from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from xhand_grasp.actual_contact_grasp_pose_catalog import validate_stage_ledger
from xhand_grasp.artifacts import file_sha256, write_json
from xhand_grasp.config import ACTIVE_ACTUATORS, load_config
from xhand_grasp.grasp_pose import controller_id, grasp_pose_id
from xhand_grasp.tuning import actual_contact_grasp_pose as campaign


TEMPLATE = Path(
    "grasp_configs/"
    "left_opposed_face_palm_down_actual_contact_grasp_pose_"
    "smooth_vertical_lift.json"
)


def _result_artifacts(
    directory: Path,
    config: dict,
    *,
    candidate_id: int,
    grasp_success: bool,
    full_success: bool,
) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / "resolved_config.json"
    trace_path = directory / "trace.npz"
    write_json(config_path, config)
    np.savez_compressed(
        trace_path,
        time=np.asarray([0.001]),
        grasp_lock_step=np.asarray(0),
        grasp_pose_actual_qpos_rad=np.asarray(
            [
                config["grasp_pose"]["nominal_joint_qpos_rad"][name]
                for name in ACTIVE_ACTUATORS
            ]
        ),
    )
    summary = {
        "passed": full_success,
        "failed_checks": [] if full_success else ["operation_median_lift_reached"],
        "stage_status": {
            "grasp_success": grasp_success,
            "manipulation_success": full_success,
            "full_success": full_success,
        },
    }
    result = {
        "complete": True,
        "candidate_id": candidate_id,
        "summary": summary,
        "artifacts": {
            "resolved_config": "resolved_config.json",
            "trace": "trace.npz",
            "sha256": {
                "resolved_config": file_sha256(config_path),
                "trace": file_sha256(trace_path),
            },
        },
    }
    result_path = directory / "result.json"
    write_json(result_path, result)
    return {
        "config_path": config_path,
        "result_path": result_path,
        "trace_path": trace_path,
        "summary": summary,
    }


def test_small_budget_runner_commits_resumes_and_publishes_best_first(
    tmp_path, monkeypatch
):
    template = load_config(TEMPLATE)
    nominal = tuple(
        template["grasp_pose"]["nominal_joint_qpos_rad"][name]
        for name in ACTIVE_ACTUATORS
    )
    fake_source = campaign.V8ActualQposSource(
        source_index=0,
        pose_id="fake-v8-pose",
        config_path=str(TEMPLATE.resolve()),
        result_path=str(TEMPLATE.resolve()),
        trace_path=str(TEMPLATE.resolve()),
        config_sha256=file_sha256(TEMPLATE),
        result_sha256=file_sha256(TEMPLATE),
        trace_sha256=file_sha256(TEMPLATE),
        stable_window_start_step=0,
        stable_window_end_step=249,
        stable_window_sample_count=250,
        actual_joint_qpos_rad=nominal,
        config=copy.deepcopy(template),
    )
    monkeypatch.setattr(
        campaign, "load_v8_actual_qpos_sources", lambda _config: (fake_source,)
    )

    static_config = copy.deepcopy(template)
    static_record = {
        "candidate_id": 101,
        "cell_index": 0,
        "cell_id": "tiny",
        "static_pass": True,
        "grasp_pose_id": grasp_pose_id(static_config),
        "controller_id": controller_id(static_config),
        "candidate_sha256": campaign.canonical_sha256(static_config),
        "config": static_config,
    }

    def fake_static(_template, _sources, output, **kwargs):
        report = output / "static" / kwargs["stage"] / "report.json"
        write_json(
            report,
            {
                "complete": True,
                "pool_by_cell": {"0": [static_record]},
            },
        )
        return campaign.CampaignStageExecution(
            records=(copy.deepcopy(static_record),),
            artifacts=(report,),
            summary={"evaluated_count": 1, "retained_static_pass_count": 1},
        )

    def fake_dynamic(_records, output, **_kwargs):
        config = copy.deepcopy(template)
        identifier = 1601
        relative = Path("candidates") / f"candidate_{identifier}"
        paths = _result_artifacts(
            output / "dynamic" / relative,
            config,
            candidate_id=identifier,
            grasp_success=True,
            full_success=False,
        )
        record = {
            "candidate_id": identifier,
            "source_candidate_id": 101,
            "candidate_sha256": campaign.canonical_sha256(config),
            "grasp_pose_id": grasp_pose_id(config),
            "controller_id": controller_id(config),
            "grasp_success": True,
            "artifact_directory": str(relative),
            "config": config,
            "summary": paths["summary"],
        }
        report = output / "dynamic" / f"{_kwargs['stage']}_report.json"
        write_json(report, {"complete": True, "candidate_id": identifier})
        return campaign.CampaignStageExecution(
            records=(record,),
            artifacts=(report,),
            summary={"dynamic_candidate_count": 1, "grasp_success_count": 1},
        )

    def fake_refinement(records, output, **kwargs):
        report = output / "static" / kwargs["stage"] / "uniform_refinement.json"
        write_json(report, {"complete": True, "refined_candidates": []})
        return campaign.CampaignStageExecution(
            records=(),
            artifacts=(report,),
            summary={
                "selected_source_count": len(records),
                "refined_candidate_count": 0,
                "refined_static_pass_count": 0,
            },
        )

    def fake_local_refinement(_records, output, **_kwargs):
        report = output / "static" / "expanded" / "joint_controller_local_refinement.json"
        config = copy.deepcopy(template)
        record = {
            "candidate_id": 1702,
            "source_candidate_id": 102,
            "candidate_sha256": campaign.canonical_sha256(config),
            "grasp_pose_id": grasp_pose_id(config),
            "controller_id": controller_id(config),
            "config": config,
        }
        write_json(report, {"complete": True, "dynamic_candidates": [record]})
        return campaign.CampaignStageExecution(
            records=(record,),
            artifacts=(report,),
            summary={
                "declared_candidate_budget": 1280,
                "generated_candidate_count": 1,
                "dynamic_promoted_count": 1,
                "selected_parent_candidate_ids": [1601],
            },
        )

    def fake_local_dynamic(records, output, **_kwargs):
        source = records[0]
        relative = Path("candidates") / f"candidate_{source['candidate_id']}"
        paths = _result_artifacts(
            output / "dynamic" / relative,
            copy.deepcopy(source["config"]),
            candidate_id=int(source["candidate_id"]),
            grasp_success=False,
            full_success=False,
        )
        record = {
            **copy.deepcopy(source),
            "grasp_success": False,
            "artifact_directory": str(relative),
            "summary": paths["summary"],
        }
        report = output / "dynamic" / "expanded_local_refinement_report.json"
        write_json(report, {"complete": True, "candidate_id": source["candidate_id"]})
        return campaign.CampaignStageExecution(
            records=(record,),
            artifacts=(report,),
            summary={"dynamic_candidate_count": 1, "grasp_success_count": 0},
        )

    def fake_measured(records, output, **kwargs):
        successful = tuple(
            copy.deepcopy(dict(value))
            for value in records
            if value.get("summary", {})
            .get("stage_status", {})
            .get("grasp_success", False)
        )
        report = (
            output
            / "dynamic"
            / "measured"
            / f"{kwargs['stage']}_report.json"
        )
        write_json(report, {"complete": True, "candidate_count": len(records)})
        return campaign.CampaignStageExecution(
            records=successful,
            artifacts=(report,),
            summary={
                "authoritative_dynamic_grasp_success_count": len(successful),
                "measured_grasp_pose_success_count": len(successful),
                "measured_grasp_pose_failure_count": 0,
                "workers": kwargs["workers"],
            },
        )

    manipulation_invocations = []

    def fake_manipulation(_records, output, **_kwargs):
        expanded = len(manipulation_invocations) % 2 == 1
        manipulation_invocations.append("expanded" if expanded else "quick")
        identifier = 1_601_000
        directory = (
            output
            / "manipulation"
            / ("expanded" if expanded else "quick")
            / f"candidate_{identifier}"
        )
        paths = _result_artifacts(
            directory,
            copy.deepcopy(template),
            candidate_id=identifier,
            grasp_success=True,
            full_success=expanded,
        )
        record = {
            "candidate_id": identifier,
            "discovery_index": 0,
            **paths,
            "initial_state_source": "configured_no_contact_reset",
            "checkpoint_used": False,
        }
        report = output / "manipulation" / (
            "expanded_target_1_report.json" if expanded else "quick_target_1_report.json"
        )
        write_json(
            report,
            {
                "complete": True,
                "full_success_count": int(expanded),
                "initial_state_source": "configured_no_contact_reset",
                "checkpoint_used": False,
            },
        )
        return campaign.CampaignStageExecution(
            records=(record,),
            artifacts=(report,),
            summary={"candidate_count": 1, "full_success_count": int(expanded)},
        )

    monkeypatch.setattr(campaign, "_run_static_stage", fake_static)
    monkeypatch.setattr(campaign, "_run_uniform_refinement_stage", fake_refinement)
    monkeypatch.setattr(campaign, "_run_dynamic_stage", fake_dynamic)
    monkeypatch.setattr(
        campaign,
        "_run_joint_controller_local_refinement_stage",
        fake_local_refinement,
    )
    monkeypatch.setattr(
        campaign, "_run_materialized_local_dynamic_stage", fake_local_dynamic
    )
    monkeypatch.setattr(
        campaign,
        "_run_measured_grasp_pose_finalization_stage",
        fake_measured,
    )
    monkeypatch.setattr(campaign, "_run_manipulation_stage", fake_manipulation)
    output = tmp_path / "campaign"
    first = campaign.run_actual_contact_grasp_pose_campaign(
        TEMPLATE,
        output,
        resume=False,
        target_success_count=1,
        workers=1,
        seed=20260821,
    )
    assert first["target_reached"]
    assert first["grasp_success_count"] == first["full_success_count"] == 1
    assert first["checkpoint_policy"] == {
        "search_probes_may_restore_latched_free_dynamics": True,
        "final_initial_state_source": "configured_no_contact_reset",
        "final_checkpoint_used": False,
    }
    for relative in first["catalogs"].values():
        catalog = json.loads((output / relative).read_text(encoding="utf-8"))
        assert "best_first" in catalog["aliases"]
        if catalog["catalog_kind"] == "manipulation":
            best = next(
                value
                for value in catalog["trajectories"]
                if value["trajectory_id"] == catalog["aliases"]["best_first"]
            )
            assert best["initial_state_source"] == "configured_no_contact_reset"
            assert best["checkpoint_used"] is False
    ledger = validate_stage_ledger(output)
    assert {
        "source_bundle",
        "evidence_grasp_anchors",
        "quick_static",
        "quick_uniform_refinement",
        "quick_dynamic",
        "quick_measured_grasp_pose_finalization",
        "quick_manipulation_1",
        "expanded_static",
        "expanded_uniform_refinement",
        "expanded_dynamic",
        "expanded_joint_controller_local_refinement",
        "expanded_local_refinement_dynamic",
        "expanded_measured_grasp_pose_finalization",
        "expanded_manipulation_1",
        "catalogs_1",
    }.issubset(ledger["stages"])
    anchor_dir = tmp_path / "late_anchor"
    _result_artifacts(
        anchor_dir,
        copy.deepcopy(template),
        candidate_id=9901,
        grasp_success=True,
        full_success=False,
    )
    with pytest.raises(RuntimeError, match="anchor inputs changed"):
        campaign._prepare_evidence_grasp_anchors(output, (anchor_dir,))

    resumed = campaign.run_actual_contact_grasp_pose_campaign(
        TEMPLATE,
        output,
        resume=True,
        target_success_count=1,
        workers=1,
        seed=20260821,
    )
    assert resumed["campaign_input_sha256"] == first["campaign_input_sha256"]
    assert resumed["catalogs"] == first["catalogs"]

    manipulation_catalog = output / resumed["catalogs"]["manipulation"]
    catalog_payload = json.loads(manipulation_catalog.read_text(encoding="utf-8"))
    member = catalog_payload["trajectories"][0]["artifacts"]["result"]
    member_path = manipulation_catalog.parent / member
    assert str(member_path.relative_to(output)) in ledger["stages"]["catalogs_1"][
        "artifacts"
    ]
    member_payload = json.loads(member_path.read_text(encoding="utf-8"))
    member_payload["summary"]["passed"] = not member_payload["summary"]["passed"]
    write_json(member_path, member_payload)
    with pytest.raises(RuntimeError, match="artifact SHA-256 mismatch"):
        validate_stage_ledger(output)


def test_nonempty_evidence_anchor_bundle_rejects_deletion_and_change(
    tmp_path, monkeypatch
):
    from xhand_grasp.tuning import actual_contact_manipulation

    monkeypatch.setattr(
        actual_contact_manipulation,
        "validate_grasp_success_source",
        lambda *_args, **_kwargs: None,
    )
    template = load_config(TEMPLATE)
    anchor = tmp_path / "anchor"
    artifacts = _result_artifacts(
        anchor,
        template,
        candidate_id=41,
        grasp_success=True,
        full_success=False,
    )
    workspace = tmp_path / "workspace"
    first = campaign._prepare_evidence_grasp_anchors(workspace, (anchor,))
    assert len(first.records) == 1
    resumed = campaign._prepare_evidence_grasp_anchors(workspace, (anchor,))
    assert len(resumed.records) == 1

    with pytest.raises(RuntimeError, match="anchor inputs changed"):
        campaign._prepare_evidence_grasp_anchors(workspace, ())

    source_result = json.loads(
        Path(artifacts["result_path"]).read_text(encoding="utf-8")
    )
    source_result["summary"]["failed_checks"] = ["tampered"]
    write_json(Path(artifacts["result_path"]), source_result)
    with pytest.raises(RuntimeError, match="anchor inputs changed"):
        campaign._prepare_evidence_grasp_anchors(workspace, (anchor,))
