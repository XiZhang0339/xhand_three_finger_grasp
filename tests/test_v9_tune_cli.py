from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import xhand_grasp.cli as cli
from xhand_grasp.actual_contact_grasp_pose_catalog import (
    build_campaign_manifest,
    commit_campaign_stage,
    export_actual_contact_grasp_pose_catalog,
    export_actual_contact_manipulation_catalog,
    initialize_or_resume_campaign,
    validate_stage_ledger,
)
from xhand_grasp.artifacts import file_sha256, write_json
from xhand_grasp.config import load_config
from xhand_grasp.viewer import resolve_replay_source


TEMPLATE = Path(
    "grasp_configs/"
    "left_opposed_face_palm_down_actual_contact_grasp_pose_"
    "smooth_vertical_lift.json"
)


def test_tune_parser_exposes_bounded_v9_resume_controls():
    parser = cli.build_parser()
    arguments = parser.parse_args(
        ["tune", "--resume", "--target-success-count", "5"]
    )
    assert arguments.resume is True
    assert arguments.target_success_count == 5
    with pytest.raises(SystemExit):
        parser.parse_args(["tune", "--target-success-count", "2"])


def test_old_schema_rejects_v9_only_tune_controls(tmp_path):
    parser = cli.build_parser()
    arguments = parser.parse_args(
        [
            "tune",
            "--config",
            "grasp_configs/left_three_finger_cube.json",
            "--output-dir",
            str(tmp_path / "out"),
            "--resume",
        ]
    )
    with pytest.raises(ValueError, match="schema-v9"):
        cli.command_tune(arguments)


def test_v9_dispatch_uses_dedicated_lazy_runner(tmp_path, monkeypatch, capsys):
    output = tmp_path / "campaign"
    observed = {}

    def fake_runner(config_path, output_dir, **kwargs):
        observed.update(
            {"config_path": config_path, "output_dir": output_dir, **kwargs}
        )
        manifest = build_campaign_manifest(config_path, seed=kwargs["seed"])
        initialize_or_resume_campaign(output_dir, manifest, resume=False)
        return {"grasp_success_count": 3, "full_success_count": 1}

    monkeypatch.setattr(cli, "_load_v9_tune_runner", lambda: fake_runner)
    arguments = cli.build_parser().parse_args(
        [
            "tune",
            "--config",
            str(TEMPLATE),
            "--output-dir",
            str(output),
            "--target-success-count",
            "1",
            "--workers",
            "3",
        ]
    )
    assert cli.command_tune(arguments) == 0
    assert observed == {
        "config_path": TEMPLATE.resolve(),
        "output_dir": output.resolve(),
        "resume": False,
        "target_success_count": 1,
        "workers": 3,
        "seed": 20260821,
    }
    payload = json.loads(capsys.readouterr().out)
    assert payload["target_reached"] is True
    assert payload["full_success_count"] == 1


def test_v9_resume_dispatch_requires_existing_output_and_runner_target(tmp_path, monkeypatch):
    output = tmp_path / "campaign"
    initialize_or_resume_campaign(
        output,
        build_campaign_manifest(TEMPLATE, seed=20260821),
        resume=False,
    )
    seen = {}

    def fake_runner(_config, _output, **kwargs):
        seen.update(kwargs)
        return {"full_success_count": 1}

    monkeypatch.setattr(cli, "_load_v9_tune_runner", lambda: fake_runner)
    arguments = cli.build_parser().parse_args(
        [
            "tune",
            "--config",
            str(TEMPLATE),
            "--output-dir",
            str(output),
            "--resume",
            "--target-success-count",
            "5",
        ]
    )
    assert cli.command_tune(arguments) == 2
    assert seen["resume"] is True
    assert seen["target_success_count"] == 5


def test_resume_manifest_rejects_each_bound_hash_mismatch(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")
    manifest = build_campaign_manifest(config, seed=20260821, source_paths=[source])
    output = initialize_or_resume_campaign(tmp_path / "campaign", manifest, resume=False)

    for field in (
        "config_sha256",
        "model_sha256",
        "uv_lock_sha256",
        "source_sha256",
        "campaign_input_sha256",
    ):
        changed = copy.deepcopy(manifest)
        changed[field] = "0" * 64
        with pytest.raises(RuntimeError, match=field):
            initialize_or_resume_campaign(output, changed, resume=True)


def test_stage_ledger_is_atomic_reusable_and_authenticates_artifacts(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")
    manifest = build_campaign_manifest(config, seed=7, source_paths=[source])
    output = initialize_or_resume_campaign(tmp_path / "campaign", manifest, resume=False)
    artifact = output / "cells" / "cell_00.json"
    write_json(artifact, {"complete": True})
    first = commit_campaign_stage(
        output,
        "quick_static",
        stage_input={"count": 110_000},
        artifacts=[artifact],
        summary={"retained": 4},
    )
    assert commit_campaign_stage(
        output,
        "quick_static",
        stage_input={"count": 110_000},
        artifacts=[artifact],
        summary={"retained": 4},
    ) == first
    assert validate_stage_ledger(output)["stages"]["quick_static"]["complete"]
    artifact.write_text("tampered", encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        initialize_or_resume_campaign(output, manifest, resume=True)


def _catalog_candidate(
    root: Path,
    identifier: int,
    *,
    grasp: bool,
    full: bool,
    override: bool = False,
) -> dict:
    directory = root / f"candidate_{identifier}"
    directory.mkdir(parents=True)
    config = load_config(TEMPLATE)
    if override:
        config["run_context"] = {"kind": "parameter_override_run"}
    config_path = directory / "resolved_config.json"
    write_json(config_path, config)
    trace_path = directory / "trace.npz"
    np.savez_compressed(trace_path, time=np.asarray([0.001]))
    summary = {
        "passed": full,
        "failed_checks": [] if full else ["operation_median_lift_reached"],
        "stage_status": {
            "grasp_success": grasp,
            "manipulation_success": full,
            "full_success": full,
        },
    }
    result = {
        "candidate_id": identifier,
        "discovery_index": identifier,
        "summary": summary,
        "artifacts": {
            "resolved_config": config_path.name,
            "trace": trace_path.name,
            "sha256": {
                "resolved_config": file_sha256(config_path),
                "trace": file_sha256(trace_path),
            },
        },
    }
    result_path = directory / "result.json"
    write_json(result_path, result)
    return {
        "candidate_id": identifier,
        "discovery_index": identifier,
        "config_path": config_path,
        "result_path": result_path,
        "trace_path": trace_path,
    }


def test_separate_catalogs_are_viewer_compatible_and_best_first_is_chronological(tmp_path):
    candidates = [
        _catalog_candidate(tmp_path, 0, grasp=True, full=True, override=True),
        _catalog_candidate(tmp_path, 1, grasp=True, full=False),
        _catalog_candidate(tmp_path, 2, grasp=True, full=True),
        _catalog_candidate(tmp_path, 3, grasp=True, full=True),
    ]
    grasp_output = tmp_path / "grasp_catalog"
    manipulation_output = tmp_path / "manipulation_catalog"
    grasp = export_actual_contact_grasp_pose_catalog(
        candidates, grasp_output, selected_count=2
    )
    manipulation = export_actual_contact_manipulation_catalog(
        candidates, manipulation_output, selected_count=2
    )

    assert grasp["success_count"] == 2
    grasp_best = grasp["aliases"]["best_first"]
    assert next(
        entry for entry in grasp["trajectories"] if entry["trajectory_id"] == grasp_best
    )["candidate_id"] == "1"
    assert manipulation["success_count"] == 2
    manipulation_best = manipulation["aliases"]["best_first"]
    assert next(
        entry
        for entry in manipulation["trajectories"]
        if entry["trajectory_id"] == manipulation_best
    )["candidate_id"] == "2"
    assert all(
        entry["candidate_id"] != "0" or "best_first" not in entry["aliases"]
        for entry in (*grasp["trajectories"], *manipulation["trajectories"])
    )
    replay = resolve_replay_source(
        catalog_path=manipulation_output / "catalog.json",
        trajectory="best_first",
    )
    assert replay.config_path.is_file()
    assert replay.trace_path.is_file()


def test_diagnostic_override_cannot_create_success_alias(tmp_path):
    only_override = _catalog_candidate(
        tmp_path, 0, grasp=True, full=True, override=True
    )
    catalog = export_actual_contact_manipulation_catalog(
        [only_override], tmp_path / "catalog", selected_count=1
    )
    assert catalog["success_count"] == 0
    assert "best_first" not in catalog["aliases"]
    assert catalog["aliases"]["best_attempt"]
    assert catalog["trajectories"][0]["classification"] == "diagnostic_override"


def test_manipulation_catalog_skips_compacted_grasp_only_candidate(tmp_path):
    compacted = _catalog_candidate(
        tmp_path, 0, grasp=True, full=False
    )
    retained = _catalog_candidate(
        tmp_path, 1, grasp=True, full=False
    )
    result_path = Path(compacted["result_path"])
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    trace_path = Path(compacted["trace_path"])
    payload["artifacts"]["trace"] = None
    payload["artifacts"]["trace_retained"] = False
    payload["artifacts"]["trace_sha256_at_evaluation"] = file_sha256(trace_path)
    payload["artifacts"]["sha256"].pop("trace")
    write_json(result_path, payload)
    trace_path.unlink()

    catalog = export_actual_contact_manipulation_catalog(
        [compacted, retained], tmp_path / "manipulation_catalog", selected_count=1
    )

    assert catalog["success_count"] == 0
    assert catalog["aliases"]["best_attempt"]
    assert [entry["candidate_id"] for entry in catalog["trajectories"]] == ["1"]


def test_grasp_catalog_rejects_compacted_grasp_success(tmp_path):
    compacted = _catalog_candidate(
        tmp_path, 0, grasp=True, full=False
    )
    result_path = Path(compacted["result_path"])
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    trace_path = Path(compacted["trace_path"])
    payload["artifacts"]["trace"] = None
    payload["artifacts"]["trace_retained"] = False
    payload["artifacts"]["trace_sha256_at_evaluation"] = file_sha256(trace_path)
    payload["artifacts"]["sha256"].pop("trace")
    write_json(result_path, payload)
    trace_path.unlink()

    with pytest.raises(ValueError, match="successful candidate"):
        export_actual_contact_grasp_pose_catalog(
            [compacted], tmp_path / "grasp_catalog", selected_count=1
        )


def test_manipulation_catalog_rejects_compacted_full_success(tmp_path):
    compacted = _catalog_candidate(
        tmp_path, 0, grasp=True, full=True
    )
    result_path = Path(compacted["result_path"])
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    trace_path = Path(compacted["trace_path"])
    payload["artifacts"]["trace"] = None
    payload["artifacts"]["trace_retained"] = False
    payload["artifacts"]["trace_sha256_at_evaluation"] = file_sha256(trace_path)
    payload["artifacts"]["sha256"].pop("trace")
    write_json(result_path, payload)
    trace_path.unlink()

    with pytest.raises(ValueError, match="successful candidate"):
        export_actual_contact_manipulation_catalog(
            [compacted], tmp_path / "manipulation_catalog", selected_count=1
        )


def test_cli_rejects_runner_catalog_that_promotes_a_diagnostic(tmp_path):
    output = tmp_path / "campaign"
    output.mkdir()
    bad_catalog = output / "catalog.json"
    write_json(
        bad_catalog,
        {
            "experiment_id": (
                "left_opposed_face_palm_down_actual_contact_grasp_pose_"
                "smooth_vertical_lift"
            ),
            "aliases": {"best_first": "override"},
            "trajectories": [
                {
                    "trajectory_id": "override",
                    "classification": "diagnostic_override",
                }
            ],
        },
    )
    with pytest.raises(RuntimeError, match="canonical success"):
        cli._publish_v9_returned_candidates(
            {"catalogs": {"manipulation": "catalog.json"}},
            output,
            target_success_count=1,
        )


def test_cli_accepts_complete_v12_contact_point_catalog_without_trajectories(
    tmp_path,
):
    output = tmp_path / "campaign"
    output.mkdir()
    catalog = output / "contact_points.json"
    experiment_id = (
        "left_opposed_face_palm_down_90mm_contact_point_targeted_actual_grasp_pose"
    )
    write_json(
        catalog,
        {
            "contact_point_catalog_schema_version": 1,
            "complete": True,
            "experiment_id": experiment_id,
            "selected_point_plan_id": None,
            "stop_reason": "no_reachable_contact_point_plan",
            "plans": [],
            "aliases": {},
        },
    )
    assert cli._publish_v9_returned_candidates(
        {"catalogs": {"contact_point": catalog.name}},
        output,
        target_success_count=1,
        experiment_id=experiment_id,
        allow_contact_point_catalog=True,
    ) == {"contact_point": catalog.name}

    payload = json.loads(catalog.read_text(encoding="utf-8"))
    payload["selected_point_plan_id"] = "missing-plan"
    write_json(catalog, payload)
    with pytest.raises(RuntimeError, match="invalid contact-point catalog"):
        cli._publish_v9_returned_candidates(
            {"catalogs": {"contact_point": catalog.name}},
            output,
            target_success_count=1,
            experiment_id=experiment_id,
            allow_contact_point_catalog=True,
        )
