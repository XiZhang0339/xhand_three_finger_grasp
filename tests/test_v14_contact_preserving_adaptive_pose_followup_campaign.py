from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import xhand_grasp.cli as cli
from xhand_grasp.actual_contact_grasp_pose_catalog import (
    initialize_or_resume_campaign,
)
from xhand_grasp.artifacts import write_json
from xhand_grasp.grasp_pose import canonical_sha256
from xhand_grasp.tuning.contact_preserving_adaptive_pose_followup import (
    AdaptiveContactModePoseSource,
    AdaptivePoseFollowupBudget,
)
from xhand_grasp.tuning.contact_preserving_adaptive_pose_followup_campaign import (
    AuthenticatedAdaptivePoseCampaignSource,
    _catalog_stage,
    _manifest,
    _run_job_stage,
    _sparse_stage_required,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "grasp_configs/left_opposed_face_palm_down_contact_preserving_planned_lift.json"
)


def _authenticated_source(tmp_path: Path) -> AuthenticatedAdaptivePoseCampaignSource:
    root = tmp_path / "source"
    root.mkdir()
    report = root / "report.json"
    report.write_text('{"complete":true}\n', encoding="utf-8")
    numerical = AdaptiveContactModePoseSource(
        search_report_sha256="1" * 64,
        first_stage_candidate_count=256,
        centers=(),
        prior_physical_config_sha256=tuple(f"{index:064x}" for index in range(256)),
        fixed_config_payload_sha256="2" * 64,
        source_authentication_id="3" * 64,
    )
    return AuthenticatedAdaptivePoseCampaignSource(
        root=root,
        report_path=report,
        numerical_source=numerical,
        artifact_paths=(report,),
        source_authentication_id="4" * 64,
    )


def test_manifest_binds_source_budget_and_both_implementation_files(
    tmp_path: Path,
) -> None:
    source = _authenticated_source(tmp_path)
    budget = AdaptivePoseFollowupBudget(candidate_count=128)
    manifest = _manifest(CONFIG, source, budget)
    assert manifest["campaign_kind"] == "contact_preserving_adaptive_pose_followup"
    assert manifest["budget"]["candidate_count"] == 128
    assert manifest["execution_contract"]["sparse_gate"] == (
        "only_when_probe_full_success_count_is_zero"
    )
    assert set(manifest["implementation_sha256"]) == {
        "contact_preserving_adaptive_pose_followup.py",
        "contact_preserving_adaptive_pose_followup_campaign.py",
    }
    assert manifest["campaign_input_sha256"] == canonical_sha256(
        {key: value for key, value in manifest.items() if key != "campaign_input_sha256"}
    )

    workspace = tmp_path / "workspace"
    initialize_or_resume_campaign(workspace, manifest, resume=False)
    source.report_path.write_text('{"complete":false}\n', encoding="utf-8")
    changed = _manifest(CONFIG, source, budget)
    with pytest.raises(RuntimeError, match="campaign_input_sha256"):
        initialize_or_resume_campaign(workspace, changed, resume=True)


def _stage_record(candidate_id: int, artifact_directory: Path, physical: str) -> dict:
    return {
        "candidate_id": candidate_id,
        "artifact_directory": str(artifact_directory),
        "physical_config_sha256": physical,
        "grasp_success": True,
        "full_success": False,
        "summary": {
            "failed_checks": ["smooth_motion_jerk_within_limit"],
            "metrics": {
                "motion_smoothness": {
                    "operation_peak_abs_filtered_jerk_m_s3": 2.8
                }
            },
        },
        "contact_mode_diagnostics": {
            "measurement_available": True,
            "taxel_count_transition_count": {"thumb": 0, "index": 0, "mid": 0},
            "simultaneous_effective_contact_duty": 1.0,
            "contact_centroid_max_step_m": {"thumb": 0.0},
        },
    }


def test_job_stage_is_atomic_resumable_and_detects_artifact_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    initialize_or_resume_campaign(
        workspace,
        {"campaign_input_sha256": canonical_sha256({"test": 1}), "test": 1},
        resume=False,
    )
    base = json.loads(CONFIG.read_text(encoding="utf-8"))
    jobs = []
    for index in range(2):
        config = copy.deepcopy(base)
        config["hand_pose"]["translation_m"][0] += index * 1e-6
        jobs.append(
            {
                "candidate_id": 100 + index,
                "physical_config_sha256": f"{index + 1:064x}",
                "job_sequence_index": index,
                "config": config,
                "job_metadata": {"adaptive_stage": "finite_difference_probe"},
            }
        )

    def fake_runner(jobs_arg, output_directory, *, workers):
        assert workers == 2
        result = []
        for job in jobs_arg:
            destination = Path(output_directory) / f"candidate_{job['candidate_id']}"
            destination.mkdir(parents=True)
            for name in ("resolved_config.json", "result.json", "trace.npz"):
                (destination / name).write_bytes(f"{job['candidate_id']}:{name}".encode())
            result.append(
                _stage_record(
                    int(job["candidate_id"]),
                    destination,
                    str(job["physical_config_sha256"]),
                )
            )
        return tuple(result)

    def fake_authenticate(root, **kwargs):
        destination = Path(root)
        return SimpleNamespace(
            artifact_paths=tuple(
                destination / name
                for name in ("resolved_config.json", "result.json", "trace.npz")
            )
        )

    import xhand_grasp.tuning.contact_preserving_adaptive_pose_followup_campaign as campaign

    monkeypatch.setattr(campaign, "run_contact_mode_pose_rescue_jobs", fake_runner)
    monkeypatch.setattr(campaign, "authenticate_v14_candidate_artifacts", fake_authenticate)
    first, report_path = _run_job_stage(
        workspace,
        "adaptive_pose_probe",
        jobs,
        workers=2,
        stage_input_extra={"source": "a" * 64},
    )
    assert first["candidate_count"] == 2
    assert report_path.is_file()

    monkeypatch.setattr(
        campaign,
        "run_contact_mode_pose_rescue_jobs",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("reran stage")),
    )
    resumed, _ = _run_job_stage(
        workspace,
        "adaptive_pose_probe",
        jobs,
        workers=2,
        stage_input_extra={"source": "a" * 64},
    )
    assert canonical_sha256(resumed) == canonical_sha256(first)

    artifact = workspace / first["records"][0]["artifact_directory"] / "trace.npz"
    artifact.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="SHA-256|hash"):
        _run_job_stage(
            workspace,
            "adaptive_pose_probe",
            jobs,
            workers=2,
            stage_input_extra={"source": "a" * 64},
        )


def _probe_report(*, hard_pass_count: int) -> dict:
    records = [
        {
            "candidate_id": index,
            "physical_config_sha256": f"{index + 1:064x}",
            "full_success": index < hard_pass_count,
        }
        for index in range(64)
    ]
    return {
        "complete": True,
        "candidate_count": 64,
        "full_success_count": hard_pass_count,
        "records": records,
    }


def test_sparse_stage_gate_requires_zero_hard_pass_and_complete_64_probes() -> None:
    assert _sparse_stage_required(_probe_report(hard_pass_count=0)) is True
    assert _sparse_stage_required(_probe_report(hard_pass_count=1)) is False
    malformed = _probe_report(hard_pass_count=0)
    malformed["records"] = malformed["records"][:-1]
    with pytest.raises(RuntimeError, match="registered budget"):
        _sparse_stage_required(malformed)


def test_catalog_publishes_only_top_five_independent_reruns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    initialize_or_resume_campaign(
        workspace,
        {"campaign_input_sha256": canonical_sha256({"test": 2}), "test": 2},
        resume=False,
    )
    records = [
        {
            "candidate_id": index,
            "artifact_directory": f"search/candidate_{index}",
            "physical_config_sha256": f"{index + 1:064x}",
            "full_success": index == 0,
        }
        for index in range(7)
    ]
    observed: dict[str, object] = {}

    def fake_materialize(selected, workspace_arg, *, target_success_count):
        observed["selected"] = [value["candidate_id"] for value in selected]
        assert workspace_arg == workspace
        assert target_success_count == 1
        return tuple(copy.deepcopy(dict(value)) for value in selected)

    def fake_publish(materialized, workspace_arg, destination, *, experiment_id):
        assert len(materialized) == 5
        catalog = destination / "manipulation" / "catalog.json"
        write_json(
            catalog,
            {
                "complete": True,
                "experiment_id": experiment_id,
                "trajectories": [],
                "aliases": {},
            },
        )
        return {"manipulation": str(catalog.relative_to(workspace_arg))}

    import xhand_grasp.tuning.contact_preserving_adaptive_pose_followup_campaign as campaign

    monkeypatch.setattr(campaign, "_materialize_publications", fake_materialize)
    monkeypatch.setattr(campaign, "_publish_rescue_viewer_catalogs", fake_publish)
    monkeypatch.setattr(
        campaign,
        "authenticated_catalog_artifact_paths",
        lambda path: (Path(path),),
    )
    result = _catalog_stage(
        records,
        workspace,
        experiment_id="left_opposed_face_palm_down_contact_preserving_planned_lift",
        target_success_count=1,
        source_report_hashes={"probe": "a" * 64},
    )
    assert observed["selected"] == [0, 1, 2, 3, 4]
    assert result["published_candidate_count"] == 5
    catalog_path = workspace / result["catalogs"]["manipulation"]
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    assert payload["adaptive_pose_followup"] is True


def test_cli_dispatches_adaptive_pose_followup_to_independent_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "formal_contact_mode_campaign"
    source.mkdir()
    output = tmp_path / "adaptive_followup"
    observed: dict[str, object] = {}

    def fake_runner(config_path, output_dir, **kwargs):
        observed.update({"config_path": config_path, "output_dir": output_dir, **kwargs})
        return {
            "probe_candidate_count": 64,
            "probe_full_success_count": 0,
            "sparse_stage_executed": True,
            "sparse_candidate_count": 64,
            "physical_unique_candidate_count": 128,
            "full_success_count": 1,
            "catalogs": {"manipulation": "catalogs/target_1/manipulation/catalog.json"},
        }

    monkeypatch.setattr(
        cli,
        "_load_contact_preserving_adaptive_pose_followup_runner",
        lambda: fake_runner,
    )
    args = cli.build_parser().parse_args(
        [
            "tune",
            "--config",
            str(CONFIG),
            "--adaptive-pose-followup-from",
            str(source),
            "--output-dir",
            str(output),
            "--workers",
            "4",
            "--target-success-count",
            "1",
        ]
    )
    assert cli.command_tune(args) == 0
    assert observed == {
        "config_path": CONFIG.resolve(),
        "output_dir": output.resolve(),
        "source_contact_mode_campaign": source.resolve(),
        "resume": False,
        "target_success_count": 1,
        "workers": 4,
        "seed": 20260821,
    }
    console = json.loads(capsys.readouterr().out)
    assert console["campaign_kind"] == "contact_preserving_adaptive_pose_followup"
    assert console["sparse_stage_executed"] is True
    assert console["target_reached"] is True
