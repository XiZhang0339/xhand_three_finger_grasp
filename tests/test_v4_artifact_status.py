from __future__ import annotations

import copy
import json
from argparse import Namespace
from pathlib import Path

import xhand_grasp.cli as cli
from xhand_grasp.artifacts import resolved_run_config, write_json
from xhand_grasp.config import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_tilted_down_aligned_contacts_grasp_then_lift.json"
)


def _summary(passed: bool) -> dict:
    return {
        "passed": passed,
        "failed_checks": [] if passed else ["synthetic"],
        "metrics": {},
        "stage_status": {
            "grasp_success": passed,
            "manipulation_success": passed,
            "full_success": passed,
        },
    }


def test_resolved_v4_nominal_and_override_status_are_distinct():
    config = load_config(CONFIG)

    nominal = resolved_run_config(config, _summary(True))
    assert nominal["experiment_status"]["classification"] == (
        "validated_aligned_contacts_nominal"
    )
    assert nominal["experiment_status"]["passed"] is True
    assert nominal["experiment_status"]["campaign_validated"] is True
    assert nominal["experiment_status"]["constant_density_passed"] is True
    assert nominal["experiment_status"]["robustness_passed"] is False

    override = copy.deepcopy(config)
    override.pop("experiment_status")
    override["cube"]["friction"] = 0.9
    override["run_context"] = {"kind": "parameter_override_run"}
    resolved = resolved_run_config(override, _summary(True))
    assert resolved["experiment_status"]["classification"] == (
        "parameter_override_run"
    )
    assert resolved["experiment_status"]["passed"] is True
    assert resolved["experiment_status"]["campaign_validated"] is False
    assert resolved["experiment_status"]["constant_density_passed"] is False
    assert resolved["experiment_status"]["run_context"] == (
        "parameter_override_run"
    )


def test_command_run_marks_v4_material_override_as_fresh_run(tmp_path, monkeypatch):
    config = load_config(CONFIG)
    config["candidate_metadata"] = {"source_friction": 0.8}
    source = tmp_path / "source.json"
    output = tmp_path / "output"
    write_json(source, config)

    def fake_run(actual, *, trace_path, video_path):
        assert trace_path is None
        assert video_path is None
        assert actual["run_context"] == {"kind": "parameter_override_run"}
        assert actual["cube"]["friction"] == 0.9
        assert "experiment_status" not in actual
        assert "candidate_metadata" not in actual
        return _summary(True)

    monkeypatch.setattr(cli, "run_simulation", fake_run)
    monkeypatch.setattr(cli, "run_metadata", lambda path: {"config": str(path)})
    monkeypatch.setattr(
        cli,
        "_normalized_margin_fields",
        lambda config, summary: {
            "normalized_acceptance_margins": {},
            "minimum_normalized_acceptance_margin": None,
            "limiting_metric": None,
        },
    )
    args = Namespace(
        config=str(source),
        hardest_from=None,
        output_dir=str(output),
        video=False,
        video_filename="nominal.mp4",
        no_trace=True,
        edge_mm=None,
        mass_g=None,
        friction=0.9,
    )

    assert cli.command_run(args) == 0
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    status = result["experiment_status"]
    assert status["classification"] == "parameter_override_run"
    assert status["passed"] is True
    assert status["campaign_validated"] is False


def test_command_robustness_persists_complete_v4_campaign(tmp_path, monkeypatch):
    config = load_config(CONFIG)
    source = tmp_path / "source.json"
    output = tmp_path / "robustness.json"
    write_json(source, config)

    failed_summary = {
        "passed": False,
        "failed_checks": ["simulation_error"],
        "error": "synthetic diagnostic",
        "checks": {},
        "metrics": {},
        "stage_status": {
            "grasp_success": False,
            "manipulation_success": False,
            "full_success": False,
        },
    }
    campaign = {
        "campaign_kind": "aligned_contacts",
        "seed": 20260821,
        "nominal_passed": False,
        "nominal_summary": copy.deepcopy(failed_summary),
        "grid_case_count": 0,
        "grid_passes": 0,
        "grid": [],
        "perturbation_trial_count": 50,
        "perturbation_passes": 0,
        "required_perturbation_passes": 45,
        "robust_passed": False,
        "perturbations": [
            {
                **copy.deepcopy(failed_summary),
                "trial": trial,
                "resolved_trial_config": copy.deepcopy(config),
            }
            for trial in range(50)
        ],
    }

    monkeypatch.setattr(cli, "preflight_config", lambda value: None)
    monkeypatch.setattr(cli, "robustness", lambda value, **kwargs: campaign)
    monkeypatch.setattr(cli, "run_metadata", lambda path: {"config": str(path)})

    args = Namespace(
        config=str(source),
        output=str(output),
        workers=3,
        seed=20260821,
    )
    assert cli.command_robustness(args) == 2

    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["campaign_kind"] == "aligned_contacts"
    assert persisted["grid"] == []
    assert persisted["grid_case_count"] == 0
    assert persisted["perturbation_trial_count"] == 50
    assert len(persisted["perturbations"]) == 50
    assert persisted["hardest_passing_config"] is None
    assert persisted["input_experiment_status"] == config["experiment_status"]
    status = persisted["config"]["experiment_status"]
    assert status["robustness_passed"] is False
    assert status["robustness_passes"] == 0
    assert status["robustness_trial_count"] == 50
    assert status["robustness_required_passes"] == 45
    assert persisted["perturbations"][0]["limiting_metric"] == "simulation_error"
    assert persisted["perturbations"][0]["error"] == "synthetic diagnostic"
