from __future__ import annotations

import copy
import json
from argparse import Namespace
from pathlib import Path

import pytest

import xhand_grasp.cli as cli
from xhand_grasp.artifacts import default_artifact_path, resolved_run_config, write_json
from xhand_grasp.config import load_config


ROOT = Path(__file__).resolve().parents[1]
V4_CONFIG = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_tilted_down_aligned_contacts_grasp_then_lift.json"
)
V5_CONFIG = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_tilted_down_far_hand_fingertip_grasp_then_lift.json"
)


def _summary(passed: bool) -> dict:
    return {
        "passed": passed,
        "failed_checks": [] if passed else ["synthetic_failure"],
        "metrics": {},
        "stage_status": {
            "grasp_success": passed,
            "manipulation_success": passed,
            "full_success": passed,
        },
    }


def _robustness_result(config: dict, *, passed: bool) -> dict:
    return {
        "campaign_kind": "far_hand_fingertip",
        "seed": 20260821,
        "nominal_passed": passed,
        "nominal_summary": _summary(passed),
        "grid_case_count": 0,
        "grid_passes": 0,
        "hardest_passing_grid_case": None,
        "grid": [],
        "perturbation_trial_count": 50,
        "perturbation_passes": 45 if passed else 44,
        "required_perturbation_passes": 45,
        "robust_passed": passed,
        "perturbations": [],
    }


def test_v5_default_artifact_paths_use_independent_registered_root():
    config = load_config(V5_CONFIG)
    root = Path(
        "artifacts/"
        "left_opposed_face_palm_tilted_down_far_hand_fingertip_grasp_then_lift"
    )

    assert default_artifact_path(config, "run") == root / "nominal"
    assert default_artifact_path(config, "tune") == root / "tune"
    assert default_artifact_path(config, "robustness") == root / "robustness.json"


def test_resolved_v5_canonical_hard_pass_has_nominal_fingertip_classification():
    config = load_config(V5_CONFIG)

    resolved = resolved_run_config(config, _summary(True))
    status = resolved["experiment_status"]

    assert resolved["cube"]["edge_m"] == pytest.approx(0.060)
    assert resolved["cube"]["mass_kg"] == pytest.approx(0.160)
    assert resolved["cube"]["friction"] == pytest.approx(0.8)
    assert status["classification"] == "validated_far_hand_fingertip_nominal"
    assert status["passed"] is True
    assert status["hard_constraints_passed"] is True
    assert status["campaign_validated"] is True
    assert status["canonical_nominal_material"] is True
    assert status["canonical_nominal_passed"] is True
    assert status["robustness_passed"] is False
    assert status["run_context"] is None
    assert status["input_classification"] == (
        "unvalidated_far_hand_fingertip_search_template"
    )


@pytest.mark.parametrize(
    ("context", "classification"),
    [
        ("parameter_override_run", "parameter_override_run"),
        ("robustness_trial", "robustness_trial"),
    ],
)
def test_v5_context_runs_never_inherit_nominal_validation(context, classification):
    config = load_config(V5_CONFIG)
    config.pop("experiment_status", None)
    config["run_context"] = {"kind": context}

    resolved = resolved_run_config(config, _summary(True))
    status = resolved["experiment_status"]

    assert status["classification"] == classification
    # The run's own hard-pass result remains visible, but it is not promoted to
    # a catalog/campaign claim.
    assert status["passed"] is True
    assert status["hard_constraints_passed"] is True
    assert status["canonical_nominal_material"] is True
    assert status["canonical_nominal_passed"] is False
    assert status["campaign_validated"] is False
    assert status["robustness_passed"] is False
    assert status["run_context"] == context


def test_v5_noncanonical_material_hard_pass_is_not_nominal_validation():
    config = load_config(V5_CONFIG)
    config["cube"]["friction"] = 0.9

    resolved = resolved_run_config(config, _summary(True))
    status = resolved["experiment_status"]

    assert status["classification"] == (
        "validated_noncanonical_far_hand_fingertip_run"
    )
    assert status["hard_constraints_passed"] is True
    assert status["passed"] is False
    assert status["canonical_nominal_material"] is False
    assert status["canonical_nominal_passed"] is False
    assert status["campaign_validated"] is False


def test_command_robustness_promotes_only_v5_canonical_campaign(
    tmp_path, monkeypatch
):
    config = load_config(V5_CONFIG)
    source = tmp_path / "v5.json"
    output = tmp_path / "robustness.json"
    write_json(source, config)
    campaign = _robustness_result(config, passed=True)

    monkeypatch.setattr(cli, "preflight_config", lambda value: None)
    monkeypatch.setattr(cli, "robustness", lambda value, **kwargs: campaign)
    monkeypatch.setattr(cli, "run_metadata", lambda path: {"config": str(path)})
    monkeypatch.setattr(cli, "_add_v2_robustness_margin_fields", lambda *args: None)

    args = Namespace(
        config=str(source),
        output=str(output),
        workers=2,
        seed=20260821,
    )
    assert cli.command_robustness(args) == 0

    persisted = json.loads(output.read_text(encoding="utf-8"))
    status = persisted["config"]["experiment_status"]
    assert persisted["robust_passed"] is True
    assert status["classification"] == "validated_far_hand_fingertip_robust"
    assert status["passed"] is True
    assert status["campaign_validated"] is True
    assert status["canonical_nominal_passed"] is True
    assert status["robustness_passed"] is True
    assert status["robustness_passes"] == 45
    assert status["robustness_trial_count"] == 50
    assert status["robustness_required_passes"] == 45


def test_v4_resolved_and_robust_classifications_remain_unchanged(
    tmp_path, monkeypatch
):
    config = load_config(V4_CONFIG)
    nominal = resolved_run_config(config, _summary(True))
    assert nominal["experiment_status"]["classification"] == (
        "validated_aligned_contacts_nominal"
    )

    source = tmp_path / "v4.json"
    output = tmp_path / "robustness.json"
    write_json(source, config)
    campaign = _robustness_result(config, passed=True)
    campaign["campaign_kind"] = "aligned_contacts"
    monkeypatch.setattr(cli, "preflight_config", lambda value: None)
    monkeypatch.setattr(cli, "robustness", lambda value, **kwargs: campaign)
    monkeypatch.setattr(cli, "run_metadata", lambda path: {"config": str(path)})
    monkeypatch.setattr(cli, "_add_v2_robustness_margin_fields", lambda *args: None)

    args = Namespace(
        config=str(source),
        output=str(output),
        workers=1,
        seed=20260821,
    )
    assert cli.command_robustness(args) == 0
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["config"]["experiment_status"]["classification"] == (
        "validated_aligned_contacts_robust"
    )
