from __future__ import annotations

import copy
from pathlib import Path

import pytest

import xhand_grasp.cli as cli
from xhand_grasp.actual_contact_grasp_pose_catalog import (
    initialize_or_resume_campaign,
)
from xhand_grasp.actual_contact_capability import (
    ACTUAL_CONTACT_SCHEMA_VERSIONS,
    JOINT_PAIR_NEAR_ZERO_CONTACT_PRESERVING_PLANNED_LIFT_TUNING_STRATEGY,
    is_actual_contact_definition,
    is_contact_preserving_planned_lift_definition,
    is_joint_pair_near_zero_contact_preserving_planned_lift_definition,
    resolve_actual_contact_definition,
    resolve_contact_preserving_planned_lift_definition,
    resolve_joint_pair_near_zero_contact_preserving_planned_lift_definition,
)
from xhand_grasp.config import load_config
from xhand_grasp.experiment import resolve_experiment
from xhand_grasp.tuning.joint_pair_near_zero_campaign import (
    JointPairNearZeroBudget,
)
from xhand_grasp.tuning.joint_pair_near_zero_campaign_runner import (
    build_v15_campaign_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
V15_CONFIG = ROOT / "grasp_configs" / (
    "left_opposed_face_palm_down_joint_pair_near_zero_"
    "contact_preserving_planned_lift.json"
)
V14_CONFIG = ROOT / "grasp_configs" / (
    "left_opposed_face_palm_down_contact_preserving_planned_lift.json"
)
V9_CONFIG = ROOT / "grasp_configs" / (
    "left_opposed_face_palm_down_actual_contact_grasp_pose_"
    "smooth_vertical_lift.json"
)


def test_schema_v15_is_a_strict_actual_contact_capability() -> None:
    v15 = load_config(V15_CONFIG)
    definition = resolve_experiment(v15)
    assert 15 in ACTUAL_CONTACT_SCHEMA_VERSIONS
    assert definition.tuning_strategy == (
        JOINT_PAIR_NEAR_ZERO_CONTACT_PRESERVING_PLANNED_LIFT_TUNING_STRATEGY
    )
    assert is_actual_contact_definition(definition)
    assert is_joint_pair_near_zero_contact_preserving_planned_lift_definition(
        definition
    )
    assert not is_contact_preserving_planned_lift_definition(definition)
    assert resolve_actual_contact_definition(v15) is definition
    assert (
        resolve_joint_pair_near_zero_contact_preserving_planned_lift_definition(
            v15
        )
        is definition
    )

    with pytest.raises(ValueError, match="schema v15"):
        resolve_joint_pair_near_zero_contact_preserving_planned_lift_definition(
            load_config(V14_CONFIG)
        )
    with pytest.raises(ValueError, match="schema v14"):
        resolve_contact_preserving_planned_lift_definition(v15)

    mismatched = copy.deepcopy(v15)
    mismatched["schema_version"] = 14
    with pytest.raises(ValueError, match="schema v15"):
        resolve_joint_pair_near_zero_contact_preserving_planned_lift_definition(
            mismatched
        )


def test_v15_manifest_and_runner_dispatch_precede_generic_v14() -> None:
    definition = resolve_experiment(load_config(V15_CONFIG))
    assert cli._load_actual_contact_tune_runner(definition) is (
        cli._run_v15_joint_pair_near_zero_tune_campaign
    )
    expected = build_v15_campaign_manifest(
        V15_CONFIG,
        budget=JointPairNearZeroBudget(seed=20260822),
    )
    routed = cli._build_actual_contact_campaign_manifest(
        definition,
        V15_CONFIG,
        seed=20260822,
    )
    assert routed == expected
    assert routed["seed"] == 20260822

    v14 = resolve_experiment(load_config(V14_CONFIG))
    assert cli._load_actual_contact_tune_runner(v14).__name__ == (
        "run_contact_preserving_planned_lift_campaign"
    )
    v9 = resolve_experiment(load_config(V9_CONFIG))
    assert cli._load_actual_contact_tune_runner(v9).__name__ == (
        "run_actual_contact_grasp_pose_campaign"
    )


def test_v15_tune_help_controls_dispatch_and_resume_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parsed = cli.build_parser().parse_args(
        [
            "tune",
            "--config",
            str(V15_CONFIG),
            "--output-dir",
            str(tmp_path / "command"),
            "--resume",
            "--target-success-count",
            "5",
            "--workers",
            "3",
            "--seed",
            "20260822",
        ]
    )
    assert parsed.resume is True
    assert parsed.target_success_count == 5
    assert parsed.workers == 3
    assert parsed.seed == 20260822

    routed: dict[str, object] = {}

    def command(args, **kwargs):
        routed.update({"args": args, **kwargs})
        return 73

    monkeypatch.setattr(cli, "_command_tune_actual_contact_grasp_pose", command)
    assert cli.command_tune(parsed) == 73
    assert routed["experiment_id"] == (
        "left_opposed_face_palm_down_joint_pair_near_zero_"
        "contact_preserving_planned_lift"
    )

    definition = resolve_experiment(load_config(V15_CONFIG))
    first = cli._build_actual_contact_campaign_manifest(
        definition, V15_CONFIG, seed=20260821
    )
    workspace = initialize_or_resume_campaign(
        tmp_path / "resume",
        first,
        resume=False,
    )
    initialize_or_resume_campaign(workspace, first, resume=True)
    changed = cli._build_actual_contact_campaign_manifest(
        definition, V15_CONFIG, seed=20260822
    )
    with pytest.raises(RuntimeError, match="seed|campaign_input_sha256"):
        initialize_or_resume_campaign(workspace, changed, resume=True)


def test_v15_cli_adapter_injects_delayed_backend_and_seeded_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = object()
    observed: dict[str, object] = {}

    def factory(*, workers: int, seed: int) -> object:
        observed["factory"] = (workers, seed)
        return backend

    def runner(config_path, output_dir, **kwargs):
        observed["runner"] = (config_path, output_dir, kwargs)
        return {
            "full_success_count": 0,
            "catalog_path": str(Path(output_dir) / "catalog.json"),
        }

    import xhand_grasp.tuning.joint_pair_near_zero_campaign_runner as module

    monkeypatch.setattr(cli, "_load_v15_campaign_backend_factory", lambda: factory)
    monkeypatch.setattr(module, "run_joint_pair_near_zero_campaign", runner)
    output = tmp_path / "campaign"
    result = cli._run_v15_joint_pair_near_zero_tune_campaign(
        V15_CONFIG,
        output,
        resume=False,
        target_success_count=5,
        workers=7,
        seed=314159,
    )
    assert observed["factory"] == (7, 314159)
    config_path, output_dir, kwargs = observed["runner"]
    assert Path(config_path) == V15_CONFIG
    assert Path(output_dir) == output
    assert kwargs["resume"] is False
    assert kwargs["target_success_count"] == 5
    assert kwargs["backend"] is backend
    assert kwargs["budget"].seed == 314159
    assert result["catalogs"] == {
        "manipulation": str(output / "catalog.json")
    }

    with pytest.raises(ValueError, match="hash-bound near-zero source"):
        cli._run_v15_joint_pair_near_zero_tune_campaign(
            V15_CONFIG,
            output,
            resume=False,
            target_success_count=1,
            workers=1,
            seed=20260821,
            evidence_anchor_paths=("unexpected",),
        )
