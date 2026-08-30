from __future__ import annotations

import json
from pathlib import Path

import pytest

import xhand_grasp.cli as cli


CONFIG = Path(
    "grasp_configs/left_opposed_face_palm_down_contact_preserving_planned_lift.json"
)


def test_tune_parser_exposes_v14_rescue_source() -> None:
    args = cli.build_parser().parse_args(
        [
            "tune",
            "--rescue-from",
            "source_campaign",
            "--output-dir",
            "new_campaign",
        ]
    )
    assert args.rescue_from == "source_campaign"


def test_tune_parser_exposes_v14_event_rescue_source() -> None:
    args = cli.build_parser().parse_args(
        [
            "tune",
            "--event-rescue-from",
            "completed_rescue",
            "--output-dir",
            "event_rescue",
        ]
    )
    assert args.event_rescue_from == "completed_rescue"


def test_tune_parser_exposes_v14_adaptive_event_rescue_source() -> None:
    args = cli.build_parser().parse_args(
        [
            "tune",
            "--adaptive-event-rescue-from",
            "completed_event_rescue",
            "--output-dir",
            "adaptive_event_rescue",
        ]
    )
    assert args.adaptive_event_rescue_from == "completed_event_rescue"


def test_tune_parser_exposes_v14_force_debias_rescue_source() -> None:
    args = cli.build_parser().parse_args(
        [
            "tune",
            "--force-debias-rescue-from",
            "completed_adaptive_rescue",
            "--output-dir",
            "force_debias_rescue",
        ]
    )
    assert args.force_debias_rescue_from == "completed_adaptive_rescue"


def test_tune_parser_exposes_v14_contact_mode_pose_rescue_source() -> None:
    args = cli.build_parser().parse_args(
        [
            "tune",
            "--contact-mode-pose-rescue-from",
            "retained_candidate",
            "--output-dir",
            "contact_mode_rescue",
        ]
    )
    assert args.contact_mode_pose_rescue_from == "retained_candidate"


def test_v14_contact_mode_pose_rescue_dispatches_to_separate_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "retained_candidate"
    source.mkdir()
    output = tmp_path / "contact_mode_rescue"
    observed: dict[str, object] = {}

    def fake_runner(config_path, output_dir, **kwargs):
        observed.update(
            {"config_path": config_path, "output_dir": output_dir, **kwargs}
        )
        return {
            "candidate_count": 256,
            "source_reproduction_candidate_count": 1,
            "new_unique_candidate_count": 255,
            "full_success_count": 1,
            "catalogs": {"manipulation": "catalogs/manipulation/catalog.json"},
        }

    monkeypatch.setattr(
        cli, "_load_contact_mode_pose_rescue_runner", lambda: fake_runner
    )
    args = cli.build_parser().parse_args(
        [
            "tune",
            "--config",
            str(CONFIG),
            "--contact-mode-pose-rescue-from",
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
        "source_candidate_directory": source.resolve(),
        "resume": False,
        "target_success_count": 1,
        "workers": 4,
        "seed": 20260821,
    }
    console = json.loads(capsys.readouterr().out)
    assert console["campaign_kind"] == "contact_preserving_contact_mode_pose_rescue"
    assert console["source_reproduction_candidate_count"] == 1
    assert console["new_unique_candidate_count"] == 255
    assert console["target_reached"] is True


def test_v14_force_debias_rescue_dispatches_to_separate_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "completed_adaptive_rescue"
    source.mkdir()
    output = tmp_path / "force_debias_rescue"
    observed: dict[str, object] = {}

    def fake_runner(config_path, output_dir, **kwargs):
        observed.update(
            {"config_path": config_path, "output_dir": output_dir, **kwargs}
        )
        return {
            "discovery_candidate_count": 160,
            "refinement_candidate_count": 512,
            "physical_unique_candidate_count": 672,
            "full_success_count": 1,
            "catalogs": {"manipulation": "catalogs/manipulation/catalog.json"},
        }

    monkeypatch.setattr(
        cli,
        "_load_contact_preserving_force_debias_rescue_runner",
        lambda: fake_runner,
    )
    args = cli.build_parser().parse_args(
        [
            "tune",
            "--config",
            str(CONFIG),
            "--force-debias-rescue-from",
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
        "source_adaptive_campaign": source.resolve(),
        "resume": False,
        "target_success_count": 1,
        "workers": 4,
        "seed": 20260821,
    }
    console = json.loads(capsys.readouterr().out)
    assert console["campaign_kind"] == "contact_preserving_force_debias_rescue"
    assert console["target_reached"] is True


def test_v14_adaptive_event_rescue_dispatches_to_separate_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "completed_event_rescue"
    source.mkdir()
    output = tmp_path / "adaptive_event_rescue"
    observed: dict[str, object] = {}

    def fake_runner(config_path, output_dir, **kwargs):
        observed.update(
            {"config_path": config_path, "output_dir": output_dir, **kwargs}
        )
        return {
            "diagnostic_candidate_count": 64,
            "exploration_candidate_count": 512,
            "local_refinement_candidate_count": 512,
            "physical_unique_candidate_count": 1088,
            "full_success_count": 1,
            "catalogs": {"manipulation": "catalogs/manipulation/catalog.json"},
        }

    monkeypatch.setattr(
        cli,
        "_load_contact_preserving_adaptive_event_rescue_runner",
        lambda: fake_runner,
    )
    args = cli.build_parser().parse_args(
        [
            "tune",
            "--config",
            str(CONFIG),
            "--adaptive-event-rescue-from",
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
        "source_event_campaign": source.resolve(),
        "resume": False,
        "target_success_count": 1,
        "workers": 4,
        "seed": 20260821,
    }
    console = json.loads(capsys.readouterr().out)
    assert console["campaign_kind"] == "contact_witness_adaptive_event_rescue"
    assert console["target_reached"] is True


def test_v14_event_rescue_dispatches_to_separate_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "completed_rescue"
    source.mkdir()
    output = tmp_path / "event_rescue"
    observed: dict[str, object] = {}

    def fake_runner(config_path, output_dir, **kwargs):
        observed.update(
            {"config_path": config_path, "output_dir": output_dir, **kwargs}
        )
        return {
            "exploration_candidate_count": 512,
            "local_refinement_candidate_count": 512,
            "full_success_count": 1,
            "catalogs": {"manipulation": "catalogs/manipulation/catalog.json"},
        }

    monkeypatch.setattr(
        cli, "_load_contact_preserving_event_rescue_runner", lambda: fake_runner
    )
    args = cli.build_parser().parse_args(
        [
            "tune",
            "--config",
            str(CONFIG),
            "--event-rescue-from",
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
        "source_rescue_campaign": source.resolve(),
        "resume": False,
        "target_success_count": 1,
        "workers": 4,
        "seed": 20260821,
    }
    console = json.loads(capsys.readouterr().out)
    assert console["campaign_kind"] == "contact_witness_event_rescue"
    assert console["target_reached"] is True


def test_v14_event_rescue_rejects_conflicting_sources(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    args = cli.build_parser().parse_args(
        [
            "tune",
            "--config",
            str(CONFIG),
            "--rescue-from",
            str(source),
            "--event-rescue-from",
            str(source),
            "--output-dir",
            str(tmp_path / "new"),
        ]
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        cli.command_tune(args)

    adaptive = cli.build_parser().parse_args(
        [
            "tune",
            "--config",
            str(CONFIG),
            "--event-rescue-from",
            str(source),
            "--adaptive-event-rescue-from",
            str(source),
            "--output-dir",
            str(tmp_path / "adaptive"),
        ]
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        cli.command_tune(adaptive)

    force_debias = cli.build_parser().parse_args(
        [
            "tune",
            "--config",
            str(CONFIG),
            "--adaptive-event-rescue-from",
            str(source),
            "--force-debias-rescue-from",
            str(source),
            "--output-dir",
            str(tmp_path / "force_debias"),
        ]
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        cli.command_tune(force_debias)

    contact_mode = cli.build_parser().parse_args(
        [
            "tune",
            "--config",
            str(CONFIG),
            "--micro-jerk-rescue-from",
            str(source),
            "--contact-mode-pose-rescue-from",
            str(source),
            "--output-dir",
            str(tmp_path / "contact_mode"),
        ]
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        cli.command_tune(contact_mode)


def test_v14_rescue_dispatch_is_separate_and_forwards_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "rescue"
    observed: dict[str, object] = {}

    def fake_runner(config_path, output_dir, **kwargs):
        observed.update(
            {"config_path": config_path, "output_dir": output_dir, **kwargs}
        )
        return {
            "phase_one_candidate_count": 1024,
            "time_warp_candidate_count": 256,
            "full_success_count": 1,
            "catalogs": {"manipulation": "catalogs/manipulation/catalog.json"},
        }

    monkeypatch.setattr(
        cli, "_load_contact_preserving_rescue_runner", lambda: fake_runner
    )
    args = cli.build_parser().parse_args(
        [
            "tune",
            "--config",
            str(CONFIG),
            "--rescue-from",
            str(source),
            "--output-dir",
            str(output),
            "--workers",
            "3",
            "--target-success-count",
            "1",
        ]
    )
    assert cli.command_tune(args) == 0
    assert observed == {
        "config_path": CONFIG.resolve(),
        "output_dir": output.resolve(),
        "source_campaign": source.resolve(),
        "reuse_refinement_from": None,
        "resume": False,
        "target_success_count": 1,
        "workers": 3,
        "seed": 20260821,
    }
    console = json.loads(capsys.readouterr().out)
    assert console["campaign_kind"] == "contact_preserving_post_campaign_rescue"
    assert console["target_reached"] is True


def test_v14_rescue_rejects_default_or_nested_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    parser = cli.build_parser()
    missing_output = parser.parse_args(
        ["tune", "--config", str(CONFIG), "--rescue-from", str(source)]
    )
    with pytest.raises(ValueError, match="explicit new --output-dir"):
        cli.command_tune(missing_output)

    nested = parser.parse_args(
        [
            "tune",
            "--config",
            str(CONFIG),
            "--rescue-from",
            str(source),
            "--output-dir",
            str(source / "rescue"),
        ]
    )
    with pytest.raises(ValueError, match="outside the immutable source"):
        cli.command_tune(nested)


def test_non_v14_experiment_rejects_rescue(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    args = cli.build_parser().parse_args(
        [
            "tune",
            "--config",
            "grasp_configs/left_three_finger_cube.json",
            "--rescue-from",
            str(source),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )
    with pytest.raises(ValueError, match="schema-v14"):
        cli.command_tune(args)


def test_refinement_reuse_requires_rescue_and_is_forwarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    reuse = tmp_path / "reuse"
    source.mkdir()
    reuse.mkdir()
    parser = cli.build_parser()
    missing_rescue = parser.parse_args(
        [
            "tune",
            "--reuse-refinement-from",
            str(reuse),
            "--output-dir",
            str(tmp_path / "unused"),
        ]
    )
    with pytest.raises(ValueError, match="requires --rescue-from"):
        cli.command_tune(missing_rescue)

    observed: dict[str, object] = {}

    def fake_runner(config_path, output_dir, **kwargs):
        observed.update(kwargs)
        return {"phase_one_candidate_count": 1024, "full_success_count": 1}

    monkeypatch.setattr(
        cli, "_load_contact_preserving_rescue_runner", lambda: fake_runner
    )
    args = parser.parse_args(
        [
            "tune",
            "--config",
            str(CONFIG),
            "--rescue-from",
            str(source),
            "--reuse-refinement-from",
            str(reuse),
            "--output-dir",
            str(tmp_path / "new"),
        ]
    )
    assert cli.command_tune(args) == 0
    assert observed["reuse_refinement_from"] == reuse.resolve()


def test_v14_robustness_sidecar_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = tmp_path / "catalog.json"
    result = tmp_path / "result.json"
    catalog.write_text("{}", encoding="utf-8")
    result.write_text("{}", encoding="utf-8")
    output = tmp_path / "robustness_sidecar"
    observed: dict[str, object] = {}

    def fake_command(args):
        observed.update(vars(args))
        return 0

    import xhand_grasp.tuning.contact_preserving_robustness_sidecar as sidecar

    monkeypatch.setattr(
        sidecar, "command_contact_preserving_robustness_sidecar", fake_command
    )
    args = cli.build_parser().parse_args(
        [
            "robustness",
            "--config",
            str(CONFIG),
            "--search-root",
            str(catalog),
            "--contact-preserving-source-result",
            str(result),
            "--contact-preserving-sidecar-output-dir",
            str(output),
            "--workers",
            "3",
        ]
    )
    assert cli.command_robustness(args) == 0
    assert observed == {
        "catalog": str(catalog),
        "source_result": str(result),
        "output_dir": str(output),
        "resume": False,
        "workers": 3,
        "seed": 20260821,
    }


def test_v14_robustness_sidecar_requires_exact_source_pair(tmp_path: Path) -> None:
    args = cli.build_parser().parse_args(
        [
            "robustness",
            "--config",
            str(CONFIG),
            "--contact-preserving-sidecar-output-dir",
            str(tmp_path / "sidecar"),
        ]
    )
    with pytest.raises(ValueError, match="exactly one --search-root"):
        cli.command_robustness(args)
