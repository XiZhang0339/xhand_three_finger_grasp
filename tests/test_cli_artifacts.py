from __future__ import annotations

import copy
import json
import subprocess
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

import xhand_grasp.cli as cli
from xhand_grasp.artifacts import (
    default_artifact_path,
    file_sha256,
    json_text,
    resolved_run_config,
    write_json,
)
from xhand_grasp.config import load_config, validate_config
from xhand_grasp.evaluation import evaluate_trace
from xhand_grasp.scene import build_model
from xhand_grasp.search import normalized_acceptance_margins
from xhand_grasp.trajectory import _phase_steps


ROOT = Path(__file__).resolve().parents[1]
V1_CONFIG = ROOT / "grasp_configs" / "left_three_finger_cube.json"
V2_CONFIG = ROOT / "grasp_configs" / "left_opposed_face_palm_down.json"
LARGE_CONFIG = ROOT / "grasp_configs" / "left_opposed_face_palm_down_large_cube.json"


def _run_args(
    config_path: Path,
    output_dir: Path,
    *,
    video: bool = False,
    no_trace: bool = False,
) -> Namespace:
    return Namespace(
        config=str(config_path),
        output_dir=str(output_dir),
        video=video,
        video_filename="smoke.mp4",
        no_trace=no_trace,
        edge_mm=None,
        mass_g=None,
        friction=None,
    )


@pytest.mark.parametrize(
    ("config_path", "command", "expected"),
    [
        (V1_CONFIG, "run", "artifacts/left_three_finger_cube/run"),
        (V1_CONFIG, "tune", "artifacts/left_three_finger_cube/tune"),
        (
            V1_CONFIG,
            "robustness",
            "artifacts/left_three_finger_cube/robustness.json",
        ),
        (
            V2_CONFIG,
            "run",
            "artifacts/left_opposed_face_palm_down/nominal",
        ),
        (V2_CONFIG, "tune", "artifacts/left_opposed_face_palm_down/tune"),
        (
            V2_CONFIG,
            "robustness",
            "artifacts/left_opposed_face_palm_down/robustness.json",
        ),
        (
            LARGE_CONFIG,
            "run",
            "artifacts/left_opposed_face_palm_down_large_cube/nominal",
        ),
        (
            LARGE_CONFIG,
            "tune",
            "artifacts/left_opposed_face_palm_down_large_cube/tune",
        ),
        (
            LARGE_CONFIG,
            "robustness",
            "artifacts/left_opposed_face_palm_down_large_cube/robustness.json",
        ),
    ],
)
def test_registered_experiment_default_artifact_paths(
    config_path, command, expected
):
    config = load_config(config_path)

    assert default_artifact_path(config, command) == Path(expected)


def test_explicit_cli_artifact_path_is_not_rerouted():
    config = load_config(V2_CONFIG)

    assert cli._resolved_default_output(
        config,
        "custom/output",
        cli.LEGACY_RUN_OUTPUT,
        "run",
    ) == "custom/output"
    assert cli._resolved_default_output(
        config,
        cli.LEGACY_RUN_OUTPUT,
        cli.LEGACY_RUN_OUTPUT,
        "run",
    ) == cli.LEGACY_RUN_OUTPUT


def test_large_tune_default_persists_full_campaign_result(tmp_path, monkeypatch):
    config = load_config(LARGE_CONFIG)
    best = {
        "config": copy.deepcopy(config),
        "material_policy": "constant_density",
        "summary": {
            "passed": False,
            "failed_checks": ["synthetic_near_miss"],
            "metrics": {"median_lift_m": 0.009},
        },
        "local_perturbation_probe": {"passes": 0, "trial_count": 16},
    }
    campaign_result = {
        "experiment_id": config["experiment_id"],
        "seed": 20260821,
        "workers": 1,
        "candidate_count": 2,
        "simulation_count": 2,
        "perturbation_probe_count": 16,
        "passing_candidates": 0,
        "best": best,
        "best_fixed_mass": {
            "config": copy.deepcopy(config),
            "summary": copy.deepcopy(best["summary"]),
            "material_policy": "fixed_20g_control",
        },
        "top_candidates": [best],
        "local_perturbation_probes": [],
        "campaign_kind": "large_cube_size",
        "campaign_status": "not_validated",
        "size_stages": {"coarse": [{"edge_mm": 36, "sample_count": 35000}]},
        "size_summaries": [{"edge_mm": 36, "hard_passes": 0}],
        "boundary_diagnostics": {"boundary_limited": False},
        "boundary_limited": False,
        "stopping_reason": "fixed_mass_no_hard_pass",
    }
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "preflight_config", lambda value: value)
    monkeypatch.setattr(cli, "run_metadata", lambda path: {"config": str(path)})
    monkeypatch.setattr(cli, "tune", lambda *args, **kwargs: campaign_result)
    args = cli.build_parser().parse_args(
        ["tune", "--config", str(LARGE_CONFIG), "--workers", "1"]
    )

    assert args.func(args) == 2
    output_dir = (
        tmp_path / "artifacts" / "left_opposed_face_palm_down_large_cube" / "tune"
    )
    persisted = json.loads(
        (output_dir / "tune_results.json").read_text(encoding="utf-8")
    )
    best_config = json.loads(
        (output_dir / "best_config.json").read_text(encoding="utf-8")
    )
    fixed_config = json.loads(
        (output_dir / "best_fixed_mass_config.json").read_text(encoding="utf-8")
    )
    density_config = json.loads(
        (output_dir / "best_constant_density_config.json").read_text(
            encoding="utf-8"
        )
    )
    assert persisted["campaign_kind"] == "large_cube_size"
    assert persisted["size_stages"] == campaign_result["size_stages"]
    assert persisted["stopping_reason"] == "fixed_mass_no_hard_pass"
    assert persisted["input_config"] == config
    assert best_config == config
    assert fixed_config == config
    assert density_config == config
    assert persisted["artifacts"]["best_fixed_mass_config"] == (
        "best_fixed_mass_config.json"
    )


def test_tune_and_robustness_refuse_to_overwrite_before_simulation(
    tmp_path, monkeypatch
):
    tune_output = tmp_path / "existing-tune"
    tune_output.mkdir()
    robustness_output = tmp_path / "existing-robustness.json"
    robustness_output.write_text("keep\n", encoding="utf-8")
    monkeypatch.setattr(cli, "preflight_config", lambda value: value)
    monkeypatch.setattr(
        cli,
        "tune",
        lambda *args, **kwargs: pytest.fail("tune must not run"),
    )
    monkeypatch.setattr(
        cli,
        "robustness",
        lambda *args, **kwargs: pytest.fail("robustness must not run"),
    )

    tune_args = cli.build_parser().parse_args(
        [
            "tune",
            "--config",
            str(LARGE_CONFIG),
            "--output-dir",
            str(tune_output),
        ]
    )
    with pytest.raises(FileExistsError, match="already exists"):
        tune_args.func(tune_args)

    robustness_args = Namespace(
        config=str(LARGE_CONFIG),
        output=str(robustness_output),
        workers=1,
        seed=20260821,
    )
    with pytest.raises(FileExistsError, match="already exists"):
        cli.command_robustness(robustness_args)
    assert robustness_output.read_text(encoding="utf-8") == "keep\n"


@pytest.mark.parametrize(
    ("mass_kind", "expected_classification", "campaign_passed"),
    [
        ("constant_density", "validated_constant_density", True),
        ("fixed_20g", "validated_fixed_mass_ablation", False),
        ("noncanonical", "validated_noncanonical_material", False),
    ],
)
def test_large_run_status_preserves_material_policy(
    mass_kind, expected_classification, campaign_passed
):
    config = load_config(LARGE_CONFIG)
    edge_m = float(config["cube"]["edge_m"])
    density_mass_kg = 0.020 * (edge_m / 0.030) ** 3
    config["cube"]["mass_kg"] = {
        "constant_density": density_mass_kg,
        "fixed_20g": 0.020,
        "noncanonical": 0.030,
    }[mass_kind]

    resolved = resolved_run_config(
        config,
        {"passed": True, "failed_checks": []},
    )

    assert resolved["experiment_status"]["classification"] == (
        expected_classification
    )
    assert resolved["experiment_status"]["hard_constraints_passed"] is True
    assert resolved["experiment_status"]["passed"] is campaign_passed
    assert resolved["experiment_status"]["campaign_validated"] is campaign_passed
    assert resolved["experiment_status"]["constant_density_passed"] is (
        mass_kind == "constant_density"
    )
    assert resolved["experiment_status"]["fixed_mass_discovery_passed"] is (
        mass_kind == "fixed_20g"
    )
    assert resolved["experiment_status"]["input_classification"] == (
        "unvalidated_fixed_mass_search_template"
    )


def test_large_failed_run_status_is_not_material_validated():
    config = load_config(LARGE_CONFIG)
    config["cube"]["mass_kg"] = 0.020 * (config["cube"]["edge_m"] / 0.030) ** 3

    resolved = resolved_run_config(
        config,
        {"passed": False, "failed_checks": ["index_contact_duty"]},
    )

    assert resolved["experiment_status"]["classification"] == "failed_run"
    assert resolved["experiment_status"]["passed"] is False
    assert resolved["experiment_status"]["hard_constraints_passed"] is False
    assert resolved["experiment_status"]["campaign_validated"] is False


def test_large_reference_density_with_non_nominal_friction_is_noncanonical():
    config = load_config(LARGE_CONFIG)
    config["cube"]["mass_kg"] = 0.020 * (
        config["cube"]["edge_m"] / 0.030
    ) ** 3
    config["cube"]["friction"] = 1.2

    resolved = resolved_run_config(
        config,
        {"passed": True, "failed_checks": []},
    )

    assert resolved["experiment_status"]["classification"] == (
        "validated_noncanonical_material"
    )
    assert resolved["experiment_status"]["hard_constraints_passed"] is True
    assert resolved["experiment_status"]["campaign_validated"] is False


def test_large_fixed_mass_hard_pass_keeps_run_exit_nonzero(tmp_path, monkeypatch):
    config = load_config(LARGE_CONFIG)
    config_path = tmp_path / "large-fixed.json"
    output_dir = tmp_path / "large-fixed-run"
    write_json(config_path, config)
    summary = {"passed": True, "failed_checks": [], "metrics": {}}
    monkeypatch.setattr(cli, "run_simulation", lambda *args, **kwargs: summary)
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

    return_code = cli.command_run(_run_args(config_path, output_dir, no_trace=True))

    result = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    assert return_code == 2
    assert result["summary"]["passed"] is True
    assert result["experiment_status"]["hard_constraints_passed"] is True
    assert result["experiment_status"]["campaign_validated"] is False
    assert result["experiment_status"]["classification"] == (
        "validated_fixed_mass_ablation"
    )


def test_large_fixed_mass_only_tune_keeps_exit_nonzero(tmp_path, monkeypatch):
    config = load_config(LARGE_CONFIG)
    best_config = copy.deepcopy(config)
    best_config["experiment_status"] = {
        "classification": "validated_fixed_mass_ablation_only",
        "passed": False,
        "hard_constraints_passed": True,
        "fixed_mass_discovery_passed": True,
        "constant_density_passed": False,
    }
    best = {
        "config": best_config,
        "summary": {"passed": True, "failed_checks": [], "metrics": {}},
        "local_perturbation_probe": {"passes": 16, "trial_count": 16},
    }
    result = {
        "candidate_count": 1,
        "simulation_count": 17,
        "perturbation_probe_count": 16,
        "passing_candidates": 1,
        "best": best,
        "top_candidates": [best],
        "local_perturbation_probes": [],
    }
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "preflight_config", lambda value: value)
    monkeypatch.setattr(cli, "run_metadata", lambda path: {"config": str(path)})
    monkeypatch.setattr(cli, "tune", lambda *args, **kwargs: result)
    args = cli.build_parser().parse_args(
        ["tune", "--config", str(LARGE_CONFIG), "--workers", "1"]
    )

    assert args.func(args) == 2


def test_run_artifact_is_atomic_hashed_self_consistent_and_decodable(tmp_path):
    config = load_config(V2_CONFIG)
    config["timing"] = {
        "settle_s": 0.05,
        "pregrasp_s": 0.05,
        "lift_s": 0.05,
        "hold_s": 0.05,
    }
    config_path = tmp_path / "input.json"
    output_dir = tmp_path / "published-run"
    write_json(config_path, config)

    return_code = cli.command_run(
        _run_args(config_path, output_dir, video=True)
    )

    assert output_dir.is_dir()
    assert not list(tmp_path.glob(".published-run.staging.*"))
    result = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    resolved = json.loads(
        (output_dir / "resolved_config.json").read_text(encoding="utf-8")
    )
    assert return_code == (0 if result["summary"]["passed"] else 2)
    assert result["config"] == resolved
    assert result["experiment_status"] == resolved["experiment_status"]
    status = result["experiment_status"]
    assert status["passed"] is result["summary"]["passed"]
    assert status["failed_checks"] == result["summary"]["failed_checks"]
    assert status["classification"] in {"validated_run", "failed_run"}
    assert status["classification"] != "initial_near_miss"
    assert status["input_classification"] == "initial_near_miss"
    expected_margins = normalized_acceptance_margins(
        result["summary"]["metrics"],
        resolved["acceptance"],
        contact_topology=resolved["contact_topology"],
    )
    assert result["normalized_acceptance_margins"] == expected_margins
    limiting_metric = min(expected_margins, key=expected_margins.get)
    assert result["limiting_metric"] == limiting_metric
    assert result["minimum_normalized_acceptance_margin"] == pytest.approx(
        expected_margins[limiting_metric]
    )
    assert result["minimum_normalized_acceptance_margin"] < 0.0

    expected_files = {
        "resolved_config": output_dir / "resolved_config.json",
        "trace": output_dir / "trace.npz",
        "video": output_dir / "smoke.mp4",
    }
    for artifact_name, artifact_path in expected_files.items():
        assert result["artifacts"][artifact_name] == artifact_path.name
        assert result["artifacts"]["sha256"][artifact_name] == file_sha256(
            artifact_path
        )
    assert result["metadata"]["config_sha256"] == result["artifacts"]["sha256"][
        "resolved_config"
    ]
    assert result["metadata"]["source_config_sha256"] == file_sha256(config_path)

    with np.load(output_dir / "trace.npz", allow_pickle=False) as archive:
        traces = {key: archive[key].copy() for key in archive.files}
    assert traces["face_order"].tolist() == [
        "+X",
        "-X",
        "+Y",
        "-Y",
        "+Z",
        "-Z",
        "EDGE_CORNER",
        "UNKNOWN",
    ]
    assert traces["finger_order"].tolist() == ["thumb", "index", "mid"]
    assert result["summary"]["video"]["simulation_step_indices"] == traces[
        "video_frame_steps"
    ].tolist()

    model, info = build_model(resolved)
    recomputed = evaluate_trace(
        model,
        info,
        resolved,
        _phase_steps(model, resolved),
        traces,
    )
    persisted_without_video = copy.deepcopy(result["summary"])
    persisted_without_video.pop("video")
    assert json_text(recomputed) == json_text(persisted_without_video)

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=codec_name,nb_read_frames",
            "-of",
            "json",
            str(output_dir / "smoke.mp4"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    assert stream["codec_name"] == "h264"
    assert int(stream["nb_read_frames"]) == result["summary"]["video"][
        "frame_count"
    ]


def test_failed_run_does_not_publish_or_leave_staging(tmp_path, monkeypatch):
    output_dir = tmp_path / "failed-run"

    def fail_simulation(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("deliberate simulation failure")

    monkeypatch.setattr(cli, "run_simulation", fail_simulation)
    with pytest.raises(RuntimeError, match="deliberate simulation failure"):
        cli.command_run(_run_args(V2_CONFIG, output_dir, no_trace=True))

    assert not output_dir.exists()
    assert not list(tmp_path.glob(".failed-run.staging.*"))


def test_run_rejects_robustness_result_without_passing_grid_case(tmp_path):
    robustness_path = tmp_path / "robustness.json"
    write_json(robustness_path, {"hardest_passing_config": None})
    args = cli.build_parser().parse_args(
        [
            "run",
            "--hardest-from",
            str(robustness_path),
            "--output-dir",
            str(tmp_path / "must-not-exist"),
        ]
    )

    with pytest.raises(ValueError, match="no usable hardest_passing_config"):
        args.func(args)
    assert not (tmp_path / "must-not-exist").exists()


def test_robustness_persists_simulation_errors_without_margin_crash(
    tmp_path, monkeypatch
):
    config = load_config(V2_CONFIG)
    config_path = tmp_path / "input.json"
    output_path = tmp_path / "robustness-error.json"
    write_json(config_path, config)
    simulation_error = {
        "passed": False,
        "failed_checks": ["simulation_error"],
        "checks": {"simulation_error": False},
        "metrics": {"median_lift_m": None},
        "error": "RuntimeError: synthetic worker failure",
    }
    grid_record = {
        "grid_index": 0,
        "edge_mm": 30,
        "mass_g": 20,
        "friction": 0.8,
        **copy.deepcopy(simulation_error),
    }
    perturbation_record = {
        "trial": 0,
        "cube": copy.deepcopy(config["cube"]),
        **copy.deepcopy(simulation_error),
    }
    monkeypatch.setattr(cli, "run_metadata", lambda path: {"config": str(path)})
    monkeypatch.setattr(
        cli,
        "robustness",
        lambda config, workers, seed: {
            "seed": seed,
            "nominal_passed": False,
            "nominal_summary": copy.deepcopy(simulation_error),
            "grid_case_count": 1,
            "grid_passes": 0,
            "hardest_passing_grid_case": None,
            "grid": [grid_record],
            "perturbation_trial_count": 1,
            "perturbation_passes": 0,
            "required_perturbation_passes": 1,
            "robust_passed": False,
            "perturbations": [perturbation_record],
        },
    )
    args = Namespace(
        config=str(config_path),
        output=str(output_path),
        workers=1,
        seed=20260821,
    )

    assert cli.command_robustness(args) == 2
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["nominal_normalized_acceptance_margins"] == {}
    assert result["nominal_minimum_normalized_acceptance_margin"] is None
    assert result["nominal_limiting_metric"] == "simulation_error"
    for record in (result["grid"][0], result["perturbations"][0]):
        assert record["error"] == "RuntimeError: synthetic worker failure"
        assert record["normalized_acceptance_margins"] == {}
        assert record["minimum_normalized_acceptance_margin"] is None
        assert record["limiting_metric"] == "simulation_error"
    assert result["grid"][0]["normalized_acceptance_margin"] is None
    assert result["hardest_passing_config"] is None


def test_robustness_persists_full_runnable_hardest_case(tmp_path, monkeypatch):
    config = load_config(V2_CONFIG)
    config_path = tmp_path / "input.json"
    output_path = tmp_path / "robustness.json"
    write_json(config_path, config)
    hardest = {
        "grid_index": 60,
        "edge_mm": 32,
        "mass_g": 10,
        "friction": 0.4,
        "passed": True,
        "failed_checks": [],
        "metrics": {"synthetic_margin": 0.01},
        "normalized_acceptance_margin": 0.01,
        "limiting_metric": "median_lift_m",
    }
    failed_grid = {
        **hardest,
        "grid_index": 0,
        "edge_mm": 26,
        "passed": False,
        "failed_checks": ["synthetic_failure"],
        "metrics": {"synthetic_margin": -0.4},
        "normalized_acceptance_margin": None,
        "limiting_metric": None,
    }
    failed_perturbation = {
        "trial": 0,
        "passed": False,
        "failed_checks": ["synthetic_failure"],
        "cube": config["cube"],
        "metrics": {"synthetic_margin": -0.2},
    }

    monkeypatch.setattr(cli, "run_metadata", lambda path: {"config": str(path)})
    monkeypatch.setattr(
        cli,
        "normalized_acceptance_margins",
        lambda metrics, acceptance, contact_topology: {
            "synthetic_margin": float(metrics["synthetic_margin"])
        },
    )
    monkeypatch.setattr(
        cli,
        "robustness",
        lambda config, workers, seed: {
            "seed": seed,
            "nominal_passed": True,
            "nominal_summary": {
                "passed": True,
                "metrics": {"synthetic_margin": 0.2},
            },
            "grid_case_count": 100,
            "grid_passes": 1,
            "hardest_passing_grid_case": hardest,
            "grid": [hardest, failed_grid],
            "perturbation_trial_count": 50,
            "perturbation_passes": 50,
            "required_perturbation_passes": 45,
            "robust_passed": True,
            "perturbations": [failed_perturbation],
        },
    )
    args = Namespace(
        config=str(config_path),
        output=str(output_path),
        workers=1,
        seed=20260821,
    )

    assert cli.command_robustness(args) == 0
    result = json.loads(output_path.read_text(encoding="utf-8"))
    hardest_config = result["hardest_passing_config"]
    assert hardest_config["cube"]["edge_m"] == pytest.approx(0.032)
    assert hardest_config["cube"]["mass_kg"] == pytest.approx(0.010)
    assert hardest_config["cube"]["friction"] == pytest.approx(0.4)
    assert hardest_config["experiment_status"]["passed"] is True
    validate_config(hardest_config)
    assert result["nominal_minimum_normalized_acceptance_margin"] == pytest.approx(
        0.2
    )
    assert result["grid"][0]["normalized_acceptance_margins"] == {
        "synthetic_margin": 0.01
    }
    assert result["grid"][1]["minimum_normalized_acceptance_margin"] == pytest.approx(
        -0.4
    )
    assert result["grid"][1]["normalized_acceptance_margin"] == pytest.approx(-0.4)
    assert result["grid"][1]["limiting_metric"] == "synthetic_margin"
    assert result["perturbations"][0][
        "minimum_normalized_acceptance_margin"
    ] == pytest.approx(-0.2)

    rerun_dir = tmp_path / "hardest-rerun"

    def fake_simulation(config, *, trace_path, video_path):
        assert config["cube"]["edge_m"] == pytest.approx(0.032)
        assert config["cube"]["mass_kg"] == pytest.approx(0.010)
        assert config["cube"]["friction"] == pytest.approx(0.4)
        assert trace_path is None
        assert video_path is None
        return {
            "passed": True,
            "failed_checks": [],
            "metrics": {"synthetic_margin": 0.01},
        }

    monkeypatch.setattr(cli, "run_simulation", fake_simulation)
    run_args = cli.build_parser().parse_args(
        [
            "run",
            "--hardest-from",
            str(output_path),
            "--output-dir",
            str(rerun_dir),
            "--no-trace",
        ]
    )
    assert run_args.func(run_args) == 0
    rerun = json.loads((rerun_dir / "result.json").read_text(encoding="utf-8"))
    assert rerun["metadata"]["source_config"] == str(output_path.resolve())
    assert rerun["metadata"]["source_config_selector"] == "hardest_passing_config"
    assert rerun["metadata"]["source_config_sha256"] == file_sha256(output_path)
    assert rerun["config"]["cube"] == hardest_config["cube"]


def test_schema_v1_resolved_config_remains_unchanged(tmp_path, monkeypatch):
    config = load_config(V1_CONFIG)
    config_path = tmp_path / "v1.json"
    output_dir = tmp_path / "v1-run"
    write_json(config_path, config)

    def fake_simulation(config, *, trace_path, video_path):
        assert trace_path is None
        assert video_path is None
        return {"passed": False, "failed_checks": ["synthetic"]}

    monkeypatch.setattr(cli, "run_simulation", fake_simulation)
    monkeypatch.setattr(
        cli,
        "run_metadata",
        lambda path: {"config_sha256": file_sha256(path)},
    )
    assert cli.command_run(_run_args(config_path, output_dir, no_trace=True)) == 2
    result = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    assert result["config"] == config
    assert "experiment_status" not in result
    assert "normalized_acceptance_margins" not in result
    assert "minimum_normalized_acceptance_margin" not in result
    assert "limiting_metric" not in result
