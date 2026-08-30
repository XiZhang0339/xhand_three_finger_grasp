from __future__ import annotations

import json
from pathlib import Path

import pytest

import xhand_grasp.cli as cli
import xhand_grasp.tuning.actual_contact_grasp_pose_robustness as robustness_module
import xhand_grasp.tuning.relative_wrist_pose_post_validation as post_validation_module


ROOT = Path(__file__).resolve().parents[1]
V10_CONFIG = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_larger_actual_contact_grasp_pose_smooth_vertical_lift.json"
)
V11_CONFIG = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_larger_relative_wrist_pose_actual_contact_smooth_vertical_lift.json"
)


def _generic_report(*, robust_passed: bool = True) -> dict:
    return {
        "selected_nominal_count": 1,
        "total_perturbation_count": 66,
        "best_robustness": {
            "perturbation_passes": 46 if robust_passed else 44,
            "required_perturbation_passes": 45,
        },
        "robust_passed": robust_passed,
    }


def _post_report(
    *, density_successes: int = 1, pose_friction_passed: bool = True
) -> dict:
    return {
        "fixed_160g": {"full_success_count": 1},
        "constant_density": {"full_success_count": density_successes},
        "best_first_pose_friction": {
            "robust_passed": pose_friction_passed,
        },
    }


def test_v11_robustness_orchestrates_local_and_density_validation(
    tmp_path, monkeypatch, capsys
):
    catalog = tmp_path / "catalog.json"
    catalog.write_text("{}\n", encoding="utf-8")
    generic_output = tmp_path / "robustness.json"
    post_output = tmp_path / "post-validation"
    observed: dict[str, object] = {}

    monkeypatch.setattr(cli, "preflight_config", lambda config: config)
    monkeypatch.setattr(
        post_validation_module,
        "load_canonical_fixed_160g_sources",
        lambda path: observed.setdefault("authenticated_catalog", Path(path))
        and (object(),),
    )

    def fake_post_validation(
        catalog_path,
        output_dir,
        *,
        resume,
        seed,
        workers,
        render_density_videos,
    ):
        observed["post_validation"] = {
            "catalog": Path(catalog_path),
            "output": Path(output_dir),
            "resume": resume,
            "seed": seed,
            "workers": workers,
            "render_density_videos": render_density_videos,
        }
        return _post_report()

    def fake_robustness(search_roots, output_path, *, workers, seed):
        observed["robustness"] = {
            "search_roots": tuple(Path(value) for value in search_roots),
            "output": Path(output_path),
            "workers": workers,
            "seed": seed,
        }
        return _generic_report()

    monkeypatch.setattr(
        post_validation_module,
        "run_relative_wrist_pose_post_validation",
        fake_post_validation,
    )
    monkeypatch.setattr(
        robustness_module, "run_v9_robustness_campaign", fake_robustness
    )
    arguments = cli.build_parser().parse_args(
        [
            "robustness",
            "--config",
            str(V11_CONFIG),
            "--search-root",
            str(catalog),
            "--output",
            str(generic_output),
            "--post-validation-output-dir",
            str(post_output),
            "--resume-post-validation",
            "--render-density-videos",
            "--workers",
            "3",
            "--seed",
            "17",
        ]
    )

    assert cli.command_robustness(arguments) == 0
    assert observed["authenticated_catalog"] == catalog.resolve()
    assert observed["post_validation"] == {
        "catalog": catalog.resolve(),
        "output": post_output.resolve(),
        "resume": True,
        "seed": 17,
        "workers": 3,
        "render_density_videos": True,
    }
    assert observed["robustness"] == {
        "search_roots": (catalog.resolve(),),
        "output": generic_output.resolve(),
        "workers": 3,
        "seed": 17,
    }
    console = json.loads(capsys.readouterr().out)
    assert console["fixed_mass_robustness_passed"] is True
    assert console["relative_wrist_pose_post_validation"] == {
        "status": "complete",
        "catalog": str(catalog.resolve()),
        "output_dir": str(post_output.resolve()),
        "fixed_160g_full_success_count": 1,
        "constant_density_full_success_count": 1,
        "pose_friction_robust_passed": True,
        "passed": True,
    }
    assert console["robust_passed"] is True


@pytest.mark.parametrize(
    ("density_successes", "pose_friction_passed"),
    ((0, True), (1, False)),
)
def test_v11_robustness_exit_requires_density_and_pose_friction_passes(
    tmp_path,
    monkeypatch,
    capsys,
    density_successes,
    pose_friction_passed,
):
    catalog = tmp_path / "catalog.json"
    catalog.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(cli, "preflight_config", lambda config: config)
    monkeypatch.setattr(
        post_validation_module,
        "load_canonical_fixed_160g_sources",
        lambda path: (object(),),
    )
    monkeypatch.setattr(
        post_validation_module,
        "run_relative_wrist_pose_post_validation",
        lambda *args, **kwargs: _post_report(
            density_successes=density_successes,
            pose_friction_passed=pose_friction_passed,
        ),
    )
    monkeypatch.setattr(
        robustness_module,
        "run_v9_robustness_campaign",
        lambda *args, **kwargs: _generic_report(),
    )
    arguments = cli.build_parser().parse_args(
        [
            "robustness",
            "--config",
            str(V11_CONFIG),
            "--search-root",
            str(catalog),
            "--output",
            str(tmp_path / "robustness.json"),
            "--post-validation-output-dir",
            str(tmp_path / "post-validation"),
        ]
    )

    assert cli.command_robustness(arguments) == 2
    console = json.loads(capsys.readouterr().out)
    assert console["fixed_mass_robustness_passed"] is True
    assert console["relative_wrist_pose_post_validation"]["passed"] is False
    assert console["robust_passed"] is False


def test_v10_robustness_keeps_existing_runner_and_console(tmp_path, monkeypatch, capsys):
    catalog = tmp_path / "catalog.json"
    catalog.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "robustness.json"
    monkeypatch.setattr(cli, "preflight_config", lambda config: config)
    monkeypatch.setattr(
        robustness_module,
        "run_v9_robustness_campaign",
        lambda *args, **kwargs: _generic_report(),
    )
    monkeypatch.setattr(
        post_validation_module,
        "run_relative_wrist_pose_post_validation",
        lambda *args, **kwargs: pytest.fail("v10 must not run v11 post-validation"),
    )
    arguments = cli.build_parser().parse_args(
        [
            "robustness",
            "--config",
            str(V10_CONFIG),
            "--search-root",
            str(catalog),
            "--output",
            str(output),
        ]
    )

    assert cli.command_robustness(arguments) == 0
    console = json.loads(capsys.readouterr().out)
    assert console == {
        "selected_nominal_count": 1,
        "total_perturbation_count": 66,
        "best_perturbation_passes": 46,
        "required_best_passes": 45,
        "robust_passed": True,
        "output": str(output.resolve()),
    }


def test_relative_post_validation_options_are_rejected_by_v10(
    tmp_path, monkeypatch
):
    catalog = tmp_path / "catalog.json"
    catalog.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(cli, "preflight_config", lambda config: config)
    monkeypatch.setattr(
        robustness_module,
        "run_v9_robustness_campaign",
        lambda *args, **kwargs: pytest.fail("rejected options must not run"),
    )
    arguments = cli.build_parser().parse_args(
        [
            "robustness",
            "--config",
            str(V10_CONFIG),
            "--search-root",
            str(catalog),
            "--post-validation-catalog",
            str(catalog),
        ]
    )

    with pytest.raises(ValueError, match="relative_wrist_pose_search capability"):
        cli.command_robustness(arguments)


def test_v11_without_canonical_full_success_skips_density_but_runs_diagnostics(
    tmp_path, monkeypatch, capsys
):
    catalog = tmp_path / "catalog.json"
    catalog.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(cli, "preflight_config", lambda config: config)

    def no_success(path):
        raise ValueError("catalog contains no canonical fixed-160-g full success")

    monkeypatch.setattr(
        post_validation_module, "load_canonical_fixed_160g_sources", no_success
    )
    monkeypatch.setattr(
        post_validation_module,
        "run_relative_wrist_pose_post_validation",
        lambda *args, **kwargs: pytest.fail("post-validation must stay gated"),
    )
    monkeypatch.setattr(
        robustness_module,
        "run_v9_robustness_campaign",
        lambda *args, **kwargs: _generic_report(robust_passed=False),
    )
    arguments = cli.build_parser().parse_args(
        [
            "robustness",
            "--config",
            str(V11_CONFIG),
            "--search-root",
            str(catalog),
            "--output",
            str(tmp_path / "robustness.json"),
        ]
    )

    assert cli.command_robustness(arguments) == 2
    console = json.loads(capsys.readouterr().out)
    assert console["relative_wrist_pose_post_validation"]["status"] == (
        "not_run_without_fixed_mass_full_success"
    )
    assert console["robust_passed"] is False
