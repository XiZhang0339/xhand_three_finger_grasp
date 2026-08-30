from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import xhand_grasp.cli as cli
from xhand_grasp.artifacts import file_sha256, write_json
from xhand_grasp.config import load_config


ROOT = Path(__file__).resolve().parents[1]
V3_CONFIG = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_larger_cube_grasp_then_lift.json"
)
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
BANDS = (10.0, 12.5, 15.0, 17.5, 20.0)


def _tune_result(config: dict, *, include_selected: bool) -> dict:
    summary = {
        "passed": False,
        "failed_checks": ["synthetic_near_miss"],
        "metrics": {"median_lift_m": 0.009},
        "stage_status": {
            "grasp_success": True,
            "manipulation_success": False,
            "full_success": False,
        },
    }
    result = {
        "experiment_id": config.get("experiment_id"),
        "candidate_count": 5,
        "simulation_count": 5,
        "perturbation_probe_count": 0,
        "passing_candidates": 0,
        "best": {
            "candidate_id": 100,
            "config": copy.deepcopy(config),
            "summary": copy.deepcopy(summary),
            "local_perturbation_probe": {"passes": 0, "trial_count": 0},
        },
    }
    if include_selected:
        result["selected_band_candidates"] = [
            {
                "candidate_id": index,
                "config": copy.deepcopy(config),
                "summary": copy.deepcopy(summary),
                "tilt_band_center_deg": band,
            }
            for index, band in enumerate(BANDS)
        ]
    return result


def _catalog_payload(experiment_id: str) -> dict:
    return {
        "trajectory_catalog_schema_version": 1,
        "experiment_id": experiment_id,
        "declared_tilt_bands_deg": list(BANDS),
        "published_tilt_bands_deg": list(BANDS),
        "missing_published_tilt_bands_deg": [],
        "passing_tilt_bands_deg": [10.0, 15.0],
        "missing_passing_tilt_bands_deg": [12.5, 17.5, 20.0],
        "trajectory_count": 5,
        "passing_trajectory_count": 2,
        "near_miss_trajectory_count": 3,
        "reported_hard_pass_count": 2,
        "all_selected_passes_reproduced": True,
        "all_reruns_full_success": False,
        "all_declared_bands_passed": False,
        "campaign_has_passing_trajectory": True,
        "include_near_misses": True,
        "provenance": {"synthetic": True},
        "trajectories": [],
    }


def _install_common_mocks(monkeypatch, config: dict, tune_result: dict) -> None:
    monkeypatch.setattr(cli, "preflight_config", lambda value: value)
    monkeypatch.setattr(cli, "run_metadata", lambda path: {"config": str(path)})
    monkeypatch.setattr(
        cli,
        "tune",
        lambda *args, **kwargs: copy.deepcopy(tune_result),
    )


@pytest.mark.parametrize(
    ("extra_args", "expected_video"),
    [([], True), (["--no-catalog-video"], False)],
)
def test_schema_v4_tune_transactionally_publishes_catalog(
    tmp_path, monkeypatch, extra_args, expected_video
):
    config = load_config(V4_CONFIG)
    tune_result = _tune_result(config, include_selected=True)
    _install_common_mocks(monkeypatch, config, tune_result)
    calls = []

    def publish(
        candidates,
        output_dir,
        *,
        include_near_misses,
        video,
    ):
        calls.append(
            {
                "candidates": copy.deepcopy(candidates),
                "output_dir": Path(output_dir),
                "include_near_misses": include_near_misses,
                "video": video,
            }
        )
        output = Path(output_dir)
        output.mkdir()
        catalog = _catalog_payload(config["experiment_id"])
        write_json(output / "catalog.json", catalog)
        return catalog

    monkeypatch.setattr(cli, "export_aligned_trajectory_catalog", publish)
    output = tmp_path / ("v4-with-video" if expected_video else "v4-no-video")
    args = cli.build_parser().parse_args(
        [
            "tune",
            "--config",
            str(V4_CONFIG),
            "--output-dir",
            str(output),
            "--workers",
            "1",
            *extra_args,
        ]
    )

    assert args.func(args) == 2

    assert len(calls) == 1
    call = calls[0]
    assert call["candidates"] == tune_result["selected_band_candidates"]
    assert call["output_dir"].name == "trajectory_catalog"
    assert call["output_dir"].parent.name.startswith(f".{output.name}.staging.")
    assert call["include_near_misses"] is True
    assert call["video"] is expected_video

    tune_results = json.loads(
        (output / "tune_results.json").read_text(encoding="utf-8")
    )
    artifacts = tune_results["artifacts"]
    catalog_path = output / artifacts["trajectory_catalog"]
    assert artifacts["trajectory_catalog"] == "trajectory_catalog/catalog.json"
    assert catalog_path.is_file()
    assert artifacts["trajectory_catalog_sha256"] == file_sha256(catalog_path)
    assert artifacts["trajectory_catalog_summary"] == {
        key: _catalog_payload(config["experiment_id"])[key]
        for key in cli._ALIGNED_CATALOG_SUMMARY_FIELDS
    }
    assert "trajectories" not in artifacts["trajectory_catalog_summary"]


def test_schema_v4_catalog_failure_rolls_back_outer_tune_directory(
    tmp_path, monkeypatch
):
    config = load_config(V4_CONFIG)
    _install_common_mocks(
        monkeypatch,
        config,
        _tune_result(config, include_selected=True),
    )
    monkeypatch.setattr(
        cli,
        "export_aligned_trajectory_catalog",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic catalog failure")
        ),
    )
    output = tmp_path / "v4-atomic-failure"
    args = cli.build_parser().parse_args(
        ["tune", "--config", str(V4_CONFIG), "--output-dir", str(output)]
    )

    with pytest.raises(RuntimeError, match="synthetic catalog failure"):
        args.func(args)

    assert not output.exists()
    assert not list(tmp_path.glob(".v4-atomic-failure.staging.*"))


def test_schema_v5_catalog_receives_best_candidate_alias_source(
    tmp_path,
    monkeypatch,
):
    config = load_config(V5_CONFIG)
    tune_result = _tune_result(config, include_selected=True)
    tune_result["selected_band_candidates"][0]["candidate_id"] = 100
    _install_common_mocks(monkeypatch, config, tune_result)
    seen = {}

    def publish(
        candidates,
        output_dir,
        *,
        include_near_misses,
        video,
        best_candidate_id,
    ):
        seen.update(
            best_candidate_id=best_candidate_id,
            candidates=copy.deepcopy(candidates),
        )
        output = Path(output_dir)
        output.mkdir()
        catalog = _catalog_payload(config["experiment_id"])
        catalog["aliases"] = {"best_attempt": "candidate_000100"}
        write_json(output / "catalog.json", catalog)
        return catalog

    monkeypatch.setattr(cli, "export_aligned_trajectory_catalog", publish)
    output = tmp_path / "v5-alias"
    args = cli.build_parser().parse_args(
        [
            "tune",
            "--config",
            str(V5_CONFIG),
            "--output-dir",
            str(output),
            "--workers",
            "1",
            "--no-catalog-video",
        ]
    )

    assert args.func(args) == 2
    assert seen["best_candidate_id"] == 100
    assert seen["candidates"] == tune_result["selected_band_candidates"]


def test_schema_v3_tune_does_not_publish_or_change_legacy_artifact_map(
    tmp_path, monkeypatch
):
    config = load_config(V3_CONFIG)
    _install_common_mocks(
        monkeypatch,
        config,
        _tune_result(config, include_selected=False),
    )
    monkeypatch.setattr(
        cli,
        "export_aligned_trajectory_catalog",
        lambda *args, **kwargs: pytest.fail("schema-v3 must not publish a catalog"),
    )
    output = tmp_path / "v3-tune"
    args = cli.build_parser().parse_args(
        ["tune", "--config", str(V3_CONFIG), "--output-dir", str(output)]
    )

    assert args.func(args) == 2

    persisted = json.loads(
        (output / "tune_results.json").read_text(encoding="utf-8")
    )
    assert persisted["artifacts"] == {
        "tune_results": "tune_results.json",
        "best_config": "best_config.json",
        "best_fixed_mass_config": None,
        "best_constant_density_config": None,
    }
