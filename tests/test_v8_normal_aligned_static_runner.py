from __future__ import annotations

import json
from pathlib import Path

import pytest

from xhand_grasp.artifacts import file_sha256
from xhand_grasp.tuning.normal_aligned_smooth_lift import (
    DEFAULT_V7_CAMPAIGN_RESULTS,
    DEFAULT_V8_TEMPLATE,
)
from xhand_grasp.tuning.normal_aligned_static_runner import (
    StaticSearchRunnerBudget,
    run_normal_aligned_static_search,
)


ROOT = Path(__file__).resolve().parents[1]


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def test_registered_runner_defaults_declare_55_by_10000_budget():
    budget = StaticSearchRunnerBudget()
    assert budget.alpha_count == 33
    assert budget.retain_per_cell == 4
    assert budget.report(registered_cell_count=55, selected_cell_count=55) == {
        "samples_per_cell": 10_000,
        "retain_per_cell": 4,
        "alpha_count": 33,
        "coarse_alpha_count": 7,
        "promotion_per_chunk": 16,
        "coarse_near_margin_m": 0.008,
        "chunk_size": 256,
        "registered_cell_count": 55,
        "selected_cell_count": 55,
        "registered_declared_sample_count": 550_000,
        "selected_declared_sample_count": 550_000,
        "selected_maximum_retained_count": 220,
    }
    with pytest.raises(ValueError, match="alpha_count"):
        StaticSearchRunnerBudget(alpha_count=2)


def test_small_spawn_campaign_commits_cells_report_configs_and_resumes(tmp_path):
    output = tmp_path / "static"
    budget = StaticSearchRunnerBudget(
        samples_per_cell=1,
        retain_per_cell=1,
        alpha_count=3,
        chunk_size=1,
    )
    report = run_normal_aligned_static_search(
        _repo_path(DEFAULT_V7_CAMPAIGN_RESULTS),
        _repo_path(DEFAULT_V8_TEMPLATE),
        output,
        workers=2,
        budget=budget,
        cell_indices=(0, 1),
    )

    assert report["complete"] is True
    assert report["evaluated_count"] == 2
    assert report["coarse_scan_count"] == 2
    assert report["full_scan_count"] == 2
    assert report["not_promoted_count"] == 0
    assert report["completed_cell_count"] == 2
    assert report["retained_count"] == 2
    assert [value["cell_index"] for value in report["cell_results"]] == [0, 1]
    assert report["dynamics_scheduled"] is False
    assert len(report["retained_configs"]) == 2

    for binding in report["cell_results"]:
        result_path = output / binding["result_path"]
        assert result_path.name == "result.json"
        assert file_sha256(result_path) == binding["result_sha256"]
        cell = json.loads(result_path.read_text(encoding="utf-8"))
        assert cell["complete"] is True
        assert cell["evaluated_count"] == 1
        assert cell["coarse_scan_count"] == 1
        assert cell["full_scan_count"] == 1
        assert cell["retained_count"] == 1
        assert len(cell["retained"]) == 1
        assert isinstance(cell["retained"][0]["config"], dict)
        assert cell["retained"][0]["static_evidence"] == {
            "scan_kind": "full_static_evidence",
            "alpha_sample_count": 3,
            "coarse_scan_used_for_pass": False,
        }

    for binding in report["retained_configs"]:
        config_path = output / binding["config_path"]
        assert file_sha256(config_path) == binding["config_file_sha256"]

    manifest = json.loads(
        (output / "campaign_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["complete"] is True
    assert manifest["campaign_input_sha256"] == report["campaign_input_sha256"]
    assert file_sha256(output / "search_report.json") == manifest[
        "search_report_sha256"
    ]

    cell_mtimes = {
        path: path.stat().st_mtime_ns for path in output.glob("cells/*/result.json")
    }
    resumed = run_normal_aligned_static_search(
        _repo_path(DEFAULT_V7_CAMPAIGN_RESULTS),
        _repo_path(DEFAULT_V8_TEMPLATE),
        output,
        workers=2,
        resume=True,
        budget=budget,
        cell_indices=(1, 0),
    )
    assert resumed == report
    assert {
        path: path.stat().st_mtime_ns for path in output.glob("cells/*/result.json")
    } == cell_mtimes


def test_resume_rejects_changed_exact_inputs(tmp_path):
    output = tmp_path / "resume"
    budget = StaticSearchRunnerBudget(
        samples_per_cell=1,
        retain_per_cell=1,
        alpha_count=3,
        chunk_size=1,
    )
    run_normal_aligned_static_search(
        _repo_path(DEFAULT_V7_CAMPAIGN_RESULTS),
        _repo_path(DEFAULT_V8_TEMPLATE),
        output,
        workers=1,
        budget=budget,
        cell_indices=(0,),
    )
    with pytest.raises(RuntimeError, match="inputs do not match"):
        run_normal_aligned_static_search(
            _repo_path(DEFAULT_V7_CAMPAIGN_RESULTS),
            _repo_path(DEFAULT_V8_TEMPLATE),
            output,
            workers=1,
            resume=True,
            budget=StaticSearchRunnerBudget(
                samples_per_cell=2,
                retain_per_cell=1,
                alpha_count=3,
                chunk_size=1,
            ),
            cell_indices=(0,),
        )
