from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import xhand_grasp.cli as cli
import xhand_grasp.tuning.actual_contact_grasp_pose_robustness as robustness_module
from xhand_grasp.actual_contact_grasp_pose_catalog import (
    bind_candidate_result_semantic_sha256,
)
from xhand_grasp.artifacts import file_sha256, write_json
from xhand_grasp.config import load_config, validate_config
from xhand_grasp.grasp_pose import canonical_sha256
from xhand_grasp.tuning.actual_contact_grasp_pose_robustness import (
    V9RobustnessSource,
    discover_v9_robustness_sources,
    generate_v9_perturbation_configs,
    run_v9_robustness_campaign,
)


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_actual_contact_grasp_pose_smooth_vertical_lift.json"
)


def _summary(*, passed: bool, thumb_rad: float = 1.50) -> dict:
    return {
        "passed": passed,
        "failed_checks": [] if passed else ["synthetic_failure"],
        "checks": {"synthetic_failure": passed},
        "stage_status": {
            "grasp_success": passed,
            "manipulation_success": passed,
            "full_success": passed,
        },
        "metrics": {
            "actual_grasp_pose": {
                "metrics": {"thumb_actual_median_rad": thumb_rad}
            }
        },
    }


def _source(
    index: int,
    *,
    passed: bool = True,
    best_first: bool = False,
) -> V9RobustnessSource:
    config = load_config(TEMPLATE)
    config.setdefault("candidate_metadata", {})["synthetic_source_index"] = index
    validate_config(config)
    return V9RobustnessSource(
        candidate_id=str(9_100_000 + index),
        config=config,
        summary=_summary(passed=passed, thumb_rad=1.50 + index * 0.001),
        discovery_index=index,
        best_first=best_first,
    )


def _write_production_catalog(root: Path) -> Path:
    member = root / "trajectory"
    member.mkdir(parents=True)
    config = load_config(TEMPLATE)
    config_path = member / "resolved_config.json"
    trace_path = member / "trace.npz"
    result_path = member / "result.json"
    write_json(config_path, config)
    trace_path.write_bytes(b"authenticated production trace")
    result = bind_candidate_result_semantic_sha256(
        {
            "actual_contact_manipulation_candidate_schema_version": 1,
            "candidate_id": 42,
            "candidate_sha256": canonical_sha256(config),
            "summary": _summary(passed=True),
            "artifacts": {
                "sha256": {
                    "resolved_config": file_sha256(config_path),
                    "trace": file_sha256(trace_path),
                }
            },
        }
    )
    write_json(result_path, result)
    catalog_path = root / "catalog.json"
    write_json(
        catalog_path,
        {
            "experiment_id": config["experiment_id"],
            "aliases": {"best_first": "trajectory"},
            "trajectories": [
                {
                    "trajectory_id": "trajectory",
                    "aliases": ["best_first"],
                    "artifacts": {
                        "resolved_config": "trajectory/resolved_config.json",
                        "result": "trajectory/result.json",
                        "trace": "trajectory/trace.npz",
                        "sha256": {
                            "resolved_config": file_sha256(config_path),
                            "result": file_sha256(result_path),
                            "trace": file_sha256(trace_path),
                        },
                    },
                }
            ],
        },
    )
    return catalog_path


def test_v9_perturbations_are_deterministic_valid_and_full_reset():
    config = load_config(TEMPLATE)
    before = copy.deepcopy(config)
    first = generate_v9_perturbation_configs(
        config,
        count=16,
        seed=20260821,
        source_candidate_id="source",
        family="per_full_success_local_16",
    )
    second = generate_v9_perturbation_configs(
        config,
        count=16,
        seed=20260821,
        source_candidate_id="source",
        family="per_full_success_local_16",
    )

    assert first == second
    assert config == before
    assert len({canonical_sha256(case) for case in first}) == 16
    for trial, case in enumerate(first):
        validate_config(case)
        assert case["run_context"] == {"kind": "robustness_trial"}
        metadata = case["candidate_metadata"]["robustness_trial"]
        assert metadata["trial"] == trial
        assert metadata["full_reset_rerun"] is True
        assert metadata["initial_state_source"] == "configured_no_contact_reset"
        assert metadata["checkpoint_used"] is False
        resolved = metadata["resolved_perturbations"]
        assert all(abs(value) <= 0.0015 for value in resolved["cube_center_xy_delta_m"])
        assert 0.0 <= resolved["cube_gap_delta_m"] <= 0.0005
        assert all(abs(value) <= 3.0 for value in resolved["cube_rpy_delta_deg"])
        assert 0.9 <= resolved["mass_scale"] <= 1.1
        assert -0.1 <= resolved["friction_delta"] <= 0.1


def test_v9_campaign_runs_five_times_16_plus_best_50_and_is_order_stable(
    tmp_path,
):
    sources = tuple(
        _source(index, best_first=index == 2)
        for index in range(6)
    )
    observed_jobs: list[tuple[int, dict]] = []

    def runner(jobs, workers):
        assert workers == 3
        observed_jobs.extend(jobs)
        return [
            {
                "candidate_id": candidate_id,
                "config": copy.deepcopy(config),
                "summary": _summary(passed=True),
            }
            for candidate_id, config in reversed(jobs)
        ]

    report = run_v9_robustness_campaign(
        [tmp_path],
        tmp_path / "report.json",
        workers=3,
        candidate_discoverer=lambda _roots: sources,
        runner=runner,
    )

    assert len(observed_jobs) == 5 * 16 + 50
    assert report["complete"] is True
    assert report["selected_nominal_count"] == 5
    assert report["total_perturbation_count"] == 130
    assert report["total_perturbation_passes"] == 130
    assert report["registered_budget_complete"] is True
    assert [entry["perturbation_count"] for entry in report["per_nominal"]] == [16] * 5
    assert report["best_robustness"]["candidate_id"] == sources[2].candidate_id
    assert report["best_robustness"]["perturbation_count"] == 50
    assert report["best_robustness"]["perturbation_passes"] == 50
    assert report["best_robustness"]["diagnostic_only_due_to_nominal_failure"] is False
    assert report["robust_passed"] is True
    assert report["stop_reason"] == "robust_passed"
    assert all(
        trial["full_reset_rerun"]
        and trial["initial_state_source"] == "configured_no_contact_reset"
        and not trial["checkpoint_used"]
        for entry in report["per_nominal"]
        for trial in entry["trials"]
    )


def test_v9_optional_local_pass_selection_chooses_independent_robust_leader(
    tmp_path,
):
    sources = tuple(_source(index, best_first=index == 0) for index in range(5))
    calls = []
    desired_passes = {
        sources[0].candidate_id: 8,
        sources[1].candidate_id: 10,
        sources[2].candidate_id: 12,
        sources[3].candidate_id: 16,
        sources[4].candidate_id: 14,
    }

    def runner(jobs, _workers):
        calls.append(len(jobs))
        results = []
        for candidate_id, config in jobs:
            evidence = config["candidate_metadata"]["robustness_trial"]
            if evidence["family"] == "per_full_success_local_16":
                passed = evidence["trial"] < desired_passes[
                    evidence["source_candidate_id"]
                ]
            else:
                passed = True
            results.append(
                {
                    "candidate_id": candidate_id,
                    "config": copy.deepcopy(config),
                    "summary": _summary(passed=passed),
                }
            )
        return list(reversed(results))

    report = run_v9_robustness_campaign(
        [tmp_path],
        tmp_path / "locally_selected_report.json",
        workers=2,
        candidate_discoverer=lambda _roots: sources,
        runner=runner,
        best_selection="local_perturbation_passes",
    )
    assert calls == [5 * 16, 50]
    assert report["best_selection_policy"] == "local_perturbation_passes"
    assert report["best_robustness"]["candidate_id"] == sources[3].candidate_id
    assert report["best_robustness"]["perturbation_passes"] == 50


def test_v9_nominal_failure_cannot_be_promoted_by_perturbations(tmp_path):
    source = _source(0, passed=False, best_first=True)

    def runner(jobs, _workers):
        return [
            {
                "candidate_id": candidate_id,
                "config": copy.deepcopy(config),
                "summary": _summary(passed=True),
            }
            for candidate_id, config in jobs
        ]

    report = run_v9_robustness_campaign(
        [tmp_path],
        tmp_path / "failure_report.json",
        workers=1,
        candidate_discoverer=lambda _roots: (source,),
        runner=runner,
    )

    assert report["selected_nominal_count"] == 0
    assert report["total_perturbation_count"] == 50
    assert report["best_robustness"]["perturbation_passes"] == 50
    assert report["best_robustness"]["diagnostic_only_due_to_nominal_failure"] is True
    assert report["registered_budget_complete"] is False
    assert report["robust_passed"] is False
    assert report["stop_reason"] == "best_first_nominal_failed_hard_constraints"


def test_v9_reduced_budget_is_explicitly_diagnostic(tmp_path):
    source = _source(0, best_first=True)

    def runner(jobs, _workers):
        return [
            {
                "candidate_id": candidate_id,
                "config": copy.deepcopy(config),
                "summary": _summary(passed=True),
            }
            for candidate_id, config in jobs
        ]

    report = run_v9_robustness_campaign(
        [tmp_path],
        tmp_path / "small_report.json",
        workers=1,
        local_perturbations=2,
        best_perturbations=3,
        max_nominal_trajectories=1,
        candidate_discoverer=lambda _roots: (source,),
        runner=runner,
    )

    assert report["total_perturbation_count"] == 5
    assert report["registered_budget_complete"] is False
    assert report["robust_passed"] is False
    assert report["stop_reason"] == "registered_perturbation_budget_incomplete"


def test_v9_discovery_never_recursively_uses_prior_robustness_trials(tmp_path):
    nominal = load_config(TEMPLATE)
    trial = generate_v9_perturbation_configs(
        nominal,
        count=1,
        seed=20260821,
        source_candidate_id="nominal",
        family="best_first_pose_material_50",
    )[0]
    for name, config in (("nominal", nominal), ("trial", trial)):
        directory = tmp_path / name
        directory.mkdir()
        write_json(directory / "resolved_config.json", config)
        write_json(
            directory / "result.json",
            {"candidate_id": name, "summary": _summary(passed=True)},
        )
        (directory / "trace.npz").write_bytes(b"synthetic")

    discovered = discover_v9_robustness_sources((tmp_path,))
    assert [source.candidate_id for source in discovered] == ["nominal"]


def test_v9_production_discovery_authenticates_result_semantics(tmp_path):
    _write_production_catalog(tmp_path)
    result_path = tmp_path / "trajectory" / "result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["summary"]["passed"] = False
    write_json(result_path, payload)

    with pytest.raises(RuntimeError, match="semantic SHA-256 mismatch"):
        discover_v9_robustness_sources((result_path,))


def test_v9_catalog_discovery_authenticates_member_hashes(tmp_path):
    catalog = _write_production_catalog(tmp_path)
    (tmp_path / "trajectory" / "trace.npz").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="catalog trace SHA-256 mismatch"):
        discover_v9_robustness_sources((catalog,))


def test_main_robustness_cli_dispatches_schema_v9_catalog(
    tmp_path, monkeypatch, capsys
):
    catalog = tmp_path / "catalog.json"
    write_json(catalog, {"synthetic": True})
    output = tmp_path / "report.json"
    observed = {}

    def fake_campaign(search_roots, output_path, *, workers, seed):
        observed.update(
            {
                "search_roots": tuple(Path(value) for value in search_roots),
                "output": Path(output_path),
                "workers": workers,
                "seed": seed,
            }
        )
        return {
            "selected_nominal_count": 2,
            "total_perturbation_count": 82,
            "best_robustness": {
                "perturbation_passes": 46,
                "required_perturbation_passes": 45,
            },
            "robust_passed": True,
        }

    monkeypatch.setattr(
        robustness_module, "run_v9_robustness_campaign", fake_campaign
    )
    arguments = cli.build_parser().parse_args(
        [
            "robustness",
            "--config",
            str(TEMPLATE),
            "--search-root",
            str(catalog),
            "--output",
            str(output),
            "--workers",
            "3",
            "--seed",
            "17",
        ]
    )

    assert cli.command_robustness(arguments) == 0
    assert observed == {
        "search_roots": (catalog.resolve(),),
        "output": output.resolve(),
        "workers": 3,
        "seed": 17,
    }
    assert '"robust_passed": true' in capsys.readouterr().out


def test_old_schema_rejects_v9_robustness_search_root(tmp_path):
    root = tmp_path / "catalog.json"
    root.touch()
    arguments = cli.build_parser().parse_args(
        [
            "robustness",
            "--config",
            "grasp_configs/left_three_finger_cube.json",
            "--search-root",
            str(root),
        ]
    )
    with pytest.raises(ValueError, match="schema-v9"):
        cli.command_robustness(arguments)
