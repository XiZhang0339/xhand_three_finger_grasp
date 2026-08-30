from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import xhand_grasp.cli as cli
import xhand_grasp.tuning.contact_point_grasp_robustness as robustness_module
from xhand_grasp.actual_contact_grasp_pose_catalog import (
    bind_candidate_result_semantic_sha256,
)
from xhand_grasp.artifacts import file_sha256, write_json
from xhand_grasp.config import load_config, validate_config
from xhand_grasp.grasp_pose import canonical_sha256, controller_id, grasp_pose_id
from xhand_grasp.tuning.contact_point_grasp_robustness import (
    BEST_FAMILY,
    LOCAL_FAMILY,
    V12_CONTACT_POINT_HARD_CHECKS,
    V12GraspRobustnessSource,
    discover_v12_grasp_robustness_sources,
    generate_v12_grasp_perturbation_configs,
    run_v12_grasp_perturbation_audit,
    v12_grasp_failure_reasons,
    v12_grasp_hard_success,
)


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_90mm_contact_point_targeted_actual_grasp_pose.json"
)


def _summary(
    *,
    grasp: bool,
    contact_points: bool | None = None,
    manipulation: bool = False,
    full: bool = False,
) -> dict:
    point_passed = grasp if contact_points is None else contact_points
    checks = {name: point_passed for name in V12_CONTACT_POINT_HARD_CHECKS}
    checks["synthetic_manipulation_check"] = manipulation
    return {
        "passed": full,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "checks": checks,
        "stage_status": {
            "grasp_success": grasp,
            "manipulation_success": manipulation,
            "full_success": full,
        },
        "metrics": {"synthetic": True},
    }


def _source(index: int, *, best: bool = False) -> V12GraspRobustnessSource:
    config = load_config(TEMPLATE)
    config.setdefault("candidate_metadata", {})["source_index"] = index
    validate_config(config)
    return V12GraspRobustnessSource(
        candidate_id=str(12_000 + index),
        config=config,
        summary=_summary(grasp=True),
        discovery_index=index,
        best=best,
    )


def _write_discovery_catalog(
    root: Path, *, classification: str, measured: bool
) -> Path:
    config = load_config(TEMPLATE)
    member = root / "trajectory_1"
    member.mkdir(parents=True)
    config_path = member / "resolved_config.json"
    trace_path = member / "trace.npz"
    result_path = member / "result.json"
    write_json(config_path, config)
    trace_path.write_bytes(b"authenticated-test-trace")
    summary = _summary(grasp=True)
    result = bind_candidate_result_semantic_sha256(
        {
            "candidate_result_schema_version": 1,
            "complete": True,
            "stage": (
                "measured_grasp_pose_finalization"
                if measured
                else "dynamic_grasp_acquisition"
            ),
            "candidate_id": 12_345,
            "candidate_sha256": canonical_sha256(config),
            "grasp_pose_id": grasp_pose_id(config),
            "controller_id": controller_id(config),
            "measured_grasp_pose_success": measured,
            "grasp_success": True,
            "summary": summary,
            "artifacts": {
                "resolved_config": "resolved_config.json",
                "trace": "trace.npz",
                "sha256": {
                    "resolved_config": file_sha256(config_path),
                    "trace": file_sha256(trace_path),
                },
            },
        }
    )
    write_json(result_path, result)
    catalog = {
        "experiment_id": config["experiment_id"],
        "aliases": (
            {"best_first": "trajectory_1"}
            if classification == "success"
            else {"best_attempt": "trajectory_1"}
        ),
        "trajectories": [
            {
                "trajectory_id": "trajectory_1",
                "classification": classification,
                "aliases": (
                    ["best_first"]
                    if classification == "success"
                    else ["best_attempt"]
                ),
                "artifacts": {
                    "resolved_config": "trajectory_1/resolved_config.json",
                    "result": "trajectory_1/result.json",
                    "trace": "trajectory_1/trace.npz",
                    "sha256": {
                        "resolved_config": file_sha256(config_path),
                        "result": file_sha256(result_path),
                        "trace": file_sha256(trace_path),
                    },
                },
            }
        ],
    }
    catalog_path = root / "catalog.json"
    write_json(catalog_path, catalog)
    return catalog_path


def test_v12_discovery_accepts_only_success_catalog_with_measured_evidence(
    tmp_path: Path,
) -> None:
    success = _write_discovery_catalog(
        tmp_path / "success", classification="success", measured=True
    )
    sources = discover_v12_grasp_robustness_sources((success,))
    assert len(sources) == 1
    assert sources[0].candidate_id == "12345"

    diagnostic = _write_discovery_catalog(
        tmp_path / "diagnostic", classification="diagnostic", measured=False
    )
    with pytest.raises(ValueError, match="no authenticated schema-v12"):
        discover_v12_grasp_robustness_sources((diagnostic,))

    mislabeled = _write_discovery_catalog(
        tmp_path / "mislabeled", classification="success", measured=False
    )
    with pytest.raises(ValueError, match="not a successful measured"):
        discover_v12_grasp_robustness_sources((mislabeled,))


def test_v12_grasp_verdict_ignores_manipulation_and_global_passed() -> None:
    summary = _summary(grasp=True, manipulation=False, full=False)
    assert summary["passed"] is False
    assert summary["stage_status"]["full_success"] is False
    assert v12_grasp_hard_success(summary) is True
    assert v12_grasp_failure_reasons(summary) == []

    failed = _summary(grasp=True, contact_points=False)
    assert v12_grasp_hard_success(failed) is False
    assert v12_grasp_failure_reasons(failed) == list(
        V12_CONTACT_POINT_HARD_CHECKS
    )


def test_v12_source_requires_nominal_grasp_and_all_point_checks() -> None:
    config = load_config(TEMPLATE)
    with pytest.raises(ValueError, match="verified schema-v12 grasp"):
        V12GraspRobustnessSource(
            candidate_id="bad-stage",
            config=config,
            summary=_summary(grasp=False),
        )
    with pytest.raises(ValueError, match="verified schema-v12 grasp"):
        V12GraspRobustnessSource(
            candidate_id="bad-point",
            config=config,
            summary=_summary(grasp=True, contact_points=False),
        )


def test_v12_perturbations_are_deterministic_valid_and_grasp_scoped() -> None:
    config = load_config(TEMPLATE)
    before = copy.deepcopy(config)
    first = generate_v12_grasp_perturbation_configs(
        config,
        count=16,
        seed=20260821,
        source_candidate_id="source",
        family=LOCAL_FAMILY,
    )
    second = generate_v12_grasp_perturbation_configs(
        config,
        count=16,
        seed=20260821,
        source_candidate_id="source",
        family=LOCAL_FAMILY,
    )
    assert first == second
    assert config == before
    assert len({canonical_sha256(case) for case in first}) == 16
    for trial, case in enumerate(first):
        validate_config(case)
        assert case["run_context"] == {"kind": "robustness_trial"}
        metadata = case["candidate_metadata"]["v12_grasp_robustness_trial"]
        assert metadata["trial"] == trial
        assert metadata["family"] == LOCAL_FAMILY
        assert metadata["success_scope"] == "grasp_only_contact_point_hard_checks"
        assert metadata["manipulation_success_required"] is False
        assert metadata["full_success_required"] is False
        assert metadata["full_reset_rerun"] is True
        assert metadata["checkpoint_used"] is False
        perturbation = metadata["resolved_perturbations"]
        assert all(
            abs(value) <= 0.0015
            for value in perturbation["cube_center_xy_delta_m"]
        )
        assert 0.0 <= perturbation["cube_gap_delta_m"] <= 0.0005
        assert all(
            abs(value) <= 3.0 for value in perturbation["cube_rpy_delta_deg"]
        )
        assert perturbation["mass_scale"] == 1.0
        assert -0.1 <= perturbation["friction_delta"] <= 0.1


def test_v12_audit_runs_each_16_plus_best_50_and_accepts_45(
    tmp_path: Path,
) -> None:
    sources = [_source(0), _source(1, best=True), _source(2)]
    observed: list[tuple[int, dict]] = []

    def runner(jobs, workers):
        assert workers == 3
        observed.extend(jobs)
        results = []
        for candidate_id, config in jobs:
            metadata = config["candidate_metadata"]["v12_grasp_robustness_trial"]
            passed = bool(
                metadata["family"] == LOCAL_FAMILY
                or metadata["trial"] < 45
            )
            results.append(
                {
                    "candidate_id": candidate_id,
                    "config": copy.deepcopy(config),
                    "summary": _summary(grasp=passed),
                }
            )
        return list(reversed(results))

    output = tmp_path / "audit.json"
    report = run_v12_grasp_perturbation_audit(
        sources,
        output,
        workers=3,
        runner=runner,
    )
    assert len(observed) == 3 * 16 + 50
    assert report["complete"] is True
    assert report["selected_grasp_count"] == 3
    assert report["total_perturbation_count"] == 98
    assert [entry["perturbation_count"] for entry in report["per_grasp"]] == [
        16,
        16,
        16,
    ]
    assert report["best_robustness"]["candidate_id"] == sources[1].candidate_id
    assert report["best_robustness"]["grasp_passes"] == 45
    assert report["best_robustness"]["robust_passed"] is True
    assert report["robust_passed"] is True
    assert report["validation_label"] == (
        "validated_fixed_160g_robust_grasp_ablation"
    )
    assert json.loads(output.read_text(encoding="utf-8")) == report
    for entry in report["per_grasp"]:
        assert [trial["trial"] for trial in entry["trials"]] == list(range(16))
    assert [trial["trial"] for trial in report["best_robustness"]["trials"]] == list(
        range(50)
    )


def test_v12_audit_worker_result_order_does_not_change_evidence(
    tmp_path: Path,
) -> None:
    sources = [_source(0, best=True), _source(1)]

    def execute(jobs, workers):
        results = [
            {
                "candidate_id": candidate_id,
                "config": copy.deepcopy(config),
                "summary": _summary(grasp=True),
            }
            for candidate_id, config in jobs
        ]
        return results if workers == 1 else list(reversed(results))

    first = run_v12_grasp_perturbation_audit(
        sources,
        tmp_path / "one.json",
        workers=1,
        runner=execute,
    )
    second = run_v12_grasp_perturbation_audit(
        sources,
        tmp_path / "four.json",
        workers=4,
        runner=execute,
    )
    for key in (
        "selected_candidate_ids",
        "total_perturbation_count",
        "total_grasp_passes",
        "per_grasp",
        "best_robustness",
        "robust_passed",
        "stop_reason",
    ):
        assert first[key] == second[key]


def test_v12_best_50_follows_highest_local_pass_count_not_initial_best(
    tmp_path: Path,
) -> None:
    initial_best = _source(0, best=True)
    stronger = _source(1, best=False)
    calls: list[list[tuple[int, dict]]] = []

    def runner(jobs, _workers):
        calls.append(copy.deepcopy(jobs))
        results = []
        for candidate_id, config in jobs:
            metadata = config["candidate_metadata"]["v12_grasp_robustness_trial"]
            if metadata["family"] == LOCAL_FAMILY:
                local_limit = (
                    8
                    if metadata["source_candidate_id"] == initial_best.candidate_id
                    else 12
                )
                passed = metadata["trial"] < local_limit
            else:
                passed = True
            results.append(
                {
                    "candidate_id": candidate_id,
                    "config": copy.deepcopy(config),
                    "summary": _summary(grasp=passed),
                }
            )
        return list(reversed(results))

    report = run_v12_grasp_perturbation_audit(
        [initial_best, stronger],
        tmp_path / "stronger.json",
        workers=2,
        runner=runner,
    )

    assert [len(jobs) for jobs in calls] == [2 * 16, 50]
    assert {
        config["candidate_metadata"]["v12_grasp_robustness_trial"][
            "source_candidate_id"
        ]
        for _, config in calls[1]
    } == {stronger.candidate_id}
    assert report["best_selection"]["candidate_id"] == stronger.candidate_id
    assert report["best_selection"]["local_grasp_passes"] == 12
    assert report["best_selection"]["selection_completed_before_best_50"] is True
    assert report["best_robustness"]["candidate_id"] == stronger.candidate_id
    assert report["total_perturbation_count"] == 2 * 16 + 50


def test_v12_audit_44_of_50_fails_and_keeps_all_failure_reasons(
    tmp_path: Path,
) -> None:
    source = _source(0, best=True)

    def runner(jobs, _workers):
        results = []
        for candidate_id, config in jobs:
            metadata = config["candidate_metadata"]["v12_grasp_robustness_trial"]
            passed = metadata["family"] == LOCAL_FAMILY or metadata["trial"] < 44
            summary = _summary(grasp=passed)
            if not passed:
                summary["failed_checks"].append("specific_base_grasp_failure")
            results.append(
                {
                    "candidate_id": candidate_id,
                    "config": copy.deepcopy(config),
                    "summary": summary,
                }
            )
        return results

    report = run_v12_grasp_perturbation_audit(
        [source],
        tmp_path / "fail.json",
        workers=1,
        runner=runner,
    )
    assert report["best_robustness"]["grasp_passes"] == 44
    assert report["robust_passed"] is False
    assert report["validation_label"] is None
    assert report["stop_reason"] == "best_grasp_pass_count_below_45_of_50"
    failed = [
        trial
        for trial in report["best_robustness"]["trials"]
        if not trial["grasp_passed"]
    ]
    assert len(failed) == 6
    assert all("specific_base_grasp_failure" in trial["failed_checks"] for trial in failed)
    assert all(
        "stage_status.grasp_success" in trial["grasp_failure_reasons"]
        for trial in failed
    )


def test_v12_audit_rejects_too_many_sources_and_incomplete_runner(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="at most five"):
        run_v12_grasp_perturbation_audit(
            [_source(index) for index in range(6)],
            tmp_path / "too_many.json",
            workers=1,
            runner=lambda jobs, workers: [],
        )
    with pytest.raises(RuntimeError, match="complete job set"):
        run_v12_grasp_perturbation_audit(
            [_source(0)],
            tmp_path / "missing.json",
            workers=1,
            runner=lambda jobs, workers: [],
        )


def test_v12_audit_accepts_campaign_record_mapping(tmp_path: Path) -> None:
    source = _source(0)

    def runner(jobs, workers):
        return [
            {
                "candidate_id": candidate_id,
                "config": copy.deepcopy(config),
                "summary": _summary(grasp=True),
            }
            for candidate_id, config in jobs
        ]

    report = run_v12_grasp_perturbation_audit(
        [
            {
                "candidate_id": source.candidate_id,
                "config": source.config,
                "summary": source.summary,
                "aliases": ["best_nominal"],
            }
        ],
        tmp_path / "mapping.json",
        workers=1,
        runner=runner,
    )
    assert report["best_robustness"]["candidate_id"] == source.candidate_id
    assert report["robust_passed"] is True


def test_v12_generator_rejects_v9_family_name() -> None:
    with pytest.raises(ValueError, match="unknown schema-v12"):
        generate_v12_grasp_perturbation_configs(
            load_config(TEMPLATE),
            count=1,
            seed=1,
            source_candidate_id="source",
            family=BEST_FAMILY.replace("v12_grasp", "best_first"),
        )


def test_cli_dispatches_v12_to_grasp_only_robustness(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    observed = {}

    def fake_discover(roots):
        observed["roots"] = tuple(roots)
        return ("authenticated-grasp",)

    def fake_audit(sources, output, *, workers, seed):
        observed.update(
            sources=tuple(sources), output=Path(output), workers=workers, seed=seed
        )
        return {
            "selected_grasp_count": 1,
            "total_perturbation_count": 66,
            "best_robustness": {
                "grasp_passes": 45,
                "required_grasp_passes": 45,
            },
            "robust_passed": True,
        }

    monkeypatch.setattr(
        robustness_module, "discover_v12_grasp_robustness_sources", fake_discover
    )
    monkeypatch.setattr(
        robustness_module, "run_v12_grasp_perturbation_audit", fake_audit
    )
    source = tmp_path / "grasp_catalog.json"
    output = tmp_path / "report.json"
    arguments = cli.build_parser().parse_args(
        [
            "robustness",
            "--config",
            str(TEMPLATE),
            "--search-root",
            str(source),
            "--output",
            str(output),
            "--workers",
            "3",
        ]
    )

    assert cli.command_robustness(arguments) == 0
    assert observed == {
        "roots": (source.resolve(),),
        "sources": ("authenticated-grasp",),
        "output": output.resolve(),
        "workers": 3,
        "seed": 20260821,
    }
    payload = json.loads(capsys.readouterr().out)
    assert payload["success_scope"] == "grasp_only_contact_point_hard_checks"
    assert payload["robust_passed"] is True


def test_cli_v12_default_robustness_discovers_both_tune_layouts(
    tmp_path: Path, monkeypatch
) -> None:
    observed = {}

    def fake_discover(roots):
        observed["roots"] = tuple(roots)
        return ("authenticated-grasp",)

    def fake_audit(_sources, _output, *, workers, seed):
        return {
            "selected_grasp_count": 1,
            "total_perturbation_count": 66,
            "best_robustness": {
                "grasp_passes": 45,
                "required_grasp_passes": 45,
            },
            "robust_passed": True,
        }

    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        robustness_module, "discover_v12_grasp_robustness_sources", fake_discover
    )
    monkeypatch.setattr(
        robustness_module, "run_v12_grasp_perturbation_audit", fake_audit
    )
    definition = cli.resolve_experiment(load_config(TEMPLATE))
    tune_root = tmp_path / definition.artifact_root / "tune"
    expected = tuple(
        root / "catalogs" / f"target_{target}" / "grasp_pose" / "catalog.json"
        for root in (tune_root / "campaign", tune_root)
        for target in (5, 1)
    )
    for path in expected:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")

    arguments = cli.build_parser().parse_args(
        [
            "robustness",
            "--config",
            str(TEMPLATE),
            "--output",
            str(tmp_path / "report.json"),
        ]
    )

    assert cli.command_robustness(arguments) == 0
    assert observed["roots"] == tuple(path.resolve() for path in expected)
