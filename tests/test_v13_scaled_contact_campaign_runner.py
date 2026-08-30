from __future__ import annotations

import json
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import xhand_grasp.cli as cli
from xhand_grasp.artifacts import file_sha256, write_json
from xhand_grasp.config import load_config
from xhand_grasp.experiment import resolve_experiment
from xhand_grasp.grasp_pose import canonical_sha256
from xhand_grasp.tuning import scaled_contact_downsize_campaign as campaign
from xhand_grasp.viewer import resolve_viewer_source


def _registered_campaign() -> SimpleNamespace:
    edges = tuple(value / 1000.0 for value in range(60, 89))
    aliases = ("best_nominal", "best_thumb_center", "best_pair_center")
    modes = ("proportional_face_yz", "absolute_face_yz")
    return SimpleNamespace(
        edges_m=edges,
        source_aliases=aliases,
        mapping_modes=modes,
        stratum_count=len(edges) * len(aliases) * len(modes),
    )


def test_v13_strata_run_full_descending_grid_without_early_stop():
    strata = campaign.campaign_strata(_registered_campaign())
    assert len(strata) == 174
    assert {value["edge_mm"] for value in strata} == set(range(60, 89))
    assert [value["stratum_index"] for value in strata] == list(range(174))
    assert strata[0] == {
        "stratum_index": 0,
        "edge_m": 0.088,
        "edge_mm": 88,
        "source_alias": "best_nominal",
        "mapping_mode": "proportional_face_yz",
    }
    assert strata[-1]["edge_mm"] == 60
    assert strata[-1]["source_alias"] == "best_pair_center"
    assert strata[-1]["mapping_mode"] == "absolute_face_yz"


def _source_catalog(tmp_path: Path) -> Path:
    trajectories = []
    aliases = {}
    for index, alias in enumerate(
        ("best_nominal", "best_thumb_center", "best_pair_center")
    ):
        member = tmp_path / f"source_{index}"
        member.mkdir()
        write_json(member / "resolved_config.json", {"source": alias})
        write_json(member / "result.json", {"hard_pass": True})
        np.savez_compressed(member / "trace.npz", time=np.asarray([0.0]))
        trajectory = f"trajectory_{index}"
        aliases[alias] = trajectory
        trajectories.append(
            {
                "trajectory_id": trajectory,
                "artifacts": {
                    "resolved_config": f"source_{index}/resolved_config.json",
                    "result": f"source_{index}/result.json",
                    "trace": f"source_{index}/trace.npz",
                    "sha256": {
                        "resolved_config": file_sha256(
                            member / "resolved_config.json"
                        ),
                        "result": file_sha256(member / "result.json"),
                        "trace": file_sha256(member / "trace.npz"),
                    },
                },
            }
        )
    catalog = tmp_path / "catalog.json"
    write_json(catalog, {"aliases": aliases, "trajectories": trajectories})
    return catalog


def test_source_catalog_audit_binds_every_alias_and_rejects_corruption(tmp_path):
    catalog = _source_catalog(tmp_path)
    aliases = _registered_campaign().source_aliases
    records = campaign._audit_source_catalog_files(catalog, aliases)
    assert [value["alias"] for value in records] == list(aliases)
    assert all(len(value["trace_sha256"]) == 64 for value in records)

    (tmp_path / "source_1" / "trace.npz").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        campaign._audit_source_catalog_files(catalog, aliases)


def _viewer_catalog(tmp_path: Path) -> tuple[Path, list[dict]]:
    trajectories = []
    records = []
    for identifier, edge_mm, mode, source in (
        (11, 60, "absolute_face_yz", "best_nominal"),
        (12, 60, "proportional_face_yz", "best_pair_center"),
        (13, 61, "absolute_face_yz", "best_thumb_center"),
    ):
        member = tmp_path / f"candidate_{identifier}"
        member.mkdir()
        config = member / "resolved_config.json"
        trace = member / "trace.npz"
        write_json(config, {"candidate": identifier})
        np.savez_compressed(trace, time=np.asarray([0.0]))
        trajectories.append(
            {
                "trajectory_id": f"candidate_{identifier}",
                "label": f"candidate_{identifier}",
                "candidate_id": str(identifier),
                "classification": "success",
                "aliases": [],
                "artifacts": {
                    "resolved_config": f"candidate_{identifier}/resolved_config.json",
                    "trace": f"candidate_{identifier}/trace.npz",
                    "sha256": {
                        "resolved_config": file_sha256(config),
                        "trace": file_sha256(trace),
                    },
                },
            }
        )
        records.append(
            {
                "candidate_id": identifier,
                "downsize_metadata": {
                    "edge_m": edge_mm / 1000.0,
                    "mapping_mode": mode,
                    "source_alias": source,
                },
            }
        )
    catalog = tmp_path / "catalog.json"
    write_json(catalog, {"aliases": {}, "trajectories": trajectories})
    return catalog, records


def test_catalog_metadata_aliases_and_viewer_selection(tmp_path):
    catalog, records = _viewer_catalog(tmp_path)
    payload = campaign._annotate_catalog(catalog, records, kind="grasp_pose")
    assert payload["aliases"]["smallest_grasp_pass"] == "candidate_11"
    assert payload["aliases"]["edge_60_best"] == "candidate_11"
    selected = resolve_viewer_source(
        catalog_path=catalog,
        catalog_edge_mm=60,
        catalog_mapping_mode="proportional_face_yz",
        catalog_source_alias="best_pair_center",
    )
    assert selected.config_path.parent.name == "candidate_12"
    with pytest.raises(ValueError, match="exceeds 2 metadata matches"):
        resolve_viewer_source(
            catalog_path=catalog,
            catalog_edge_mm=60,
            catalog_rank=3,
        )


def test_catalog_publication_keeps_all_per_edge_passes_beyond_cli_target(
    tmp_path,
):
    artifact_root = tmp_path / "artifacts"
    records = []
    for edge_index, edge_mm in enumerate((60, 61)):
        # Put a near miss first to prove it cannot consume the three-pass cap.
        for local_index in range(4):
            identifier = edge_mm * 100 + local_index
            member = artifact_root / f"candidate_{identifier}"
            member.mkdir(parents=True)
            write_json(member / "resolved_config.json", {"candidate": identifier})
            write_json(member / "result.json", {"candidate": identifier})
            np.savez_compressed(member / "trace.npz", time=np.asarray([0.0]))
            passed = local_index > 0
            records.append(
                {
                    "candidate_id": identifier,
                    "artifact_directory": member.name,
                    "summary": {
                        "passed": passed,
                        "stage_status": {
                            "grasp_success": passed,
                            "manipulation_success": passed,
                            "full_success": passed,
                        },
                    },
                    "downsize_metadata": {
                        "edge_m": edge_mm / 1000.0,
                        "mapping_mode": (
                            "absolute_face_yz"
                            if local_index % 2 == 0
                            else "proportional_face_yz"
                        ),
                        "source_alias": f"source_{local_index}",
                    },
                }
            )

    observed = {}

    def exporter(candidates, output, *, selected_count):
        materialized = list(candidates)
        observed["ids"] = [value["candidate_id"] for value in materialized]
        observed["selected_count"] = selected_count
        output.mkdir(parents=True)
        trajectories = [
            {
                "trajectory_id": f"candidate_{value['candidate_id']}",
                "candidate_id": str(value["candidate_id"]),
                "classification": "success",
                "aliases": [],
                "artifacts": {},
            }
            for value in materialized
        ]
        payload = {"aliases": {}, "trajectories": trajectories}
        write_json(output / "catalog.json", payload)
        return payload

    catalog = campaign._publish_catalog(
        records,
        artifact_root,
        tmp_path / "catalog",
        target_success_count=1,
        kind="grasp_pose",
        exporter=exporter,
    )
    assert catalog is not None
    assert observed["selected_count"] == 6
    assert len(observed["ids"]) == 6
    assert all(identifier % 100 != 0 for identifier in observed["ids"])
    payload = json.loads(catalog.read_text(encoding="utf-8"))
    assert {"edge_60_best", "edge_61_best", "smallest_grasp_pass"} <= set(
        payload["aliases"]
    )


def test_default_robustness_runners_bind_published_sources_and_scoped_labels(
    tmp_path, monkeypatch
):
    import xhand_grasp.tuning.actual_contact_grasp_pose_robustness as lift_module
    import xhand_grasp.tuning.contact_point_grasp_robustness as grasp_module
    from xhand_grasp.tuning.actual_contact_grasp_pose_robustness import (
        V9RobustnessSource,
    )

    template = load_config(
        "grasp_configs/"
        "left_opposed_face_palm_down_scaled_centered_spread_actual_grasp_then_lift.json"
    )
    registered = resolve_experiment(template).scaled_contact_downsize_campaign
    catalog_path = tmp_path / "catalog.json"
    write_json(
        catalog_path,
        {
            "aliases": {"best_nominal": "trajectory_2", "best_first": "trajectory_1"},
            "trajectories": [
                {"trajectory_id": "trajectory_1", "candidate_id": "1"},
                {"trajectory_id": "trajectory_2", "candidate_id": "2"},
            ],
        },
    )
    hard_summary = {
        "passed": True,
        "failed_checks": [],
        "stage_status": {
            "grasp_success": True,
            "manipulation_success": True,
            "full_success": True,
        },
        "metrics": {},
    }
    lift_sources = tuple(
        V9RobustnessSource(
            candidate_id=str(identifier),
            config=template,
            summary=hard_summary,
            discovery_index=identifier - 1,
            best_first=identifier == 1,
        )
        for identifier in (1, 2)
    )
    observed = {}
    monkeypatch.setattr(
        lift_module,
        "discover_v9_robustness_sources",
        lambda _roots: lift_sources,
    )

    def fake_lift(_roots, _output, **kwargs):
        rebound = kwargs["candidate_discoverer"](_roots)
        observed["lift_best"] = [
            value.candidate_id for value in rebound if value.best_first
        ]
        observed["lift_limit"] = kwargs["max_nominal_trajectories"]
        return {"complete": True, "robust_passed": True}

    monkeypatch.setattr(lift_module, "run_v9_robustness_campaign", fake_lift)
    lift = campaign._default_robustness_runner(
        catalog_path,
        tmp_path / "lift.json",
        workers=2,
        seed=20260821,
        campaign=registered,
    )
    assert observed == {"lift_best": ["2"], "lift_limit": 87}
    assert lift["best_50_selection_alias"] == "best_nominal"

    published_sources = [SimpleNamespace(config=template) for _ in range(6)]
    monkeypatch.setattr(
        grasp_module,
        "discover_v12_grasp_robustness_sources",
        lambda _roots, **kwargs: (
            observed.__setitem__("grasp_discovery_limit", kwargs["maximum_nominal_grasps"])
            or tuple(published_sources)
        ),
    )

    def fake_grasp(sources, _output, **kwargs):
        observed["grasp_count"] = len(sources)
        observed["grasp_limit"] = kwargs["maximum_nominal_grasps"]
        return {"complete": True, "robust_passed": True}

    monkeypatch.setattr(grasp_module, "run_v12_grasp_perturbation_audit", fake_grasp)
    grasp = campaign._default_grasp_robustness_runner(
        (),
        tmp_path / "grasp.json",
        workers=2,
        seed=20260821,
        campaign=registered,
        catalog_path=catalog_path,
    )
    assert observed["grasp_discovery_limit"] == 87
    assert observed["grasp_count"] == 6
    assert observed["grasp_limit"] == 87
    assert grasp["validation_label"] == (
        "validated_fixed_160g_scaled_grasp_ablation"
    )
    assert grasp["full_lift_robustness_claimed"] is False

def test_cli_dispatches_scaled_campaign_and_exposes_metadata_selectors():
    definition = SimpleNamespace(
        scaled_contact_downsize_campaign=object(), contact_point_search=None
    )
    runner = cli._load_actual_contact_tune_runner(definition)
    assert runner is campaign.run_scaled_contact_downsize_campaign

    parsed = cli.build_parser().parse_args(
        [
            "view",
            "--catalog",
            "catalog.json",
            "--catalog-edge-mm",
            "67",
            "--catalog-mapping-mode",
            "absolute_face_yz",
            "--catalog-source-alias",
            "best_nominal",
            "--catalog-rank",
            "2",
        ]
    )
    assert parsed.catalog_edge_mm == 67.0
    assert parsed.catalog_mapping_mode == "absolute_face_yz"
    assert parsed.catalog_source_alias == "best_nominal"
    assert parsed.catalog_rank == 2


def test_production_static_one_and_spawn_workers_are_deterministic(tmp_path):
    template = load_config(
        "grasp_configs/"
        "left_opposed_face_palm_down_scaled_centered_spread_actual_grasp_then_lift.json"
    )
    registered = resolve_experiment(template).scaled_contact_downsize_campaign
    source = campaign._default_source_loader(registered)["best_nominal"]
    mini = SimpleNamespace(
        edges_m=(0.088,),
        source_aliases=("best_nominal",),
        mapping_modes=("proportional_face_yz",),
        stratum_count=1,
        dls_starts_per_stratum=2,
        max_dls_iterations=1,
        static_retain_per_stratum=2,
        contact_target_radius_m=registered.contact_target_radius_m,
        minimum_edge_guard_m=registered.minimum_edge_guard_m,
    )
    serial = campaign._default_static_runner(
        template,
        mini,
        {"best_nominal": source},
        workspace=tmp_path / "serial",
        resume=False,
        seed=20260821,
        workers=1,
    )
    spawned = campaign._default_static_runner(
        template,
        mini,
        {"best_nominal": source},
        workspace=tmp_path / "spawned",
        resume=False,
        seed=20260821,
        workers=2,
    )
    assert [value["candidate_id"] for value in serial] == [
        value["candidate_id"] for value in spawned
    ]
    assert canonical_sha256(serial) == canonical_sha256(spawned)
    assert all(value["config"]["schema_version"] == 13 for value in serial)
    assert all("dls_job_error" not in value["static_metrics"] for value in serial)
    assert all(len(value["config"]["scaled_contact_mapping"]) == 14 for value in serial)
    resumed = campaign._default_static_runner(
        template,
        mini,
        {"best_nominal": source},
        workspace=tmp_path / "serial",
        resume=True,
        seed=20260821,
        workers=2,
    )
    assert canonical_sha256(resumed) == canonical_sha256(serial)


def test_injected_runner_commits_every_stage_and_resume_reuses_it(
    tmp_path, monkeypatch
):
    config_path = Path(
        "grasp_configs/"
        "left_opposed_face_palm_down_scaled_centered_spread_actual_grasp_then_lift.json"
    ).resolve()
    template = load_config(config_path)
    calls = {name: 0 for name in ("static", "dynamic", "local", "measured", "manip")}
    manipulation_targets = []

    def static_runner(_template, _campaign, _sources, **_kwargs):
        calls["static"] += 1
        return (
            {
                "candidate_id": 13001,
                "candidate_sha256": canonical_sha256(template),
                "config": copy.deepcopy(template),
                "static_pass": True,
                "static_rank": [False, 13001],
                "downsize_metadata": {
                    "edge_m": 0.088,
                    "mapping_mode": "proportional_face_yz",
                    "source_alias": "best_nominal",
                },
            },
        )

    def artifact_record(root: Path, identifier: int, *, measured=False, full=False):
        member = root / ("measured" if measured else "candidates") / f"candidate_{identifier}"
        member.mkdir(parents=True, exist_ok=True)
        write_json(member / "resolved_config.json", template)
        np.savez_compressed(member / "trace.npz", time=np.asarray([0.0]))
        summary = {
            "passed": full,
            "failed_checks": [] if full else ["operation_median_lift_reached"],
            "stage_status": {
                "grasp_success": True,
                "manipulation_success": full,
                "full_success": full,
            },
        }
        write_json(member / "result.json", {"summary": summary})
        return {
            "candidate_id": identifier,
            "candidate_sha256": canonical_sha256(template),
            "config": copy.deepcopy(template),
            "summary": summary,
            "grasp_success": True,
            "measured_grasp_pose_success": measured,
            "artifact_directory": str(member.relative_to(root)),
            "downsize_metadata": {
                "edge_m": 0.088,
                "mapping_mode": "proportional_face_yz",
                "source_alias": "best_nominal",
            },
        }

    def dynamic_runner(_records, root, **_kwargs):
        calls["dynamic"] += 1
        return (artifact_record(Path(root), 13002),)

    def local_runner(_records, workspace, **_kwargs):
        calls["local"] += 1
        return (artifact_record(Path(workspace) / "dynamic", 13003),)

    def measured_runner(_records, root, **_kwargs):
        calls["measured"] += 1
        return (artifact_record(Path(root), 13004, measured=True),)

    def manipulation_runner(_records, workspace, **_kwargs):
        calls["manip"] += 1
        manipulation_targets.append(int(_kwargs["target_success_count"]))
        member = Path(workspace) / "manipulation" / "candidate_13005"
        member.mkdir(parents=True, exist_ok=True)
        write_json(member / "resolved_config.json", template)
        np.savez_compressed(member / "trace.npz", time=np.asarray([0.0]))
        summary = {
            "passed": True,
            "failed_checks": [],
            "stage_status": {
                "grasp_success": True,
                "manipulation_success": True,
                "full_success": True,
            },
        }
        write_json(member / "result.json", {"summary": summary})
        return (
            {
                "candidate_id": 13005,
                "candidate_sha256": canonical_sha256(template),
                "config": copy.deepcopy(template),
                "summary": summary,
                "config_path": member / "resolved_config.json",
                "result_path": member / "result.json",
                "trace_path": member / "trace.npz",
                "downsize_metadata": {
                    "edge_m": 0.088,
                    "mapping_mode": "proportional_face_yz",
                    "source_alias": "best_nominal",
                },
            },
        )

    def catalog_exporter(candidates, output, **_kwargs):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        trajectories = []
        for item in candidates:
            trajectory_id = f"candidate_{item['candidate_id']}"
            trajectories.append(
                {
                    "trajectory_id": trajectory_id,
                    "label": trajectory_id,
                    "candidate_id": str(item["candidate_id"]),
                    "classification": "success",
                    "aliases": ["best_first"],
                    "artifacts": {},
                }
            )
        payload = {
            "experiment_id": template["experiment_id"],
            "aliases": {"best_first": trajectories[0]["trajectory_id"]},
            "trajectories": trajectories,
        }
        write_json(output / "catalog.json", payload)
        return payload

    def robustness_runner(*args, **_kwargs):
        output = Path(args[1])
        payload = {"complete": True, "best_pass_count": 50}
        write_json(output, payload)
        return payload

    monkeypatch.setattr(
        campaign, "authenticated_catalog_artifact_paths", lambda path: (Path(path),)
    )
    backend = campaign.CampaignBackend(
        source_loader=campaign._default_source_loader,
        static_runner=static_runner,
        dynamic_runner=dynamic_runner,
        local_grasp_runner=local_runner,
        measured_runner=measured_runner,
        manipulation_runner=manipulation_runner,
        grasp_catalog_exporter=catalog_exporter,
        manipulation_catalog_exporter=catalog_exporter,
        grasp_robustness_runner=robustness_runner,
        robustness_runner=robustness_runner,
    )
    output = tmp_path / "campaign"
    result = campaign.run_scaled_contact_downsize_campaign(
        config_path,
        output,
        resume=False,
        target_success_count=1,
        workers=1,
        backend=backend,
    )
    assert result["target_reached"] is True
    assert result["grasp_success_count"] == 1
    assert result["full_success_count"] == 1
    assert calls == {"static": 1, "dynamic": 1, "local": 1, "measured": 1, "manip": 1}
    assert manipulation_targets == [87]
    ledger = campaign.validate_stage_ledger(output)
    assert {
        "source_audit",
        "static_downsize_scan",
        "dynamic_grasp",
        "local_grasp_refinement",
        "measured_grasp_finalization",
        "manipulation_source",
        "manipulation_target_1",
        "catalog_target_1",
        "grasp_robustness_target_1",
        "robustness_target_1",
    } <= set(ledger["stages"])

    repeated = campaign.run_scaled_contact_downsize_campaign(
        config_path,
        output,
        resume=True,
        target_success_count=1,
        workers=2,
        backend=backend,
    )
    assert repeated["target_reached"] is True
    assert calls == {"static": 1, "dynamic": 1, "local": 1, "measured": 1, "manip": 1}
    assert manipulation_targets == [87]

    expanded = campaign.run_scaled_contact_downsize_campaign(
        config_path,
        output,
        resume=True,
        target_success_count=5,
        workers=2,
        backend=backend,
    )
    assert expanded["target_reached"] is False
    assert calls == {"static": 1, "dynamic": 1, "local": 1, "measured": 1, "manip": 1}
    assert manipulation_targets == [87]
    expanded_ledger = campaign.validate_stage_ledger(output)
    assert {
        "manipulation_source",
        "manipulation_target_1",
        "manipulation_target_5",
        "catalog_target_1",
        "catalog_target_5",
    } <= set(expanded_ledger["stages"])
    assert (
        output / "manipulation" / "target_1_report.json"
    ).is_file()
    assert (
        output / "manipulation" / "target_5_report.json"
    ).is_file()


def test_near_miss_still_seeds_the_next_descending_edge(tmp_path, monkeypatch):
    template = load_config(
        "grasp_configs/"
        "left_opposed_face_palm_down_scaled_centered_spread_actual_grasp_then_lift.json"
    )
    registered = resolve_experiment(template).scaled_contact_downsize_campaign
    source = campaign._default_source_loader(registered)["best_nominal"]
    mini = SimpleNamespace(
        edges_m=(0.087, 0.088),
        source_aliases=("best_nominal",),
        mapping_modes=("proportional_face_yz",),
        stratum_count=2,
        dls_starts_per_stratum=1,
        max_dls_iterations=1,
        static_retain_per_stratum=1,
        contact_target_radius_m=registered.contact_target_radius_m,
        minimum_edge_guard_m=registered.minimum_edge_guard_m,
    )
    observed_previous = []

    def fake_job(job):
        observed_previous.append(copy.deepcopy(job.get("previous_edge_config")))
        config = copy.deepcopy(job["template"])
        config["cube"]["edge_m"] = job["stratum"]["edge_m"]
        return {
            "candidate_id": job["candidate_id"],
            "candidate_sha256": canonical_sha256(config),
            "config": config,
            "static_pass": False,
            "static_rank": [True, job["candidate_id"]],
            "static_metrics": {"real_witness_near_miss": True},
            "downsize_metadata": {
                **job["stratum"],
                "dls_start_index": job["start_index"],
            },
        }

    monkeypatch.setattr(campaign, "_solve_static_job", fake_job)
    campaign._default_static_runner(
        template,
        mini,
        {"best_nominal": source},
        workspace=tmp_path,
        resume=False,
        seed=20260821,
        workers=1,
    )
    assert observed_previous[0] is None
    assert observed_previous[1]["cube"]["edge_m"] == pytest.approx(0.088)

    progress = tmp_path / "static_progress" / "stratum_000.json"
    payload = json.loads(progress.read_text(encoding="utf-8"))
    payload["records"][0]["static_pass"] = True
    write_json(progress, payload)
    with pytest.raises(RuntimeError, match="static progress changed"):
        campaign._default_static_runner(
            template,
            mini,
            {"best_nominal": source},
            workspace=tmp_path,
            resume=True,
            seed=20260821,
            workers=1,
        )


@pytest.mark.parametrize(
    ("start_index", "has_previous", "expected"),
    (
        (0, True, ("previous_edge_continuation", 0, True)),
        (
            0,
            False,
            ("reference_edge_direct_alternate_damping", 0, False),
        ),
        (1, True, ("reference_edge_direct", 0, False)),
        (2, True, ("reference_edge_local_branch_1", 1, False)),
        (3, True, ("reference_edge_local_branch_2", 2, False)),
    ),
)
def test_static_four_start_initialization_schedule(
    start_index, has_previous, expected
):
    assert campaign._static_initialization_branch(
        start_index, has_previous_edge=has_previous
    ) == expected


def test_static_worker_passes_declared_initialization_to_solver_and_metadata(
    monkeypatch,
):
    from xhand_grasp.tuning import contact_point_downsize as downsize

    template = load_config(
        "grasp_configs/"
        "left_opposed_face_palm_down_scaled_centered_spread_actual_grasp_then_lift.json"
    )
    registered = resolve_experiment(template).scaled_contact_downsize_campaign
    evidence = campaign._default_source_loader(registered)["best_nominal"]
    calls = []

    class Solved:
        def as_dynamic_record(self, identifier, *, source_id):
            config = copy.deepcopy(template)
            config.setdefault("candidate_metadata", {})[
                "contact_point_downsize"
            ] = {}
            return {
                "candidate_id": identifier,
                "source_id": source_id,
                "config": config,
                "candidate_sha256": canonical_sha256(config),
                "static_pass": True,
                "static_rank": [False, identifier],
                "static_metrics": {},
            }

    def fake_solve(*args, **kwargs):
        calls.append(
            (
                kwargs["start_index"],
                kwargs["previous_edge_config"] is not None,
            )
        )
        return Solved()

    monkeypatch.setattr(downsize, "solve_downsize_static_candidate", fake_solve)
    stratum = campaign.campaign_strata(registered)[0]
    previous = copy.deepcopy(template)
    expected = (
        (0, True, "previous_edge_continuation"),
        (0, False, "reference_edge_direct"),
        (1, False, "reference_edge_local_branch_1"),
        (2, False, "reference_edge_local_branch_2"),
    )
    for start_index, (solver_start, uses_previous, branch) in enumerate(expected):
        record = campaign._solve_static_job(
            {
                "start_index": start_index,
                "candidate_id": start_index + 1,
                "stratum": stratum,
                "seed": registered.seed,
                "evidence": evidence,
                "template": template,
                "target_radius_m": registered.contact_target_radius_m,
                "edge_guard_m": registered.minimum_edge_guard_m,
                "max_iterations": 1,
                "previous_edge_config": previous,
            }
        )
        assert calls[-1] == (solver_start, uses_previous)
        assert record["downsize_metadata"]["initialization_branch"] == branch
        persisted = record["config"]["candidate_metadata"][
            "contact_point_downsize"
        ]
        assert persisted["initialization_branch"] == branch
        assert persisted["campaign_start_index"] == start_index
        assert persisted["solver_start_index"] == solver_start
        assert record["candidate_sha256"] == canonical_sha256(record["config"])


def test_all_static_failures_skip_dynamic_without_validating_fallback(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        campaign,
        "run_actual_contact_dynamic_grasp_candidates",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    records = (
        {
            "candidate_id": index,
            "static_pass": False,
            "config": {"intentionally": "not a runnable fallback"},
            "downsize_metadata": {
                "edge_m": 0.088,
                "mapping_mode": "absolute_face_yz",
            },
        }
        for index in range(4)
    )
    assert campaign._default_batched_dynamic_runner(
        records,
        tmp_path,
        workers=1,
        resume=False,
        seed=20260821,
        controller_seed_count=6,
    ) == ()
    assert calls == []


def test_dynamic_preflight_rejects_every_invalid_static_pass_before_writes(
    tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        campaign,
        "run_actual_contact_dynamic_grasp_candidates",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    config = load_config(
        "grasp_configs/"
        "left_opposed_face_palm_down_scaled_centered_spread_actual_grasp_then_lift.json"
    )
    config["hand_pose"]["rpy_deg"][2] = 15.0170086662682
    record = {
        "candidate_id": 13041287864844940,
        "static_pass": True,
        "config": config,
        "downsize_metadata": {
            "stratum_index": 7,
            "edge_m": 0.087,
            "edge_mm": 87,
            "mapping_mode": "absolute_face_yz",
            "source_alias": "best_nominal",
        },
    }
    with pytest.raises(
        RuntimeError,
        match=r"candidate 13041287864844940 .*stratum=7.*hand yaw",
    ):
        campaign._default_batched_dynamic_runner(
            (record,),
            tmp_path / "dynamic",
            workers=1,
            resume=False,
            seed=20260821,
            controller_seed_count=6,
        )
    assert calls == []
    assert not (tmp_path / "dynamic").exists()
