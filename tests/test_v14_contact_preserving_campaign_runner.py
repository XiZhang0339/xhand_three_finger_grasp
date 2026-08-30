from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from xhand_grasp.artifacts import file_sha256, write_json
from xhand_grasp.cli import (
    _build_actual_contact_campaign_manifest,
    _load_actual_contact_tune_runner,
)
from xhand_grasp.config import load_config
from xhand_grasp.experiment import resolve_experiment
from xhand_grasp.tuning.contact_preserving_planned_lift_campaign import (
    AUXILIARY_SOURCE_CANDIDATE_ID,
    PRIMARY_SOURCE_CANDIDATE_ID,
    CampaignBackend,
    authenticate_v13_grasp_rescue_anchors,
    authenticate_v13_grasp_sources,
    build_grasp_rescue_jobs,
    build_contact_preserving_planned_lift_manifest,
    build_v14_source_pairs,
    run_contact_preserving_planned_lift_campaign,
)
from xhand_grasp.tuning.actual_contact_grasp_pose_robustness import (
    generate_v9_perturbation_configs,
)
from xhand_grasp.tuning.contact_constrained_planner import (
    v14_grasp_object_pair_id,
    v14_grasp_pose_id,
    v14_object_config_id,
)
import xhand_grasp.tuning.contact_preserving_planned_lift_campaign as v14_campaign


CONFIG = Path(
    "grasp_configs/left_opposed_face_palm_down_contact_preserving_planned_lift.json"
)


def _definition():
    return resolve_experiment(load_config(CONFIG))


def _empty_rescue_runner(
    template,
    definition,
    bundle,
    rescue_anchors,
    workspace,
    *,
    workers,
    resume,
):
    del template, workspace, workers, resume
    jobs = build_grasp_rescue_jobs(definition, bundle, rescue_anchors)
    groups = []
    grouped = {}
    for job in jobs:
        grouped.setdefault(job.group_id, []).append(job)
    for group_id, values in grouped.items():
        groups.append(
            {
                "complete": True,
                "group_id": group_id,
                "family": values[0].family,
                "edge_m": values[0].edge_m,
                "mapping_mode": values[0].mapping_mode,
                "declared_candidate_count": len(values),
                "candidate_count": len(values),
                "grasp_success_count": 0,
                "selected_candidate_ids": [],
                "failure_reason_counts": {"injected_no_physics": len(values)},
                "report_path": f"injected/{group_id}.json",
            }
        )
    return {
        "report": {
            "v14_grasp_rescue_report_schema_version": 1,
            "complete": True,
            "static_anchor_report_sha256": rescue_anchors.report_sha256,
            "declared_edge_mapping_group_count": len(definition.contact_preserving_planned_lift_campaign.edges_m) * 2,
            "executed_edge_mapping_group_count": len(definition.contact_preserving_planned_lift_campaign.edges_m) * 2,
            "priority_group_count": 2,
            "declared_candidate_count": len(jobs),
            "executed_candidate_count": len(jobs),
            "grasp_success_count": 0,
            "selected_count": 0,
            "groups": groups,
            "selected_records": [],
        },
        "artifacts": (),
    }


def test_v14_source_audit_and_pair_priority_are_exact() -> None:
    template = load_config(CONFIG)
    definition = resolve_experiment(template)
    bundle = authenticate_v13_grasp_sources(definition)
    assert bundle.catalog_sha256 == (
        definition.contact_preserving_planned_lift_campaign.source_grasp_catalog_sha256
    )
    assert len(bundle.published_candidate_ids) == 18
    assert len(bundle.sources) == 22
    assert [value["warm_start_alias"] for value in bundle.warm_start_records] == [
        "v13_79mm_local_08",
        "v13_79mm_local_09",
    ]
    assert all(value.config_path.is_file() for value in bundle.sources)
    assert all(value.trace_path.is_file() for value in bundle.sources)

    pairs = build_v14_source_pairs(template, definition, bundle)
    assert len(pairs) == 22
    assert pairs[0]["source_candidate_id"] == PRIMARY_SOURCE_CANDIDATE_ID
    assert pairs[1]["source_candidate_id"] == AUXILIARY_SOURCE_CANDIDATE_ID
    assert len(pairs[0]["warm_start_records"]) == 2
    assert pairs[1]["warm_start_records"] == []
    for pair in pairs:
        config = pair["config"]
        assert config["schema_version"] == 14
        assert "scaled_contact_mapping" not in config
        assert config["cube"]["mass_kg"] == 0.160
        assert config["cube"]["friction"] == 0.8
        assert config["object_config_id"] == pair["object_config_id"]
        assert config["grasp_pose_id"] == pair["grasp_pose_id"]
        assert config["grasp_object_pair_id"] == pair["grasp_object_pair_id"]


def test_v14_formal_79mm_warm_start_and_asymmetric_feedback_are_registered() -> None:
    template = load_config(CONFIG)
    definition = resolve_experiment(template)
    bundle = authenticate_v13_grasp_sources(definition)
    pair = build_v14_source_pairs(template, definition, bundle)[0]
    bounds = definition.search_bounds.manipulation_delta_rad
    assert bounds is not None
    plans = v14_campaign._warm_start_plan_records(pair, pair["config"], bounds)
    refined = next(
        value
        for value in plans
        if value["warm_start_alias"] == "v14_79mm_c3_contact_compensated_v1"
    )
    terminal = refined["config"]["control"]["manipulation_delta_rad"]
    assert terminal["left_hand_index_joint2_actuator"] == pytest.approx(0.535)
    assert terminal["left_hand_mid_joint1_actuator"] == pytest.approx(
        -0.09743618044100734
    )
    assert terminal["left_hand_mid_joint2_actuator"] == pytest.approx(
        0.27082973148714146
    )
    assert refined["terminal_path_rejected"] is False
    assert refined["config"]["candidate_metadata"]["formal_warm_start"][
        "requires_full_reset_rerun"
    ] is True

    feedback = v14_campaign._feedback_variants(pair["config"], count=64)
    assert len(feedback) == 64
    base = pair["config"]["contact_feedback"]
    assert feedback[0]["kp_rad_per_n"]["thumb"] == pytest.approx(
        base["kp_rad_per_n"]["thumb"] * 0.01
    )
    assert feedback[0]["kp_rad_per_n"]["index"] == pytest.approx(
        base["kp_rad_per_n"]["index"] * 0.01
    )
    assert feedback[0]["kp_rad_per_n"]["mid"] == pytest.approx(
        base["kp_rad_per_n"]["mid"] * 0.30
    )
    assert set(feedback[0]["ki_rad_per_n_s"].values()) == {0.0}


def test_v14_candidate_resume_binds_requested_id_and_config(tmp_path: Path) -> None:
    template = load_config(CONFIG)
    definition = resolve_experiment(template)
    bundle = authenticate_v13_grasp_sources(definition)
    config = build_v14_source_pairs(template, definition, bundle)[0]["config"]
    destination = tmp_path / "candidate"
    destination.mkdir()
    config_path = destination / "resolved_config.json"
    trace_path = destination / "trace.npz"
    write_json(config_path, config)
    np.savez_compressed(trace_path, time=np.asarray((0.0,)))
    payload = {
        "contact_preserving_candidate_schema_version": 1,
        "complete": True,
        "candidate_id": 14001,
        "experiment_id": config["experiment_id"],
        "classification": "near_miss",
        "full_success": False,
        "grasp_success": True,
        "summary": {"passed": False},
        "artifacts": {
            "resolved_config": "resolved_config.json",
            "trace": "trace.npz",
            "sha256": {
                "resolved_config": file_sha256(config_path),
                "trace": file_sha256(trace_path),
            },
        },
    }
    payload = v14_campaign.bind_candidate_result_semantic_sha256(payload)
    write_json(destination / "result.json", payload)
    assert v14_campaign._run_full_reset_candidate(
        config, destination, 14001
    )["candidate_id"] == 14001
    with pytest.raises(RuntimeError, match="wrong candidate ID"):
        v14_campaign._run_full_reset_candidate(config, destination, 14002)
    changed = copy.deepcopy(config)
    changed["candidate_metadata"]["resume_probe"] = True
    with pytest.raises(RuntimeError, match="requested config changed"):
        v14_campaign._run_full_reset_candidate(changed, destination, 14001)


def test_v14_robustness_perturbation_rebinds_object_pair_and_controller_ids() -> None:
    template = load_config(CONFIG)
    definition = resolve_experiment(template)
    bundle = authenticate_v13_grasp_sources(definition)
    config = build_v14_source_pairs(template, definition, bundle)[0]["config"]
    # A formal plan/controller ID is required before this config can be a
    # robustness source.
    bounds = definition.search_bounds.manipulation_delta_rad
    assert bounds is not None
    planned = v14_campaign._warm_start_plan_records(config | {
        "source_candidate_id": PRIMARY_SOURCE_CANDIDATE_ID,
        "grasp_object_pair_id": config["grasp_object_pair_id"],
        "warm_start_records": list(bundle.warm_start_records),
    }, config, bounds)[0]["config"]
    trial = generate_v9_perturbation_configs(
        planned,
        count=1,
        seed=20260821,
        source_candidate_id=PRIMARY_SOURCE_CANDIDATE_ID,
        family="per_full_success_local_16",
    )[0]
    assert trial["object_config_id"] == v14_object_config_id(trial)
    assert trial["grasp_pose_id"] == v14_grasp_pose_id(trial)
    assert trial["grasp_object_pair_id"] == v14_grasp_object_pair_id(trial)
    assert trial["object_config_id"] != planned["object_config_id"]
    assert trial["grasp_object_pair_id"] != planned["grasp_object_pair_id"]
    evidence = trial["candidate_metadata"]["robustness_trial"]
    assert evidence["source_object_config_id"] == planned["object_config_id"]
    assert evidence["source_grasp_object_pair_id"] == planned[
        "grasp_object_pair_id"
    ]


def test_v14_rescue_schedule_covers_every_edge_mapping_and_priority_budget() -> None:
    definition = _definition()
    bundle = authenticate_v13_grasp_sources(definition)
    anchors = authenticate_v13_grasp_rescue_anchors(definition, bundle)
    jobs = build_grasp_rescue_jobs(definition, bundle, anchors)
    assert len(anchors.anchors) == 29 * 2
    assert len(jobs) == 29 * 2 * 32 + 2 * 256 == 2368
    assert len({value.candidate_id for value in jobs}) == len(jobs)
    regular = [value for value in jobs if value.family == "edge_mapping_local"]
    assert {
        (round(value.edge_m, 6), value.mapping_mode) for value in regular
    } == {
        (edge / 1000.0, mode)
        for edge in range(60, 89)
        for mode in ("absolute_face_yz", "proportional_face_yz")
    }
    counts = {}
    for value in jobs:
        counts[value.group_id] = counts.get(value.group_id, 0) + 1
    assert set(counts.values()) == {32, 256}
    assert sum(value == 32 for value in counts.values()) == 58
    assert sum(value == 256 for value in counts.values()) == 2
    repeated = build_grasp_rescue_jobs(definition, bundle, anchors)
    assert [value.descriptor() for value in repeated] == [
        value.descriptor() for value in jobs
    ]


def test_v14_rescued_source_is_integrated_before_pair_ranking() -> None:
    template = load_config(CONFIG)
    definition = resolve_experiment(template)
    bundle = authenticate_v13_grasp_sources(definition)
    rescue = replace(
        bundle.sources[2],
        candidate_id=14_099_999_999_999_999,
        published_in_catalog=False,
    )
    pairs = build_v14_source_pairs(template, definition, bundle, (rescue,))
    assert len(pairs) == 23
    integrated = next(
        value for value in pairs if value["source_candidate_id"] == rescue.candidate_id
    )
    assert integrated["priority_role"] == "per_edge_mapping_grasp_rescue"


def test_v14_rescue_group_is_atomic_hash_bound_and_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    template = load_config(CONFIG)
    definition = resolve_experiment(template)
    bundle = authenticate_v13_grasp_sources(definition)
    anchors = authenticate_v13_grasp_rescue_anchors(definition, bundle)
    jobs = build_grasp_rescue_jobs(definition, bundle, anchors)[:2]
    base = v14_campaign.materialize_v14_static_rescue_config(
        template, definition, anchors.anchors[0]
    )
    calls = 0

    def fake_job(job):
        nonlocal calls
        calls += 1
        destination = Path(job["destination"])
        destination.mkdir(parents=True, exist_ok=True)
        config_path = destination / "resolved_config.json"
        trace_path = destination / "trace.npz"
        result_path = destination / "result.json"
        write_json(config_path, job["config"])
        np.savez_compressed(trace_path, time=np.asarray((0.001,)))
        payload = v14_campaign.bind_candidate_result_semantic_sha256(
            {
                "contact_preserving_candidate_schema_version": 1,
                "complete": True,
                "candidate_id": int(job["candidate_id"]),
                "experiment_id": job["config"]["experiment_id"],
                "classification": "near_miss",
                "full_success": False,
                "grasp_success": False,
                "summary": {
                    "stage_status": {
                        "grasp_success": False,
                        "full_success": False,
                        "failure_reason": "injected",
                    }
                },
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
        write_json(result_path, payload)
        return payload

    monkeypatch.setattr(v14_campaign, "_run_candidate_job", fake_job)
    workspace = tmp_path.resolve()
    first, artifacts = v14_campaign._load_or_run_rescue_group(
        template,
        definition,
        workspace,
        jobs,
        base,
        workers=1,
        seed=20260821,
    )
    assert first["candidate_count"] == 2
    assert calls == 2
    assert all(path.is_file() for path in artifacts)

    def must_not_run(_job):
        raise AssertionError("committed group should be loaded, not re-executed")

    monkeypatch.setattr(v14_campaign, "_run_candidate_job", must_not_run)
    second, _ = v14_campaign._load_or_run_rescue_group(
        template,
        definition,
        workspace,
        jobs,
        base,
        workers=4,
        seed=20260821,
    )
    assert second == first
    trace = next(workspace.glob("grasp_rescue/groups/*/candidates/*/trace.npz"))
    trace.write_bytes(trace.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        v14_campaign._load_or_run_rescue_group(
            template,
            definition,
            workspace,
            jobs,
            base,
            workers=1,
            seed=20260821,
        )


def test_v14_manifest_and_cli_dispatch_use_capability() -> None:
    definition = _definition()
    expected = build_contact_preserving_planned_lift_manifest(
        CONFIG, seed=20260821
    )
    routed = _build_actual_contact_campaign_manifest(
        definition, CONFIG, seed=20260821
    )
    assert routed == expected
    assert len(expected["v13_measured_source_records"]) == 22
    assert _load_actual_contact_tune_runner(definition).__name__ == (
        "run_contact_preserving_planned_lift_campaign"
    )


def test_v14_runner_commits_and_resumes_injected_physics(tmp_path: Path) -> None:
    calls = {"rescue": 0, "plan": 0, "candidate": 0}

    def rescue_runner(*args, **kwargs):
        calls["rescue"] += 1
        return _empty_rescue_runner(*args, **kwargs)

    def plan_runner(pair, source, output_dir):
        del source, output_dir
        calls["plan"] += 1
        return (
            {
                "candidate_id": 14_000_000_000_001 + calls["plan"],
                "source_candidate_id": pair["source_candidate_id"],
                "grasp_object_pair_id": pair["grasp_object_pair_id"],
                "contact_feasible": True,
                "path_rms_error": 0.0,
                "terminal_path_error": 0.0,
                "config": copy.deepcopy(pair["config"]),
            },
        )

    def candidate_runner(
        plan_records,
        workspace,
        *,
        target_success_count,
        workers,
        feedback_candidates_per_plan,
        maximum_candidate_count,
    ):
        del target_success_count, workers
        assert feedback_candidates_per_plan == 64
        assert maximum_candidate_count == 16 * 4 * 64
        calls["candidate"] += 1
        record = plan_records[0]
        candidate_id = 14_900_000_000_001
        root = workspace / "candidates" / f"candidate_{candidate_id}"
        root.mkdir(parents=True, exist_ok=True)
        config_path = root / "resolved_config.json"
        trace_path = root / "trace.npz"
        result_path = root / "result.json"
        write_json(config_path, record["config"])
        np.savez_compressed(trace_path, time=np.asarray((0.001,)))
        payload = {
            "contact_preserving_candidate_schema_version": 1,
            "complete": True,
            "candidate_id": candidate_id,
            "experiment_id": record["config"]["experiment_id"],
            "classification": "success",
            "full_success": True,
            "grasp_success": True,
            "summary": {
                "passed": True,
                "stage_status": {"grasp_success": True, "full_success": True},
            },
            "contact_maintenance": {
                "satisfied": True,
                "forbidden_contact": False,
                "active_nondistal_contact": False,
                "contact_loss_count": 0,
                "minimum_valid_duty": 1.0,
                "minimum_force_margin_n": 0.1,
                "maximum_tangent_slip_m": 0.0,
            },
            "path_tracking": {"rms_error": 0.0, "terminal_error": 0.0},
            "artifact_directory": f"candidates/candidate_{candidate_id}",
            "artifacts": {
                "resolved_config": "resolved_config.json",
                "trace": "trace.npz",
                "sha256": {
                    "resolved_config": file_sha256(config_path),
                    "trace": file_sha256(trace_path),
                },
            },
        }
        write_json(result_path, payload)
        return (payload,)

    backend = CampaignBackend(
        source_authenticator=authenticate_v13_grasp_sources,
        grasp_rescue_runner=rescue_runner,
        plan_runner=plan_runner,
        candidate_runner=candidate_runner,
    )
    output = tmp_path / "campaign"
    first = run_contact_preserving_planned_lift_campaign(
        CONFIG,
        output,
        resume=False,
        target_success_count=1,
        workers=1,
        backend=backend,
    )
    assert first["full_success_count"] == 1
    assert first["target_reached"] is True
    assert calls == {"rescue": 1, "plan": 16, "candidate": 1}
    catalog = output / first["catalogs"]["manipulation"]
    payload = json.loads(catalog.read_text(encoding="utf-8"))
    assert payload["aliases"]["best_first"] == payload["aliases"]["pair_rank_01"]

    second = run_contact_preserving_planned_lift_campaign(
        CONFIG,
        output,
        resume=True,
        target_success_count=1,
        workers=4,
        backend=backend,
    )
    assert second["full_success_count"] == 1
    assert calls == {"rescue": 1, "plan": 16, "candidate": 1}

    trace = output / "candidates/candidate_14900000000001/trace.npz"
    trace.write_bytes(trace.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        run_contact_preserving_planned_lift_campaign(
            CONFIG,
            output,
            resume=True,
            target_success_count=1,
            workers=1,
            backend=backend,
        )
