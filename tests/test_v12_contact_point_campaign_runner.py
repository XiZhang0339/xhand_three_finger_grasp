from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from xhand_grasp.artifacts import write_json
from xhand_grasp.config import ACTIVE_ACTUATORS, load_config
from xhand_grasp.grasp_pose import canonical_sha256, controller_id, grasp_pose_id
from xhand_grasp.tuning.actual_qpos_sources import load_actual_qpos_sources
from xhand_grasp.tuning.contact_point_targeted_search import (
    ContactPointGenerationResult,
    ContactPointPlan,
    GeneratedContactPointPlan,
    evaluate_contact_point_plan_geometry,
    ContactPointSearchPolicy,
)
from xhand_grasp.tuning import contact_point_targeted_campaign as campaign


TEMPLATE = Path(
    "grasp_configs/"
    "left_opposed_face_palm_down_90mm_contact_point_targeted_actual_grasp_pose.json"
)


def _static_record(config: dict, candidate_id: int) -> dict:
    plan = ContactPointPlan.from_config(config["contact_point_plan"])
    return {
        "candidate_id": candidate_id,
        "source_id": "test-source",
        "config": copy.deepcopy(config),
        "candidate_sha256": canonical_sha256(config),
        "grasp_pose_id": grasp_pose_id(config),
        "controller_id": controller_id(config),
        "point_plan_id": plan.point_plan_id,
        "signed_orbit_deg": 0.0,
        "clockwise_orbit_deg": 0.0,
        "static_pass": True,
        "static_metrics": {
            "point_target": {
                "point_plan_id": plan.point_plan_id,
                "point_distance_m": [0.001, 0.001, 0.001],
                "static_acceptance": {"passed": True, "reasons": []},
            },
            "minimum_active_nondistal_gap_m": 0.003,
            "nominal_minimum_forbidden_hand_gap_m": 0.0025,
            "nominal_maximum_all_distal_penetration_m": 0.0002,
            "precontact_minimum_hand_gap_m": 0.002,
        },
        "static_rank": (False, 0, 0, 0.001, 0.003, 0.0, 0.0, 0.0, 10.0, candidate_id),
    }


def _write_candidate(root: Path, config: dict, candidate_id: int, *, measured: bool) -> dict:
    relative = Path("measured" if measured else "candidates") / f"candidate_{candidate_id}"
    directory = root / relative
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "resolved_config.json", config)
    np.savez_compressed(directory / "trace.npz", time=np.asarray([0.0, 0.001]))
    summary = {
        "passed": False,
        "failed_checks": ["manipulation_not_part_of_v12_grasp_campaign"],
        "stage_status": {
            "grasp_success": True,
            "manipulation_success": False,
            "full_success": False,
        },
        "metrics": {
            "contact_point_targeting": {
                "acquisition_window": {
                    "per_finger": {
                        finger: {"tangent_error_max_m": 0.001}
                        for finger in ("thumb", "index", "mid")
                    }
                }
            }
        },
    }
    result = {
        "candidate_result_schema_version": 1,
        "complete": True,
        "stage": (
            "measured_grasp_pose_finalization"
            if measured
            else "actual_contact_dynamic_grasp"
        ),
        "candidate_id": candidate_id,
        "candidate_sha256": canonical_sha256(config),
        "grasp_pose_id": grasp_pose_id(config),
        "controller_id": controller_id(config),
        "measured_grasp_pose_success": measured,
        "grasp_success": True,
        "summary": summary,
        "artifacts": {"resolved_config": "resolved_config.json", "trace": "trace.npz"},
    }
    result = campaign.bind_candidate_result_semantic_sha256(result)
    write_json(directory / "result.json", result)
    return {
        "candidate_id": candidate_id,
        "candidate_sha256": canonical_sha256(config),
        "grasp_pose_id": grasp_pose_id(config),
        "controller_id": controller_id(config),
        "config": copy.deepcopy(config),
        "summary": summary,
        "grasp_success": True,
        "measured_grasp_pose_success": measured,
        "result_semantic_sha256": result["result_semantic_sha256"],
        "artifact_directory": str(relative),
    }


def test_source_transplant_preserves_v12_cube_and_relative_height_bridge():
    template = load_config(TEMPLATE)
    source = load_actual_qpos_sources(template)[0]
    plan = ContactPointPlan.from_config(template["contact_point_plan"])
    materialized = campaign.materialize_v12_source_seed(template, source, plan)

    assert materialized["experiment_id"] == template["experiment_id"]
    assert materialized["cube"] == template["cube"]
    assert materialized["cube"]["edge_m"] == 0.090
    assert np.isclose(
        materialized["hand_pose"]["translation_m"][2]
        - source.config["hand_pose"]["translation_m"][2],
        0.0005,
    )
    assert materialized["contact_point_plan"]["point_plan_id"] == plan.point_plan_id
    assert set(materialized["control"]["manipulation_delta_rad"].values()) == {0.0}
    assert materialized["candidate_metadata"]["contact_point_target_search"][
        "cube_pose_sampled"
    ] is False


def test_static_safety_margin_uses_all_real_v12_evidence_fields():
    template = load_config(TEMPLATE)
    policy = ContactPointSearchPolicy.from_config(template)
    plan = ContactPointPlan.from_config(template["contact_point_plan"])
    source = load_actual_qpos_sources(template)[0]
    record = _static_record(
        campaign.materialize_v12_source_seed(template, source, plan), 17
    )
    assert campaign._static_safety_margin(record, policy) == 0.0018
    del record["static_metrics"]["nominal_minimum_forbidden_hand_gap_m"]
    assert campaign._static_safety_margin(record, policy) == -np.inf


def test_v12_rank_uses_local_perturbations_before_real_acquisition_error():
    def record(identifier: int, passes: int, error: float) -> dict:
        return {
            "candidate_id": identifier,
            "measured_grasp_pose_success": True,
            "local_perturbation_pass_count": passes,
            "manipulability_prescreen": {"manipulability_score": 0.5},
            "summary": {
                "stage_status": {"grasp_success": True},
                "metrics": {
                    "contact_point_targeting": {
                        "acquisition_window": {
                            "per_finger": {
                                finger: {"tangent_error_max_m": error}
                                for finger in ("thumb", "index", "mid")
                            }
                        }
                    }
                },
            },
            "config": load_config(TEMPLATE),
        }

    fewer_passes_better_points = record(1, 14, 0.0001)
    more_passes_worse_points = record(2, 15, 0.0019)
    assert campaign._grasp_rank(more_passes_worse_points) < campaign._grasp_rank(
        fewer_passes_better_points
    )


def _complete_v12_rank_record(
    identifier: int,
    *,
    thumb_actual: float = 1.50,
    point_error: float = 0.001,
    residual: float = 0.2,
    forces: tuple[float, float, float] = (2.0, 1.0, 1.0),
    peak_total_force: float = 4.0,
) -> dict:
    config = load_config(TEMPLATE)
    return {
        "candidate_id": identifier,
        "measured_grasp_pose_success": True,
        "local_perturbation_pass_count": 16,
        "manipulability_prescreen": {"weighted_residual_norm": residual},
        "config": config,
        "summary": {
            "stage_status": {"grasp_success": True},
            "metrics": {
                "actual_grasp_pose": {
                    "metrics": {
                        "thumb_actual_median_rad": thumb_actual,
                        "maximum_nominal_joint_error_rad": 0.01,
                        "maximum_joint_stability_span_rad": 0.01,
                    }
                },
                "pose_preservation": {
                    "max_translation_m": 0.0002,
                    "max_orientation_drift_deg": 0.3,
                },
                "closure_alignment": {
                    "close": {
                        "max_p95_angle_deg": 15.0,
                        "per_finger": {
                            finger: {"angle_p95_deg": 15.0}
                            for finger in ("thumb", "index", "mid")
                        },
                    }
                },
                "contact_point_targeting": {
                    "target_radius_m": 0.002,
                    "acquisition_window": {
                        "per_finger": {
                            finger: {"tangent_error_max_m": point_error}
                            for finger in ("thumb", "index", "mid")
                        }
                    },
                },
                "verify_peak_target_face_force_n": dict(
                    zip(("thumb", "index", "mid"), forces)
                ),
                # These are legacy v9 rank keys and must not affect v12.
                "peak_total_distal_contact_force_n": peak_total_force,
                "actuator_saturation_fraction": 0.02,
            },
        },
    }


def test_v12_rank_ignores_thumb_1p50_and_legacy_peak_force_but_penalizes_pair_imbalance():
    far_thumb_high_peak = _complete_v12_rank_record(
        1, thumb_actual=1.40, peak_total_force=100.0
    )
    near_thumb_low_peak = _complete_v12_rank_record(
        2, thumb_actual=1.50, peak_total_force=1.0
    )
    # With every v12 key tied, deterministic candidate ID wins.  The generic
    # v9 thumb-distance/peak-force keys therefore cannot leak into this rank.
    assert campaign._grasp_rank(far_thumb_high_peak) < campaign._grasp_rank(
        near_thumb_low_peak
    )

    pair_imbalanced = _complete_v12_rank_record(
        3, forces=(2.0, 0.1, 1.9)
    )
    pair_balanced = _complete_v12_rank_record(
        4, forces=(2.0, 1.0, 1.0)
    )
    # Both have thumb == index + middle.  The index/middle imbalance must still
    # rank the first record later despite its lower candidate ID.
    assert campaign._grasp_rank(pair_balanced) < campaign._grasp_rank(
        pair_imbalanced
    )


def test_v12_rank_minimum_margin_includes_point_radius_and_missing_is_fail_closed():
    centred = _complete_v12_rank_record(8, point_error=0.001)
    near_boundary = _complete_v12_rank_record(7, point_error=0.0019)
    centred_evidence = campaign.v12_grasp_rank_evidence(centred)
    boundary_evidence = campaign.v12_grasp_rank_evidence(near_boundary)
    assert centred_evidence["minimum_acceptance_margin"] == pytest.approx(0.5)
    assert boundary_evidence["minimum_acceptance_margin"] == pytest.approx(0.05)
    assert campaign._grasp_rank(centred) < campaign._grasp_rank(near_boundary)

    missing = _complete_v12_rank_record(6)
    del missing["summary"]["metrics"]["contact_point_targeting"]
    missing_evidence = campaign.v12_grasp_rank_evidence(missing)
    assert missing_evidence["minimum_acceptance_margin"] is None
    assert missing_evidence["maximum_target_point_error_m"] is None
    assert campaign._grasp_rank(centred) < campaign._grasp_rank(missing)


def test_dynamic_grasp_without_measured_finalization_is_diagnostic_only(tmp_path):
    template = load_config(TEMPLATE)
    source = load_actual_qpos_sources(template)[0]
    plan = ContactPointPlan.from_config(template["contact_point_plan"])
    config = campaign.materialize_v12_source_seed(template, source, plan)
    dynamic_root = tmp_path / "dynamic"
    raw_dynamic = _write_candidate(dynamic_root, config, 701, measured=False)
    assert raw_dynamic["summary"]["stage_status"]["grasp_success"] is True
    candidates = campaign._catalog_candidates((raw_dynamic,), dynamic_root)
    catalog_root = tmp_path / "catalog"
    catalog = campaign._publish_v12_diagnostic_catalog(
        candidates[0],
        catalog_root,
        experiment_id=template["experiment_id"],
        point_plan_id=plan.point_plan_id,
        target_success_count=1,
    )
    assert catalog["success_count"] == 0
    assert catalog["aliases"] == {
        "best_attempt": "grasp_pose_diagnostic_701"
    }
    assert "best_first" not in catalog["aliases"]
    assert "best_nominal" not in catalog["aliases"]
    trajectory = catalog["trajectories"][0]
    assert trajectory["classification"] == "diagnostic"
    assert trajectory["grasp_success"] is False
    assert trajectory["raw_dynamic_grasp_success"] is True
    diagnostic_result = json.loads(
        (catalog_root / trajectory["artifacts"]["result"]).read_text(
            encoding="utf-8"
        )
    )
    assert diagnostic_result["summary"]["stage_status"]["grasp_success"] is False
    assert diagnostic_result["measured_grasp_pose_success"] is False
    assert diagnostic_result["diagnostic_source_evidence"][
        "raw_dynamic_grasp_success"
    ] is True


def test_campaign_counts_measured_grasp_not_full_manipulation(tmp_path, monkeypatch):
    template = load_config(TEMPLATE)
    sources = load_actual_qpos_sources(template)
    policy = ContactPointSearchPolicy.from_config(template)
    plan = ContactPointPlan.from_config(template["contact_point_plan"])
    generated = GeneratedContactPointPlan(
        0, plan, evaluate_contact_point_plan_geometry(plan, policy)
    )
    generation = ContactPointGenerationResult(
        seed=policy.seed,
        sample_count=policy.sample_count,
        eligible_count=1,
        retained=(generated,),
    )
    seeded = campaign.materialize_v12_source_seed(template, sources[0], plan)
    call = {"dls": 0, "dynamic": 0}

    # Keep the orchestration's exact registered reachability cardinality while
    # replacing its expensive numerical content with one deterministic record.
    monkeypatch.setattr(
        campaign,
        "_policy_jobs",
        lambda *_args, **_kwargs: tuple({"slot": index} for index in range(3584)),
    )
    monkeypatch.setattr(
        campaign,
        "_static_pose_jobs",
        lambda *_args, **_kwargs: ({"slot": 0},),
    )

    def fake_dls(_jobs, **_kwargs):
        call["dls"] += 1
        return (_static_record(seeded, 100 + call["dls"]),)

    monkeypatch.setattr(campaign, "_run_dls_jobs", fake_dls)

    def fake_dynamic(records, output, **_kwargs):
        call["dynamic"] += 1
        if not records:
            return ()
        return (
            _write_candidate(
                Path(output), records[0]["config"], 1000 + call["dynamic"], measured=False
            ),
        )

    def fake_measured(records, output, **_kwargs):
        if not records:
            return ()
        return (
            _write_candidate(Path(output), records[0]["config"], 2001, measured=True),
        )

    def fake_manipulability(_config, _trace, _result, **_kwargs):
        score = {
            "probe_count": 17,
            "manipulability_score": 0.75,
            "success_evidence": False,
        }
        return score, tuple({"probe": index} for index in range(17))

    def fake_robustness(_selected, **_kwargs):
        return {
            "complete": True,
            "grasp_only_robustness_schema_version": 1,
            "robust_grasp": False,
            "best_pass_count": 44,
            "required_best_pass_count": 45,
            "trials": [],
        }

    catalog_calls = []

    def fake_catalog(candidates, output, **_kwargs):
        catalog_calls.append(str(output))
        output = Path(output)
        output.mkdir(parents=True)
        candidate = candidates[0]
        trajectory_id = f"grasp_pose_01_{candidate['candidate_id']}"
        member = output / trajectory_id
        member.mkdir()
        resolved_config = json.loads(
            Path(candidate["config_path"]).read_text(encoding="utf-8")
        )
        write_json(member / "resolved_config.json", resolved_config)
        for name, source in (
            ("result.json", Path(candidate["result_path"])),
            ("trace.npz", Path(candidate["trace_path"])),
        ):
            (member / name).write_bytes(source.read_bytes())
        hashes = {
            "resolved_config": campaign.file_sha256(
                member / "resolved_config.json"
            ),
            "result": campaign.file_sha256(member / "result.json"),
            "trace": campaign.file_sha256(member / "trace.npz"),
        }
        payload = {
            "experiment_id": template["experiment_id"],
            "catalog_kind": "grasp_pose",
            "requested_success_count": int(_kwargs["selected_count"]),
            "aliases": {"best_first": trajectory_id, "best_nominal": trajectory_id},
            "selection": {},
            "trajectories": [
                {
                    "trajectory_id": trajectory_id,
                    "candidate_id": candidate["candidate_id"],
                    "classification": "success",
                    "aliases": ["best_first", "best_nominal"],
                    "artifacts": {
                        "resolved_config": f"{trajectory_id}/resolved_config.json",
                        "result": f"{trajectory_id}/result.json",
                        "trace": f"{trajectory_id}/trace.npz",
                        "video": None,
                        "sha256": hashes,
                    },
                }
            ],
        }
        write_json(output / "catalog.json", payload)
        return payload

    monkeypatch.setattr(
        campaign,
        "authenticated_catalog_artifact_paths",
        lambda path: (Path(path),),
    )
    backend = campaign.CampaignBackend(
        source_loader=lambda _config: sources,
        point_generator=lambda _policy: generation,
        dynamic_runner=fake_dynamic,
        measured_runner=fake_measured,
        manipulability_runner=fake_manipulability,
        robustness_runner=fake_robustness,
        catalog_exporter=fake_catalog,
    )
    output = tmp_path / "campaign"
    result = campaign.run_contact_point_targeted_campaign(
        TEMPLATE,
        output,
        resume=False,
        target_success_count=1,
        workers=1,
        seed=20260821,
        backend=backend,
    )

    assert result["target_reached"] is True
    assert result["grasp_success_count"] == 1
    assert result["full_success_count"] == 0
    assert result["manipulation_required_for_success"] is False
    assert result["manipulability_prescreen_is_success_evidence"] is False
    assert result["point_plan_id"] == plan.point_plan_id
    assert (output / result["catalogs"]["grasp_pose"]).is_file()
    ledger = campaign.validate_stage_ledger(output)
    assert "frozen_contact_point_plan" in ledger["stages"]
    assert "point_targeted_manipulability" in ledger["stages"]

    # Simulate power loss after the publisher's atomic directory rename but
    # before the catalog stage reached the ledger.  Resume authenticates and
    # commits the existing directory instead of invoking the exporter again.
    ledger_path = output / "stage_ledger.json"
    interrupted = json.loads(ledger_path.read_text(encoding="utf-8"))
    interrupted["stages"].pop("point_targeted_grasp_catalog_1")
    write_json(ledger_path, interrupted)
    repeated = campaign.run_contact_point_targeted_campaign(
        TEMPLATE,
        output,
        resume=True,
        target_success_count=1,
        workers=1,
        seed=20260821,
        backend=backend,
    )
    assert repeated["target_reached"] is True
    assert len(catalog_calls) == 1

    resumed = campaign.run_contact_point_targeted_campaign(
        TEMPLATE,
        output,
        resume=True,
        target_success_count=5,
        workers=1,
        seed=20260821,
        backend=backend,
    )
    assert resumed["grasp_success_count"] == 1
    assert resumed["target_reached"] is False
    assert resumed["stop_reason"] == "declared_budget_exhausted_before_target_grasp_count"

    measured_trace = output / "dynamic" / "measured" / "candidate_2001" / "trace.npz"
    measured_trace.write_bytes(measured_trace.read_bytes() + b"tampered")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        campaign.run_contact_point_targeted_campaign(
            TEMPLATE,
            output,
            resume=True,
            target_success_count=5,
            workers=1,
            seed=20260821,
            backend=backend,
        )


def test_no_reachable_plan_still_commits_diagnostic_contact_point_catalog(
    tmp_path, monkeypatch
):
    template = load_config(TEMPLATE)
    sources = load_actual_qpos_sources(template)
    policy = ContactPointSearchPolicy.from_config(template)
    plan = ContactPointPlan.from_config(template["contact_point_plan"])
    generated = GeneratedContactPointPlan(
        0, plan, evaluate_contact_point_plan_geometry(plan, policy)
    )
    generation = ContactPointGenerationResult(
        seed=policy.seed,
        sample_count=policy.sample_count,
        eligible_count=1,
        retained=(generated,),
    )
    config = campaign.materialize_v12_source_seed(template, sources[0], plan)
    failed = _static_record(config, 999)
    failed["static_pass"] = False
    failed["static_metrics"]["point_target"]["static_acceptance"] = {
        "passed": False,
        "reasons": ["point_target_radius_failed:thumb"],
    }
    monkeypatch.setattr(
        campaign,
        "_policy_jobs",
        lambda *_args, **_kwargs: tuple({"slot": index} for index in range(3584)),
    )
    monkeypatch.setattr(
        campaign, "_run_dls_jobs", lambda *_args, **_kwargs: (failed,)
    )
    output = tmp_path / "no_reach"
    result = campaign.run_contact_point_targeted_campaign(
        TEMPLATE,
        output,
        resume=False,
        target_success_count=1,
        workers=1,
        seed=20260821,
        backend=campaign.CampaignBackend(
            source_loader=lambda _config: sources,
            point_generator=lambda _policy: generation,
        ),
    )
    assert result["stop_reason"] == "no_reachable_contact_point_plan"
    catalog_path = output / result["catalogs"]["contact_point"]
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    assert catalog["selected_point_plan_id"] is None
    assert len(catalog["plans"]) == 1
    assert len(catalog["reachability_candidates"]) == 1
    assert "frozen_contact_point_plan" not in campaign.validate_stage_ledger(output)[
        "stages"
    ]
