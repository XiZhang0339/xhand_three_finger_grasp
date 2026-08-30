from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import xhand_grasp.high_thumb_size_catalog as catalog_module
from xhand_grasp.config import load_config
from xhand_grasp.pose_preserving_seed_catalog import REQUIRED_ACQUISITION_CHECKS
from xhand_grasp.scene import rpy_degrees_to_quaternion
from xhand_grasp.high_thumb_size_catalog import (
    CAMPAIGN_KIND,
    EXPERIMENT_ID,
    CampaignCandidate,
    export_high_thumb_size_catalog,
    select_diverse_acquisition_candidates,
    thumb_bend_band,
)
from xhand_grasp.viewer import resolve_viewer_source


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_high_thumb_variable_size_"
    "pose_preserving_grasp_then_lift.json"
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selection_candidate(
    candidate_id: str,
    *,
    edge_mm: int,
    thumb: float,
    family: str,
    pose_margin: float = 0.5,
) -> CampaignCandidate:
    return CampaignCandidate(
        candidate_id=candidate_id,
        stage="acquisition",
        search_stage="exact",
        source_family_id=family,
        source_trajectory_id=f"source_{family}",
        edge_m=edge_mm / 1000.0,
        thumb_target_rad=thumb,
        acquisition_success=True,
        pose_preservation_success=True,
        lift_success=False,
        pose_margin=pose_margin,
        stable_grasp_margin=0.4,
        all_state_gate_steps=250.0,
        alignment_duty=1.0,
        alignment_height_p95_m=0.004,
        pad_force_fraction=1.0,
        target_face_duty=1.0,
        peak_total_distal_force_n=1.0,
        saturation_duty=0.0,
        config={},
        result={},
        config_path=Path("config.json"),
        result_path=Path("result.json"),
        trace_path=Path("trace.npz"),
        video_path=None,
        artifact_sha256={},
        trace_validation={"passed": True, "failed_checks": []},
    )


def _diverse_candidates() -> list[CampaignCandidate]:
    bands = (1.30, 1.35, 1.40)
    values: list[CampaignCandidate] = []
    for band_index, thumb in enumerate(bands):
        for local_index, edge in enumerate((52, 54, 56, 58)):
            values.append(
                _selection_candidate(
                    f"c{band_index}{local_index}",
                    edge_mm=edge,
                    thumb=thumb,
                    family=("seed_a", "seed_b", "seed_c")[
                        (band_index + local_index) % 3
                    ],
                    pose_margin=0.4 + local_index * 0.01,
                )
            )
    return values


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1.25, "1p25_to_1p31"),
        (1.31, "1p25_to_1p31"),
        (1.32, "gt_1p31_to_1p38"),
        (1.38, "gt_1p31_to_1p38"),
        (1.39, "gt_1p38_to_1p45"),
        (1.45, "gt_1p38_to_1p45"),
    ],
)
def test_thumb_bend_bands_have_canonical_closed_boundaries(
    value: float, expected: str
) -> None:
    assert thumb_bend_band(value) == expected


def test_result_experiment_status_is_authoritative_over_a_stale_summary() -> None:
    result = {
        "acquisition_success": True,
        "lift_success": True,
        "experiment_status": {
            "passed": False,
            "grasp_success": False,
            "full_success": False,
        },
        "summary": {
            "passed": True,
            "stage_status": {"grasp_success": True, "full_success": True},
        },
    }

    assert catalog_module._stage_grasp_success(result) is False
    assert catalog_module._stage_lift_success(result) is False


def test_missing_soft_rank_metric_uses_deterministic_fallback() -> None:
    result = {"rank_metrics": {"grasp_stability_margin": None}}

    assert (
        catalog_module._rank_metric(
            result,
            ("grasp_stability_margin", "stable_grasp_margin"),
            0.0,
        )
        == 0.0
    )


def test_near_miss_rank_prefers_real_gate_evidence_over_larger_thumb_target() -> None:
    high_target_no_gate = replace(
        _selection_candidate(
            "high_target_no_gate",
            edge_mm=60,
            thumb=1.45,
            family="seed_a",
        ),
        acquisition_success=False,
        pose_preservation_success=False,
        pose_margin=-4.0,
        all_state_gate_steps=0.0,
        alignment_duty=0.0,
    )
    lower_target_long_gate = replace(
        _selection_candidate(
            "lower_target_long_gate",
            edge_mm=60,
            thumb=1.30,
            family="seed_b",
        ),
        acquisition_success=False,
        pose_preservation_success=True,
        pose_margin=0.4,
        all_state_gate_steps=220.0,
        alignment_duty=0.9,
    )

    assert catalog_module._near_miss_rank(
        lower_target_long_gate
    ) < catalog_module._near_miss_rank(high_target_no_gate)


def test_diverse_selection_is_worker_order_independent_and_meets_every_quota() -> None:
    forward = select_diverse_acquisition_candidates(_diverse_candidates())
    reverse = select_diverse_acquisition_candidates(reversed(_diverse_candidates()))

    assert [value.candidate_id for value in forward] == [
        value.candidate_id for value in reverse
    ]
    assert len(forward) == 12
    assert len({value.edge_mm for value in forward}) >= 4
    assert len({value.source_family_id for value in forward}) >= 3
    assert {
        band: sum(value.bend_band == band for value in forward)
        for band in {value.bend_band for value in forward}
    } == {
        "1p25_to_1p31": 4,
        "gt_1p31_to_1p38": 4,
        "gt_1p38_to_1p45": 4,
    }
    pair_counts = {
        pair: sum(value.edge_target_pair == pair for value in forward)
        for pair in {value.edge_target_pair for value in forward}
    }
    assert max(pair_counts.values()) <= 2


def test_diverse_selection_never_fills_a_missing_band_with_duplicate_successes() -> None:
    candidates = [
        value
        for value in _diverse_candidates()
        if value.bend_band != "gt_1p38_to_1p45"
    ]
    candidates.extend(copy.deepcopy(candidates[:4]))
    for index, candidate in enumerate(candidates[-4:]):
        object.__setattr__(candidate, "candidate_id", f"duplicate_{index}")

    with pytest.raises(ValueError, match="bend-band quota is infeasible"):
        select_diverse_acquisition_candidates(candidates)


def _trace(
    edge_m: float,
    *,
    cube_rpy_deg: tuple[float, float, float],
    translation_m: float = 0.0002,
) -> dict[str, np.ndarray]:
    total = 300
    acquisition = 249
    time = np.arange(1, total + 1, dtype=np.float64) * 0.001
    states = np.full(total, "VERIFY", dtype="<U10")
    states[:25] = "SETTLE"
    center_z = 0.084 + edge_m / 2.0
    initial = np.asarray([0.071, -0.027, center_z], dtype=np.float64)
    positions = np.repeat(initial[None, :], total, axis=0)
    positions[25:, 0] += translation_m
    initial_quaternion = rpy_degrees_to_quaternion(cube_rpy_deg)
    quaternions = np.repeat(initial_quaternion[None, :], total, axis=0)
    return {
        "time": time,
        "cube_pos": positions,
        "cube_quat": quaternions,
        "initial_cube_pos_m": initial,
        "initial_cube_quat": initial_quaternion,
        "grasp_acquisition_step": np.asarray(acquisition, dtype=np.int64),
        "manipulation_start_step": np.asarray(250, dtype=np.int64),
        "manipulation_end_step": np.asarray(299, dtype=np.int64),
        "termination_step": np.asarray(299, dtype=np.int64),
        "control_state": states,
        "support_contact": np.ones(total, dtype=bool),
        "hand_cube_contact": np.concatenate(
            (np.zeros(25, dtype=bool), np.ones(total - 25, dtype=bool))
        ),
        "grasp_acquired": np.arange(total) >= acquisition,
        "manipulation_progress": np.zeros(total, dtype=np.float64),
        "actuator_order": np.asarray(("thumb", "index", "mid")),
    }


def _write_candidate(
    root: Path,
    *,
    candidate_id: str,
    edge_mm: int,
    thumb: float,
    family: str,
    stage: str = "acquisition",
) -> tuple[dict, dict]:
    directory = root / "candidates" / f"candidate_{candidate_id}"
    directory.mkdir(parents=True)
    full = stage == "lift"
    search_stage = "lift" if full else "exact"
    config = load_config(CONFIG)
    config["cube"]["edge_m"] = edge_mm / 1000.0
    config["cube"]["mass_kg"] = 0.160
    config["cube"]["friction"] = 0.8
    config["control"]["grasp_targets_rad"][
        "left_hand_thumb_bend_joint_actuator"
    ] = thumb
    config["candidate_metadata"] = {
        "candidate_id": candidate_id,
        "stage": search_stage,
        "source_family_id": family,
        "source_trajectory_id": f"source_{family}",
        "edge_m": edge_mm / 1000.0,
        "thumb_target_rad": thumb,
        "cube_pose_sampled": False,
        "free_cube_pose_reset_during_run": False,
    }
    config_path = directory / "resolved_config.json"
    config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    trace_path = directory / "trace.npz"
    np.savez_compressed(
        trace_path,
        **_trace(
            edge_mm / 1000.0,
            cube_rpy_deg=tuple(config["cube"]["rpy_deg"]),
        ),
    )
    summary = {
        "passed": full,
        "checks": {name: True for name in REQUIRED_ACQUISITION_CHECKS},
        "stage_status": {
            "grasp_success": True,
            "manipulation_success": full,
            "full_success": full,
        },
        "metrics": {
            "grasp_acquisition_step": 249,
            "pose_preservation": {
                "max_translation_m": 0.0002,
                "max_orientation_drift_deg": 0.0,
            },
        },
    }
    candidate_sha = _canonical_sha256(config)
    result = {
        "candidate_result_schema_version": 1,
        "campaign_kind": CAMPAIGN_KIND,
        "complete": True,
        "stage": stage,
        "search_stage": search_stage,
        "candidate_id": candidate_id,
        "candidate_sha256": candidate_sha,
        "source_family_id": family,
        "source_trajectory_id": f"source_{family}",
        "edge_m": edge_mm / 1000.0,
        "thumb_target_rad": thumb,
        "summary": summary,
        "classification": "validated_lift" if full else "validated_grasp",
        "grasp_success": True,
        "acquisition_success": True,
        "pose_preservation_success": True,
        "lift_success": full,
        "rank_metrics": {
            "max_translation_before_acquisition_m": 0.0002,
            "max_orientation_before_acquisition_deg": 0.0,
            "stable_grasp_margin": 0.4,
            "alignment_duty": 1.0,
            "pad_force_fraction": 1.0,
            "saturation_duty": 0.0,
        },
        "artifacts": {
            "resolved_config": config_path.name,
            "trace": trace_path.name,
            "video": None,
            "sha256": {
                "resolved_config": _file_sha256(config_path),
                "trace": _file_sha256(trace_path),
            },
        },
    }
    result_path = directory / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_record = {
        "candidate_id": candidate_id,
        "candidate_sha256": candidate_sha,
        "stage": stage,
        "search_stage": search_stage,
        "source_family_id": family,
        "edge_m": edge_mm / 1000.0,
        "thumb_target_rad": thumb,
    }
    campaign_record = {
        "candidate_id": candidate_id,
        "stage": stage,
        "search_stage": search_stage,
        "candidate_sha256": candidate_sha,
        "result": str(result_path.relative_to(root)),
        "result_sha256": _file_sha256(result_path),
        "classification": result["classification"],
        "acquisition_success": True,
        "pose_preservation_success": True,
        "lift_success": full,
    }
    return manifest_record, campaign_record


def _write_campaign(root: Path, *, include_lift: bool = True) -> None:
    root.mkdir(parents=True)
    manifest_records = []
    campaign_records = []
    for candidate in _diverse_candidates():
        manifest, campaign = _write_candidate(
            root,
            candidate_id=candidate.candidate_id,
            edge_mm=candidate.edge_mm,
            thumb=candidate.thumb_target_rad,
            family=candidate.source_family_id,
        )
        manifest_records.append(manifest)
        campaign_records.append(campaign)
    if include_lift:
        manifest, campaign = _write_candidate(
            root,
            candidate_id="lift_000",
            edge_mm=58,
            thumb=1.40,
            family="seed_c",
            stage="lift",
        )
        manifest_records.append(manifest)
        campaign_records.append(campaign)
    shared_hash = "a" * 64
    manifest = {
        "campaign_schema_version": 1,
        "campaign_kind": CAMPAIGN_KIND,
        "experiment_id": EXPERIMENT_ID,
        "campaign_input_sha256": shared_hash,
        "budget": {"perturbations_per_grasp": 16},
        "candidates": manifest_records,
    }
    campaign = {
        "campaign_schema_version": 1,
        "complete": True,
        "campaign_kind": CAMPAIGN_KIND,
        "experiment_id": EXPERIMENT_ID,
        "campaign_input_sha256": shared_hash,
        "selected_grasp_count": 12,
        "selection_satisfied": True,
        "selected_grasps": [
            {
                "candidate_id": record["candidate_id"],
                "source_family_id": next(
                    item["source_family_id"]
                    for item in manifest_records
                    if item["candidate_id"] == record["candidate_id"]
                ),
                "edge_m": next(
                    item["edge_m"]
                    for item in manifest_records
                    if item["candidate_id"] == record["candidate_id"]
                ),
                "thumb_target_rad": next(
                    item["thumb_target_rad"]
                    for item in manifest_records
                    if item["candidate_id"] == record["candidate_id"]
                ),
                "perturbation_count": 16,
                "perturbation_pass_count": 16,
            }
            for record in campaign_records
            if record["stage"] == "acquisition"
        ],
        "candidate_results": campaign_records,
    }
    (root / "campaign_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "campaign_results.json").write_text(
        json.dumps(campaign, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _mark_exact_candidate_as_near_miss(
    root: Path,
    candidate_id: str,
    *,
    all_state_gate_steps: int,
    include_video: bool = False,
) -> None:
    result_path = (
        root / "candidates" / f"candidate_{candidate_id}" / "result.json"
    )
    result = json.loads(result_path.read_text())
    result["classification"] = "high_thumb_pose_preserved_grasp_not_acquired"
    result["grasp_success"] = False
    result["acquisition_success"] = False
    result["pose_preservation_success"] = True
    result["lift_success"] = False
    result["summary"]["stage_status"]["grasp_success"] = False
    result["summary"]["checks"]["stable_grasp_acquired"] = False
    result["rank_metrics"]["all_state_gate_steps"] = all_state_gate_steps
    if include_video:
        video_path = result_path.parent / "trajectory.mp4"
        video_path.write_bytes(b"authenticated-test-video")
        result["artifacts"]["video"] = video_path.name
        result["artifacts"]["sha256"]["video"] = _file_sha256(video_path)
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    campaign_path = root / "campaign_results.json"
    campaign = json.loads(campaign_path.read_text())
    binding = next(
        value
        for value in campaign["candidate_results"]
        if str(value["candidate_id"]) == candidate_id
    )
    binding.update(
        {
            "classification": result["classification"],
            "acquisition_success": False,
            "pose_preservation_success": True,
            "lift_success": False,
            "result_sha256": _file_sha256(result_path),
        }
    )
    campaign_path.write_text(
        json.dumps(campaign, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_export_authenticates_selects_reports_and_writes_viewer_catalog(
    tmp_path: Path,
) -> None:
    source = tmp_path / "campaign"
    output = tmp_path / "catalog"
    _write_campaign(source)

    catalog = export_high_thumb_size_catalog(source, output)

    assert catalog["selected_grasp_count"] == 12
    assert catalog["validated_lift_count"] == 1
    assert catalog["trajectory_count"] == 13
    assert catalog["highest_thumb_target_rad"] == pytest.approx(1.40)
    assert "highest_thumb_target" in catalog["aliases"]
    assert "best_pose_margin" in catalog["aliases"]
    assert "best_lift" in catalog["aliases"]
    assert set(catalog["aliases"]).issuperset(
        {f"grasp_{candidate.candidate_id}" for candidate in _diverse_candidates()}
    )
    source_for_viewer = resolve_viewer_source(
        catalog_path=output / "catalog.json", trajectory="highest_thumb_target"
    )
    assert source_for_viewer.config_path.is_file()
    assert source_for_viewer.trace_path is not None
    assert source_for_viewer.trace_path.is_file()
    grasp_catalog = json.loads((output / "grasp_catalog.json").read_text())
    assert grasp_catalog["catalog_stage"] == "acquisition"
    assert grasp_catalog["trajectory_count"] == 12
    assert all(
        entry["stage"] == "acquisition"
        for entry in grasp_catalog["trajectories"]
    )
    assert not any(alias.startswith("lift_") for alias in grasp_catalog["aliases"])
    grasp_viewer_source = resolve_viewer_source(
        catalog_path=output / "grasp_catalog.json",
        trajectory="highest_thumb_target",
    )
    assert grasp_viewer_source.config_path == source_for_viewer.config_path
    lift_catalog = json.loads((output / "lift_catalog.json").read_text())
    assert lift_catalog["catalog_stage"] == "lift"
    assert lift_catalog["trajectory_count"] == 1
    assert set(lift_catalog["aliases"]) == {"lift_lift_000", "best_lift"}
    lift_viewer_source = resolve_viewer_source(
        catalog_path=output / "lift_catalog.json", trajectory="best_lift"
    )
    assert lift_viewer_source.config_path.is_file()
    report = json.loads((output / "selection_report.json").read_text())
    assert report["quota_achieved"]["distinct_edges_mm"] == [52, 54, 56, 58]
    assert len(report["quota_achieved"]["distinct_seed_families"]) == 3
    assert report["quota_achieved"]["maximum_edge_target_pair_count"] <= 2
    for subcatalog in ("grasp", "lift"):
        member = output / catalog["catalogs"][subcatalog]["path"]
        assert _file_sha256(member) == catalog["catalogs"][subcatalog]["sha256"]
        assert report["catalogs"][subcatalog] == catalog["catalogs"][subcatalog]
    assert (output / "EXPERIMENT_REPORT.md").is_file()
    markdown_report = output / catalog["experiment_report"]["path"]
    assert _file_sha256(markdown_report) == catalog["experiment_report"]["sha256"]
    for entry in catalog["trajectories"]:
        artifacts = entry["artifacts"]
        for field in ("resolved_config", "result", "trace", "grasp_trace"):
            path = output / artifacts[field]
            assert _file_sha256(path) == artifacts["sha256"][field]


def test_export_rejects_trace_tampering_before_creating_output(tmp_path: Path) -> None:
    source = tmp_path / "campaign"
    output = tmp_path / "catalog"
    _write_campaign(source, include_lift=False)
    trace = source / "candidates" / "candidate_c00" / "trace.npz"
    with trace.open("ab") as stream:
        stream.write(b"tampered")

    with pytest.raises(ValueError, match="SHA-256 mismatch for trace"):
        export_high_thumb_size_catalog(source, output)
    assert not output.exists()


def test_export_rejects_result_status_not_bound_by_campaign(tmp_path: Path) -> None:
    source = tmp_path / "campaign"
    _write_campaign(source, include_lift=False)
    campaign_path = source / "campaign_results.json"
    campaign = json.loads(campaign_path.read_text())
    campaign["candidate_results"][0]["acquisition_success"] = False
    campaign_path.write_text(
        json.dumps(campaign, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="acquisition_success disagrees"):
        export_high_thumb_size_catalog(source, tmp_path / "catalog")


def test_lift_catalog_is_honestly_empty_when_no_full_success_exists(
    tmp_path: Path,
) -> None:
    source = tmp_path / "campaign"
    output = tmp_path / "catalog"
    _write_campaign(source, include_lift=False)

    catalog = export_high_thumb_size_catalog(source, output)

    assert catalog["validated_lift_count"] == 0
    assert "best_lift" not in catalog["aliases"]
    lift = json.loads((output / "lift_catalog.json").read_text())
    assert lift["trajectory_count"] == 0
    assert lift["validated_trajectory_count"] == 0
    assert lift["campaign_has_validated_trajectory"] is False
    assert lift["aliases"] == {}
    assert lift["trajectories"] == []


def test_incomplete_diversity_is_reported_without_success_aliases_or_exception(
    tmp_path: Path,
) -> None:
    source = tmp_path / "campaign"
    output = tmp_path / "catalog"
    _write_campaign(source, include_lift=False)
    campaign_path = source / "campaign_results.json"
    campaign = json.loads(campaign_path.read_text())
    campaign["selected_grasps"] = campaign["selected_grasps"][:5]
    campaign["selected_grasp_count"] = 5
    campaign["selection_satisfied"] = False
    campaign_path.write_text(
        json.dumps(campaign, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    catalog = export_high_thumb_size_catalog(source, output)

    assert catalog["quota_satisfied"] is False
    assert catalog["selected_grasp_count"] == 5
    assert catalog["quota_deficiencies"]
    assert "best_nominal" not in catalog["aliases"]
    assert "highest_thumb_target" not in catalog["aliases"]
    assert "best_pose_margin" not in catalog["aliases"]
    assert "best_attempt" in catalog["aliases"]
    assert all(
        alias.startswith("grasp_") or alias == "best_attempt"
        for alias in catalog["aliases"]
    )
    report = json.loads((output / "selection_report.json").read_text())
    assert report["quota_satisfied"] is False
    assert report["quota_deficiencies"] == catalog["quota_deficiencies"]


def test_zero_selected_passes_still_publish_empty_honest_catalogs(
    tmp_path: Path,
) -> None:
    source = tmp_path / "campaign"
    output = tmp_path / "catalog"
    _write_campaign(source, include_lift=False)
    _mark_exact_candidate_as_near_miss(
        source, "c00", all_state_gate_steps=225, include_video=True
    )
    campaign_path = source / "campaign_results.json"
    campaign = json.loads(campaign_path.read_text())
    campaign["selected_grasps"] = []
    campaign["selected_grasp_count"] = 0
    campaign["selection_satisfied"] = False
    campaign_path.write_text(
        json.dumps(campaign, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    catalog = export_high_thumb_size_catalog(source, output)

    assert catalog["quota_satisfied"] is False
    assert catalog["trajectory_count"] == 0
    assert catalog["aliases"] == {}
    assert catalog["highest_thumb_target_rad"] is None
    grasp = json.loads((output / "grasp_catalog.json").read_text())
    lift = json.loads((output / "lift_catalog.json").read_text())
    assert grasp["trajectories"] == []
    assert grasp["aliases"] == {}
    assert lift["trajectories"] == []
    assert lift["aliases"] == {}
    assert "best_near_miss" not in catalog["aliases"]
    diagnostics = catalog["diagnostics"]["best_near_miss"]
    assert diagnostics is not None
    assert diagnostics["directory"] == "diagnostics/best_near_miss"
    for field in ("resolved_config", "result", "trace", "video"):
        diagnostic_path = output / diagnostics[field]
        assert diagnostic_path.is_file()
        assert _file_sha256(diagnostic_path) == diagnostics["sha256"][field]
    report = json.loads((output / "selection_report.json").read_text())
    near_miss = report["best_near_miss"]
    assert near_miss["candidate_id"] == "c00"
    assert near_miss["rank_evidence"]["all_state_gate_steps"] == 225
    assert near_miss["artifacts"] == diagnostics
    source_provenance = diagnostics["source_provenance"]
    assert Path(source_provenance["resolved_config"]).is_file()
    assert Path(source_provenance["result"]).is_file()
    assert Path(source_provenance["trace"]).is_file()
