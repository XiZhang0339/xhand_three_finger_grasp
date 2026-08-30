from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from xhand_grasp.grasp_pose import canonical_sha256
from xhand_grasp.tuning import contact_preserving_micro_jerk_rescue as numerical
from xhand_grasp.tuning import contact_preserving_micro_jerk_rescue_campaign as campaign


SOURCE = Path(
    "artifacts/left_opposed_face_palm_down_contact_preserving_planned_lift/"
    "tune/formal_campaign_v14_1_force_debias_rescue_v2"
)
CONFIG = Path(
    "artifacts/left_opposed_face_palm_down_contact_preserving_planned_lift/"
    "tune/micro_jerk_design_evidence_v1/candidate_best_round2/resolved_config.json"
)


def test_manifest_binds_supplemental_source_budget_and_exclusions():
    manifest = campaign.build_contact_preserving_micro_jerk_rescue_manifest(CONFIG, SOURCE)
    assert manifest["campaign_kind"] == "contact_preserving_bounded_micro_jerk_rescue"
    assert manifest["micro_jerk_budget"] == {
        "center_count": 8,
        "sensitivity_per_center": 32,
        "trust_radii": [0.015, 0.03, 0.06],
        "trust_per_radius_per_center": 32,
        "candidate_count": 1024,
        "publication_candidate_count": 5,
        "candidate_execution": "fresh_full_reset_free_dynamics",
        "catalog_execution": "independent_top_five_fresh_full_reset_rerun",
        "global_physical_exclusion": True,
    }
    source = manifest["source_authentication"]
    assert source["centers"][0]["candidate_id"] == 15043844903285998
    assert len(source["prior_physical_plan_sha256"]) == 673
    assert manifest["campaign_input_sha256"] == canonical_sha256(
        {key: value for key, value in manifest.items() if key != "campaign_input_sha256"}
    )


def test_campaign_job_normalization_reauthenticates_fixed_schedule():
    api = campaign._micro_jerk_api()
    source = api.source_authenticator(SOURCE)
    raw = api.jobs_builder(source)
    jobs = campaign._normalize_jobs(raw, api=api, source=source)
    assert len(jobs) == 1024
    assert jobs[0]["job_sequence_index"] == 0
    assert jobs[-1]["job_sequence_index"] == 1023
    assert all(value["job_metadata"]["runner_stage"] == "micro_refinement" for value in jobs)


def test_campaign_job_normalization_rejects_tamper():
    api = campaign._micro_jerk_api()
    source = api.source_authenticator(SOURCE)
    raw = list(api.jobs_builder(source))
    raw[0] = copy.deepcopy(raw[0])
    raw[0]["candidate_id"] += 1
    with pytest.raises(RuntimeError):
        campaign._normalize_jobs(raw, api=api, source=source)


def test_branch_stability_counts_taxel_transitions_and_valid_witness_jumps(tmp_path):
    length = 12
    taxels = np.ones((length, 3), dtype=np.int64)
    taxels[5:, 0] = 2
    taxels[7:, 1] = 3
    centroids = np.zeros((length, 3, 3), dtype=np.float64)
    centroids[6:, 0, 1] = 0.0004
    centroids[8:, 2, 2] = 0.0007
    valid = np.ones((length, 3), dtype=bool)
    valid[8, 2] = False
    path = tmp_path / "trace.npz"
    np.savez_compressed(
        path,
        distal_active_taxel_count=taxels,
        target_face_contact_centroid_cube_local_m=centroids,
        target_face_contact_centroid_valid=valid,
        manipulation_start_step=np.asarray(2),
        manipulation_end_step=np.asarray(11),
    )
    result = campaign.branch_stability_from_trace(path)
    assert result["taxel_switch_count_per_finger"] == [1, 1, 0]
    assert result["total_taxel_switch_count"] == 2
    assert result["thumb_taxel_switch_count"] == 1
    assert result["maximum_witness_jump_m"] == pytest.approx(0.0004)


def test_phase_payload_reports_physical_uniqueness():
    records = [
        {
            "candidate_id": index,
            "full_success": index == 0,
            "rescue_job": {"physical_plan_sha256": f"{index + 1:064x}"},
        }
        for index in range(3)
    ]
    payload = campaign._phase_payload(records, "a" * 64)
    assert payload["candidate_count"] == 3
    assert payload["full_success_count"] == 1
    assert payload["physical_unique_candidate_count"] == 3
    assert payload["physical_duplicate_candidate_count"] == 0


def test_branch_stability_rejects_shape_damage(tmp_path):
    path = tmp_path / "trace.npz"
    np.savez_compressed(
        path,
        distal_active_taxel_count=np.ones((4, 2), dtype=np.int64),
        target_face_contact_centroid_cube_local_m=np.zeros((4, 3, 3)),
        target_face_contact_centroid_valid=np.ones((4, 3), dtype=bool),
        manipulation_start_step=np.asarray(1),
        manipulation_end_step=np.asarray(3),
    )
    with pytest.raises(RuntimeError, match="shapes"):
        campaign.branch_stability_from_trace(path)
