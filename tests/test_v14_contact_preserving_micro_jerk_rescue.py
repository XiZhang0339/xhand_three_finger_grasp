from __future__ import annotations

import copy
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest

from xhand_grasp.tuning import contact_preserving_micro_jerk_rescue as micro


SOURCE = Path(
    "artifacts/left_opposed_face_palm_down_contact_preserving_planned_lift/"
    "tune/formal_campaign_v14_1_force_debias_rescue_v2"
)


def test_compact_band_is_c2_and_has_exact_support():
    start, peak, end = 0.2, 0.5, 0.9
    assert micro.compact_c2_band([start - 1e-6, start, peak, end, end + 1e-6], start=start, peak=peak, end=end).tolist() == [0.0, 0.0, 1.0, 0.0, 0.0]
    spacing = 1e-5
    for junction, expected in ((start, 0.0), (peak, 1.0), (end, 0.0)):
        values = micro.compact_c2_band(
            [junction - 2 * spacing, junction - spacing, junction, junction + spacing, junction + 2 * spacing],
            start=start,
            peak=peak,
            end=end,
        )
        first_left = (values[2] - values[1]) / spacing
        first_right = (values[3] - values[2]) / spacing
        second_left = (values[2] - 2 * values[1] + values[0]) / spacing**2
        second_right = (values[4] - 2 * values[3] + values[2]) / spacing**2
        assert values[2] == expected
        assert abs(first_left) < 1e-6 and abs(first_right) < 1e-6
        # The analytic minimum-jerk second derivative is exactly zero at the
        # join.  The five-point one-sided estimate divides values of order
        # 1e-13 by h**2, so leave a small floating-point cancellation margin.
        assert abs(second_left) < 0.03 and abs(second_right) < 0.03


def test_authenticated_source_includes_round2_and_fixed_center_design():
    source = micro.authenticate_micro_jerk_source(SOURCE)
    assert len(source.centers) == 8
    assert Counter(value.center_kind for value in source.centers) == {
        "jerk_only": 6,
        "contact_boundary": 2,
    }
    assert source.centers[0].candidate_id == 15043844903285998
    assert "micro_jerk_design_evidence_v1" in str(source.centers[0].result_path)
    assert len(source.prior_physical_plan_sha256) == 673
    assert {0.18, 0.22}.issubset(
        {value.event_half_width_progress for value in source.centers}
    )
    assert len({value.source_candidate_id for value in source.centers}) >= 2


def test_fixed_1024_schedule_is_unique_sparse_and_authentic():
    source = micro.authenticate_micro_jerk_source(SOURCE)
    jobs = micro.build_micro_jerk_jobs(source)
    assert len(jobs) == micro.MICRO_CANDIDATE_COUNT == 1024
    assert len({value["candidate_id"] for value in jobs}) == 1024
    assert len({value["physical_plan_sha256"] for value in jobs}) == 1024
    assert not {value["physical_plan_sha256"] for value in jobs}.intersection(
        source.prior_physical_plan_sha256
    )
    grouped = defaultdict(list)
    for job in jobs:
        micro.authenticate_micro_jerk_job(
            job,
            source,
            expected_excluded_physical_plan_sha256=source.prior_physical_plan_sha256,
        )
        grouped[job["source_center_id"]].append(job)
    assert len(grouped) == 8
    for values in grouped.values():
        assert Counter(value["sampling_mode"] for value in values) == {
            "deterministic_sensitivity": 32,
            "trust_shell": 96,
        }
        assert Counter(
            value["trust_radius"]
            for value in values
            if value["sampling_mode"] == "trust_shell"
        ) == {0.015: 32, 0.03: 32, 0.06: 32}
        for value in values:
            if value["sampling_mode"] != "trust_shell":
                continue
            parameters = value["normalized_parameters"]
            active_clusters = 0
            for band in ("early", "main", "terminal"):
                names = [
                    f"band:{band}:thumb",
                    f"band:{band}:index",
                    f"band:{band}:mid",
                    f"band:{band}:center_shift",
                    f"band:{band}:width_scale_delta",
                ]
                active_clusters += int(any(abs(float(parameters[name])) > 0.0 for name in names))
            assert active_clusters == 1


def test_job_authentication_rejects_parameter_tamper():
    source = micro.authenticate_micro_jerk_source(SOURCE)
    job = copy.deepcopy(micro.build_micro_jerk_jobs(source)[0])
    name = next(iter(job["normalized_parameters"]))
    job["normalized_parameters"][name] += 1e-6
    with pytest.raises(RuntimeError, match="payload SHA-256"):
        micro.authenticate_micro_jerk_job(
            job,
            source,
            expected_excluded_physical_plan_sha256=source.prior_physical_plan_sha256,
        )


def test_hard_first_rank_uses_branch_only_as_soft_tie_break():
    base = {
        "candidate_id": 4,
        "full_success": False,
        "summary": {
            "failed_checks": ["smooth_motion_jerk_within_limit"],
            "metrics": {
                "motion_smoothness": {
                    "operation_peak_abs_filtered_jerk_m_s3": 3.1,
                }
            },
        },
    }
    branchy = {**copy.deepcopy(base), "candidate_id": 5, "branch_stability": {"total_taxel_switch_count": 8, "maximum_witness_jump_m": 0.001}}
    stable = {**copy.deepcopy(base), "candidate_id": 6, "branch_stability": {"total_taxel_switch_count": 1, "maximum_witness_jump_m": 0.002}}
    success = {**copy.deepcopy(branchy), "candidate_id": 7, "full_success": True, "summary": {"failed_checks": [], "metrics": base["summary"]["metrics"]}}
    assert micro.micro_jerk_candidate_rank(success) < micro.micro_jerk_candidate_rank(stable)
    assert micro.micro_jerk_candidate_rank(stable) < micro.micro_jerk_candidate_rank(branchy)
