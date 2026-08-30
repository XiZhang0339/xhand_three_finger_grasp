from __future__ import annotations

import copy
import random
from collections import Counter

import numpy as np
import pytest

from xhand_grasp.config import ACTIVE_ACTUATORS, load_config
from xhand_grasp.experiment import ManipulationPlanParameters
from xhand_grasp.tuning.contact_preserving_joint_refinement import (
    resolve_joint_refinement_limits,
)
from xhand_grasp.tuning.contact_preserving_time_warp import (
    COEFFICIENT_BOUNDS,
    DEFAULT_TOTAL_CANDIDATES,
    MAX_SEGMENT_DURATION_S,
    MIN_SEGMENT_DURATION_S,
    TimeWarpBudget,
    actuator_bezier_hull,
    apply_time_warp_to_config,
    authenticate_time_warp_job,
    build_contact_preserving_time_warp_jobs,
    stable_sort_time_warp_jobs,
    time_warp_campaign_manifest,
    validate_time_warped_config,
    warp_knot_times,
    zero_mean_gaussian_bases,
)


@pytest.fixture(scope="module")
def resolved_config():
    return load_config(
        "grasp_configs/left_opposed_face_palm_down_contact_preserving_planned_lift.json"
    )


@pytest.fixture(scope="module")
def real_limits(resolved_config):
    return resolve_joint_refinement_limits(resolved_config)


def _parents(config, count=4):
    # Equal summaries deliberately fall through to stable candidate ID.
    return [
        {
            "candidate_id": 14_000_000_000_950_000 + index,
            "config": copy.deepcopy(config),
            "full_success": False,
        }
        for index in range(count)
    ]


@pytest.fixture(scope="module")
def generated_jobs(resolved_config, real_limits):
    return build_contact_preserving_time_warp_jobs(
        _parents(resolved_config),
        limits_resolver=lambda _config: real_limits,
        validate_configs=False,
    )


def test_gaussian_bases_are_zero_mean_and_have_declared_sign_response():
    bases = zero_mean_gaussian_bases()
    assert bases.shape == (2, 20)
    np.testing.assert_allclose(np.mean(bases, axis=1), 0.0, atol=1e-16)
    # +a1 lengthens the first diagnostic band and -a2 shortens the second.
    assert bases[0, 9] > 0.0
    assert bases[1, 13] > 0.0


def test_zero_warp_reproduces_parent_times_and_plan_id_exactly(
    resolved_config, real_limits
):
    parent_times = resolved_config["manipulation_plan"]["knot_times_s"]
    warped = warp_knot_times(parent_times, 0.0, 0.0)
    assert warped.tolist() == parent_times
    candidate = apply_time_warp_to_config(
        resolved_config,
        a1=0.0,
        a2=0.0,
        limits=real_limits,
    )
    assert candidate["manipulation_plan"]["knot_times_s"] == parent_times
    assert (
        candidate["manipulation_plan"]["plan_id"]
        == resolved_config["manipulation_plan"]["plan_id"]
    )


@pytest.mark.parametrize(
    "a1,a2",
    [(-0.45, -0.45), (-0.45, 0.45), (0.45, -0.45), (0.45, 0.45)],
)
def test_softmax_warp_is_positive_bounded_and_exactly_three_seconds(
    resolved_config, a1, a2
):
    times = warp_knot_times(
        resolved_config["manipulation_plan"]["knot_times_s"], a1, a2
    )
    durations = np.diff(times)
    assert times[0] == 0.0
    assert times[-1] == 3.0
    assert float(np.min(durations)) >= MIN_SEGMENT_DURATION_S - 1e-12
    assert float(np.max(durations)) <= MAX_SEGMENT_DURATION_S + 1e-12
    assert np.all(durations > 0.0)


def test_diagnostic_best_sign_slows_047_and_speeds_070_bands(resolved_config):
    parent = np.diff(resolved_config["manipulation_plan"]["knot_times_s"])
    warped = np.diff(
        warp_knot_times(
            resolved_config["manipulation_plan"]["knot_times_s"], 0.45, -0.45
        )
    )
    # This protects the sign convention proven by the 79-mm diagnostic:
    # a1=+.45/a2=-.45 reduced jerk while retaining all three contacts.
    assert float(np.mean(warped[8:11])) > float(np.mean(parent[8:11]))
    assert float(np.mean(warped[13:15])) < float(np.mean(parent[13:15]))


def test_warp_coefficient_and_parent_contract_fail_closed(resolved_config):
    times = resolved_config["manipulation_plan"]["knot_times_s"]
    with pytest.raises(ValueError, match="coefficients"):
        warp_knot_times(times, COEFFICIENT_BOUNDS[1] + 1e-6, 0.0)
    bad = list(times)
    bad[1] = 0.01
    with pytest.raises(ValueError, match="90--240"):
        warp_knot_times(bad, 0.0, 0.0)


def test_apply_recomputes_plan_and_controller_and_preserves_pair(
    resolved_config, real_limits
):
    candidate = apply_time_warp_to_config(
        resolved_config,
        a1=0.45,
        a2=-0.45,
        limits=real_limits,
    )
    assert (
        candidate["manipulation_plan"]["plan_id"]
        != resolved_config["manipulation_plan"]["plan_id"]
    )
    assert candidate["controller_id"]
    assert candidate["object_config_id"]
    assert candidate["grasp_pose_id"]
    assert candidate["grasp_object_pair_id"]
    for name in ACTIVE_ACTUATORS:
        assert (
            candidate["manipulation_plan"]["actuator_waypoints_rad"][name]
            == resolved_config["manipulation_plan"]["actuator_waypoints_rad"][name]
        )
    report = validate_time_warped_config(candidate, limits=real_limits)
    assert report["knot_count"] == 21


def test_terminal_and_preload_perturbations_are_isolated(
    resolved_config, real_limits
):
    terminal_offset = {name: 0.0 for name in ACTIVE_ACTUATORS}
    terminal_offset[ACTIVE_ACTUATORS[0]] = -0.005
    terminal = apply_time_warp_to_config(
        resolved_config,
        a1=0.1,
        a2=-0.1,
        terminal_offset_rad=terminal_offset,
        category="warp_terminal",
        limits=real_limits,
    )
    assert terminal["control"]["contact_preload_targets_rad"] == resolved_config[
        "control"
    ]["contact_preload_targets_rad"]
    assert terminal["control"]["manipulation_delta_rad"][ACTIVE_ACTUATORS[0]] == pytest.approx(
        resolved_config["control"]["manipulation_delta_rad"][ACTIVE_ACTUATORS[0]]
        - 0.005
    )

    preload_offset = {name: 0.0 for name in ACTIVE_ACTUATORS}
    preload_offset[ACTIVE_ACTUATORS[-1]] = 0.003
    preload = apply_time_warp_to_config(
        resolved_config,
        a1=-0.1,
        a2=0.1,
        preload_offset_rad=preload_offset,
        category="warp_preload",
        limits=real_limits,
    )
    assert preload["control"]["manipulation_delta_rad"] == resolved_config[
        "control"
    ]["manipulation_delta_rad"]
    assert preload["control"]["contact_preload_targets_rad"][ACTIVE_ACTUATORS[-1]] == pytest.approx(
        resolved_config["control"]["contact_preload_targets_rad"][ACTIVE_ACTUATORS[-1]]
        + 0.003
    )


def test_bezier_hull_is_continuous_limit_evidence(resolved_config):
    hull = actuator_bezier_hull(resolved_config)
    assert hull["controls_rad"].shape == (20, 6, len(ACTIVE_ACTUATORS))
    waypoint_values = np.stack(
        [
            resolved_config["manipulation_plan"]["actuator_waypoints_rad"][name]
            for name in ACTIVE_ACTUATORS
        ],
        axis=1,
    )
    assert np.all(hull["relative_lower_rad"] <= np.min(waypoint_values, axis=0) + 1e-12)
    assert np.all(hull["relative_upper_rad"] >= np.max(waypoint_values, axis=0) - 1e-12)


def test_default_builder_is_exact_256_with_declared_four_parent_split(generated_jobs):
    assert len(generated_jobs) == DEFAULT_TOTAL_CANDIDATES == 256
    assert Counter(job["job_metadata"]["category"] for job in generated_jobs) == {
        "warp_only": 128,
        "warp_terminal": 64,
        "warp_preload": 64,
    }
    assert Counter(job["parent_candidate_id"] for job in generated_jobs) == {
        14_000_000_000_950_000 + index: 64 for index in range(4)
    }
    assert [job["job_sequence_index"] for job in generated_jobs] == list(range(256))


def test_boundary_parent_falls_back_per_component_and_still_builds_256_unique(
    resolved_config, real_limits
):
    """A terminal direction with zero knot headroom must not abort the batch.

    The selected actuator has exactly 20 x 0.04-rad waypoint increments.  Any
    positive terminal perturbation violates ``max_knot_delta_rad`` at every
    nonzero scale, while the independently requested time warp remains safe.
    This models the formal rescue parent which previously stopped generation
    after all 17 joint-backoff attempts.
    """

    boundary = copy.deepcopy(resolved_config)
    old_plan = ManipulationPlanParameters.from_config(boundary["manipulation_plan"])
    boundary_name = "left_hand_thumb_rota_joint2_actuator"
    terminal = 0.8
    waypoints = {
        name: tuple(values)
        for name, values in old_plan.actuator_waypoints_rad.items()
    }
    waypoints[boundary_name] = tuple(np.linspace(0.0, terminal, 21))
    boundary_plan = ManipulationPlanParameters(
        schema_version=old_plan.schema_version,
        profile=old_plan.profile,
        duration_s=old_plan.duration_s,
        knot_times_s=old_plan.knot_times_s,
        actuator_waypoints_rad=waypoints,
        desired_cube_position_delta_m=old_plan.desired_cube_position_delta_m,
        desired_cube_rotation_vector_rad=old_plan.desired_cube_rotation_vector_rad,
        max_knot_delta_rad=old_plan.max_knot_delta_rad,
        trust_region_backtracks=old_plan.trust_region_backtracks,
    )
    boundary["manipulation_plan"] = boundary_plan.as_config()
    boundary["control"]["manipulation_delta_rad"][boundary_name] = terminal

    jobs = build_contact_preserving_time_warp_jobs(
        _parents(boundary, count=1),
        limits_resolver=lambda _config: real_limits,
        validate_configs=False,
    )
    assert len(jobs) == 256
    assert len({job["candidate_id"] for job in jobs}) == 256
    assert len({job["candidate_sha256"] for job in jobs}) == 256
    assert len({job["config_sha256"] for job in jobs}) == 256

    fallbacks = [
        job
        for job in jobs
        if job["job_metadata"]["fallback_reason"]
        == "terminal_offset_zero_no_safe_headroom"
    ]
    assert fallbacks
    for job in fallbacks:
        metadata = job["job_metadata"]
        assert metadata["category"] == "warp_terminal"
        assert metadata["applied_scales"]["coefficient"] > 0.0
        assert metadata["applied_scales"]["terminal"] == 0.0
        assert metadata["applied_scales"]["preload"] == 0.0
        assert metadata["applied_backoff_scale"] is None
        assert any(abs(value) > 0.0 for value in metadata["applied_coefficients"])
        assert all(value == 0.0 for value in metadata["terminal_offset_rad"].values())
        assert all(value == 0.0 for value in metadata["preload_offset_rad"].values())
        assert (
            job["config"]["control"]["manipulation_delta_rad"]
            == boundary["control"]["manipulation_delta_rad"]
        )
        authenticate_time_warp_job(job)


def test_fewer_parents_keep_256_and_publish_explicit_allocation(resolved_config):
    manifest = time_warp_campaign_manifest(_parents(resolved_config, count=3))
    assert manifest["selected_parent_count"] == 3
    assert sum(parent["total"] for parent in manifest["parents"]) == 256
    category_totals = Counter()
    for parent in manifest["parents"]:
        category_totals.update(parent["allocation"])
    assert category_totals == {"warp_only": 128, "warp_terminal": 64, "warp_preload": 64}
    assert manifest["fewer_parent_allocation"]


def test_generation_and_sort_are_worker_order_independent(
    resolved_config, real_limits, generated_jobs
):
    shuffled_parents = _parents(resolved_config)
    random.Random(3117).shuffle(shuffled_parents)
    repeated = build_contact_preserving_time_warp_jobs(
        shuffled_parents,
        limits_resolver=lambda _config: real_limits,
        validate_configs=False,
    )
    signature = lambda jobs: [
        (job["candidate_id"], job["candidate_sha256"], job["config_sha256"])
        for job in jobs
    ]
    assert signature(repeated) == signature(generated_jobs)
    completion = list(copy.deepcopy(generated_jobs))
    random.Random(7229).shuffle(completion)
    assert signature(stable_sort_time_warp_jobs(completion)) == signature(generated_jobs)


def test_authentication_detects_config_and_summary_tampering(generated_jobs):
    job = copy.deepcopy(generated_jobs[7])
    authenticated = authenticate_time_warp_job(job)
    assert authenticated["candidate_id"] == job["candidate_id"]

    changed = copy.deepcopy(job)
    changed["config"]["manipulation_plan"]["knot_times_s"][1] += 1e-6
    with pytest.raises(ValueError):
        authenticate_time_warp_job(changed)

    changed = copy.deepcopy(job)
    changed["job_metadata"]["applied_coefficients"][0] += 1e-6
    with pytest.raises(ValueError):
        authenticate_time_warp_job(changed)


def test_budget_fails_closed_if_not_exactly_256():
    with pytest.raises(ValueError, match="exactly 256"):
        TimeWarpBudget(total_candidate_count=255)
