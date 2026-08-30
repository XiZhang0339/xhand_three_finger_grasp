from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from xhand_grasp.config import ACTIVE_ACTUATORS
from xhand_grasp.experiment import ContactForceTargets, ManipulationPlanParameters
from xhand_grasp.grasp_pose import canonical_sha256
from xhand_grasp.tuning import contact_preserving_force_debias_rescue as debias


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "artifacts/left_opposed_face_palm_down_contact_preserving_planned_lift"
    / "tune/formal_campaign_v14_1_adaptive_event_rescue_v2"
)


@pytest.fixture(scope="module")
def source():
    return debias.authenticate_force_debias_source(SOURCE)


@pytest.fixture(scope="module")
def discovery_jobs(source):
    return debias.build_force_debias_jobs(
        source,
        stage="discovery",
        total_count=160,
        seed=debias.DEFAULT_SEED,
    )


def test_broad_force_debias_envelope_is_c2_with_declared_support():
    progress = np.linspace(0.0, 1.0, 10001)
    values = debias.broad_c2_force_debias_envelope(progress)
    assert np.all(values[(progress <= 0.20)] == 0.0)
    assert np.all(values[(progress >= 0.40) & (progress <= 0.85)] == 1.0)
    assert values[-1] == 0.0
    assert np.min(values) == 0.0
    assert np.max(values) == 1.0

    # Quintic minimum jerk has zero first and second derivative at every
    # junction; symmetric finite differences make this a numerical contract.
    step = progress[1] - progress[0]
    for boundary in (0.20, 0.40, 0.85, 1.0):
        index = int(round(boundary / step))
        if 1 <= index < len(values) - 1:
            first = (values[index + 1] - values[index - 1]) / (2.0 * step)
            second = (values[index + 1] - 2.0 * values[index] + values[index - 1]) / step**2
            assert abs(first) < 2e-4
            assert abs(second) < 0.40


def test_formal_adaptive_top_five_are_hash_authenticated(source):
    assert source.source_authentication_id == (
        "01c274b61da7ea2ce8a3cd4f8ad2ac36d48503678350a2815e7804010be4745c"
    )
    assert [center.candidate_id for center in source.centers] == [
        14883962737215073,
        14877912770329188,
        14816247784204292,
        14894512474483578,
        14831445898922445,
    ]
    assert len(source.artifact_paths) == 43
    assert all(path.is_file() for path in source.artifact_paths)
    assert all(center.trace_path.is_file() for center in source.centers)


def test_wide_event_polytopes_keep_registered_plan_boundaries(source):
    center = source.centers[0]
    descriptor = debias.build_force_debias_descriptor(center)
    for width in debias.EVENT_HALF_WIDTHS_PROGRESS:
        polytope = debias.derive_force_debias_polytope(
            center.config,
            descriptor,
            event_half_width_progress=width,
        )
        assert polytope.bump_half_width_progress == width
        assert tuple(polytope.parameter_names[:3]) == (
            "broad_unload:thumb",
            "broad_unload:index",
            "broad_unload:mid",
        )
        assert tuple(polytope.upper_bounds[:3]) == pytest.approx((0.035,) * 3)
        projected = polytope.project(np.asarray(polytope.upper_bounds))
        assert polytope.contains(projected, tolerance=1e-10)
        assert np.all(projected >= np.asarray(polytope.lower_bounds) - 1e-12)
        assert np.all(projected <= np.asarray(polytope.upper_bounds) + 1e-12)


def test_discovery_is_deterministic_unique_balanced_and_covers_feedback_bounds(
    source, discovery_jobs
):
    repeated = debias.build_force_debias_jobs(
        source,
        stage="discovery",
        total_count=160,
        seed=debias.DEFAULT_SEED,
    )
    assert discovery_jobs == repeated
    assert len({job["candidate_id"] for job in discovery_jobs}) == 160
    assert len({job["physical_plan_sha256"] for job in discovery_jobs}) == 160
    for center in source.centers:
        jobs = [
            job for job in discovery_jobs if job["source_center_id"] == center.center_id
        ]
        assert len(jobs) == 32
        scales = [job["feedback_parameters"]["operation_scale"] for job in jobs]
        assert min(scales) == 0.70
        assert max(scales) == 1.00
    assert {job["event_half_width_progress"] for job in discovery_jobs} == {
        0.18,
        0.22,
        0.26,
    }
    assert {job["feedback_parameters"]["filter_time_constant_s"] for job in discovery_jobs} == {
        0.005,
        0.008,
        0.012,
    }
    for key, bounds in (
        ("kp_multiplier", (1.0, 3.0)),
        ("ki_multiplier", (1.0, 2.0)),
        ("integral_limit_n_s", (0.5, 1.25)),
        ("correction_limit_rad", (0.06, 0.09)),
        ("operation_scale", (0.70, 1.00)),
    ):
        values = [job["feedback_parameters"][key] for job in discovery_jobs]
        assert min(values) >= bounds[0]
        assert max(values) <= bounds[1]


def test_scale_identity_and_plan_limits_are_bound_to_each_job(source, discovery_jobs):
    legacy_target_ids = {
        center.center_id: center.config["contact_force_targets_n"]["target_id"]
        for center in source.centers
    }
    for job in discovery_jobs:
        config = job["config"]
        parameters = job["feedback_parameters"]
        targets = ContactForceTargets.from_config(config["contact_force_targets_n"])
        assert targets.operation_scale == pytest.approx(parameters["operation_scale"])
        if parameters["operation_scale"] == 1.0:
            assert "operation_scale" not in config["contact_force_targets_n"]
            assert targets.target_id == legacy_target_ids[job["source_center_id"]]
        else:
            assert config["contact_force_targets_n"]["operation_scale"] == pytest.approx(
                parameters["operation_scale"]
            )
            assert targets.target_id != legacy_target_ids[job["source_center_id"]]
        plan = ManipulationPlanParameters.from_config(config["manipulation_plan"])
        assert max(
            abs(following - previous)
            for name in ACTIVE_ACTUATORS
            for previous, following in zip(
                plan.actuator_waypoints_rad[name],
                plan.actuator_waypoints_rad[name][1:],
            )
        ) <= plan.max_knot_delta_rad + 1e-12


def test_discovery_anchor_identity_and_small_radius_shells(source, discovery_jobs):
    expected_anchors = {
        "near_identity_thumb",
        "near_identity_index",
        "near_identity_mid",
        "operation_scale_0p70",
        "operation_scale_0p85",
        "kp_multiplier_2p0",
        "ki_multiplier_1p5",
        "integral_limit_1p0",
        "correction_limit_0p075",
        "filter_tau_8ms",
        "all_finger_broad_0p001",
        "all_finger_broad_0p002",
    }
    for center in source.centers:
        jobs = [
            job for job in discovery_jobs if job["source_center_id"] == center.center_id
        ]
        anchors = [job for job in jobs if job["sampling_mode"] == "structured_anchor"]
        small = [job for job in jobs if job["sampling_mode"] == "small_radius_lhs"]
        assert len(anchors) == 12
        assert len(small) == 20
        assert {job["structured_anchor"] for job in anchors} == expected_anchors
        assert [sum(job["normalized_radius"] == radius for job in small) for radius in (0.05, 0.10, 0.20)] == [7, 7, 6]
        for job in jobs:
            sample_index = job["lhs_pool_row_index"]
            assert job["event_half_width_progress"] == debias.EVENT_HALF_WIDTHS_PROGRESS[
                sample_index % 3
            ]

        near = sorted(
            (job for job in anchors if job["structured_anchor"].startswith("near_identity")),
            key=lambda value: value["structured_anchor"],
        )
        assert all(
            sum(
                abs(value)
                for key, value in job["projected_parameters"].items()
                if not key.startswith("feedback:")
            )
            == pytest.approx(1e-6, abs=1e-14)
            for job in near
        )
        for job in small:
            radius = job["normalized_radius"]
            for key, value in job["projected_parameters"].items():
                if key.startswith("broad_unload:"):
                    assert 0.0 <= value <= radius * 0.035 + 1e-12
                elif key.startswith("event:"):
                    bound = 0.015 if key.endswith(":tangent") else 0.006
                    assert abs(value) <= radius * bound + 1e-12
            feedback = job["feedback_parameters"]
            assert 1.0 <= feedback["kp_multiplier"] <= 1.0 + 2.0 * radius
            assert 1.0 <= feedback["ki_multiplier"] <= 1.0 + radius
            assert 0.5 <= feedback["integral_limit_n_s"] <= 0.5 + 0.75 * radius
            assert 0.06 <= feedback["correction_limit_rad"] <= 0.06 + 0.03 * radius
            assert 1.0 - 0.30 * radius <= feedback["operation_scale"] <= 1.0


def test_job_authentication_rejects_payload_id_and_physical_tampering(
    source, discovery_jobs
):
    job = discovery_jobs[0]
    authenticated = debias.authenticate_force_debias_job(
        job,
        source,
        expected_stage="discovery",
        expected_excluded_physical_plan_sha256=(),
    )
    assert authenticated == job

    tampered = copy.deepcopy(job)
    tampered["feedback_parameters"]["operation_scale"] = 0.95
    with pytest.raises(RuntimeError, match="payload SHA-256"):
        debias.authenticate_force_debias_job(tampered, source)

    rebound = copy.deepcopy(job)
    rebound["candidate_id"] += 1
    core = {
        key: value
        for key, value in rebound.items()
        if key not in {"config", "candidate_payload_sha256"}
    }
    rebound["candidate_payload_sha256"] = canonical_sha256(core)
    with pytest.raises(RuntimeError, match="candidate ID"):
        debias.authenticate_force_debias_job(rebound, source)

    config_changed = copy.deepcopy(job)
    config_changed["config"]["planner_id"] = "0" * 64
    with pytest.raises(RuntimeError, match="config is not reproducible"):
        debias.authenticate_force_debias_job(config_changed, source)


def test_refinement_uses_anchor_and_global_physical_exclusion(source, discovery_jobs):
    excluded = tuple(job["physical_plan_sha256"] for job in discovery_jobs)
    refined = debias.build_force_debias_jobs(
        source,
        stage="refinement",
        total_count=64,
        seed=debias.DEFAULT_SEED,
        local_index_offset=160,
        refinement_records=discovery_jobs[:8],
        excluded_physical_plan_sha256=excluded,
    )
    assert len(refined) == 64
    assert len({job["physical_plan_sha256"] for job in refined}) == 64
    assert not set(excluded).intersection(
        job["physical_plan_sha256"] for job in refined
    )
    debias.authenticate_force_debias_job(
        refined[0],
        source,
        expected_stage="refinement",
        expected_excluded_physical_plan_sha256=excluded,
    )
