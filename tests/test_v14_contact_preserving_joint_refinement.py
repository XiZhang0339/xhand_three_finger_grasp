from __future__ import annotations

import copy
import random
from collections import Counter

import numpy as np
import pytest

from xhand_grasp.config import ACTIVE_ACTUATORS, load_config
from xhand_grasp.grasp_pose import canonical_sha256
from xhand_grasp.tuning.contact_preserving_joint_refinement import (
    CANDIDATES_PER_PARENT,
    GLOBAL_PARENT_COUNT,
    JointRefinementBudget,
    build_joint_refinement_job_specs,
    resolve_joint_refinement_limits,
    stable_sort_joint_refinement_jobs,
)


@pytest.fixture(scope="module")
def resolved_config():
    return load_config(
        "grasp_configs/left_opposed_face_palm_down_contact_preserving_planned_lift.json"
    )


@pytest.fixture(scope="module")
def real_limits(resolved_config):
    return resolve_joint_refinement_limits(resolved_config)


def _parents(config):
    # Contact-first ranking falls through to the stable candidate ID for these
    # deliberately equal synthetic result summaries.
    return [
        {
            "candidate_id": 14_000_000_000_001_000 + index,
            "config": copy.deepcopy(config),
            "full_success": False,
        }
        for index in range(GLOBAL_PARENT_COUNT)
    ]


def _signature(jobs):
    return tuple(
        (
            value["candidate_id"],
            value["candidate_sha256"],
            value["parent_candidate_id"],
            value["parent_rank"],
            value["local_index"],
            canonical_sha256(value["config"]),
        )
        for value in jobs
    )


def test_v14_global_joint_refinement_budget_is_exactly_eight_by_128(
    resolved_config, real_limits
):
    jobs = build_joint_refinement_job_specs(
        _parents(resolved_config),
        limits_resolver=lambda _config: real_limits,
        validate_configs=False,
    )
    assert len(jobs) == GLOBAL_PARENT_COUNT * CANDIDATES_PER_PARENT == 1024
    assert Counter(value["parent_candidate_id"] for value in jobs) == {
        14_000_000_000_001_000 + index: CANDIDATES_PER_PARENT
        for index in range(GLOBAL_PARENT_COUNT)
    }
    assert Counter(value["local_index"] for value in jobs) == {
        index: GLOBAL_PARENT_COUNT for index in range(CANDIDATES_PER_PARENT)
    }
    assert [value["job_sequence_index"] for value in jobs] == list(range(1024))


def test_generation_and_result_order_do_not_depend_on_worker_order(
    resolved_config, real_limits
):
    budget = JointRefinementBudget(candidates_per_parent=5)
    parents = _parents(resolved_config)
    forward = build_joint_refinement_job_specs(
        parents,
        budget=budget,
        limits_resolver=lambda _config: real_limits,
        validate_configs=False,
    )
    shuffled_parents = copy.deepcopy(parents)
    random.Random(8871).shuffle(shuffled_parents)
    shuffled = build_joint_refinement_job_specs(
        shuffled_parents,
        budget=budget,
        limits_resolver=lambda _config: real_limits,
        validate_configs=False,
    )
    assert _signature(shuffled) == _signature(forward)

    completion_order = list(copy.deepcopy(forward))
    random.Random(9903).shuffle(completion_order)
    assert _signature(stable_sort_joint_refinement_jobs(completion_order)) == _signature(
        forward
    )


def test_refinement_respects_real_preload_plan_and_node_bounds(
    resolved_config, real_limits
):
    budget = JointRefinementBudget(candidates_per_parent=8)
    jobs = build_joint_refinement_job_specs(
        _parents(resolved_config),
        budget=budget,
        limits_resolver=lambda _config: real_limits,
    )
    for job in jobs:
        config = job["config"]
        preload = config["control"]["contact_preload_targets_rad"]
        terminal = config["control"]["manipulation_delta_rad"]
        waypoints = config["manipulation_plan"]["actuator_waypoints_rad"]
        maximum_node_delta = float(config["manipulation_plan"]["max_knot_delta_rad"])
        for name in ACTIVE_ACTUATORS:
            command_lower, command_upper = real_limits.command_target_rad[name]
            preload_lower, preload_upper = real_limits.preload_target_rad[name]
            registered_lower, registered_upper = (
                real_limits.registered_plan_delta_rad[name]
            )
            assert command_lower - 1e-12 <= preload[name] <= command_upper + 1e-12
            assert preload_lower - 1e-12 <= preload[name] <= preload_upper + 1e-12
            assert registered_lower - 1e-12 <= terminal[name] <= registered_upper + 1e-12
            assert (
                command_lower - 1e-12
                <= preload[name] + terminal[name]
                <= command_upper + 1e-12
            )
            values = np.asarray(waypoints[name], dtype=np.float64)
            assert values[0] == pytest.approx(0.0, abs=1e-14)
            assert values[-1] == pytest.approx(terminal[name], abs=1e-12)
            assert np.max(np.abs(np.diff(values))) <= maximum_node_delta + 1e-12


def test_candidate_identity_is_stable_and_seed_bound(resolved_config, real_limits):
    parents = _parents(resolved_config)
    budget = JointRefinementBudget(candidates_per_parent=4, seed=20260821)
    first = build_joint_refinement_job_specs(
        parents,
        budget=budget,
        limits_resolver=lambda _config: real_limits,
        validate_configs=False,
    )
    repeated = build_joint_refinement_job_specs(
        copy.deepcopy(parents),
        budget=budget,
        limits_resolver=lambda _config: real_limits,
        validate_configs=False,
    )
    assert _signature(first) == _signature(repeated)
    assert len({value["candidate_id"] for value in first}) == len(first)
    assert all(
        value["candidate_sha256"]
        == value["config"]["candidate_metadata"]["v14_joint_local_refinement"][
            "candidate_sha256"
        ]
        for value in first
    )

    different_seed = build_joint_refinement_job_specs(
        parents,
        budget=JointRefinementBudget(candidates_per_parent=4, seed=20260822),
        limits_resolver=lambda _config: real_limits,
        validate_configs=False,
    )
    assert [value["candidate_id"] for value in different_seed] != [
        value["candidate_id"] for value in first
    ]
