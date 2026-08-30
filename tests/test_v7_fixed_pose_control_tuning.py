from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from xhand_grasp.config import load_config, validate_config
from xhand_grasp.tuning.fixed_pose_control import (
    DEFAULT_BASE_CONFIG,
    DEFAULT_CONTROL_CENTER,
    assert_control_only_candidate,
    candidate_rank,
    control_vector,
    frozen_context_sha256,
    generate_stage1_candidates,
    generate_stage2_candidates,
    materialize_control_candidate,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def base() -> dict:
    return load_config(ROOT / DEFAULT_BASE_CONFIG)


def test_materializer_changes_only_control_and_metadata(base):
    candidate = materialize_control_candidate(
        base,
        DEFAULT_CONTROL_CENTER,
        candidate_id=77,
        stage="test",
    )
    validate_config(candidate)
    assert_control_only_candidate(base, candidate)
    assert frozen_context_sha256(candidate) == frozen_context_sha256(base)
    for name, value in base.items():
        if name not in {"control", "candidate_metadata", "run_context"}:
            assert candidate[name] == value
    assert candidate["control"]["manipulation_delta_rad"] == base["control"][
        "manipulation_delta_rad"
    ]
    assert candidate["control"]["grasp_targets_rad"][
        "left_hand_thumb_bend_joint_actuator"
    ] == base["control"]["grasp_targets_rad"][
        "left_hand_thumb_bend_joint_actuator"
    ]


def test_control_only_guard_rejects_pose_or_cube_changes(base):
    candidate = materialize_control_candidate(
        base, DEFAULT_CONTROL_CENTER, candidate_id=78, stage="test"
    )
    moved = copy.deepcopy(candidate)
    moved["hand_pose"]["translation_m"][0] += 1e-6
    with pytest.raises(ValueError, match="frozen experiment context"):
        assert_control_only_candidate(base, moved)
    resized = copy.deepcopy(candidate)
    resized["cube"]["edge_m"] += 0.001
    with pytest.raises(ValueError, match="frozen experiment context"):
        assert_control_only_candidate(base, resized)


def test_stage1_generation_is_deterministic_and_contains_anchor(base):
    first = generate_stage1_candidates(base, count=24, seed=20260821)
    second = generate_stage1_candidates(base, count=24, seed=20260821)
    assert first == second
    np.testing.assert_allclose(control_vector(first[0]), DEFAULT_CONTROL_CENTER)
    assert len({item["candidate_metadata"]["candidate_id"] for item in first}) == 24
    assert all(frozen_context_sha256(item) == frozen_context_sha256(base) for item in first)


def test_stage2_keeps_ranked_parents_and_is_deterministic(base):
    configs = generate_stage1_candidates(base, count=4, seed=3)
    ranked = tuple(
        {
            "candidate_id": config["candidate_metadata"]["candidate_id"],
            "config": config,
            "acquisition_success": False,
            "pose_preservation_success": False,
            "verify_gate_steps": 10 - index,
            "translation_max_m": 0.0004,
            "orientation_max_deg": 0.8,
            "opposed_normal_force_imbalance_n": 0.1,
            "contact_onset_span_steps": 10,
            "actuator_saturation_fraction": 0.1,
        }
        for index, config in enumerate(configs)
    )
    first = generate_stage2_candidates(base, ranked, count=8, seed_count=2, seed=9)
    second = generate_stage2_candidates(base, ranked, count=8, seed_count=2, seed=9)
    assert first == second
    np.testing.assert_allclose(control_vector(first[0]), control_vector(configs[0]))
    np.testing.assert_allclose(control_vector(first[1]), control_vector(configs[1]))
    assert all(frozen_context_sha256(item) == frozen_context_sha256(base) for item in first)


def test_rank_prioritizes_hard_pass_then_gate_window():
    base_record = {
        "candidate_id": 2,
        "acquisition_success": False,
        "pose_preservation_success": False,
        "verify_gate_steps": 249,
        "translation_max_m": 0.00049,
        "orientation_max_deg": 0.99,
        "opposed_normal_force_imbalance_n": 0.0,
        "contact_onset_span_steps": 0,
        "actuator_saturation_fraction": 0.0,
    }
    passing = {**base_record, "candidate_id": 3, "acquisition_success": True, "pose_preservation_success": True, "verify_gate_steps": 250}
    shorter = {**base_record, "candidate_id": 1, "verify_gate_steps": 200}
    assert candidate_rank(passing) < candidate_rank(base_record) < candidate_rank(shorter)
