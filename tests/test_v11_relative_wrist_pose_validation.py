from __future__ import annotations

import copy
from pathlib import Path

import pytest

from xhand_grasp.config import load_config, validate_config
from xhand_grasp.tuning.relative_wrist_pose_validation import (
    CONSTANT_DENSITY_VALIDATION_LABELS,
    FIXED_160G_VALIDATION_LABELS,
    PERTURBATION_COUNT,
    REQUIRED_PERTURBATION_PASSES,
    apply_pose_friction_perturbation_case,
    classify_pose_friction_robustness,
    constant_density_mass_kg,
    density_revalidation_cases,
    generate_pose_friction_perturbation_cases,
    validation_label,
)


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "grasp_configs" / (
    "left_opposed_face_palm_down_larger_relative_wrist_pose_actual_contact_"
    "smooth_vertical_lift.json"
)


def test_equal_density_mass_covers_every_registered_85_to_104_mm_edge():
    cases = density_revalidation_cases()

    assert len(cases) == 20
    assert [case.edge_mm for case in cases] == pytest.approx(range(85, 105))
    assert cases[0].mass_kg == pytest.approx(
        740.7407407407408 * 0.085**3
    )
    assert cases[-1].mass_kg == pytest.approx(
        740.7407407407408 * 0.104**3
    )
    assert all(
        case.mass_kg < following.mass_kg
        for case, following in zip(cases, cases[1:])
    )
    assert cases[0].as_dict()["family"] == "constant_density"
    with pytest.raises(ValueError, match="registered"):
        constant_density_mass_kg(0.084)
    with pytest.raises(ValueError, match="registered"):
        constant_density_mass_kg(0.105)


def test_fixed_and_equal_density_labels_cannot_be_conflated():
    assert FIXED_160G_VALIDATION_LABELS == {
        "grasp": "validated_fixed_160g_grasp_ablation",
        "manipulation": "validated_fixed_160g_manipulation_ablation",
        "robust": "validated_fixed_160g_robust_full_success_ablation",
    }
    assert CONSTANT_DENSITY_VALIDATION_LABELS == {
        "grasp": "validated_constant_density_grasp",
        "manipulation": "validated_constant_density_manipulation",
        "robust": "validated_constant_density_robust_full_success",
    }
    assert validation_label("fixed_160g", "grasp").endswith("_ablation")
    assert validation_label("constant_density", "grasp") == (
        "validated_constant_density_grasp"
    )
    with pytest.raises(ValueError, match="family"):
        validation_label("same_material", "grasp")  # type: ignore[arg-type]


def test_pose_friction_cases_are_deterministic_complete_and_in_bounds():
    first = generate_pose_friction_perturbation_cases(seed=20260821)
    second = generate_pose_friction_perturbation_cases(seed=20260821)
    different = generate_pose_friction_perturbation_cases(seed=20260822)

    assert first == second
    assert first != different
    assert len(first) == PERTURBATION_COUNT == 50
    assert [case.trial for case in first] == list(range(50))
    for case in first:
        assert all(abs(value) <= 0.0015 for value in case.cube_center_xy_delta_m)
        assert 0.0 <= case.cube_gap_delta_m <= 0.0005
        assert all(abs(value) <= 3.0 for value in case.cube_rpy_delta_deg)
        assert abs(case.friction_delta) <= 0.1
        assert case.resolved_perturbations()["mass_scale"] == 1.0
        assert case.full_reset_rerun is True
        assert case.initial_state_source == "configured_no_contact_reset"
        assert case.checkpoint_used is False
    with pytest.raises(ValueError, match="non-negative integer"):
        generate_pose_friction_perturbation_cases(seed=-1)


def test_pose_friction_case_materialization_preserves_fixed_mass_and_source():
    config = load_config(TEMPLATE)
    before = copy.deepcopy(config)
    case = generate_pose_friction_perturbation_cases()[7]

    trial = apply_pose_friction_perturbation_case(
        config, case, source_candidate_id="candidate-7"
    )

    assert config == before
    assert trial["cube"]["mass_kg"] == pytest.approx(0.160)
    assert trial["run_context"] == {"kind": "robustness_trial"}
    metadata = trial["candidate_metadata"]["robustness_trial"]
    assert metadata["trial"] == 7
    assert metadata["source_candidate_id"] == "candidate-7"
    assert metadata["resolved_perturbations"]["mass_scale"] == 1.0
    assert metadata["full_reset_rerun"] is True
    assert metadata["checkpoint_used"] is False
    validate_config(trial)

    prior_trial = copy.deepcopy(config)
    prior_trial["run_context"] = {"kind": "robustness_trial"}
    with pytest.raises(ValueError, match="canonical nominal"):
        apply_pose_friction_perturbation_case(
            prior_trial, case, source_candidate_id="recursive"
        )


@pytest.mark.parametrize(
    ("passes", "nominal", "expected_pass", "reason"),
    [
        (45, True, True, "robust_passed"),
        (44, True, False, "perturbation_pass_count_below_45_of_50"),
        (50, False, False, "nominal_failed_full_success"),
    ],
)
def test_robustness_classification_applies_exact_45_of_50_gate(
    passes, nominal, expected_pass, reason
):
    flags = [True] * passes + [False] * (50 - passes)

    result = classify_pose_friction_robustness(
        flags, nominal_full_success=nominal, family="fixed_160g"
    )

    assert result.perturbation_count == 50
    assert result.perturbation_passes == passes
    assert result.required_perturbation_passes == REQUIRED_PERTURBATION_PASSES
    assert result.registered_budget_complete is True
    assert result.robust_passed is expected_pass
    assert result.stop_reason == reason
    assert result.validation_label == (
        "validated_fixed_160g_robust_full_success_ablation"
        if expected_pass
        else None
    )


def test_incomplete_budget_and_constant_density_classification_are_explicit():
    incomplete = classify_pose_friction_robustness(
        [True] * 49,
        nominal_full_success=True,
    )
    equal_density = classify_pose_friction_robustness(
        [True] * 45 + [False] * 5,
        nominal_full_success=True,
        family="constant_density",
    )

    assert incomplete.registered_budget_complete is False
    assert incomplete.robust_passed is False
    assert incomplete.stop_reason == "registered_perturbation_budget_incomplete"
    assert equal_density.robust_passed is True
    assert equal_density.validation_label == (
        "validated_constant_density_robust_full_success"
    )
    with pytest.raises(ValueError, match="boolean"):
        classify_pose_friction_robustness(
            [True] * 49 + [1],  # type: ignore[list-item]
            nominal_full_success=True,
        )
