from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from xhand_grasp.aligned_contacts_search import (
    aligned_contact_candidate_rank,
    budget_manifest,
    deterministic_rank_results,
    select_per_band,
    static_screen_jobs,
    tune_aligned_contacts,
)
from xhand_grasp.config import (
    load_config,
    resolved_pose_constraint_values,
    validate_config,
)
from xhand_grasp.experiment import get_experiment, resolve_experiment
from xhand_grasp.experiments.opposed_face_palm_down import SEARCH_FACE_ASSIGNMENTS
from xhand_grasp.experiments.opposed_face_palm_tilted_down_aligned_contacts_grasp_then_lift import (
    ALIGNED_CONTACT_CAMPAIGN,
    CONTACT_ALIGNMENT,
    EXPERIMENT_DEFINITION,
    EXPERIMENT_ID,
    POSE_CONSTRAINTS,
    REFERENCE_HAND_TRANSLATION_M,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_tilted_down_aligned_contacts_grasp_then_lift.json"
)
SOURCE_CONFIG_PATH = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_larger_cube_relative_pose_rescue_validated.json"
)


def _raw_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _result(
    candidate_id: int,
    band: float,
    *,
    passed: bool = True,
    margin: float = 0.2,
    aligned_duty: float = 0.8,
    p95_spread: float = 0.003,
    topology: float = 0.75,
    force: float = 3.0,
    drift: float = 2.0,
    saturation: float = 0.1,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "tilt_band_center_deg": band,
        "summary": {
            "passed": passed,
            "stage_status": {"full_success": passed},
            "metrics": {
                "minimum_normalized_margin": margin,
                "operation_aligned_contact_duty": aligned_duty,
                "operation_contact_height_spread_p95_m": p95_spread,
                "operation_target_face_simultaneous_duty": topology,
                "peak_total_distal_contact_force_n": force,
                "orientation_drift_deg": drift,
                "actuator_saturation_fraction": saturation,
            },
        },
    }


def test_v4_definition_is_registered_and_template_is_strictly_resolved():
    config = load_config(CONFIG_PATH)
    source = load_config(SOURCE_CONFIG_PATH)

    assert config["schema_version"] == 4
    assert resolve_experiment(config) is EXPERIMENT_DEFINITION
    assert get_experiment(EXPERIMENT_ID) is EXPERIMENT_DEFINITION
    assert EXPERIMENT_DEFINITION.tuning_strategy == "aligned_contacts"
    assert EXPERIMENT_DEFINITION.candidate_faces == SEARCH_FACE_ASSIGNMENTS
    assert config["pose_constraints"] == POSE_CONSTRAINTS.as_config()
    assert config["contact_alignment"] == CONTACT_ALIGNMENT.as_config()
    assert config["aligned_contact_campaign"] == (
        ALIGNED_CONTACT_CAMPAIGN.as_config()
    )
    assert tuple(config["pose_constraints"]["reference_hand_translation_m"]) == (
        pytest.approx(source["hand_pose"]["translation_m"])
    )
    assert tuple(config["pose_constraints"]["reference_hand_translation_m"]) == (
        pytest.approx(REFERENCE_HAND_TRANSLATION_M)
    )
    assert resolved_pose_constraint_values(config) == pytest.approx(
        {
            "finger_down_tilt_deg": 15.0,
            "palm_plane_ground_angle_deg": 15.0,
            "palm_press_depth_m": 0.005,
            "cube_position_in_root_m": (
                0.05138553622355406,
                -0.02886063167725714,
                0.10872576454433164,
            ),
        }
    )
    assert config["cube"] == source["cube"]
    assert config["control"] == source["control"]
    assert config["experiment_status"]["full_success"] is False


def test_v4_pose_material_domain_and_budgets_are_exact():
    bounds = EXPERIMENT_DEFINITION.search_bounds
    campaign = ALIGNED_CONTACT_CAMPAIGN

    assert POSE_CONSTRAINTS.finger_down_tilt_deg == (10.0, 20.0)
    assert POSE_CONSTRAINTS.palm_plane_ground_angle_deg == (10.0, 20.0)
    assert POSE_CONSTRAINTS.palm_press_depth_m == (0.002, 0.008)
    assert CONTACT_ALIGNMENT.max_height_spread_m == pytest.approx(0.005)
    assert CONTACT_ALIGNMENT.verify_continuous_s == pytest.approx(0.25)
    assert CONTACT_ALIGNMENT.operation_aligned_duty == pytest.approx(0.70)
    assert campaign.edges_m == pytest.approx(
        (0.059, 0.060, 0.061, 0.062, 0.063, 0.064)
    )
    assert campaign.density_kg_m3 == pytest.approx(740.7407407407407)
    assert campaign.friction == pytest.approx(0.8)
    assert campaign.tilt_band_centers_deg == (
        10.0,
        12.5,
        15.0,
        17.5,
        20.0,
    )
    assert bounds.hand_roll_deg == (-5.0, 5.0)
    assert bounds.hand_yaw_deg == (-8.0, 2.0)
    assert bounds.cube_position_in_root_m["y"] == (-0.036, -0.022)
    assert bounds.cube_position_in_root_m["z"] == (0.101, 0.117)
    assert bounds.cube_yaw_deg == (20.0, 40.0)
    assert bounds.seed == 20260821
    assert EXPERIMENT_DEFINITION.robustness.perturbation_count == 50
    assert EXPERIMENT_DEFINITION.robustness.required_pass_count == 45


def test_budget_manifest_and_static_jobs_are_complete_and_deterministic():
    manifest = budget_manifest()
    counts = manifest["nominal_counts"]

    assert counts == {
        "static_job_count": 30,
        "static_sample_count": 600_000,
        "dynamic_candidate_count": 640,
        "grasp_refinement_count": 2_560,
        "manipulation_refinement_count": 1_280,
        "exact_candidate_count": 80,
    }
    assert manifest["per_band"]["dynamic_candidate_count"] == 128
    assert manifest["per_band"]["grasp_refine_seed_count"] == 4
    assert manifest["per_band"]["grasp_refine_per_seed"] == 128
    assert manifest["per_band"]["manipulation_seed_count"] == 2
    assert manifest["per_band"]["manipulation_refine_per_seed"] == 128
    assert manifest["per_band"]["exact_candidate_count"] == 16
    assert manifest["selection_policy"]["fill_missing_bands"] is False

    jobs = static_screen_jobs()
    assert len(jobs) == 6 * 5
    assert sum(job.samples for job in jobs) == 600_000
    assert all(job.face_sample_counts == (5_000, 5_000, 5_000, 5_000) for job in jobs)
    assert jobs == static_screen_jobs()
    assert len({job.seed for job in jobs}) == len(jobs)


@pytest.mark.parametrize(
    ("better_changes", "worse_changes"),
    [
        ({"passed": True}, {"passed": False}),
        ({"margin": 0.2}, {"margin": 0.1}),
        ({"aligned_duty": 0.8}, {"aligned_duty": 0.7}),
        ({"p95_spread": 0.003}, {"p95_spread": 0.004}),
        ({"topology": 0.8}, {"topology": 0.7}),
        ({"force": 2.0}, {"force": 3.0}),
        ({"drift": 1.0}, {"drift": 2.0}),
        ({"saturation": 0.1}, {"saturation": 0.2}),
    ],
)
def test_rank_uses_each_declared_metric_in_order(better_changes, worse_changes):
    better = _result(2, 15.0, **better_changes)
    worse = _result(1, 15.0, **worse_changes)

    assert aligned_contact_candidate_rank(better) > aligned_contact_candidate_rank(
        worse
    )


def test_rank_uses_lower_candidate_id_as_total_order_tie_break():
    lower_id = _result(1, 15.0)
    higher_id = _result(2, 15.0)

    assert deterministic_rank_results([higher_id, lower_id]) == (
        lower_id,
        higher_id,
    )
    with pytest.raises(ValueError, match="unique"):
        deterministic_rank_results([lower_id, copy.deepcopy(lower_id)])


def test_rank_reads_real_nested_alignment_summary_metrics():
    better = _result(2, 15.0)
    worse = _result(1, 15.0)
    for result, duty, spread in ((better, 0.8, 0.003), (worse, 0.7, 0.004)):
        metrics = result["summary"]["metrics"]
        metrics.pop("operation_aligned_contact_duty")
        metrics.pop("operation_contact_height_spread_p95_m")
        metrics["contact_alignment"] = {
            "operation": {
                "aligned_duty": duty,
                "height_spread_p95_m": spread,
            }
        }

    assert aligned_contact_candidate_rank(better) > aligned_contact_candidate_rank(
        worse
    )


def test_per_band_selection_never_fills_an_empty_or_short_band():
    low = _result(1, 10.0, margin=0.1)
    high = _result(2, 10.0, margin=0.2)
    only = _result(3, 15.0, margin=0.3)

    selected = select_per_band([low, only, high], per_band=2)

    assert tuple(selected) == (10.0, 12.5, 15.0, 17.5, 20.0)
    assert selected[10.0] == (high, low)
    assert selected[12.5] == ()
    assert selected[15.0] == (only,)
    assert selected[17.5] == ()
    assert selected[20.0] == ()


def test_executable_campaign_preserves_bands_gates_and_id_binding():
    config = _raw_config()
    screen_calls = []

    def screen_candidates(base, job, retain):
        screen_calls.append((job.edge_m, job.tilt_band_center_deg, retain))
        return [copy.deepcopy(base)]

    def refine_candidates(parent, *, count, band_center_deg, seed, stage):
        assert count == 1
        candidate = copy.deepcopy(parent)
        candidate.setdefault("search_metadata", {})["refine_stage"] = stage
        return [candidate]

    def perturb_cases(parent, *, count, seed):
        assert count == 50
        return [copy.deepcopy(parent) for _ in range(count)]

    def run_candidates(payloads, workers):
        assert workers == 2
        results = []
        for candidate_id, candidate in reversed(payloads):
            results.append(
                {
                    **_result(
                        candidate_id,
                        candidate["search_metadata"]["tilt_band_center_deg"],
                    ),
                    "config": copy.deepcopy(candidate),
                }
            )
            results[-1]["summary"]["stage_status"]["grasp_success"] = True
        return results

    report = tune_aligned_contacts(
        config,
        workers=2,
        run_candidates=run_candidates,
        screen_candidates=screen_candidates,
        refine_candidates=refine_candidates,
        perturb_cases=perturb_cases,
        dynamic_candidates_per_band=1,
        grasp_refine_seed_count_per_band=1,
        grasp_refine_per_seed=1,
        manipulation_seed_count_per_band=1,
        manipulation_refine_per_seed=1,
        exact_candidates_per_band=1,
    )

    assert len(screen_calls) == 30
    assert report["actual_counts"] == {
        "dynamic_candidate_count": 5,
        "grasp_refinement_count": 5,
        "manipulation_refinement_count": 5,
        "exact_candidate_count": 5,
        "perturbation_count": 250,
    }
    assert report["perturbation_passes_by_finalist"] == {
        candidate["candidate_id"]: 50 for candidate in report["finalists"]
    }
    assert len(report["robust_finalists"]) == 5
    assert report["campaign_success"] is True


@pytest.mark.parametrize(
    ("section", "mutate"),
    [
        (
            "pose_constraints",
            lambda config: config["pose_constraints"].__setitem__(
                "palm_press_depth_m", [0.001, 0.008]
            ),
        ),
        (
            "contact_alignment",
            lambda config: config["contact_alignment"].__setitem__(
                "max_height_spread_m", 0.006
            ),
        ),
        (
            "aligned_contact_campaign",
            lambda config: config["aligned_contact_campaign"]["edges_m"].pop(),
        ),
        (
            "search.budget",
            lambda config: config["search"]["budget"].__setitem__(
                "dynamic_candidate_count", 639
            ),
        ),
        (
            "constant-density",
            lambda config: config["cube"].__setitem__("mass_kg", 0.020),
        ),
    ],
)
def test_v4_versioned_contract_rejects_drift(section, mutate):
    config = copy.deepcopy(_raw_config())
    mutate(config)

    with pytest.raises(ValueError, match=section):
        validate_config(config)


def test_v4_definition_cannot_be_selected_from_schema_v3():
    config = copy.deepcopy(_raw_config())
    config["schema_version"] = 3

    with pytest.raises(ValueError, match="schema_version 4"):
        resolve_experiment(config)


def test_catalog_pose_is_strict_but_explicit_run_overrides_reach_evaluation():
    invalid_pose = copy.deepcopy(_raw_config())
    invalid_pose["hand_pose"]["rpy_deg"][1] = 90.0
    with pytest.raises(ValueError, match="resolved finger_down_tilt_deg"):
        validate_config(invalid_pose)

    invalid_pose["cube"].update(
        {"edge_m": 0.070, "mass_kg": 0.25, "friction": 0.4}
    )
    invalid_pose["run_context"] = {"kind": "parameter_override_run"}
    validate_config(invalid_pose)

    bad_context = copy.deepcopy(invalid_pose)
    bad_context["run_context"] = {"kind": "unversioned_override"}
    with pytest.raises(ValueError, match="run_context.kind"):
        validate_config(bad_context)
