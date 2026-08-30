from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from xhand_grasp.aligned_contacts_tuning import (
    ALIGNED_PERTURBATION_RANGES,
    AlignedStaticJob,
    AlignedTuningBudget,
    StaticScreenOutcome,
    aligned_static_candidate_advances,
    generate_aligned_perturbation_configs,
    materialize_aligned_candidate,
    run_parallel_static_screen,
    tune_aligned_contacts,
)
from xhand_grasp.config import (
    ACTIVE_ACTUATORS,
    load_config,
    resolved_pose_constraint_values,
    validate_config,
)
from xhand_grasp.experiment import resolve_experiment


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_tilted_down_aligned_contacts_grasp_then_lift.json"
)


@pytest.fixture
def v4_config() -> dict:
    return load_config(CONFIG_PATH)


def _candidate(
    config: dict,
    *,
    band: float,
    face_index: int = 0,
    static_id: int | None = None,
) -> dict:
    definition = resolve_experiment(config)
    return materialize_aligned_candidate(
        config,
        edge_m=0.062,
        tilt_band_center_deg=band,
        roll_deg=0.0,
        yaw_deg=-2.6325407810459214,
        press_depth_m=0.005,
        cube_in_root_y_m=-0.028860631677,
        cube_in_root_z_m=0.108725764544,
        cube_yaw_deg=30.0,
        grasp_targets_rad=config["control"]["grasp_targets_rad"],
        target_assignment=definition.candidate_faces[face_index],
        static_candidate_id=static_id,
    )


def _summary(
    *,
    grasp: bool,
    full: bool,
    aligned_duty: float,
) -> dict:
    return {
        "passed": full,
        "failed_checks": [] if full else ["mock_gate"],
        "checks": {"mock_gate": full},
        "stage_status": {
            "grasp_success": grasp,
            "manipulation_success": full,
            "full_success": full,
        },
        "metrics": {
            "contact_alignment": {
                "operation": {
                    "aligned_duty": aligned_duty,
                    "height_spread_p95_m": 0.003,
                }
            },
            "operation_target_face_simultaneous_duty": 0.8,
            "peak_total_distal_contact_force_n": 2.0,
            "orientation_drift_deg": 1.0,
            "actuator_saturation_fraction": 0.05,
        },
    }


def test_materialized_candidate_resolves_pose_material_face_and_all_controls(v4_config):
    definition = resolve_experiment(v4_config)
    campaign = definition.aligned_contact_campaign
    assert campaign is not None

    for band in campaign.tilt_band_centers_deg:
        for face_index, assignment in enumerate(definition.candidate_faces):
            candidate = _candidate(
                v4_config,
                band=band,
                face_index=face_index,
                static_id=face_index,
            )
            resolved = resolved_pose_constraint_values(candidate)

            validate_config(candidate)
            assert candidate["cube"]["mass_kg"] == pytest.approx(
                campaign.constant_density_mass_kg(0.062), abs=1e-15
            )
            assert candidate["cube"]["friction"] == pytest.approx(0.8)
            assert candidate["contact_topology"]["target_faces"] == assignment.as_dict()
            assert set(candidate["control"]["grasp_targets_rad"]) == set(
                ACTIVE_ACTUATORS
            )
            assert candidate["control"]["manipulation_delta_rad"] == {
                name: 0.0 for name in ACTIVE_ACTUATORS
            }
            assert resolved["palm_press_depth_m"] == pytest.approx(0.005, abs=1e-14)
            assert resolved["cube_position_in_root_m"][1:] == pytest.approx(
                (-0.028860631677, 0.108725764544), abs=1e-12
            )
            assert 10.0 - 1e-12 <= resolved["finger_down_tilt_deg"] <= 20.0 + 1e-12
            assert (
                10.0 - 1e-12
                <= resolved["palm_plane_ground_angle_deg"]
                <= 20.0 + 1e-12
            )


def test_v4_static_gate_does_not_require_clean_or_estimated_alignment():
    diagnostic = {
        "forbidden_contact": False,
        "max_penetration_m": 0.002,
        "clean_target_contact_count": 0,
        "near_target_face_count": 3,
        "target_site_signed_distance_m": [-0.002, 0.0, 0.003],
        "contact_height_aligned_estimate": False,
    }

    assert aligned_static_candidate_advances(
        diagnostic, max_penetration_m=0.002
    )
    diagnostic["target_site_signed_distance_m"][1] = 0.0031
    assert not aligned_static_candidate_advances(
        diagnostic, max_penetration_m=0.002
    )


def test_tiny_mujoco_static_job_records_gate_and_signed_distance_semantics(v4_config):
    job = AlignedStaticJob(
        job_id=0,
        edge_m=0.062,
        tilt_band_center_deg=15.0,
        samples=4,
        seed=17,
        face_sample_counts=(1, 1, 1, 1),
    )

    outcome = run_parallel_static_screen(
        v4_config,
        jobs=[job],
        retain_per_band=4,
        workers=1,
    )

    assert outcome.sample_count == 4
    record = outcome.job_records[0]
    assert record["valid_pose_count"] == 4
    assert record["retained_count"] <= record["eligible_count"] <= 4
    assert len(record["face_gate_counts"]) == 4
    assert len(record["face_assignments"]) == 4
    assert all("target_faces" in item for item in record["face_gate_counts"])
    assert sum(item["clean_three_count"] for item in record["face_gate_counts"]) == (
        record["clean_three_count"]
    )
    assert sum(item["near_three_count"] for item in record["face_gate_counts"]) == (
        record["near_three_count"]
    )
    assert record["stop_reason"] in {
        "eligible_candidates_retained",
        "no_static_candidate_passed_geometry_gate",
    }
    near = outcome.near_miss_by_band[15.0]
    assert near is not None
    diagnostic = near["diagnostic"]
    assert len(diagnostic["target_nearest_signed_distance_m"]) == 3
    assert len(diagnostic["target_nearest_absolute_distance_m"]) == 3
    for signed, absolute in zip(
        diagnostic["target_nearest_signed_distance_m"],
        diagnostic["target_nearest_absolute_distance_m"],
    ):
        if np.isfinite(signed):
            assert absolute == pytest.approx(abs(signed), abs=1e-15)


def test_best_overall_perturbations_are_independent_bounded_and_deterministic(
    v4_config,
):
    # Use the lower discovery boundary so negative tilt draws prove the
    # robustness envelope is not silently clipped back to 10 degrees.
    parent = _candidate(v4_config, band=10.0)
    definition = resolve_experiment(parent)
    campaign = definition.aligned_contact_campaign
    assert campaign is not None
    first = generate_aligned_perturbation_configs(parent, count=50, seed=71)
    second = generate_aligned_perturbation_configs(parent, count=50, seed=71)

    assert first == second
    assert len(first) == 50
    parent_xy = np.asarray(parent["cube"]["center_xy_m"])
    parent_cube_rpy = np.asarray(parent["cube"]["rpy_deg"])
    parent_hand_rpy = np.asarray(parent["hand_pose"]["rpy_deg"])
    parent_pose = resolved_pose_constraint_values(parent)
    xy_deltas = []
    relative_deltas = []
    tilt_deltas = []
    press_deltas = []
    for candidate in first:
        validate_config(candidate)
        metadata = candidate["candidate_metadata"]["resolved_perturbations"]
        assert candidate["candidate_metadata"]["robustness_seed"] == 71
        cube_xy_delta = np.asarray(candidate["cube"]["center_xy_m"]) - parent_xy
        cube_rpy_delta = np.asarray(candidate["cube"]["rpy_deg"]) - parent_cube_rpy
        resolved = resolved_pose_constraint_values(candidate)

        assert candidate["cube"]["edge_m"] == parent["cube"]["edge_m"]
        assert np.max(np.abs(cube_xy_delta)) <= 0.0015 + 1e-12
        assert np.max(np.abs(cube_rpy_delta)) <= 3.0 + 1e-12
        assert 0.0 <= candidate["cube"]["z_offset_m"] <= 0.0005
        assert abs(candidate["hand_pose"]["rpy_deg"][0] - parent_hand_rpy[0]) <= 1.0
        assert abs(candidate["hand_pose"]["rpy_deg"][2] - parent_hand_rpy[2]) <= 1.0
        assert abs(
            resolved["finger_down_tilt_deg"]
            - parent_pose["finger_down_tilt_deg"]
        ) <= 1.0 + 1e-12
        assert abs(
            resolved["palm_press_depth_m"] - parent_pose["palm_press_depth_m"]
        ) <= 0.001 + 1e-12
        reference_mass = campaign.constant_density_mass_kg(
            candidate["cube"]["edge_m"]
        )
        assert 0.9 <= candidate["cube"]["mass_kg"] / reference_mass <= 1.1
        assert 0.7 <= candidate["cube"]["friction"] <= 0.9
        np.testing.assert_allclose(
            metadata["cube_center_xy_delta_m"], cube_xy_delta, atol=1e-15
        )
        np.testing.assert_allclose(
            candidate["candidate_metadata"]["cube_in_root_m"],
            resolved["cube_position_in_root_m"],
            atol=1e-15,
        )
        xy_deltas.append(cube_xy_delta)
        relative_deltas.append(np.asarray(metadata["cube_in_root_delta_m"]))
        tilt_deltas.append(metadata["finger_down_tilt_delta_deg"])
        press_deltas.append(metadata["palm_press_depth_delta_m"])

    # Cube XY is not cancelled by translating the root with it.
    assert any(np.linalg.norm(value) > 1e-6 for value in xy_deltas)
    assert any(np.linalg.norm(value[:2]) > 1e-6 for value in relative_deltas)
    assert min(tilt_deltas) < -0.5
    assert max(tilt_deltas) > 0.5
    assert min(
        resolved_pose_constraint_values(candidate)["finger_down_tilt_deg"]
        for candidate in first
    ) < 10.0
    assert min(press_deltas) < -0.0005
    assert max(press_deltas) > 0.0005
    assert ALIGNED_PERTURBATION_RANGES.as_dict()["cube_rpy_delta_deg"] == [
        -3.0,
        3.0,
    ]


def test_perturbation_generator_resolves_unannotated_template_band_and_rejects_tie(
    v4_config,
):
    v4_config["experiment_status"] = {
        "classification": "validated_aligned_contacts_robust",
        "passed": True,
        "robustness_passed": True,
    }
    v4_config["candidate_metadata"] = {
        "candidate_id": 91,
        "search_stage": "exact_1ms_confirmation",
        "material_policy": "constant_density",
    }
    generated = generate_aligned_perturbation_configs(
        v4_config, count=2, seed=72
    )
    assert all("experiment_status" not in candidate for candidate in generated)
    assert all(
        candidate["candidate_metadata"]["search_stage"] == "robustness_trial"
        for candidate in generated
    )
    assert all(
        candidate["candidate_metadata"]["material_policy"]
        == "local_pose_and_material_perturbation"
        for candidate in generated
    )
    assert generated[0]["candidate_metadata"]["parent_candidate_metadata"] == (
        v4_config["candidate_metadata"]
    )
    assert [
        candidate["candidate_metadata"]["tilt_band_center_deg"]
        for candidate in generated
    ] == [15.0, 15.0]

    tied = materialize_aligned_candidate(
        v4_config,
        edge_m=0.062,
        tilt_band_center_deg=10.0,
        finger_down_tilt_deg=11.25,
        roll_deg=0.0,
        yaw_deg=-2.6325407810459214,
        press_depth_m=0.005,
        cube_in_root_y_m=-0.028860631677,
        cube_in_root_z_m=0.108725764544,
        cube_yaw_deg=30.0,
        grasp_targets_rad=v4_config["control"]["grasp_targets_rad"],
        target_assignment=resolve_experiment(v4_config).candidate_faces[0],
    )
    tied.pop("candidate_metadata", None)
    with pytest.raises(ValueError, match="equidistant"):
        generate_aligned_perturbation_configs(tied, count=1, seed=72)


def test_tiny_orchestration_enforces_band_gates_exact_rerun_and_catalog(v4_config):
    definition = resolve_experiment(v4_config)
    bands = definition.aligned_contact_campaign.tilt_band_centers_deg
    groups = {
        float(band): tuple(
            _candidate(v4_config, band=band, face_index=index, static_id=100 * i + index)
            for index in range(4)
        )
        for i, band in enumerate(bands)
        if band != 20.0
    }

    def static_runner(config, **kwargs):
        del config, kwargs
        return StaticScreenOutcome(
            sample_count=30,
            retained_by_band={float(band): groups.get(float(band), ()) for band in bands},
            diagnostics_by_band={float(band): () for band in bands},
            near_miss_by_band={float(band): None for band in bands},
            job_records=(),
        )

    def runner(payloads, workers):
        assert workers == 2
        results = []
        for candidate_id, config in payloads:
            stage = config["candidate_metadata"]["search_stage"]
            band = float(config["candidate_metadata"]["tilt_band_center_deg"])
            grasp = stage == "pose_grasp_refinement" and band in (10.0, 15.0)
            full = stage in ("manipulation_refinement", "exact_1ms_confirmation") and band == 10.0
            if stage == "best_overall_robustness":
                grasp = True
                full = True
            results.append(
                {
                    "candidate_id": candidate_id,
                    "config": copy.deepcopy(config),
                    "summary": _summary(
                        grasp=grasp or full,
                        full=full,
                        aligned_duty=0.7 + band / 100.0,
                    ),
                }
            )
        return list(reversed(results))

    result = tune_aligned_contacts(
        v4_config,
        workers=2,
        seed=73,
        run_candidates=runner,
        static_runner=static_runner,
        budget=AlignedTuningBudget(
            static_samples_per_edge_band=1,
            dynamic_candidates_per_band=4,
            grasp_refine_seed_count_per_band=1,
            grasp_refine_per_seed=1,
            manipulation_seed_count_per_band=1,
            manipulation_refine_per_seed=1,
            exact_candidates_per_band=1,
            perturbation_count=2,
        ),
    )

    assert result["stage_counts"] == {
        "static_sample_count": 30,
        "static_retained_count": 16,
        "dynamic_close_verify_count": 16,
        "grasp_refinement_count": 4,
        "manipulation_refinement_count": 2,
        "exact_1ms_confirmation_count": 2,
        "catalog_diagnostic_rerun_count": 1,
        "per_band_finalist_count": 1,
        "best_overall_perturbation_count": 2,
    }
    assert len(result["finalists"]) == 1
    assert len(result["selected_band_candidates"]) == 5
    assert [
        item["config"]["candidate_metadata"]["tilt_band_center_deg"]
        for item in result["selected_band_candidates"]
    ] == list(bands)
    assert result["per_band"]["20"]["dynamic_count"] == 0
    selected_20 = result["selected_band_candidates"][-1]
    assert selected_20["search_stage"] == "catalog_diagnostic_rerun"
    assert result["per_band"]["12.5"]["manipulation_refinement_count"] == 0
    assert result["per_band"]["15"]["manipulation_refinement_count"] == 1
    assert result["best"]["search_stage"] == "exact_1ms_confirmation"
    assert result["best"]["summary"]["metrics"][
        "operation_aligned_contact_duty"
    ] == pytest.approx(0.8)
    assert result["best"]["config"]["experiment_status"]["passed"] is True
    assert result["best"]["config"]["experiment_status"][
        "robustness_passed"
    ] is True
    assert len(result["catalog_candidates"]) == 2
    assert result["perturbation_seed"] == 400_000_073
    assert result["best"]["local_perturbation_probe"]["seed"] == 400_000_073
    assert result["candidate_count"] == 24
    assert result["diagnostic_simulation_count"] == 1
    assert result["simulation_count"] == 27
    assert set(result["per_size"]) == {"59", "60", "61", "62", "63", "64"}
    assert result["per_size"]["62"]["dynamic_count"] == 16
    assert result["per_size"]["62"]["exact_hard_pass_count"] == 1


def test_tuner_rejects_viewer_override_context_before_any_work(v4_config):
    v4_config["run_context"] = {"kind": "parameter_override_run"}

    def forbidden(*args, **kwargs):  # pragma: no cover - assertion path
        raise AssertionError("override input must be rejected before search")

    with pytest.raises(ValueError, match="tuning requires a canonical config"):
        tune_aligned_contacts(
            v4_config,
            workers=1,
            seed=20260821,
            run_candidates=forbidden,
            static_runner=forbidden,
        )


def test_all_static_gate_failures_stop_honestly_but_rerun_one_artifact_per_band(
    v4_config,
):
    definition = resolve_experiment(v4_config)
    bands = definition.aligned_contact_campaign.tilt_band_centers_deg

    def empty_static(config, **kwargs):
        del config, kwargs
        return StaticScreenOutcome(
            sample_count=30,
            retained_by_band={float(band): () for band in bands},
            diagnostics_by_band={float(band): () for band in bands},
            near_miss_by_band={float(band): None for band in bands},
            job_records=(),
        )

    def diagnostic_runner(payloads, workers):
        del workers
        return [
            {
                "candidate_id": candidate_id,
                "config": copy.deepcopy(config),
                "summary": _summary(grasp=False, full=False, aligned_duty=0.0),
            }
            for candidate_id, config in payloads
        ]

    result = tune_aligned_contacts(
        v4_config,
        workers=1,
        seed=79,
        run_candidates=diagnostic_runner,
        static_runner=empty_static,
        budget=AlignedTuningBudget(
            static_samples_per_edge_band=1,
            dynamic_candidates_per_band=4,
            grasp_refine_seed_count_per_band=1,
            grasp_refine_per_seed=1,
            manipulation_seed_count_per_band=1,
            manipulation_refine_per_seed=1,
            exact_candidates_per_band=1,
            perturbation_count=2,
        ),
    )

    assert result["stop_reason"] == "no_static_candidate_passed_geometry_gate"
    assert result["candidate_count"] == 0
    assert result["diagnostic_simulation_count"] == 5
    assert result["perturbation_probe_count"] == 2
    assert len(result["catalog_candidates"]) == 2
    assert result["simulation_count"] == 7
    assert result["robustness_success"] is False
    assert len(result["selected_band_candidates"]) == 5
    assert all(
        item["search_stage"] == "catalog_diagnostic_rerun"
        for item in result["selected_band_candidates"]
    )


def test_stage_runner_rejects_candidate_id_to_config_rebinding(v4_config):
    definition = resolve_experiment(v4_config)
    bands = definition.aligned_contact_campaign.tilt_band_centers_deg
    groups = {
        float(band): tuple(
            _candidate(v4_config, band=band, face_index=index)
            for index in range(4)
        )
        for band in bands
    }

    def static_runner(config, **kwargs):
        del config, kwargs
        return groups

    def bad_runner(payloads, workers):
        del workers
        candidate_id, config = payloads[0]
        rebound = copy.deepcopy(config)
        rebound["cube"]["friction"] += 0.01
        return [
            {
                "candidate_id": candidate_id,
                "config": rebound,
                "summary": _summary(grasp=False, full=False, aligned_duty=0.0),
            }
        ]

    with pytest.raises(RuntimeError, match="different configuration|exactly match"):
        tune_aligned_contacts(
            v4_config,
            workers=1,
            seed=83,
            run_candidates=bad_runner,
            static_runner=static_runner,
            budget=AlignedTuningBudget(
                static_samples_per_edge_band=1,
                dynamic_candidates_per_band=4,
                grasp_refine_seed_count_per_band=1,
                grasp_refine_per_seed=1,
                manipulation_seed_count_per_band=1,
                manipulation_refine_per_seed=1,
                exact_candidates_per_band=1,
                perturbation_count=0,
            ),
        )
