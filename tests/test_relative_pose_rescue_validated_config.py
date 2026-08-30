from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from xhand_grasp.artifacts import resolved_run_config
from xhand_grasp.config import ACTIVE_ACTUATORS, load_config, validate_config
from xhand_grasp.experiment import resolve_experiment
from xhand_grasp.v2_search import _rpy_matrix


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_larger_cube_relative_pose_rescue_validated.json"
)
EXPECTED_CUBE_IN_ROOT_M = np.asarray(
    [0.08369118081660427, -0.02886063167725713, 0.10872576454433161]
)
EXPECTED_GRASP = np.asarray(
    [
        1.019914111112225,
        0.45,
        0.6156759841937474,
        -0.0049579503195133975,
        0.4434969320225719,
        1.6035425081988066,
        0.7608503089023541,
        1.3064070166012365,
    ]
)
EXPECTED_DELTA = np.asarray(
    [
        -0.1991315467092353,
        0.09007911014711634,
        0.36111262803013183,
        -0.0007398344932933489,
        -0.1236577010107487,
        0.1347433499994188,
        0.03704509887157975,
        0.41101557911873393,
    ]
)


def test_validated_relative_pose_rescue_config_is_exact_and_runnable():
    config = load_config(CONFIG_PATH)
    validate_config(config)
    definition = resolve_experiment(config)
    bounds = definition.search_bounds

    assert config["experiment_status"] == {
        "classification": "validated_constant_density",
        "passed": True,
        "grasp_success": True,
        "manipulation_success": True,
        "full_success": True,
        "fixed_mass_discovery_passed": False,
        "constant_density_passed": True,
        "robustness_passed": False,
        "note": (
            "The exact nominal constant-density configuration passed all 54 "
            "declared hard checks in two recorded identical runs. Fixed-mass "
            "discovery and the registered robustness campaign are not claimed "
            "by this source config."
        ),
    }
    assert config["hand_pose"]["translation_m"] == pytest.approx(
        [-0.036454669709448065, 0.008499986928773153, 0.19777491509344874]
    )
    assert config["hand_pose"]["rpy_deg"] == pytest.approx(
        [0.8800118717318188, 89.5166867019805, -2.6325407810459214]
    )
    assert config["cube"]["edge_m"] == pytest.approx(0.062)
    assert config["cube"]["rpy_deg"] == pytest.approx(
        [0.0, 0.0, 30.589880839844653]
    )

    expected_mass = definition.size_campaign.constant_density_mass_kg(0.062)
    assert expected_mass == pytest.approx(0.17653925925925934, abs=1e-15)
    assert config["cube"]["mass_kg"] == pytest.approx(expected_mass, abs=1e-15)
    assert config["cube"]["friction"] == pytest.approx(
        definition.size_campaign.discovery_friction
    )

    grasp = np.asarray(
        [config["control"]["grasp_targets_rad"][name] for name in ACTIVE_ACTUATORS]
    )
    delta = np.asarray(
        [
            config["control"]["manipulation_delta_rad"][name]
            for name in ACTIVE_ACTUATORS
        ]
    )
    np.testing.assert_array_equal(grasp, EXPECTED_GRASP)
    np.testing.assert_array_equal(delta, EXPECTED_DELTA)
    assert bounds.contains_targets(config["control"]["grasp_targets_rad"])
    assert bounds.contains_manipulation_delta(
        config["control"]["manipulation_delta_rad"]
    )
    for name, endpoint in zip(ACTIVE_ACTUATORS, grasp + delta):
        lower, upper = bounds.actuator_targets_rad[name]
        assert lower <= endpoint <= upper

    cube_world = np.asarray(
        [
            *config["cube"]["center_xy_m"],
            config["scene"]["support_top_z_m"] + config["cube"]["edge_m"] / 2.0,
        ]
    )
    root_rotation = _rpy_matrix(config["hand_pose"]["rpy_deg"])
    actual_cube_in_root = root_rotation.T @ (
        cube_world - np.asarray(config["hand_pose"]["translation_m"])
    )
    np.testing.assert_allclose(
        actual_cube_in_root, EXPECTED_CUBE_IN_ROOT_M, rtol=0.0, atol=1e-12
    )

    provenance = config["validation_provenance"]
    assert provenance["stable_parent_candidate_id"] == 187
    assert provenance["selected_local_candidate_id"] == 44
    assert provenance["cube_position_in_root_m"] == pytest.approx(
        EXPECTED_CUBE_IN_ROOT_M
    )
    assert provenance["bounded_operation_search"] == {
        "declared_attempt_budget": 256,
        "conservative_attempts_consumed": 204,
        "unspent_attempts": 52,
        "pre_local_attempts": 11,
        "diagnostic_batch_attempts_counted": 96,
        "effective_unique_local_samples": 96,
        "effective_full_passes": 11,
        "exact_confirmation_attempts": 1,
        "note": (
            "The first 96-point local batch completed simulation but its ephemeral "
            "postprocessor discarded summaries; those attempts remain conservatively "
            "counted. The corrected 96-point batch contained 11 full passes."
        ),
    }
    assert provenance["deterministic_replay"]["recorded_exact_runs"] == 2
    assert provenance["deterministic_replay"]["recorded_results_identical"] is True

    resolved = resolved_run_config(
        config,
        {
            "passed": True,
            "failed_checks": [],
            "stage_status": {
                "grasp_success": True,
                "manipulation_success": True,
                "full_success": True,
            },
        },
    )
    status = resolved["experiment_status"]
    assert status["classification"] == "validated_constant_density"
    assert status["passed"] is True
    assert status["campaign_validated"] is True
    assert status["constant_density_passed"] is True
    assert status["full_success"] is True
    assert status["input_classification"] == "validated_constant_density"
