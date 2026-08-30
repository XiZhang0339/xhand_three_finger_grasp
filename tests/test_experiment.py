from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from xhand_grasp.config import validate_config
from xhand_grasp.experiment import (
    DEFAULT_V1_EXPERIMENT_ID,
    LEGACY_V1_EXPERIMENT,
    OpposedFaceAssignment,
    get_experiment,
    opposite_face,
    registered_experiments,
    resolve_experiment,
)
from xhand_grasp.experiments.opposed_face_palm_down import (
    EXPERIMENT_ID,
    OPPOSED_FACE_PALM_DOWN,
    SEARCH_FACE_ASSIGNMENTS,
    TARGET_FACES,
)


ROOT = Path(__file__).resolve().parents[1]
V1_CONFIG = ROOT / "grasp_configs" / "left_three_finger_cube.json"
V2_CONFIG = ROOT / "grasp_configs" / "left_opposed_face_palm_down.json"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_builtin_experiments_are_registered_by_id():
    registered = registered_experiments()
    assert registered[DEFAULT_V1_EXPERIMENT_ID] is LEGACY_V1_EXPERIMENT
    assert registered[EXPERIMENT_ID] is OPPOSED_FACE_PALM_DOWN
    assert get_experiment(EXPERIMENT_ID) is OPPOSED_FACE_PALM_DOWN
    with pytest.raises(ValueError, match="unknown experiment_id"):
        get_experiment("does_not_exist")


def test_opposed_face_assignment_enforces_exact_topology():
    assert TARGET_FACES.as_dict() == {
        "thumb": "-X",
        "index": "+X",
        "mid": "+X",
    }
    assert opposite_face(TARGET_FACES.thumb) == TARGET_FACES.index
    assert OpposedFaceAssignment.from_mapping(TARGET_FACES.as_dict()) == TARGET_FACES

    with pytest.raises(ValueError, match="same cube face"):
        OpposedFaceAssignment(thumb="-X", index="+X", mid="+Y")
    with pytest.raises(ValueError, match="exact opposite"):
        OpposedFaceAssignment(thumb="-Y", index="+X", mid="+X")
    with pytest.raises(ValueError, match="exactly"):
        OpposedFaceAssignment.from_mapping({"thumb": "-X", "index": "+X"})


def test_schema_v1_without_experiment_id_resolves_to_legacy_default():
    config = _read(V1_CONFIG)
    assert "experiment_id" not in config
    definition = resolve_experiment(config)
    assert definition is LEGACY_V1_EXPERIMENT
    assert definition.experiment_id == DEFAULT_V1_EXPERIMENT_ID
    assert definition.evaluation.target_faces is None


def test_schema_v2_resolves_explicit_definition_and_does_not_claim_success():
    config = _read(V2_CONFIG)
    definition = resolve_experiment(config)
    assert config["schema_version"] == 2
    assert config["experiment_id"] == EXPERIMENT_ID
    assert definition is OPPOSED_FACE_PALM_DOWN
    assert config["experiment_status"]["classification"] == "initial_near_miss"
    assert config["experiment_status"]["passed"] is False
    assert config["initial_near_miss"]["dynamic_result"]["passed"] is False
    assert config["contact_topology"]["target_faces"] == TARGET_FACES.as_dict()
    assert config["contact_topology"] == definition.evaluation.contact_topology_config()

    with pytest.raises(ValueError, match="requires experiment_id"):
        resolve_experiment({"schema_version": 2})

    invalid = dict(config)
    invalid["contact_topology"] = dict(config["contact_topology"])
    invalid["contact_topology"]["target_faces"] = {
        "thumb": "-X",
        "index": "+X",
        "mid": "+Y",
    }
    with pytest.raises(ValueError, match="same cube face"):
        resolve_experiment(invalid)


def test_palm_down_topology_thresholds_are_fully_declared():
    evaluation = OPPOSED_FACE_PALM_DOWN.evaluation
    assert evaluation.target_faces == TARGET_FACES
    assert evaluation.palm_normal_local_axis == "+X"
    assert evaluation.world_down_axis == "-Z"
    assert evaluation.max_palm_down_angle_deg == 30.0
    assert evaluation.min_face_normal_alignment == 0.95
    assert evaluation.surface_tolerance_m == pytest.approx(0.00005)
    assert evaluation.min_face_edge_margin_m == pytest.approx(0.0005)
    assert evaluation.min_contact_force_n == pytest.approx(0.05)
    assert evaluation.min_target_force_purity == pytest.approx(0.95)
    assert dict(evaluation.min_target_face_contact_duty) == {
        "thumb": 0.8,
        "index": 0.8,
        "mid": 0.8,
    }
    assert evaluation.min_simultaneous_topology_duty == 0.7
    assert evaluation.max_off_target_duty == 0.01
    assert evaluation.max_off_target_run_s == pytest.approx(0.010)
    assert evaluation.forbid_active_nondistal is True


def test_search_enumerates_all_four_horizontal_shared_faces():
    config = _read(V2_CONFIG)
    shared_faces = tuple(assignment.index for assignment in SEARCH_FACE_ASSIGNMENTS)
    assert shared_faces == ("+X", "-X", "+Y", "-Y")
    assert tuple(config["search"]["shared_faces"]) == shared_faces
    assert OPPOSED_FACE_PALM_DOWN.candidate_faces == SEARCH_FACE_ASSIGNMENTS
    for assignment in SEARCH_FACE_ASSIGNMENTS:
        assert assignment.index == assignment.mid
        assert opposite_face(assignment.thumb) == assignment.index


def test_pitch75_near_miss_and_final_targets_are_inside_search_domain():
    config = _read(V2_CONFIG)
    bounds = OPPOSED_FACE_PALM_DOWN.search_bounds
    assert bounds.palm_pitch_values_deg == (60.0, 65.0, 70.0, 75.0, 80.0)
    assert bounds.palm_pitch_deg == (60.0, 80.0)
    assert bounds.contains_pitch(config["hand_pose"]["rpy_deg"][1])
    assert bounds.contains_cube_position(
        config["initial_near_miss"]["cube_position_in_root_m"]
    )
    assert bounds.contains_targets(config["control"]["final_targets_rad"])
    assert not bounds.contains_pitch(59.999)
    assert not bounds.contains_cube_position([0.0549, -0.027, 0.099])


def test_search_and_robustness_budgets_are_reproducible():
    config = _read(V2_CONFIG)
    search = OPPOSED_FACE_PALM_DOWN.search_bounds
    assert search.seed == 20260821
    assert search.kinematic_samples_per_pitch == 50_000
    assert search.dynamic_candidate_count == 512
    assert search.local_refine_seed_count * search.local_refine_per_seed == 2048
    assert search.final_candidate_count * search.perturbations_per_final_candidate == 256
    assert search.fallback_kinematic_samples_per_pitch == 5_000
    assert search.fallback_candidate_count == 2048
    assert config["search"]["seed"] == search.seed
    assert config["search"]["budget"] == {
        "palm_pitch_values_deg": list(search.palm_pitch_values_deg),
        "kinematic_samples_per_pitch": search.kinematic_samples_per_pitch,
        "dynamic_candidate_count": search.dynamic_candidate_count,
        "local_refine_seed_count": search.local_refine_seed_count,
        "local_refine_per_seed": search.local_refine_per_seed,
        "final_candidate_count": search.final_candidate_count,
        "perturbations_per_final_candidate": search.perturbations_per_final_candidate,
        "fallback_kinematic_samples_per_pitch": (
            search.fallback_kinematic_samples_per_pitch
        ),
        "fallback_candidate_count": search.fallback_candidate_count,
    }

    robustness = OPPOSED_FACE_PALM_DOWN.robustness
    assert robustness.grid_case_count == 100
    assert robustness.perturbation_count == 50
    assert robustness.required_pass_count == 45
    assert robustness.seed == 20260821


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fallback_kinematic_samples_per_pitch", 0),
        ("fallback_candidate_count", 1024),
    ],
)
def test_v2_config_rejects_non_versioned_fallback_budgets(field, value):
    config = copy.deepcopy(_read(V2_CONFIG))
    config["search"]["budget"][field] = value

    with pytest.raises(ValueError, match="search.budget"):
        validate_config(config)


def test_v2_config_requires_explicit_fallback_budgets():
    config = copy.deepcopy(_read(V2_CONFIG))
    del config["search"]["budget"]["fallback_candidate_count"]

    with pytest.raises(ValueError, match="search.budget"):
        validate_config(config)
