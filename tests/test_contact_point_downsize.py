from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from xhand_grasp.artifacts import file_sha256
from xhand_grasp.config import ACTIVE_ACTUATORS, load_config
from xhand_grasp.evaluation import face_from_label
from xhand_grasp.experiment import ScaledContactMappingParameters, resolve_experiment
from xhand_grasp.grasp_pose import canonical_sha256
from xhand_grasp.tuning import contact_point_downsize as downsize
from xhand_grasp.tuning import contact_point_targeted_search as targeting


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_larger_relative_wrist_pose_actual_contact_smooth_vertical_lift.json"
)
V13_CONFIG_PATH = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_scaled_centered_spread_actual_grasp_then_lift.json"
)


def _trace_payload(config: dict, *, count: int = 250) -> dict[str, np.ndarray]:
    local = np.asarray(
        (
            (-0.042, 0.004, 0.014),
            (0.042, -0.014, 0.011),
            (0.042, 0.008, 0.0105),
        ),
        dtype=np.float64,
    )
    cube_pos = np.tile(np.asarray((0.071, -0.027, 0.2)), (count, 1))
    cube_quat = np.tile(np.asarray((1.0, 0.0, 0.0, 0.0)), (count, 1))
    force = np.zeros((count, 3, 8), dtype=np.float64)
    moment = np.zeros((count, 3, 8, 3), dtype=np.float64)
    centroid = np.zeros((count, 3, 3), dtype=np.float64)
    valid = np.ones((count, 3), dtype=bool)
    for finger_index, finger in enumerate(("thumb", "index", "mid")):
        face_index = int(
            face_from_label(config["contact_topology"]["target_faces"][finger])
        )
        weights = 0.5 + 0.001 * np.arange(count) + 0.1 * finger_index
        world = cube_pos + local[finger_index]
        force[:, finger_index, face_index] = weights
        moment[:, finger_index, face_index] = weights[:, None] * world
        centroid[:, finger_index] = world
    return {
        "distal_face_force_n": force,
        "distal_face_position_moment_n_m": moment,
        "cube_pos": cube_pos,
        "cube_quat": cube_quat,
        "grasp_stable_window_start_step": np.asarray(0),
        "grasp_stable_window_end_step": np.asarray(count - 1),
        "target_face_contact_centroid_world_m": centroid,
        "target_face_contact_centroid_valid": valid,
    }


@pytest.fixture
def authenticated_source(tmp_path: Path):
    config = load_config(CONFIG_PATH)
    config_path = tmp_path / "config.json"
    result_path = tmp_path / "result.json"
    trace_path = tmp_path / "trace.npz"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    result_path.write_text(
        json.dumps(
            {
                "record": {
                    "hard_pass": True,
                    "strict_grasp_success": True,
                    "stable_window_start_step": 0,
                    "stable_window_end_step": 249,
                }
            }
        ),
        encoding="utf-8",
    )
    np.savez(trace_path, **_trace_payload(config))
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "aliases": {"best_nominal": "measured"},
                "trajectories": [
                    {
                        "trajectory_id": "measured",
                        "label": "measured",
                        "aliases": [],
                        "hard_pass": True,
                        "artifacts": {
                            "resolved_config": config_path.name,
                            "result": result_path.name,
                            "trace": trace_path.name,
                            "sha256": {
                                "resolved_config": file_sha256(config_path),
                                "result": file_sha256(result_path),
                                "trace": file_sha256(trace_path),
                            },
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return downsize.audit_downsize_source(catalog_path), catalog_path, trace_path


def test_schedule_is_exact_descending_prefix():
    values = downsize.descending_edge_schedule()
    assert len(values) == 29
    assert values[0] == pytest.approx(0.088)
    assert values[-1] == pytest.approx(0.060)
    assert np.diff(values) == pytest.approx(-0.001)
    with pytest.raises(ValueError, match="align"):
        downsize.descending_edge_schedule(maximum_edge_m=0.0885)


def test_mapping_modes_and_whole_target_circle_edge_guard():
    source = np.asarray(
        ((-0.0445, 0.004, 0.014), (0.0445, -0.014, 0.011), (0.0445, 0.008, 0.0105))
    )
    absolute = downsize.build_downsize_contact_point_plan(
        0.089, 0.060, source, mapping_mode="absolute"
    )
    proportional = downsize.build_downsize_contact_point_plan(
        0.089, 0.060, source, mapping_mode="proportional"
    )
    assert downsize.MAPPING_MODES == (
        "proportional_face_yz",
        "absolute_face_yz",
    )
    assert absolute.points["index"].y_m == pytest.approx(-0.014)
    assert proportional.points["index"].y_m == pytest.approx(-0.014 * 60 / 89)
    too_close = source.copy()
    too_close[0, 1] = 0.0275001
    with pytest.raises(ValueError, match=r"radius\+guard"):
        downsize.build_downsize_contact_point_plan(0.089, 0.060, too_close)


def test_force_weighted_window_requires_all_250_samples():
    config = load_config(CONFIG_PATH)
    trace = _trace_payload(config)
    evidence = downsize.extract_stable_window_force_weighted_centroids(trace, config)
    assert evidence.sample_count == 250
    assert evidence.valid_counts == (250, 250, 250)
    assert evidence.centroid_cube_local_m[0] == pytest.approx(
        (-0.042, 0.004, 0.014)
    )
    broken = copy.deepcopy(trace)
    face = int(face_from_label(config["contact_topology"]["target_faces"]["mid"]))
    broken["distal_face_force_n"][-1, 2, face] = 0.0
    with pytest.raises(ValueError, match="every finger"):
        downsize.extract_stable_window_force_weighted_centroids(broken, config)


def test_source_audit_hashes_trace_and_canonical_contact_evidence(
    authenticated_source, tmp_path: Path
):
    evidence, catalog_path, trace_path = authenticated_source
    payload = evidence.reference_contact_payload("best_nominal")
    assert set(payload) == {
        "source_alias",
        "config_sha256",
        "result_sha256",
        "trace_sha256",
        "stable_window_start_step",
        "stable_window_end_step",
        "reference_edge_m",
        "reference_target_face_yz_m",
    }
    plan = downsize.build_downsize_contact_point_plan(
        evidence.source_edge_m,
        0.080,
        evidence.stable_window.centroid_cube_local_m,
        mapping_mode="proportional",
    )
    mapping = evidence.scaled_contact_mapping(
        "proportional", plan, "best_nominal"
    )
    assert evidence.as_dict()["stable_window"]["sample_count"] == 250
    assert set(mapping) == {
        "source_alias",
        "mapping_mode",
        "schema_version",
        "config_sha256",
        "result_sha256",
        "trace_sha256",
        "stable_window_start_step",
        "stable_window_end_step",
        "reference_edge_m",
        "reference_contact_evidence_sha256",
        "reference_target_face_yz_m",
        "target_edge_m",
        "derived_target_face_yz_m",
        "contact_point_plan_id",
    }
    assert mapping["reference_contact_evidence_sha256"] == canonical_sha256(payload)
    parsed = ScaledContactMappingParameters.from_config(mapping)
    assert parsed.contact_point_plan_id == plan.point_plan_id

    with trace_path.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="trace SHA-256 mismatch"):
        downsize.audit_downsize_source(catalog_path)


def test_seed_mapping_preserves_cube_in_root_relation(authenticated_source):
    evidence, _, _ = authenticated_source
    mapped = downsize.map_downsize_seed_config(evidence.config, 0.080)
    assert mapped.config["cube"]["edge_m"] == pytest.approx(0.080)
    assert mapped.config["hand_pose"]["rpy_deg"] == evidence.config["hand_pose"][
        "rpy_deg"
    ]
    assert mapped.hand_translation_delta_world_m == pytest.approx((0.0, 0.0, -0.0025))


def test_previous_edge_continuation_uses_cube_relative_hand_pose(
    authenticated_source,
):
    evidence, _, _ = authenticated_source
    previous = downsize.map_downsize_seed_config(evidence.config, 0.081).config
    target = downsize.map_downsize_seed_config(evidence.config, 0.080).config
    variables = downsize.point_target_variables_from_previous_edge(target, previous)
    assert variables.root_delta_cube_m == pytest.approx((0.0, 0.0, 0.0))
    assert variables.wrist_local_rotvec_rad == pytest.approx((0.0, 0.0, 0.0))
    assert variables.actual_joint_qpos_rad == pytest.approx(
        tuple(
            previous["grasp_pose"]["nominal_joint_qpos_rad"][name]
            for name in ACTIVE_ACTUATORS
        )
    )


def test_v13_materialized_wrist_yaw_is_checked_beyond_local_rotvec_bounds():
    config = load_config(V13_CONFIG_PATH)
    plan = targeting.assert_frozen_contact_point_plan(config)
    policy = downsize._downsize_point_policy(config, plan, edge_guard_m=0.0005)
    variables = targeting.PointTargetVariables(
        tuple(
            float(config["grasp_pose"]["nominal_joint_qpos_rad"][name])
            for name in ACTIVE_ACTUATORS
        ),
        (-4.533966163047875e-05, 0.00042824451746866963, 0.0006579366024345778),
        tuple(
            np.radians((0.10303480990479583, 0.3242935990381595, -0.2384305380802731))
        ),
    )
    joint_bounds = resolve_experiment(config).search_bounds.actuator_targets_rad
    candidate = targeting.materialize_point_target_candidate(
        config, variables, signed_orbit_deg=0.0
    )
    assert candidate["hand_pose"]["rpy_deg"][2] == pytest.approx(
        15.0170086662682
    )
    reasons = targeting.point_target_boundary_violations(
        config,
        variables,
        policy,
        signed_orbit_deg=0.0,
        joint_bounds=joint_bounds,
    )
    assert "registered_search_bound_out_of_bounds:hand_yaw_deg" in reasons

    # Schema-v12 keeps its archived numerical boundary path unchanged.
    legacy = copy.deepcopy(config)
    legacy["schema_version"] = 12
    legacy_reasons = targeting.point_target_boundary_violations(
        legacy,
        variables,
        policy,
        signed_orbit_deg=0.0,
        joint_bounds=joint_bounds,
    )
    assert "registered_search_bound_out_of_bounds:hand_yaw_deg" not in legacy_reasons


def test_dls_materialization_binds_top_level_mapping_and_plan(
    authenticated_source, monkeypatch
):
    evidence, _, _ = authenticated_source
    captured = {}

    def fake_solve(base, **kwargs):
        captured["base"] = copy.deepcopy(base)
        captured["initial"] = kwargs["initial_variables"].as_array()
        return SimpleNamespace(acceptance=SimpleNamespace(passed=True))

    monkeypatch.setattr(downsize, "solve_point_target_dls", fake_solve)
    bounds = {name: (-2.0, 2.0) for name in ACTIVE_ACTUATORS}
    candidate = downsize.solve_downsize_static_candidate(
        evidence.config,
        0.080,
        evidence.stable_window.centroid_cube_local_m,
        source_id="best_nominal",
        source_evidence=evidence,
        mapping_mode="proportional",
        evaluator=lambda _: None,
        joint_bounds=bounds,
        start_index=1,
        max_iterations=4,
        check_pose_constraints=False,
    )
    mapping = captured["base"]["scaled_contact_mapping"]
    assert mapping == evidence.scaled_contact_mapping(
        "proportional_face_yz", candidate.point_plan, "best_nominal"
    )
    assert captured["base"]["contact_point_plan"]["point_plan_id"] == (
        candidate.point_plan.point_plan_id
    )
    assert not np.array_equal(
        captured["initial"][:8],
        np.asarray(
            [
                evidence.config["grasp_pose"]["nominal_joint_qpos_rad"][name]
                for name in ACTIVE_ACTUATORS
            ]
        ),
    )
