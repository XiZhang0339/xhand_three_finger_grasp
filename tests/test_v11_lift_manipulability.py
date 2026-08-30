from __future__ import annotations

import copy
import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import xhand_grasp.tuning.relative_wrist_pose_lift_manipulability as lift_module
from xhand_grasp.artifacts import file_sha256
from xhand_grasp.actual_contact_grasp_pose_catalog import (
    bind_candidate_result_semantic_sha256,
)
from xhand_grasp.config import ACTIVE_ACTUATORS, load_config, validate_config
from xhand_grasp.grasp_pose import canonical_sha256
from xhand_grasp.tuning.actual_contact_grasp_pose_dynamic import (
    expand_controller_candidates,
)
from xhand_grasp.tuning.relative_wrist_pose_lift_manipulability import (
    DIAGNOSTIC_CLOCKWISE_ORBITS_DEG,
    DIAGNOSTIC_ROOT_X_OFFSETS_M,
    DIAGNOSTIC_WRIST_LOCAL_ROTVECS_DEG,
    DEFAULT_EDGES_M,
    DEFAULT_ORBITS_DEG,
    FreeBodySearchHooks,
    LiftPhysicalScreen,
    LiftManipulabilityPolicy,
    MAX_STATIC_CANDIDATE_ID,
    MeasuredGraspSource,
    SupportModeObservation,
    dynamic_grasp_inputs_from_physical_screen,
    evaluate_support_mode_readiness,
    generate_exact_lift_diagnostic_grid,
    generate_lift_manipulability_candidates,
    load_measured_grasp_source,
    load_physical_screen_artifacts,
    physical_screen_lift_candidates,
    reauthenticate_measured_grasp_source,
    resize_measured_source,
    run_free_body_probe_search,
    run_or_resume_generation_campaign,
    run_supported_dynamic_grasp_screen,
    select_lift_candidates_with_edge_quota,
    write_physical_screen_artifacts,
)
from xhand_grasp.tuning.pose_preserving_seed_campaign import (
    canonical_sha256 as payload_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
V11_CONFIG = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_larger_relative_wrist_pose_actual_contact_"
    "smooth_vertical_lift.json"
)
REAL_V11_MEASURED = (
    ROOT
    / "artifacts"
    / "left_opposed_face_palm_down_larger_relative_wrist_pose_actual_contact_"
    "smooth_vertical_lift"
    / "tune"
    / "dynamic_centroid_recovery_v4_from_coordinate_v3"
    / "dynamic"
    / "measured"
    / "candidate_4337143107430657833"
)
REAL_V10_CATALOG = (
    ROOT
    / "artifacts"
    / "left_opposed_face_palm_down_larger_actual_contact_grasp_pose_"
    "smooth_vertical_lift"
    / "tune"
    / "campaign"
    / "catalogs"
    / "target_1"
    / "grasp_pose"
    / "catalog.json"
)


def _balanced_observation(*, height_spread_m: float = 0.001) -> SupportModeObservation:
    return SupportModeObservation(
        target_face_force_n=(2.0, 1.0, 1.0),
        contact_centroid_cube_m=(
            (-0.04, 0.0, 0.0),
            (0.04, -0.01, 0.0),
            (0.04, 0.01, 0.0),
        ),
        target_faces=("-X", "+X", "+X"),
        height_spread_p95_m=height_spread_m,
        tactile_nearest_distance_p95_m=(0.002, 0.002, 0.002),
        maximum_taxel_assignment_distance_m=0.006,
        support_retained=True,
    )


def _synthetic_source() -> MeasuredGraspSource:
    config = load_config(V11_CONFIG)
    actual = tuple(
        float(config["grasp_pose"]["nominal_joint_qpos_rad"][name])
        for name in ACTIVE_ACTUATORS
    )
    observation = _balanced_observation()
    readiness = evaluate_support_mode_readiness(
        observation,
        mass_kg=0.160,
        sliding_friction=0.8,
        gravity_m_s2=(0.0, 0.0, -9.81),
    )
    digests = {
        "config_sha256": "1" * 64,
        "result_sha256": "2" * 64,
        "trace_sha256": "3" * 64,
    }
    return MeasuredGraspSource(
        config_path="synthetic/resolved_config.json",
        result_path="synthetic/result.json",
        trace_path="synthetic/trace.npz",
        **digests,
        source_id=payload_sha256(digests),
        stable_window_start_step=100,
        stable_window_end_step=349,
        actual_joint_qpos_rad=actual,
        config=config,
        result={"summary": {"stage_status": {"grasp_success": True}}},
        observation=observation,
        readiness=readiness,
    )


def _tiny_policy() -> LiftManipulabilityPolicy:
    return LiftManipulabilityPolicy(
        edges_m=(0.087, 0.089),
        clockwise_orbits_deg=(0.0, 2.5),
        samples_per_cell=4,
        retained_per_cell=1,
        minimum_per_edge=2,
        selected_total=4,
        non_thumb_joint_radius_rad=0.005,
        root_radius_m=(0.0002, 0.0002, 0.0002),
        wrist_radius_deg=(0.1, 0.1, 0.1),
    )


def _eligible_static_screen() -> LiftPhysicalScreen:
    generated = generate_lift_manipulability_candidates(
        _synthetic_source(), policy=_tiny_policy()
    )
    proposal = copy.deepcopy(generated.records[0])

    class StaticResult:
        static_geometry_pass = True

        def as_dict(self) -> dict[str, object]:
            return {
                "static_geometry_pass": True,
                "missing_target_witness_count": 0,
                "off_target_distal_penetrating_count": 0,
                "minimum_active_nondistal_gap_m": 0.001,
                "contact_height_spread_m": 0.001,
                "nominal_minimum_forbidden_hand_gap_m": 0.001,
                "nominal_maximum_all_distal_penetration_m": 0.0001,
                "precontact_minimum_hand_gap_m": 0.001,
            }

    def evaluator_factory(config: object) -> tuple[object, dict[str, tuple[float, float]]]:
        return (lambda value: None), {
            name: (-2.0, 2.0) for name in ACTIVE_ACTUATORS if "thumb_bend" not in name
        }

    def solver(config: object, **kwargs: object) -> object:
        return SimpleNamespace(
            config=copy.deepcopy(config),
            variables=kwargs["initial_variables"],
            static_result=StaticResult(),
            diagnostics={"final_contact_objective": 0.0},
            stop_reason="test_static_pass",
        )

    return physical_screen_lift_candidates(
        [proposal], evaluator_factory=evaluator_factory, solver=solver
    )


def test_declared_grid_and_per_size_quota_are_versioned() -> None:
    policy = LiftManipulabilityPolicy()

    assert DEFAULT_EDGES_M == (0.087, 0.089, 0.091, 0.093)
    assert DEFAULT_ORBITS_DEG == (0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0)
    assert policy.minimum_per_edge == 8
    assert policy.selected_total == 32
    assert len(policy.edges_m) * len(policy.clockwise_orbits_deg) == 28
    with pytest.raises(ValueError, match="cannot satisfy every per-edge quota"):
        LiftManipulabilityPolicy(selected_total=31)
    with pytest.raises(ValueError, match="exceeds the retained candidate budget"):
        LiftManipulabilityPolicy(
            edges_m=(0.087,),
            clockwise_orbits_deg=(0.0,),
            samples_per_cell=1,
            retained_per_cell=1,
            minimum_per_edge=1,
            selected_total=2,
        )


def test_support_mode_readiness_combines_force_friction_moment_and_soft_height() -> None:
    ready = evaluate_support_mode_readiness(
        _balanced_observation(height_spread_m=0.0025),
        mass_kg=0.160,
        sliding_friction=0.8,
        gravity_m_s2=(0.0, 0.0, -9.81),
    )

    assert ready.passed
    assert ready.force_balance_ratio == pytest.approx(1.0)
    assert ready.friction_margin_n == pytest.approx(3.2 - 0.160 * 9.81)
    assert ready.line_of_action_moment_n_m == pytest.approx(0.0)
    assert ready.height_soft_excess_m == pytest.approx(0.0005)
    assert ready.tactile_path_margin_m == pytest.approx(0.004)
    assert ready.as_dict()["policy"]["hard_height_spread_m"] == pytest.approx(
        0.005
    )
    assert ready.as_dict()["policy"]["minimum_tactile_path_margin_m"] == 0.0

    unbalanced = replace(
        _balanced_observation(),
        target_face_force_n=(0.5, 1.0, 1.0),
    )
    failed = evaluate_support_mode_readiness(
        unbalanced,
        mass_kg=0.160,
        sliding_friction=0.8,
        gravity_m_s2=(0.0, 0.0, -9.81),
    )
    assert not failed.passed
    assert not failed.checks["opposed_normal_force_balanced"]
    assert not failed.checks["friction_lift_margin"]


@pytest.mark.skipif(
    not (REAL_V11_MEASURED / "trace.npz").is_file(),
    reason="sealed v11 measured trace is unavailable",
)
def test_real_measured_source_reports_the_power_recovery_failure_mode() -> None:
    source = load_measured_grasp_source(
        measured_config_path=REAL_V11_MEASURED / "resolved_config.json"
    )

    assert source.stable_window_end_step - source.stable_window_start_step + 1 == 250
    assert source.observation.target_face_force_n == pytest.approx(
        (0.6048276003, 0.9823778112, 0.7928956589)
    )
    assert source.readiness.force_balance_ratio == pytest.approx(0.3406954537)
    assert source.readiness.friction_margin_n == pytest.approx(-0.6018758395)
    assert source.readiness.line_of_action_moment_n_m == pytest.approx(
        0.01300212019
    )
    assert not source.readiness.passed

    source.config["cube"]["friction"] = 0.81
    with pytest.raises(RuntimeError, match="config mutated"):
        reauthenticate_measured_grasp_source(source)


@pytest.mark.skipif(
    not REAL_V10_CATALOG.is_file(),
    reason="sealed v10 trajectory catalog is unavailable",
)
def test_catalog_alias_loads_and_hash_tampering_is_rejected(tmp_path: Path) -> None:
    source = load_measured_grasp_source(
        catalog_path=REAL_V10_CATALOG,
        trajectory="best_first",
    )
    assert source.catalog_path == str(REAL_V10_CATALOG.resolve())
    assert source.trajectory_id is not None
    assert source.config["schema_version"] == 10

    original_catalog = json.loads(REAL_V10_CATALOG.read_text(encoding="utf-8"))
    trajectory_id = original_catalog["aliases"]["best_first"]
    record = next(
        value
        for value in original_catalog["trajectories"]
        if value["trajectory_id"] == trajectory_id
    )
    copied = tmp_path / "trajectory"
    copied.mkdir()
    source_base = REAL_V10_CATALOG.parent
    copied_paths = {}
    for name in ("resolved_config", "result", "trace"):
        source_path = source_base / record["artifacts"][name]
        target = copied / source_path.name
        shutil.copy2(source_path, target)
        copied_paths[name] = target
    catalog = {
        "complete": True,
        "experiment_id": original_catalog["experiment_id"],
        "aliases": {"best_nominal": "copied"},
        "trajectories": [
            {
                "trajectory_id": "copied",
                "grasp_success": True,
                "artifacts": {
                    **{name: f"trajectory/{path.name}" for name, path in copied_paths.items()},
                    "sha256": {
                        name: file_sha256(path) for name, path in copied_paths.items()
                    },
                },
            }
        ],
    }
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    copied_paths["resolved_config"].write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="catalog member SHA-256 mismatch"):
        load_measured_grasp_source(catalog_path=catalog_path)


def test_resize_and_generation_cover_all_relative_wrist_variables_deterministically() -> None:
    source = _synthetic_source()
    original = copy.deepcopy(source.config)
    resized = resize_measured_source(source, 0.093)

    assert source.config == original
    assert resized["cube"]["edge_m"] == pytest.approx(0.093)
    assert resized["cube"]["mass_kg"] == pytest.approx(0.160)
    assert resized["cube"]["center_xy_m"] == original["cube"]["center_xy_m"]
    assert resized["hand_pose"]["translation_m"][2] == pytest.approx(
        original["hand_pose"]["translation_m"][2] + 0.004
    )

    policy = _tiny_policy()
    first = generate_lift_manipulability_candidates(source, policy=policy)
    second = generate_lift_manipulability_candidates(source, policy=policy)
    assert first.report["declared_cell_count"] == 4
    assert first.report["declared_sample_count"] == 16
    assert [value["candidate_id"] for value in first.records] == [
        value["candidate_id"] for value in second.records
    ]
    assert [value["candidate_sha256"] for value in first.records] == [
        value["candidate_sha256"] for value in second.records
    ]
    assert {(value["edge_m"], value["clockwise_orbit_deg"]) for value in first.records} == {
        (0.087, 0.0),
        (0.087, 2.5),
        (0.089, 0.0),
        (0.089, 2.5),
    }
    vectors = [
        value["config"]["candidate_metadata"]["lift_manipulability_search"][
            "variables"
        ]
        for value in first.records
    ]
    assert all(len(vector) == 13 for vector in vectors)
    assert any(vector[:7] != vectors[0][:7] for vector in vectors[1:])
    assert any(vector[7:10] != vectors[0][7:10] for vector in vectors[1:])
    assert any(vector[10:13] != vectors[0][10:13] for vector in vectors[1:])
    for record in first.records:
        config = record["config"]
        validate_config(config)
        assert config["cube"]["mass_kg"] == pytest.approx(0.160)
        assert config["candidate_metadata"]["lift_manipulability_search"][
            "cube_pose_sampled"
        ] is False
        assert config["grasp_pose"]["nominal_joint_qpos_rad"][
            "left_hand_thumb_bend_joint_actuator"
        ] == pytest.approx(source.actual_joint_qpos_rad[0])

    selected = select_lift_candidates_with_edge_quota(
        first.records,
        edges_m=policy.edges_m,
        minimum_per_edge=policy.minimum_per_edge,
        selected_total=policy.selected_total,
    )
    assert len(selected) == 4
    assert {value["edge_m"] for value in selected} == {0.087, 0.089}


def test_v10_catalog_source_is_promoted_only_in_generated_candidates() -> None:
    if not REAL_V10_CATALOG.is_file():
        pytest.skip("sealed v10 trajectory catalog is unavailable")
    source = load_measured_grasp_source(
        catalog_path=REAL_V10_CATALOG,
        trajectory="best_first",
    )
    policy = LiftManipulabilityPolicy(
        edges_m=(0.087,),
        clockwise_orbits_deg=(0.0,),
        samples_per_cell=1,
        retained_per_cell=1,
        minimum_per_edge=1,
        selected_total=1,
        non_thumb_joint_radius_rad=0.001,
        root_radius_m=(0.0001, 0.0001, 0.0001),
        wrist_radius_deg=(0.05, 0.05, 0.05),
    )

    generated = generate_lift_manipulability_candidates(source, policy=policy)

    assert source.config["schema_version"] == 10
    candidate = generated.records[0]["config"]
    assert candidate["schema_version"] == 11
    assert candidate["candidate_metadata"]["source_schema_version"] == 10
    assert candidate["candidate_metadata"]["promoted_from_measured_source_id"] == (
        source.source_id
    )


@pytest.mark.skipif(
    not (REAL_V11_MEASURED / "trace.npz").is_file(),
    reason="sealed v11 measured trace is unavailable",
)
def test_exact_27_point_grid_is_deterministic_and_uses_safe_static_ids() -> None:
    source = load_measured_grasp_source(
        measured_config_path=REAL_V11_MEASURED / "resolved_config.json"
    )
    first = generate_exact_lift_diagnostic_grid(source)
    second = generate_exact_lift_diagnostic_grid(source)

    assert len(first.records) == 27
    assert [value["candidate_id"] for value in first.records] == [
        value["candidate_id"] for value in second.records
    ]
    assert {value["clockwise_orbit_deg"] for value in first.records} == set(
        DIAGNOSTIC_CLOCKWISE_ORBITS_DEG
    )
    searches = [
        value["config"]["candidate_metadata"]["lift_manipulability_search"]
        for value in first.records
    ]
    assert {value["root_x_offset_from_source_m"] for value in searches} == set(
        DIAGNOSTIC_ROOT_X_OFFSETS_M
    )
    assert {tuple(value["wrist_local_rotvec_deg"]) for value in searches} == set(
        DIAGNOSTIC_WRIST_LOCAL_ROTVECS_DEG
    )
    assert all(value["solver_clockwise_orbit_deg"] == 0.0 for value in first.records)
    assert sum(bool(value["configuration_valid"]) for value in first.records) == 9
    assert max(value["candidate_id"] for value in first.records) <= MAX_STATIC_CANDIDATE_ID
    assert (16 * MAX_STATIC_CANDIDATE_ID + 15) * 1000 + 999 <= 2**63 - 1


@pytest.mark.skipif(
    not (REAL_V11_MEASURED / "trace.npz").is_file(),
    reason="sealed v11 measured trace is unavailable",
)
def test_physical_screen_is_not_grasp_evidence_and_round_trips(tmp_path: Path) -> None:
    source = load_measured_grasp_source(
        measured_config_path=REAL_V11_MEASURED / "resolved_config.json"
    )
    proposal = next(
        value
        for value in generate_exact_lift_diagnostic_grid(source).records
        if value["configuration_valid"]
    )
    derived_precontact: dict[str, float] = {}

    class StaticResult:
        static_geometry_pass = True

        def as_dict(self) -> dict[str, object]:
            return {
                "static_geometry_pass": True,
                "missing_target_witness_count": 0,
                "off_target_distal_penetrating_count": 0,
                "minimum_active_nondistal_gap_m": 0.001,
                "contact_height_spread_m": 0.001,
                "nominal_minimum_forbidden_hand_gap_m": 0.001,
                "nominal_maximum_all_distal_penetration_m": 0.0001,
                "precontact_minimum_hand_gap_m": 0.001,
            }

    def evaluator_factory(config: object) -> tuple[object, dict[str, tuple[float, float]]]:
        return (lambda value: None), {
            name: (-2.0, 2.0) for name in ACTIVE_ACTUATORS if "thumb_bend" not in name
        }

    def solver(config: object, **kwargs: object) -> object:
        refined = copy.deepcopy(config)
        name = ACTIVE_ACTUATORS[1]
        refined["control"]["precontact_targets_rad"][name] += 0.001
        derived_precontact[name] = refined["control"]["precontact_targets_rad"][name]
        return SimpleNamespace(
            config=refined,
            variables=kwargs["initial_variables"],
            static_result=StaticResult(),
            diagnostics={"final_contact_objective": 0.0},
            stop_reason="test_static_pass",
        )

    screen = physical_screen_lift_candidates(
        [proposal], evaluator_factory=evaluator_factory, solver=solver
    )
    record = screen.records[0]
    assert record["static_geometry_pass"]
    assert record["eligible_for_dynamic_grasp"]
    assert not record["grasp_success"]
    assert not record["static_is_grasp_success_evidence"]
    changed_name = ACTIVE_ACTUATORS[1]
    assert record["config"]["control"]["precontact_targets_rad"][changed_name] == pytest.approx(
        derived_precontact[changed_name]
    )
    inputs = dynamic_grasp_inputs_from_physical_screen(screen)
    assert inputs[0]["candidate_id"] == record["candidate_id"]
    assert inputs[0]["lift_candidate_sha256"] == record["lift_candidate_sha256"]

    manifest = write_physical_screen_artifacts(
        screen, tmp_path / "physical", input_sha256="a" * 64
    )
    assert manifest["dynamic_eligible_count"] == 1
    loaded = load_physical_screen_artifacts(
        tmp_path / "physical", expected_input_sha256="a" * 64
    )
    assert loaded.records[0]["candidate_sha256"] == record["candidate_sha256"]
    assert not loaded.records[0]["grasp_success"]


def test_physical_screen_rejects_stale_hash_mixed_sources_and_model_drift() -> None:
    source = _synthetic_source()
    policy = LiftManipulabilityPolicy(
        edges_m=(0.087,),
        clockwise_orbits_deg=(0.0,),
        samples_per_cell=4,
        retained_per_cell=1,
        minimum_per_edge=1,
        selected_total=1,
        non_thumb_joint_radius_rad=0.005,
        root_radius_m=(0.0002, 0.0002, 0.0002),
        wrist_radius_deg=(0.1, 0.1, 0.1),
    )
    base = copy.deepcopy(
        generate_lift_manipulability_candidates(source, policy=policy).records[0]
    )
    clone = copy.deepcopy(base)
    clone["candidate_id"] += 1
    clone["config"]["candidate_metadata"]["candidate_id"] = clone["candidate_id"]
    clone["config"]["candidate_metadata"]["lift_manipulability_search"][
        "static_candidate_id"
    ] = clone["candidate_id"]
    clone["candidate_sha256"] = canonical_sha256(clone["config"])
    records = [base, clone]

    stale = copy.deepcopy(records[0])
    stale["config"]["cube"]["friction"] = 0.81
    with pytest.raises(RuntimeError, match="config SHA-256 mismatch"):
        physical_screen_lift_candidates([stale])

    mixed = copy.deepcopy(records)
    foreign_source = "f" * 64
    mixed[1]["source_id"] = foreign_source
    mixed[1]["config"]["candidate_metadata"][
        "source_measured_grasp_id"
    ] = foreign_source
    mixed[1]["candidate_sha256"] = canonical_sha256(mixed[1]["config"])
    with pytest.raises(RuntimeError, match="one measured source"):
        physical_screen_lift_candidates(mixed)

    changed_model = copy.deepcopy(records)
    changed_model[1]["config"]["scene"]["support_radius_m"] += 0.001
    changed_model[1]["candidate_sha256"] = canonical_sha256(
        changed_model[1]["config"]
    )
    with pytest.raises(RuntimeError, match="changed evaluator model inputs"):
        physical_screen_lift_candidates(changed_model)


def test_dynamic_runner_outputs_are_rebound_and_paths_are_confined(
    tmp_path: Path,
) -> None:
    screen = _eligible_static_screen()
    forged = copy.deepcopy(
        generate_lift_manipulability_candidates(
            _synthetic_source(), policy=_tiny_policy()
        ).records[0]
    )
    forged["eligible_for_dynamic_grasp"] = True
    with pytest.raises(RuntimeError, match="physical collision-witness pass"):
        dynamic_grasp_inputs_from_physical_screen(
            LiftPhysicalScreen((forged,), {"complete": True})
        )

    def records_for(
        inputs: object, output: object, **kwargs: object
    ) -> list[dict[str, object]]:
        jobs = expand_controller_candidates(
            inputs, controller_seed_count=int(kwargs["controller_seed_count"])
        )
        records: list[dict[str, object]] = []
        for job in jobs:
            relative = f"candidates/candidate_{int(job['candidate_id'])}"
            directory = Path(output) / relative
            directory.mkdir(parents=True, exist_ok=True)
            config_path = directory / "resolved_config.json"
            config_path.write_text(
                json.dumps(job["config"], sort_keys=True), encoding="utf-8"
            )
            summary = {"stage_status": {"grasp_success": False}}
            payload = bind_candidate_result_semantic_sha256(
                {
                    "candidate_result_schema_version": 1,
                    "complete": True,
                    "campaign_kind": "actual_contact_grasp_pose_dynamic_acquisition",
                    "stage": "dynamic_grasp_acquisition",
                    "candidate_id": int(job["candidate_id"]),
                    "source_candidate_id": int(job["source_candidate_id"]),
                    "controller_seed_index": int(job["controller_seed_index"]),
                    "candidate_sha256": job["candidate_sha256"],
                    "grasp_pose_id": job["grasp_pose_id"],
                    "controller_id": job["controller_id"],
                    "grasp_success": False,
                    "classification": "actual_contact_grasp_pose_near_miss",
                    "rank_evidence": {"grasp_success": False},
                    "actual_grasp_pose": {},
                    "summary": summary,
                    "artifacts": {
                        "resolved_config": "resolved_config.json",
                        "trace": None,
                        "trace_retained": False,
                        "trace_sha256_at_evaluation": "d" * 64,
                        "sha256": {
                            "resolved_config": file_sha256(config_path),
                        },
                    },
                }
            )
            (directory / "result.json").write_text(
                json.dumps(payload, sort_keys=True), encoding="utf-8"
            )
            records.append(
                {
                    **copy.deepcopy(job),
                    **payload,
                    "config": copy.deepcopy(job["config"]),
                    "artifact_directory": relative,
                }
            )
        return records

    def good_runner(inputs: object, output: object, **kwargs: object) -> object:
        return records_for(inputs, output, **kwargs)

    result = run_supported_dynamic_grasp_screen(
        screen,
        tmp_path / "good",
        controller_seed_count=1,
        dynamic_runner=good_runner,
    )
    assert result.report["dynamic_candidate_count"] == 1
    assert result.report["support_mode_ready_count"] == 0
    assert not result.report["production_dynamic_runner_attested"]
    assert "hard_height_spread_m" in result.report["hard_thresholds"]
    assert "minimum_tactile_path_margin_m" in result.report["hard_thresholds"]

    def traversing_runner(inputs: object, output: object, **kwargs: object) -> object:
        values = records_for(inputs, output, **kwargs)
        values[0]["artifact_directory"] = "../../sealed_grasp"
        return values

    with pytest.raises(RuntimeError, match="confined relative path"):
        run_supported_dynamic_grasp_screen(
            screen,
            tmp_path / "traversal",
            controller_seed_count=1,
            dynamic_runner=traversing_runner,
        )

    def rebound_runner(inputs: object, output: object, **kwargs: object) -> object:
        values = records_for(inputs, output, **kwargs)
        values[0]["config"]["control"]["contact_preload_targets_rad"][
            ACTIVE_ACTUATORS[0]
        ] += 0.01
        return values

    with pytest.raises(RuntimeError, match="different candidate config"):
        run_supported_dynamic_grasp_screen(
            screen,
            tmp_path / "rebound",
            controller_seed_count=1,
            dynamic_runner=rebound_runner,
        )

    def swapped_artifact_runner(
        inputs: object, output: object, **kwargs: object
    ) -> object:
        values = records_for(inputs, output, **kwargs)
        directory = Path(output) / str(values[0]["artifact_directory"])
        unrelated = copy.deepcopy(values[0]["config"])
        unrelated["control"]["contact_preload_targets_rad"][
            ACTIVE_ACTUATORS[0]
        ] += 0.01
        (directory / "resolved_config.json").write_text(
            json.dumps(unrelated, sort_keys=True), encoding="utf-8"
        )
        return values

    with pytest.raises(RuntimeError, match="persisted dynamic config"):
        run_supported_dynamic_grasp_screen(
            screen,
            tmp_path / "artifact_swap",
            controller_seed_count=1,
            dynamic_runner=swapped_artifact_runner,
        )

    def foreign_runner(inputs: object, output: object, **kwargs: object) -> object:
        values = records_for(inputs, output, **kwargs)
        values[0]["source_candidate_id"] += 1
        return values

    with pytest.raises(RuntimeError, match="source candidate ID"):
        run_supported_dynamic_grasp_screen(
            screen,
            tmp_path / "foreign",
            controller_seed_count=1,
            dynamic_runner=foreign_runner,
        )


def test_free_body_hook_runs_probes_then_full_resets(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    checkpoint = SimpleNamespace(model="model")

    def prepare(config: object, trace: object, result: object) -> object:
        calls.append(("prepare", trace))
        return checkpoint

    def bounds(model: object, config: object) -> dict[str, tuple[float, float]]:
        calls.append(("bounds", model))
        return {name: (-0.1, 0.1) for name in ACTIVE_ACTUATORS}

    def probes(value: object, *, budget: object) -> tuple[dict, ...]:
        calls.append(("probes", value))
        return tuple({"probe": index} for index in range(17))

    def fit(values: object, limits: object, **kwargs: object) -> dict:
        calls.append(("fit", len(values), kwargs["target_response_6d"]))
        return {"solution_delta_rad": {name: 0.0 for name in ACTIVE_ACTUATORS}}

    def deltas(response: object, limits: object, **kwargs: object) -> tuple[dict, ...]:
        calls.append(("deltas", kwargs["count"]))
        return tuple(
            {name: 0.001 * index for name in ACTIVE_ACTUATORS}
            for index in range(2)
        )

    def full(config: object, values: object, **kwargs: object) -> dict:
        calls.append(("full", kwargs["workers"], tuple(kwargs["candidate_ids"])))
        return {"candidate_count": len(values)}

    budget = SimpleNamespace(
        target_upward_m=0.011,
        ridge=1e-4,
        inward_preload_weight=2e-3,
        trust_candidate_count=2,
        seed=20260821,
        trust_radius_fraction=0.15,
        wide_candidate_fraction=0.4,
        wide_radius_fraction=0.45,
    )
    authenticated: list[str] = []

    def reauthenticate(source: MeasuredGraspSource) -> MeasuredGraspSource:
        authenticated.append(source.source_id)
        return source

    monkeypatch.setattr(
        lift_module, "reauthenticate_measured_grasp_source", reauthenticate
    )
    result = run_free_body_probe_search(
        _synthetic_source(),
        budget=budget,
        workers=3,
        hooks=FreeBodySearchHooks(prepare, bounds, probes, fit, deltas, full),
    )

    assert result["checkpoint_search_only"]
    assert not result["final_candidates_rerun_from_initial_state"]
    assert not result["execution_attested"]
    assert result["probe_count"] == 17
    assert result["candidate_count"] == 2
    assert authenticated == [_synthetic_source().source_id]
    assert calls[-1] == ("full", 3, (0, 1))


def test_atomic_campaign_resumes_a_committed_prefix_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    source = _synthetic_source()
    policy = _tiny_policy()
    output = tmp_path / "campaign"
    complete = run_or_resume_generation_campaign(source, output, policy=policy)
    assert complete.manifest["complete"]
    assert complete.manifest["input"]["model"]["sha256"] == file_sha256(
        ROOT / "xhand_left.xml"
    )
    assert complete.manifest["input"]["uv_lock"]["sha256"] == file_sha256(
        ROOT / "uv.lock"
    )
    assert [value["name"] for value in complete.ledger["stages"]] == [
        "authenticated_source",
        "candidate_generation",
        "per_edge_selection",
        "complete_manifest",
    ]

    # Simulate power loss immediately after the generation stage commit.  The
    # selection file may exist, but it is not trusted until re-bound.
    ledger_path = output / "campaign_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["stages"] = ledger["stages"][:2]
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    manifest_path = output / "campaign_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["complete"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    resumed = run_or_resume_generation_campaign(
        source,
        output,
        policy=policy,
        resume=True,
    )
    assert resumed.manifest["complete"]
    assert [value["name"] for value in resumed.ledger["stages"]] == [
        "authenticated_source",
        "candidate_generation",
        "per_edge_selection",
        "complete_manifest",
    ]
    assert [value["candidate_sha256"] for value in resumed.selected_records] == [
        value["candidate_sha256"] for value in complete.selected_records
    ]

    retained_path = output / "static" / "retained_candidates.json"
    retained_path.write_text(
        retained_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="ledger artifact changed"):
        run_or_resume_generation_campaign(
            source,
            output,
            policy=policy,
            resume=True,
        )
