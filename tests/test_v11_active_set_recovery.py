from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from xhand_grasp.artifacts import aggregate_source_sha256
from xhand_grasp.config import ACTIVE_ACTUATORS
from xhand_grasp.tuning.relative_wrist_pose_active_set_recovery import (
    _commit_stage,
    _load_committed_batch,
    _run_or_resume_fresh_full_scene_filter,
    _write_or_authenticate,
    boundary_release_initial_variables,
    build_recovery_base_manifest,
    build_recovery_run_manifest,
    candidate_has_complete_v11_safety,
    classify_deep_scene_contacts,
    fresh_full_scene_source_filter,
    full_scene_model_signature,
    normalized_gate_violations,
    recovery_gate_merit,
    recovery_stop_reason,
    select_recovery_batches,
    select_second_batch_for_failed_edges,
    validate_recovery_paths,
)


def _candidate(
    identifier: int,
    *,
    edge_mm: int = 85,
    orbit_deg: float = 0.0,
    gap_m: float = 0.00015,
    height_m: float = 0.0,
    safe: bool = True,
) -> dict:
    witnesses = {
        finger: {
            "signed_gap_m": gap_m,
            "normal_alignment": 1.0,
            "edge_margin_m": 0.002,
        }
        for finger in ("thumb", "index", "mid")
    }
    retreats = {
        finger: {
            "measured_outward_retreat_m": 0.003,
            "closure_angle_deg": 10.0,
        }
        for finger in ("thumb", "index", "mid")
    }
    return {
        "candidate_id": identifier,
        "edge_m": edge_mm / 1000.0,
        "clockwise_orbit_deg": orbit_deg,
        "static_metrics": {
            "target_witness": witnesses,
            "retreat_evidence": retreats,
            "contact_height_spread_m": height_m,
            "minimum_active_nondistal_gap_m": 0.001,
            "precontact_minimum_hand_gap_m": 0.001,
            "nominal_minimum_forbidden_hand_gap_m": 0.001,
            "nominal_maximum_all_distal_penetration_m": 0.001,
            "precontact_geometry_evaluated": safe,
            "off_target_distal_penetrating_count": 0,
            "missing_target_witness_count": 0,
            "cube_freejoint_qpos_unchanged": True,
        },
    }


def _fresh_candidate(identifier: int, *, edge_mm: int = 85) -> dict:
    value = _candidate(identifier, edge_mm=edge_mm)
    value["candidate_sha256"] = f"sha-{identifier}"
    value["static_metrics"]["nominal_joint_qpos_rad"] = [float(identifier)] * 8
    value["static_metrics"]["precontact_joint_qpos_rad"] = [
        float(identifier) + 0.25
    ] * 8
    value["config"] = {
        "schema_version": 11,
        "experiment_id": "v11-test",
        "side": "left",
        "cube": {
            "edge_m": edge_mm / 1000.0,
            "center_xy_m": [0.071, -0.027],
            "rpy_deg": [0.0, 0.0, 27.6],
            "mass_kg": 0.16,
            "friction": 0.8,
        },
        "scene": {
            "support_top_z_m": 0.084,
            "support_radius_m": 0.006,
            "floor_z_m": -0.015,
        },
        # Deliberately different: the filter must use static_metrics instead.
        "control": {"precontact_targets_rad": {"ignored": 99.0}},
    }
    return value


def test_safety_filter_fails_closed_before_merit() -> None:
    valid = _candidate(1)
    assert candidate_has_complete_v11_safety(valid)
    missing_proof = _candidate(2)
    del missing_proof["static_metrics"]["nominal_minimum_forbidden_hand_gap_m"]
    assert not candidate_has_complete_v11_safety(missing_proof)
    penetrated = _candidate(3)
    penetrated["static_metrics"]["nominal_maximum_all_distal_penetration_m"] = 0.00201
    assert not candidate_has_complete_v11_safety(penetrated)


def test_normalized_max_then_l2_merit() -> None:
    passed = _candidate(7)
    gap_near_miss = _candidate(8, gap_m=0.00075)
    height_near_miss = _candidate(9, height_m=0.0075)
    assert max(normalized_gate_violations(passed)) == 0.0
    # Three equal gap errors have a larger L2 merit than one height error.
    # The deterministic ID is the final tie break only after both merits.
    assert recovery_gate_merit(gap_near_miss) == (1.0, 3.0**0.5, 8)
    assert recovery_gate_merit(height_near_miss)[:2] == pytest.approx((0.5, 0.5))
    assert recovery_gate_merit(passed) < recovery_gate_merit(height_near_miss)


def test_two_edge_batches_are_orbit_covering_nonoverlapping_and_deterministic() -> None:
    records = []
    identifier = 100
    for edge in (85, 104):
        for orbit in (0.0, 2.5, 5.0):
            for repeat in range(3):
                records.append(
                    _candidate(
                        identifier,
                        edge_mm=edge,
                        orbit_deg=orbit,
                        height_m=repeat * 0.0001,
                    )
                )
                identifier += 1
    records.append(_candidate(999, safe=False))
    forward = select_recovery_batches(records, per_edge=4)
    reverse = select_recovery_batches(tuple(reversed(records)), per_edge=4)
    assert [value["candidate_id"] for value in forward.first] == [
        value["candidate_id"] for value in reverse.first
    ]
    assert [value["candidate_id"] for value in forward.second] == [
        value["candidate_id"] for value in reverse.second
    ]
    assert len(forward.first) == 8
    assert len(forward.second) == 8
    assert not (
        {value["candidate_id"] for value in forward.first}
        & {value["candidate_id"] for value in forward.second}
    )
    for edge in (0.085, 0.104):
        first_orbits = {
            value["clockwise_orbit_deg"]
            for value in forward.first
            if value["edge_m"] == edge
        }
        assert first_orbits == {0.0, 2.5, 5.0}
    assert forward.unsafe_candidate_count == 1


def test_second_batch_is_decided_independently_for_each_edge() -> None:
    second = (_candidate(1, edge_mm=85), _candidate(2, edge_mm=104))
    first_results = (
        {**_candidate(3, edge_mm=85), "static_pass": True},
        {**_candidate(4, edge_mm=104), "static_pass": False},
    )
    selected = select_second_batch_for_failed_edges(second, first_results)
    assert [value["candidate_id"] for value in selected] == [2]


def test_fresh_filter_is_deterministic_reuses_model_and_uses_recorded_qpos() -> None:
    compile_signatures: list[str] = []
    calls: list[tuple[int, tuple[float, ...], tuple[float, ...]]] = []

    class Gate:
        def __init__(self, config: dict) -> None:
            compile_signatures.append(full_scene_model_signature(config))

        def evaluate(
            self,
            config: dict,
            *,
            nominal_joint_qpos_rad: list[float],
            precontact_joint_qpos_rad: list[float],
        ) -> dict:
            identifier = int(nominal_joint_qpos_rad[0])
            calls.append(
                (
                    identifier,
                    tuple(nominal_joint_qpos_rad),
                    tuple(precontact_joint_qpos_rad),
                )
            )
            return {
                "passed": identifier != 2,
                "violations": [] if identifier != 2 else [{"phase": "nominal"}],
            }

    records = (_fresh_candidate(3, edge_mm=86), _fresh_candidate(2), _fresh_candidate(1))
    forward = fresh_full_scene_source_filter(records, gate_factory=Gate)
    reverse = fresh_full_scene_source_filter(tuple(reversed(records)), gate_factory=Gate)
    assert [value["candidate_id"] for value in forward.accepted] == [1, 3]
    assert [value["candidate_id"] for value in reverse.accepted] == [1, 3]
    assert forward.stored_v11_safe_count == 3
    assert forward.fresh_full_scene_safe_count == 2
    assert forward.model_compile_count == 2
    # Two independent runs compile two signatures each; candidates 1/2 share.
    assert len(compile_signatures) == 4
    first_call = next(value for value in calls if value[0] == 1)
    assert first_call[1] == (1.0,) * 8
    assert first_call[2] == (1.25,) * 8


def test_fresh_filter_resume_loads_committed_report_without_gate_call(
    tmp_path: Path,
) -> None:
    records = (_fresh_candidate(1), _fresh_candidate(2))
    calls = 0

    class Gate:
        def __init__(self, config: dict) -> None:
            nonlocal calls
            calls += 1

        def evaluate(self, config: dict, **kwargs: object) -> dict:
            return {"passed": True, "violations": []}

    first = _run_or_resume_fresh_full_scene_filter(
        records,
        tmp_path,
        recovery_input_sha256="base",
        resume=False,
        gate_factory=Gate,
    )
    assert calls == 1 and first.fresh_full_scene_safe_count == 2

    def forbidden_factory(config: dict) -> object:
        raise AssertionError("resume must not compile or reevaluate the gate")

    restored = _run_or_resume_fresh_full_scene_filter(
        records,
        tmp_path,
        recovery_input_sha256="base",
        resume=True,
        gate_factory=forbidden_factory,
    )
    assert restored.fresh_full_scene_safe_count == 2
    assert [value["candidate_id"] for value in restored.accepted] == [1, 2]


def test_fresh_filter_resume_rejects_report_tampering(tmp_path: Path) -> None:
    records = (_fresh_candidate(1),)

    class Gate:
        def __init__(self, config: dict) -> None:
            pass

        def evaluate(self, config: dict, **kwargs: object) -> dict:
            return {"passed": True, "violations": []}

    _run_or_resume_fresh_full_scene_filter(
        records,
        tmp_path,
        recovery_input_sha256="base",
        resume=False,
        gate_factory=Gate,
    )
    report = tmp_path / "static" / "fresh_full_scene_source_filter.json"
    report.write_text('{"complete": false}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="artifact changed"):
        _run_or_resume_fresh_full_scene_filter(
            records,
            tmp_path,
            recovery_input_sha256="base",
            resume=True,
            gate_factory=Gate,
        )


_REAL_V11_PARENT = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "left_opposed_face_palm_down_larger_relative_wrist_pose_actual_contact_smooth_vertical_lift"
    / "tune"
    / "campaign_6d_r2"
)


def _sealed_parent_matches_active_sources() -> bool:
    """Return whether the optional production fixture matches this checkout.

    The recovery authenticator deliberately rejects a campaign after any of
    its bound source files changes.  That is the correct production behavior,
    but it also means this optional, artifact-backed regression is only
    meaningful for the exact checkout that created the sealed campaign.
    Integrity rejection itself is covered by the synthetic authentication
    tests, so a later source edit should skip this historical count regression
    instead of turning an otherwise healthy suite red.
    """

    manifest_path = _REAL_V11_PARENT / "campaign_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_paths = [
            (
                Path(value).expanduser()
                if Path(value).expanduser().is_absolute()
                else Path(__file__).resolve().parents[1] / Path(value)
            ).resolve()
            for value in manifest["source_files"]
        ]
        return aggregate_source_sha256(source_paths) == str(
            manifest["source_sha256"]
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


@pytest.mark.skipif(
    not (_REAL_V11_PARENT / "campaign_result_target_1.json").is_file()
    or not _sealed_parent_matches_active_sources(),
    reason="sealed production v11 parent is unavailable or belongs to another source checkout",
)
def test_real_parent_fresh_full_scene_regression_is_1067_to_462() -> None:
    from xhand_grasp.tuning.relative_wrist_pose_recovery_auth import (
        authenticate_recovery_parent,
    )

    parent = authenticate_recovery_parent(_REAL_V11_PARENT)
    result = fresh_full_scene_source_filter(parent.candidates)
    assert result.input_candidate_count == 4200
    assert result.stored_v11_safe_count == 1067
    assert result.fresh_full_scene_safe_count == 462
    assert result.model_compile_count == 16


def test_output_must_not_be_parent_or_descendant(tmp_path: Path) -> None:
    parent = tmp_path / "campaign"
    parent.mkdir()
    with pytest.raises(ValueError, match="disjoint"):
        validate_recovery_paths(parent, parent)
    with pytest.raises(ValueError, match="disjoint"):
        validate_recovery_paths(parent, parent / "recovery")
    with pytest.raises(ValueError, match="disjoint"):
        validate_recovery_paths(parent, tmp_path)
    resolved_parent, output = validate_recovery_paths(parent, tmp_path / "recovery")
    assert resolved_parent == parent.resolve()
    assert output == (tmp_path / "recovery").resolve()


def test_full_scene_contact_filter_only_exempts_cube_support() -> None:
    contacts = (
        {"geom1_id": 62, "geom2_id": 1, "distance_m": -0.010},
        {
            "geom1_id": 20,
            "geom2_id": 31,
            "distance_m": -0.004717,
            "geom1_name": "index",
            "geom2_name": "middle",
        },
        {"geom1_id": 21, "geom2_id": 1, "distance_m": -0.002},
    )
    violations = classify_deep_scene_contacts(
        contacts, cube_geom_id=62, support_geom_id=1
    )
    assert len(violations) == 1
    assert violations[0]["geom1_name"] == "index"
    assert violations[0]["geom2_name"] == "middle"
    assert violations[0]["distance_m"] == pytest.approx(-0.004717)


def test_boundary_release_moves_bound_inward_and_prioritizes_same_finger() -> None:
    nominal = {name: 0.5 for name in ACTIVE_ACTUATORS}
    nominal["left_hand_index_bend_joint_actuator"] = -0.15
    config = {
        "grasp_pose": {"nominal_joint_qpos_rad": nominal},
        "candidate_metadata": {
            "relative_wrist_pose_search": {
                "root_delta_cube_m": [0.0, 0.0, 0.0],
                "wrist_local_rotvec_deg": [0.0, 0.0, 0.0],
            }
        },
    }
    non_thumb = [name for name in ACTIVE_ACTUATORS if "thumb_bend" not in name]
    bounds = {name: (-0.15, 1.8) for name in non_thumb}
    seeds = boundary_release_initial_variables(config, bounds)
    assert seeds[0][0] == "left_hand_index_bend_joint_actuator:inward_0.010"
    assert seeds[0][1].non_thumb_joint_qpos_rad[2] == pytest.approx(-0.14)
    assert "left_hand_index_joint1_actuator:-0.003" in seeds[1][0]


def test_solve_one_builds_one_context_shared_by_default_seed_and_fresh_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import xhand_grasp.tuning.relative_wrist_pose_active_set as active_set
    import xhand_grasp.tuning.relative_wrist_pose_active_set_recovery as recovery

    metrics = _candidate(1)["static_metrics"]
    metrics["nominal_joint_qpos_rad"] = [0.0] * 8
    metrics["precontact_joint_qpos_rad"] = [0.0] * 8

    class Static:
        static_geometry_pass = True
        nominal_joint_qpos_rad = (0.0,) * 8
        precontact_joint_qpos_rad = (0.0,) * 8

        def as_dict(self) -> dict:
            return metrics

    context = SimpleNamespace(joint_bounds={"joint": (0.0, 1.0)})
    build_calls = 0
    solve_contexts: list[object] = []
    fresh_contexts: list[object] = []

    def build_context(config: dict) -> object:
        nonlocal build_calls
        build_calls += 1
        return context

    def solve(config: dict, **kwargs: object) -> object:
        solve_contexts.append(kwargs["evaluation_context"])
        promoted = len(solve_contexts) > 1
        return SimpleNamespace(
            config={"candidate_metadata": {}},
            static_result=Static(),
            diagnostics={"promotion_config_valid": promoted},
            stop_reason="static_geometry_pass" if promoted else "near_miss",
        )

    def fresh(ctx: object, config: dict) -> tuple[object, dict]:
        fresh_contexts.append(ctx)
        # Keep the synthetic child non-promotable so no production config
        # validation is required in this orchestration-only test.
        return SimpleNamespace(safe=False, static_result=Static()), {
            "passed": True,
            "violations": [],
        }

    monkeypatch.setattr(active_set, "build_active_set_evaluation_context", build_context)
    monkeypatch.setattr(active_set, "solve_orientation_aware_active_set_dls", solve)
    monkeypatch.setattr(active_set, "evaluate_active_set_candidate", fresh)
    monkeypatch.setattr(
        recovery,
        "boundary_release_initial_variables",
        lambda config, bounds: (("release", object()),),
    )
    monkeypatch.setattr(recovery, "grasp_pose_id", lambda config: "pose")
    monkeypatch.setattr(recovery, "controller_id", lambda config: "controller")
    source = {
        "candidate_id": 10,
        "edge_m": 0.085,
        "clockwise_orbit_deg": 0.0,
        "config": {"candidate_metadata": {}},
    }
    result = recovery._solve_one((source, 1, 0))
    assert result["static_pass"] is False
    assert build_calls == 1
    assert solve_contexts == [context, context]
    assert fresh_contexts == [context]


def test_downstream_local_refinement_call_order_resume_and_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import xhand_grasp.experiment as experiment_module
    import xhand_grasp.tuning.actual_contact_grasp_pose as campaign_module
    import xhand_grasp.tuning.relative_wrist_pose_active_set_recovery as recovery

    calls: list[str] = []
    artifacts = {
        name: tmp_path / f"{name}.json"
        for name in ("base", "local_refinement", "local_dynamic")
    }
    for name, path in artifacts.items():
        path.write_text(json.dumps({"stage": name}), encoding="utf-8")

    base_records = (
        {"candidate_id": 1, "candidate_sha256": "base-1", "summary": {}},
        {"candidate_id": 2, "candidate_sha256": "base-2", "summary": {}},
    )
    local_candidates = (
        {"candidate_id": 11, "candidate_sha256": "local-static-1"},
    )
    local_records = (
        {"candidate_id": 21, "candidate_sha256": "local-dynamic-1"},
    )

    def base(*args: object, **kwargs: object) -> object:
        calls.append("base")
        return SimpleNamespace(
            records=base_records,
            artifacts=(artifacts["base"],),
            summary={"dynamic_candidate_count": 2, "grasp_success_count": 0},
        )

    def refine(*args: object, **kwargs: object) -> object:
        calls.append("refine")
        assert kwargs["top_count"] == 40
        assert kwargs["candidates_per_pose"] == 64
        return SimpleNamespace(
            records=local_candidates,
            artifacts=(artifacts["local_refinement"],),
            summary={
                "selected_pose_count": 2,
                "generated_candidate_count": 128,
                "dynamic_promoted_count": 1,
            },
        )

    def local_dynamic(*args: object, **kwargs: object) -> object:
        calls.append("local_dynamic")
        return SimpleNamespace(
            records=local_records,
            artifacts=(artifacts["local_dynamic"],),
            summary={"dynamic_candidate_count": 1, "grasp_success_count": 1},
        )

    def merge(first: object, second: object) -> object:
        calls.append("merge")
        return SimpleNamespace(
            records=(*first.records, *second.records),
            artifacts=(*first.artifacts, *second.artifacts),
            summary={"dynamic_candidate_count": 3, "grasp_success_count": 1},
        )

    definition = SimpleNamespace(
        actual_contact_grasp_pose_campaign=SimpleNamespace(
            local_pose_count=40, local_refine_per_pose=64
        ),
        relative_wrist_pose_search=object(),
    )
    monkeypatch.setattr(campaign_module, "_run_dynamic_stage", base)
    monkeypatch.setattr(
        campaign_module, "_run_joint_controller_local_refinement_stage", refine
    )
    monkeypatch.setattr(
        campaign_module, "_run_materialized_local_dynamic_stage", local_dynamic
    )
    monkeypatch.setattr(campaign_module, "_merge_dynamic_executions", merge)
    monkeypatch.setattr(experiment_module, "resolve_experiment", lambda config: definition)
    static = ({
        "static_pass": True,
        "recovery_static_promotable": True,
        "initial_contact_safety_pass": True,
        "config": {"experiment_id": "v11"},
    },)
    merged, summary = recovery._run_dynamic_with_optional_local_refinement(
        static,
        tmp_path,
        workers=3,
        seed=20260821,
        target_success_count=1,
        recovery_input_sha256="base-input",
    )
    assert calls == ["base", "refine", "local_dynamic", "merge"]
    assert len(merged.records) == 3
    assert summary["local_refinement_executed"] is True
    assert summary["base_dynamic_candidate_count"] == 2
    assert summary["local_refinement_selected_pose_count"] == 2
    assert summary["local_refinement_generated_candidate_count"] == 128
    assert summary["local_dynamic_candidate_count"] == 1
    assert summary["merged_grasp_success_count"] == 1

    # A second invocation exercises every private stage's resume path, while
    # our ledger/report commits remain idempotent and authenticated.
    recovery._run_dynamic_with_optional_local_refinement(
        static,
        tmp_path,
        workers=3,
        seed=20260821,
        target_success_count=1,
        recovery_input_sha256="base-input",
    )
    ledger = json.loads(
        (tmp_path / "recovery_stage_ledger.json").read_text(encoding="utf-8")
    )
    assert [value["name"] for value in ledger["stages"]] == [
        "recovery_dynamic",
        "recovery_local_refinement_decision_1",
        "recovery_joint_controller_local_refinement",
        "recovery_local_refinement_dynamic",
        "recovery_dynamic_merge_1",
    ]
    artifacts["local_dynamic"].write_text('{"tampered": true}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="artifact changed"):
        recovery._run_dynamic_with_optional_local_refinement(
            static,
            tmp_path,
            workers=3,
            seed=20260821,
            target_success_count=1,
            recovery_input_sha256="base-input",
        )


def test_downstream_skips_local_refinement_when_base_grasp_target_reached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import xhand_grasp.tuning.actual_contact_grasp_pose as campaign_module
    import xhand_grasp.tuning.relative_wrist_pose_active_set_recovery as recovery

    artifact = tmp_path / "base.json"
    artifact.write_text("{}", encoding="utf-8")
    base_record = {"candidate_id": 1, "candidate_sha256": "base", "summary": {}}
    monkeypatch.setattr(
        campaign_module,
        "_run_dynamic_stage",
        lambda *args, **kwargs: SimpleNamespace(
            records=(base_record,),
            artifacts=(artifact,),
            summary={"dynamic_candidate_count": 1, "grasp_success_count": 1},
        ),
    )
    monkeypatch.setattr(
        campaign_module,
        "_run_joint_controller_local_refinement_stage",
        lambda *args, **kwargs: pytest.fail("local refinement must be skipped"),
    )
    static = ({
        "static_pass": True,
        "recovery_static_promotable": True,
        "initial_contact_safety_pass": True,
        "config": {},
    },)
    merged, summary = recovery._run_dynamic_with_optional_local_refinement(
        static,
        tmp_path,
        workers=1,
        seed=20260821,
        target_success_count=1,
        recovery_input_sha256="base-input",
    )
    assert merged.records == (base_record,)
    assert summary["local_refinement_executed"] is False
    assert summary["decision_reason"] == "base_grasp_target_reached"
    assert summary["local_dynamic_candidate_count"] == 0


def test_downstream_rejects_static_source_without_fresh_safety(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import xhand_grasp.tuning.actual_contact_grasp_pose as campaign_module
    import xhand_grasp.tuning.relative_wrist_pose_active_set_recovery as recovery

    monkeypatch.setattr(
        campaign_module,
        "_run_dynamic_stage",
        lambda *args, **kwargs: pytest.fail("unsafe source must not run dynamics"),
    )
    with pytest.raises(RuntimeError, match="freshly safe static source"):
        recovery._run_dynamic_with_optional_local_refinement(
            ({"static_pass": True, "config": {}},),
            tmp_path,
            workers=1,
            seed=20260821,
            target_success_count=1,
            recovery_input_sha256="base-input",
        )


def test_atomic_ledger_binds_stage_input_and_artifact_hash(tmp_path: Path) -> None:
    output = tmp_path / "recovery"
    artifact = output / "static" / "report.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"complete": true}\n', encoding="utf-8")
    _commit_stage(
        output,
        "static",
        stage_input={"parent": "abc"},
        artifacts=(artifact,),
        summary={"count": 1},
    )
    # Exact resume is idempotent.
    _commit_stage(
        output,
        "static",
        stage_input={"parent": "abc"},
        artifacts=(artifact,),
        summary={"count": 1},
    )
    ledger = json.loads(
        (output / "recovery_stage_ledger.json").read_text(encoding="utf-8")
    )
    assert len(ledger["stages"]) == 1
    with pytest.raises(RuntimeError, match="resume input changed"):
        _commit_stage(
            output,
            "static",
            stage_input={"parent": "changed"},
            artifacts=(artifact,),
            summary={"count": 1},
        )
    artifact.write_text('{"complete": false}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="artifact changed"):
        _commit_stage(
            output,
            "another",
            stage_input={},
            artifacts=(artifact,),
            summary={},
        )


def test_worker_count_may_change_for_an_identical_committed_stage(tmp_path: Path) -> None:
    output = tmp_path / "recovery"
    artifact = output / "batch.json"
    artifact.parent.mkdir()
    artifact.write_text('{"complete": true}\n', encoding="utf-8")
    _commit_stage(
        output,
        "batch",
        stage_input={"parent": "abc", "workers": 12},
        artifacts=(artifact,),
        summary={},
    )
    _commit_stage(
        output,
        "batch",
        stage_input={"parent": "abc", "workers": 3},
        artifacts=(artifact,),
        summary={},
    )
    ledger = json.loads(
        (output / "recovery_stage_ledger.json").read_text(encoding="utf-8")
    )
    assert len(ledger["stages"]) == 1


def test_committed_batch_is_loaded_without_reexecution(tmp_path: Path) -> None:
    output = tmp_path / "recovery"
    path = output / "static" / "active_set_batch_1.json"
    path.parent.mkdir(parents=True)
    payload = {"complete": True, "batch": 1, "candidates": [{"candidate_id": 7}]}
    path.write_text(json.dumps(payload), encoding="utf-8")
    stage_input = {"parent": "abc", "workers": 12}
    _commit_stage(
        output,
        "active_set_batch_1",
        stage_input=stage_input,
        artifacts=(path,),
        summary={"candidate_count": 1},
    )
    loaded = _load_committed_batch(
        output,
        stage_name="active_set_batch_1",
        path=path,
        batch_index=1,
        stage_input={"parent": "abc", "workers": 2},
    )
    assert loaded == ({"candidate_id": 7},)


def test_parent_snapshot_is_always_authenticated_on_resume(tmp_path: Path) -> None:
    path = tmp_path / "parent_snapshot.json"
    _write_or_authenticate(path, {"snapshot_sha256": "a"}, resume=False)
    _write_or_authenticate(path, {"snapshot_sha256": "a"}, resume=True)
    with pytest.raises(RuntimeError, match="resume input changed"):
        _write_or_authenticate(path, {"snapshot_sha256": "b"}, resume=True)


def test_base_manifest_excludes_run_target_and_workers(tmp_path: Path) -> None:
    manifest = build_recovery_base_manifest(
        parent=tmp_path / "parent",
        experiment_id="v11",
        campaign_input_sha256="campaign",
        source_bundle_sha256="sources",
        parent_snapshot_sha256="snapshot",
        seed=20260821,
    )
    assert "workers" not in manifest
    assert "target_success_count" not in manifest
    assert manifest["relative_wrist_active_set_recovery_manifest_schema_version"] == 2
    assert manifest["source_bundle_sha256"] == "sources"
    assert "xhand_grasp.tuning.relative_wrist_pose_active_set" in manifest["source_code"]
    assert "xhand_grasp.tuning.actual_contact_grasp_pose" in manifest["source_code"]
    run_one = build_recovery_run_manifest(
        recovery_input_sha256=manifest["recovery_input_sha256"],
        target_success_count=1,
        workers=12,
    )
    run_five = build_recovery_run_manifest(
        recovery_input_sha256=manifest["recovery_input_sha256"],
        target_success_count=5,
        workers=3,
    )
    assert run_one["target_success_count"] == 1 and run_one["workers"] == 12
    assert run_five["target_success_count"] == 5 and run_five["workers"] == 3


@pytest.mark.parametrize(
    ("counts", "expected"),
    (
        ((0, 0, 0, 0), "active_set_static_budget_exhausted"),
        ((1, 0, 0, 0), "dynamic_grasp_not_verified"),
        ((1, 2, 0, 0), "measured_grasp_pose_not_verified"),
        ((1, 2, 1, 0), "manipulation_full_success_not_verified"),
        ((1, 2, 1, 1), "manipulation_full_success_verified"),
    ),
)
def test_stop_reason_distinguishes_evidence_stage(
    counts: tuple[int, int, int, int], expected: str
) -> None:
    assert recovery_stop_reason(
        static_pass_count=counts[0],
        grasp_success_count=counts[1],
        measured_grasp_pose_count=counts[2],
        full_success_count=counts[3],
    ) == expected
