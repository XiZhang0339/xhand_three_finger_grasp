from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from xhand_grasp.artifacts import write_json
from xhand_grasp.config import ACTIVE_ACTUATORS, load_config
from xhand_grasp.grasp_pose import controller_id, grasp_pose_id
from xhand_grasp.tuning.actual_contact_grasp_pose_dynamic import (
    CONTROLLER_SEED_COUNT,
    compact_dynamic_candidate_artifacts,
    dynamic_grasp_rank_evidence,
    expand_controller_candidates,
    generate_controller_seeds,
    grasp_stage_succeeded,
    materialize_controller_candidate,
    promotable_grasp_results,
    rank_dynamic_grasp_results,
    run_or_resume_dynamic_candidates,
)


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_actual_contact_grasp_pose_smooth_vertical_lift.json"
)


def _config() -> dict:
    return load_config(TEMPLATE)


def _source(identifier: int, config: dict | None = None) -> dict:
    return {"candidate_id": identifier, "config": _config() if config is None else config}


def _summary(
    *,
    success: bool,
    thumb: float = 1.50,
    error: float = 0.010,
    span: float = 0.010,
    translation: float = 0.0002,
    orientation: float = 0.3,
    closure: float = 15.0,
    force: float = 2.0,
    saturation: float = 0.02,
) -> dict:
    return {
        "stage_status": {
            "grasp_success": success,
            "manipulation_success": False,
            "full_success": False,
        },
        "metrics": {
            "actual_grasp_pose": {
                "passed": success,
                "metrics": {
                    "thumb_actual_median_rad": thumb,
                    "maximum_nominal_joint_error_rad": error,
                    "maximum_joint_stability_span_rad": span,
                },
            },
            "pose_preservation": {
                "max_translation_m": translation,
                "max_orientation_drift_deg": orientation,
            },
            "closure_alignment": {
                "close": {
                    "max_p95_angle_deg": closure,
                    "per_finger": {
                        name: {"angle_p95_deg": closure}
                        for name in ("thumb", "index", "mid")
                    },
                }
            },
            "peak_total_distal_contact_force_n": force,
            "actuator_saturation_fraction": saturation,
        },
        "checks": {},
        "passed": False,
        "failed_checks": [],
    }


def _executed(candidate: dict, summary: dict) -> dict:
    return {
        **{key: copy.deepcopy(value) for key, value in candidate.items() if key != "config"},
        "config": copy.deepcopy(candidate["config"]),
        "summary": summary,
        "grasp_success": bool(summary["stage_status"]["grasp_success"]),
    }


def test_eight_controller_seeds_are_deterministic_and_prefix_stable() -> None:
    config = _config()
    first = generate_controller_seeds(config, source_candidate_id=73)
    second = generate_controller_seeds(config, source_candidate_id=73)
    prefix = generate_controller_seeds(config, source_candidate_id=73, count=3)
    assert len(first) == CONTROLLER_SEED_COUNT == 8
    assert first == second
    assert prefix == first[:3]
    assert first[0].kind == "measured_sync_best_margin"
    assert {value.close_s for value in first}.issubset({1.0, 1.25, 1.5})


def test_controller_seed_never_relabels_nominal_actual_pose() -> None:
    config = _config()
    nominal = copy.deepcopy(config["grasp_pose"]["nominal_joint_qpos_rad"])
    pose_identifier = grasp_pose_id(config)
    records = [
        materialize_controller_candidate(
            _source(7, config), spec
        )
        for spec in generate_controller_seeds(config, source_candidate_id=7)
    ]
    assert len({record["candidate_id"] for record in records}) == 8
    assert len({record["controller_id"] for record in records}) == 8
    for record in records:
        resolved = record["config"]
        assert resolved["grasp_pose"]["nominal_joint_qpos_rad"] == nominal
        assert grasp_pose_id(resolved) == pose_identifier == record["grasp_pose_id"]
        assert controller_id(resolved) == record["controller_id"]
        assert all(
            resolved["control"]["manipulation_delta_rad"][name] == 0.0
            for name in ACTIVE_ACTUATORS
        )


def test_expansion_is_source_order_independent() -> None:
    sources = [_source(9), _source(3)]
    forward = expand_controller_candidates(sources, controller_seed_count=2)
    reverse = expand_controller_candidates(list(reversed(sources)), controller_seed_count=2)
    assert forward == reverse
    assert [record["candidate_id"] for record in forward] == [48, 49, 144, 145]


def test_rank_prioritizes_authoritative_grasp_then_requested_soft_order() -> None:
    config = _config()
    records = [
        {
            "candidate_id": 4,
            "config": config,
            "summary": _summary(success=False, thumb=1.50),
        },
        {
            "candidate_id": 3,
            "config": config,
            "summary": _summary(success=True, thumb=1.54, error=0.005),
        },
        {
            "candidate_id": 2,
            "config": config,
            "summary": _summary(success=True, thumb=1.50, error=0.020),
        },
        {
            "candidate_id": 1,
            "config": config,
            "summary": _summary(success=True, thumb=1.50, error=0.005),
        },
    ]
    ranked = rank_dynamic_grasp_results(reversed(records))
    assert [record["candidate_id"] for record in ranked] == [1, 2, 3, 4]
    evidence = dynamic_grasp_rank_evidence(ranked[0])
    assert evidence["thumb_distance_from_1p50_rad"] == pytest.approx(0.0)
    assert evidence["stability_min_normalized_margin"] > 0.0


def test_only_stage_status_grasp_success_can_advance() -> None:
    config = _config()
    false_stage = {
        "candidate_id": 1,
        "config": config,
        "grasp_success": True,
        "summary": _summary(success=False),
    }
    true_stage = {
        "candidate_id": 2,
        "config": config,
        "grasp_success": False,
        "summary": _summary(success=True),
    }
    assert not grasp_stage_succeeded(false_stage)
    assert grasp_stage_succeeded(true_stage)
    assert [value["candidate_id"] for value in promotable_grasp_results([false_stage, true_stage])] == [2]


def test_injected_executor_worker_order_cannot_change_result_order(tmp_path: Path) -> None:
    candidates = expand_controller_candidates(
        [_source(8), _source(2)], controller_seed_count=2
    )
    observed_workers: list[int] = []

    def reverse_executor(jobs, workers):
        observed_workers.append(workers)
        return tuple(
            _executed(dict(job), _summary(success=int(job["candidate_id"]) % 2 == 0))
            for job in reversed(jobs)
        )

    results = run_or_resume_dynamic_candidates(
        candidates,
        tmp_path,
        workers=3,
        executor=reverse_executor,
    )
    assert observed_workers == [3]
    assert [value["candidate_id"] for value in results] == sorted(
        value["candidate_id"] for value in candidates
    )


def test_atomic_artifacts_resume_and_failure_trace_compaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import xhand_grasp.simulation as simulation

    candidates = expand_controller_candidates([_source(5)], controller_seed_count=3)

    def fake_run(config, *, trace_path=None, video_path=None):
        assert video_path is None
        np.savez_compressed(trace_path, marker=np.asarray([1, 2, 3]))
        return _summary(success=False)

    monkeypatch.setattr(simulation, "run_simulation", fake_run)
    first = run_or_resume_dynamic_candidates(candidates, tmp_path, workers=1)
    assert all(not value["reused"] for value in first)
    for value in first:
        directory = tmp_path / value["artifact_directory"]
        assert {path.name for path in directory.iterdir()} == {
            "resolved_config.json",
            "result.json",
            "trace.npz",
        }
        assert value["actual_grasp_pose"]["metrics"]["thumb_actual_median_rad"] == 1.5

    compacted = compact_dynamic_candidate_artifacts(
        first, tmp_path, retain_failure_trace_count=1
    )
    assert sum(
        (tmp_path / value["artifact_directory"] / "trace.npz").is_file()
        for value in compacted
    ) == 1

    # Compaction is irreversible: a later subset ranking may ask to retain a
    # failure whose trace a previous global pass already removed.  It must
    # preserve the compacted state and digest instead of claiming that the
    # missing trace was restored.
    compacted_again = compact_dynamic_candidate_artifacts(
        compacted, tmp_path, retain_failure_trace_count=3
    )
    assert sum(
        (tmp_path / value["artifact_directory"] / "trace.npz").is_file()
        for value in compacted_again
    ) == 1
    for value in compacted_again:
        directory = tmp_path / value["artifact_directory"]
        payload = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        if not (directory / "trace.npz").is_file():
            assert payload["artifacts"]["trace_retained"] is False
            assert len(payload["artifacts"]["trace_sha256_at_evaluation"]) == 64

    def must_not_execute(jobs, workers):  # pragma: no cover - assertion path
        raise AssertionError("resume should authenticate compact records")

    resumed = run_or_resume_dynamic_candidates(
        candidates,
        tmp_path,
        workers=4,
        resume=True,
        executor=must_not_execute,
    )
    assert [value["candidate_id"] for value in resumed] == [
        value["candidate_id"] for value in first
    ]
    assert all(value["reused"] for value in resumed)

    tampered_path = tmp_path / resumed[0]["artifact_directory"] / "result.json"
    tampered = json.loads(tampered_path.read_text(encoding="utf-8"))
    tampered["summary"]["passed"] = True
    write_json(tampered_path, tampered)
    with pytest.raises(RuntimeError, match="semantic SHA-256 mismatch"):
        run_or_resume_dynamic_candidates(
            candidates,
            tmp_path,
            workers=1,
            resume=True,
            executor=must_not_execute,
        )


def test_invalid_seed_and_duplicate_source_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="count"):
        generate_controller_seeds(_config(), count=0)
    with pytest.raises(ValueError, match="unique"):
        expand_controller_candidates([_source(1), _source(1)], controller_seed_count=1)
