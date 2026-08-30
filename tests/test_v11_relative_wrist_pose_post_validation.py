from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from xhand_grasp.actual_contact_grasp_pose_catalog import (
    bind_candidate_result_semantic_sha256,
)
from xhand_grasp.artifacts import file_sha256, write_json
from xhand_grasp.config import load_config, validate_config
from xhand_grasp.tuning.relative_wrist_pose_post_validation import (
    load_canonical_fixed_160g_sources,
    run_relative_wrist_pose_post_validation,
)
from xhand_grasp.tuning.relative_wrist_pose_validation import (
    constant_density_mass_kg,
    materialize_constant_density_revalidation,
)
from xhand_grasp.viewer import resolve_viewer_source


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "grasp_configs" / (
    "left_opposed_face_palm_down_larger_relative_wrist_pose_actual_contact_"
    "smooth_vertical_lift.json"
)


def _summary(passed: bool) -> dict:
    return {
        "passed": passed,
        "failed_checks": [] if passed else ["synthetic_failure"],
        "stage_status": {
            "grasp_success": passed,
            "manipulation_success": passed,
            "full_success": passed,
        },
    }


def _source_catalog(tmp_path: Path, *, candidate_id: int = 811) -> Path:
    catalog_root = tmp_path / "source_catalog"
    member = catalog_root / f"manipulation_01_{candidate_id}"
    member.mkdir(parents=True)
    config = load_config(TEMPLATE)
    config["cube"]["edge_m"] = 0.085
    validate_config(config)
    config_path = member / "resolved_config.json"
    trace_path = member / "trace.npz"
    result_path = member / "result.json"
    write_json(config_path, config)
    np.savez_compressed(trace_path, time_s=np.array([0.0, 0.001]))
    result = bind_candidate_result_semantic_sha256(
        {
            "actual_contact_manipulation_candidate_schema_version": 1,
            "complete": True,
            "candidate_id": candidate_id,
            "discovery_index": 3,
            "full_success": True,
            "summary": _summary(True),
            "artifacts": {
                "resolved_config": config_path.name,
                "trace": trace_path.name,
                "trace_retained": True,
                "sha256": {
                    "resolved_config": file_sha256(config_path),
                    "trace": file_sha256(trace_path),
                },
            },
        }
    )
    write_json(result_path, result)
    trajectory_id = f"manipulation_01_{candidate_id}"
    catalog = {
        "actual_contact_grasp_pose_catalog_schema_version": 1,
        "trajectory_catalog_schema_version": 1,
        "experiment_id": config["experiment_id"],
        "catalog_kind": "manipulation",
        "complete": True,
        "aliases": {"best_first": trajectory_id, "best_nominal": trajectory_id},
        "trajectories": [
            {
                "trajectory_id": trajectory_id,
                "label": trajectory_id,
                "aliases": ["best_first", "best_nominal"],
                "classification": "success",
                "candidate_id": str(candidate_id),
                "discovery_index": 3,
                "full_success": True,
                "artifacts": {
                    "resolved_config": f"{trajectory_id}/resolved_config.json",
                    "result": f"{trajectory_id}/result.json",
                    "trace": f"{trajectory_id}/trace.npz",
                    "video": None,
                    "sha256": {
                        "resolved_config": file_sha256(config_path),
                        "result": file_sha256(result_path),
                        "trace": file_sha256(trace_path),
                    },
                },
            }
        ],
    }
    catalog_path = catalog_root / "catalog.json"
    write_json(catalog_path, catalog)
    return catalog_path


class _FakeRunner:
    def __init__(self, *, density_pass: bool = True, robustness_passes: int = 45):
        self.calls: list[dict] = []
        self.density_pass = density_pass
        self.robustness_passes = robustness_passes

    def __call__(self, config, *, trace_path, video_path):
        self.calls.append(copy.deepcopy(config))
        metadata = config.get("candidate_metadata", {})
        post = metadata.get("post_validation", {})
        robust = metadata.get("robustness_trial", {})
        if post.get("family") == "constant_density_full_reset":
            passed = self.density_pass
        elif robust:
            passed = int(robust["trial"]) < self.robustness_passes
        else:
            passed = True
        if trace_path is not None:
            np.savez_compressed(
                trace_path,
                time_s=np.array([0.0, 0.001]),
                cube_mass_kg=np.array([config["cube"]["mass_kg"]]),
            )
        if video_path is not None:
            Path(video_path).write_bytes(b"synthetic-mp4")
        return _summary(passed)


def test_density_materializer_changes_only_validation_material_and_context():
    config = load_config(TEMPLATE)
    before = copy.deepcopy(config)

    resolved = materialize_constant_density_revalidation(
        config, source_candidate_id="candidate-1"
    )

    assert config == before
    assert resolved["cube"]["edge_m"] == config["cube"]["edge_m"]
    assert resolved["cube"]["mass_kg"] == pytest.approx(
        constant_density_mass_kg(config["cube"]["edge_m"])
    )
    assert resolved["run_context"] == {"kind": "robustness_trial"}
    evidence = resolved["candidate_metadata"]["post_validation"]
    assert evidence["full_reset_rerun"] is True
    assert evidence["checkpoint_used"] is False
    validate_config(resolved)


def test_post_validation_is_atomic_resumable_and_labels_only_real_passes(tmp_path):
    catalog = _source_catalog(tmp_path)
    output = tmp_path / "post_validation"
    runner = _FakeRunner(density_pass=True, robustness_passes=45)

    report = run_relative_wrist_pose_post_validation(
        catalog,
        output,
        simulation_runner=runner,
    )

    assert len(runner.calls) == 52  # fixed + density + exact 50 perturbations
    assert report["fixed_160g"]["full_success_count"] == 1
    assert report["constant_density"]["full_success_count"] == 1
    assert report["best_first_pose_friction"]["robust_passed"] is True
    assert report["best_first_pose_friction"]["perturbation_passes"] == 45
    assert report["best_first_pose_friction"]["validation_label"].endswith(
        "_ablation"
    )
    fixed_catalog = json.loads((output / "fixed_160g_catalog.json").read_text())
    density_catalog = json.loads(
        (output / "constant_density_catalog.json").read_text()
    )
    assert fixed_catalog["aliases"]["best_first"]
    assert density_catalog["validation_label"] == (
        "validated_constant_density_manipulation"
    )
    assert all(
        entry["validation_label"] is not None
        for entry in density_catalog["trajectories"]
        if entry["classification"] == "success"
    )
    viewer_selections = (
        (output / "fixed_160g_catalog.json", "best_first"),
        (output / "constant_density_catalog.json", "best_nominal"),
        (output / "pose_friction_catalog.json", "first_passing_trial"),
    )
    for catalog_path, selector in viewer_selections:
        viewer_source = resolve_viewer_source(
            catalog_path=catalog_path,
            trajectory=selector,
        )
        assert viewer_source.from_catalog is True
        assert viewer_source.config_path.is_file()
        assert viewer_source.trace_path is not None
        assert viewer_source.trace_path.is_file()
    progress = json.loads((output / "progress.json").read_text())
    assert progress == {
        "complete": True,
        "committed_job_count": 52,
        "committed_jobs": sorted(progress["committed_jobs"]),
    }

    resumed_runner = _FakeRunner()
    resumed = run_relative_wrist_pose_post_validation(
        catalog,
        output,
        resume=True,
        simulation_runner=resumed_runner,
    )
    assert resumed == report
    assert resumed_runner.calls == []
    with pytest.raises(RuntimeError, match="manifest/input hash mismatch"):
        run_relative_wrist_pose_post_validation(
            catalog,
            output,
            resume=True,
            workers=2,
            simulation_runner=_FakeRunner(),
        )


def test_density_and_robustness_failures_never_receive_success_labels(tmp_path):
    catalog = _source_catalog(tmp_path)
    output = tmp_path / "failed_validation"
    runner = _FakeRunner(density_pass=False, robustness_passes=44)

    report = run_relative_wrist_pose_post_validation(
        catalog,
        output,
        simulation_runner=runner,
    )

    assert report["fixed_160g"]["validation_label"] == (
        "validated_fixed_160g_manipulation_ablation"
    )
    assert report["constant_density"]["validation_label"] is None
    assert report["best_first_pose_friction"]["validation_label"] is None
    density_catalog = json.loads(
        (output / "constant_density_catalog.json").read_text()
    )
    assert density_catalog["success_count"] == 0
    assert density_catalog["validation_label"] is None
    assert "best_nominal" not in density_catalog["aliases"]
    assert density_catalog["aliases"]["best_attempt"]
    assert density_catalog["trajectories"][0]["validation_label"] is None


def test_optional_video_is_created_only_for_passing_density_job(tmp_path):
    catalog = _source_catalog(tmp_path)
    output = tmp_path / "with_video"
    runner = _FakeRunner(density_pass=True)

    run_relative_wrist_pose_post_validation(
        catalog,
        output,
        simulation_runner=runner,
        render_density_videos=True,
    )

    assert len(runner.calls) == 53
    density_catalog = json.loads(
        (output / "constant_density_catalog.json").read_text()
    )
    entry = density_catalog["trajectories"][0]
    assert entry["classification"] == "success"
    assert entry["artifacts"]["video"].endswith("trajectory.mp4")
    assert (output / entry["artifacts"]["video"]).read_bytes() == b"synthetic-mp4"
    assert all(
        record.get("candidate_metadata", {}).get("post_validation", {}).get("family")
        != "constant_density_full_reset"
        or record["cube"]["mass_kg"] > 0.160
        for record in runner.calls
    )


def test_catalog_and_resume_hash_tampering_fail_closed(tmp_path):
    catalog = _source_catalog(tmp_path)
    sources = load_canonical_fixed_160g_sources(catalog)
    assert len(sources) == 1 and sources[0].best_first
    output = tmp_path / "tamper"
    run_relative_wrist_pose_post_validation(
        catalog, output, simulation_runner=_FakeRunner()
    )

    trace = sources[0].trace_path
    trace.write_bytes(trace.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        run_relative_wrist_pose_post_validation(
            catalog,
            output,
            resume=True,
            simulation_runner=_FakeRunner(),
        )


def test_workers_one_and_two_commit_the_same_deterministic_catalog_order(tmp_path):
    catalog = _source_catalog(tmp_path)
    serial_output = tmp_path / "serial"
    parallel_output = tmp_path / "parallel"

    run_relative_wrist_pose_post_validation(
        catalog,
        serial_output,
        workers=1,
        simulation_runner=_FakeRunner(robustness_passes=45),
    )
    run_relative_wrist_pose_post_validation(
        catalog,
        parallel_output,
        workers=2,
        simulation_runner=_FakeRunner(robustness_passes=45),
    )

    for name in (
        "fixed_160g_catalog.json",
        "constant_density_catalog.json",
        "pose_friction_catalog.json",
    ):
        serial = json.loads((serial_output / name).read_text())
        parallel = json.loads((parallel_output / name).read_text())
        assert [entry["trajectory_id"] for entry in serial["trajectories"]] == [
            entry["trajectory_id"] for entry in parallel["trajectories"]
        ]
        assert serial["aliases"] == parallel["aliases"]
        assert [entry["full_success"] for entry in serial["trajectories"]] == [
            entry["full_success"] for entry in parallel["trajectories"]
        ]
    serial_progress = json.loads((serial_output / "progress.json").read_text())
    parallel_progress = json.loads((parallel_output / "progress.json").read_text())
    assert serial_progress["committed_jobs"] == parallel_progress["committed_jobs"]
    assert len(parallel_progress["committed_jobs"]) == len(
        set(parallel_progress["committed_jobs"])
    )

    with pytest.raises(ValueError, match="positive integer"):
        run_relative_wrist_pose_post_validation(
            catalog,
            tmp_path / "invalid_workers",
            workers=0,
            simulation_runner=_FakeRunner(),
        )
