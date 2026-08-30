from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from xhand_grasp.actual_contact_grasp_pose_catalog import (
    bind_candidate_result_semantic_sha256,
    commit_campaign_stage,
    initialize_or_resume_campaign,
    validate_stage_ledger,
)
from xhand_grasp.artifacts import file_sha256, write_json
from xhand_grasp.config import load_config
from xhand_grasp.grasp_pose import canonical_sha256
from xhand_grasp.tuning.contact_preserving_robustness_sidecar import (
    RobustnessSidecarBackend,
    authenticate_completed_v14_manipulation_catalog,
    contact_preserving_robustness_sidecar_console,
    run_contact_preserving_robustness_sidecar,
)
from xhand_grasp.tuning.contact_constrained_planner import (
    v14_grasp_object_pair_id,
    v14_grasp_pose_id,
    v14_object_config_id,
)
from xhand_grasp.v14_identity import (
    v14_base_controller_id,
    v14_sequential_planner_id,
)


CONFIG = Path(
    "grasp_configs/left_opposed_face_palm_down_contact_preserving_planned_lift.json"
)
EXPERIMENT_ID = "left_opposed_face_palm_down_contact_preserving_planned_lift"
CANDIDATE_ID = "14000000000000001"


def _source(tmp_path: Path, *, success: bool = True) -> tuple[Path, Path]:
    root = tmp_path / "source"
    initialize_or_resume_campaign(
        root,
        {
            "campaign_manifest_schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "campaign_input_sha256": "a" * 64,
        },
        resume=False,
    )
    member = root / "catalogs/target_1/manipulation/pair_rank_01"
    member.mkdir(parents=True)
    config = load_config(CONFIG)
    config["object_config_id"] = v14_object_config_id(config)
    config["grasp_pose_id"] = v14_grasp_pose_id(config)
    config["grasp_object_pair_id"] = v14_grasp_object_pair_id(config)
    report_id = canonical_sha256({"test": "robustness-sidecar-report"})
    attempt_id = canonical_sha256({"test": "robustness-sidecar-attempt"})
    config.setdefault("candidate_metadata", {})[
        "sequential_checkpoint_planning"
    ] = {
        "report_id": report_id,
        "attempt_report_id": attempt_id,
    }
    config["planner_id"] = v14_sequential_planner_id(report_id, attempt_id)
    config["controller_id"] = v14_base_controller_id(
        config, bind_planner=False
    )
    config_path = member / "resolved_config.json"
    trace_path = member / "trace.npz"
    result_path = member / "result.json"
    write_json(config_path, config)
    np.savez_compressed(trace_path, time=np.asarray((0.0, 0.001)))
    summary = {
        "passed": success,
        "failed_checks": [] if success else ["smooth_peak_jerk"],
        "stage_status": {
            "grasp_success": True,
            "manipulation_success": success,
            "full_success": success,
        },
        "metrics": {},
    }
    candidate = bind_candidate_result_semantic_sha256(
        {
            "contact_preserving_candidate_schema_version": 1,
            "complete": True,
            "candidate_id": int(CANDIDATE_ID),
            "experiment_id": EXPERIMENT_ID,
            "classification": "success" if success else "near_miss",
            "grasp_success": True,
            "full_success": success,
            "summary": summary,
        }
    )
    write_json(result_path, candidate)
    catalog_path = member.parent / "catalog.json"
    trajectory_id = f"pair_rank_01_{CANDIDATE_ID}"
    catalog = {
        "contact_preserving_viewer_catalog_schema_version": 1,
        "trajectory_catalog_schema_version": 1,
        "catalog_kind": "manipulation",
        "complete": True,
        "experiment_id": EXPERIMENT_ID,
        "production_trajectory_video_policy": (
            "deterministic_full_reset_rerun_ffprobe_and_full_decode"
        ),
        "success_count": int(success),
        "aliases": (
            {
                "best_first": trajectory_id,
                "best_nominal": trajectory_id,
                "pair_rank_01": trajectory_id,
            }
            if success
            else {"best_attempt": trajectory_id, "pair_rank_01": trajectory_id}
        ),
        "best_grasp_object_pairs": {},
        "trajectories": [
            {
                "trajectory_id": trajectory_id,
                "candidate_id": CANDIDATE_ID,
                "classification": "success" if success else "diagnostic",
                "grasp_success": True,
                "full_success": success,
                "artifacts": {
                    "resolved_config": f"pair_rank_01/{config_path.name}",
                    "result": f"pair_rank_01/{result_path.name}",
                    "trace": f"pair_rank_01/{trace_path.name}",
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
    write_json(catalog_path, catalog)
    campaign_result = root / "event_rescue_result_target_1.json"
    write_json(
        campaign_result,
        {
            "contact_preserving_event_rescue_result_schema_version": 1,
            "complete": True,
            "experiment_id": EXPERIMENT_ID,
            "target_reached": success,
            "full_success_count": int(success),
            "catalogs": {
                "manipulation": str(catalog_path.relative_to(root)),
            },
        },
    )
    commit_campaign_stage(
        root,
        "published",
        stage_input={"catalog": file_sha256(catalog_path)},
        artifacts=(
            config_path,
            trace_path,
            result_path,
            catalog_path,
            campaign_result,
        ),
    )
    return catalog_path, campaign_result


def _trials(count: int, family: str) -> list[dict]:
    return [
        {
            "trial": index,
            "family": family,
            "source_candidate_id": CANDIDATE_ID,
            "passed": True,
        }
        for index in range(count)
    ]


def _report(passes: int) -> dict:
    robust = passes >= 45
    return {
        "v9_robustness_report_schema_version": 1,
        "contact_preserving_robustness_schema_version": 1,
        "complete": True,
        "experiment_id": EXPERIMENT_ID,
        "seed": 20260821,
        "selected_nominal_count": 1,
        "selected_nominal_candidate_ids": [CANDIDATE_ID],
        "local_perturbations_per_nominal": 16,
        "best_perturbation_count": 50,
        "registered_budget_complete": True,
        "total_perturbation_count": 66,
        "per_nominal": [
            {
                "candidate_id": CANDIDATE_ID,
                "perturbation_count": 16,
                "perturbation_passes": 16,
                "trials": _trials(16, "per_full_success_local_16"),
            }
        ],
        "best_robustness": {
            "candidate_id": CANDIDATE_ID,
            "perturbation_count": 50,
            "perturbation_passes": passes,
            "required_perturbation_passes": 45,
            "robust_passed": robust,
            "trials": _trials(50, "best_first_pose_material_50"),
        },
        "robust_passed": robust,
        "best_selection_policy": "local_perturbation_passes",
        "best_50_selection_alias": "best_robust_local_16_leader",
        "best_50_candidate_id": CANDIDATE_ID,
        "nominal_best_candidate_id": CANDIDATE_ID,
        "required_passes": 45,
        "robust_success": robust,
    }


def _backend(passes: int, calls: list[dict]) -> RobustnessSidecarBackend:
    def runner(catalog, output, *, workers, seed, campaign):
        calls.append(
            {
                "catalog": Path(catalog),
                "output": Path(output),
                "workers": workers,
                "seed": seed,
                "campaign": campaign,
            }
        )
        # The production helper first writes its base v9 report.  The sidecar
        # must atomically replace this with the complete returned v14 payload.
        write_json(output, {"incomplete_base_report": True})
        return _report(passes)

    return RobustnessSidecarBackend(robustness_runner=runner)


def test_sidecar_commits_full_report_and_fresh_best_robust_catalog(
    tmp_path: Path,
) -> None:
    catalog, result = _source(tmp_path)
    source_hashes = (file_sha256(catalog), file_sha256(result))
    calls: list[dict] = []
    output = tmp_path / "sidecar"
    report = run_contact_preserving_robustness_sidecar(
        catalog,
        result,
        output,
        resume=False,
        workers=3,
        backend=_backend(45, calls),
    )
    assert len(calls) == 1
    assert calls[0]["workers"] == 3
    assert report["robust_success"] is True
    assert report["source"]["source_catalog_sha256"] == source_hashes[0]
    assert report["source"]["source_result_sha256"] == source_hashes[1]
    report_path = output / "evidence/perturbation_report.json"
    persisted = json.loads(report_path.read_text())
    assert "incomplete_base_report" not in persisted
    assert persisted["contact_preserving_robustness_sidecar_schema_version"] == 1
    robust_catalog = output / persisted["robust_catalog"]
    robust = json.loads(robust_catalog.read_text())
    assert robust["aliases"]["best_robust"] == robust["aliases"]["best_nominal"]
    assert robust["catalog_kind"] == "robust_manipulation"
    validate_stage_ledger(output)
    assert (file_sha256(catalog), file_sha256(result)) == source_hashes

    resumed = run_contact_preserving_robustness_sidecar(
        catalog,
        result,
        output,
        resume=True,
        workers=8,
        backend=_backend(45, calls),
    )
    assert canonical_sha256(resumed) == canonical_sha256(report)
    assert len(calls) == 1


def test_sidecar_does_not_publish_alias_below_45_of_50(tmp_path: Path) -> None:
    catalog, result = _source(tmp_path)
    output = tmp_path / "sidecar"
    report = run_contact_preserving_robustness_sidecar(
        catalog,
        result,
        output,
        resume=False,
        workers=1,
        backend=_backend(44, []),
    )
    assert report["robust_success"] is False
    assert report["robust_catalog"] is None
    assert not (output / "evidence/robust_manipulation").exists()
    console = contact_preserving_robustness_sidecar_console(report, output)
    assert console["best_perturbation_passes"] == 44
    assert console["required_best_passes"] == 45


def test_sidecar_rejects_overwrite_and_tampered_committed_evidence(
    tmp_path: Path,
) -> None:
    catalog, result = _source(tmp_path)
    output = tmp_path / "sidecar"
    backend = _backend(45, [])
    run_contact_preserving_robustness_sidecar(
        catalog,
        result,
        output,
        resume=False,
        workers=1,
        backend=backend,
    )
    with pytest.raises(FileExistsError):
        run_contact_preserving_robustness_sidecar(
            catalog,
            result,
            output,
            resume=False,
            workers=1,
            backend=backend,
        )
    report_path = output / "evidence/perturbation_report.json"
    changed = json.loads(report_path.read_text())
    changed["robust_success"] = False
    write_json(report_path, changed)
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        run_contact_preserving_robustness_sidecar(
            catalog,
            result,
            output,
            resume=True,
            workers=1,
            backend=backend,
        )


def test_sidecar_rejects_partial_resume_and_tampered_source(tmp_path: Path) -> None:
    catalog, result = _source(tmp_path)
    output = tmp_path / "sidecar"

    def interrupted(_catalog, output_path, **_kwargs):
        write_json(output_path, {"partial": True})
        raise RuntimeError("power loss")

    backend = RobustnessSidecarBackend(robustness_runner=interrupted)
    with pytest.raises(RuntimeError, match="power loss"):
        run_contact_preserving_robustness_sidecar(
            catalog,
            result,
            output,
            resume=False,
            workers=1,
            backend=backend,
        )
    with pytest.raises(RuntimeError, match="partial"):
        run_contact_preserving_robustness_sidecar(
            catalog,
            result,
            output,
            resume=True,
            workers=1,
            backend=backend,
        )

    second_catalog, second_result = _source(tmp_path / "changed")
    second_output = tmp_path / "changed_sidecar"
    run_contact_preserving_robustness_sidecar(
        second_catalog,
        second_result,
        second_output,
        resume=False,
        workers=1,
        backend=_backend(44, []),
    )
    payload = json.loads(second_result.read_text())
    payload["full_success_count"] = 2
    write_json(second_result, payload)
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        run_contact_preserving_robustness_sidecar(
            second_catalog,
            second_result,
            second_output,
            resume=True,
            workers=1,
            backend=_backend(44, []),
        )


def test_source_authentication_rejects_diagnostic_only_catalog(tmp_path: Path) -> None:
    catalog, result = _source(tmp_path, success=False)
    with pytest.raises(RuntimeError, match="no full manipulation success"):
        authenticate_completed_v14_manipulation_catalog(catalog, result)
