from __future__ import annotations

import copy
import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from xhand_grasp.actual_contact_grasp_pose_catalog import (
    build_campaign_manifest,
    commit_campaign_stage,
    initialize_or_resume_campaign,
)
from xhand_grasp.artifacts import REPO_ROOT, file_sha256, write_json
from xhand_grasp.config import load_config
from xhand_grasp.grasp_pose import controller_id, grasp_pose_id
from xhand_grasp.tuning.actual_contact_grasp_pose import (
    _static_cell_input_sha256,
    actual_contact_search_cells,
)
from xhand_grasp.tuning.pose_preserving_seed_campaign import canonical_sha256
from xhand_grasp.tuning.relative_wrist_pose_recovery_auth import (
    authenticate_recovery_parent,
)


CONFIG_PATH = (
    REPO_ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_larger_relative_wrist_pose_actual_contact_smooth_vertical_lift.json"
)


def _candidate(template: dict, cell: object) -> dict:
    config = copy.deepcopy(template)
    config["cube"]["edge_m"] = cell.edge_m
    thumb = "left_hand_thumb_bend_joint_actuator"
    config["grasp_pose"]["nominal_joint_qpos_rad"][thumb] = (
        cell.thumb_actual_center_rad
    )
    config["control"]["contact_preload_targets_rad"][thumb] = (
        cell.thumb_actual_center_rad
    )
    identifier = 101_000_000_000_001
    config["candidate_metadata"] = {
        "campaign_kind": "actual_contact_grasp_pose_static_search",
        "candidate_id": identifier,
        "cell_index": cell.cell_index,
        "cell_id": cell.cell_id,
        "relative_wrist_pose_search": {
            "clockwise_orbit_deg": cell.clockwise_orbit_deg,
        },
    }
    return {
        "candidate_id": identifier,
        **cell.as_dict(),
        "grasp_pose_id": grasp_pose_id(config),
        "controller_id": controller_id(config),
        "candidate_sha256": canonical_sha256(config),
        "static_pass": False,
        "static_metrics": {"static_geometry_pass": False},
        "config": config,
    }


@pytest.fixture(scope="module")
def recovery_parent(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("v11_recovery_parent") / "campaign"
    seed = 20260821
    manifest = build_campaign_manifest(
        CONFIG_PATH,
        seed=seed,
        source_paths=(REPO_ROOT / "xhand_grasp" / "grasp_pose.py",),
    )
    initialize_or_resume_campaign(output, manifest, resume=False)
    template = load_config(CONFIG_PATH)
    cells = actual_contact_search_cells(template)
    assert len(cells) == 700

    sources = [{"source_id": "authenticated_fixture", "eligible": True}]
    source_sha = canonical_sha256(sources)
    source_bundle = {
        "actual_contact_source_bundle_schema_version": 2,
        "complete": True,
        "experiment_id": template["experiment_id"],
        "source_manifest_sha256": manifest[
            "actual_qpos_source_manifest_sha256"
        ],
        "source_count": len(sources),
        "source_bundle_sha256": source_sha,
        "sources": sources,
    }
    source_path = output / "sources" / "source_bundle.json"
    write_json(source_path, source_bundle)
    commit_campaign_stage(
        output,
        "source_bundle",
        stage_input={"campaign_input_sha256": manifest["campaign_input_sha256"]},
        artifacts=(source_path,),
    )

    first = _candidate(template, cells[0])
    quick_pools: dict[int, tuple[dict, ...]] = {}
    for stage, start, samples in (("quick", 0, 1000), ("expanded", 1000, 2000)):
        paths = []
        for cell in cells:
            pool = (first,) if stage == "quick" and cell.cell_index == 0 else ()
            prior = quick_pools.get(cell.cell_index, ()) if stage == "expanded" else ()
            prior_sha = (
                canonical_sha256(
                    [
                        {
                            "candidate_id": int(value["candidate_id"]),
                            "candidate_sha256": str(value["candidate_sha256"]),
                            "static_pass": bool(value.get("static_pass")),
                        }
                        for value in prior
                    ]
                )
                if prior
                else None
            )
            input_sha = _static_cell_input_sha256(
                campaign_input_sha256=manifest["campaign_input_sha256"],
                source_bundle_sha256=source_sha,
                stage=stage,
                cell=cell,
                start_index=start,
                sample_count=samples,
                retain_count=3,
                seed=seed,
                prior_pool_sha256=prior_sha,
            )
            payload = {
                "actual_contact_static_cell_schema_version": 1,
                "complete": True,
                "stage": stage,
                **cell.as_dict(),
                "cell_input_sha256": input_sha,
                "start_index": start,
                "evaluated_count": samples,
                "static_pass_count": 0,
                "new_pool": list(pool),
            }
            path = output / "static" / stage / f"cell_{cell.cell_index:02d}.json"
            write_json(path, payload)
            paths.append(path)
            if stage == "quick":
                quick_pools[cell.cell_index] = pool
        report = {
            "actual_contact_static_stage_schema_version": 1,
            "complete": True,
            "stage": stage,
            "campaign_input_sha256": manifest["campaign_input_sha256"],
            "source_bundle_sha256": source_sha,
            "cell_count": 700,
            "sample_start_index": start,
            "samples_per_cell": samples,
            "pool_retain_per_cell": 3,
            "static_pass_observation_count": 0,
            "prior_pool_static_pass_count": 0,
            "retained_static_pass_count": 0,
        }
        report_path = output / "static" / stage / "report.json"
        write_json(report_path, report)
        paths.append(report_path)
        commit_campaign_stage(
            output,
            f"{stage}_static",
            stage_input={"stage": stage},
            artifacts=paths,
        )
        refinement = {
            "relative_wrist_refinement_schema_version": 1,
            "complete": True,
            "stage": stage,
            "method": "orientation_aware_actual_contact_dls_13_variables",
            "variable_count": 13,
            "selected_source_count": 240,
            "selected_static_pass_count": 0,
            "refined_candidates": [],
        }
        refinement_path = (
            output / "static" / stage / "relative_wrist_orientation_refinement.json"
        )
        write_json(refinement_path, refinement)
        commit_campaign_stage(
            output,
            f"{stage}_relative_wrist_orientation_refinement",
            stage_input={"stage": stage},
            artifacts=(refinement_path,),
        )

    result = {
        "actual_contact_grasp_pose_campaign_result_schema_version": 1,
        "complete": True,
        "experiment_id": template["experiment_id"],
        "campaign_input_sha256": manifest["campaign_input_sha256"],
        "target_reached": False,
        "static_pass_count": 0,
        "grasp_success_count": 0,
        "full_success_count": 0,
    }
    result_path = output / "campaign_result_target_1.json"
    write_json(result_path, result)
    commit_campaign_stage(
        output,
        "campaign_result_1",
        stage_input={"target": 1},
        artifacts=(result_path,),
    )
    return output


@contextmanager
def _mutate_committed_artifact(parent: Path, relative: str):
    path = parent / relative
    ledger_path = parent / "stage_ledger.json"
    original_bytes = path.read_bytes()
    original_ledger = ledger_path.read_bytes()
    try:
        yield path
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        owners = [
            value
            for value in ledger["stages"].values()
            if relative in value["artifacts"]
        ]
        assert len(owners) == 1
        owners[0]["artifacts"][relative] = file_sha256(path)
        write_json(ledger_path, ledger)
    finally:
        path.write_bytes(original_bytes)
        ledger_path.write_bytes(original_ledger)


def test_authenticates_all_cells_and_loads_cell_pool_not_report(recovery_parent: Path):
    authenticated = authenticate_recovery_parent(recovery_parent)
    assert authenticated.candidate_count == 1
    assert authenticated.snapshot["authenticated_cell_count"] == 1400
    assert authenticated.snapshot["candidate_count"] == 1
    assert authenticated.snapshot["parent_result_counts"] == {
        "static_pass_count": 0,
        "grasp_success_count": 0,
        "full_success_count": 0,
    }
    candidate = authenticated.candidates[0]
    assert candidate["candidate_id"] == 101_000_000_000_001
    assert authenticated.candidate_by_id[candidate["candidate_id"]][
        "candidate_sha256"
    ] == candidate["candidate_sha256"]
    body = dict(authenticated.snapshot)
    digest = body.pop("snapshot_sha256")
    assert canonical_sha256(body) == digest


def test_rejects_candidate_tamper_even_if_ledger_digest_is_rebound(
    recovery_parent: Path,
):
    relative = "static/quick/cell_00.json"
    with _mutate_committed_artifact(recovery_parent, relative) as path:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["new_pool"][0]["config"]["hand_pose"]["translation_m"][0] += 0.001
        write_json(path, payload)
        ledger_path = recovery_parent / "stage_ledger.json"
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        ledger["stages"]["quick_static"]["artifacts"][relative] = file_sha256(path)
        write_json(ledger_path, ledger)
        with pytest.raises(RuntimeError, match="candidate config SHA-256 mismatch"):
            authenticate_recovery_parent(recovery_parent)


def test_rejects_cell_input_tamper_and_nonzero_or_incomplete_result(
    recovery_parent: Path,
):
    relative = "static/expanded/cell_00.json"
    with _mutate_committed_artifact(recovery_parent, relative) as path:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["cell_input_sha256"] = "0" * 64
        write_json(path, payload)
        ledger = json.loads((recovery_parent / "stage_ledger.json").read_text())
        ledger["stages"]["expanded_static"]["artifacts"][relative] = file_sha256(path)
        write_json(recovery_parent / "stage_ledger.json", ledger)
        with pytest.raises(RuntimeError, match="input SHA-256 mismatch"):
            authenticate_recovery_parent(recovery_parent)

    result_relative = "campaign_result_target_1.json"
    with _mutate_committed_artifact(recovery_parent, result_relative) as path:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["grasp_success_count"] = 1
        write_json(path, payload)
        ledger = json.loads((recovery_parent / "stage_ledger.json").read_text())
        ledger["stages"]["campaign_result_1"]["artifacts"][result_relative] = file_sha256(path)
        write_json(recovery_parent / "stage_ledger.json", ledger)
        with pytest.raises(RuntimeError, match="grasp_success_count is nonzero"):
            authenticate_recovery_parent(recovery_parent)

    with _mutate_committed_artifact(recovery_parent, result_relative) as path:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["complete"] = False
        write_json(path, payload)
        ledger = json.loads((recovery_parent / "stage_ledger.json").read_text())
        ledger["stages"]["campaign_result_1"]["artifacts"][result_relative] = file_sha256(path)
        write_json(recovery_parent / "stage_ledger.json", ledger)
        with pytest.raises(RuntimeError, match="result is incomplete"):
            authenticate_recovery_parent(recovery_parent)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("edge", "edge_m differs from its cell"),
        ("thumb", "actual thumb differs from its cell"),
        ("orbit", "orbit differs from its cell"),
    ),
)
def test_rejects_candidate_cell_coordinate_mismatch(
    recovery_parent: Path, mutation: str, message: str
):
    relative = "static/quick/cell_00.json"
    with _mutate_committed_artifact(recovery_parent, relative) as path:
        payload = json.loads(path.read_text(encoding="utf-8"))
        candidate = payload["new_pool"][0]
        if mutation == "edge":
            candidate["edge_m"] = 0.086
        elif mutation == "thumb":
            thumb = "left_hand_thumb_bend_joint_actuator"
            candidate["config"]["grasp_pose"]["nominal_joint_qpos_rad"][thumb] = 1.45
            candidate["candidate_sha256"] = canonical_sha256(candidate["config"])
            candidate["grasp_pose_id"] = grasp_pose_id(candidate["config"])
        else:
            candidate["config"]["candidate_metadata"][
                "relative_wrist_pose_search"
            ]["clockwise_orbit_deg"] = 2.5
            candidate["candidate_sha256"] = canonical_sha256(candidate["config"])
        write_json(path, payload)
        ledger_path = recovery_parent / "stage_ledger.json"
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        ledger["stages"]["quick_static"]["artifacts"][relative] = file_sha256(path)
        write_json(ledger_path, ledger)
        with pytest.raises(RuntimeError, match=message):
            authenticate_recovery_parent(recovery_parent)
