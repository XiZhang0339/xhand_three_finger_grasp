from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from xhand_grasp.actual_contact_grasp_pose_catalog import (
    initialize_or_resume_campaign,
    validate_stage_ledger,
)
from xhand_grasp.artifacts import write_json
from xhand_grasp.grasp_pose import canonical_sha256
import xhand_grasp.tuning.contact_preserving_force_debias_rescue_campaign as campaign


def _source(tmp_path: Path) -> SimpleNamespace:
    root = tmp_path / "source"
    root.mkdir()
    evidence = root / "result.json"
    write_json(evidence, {"complete": True})
    centers = []
    for index in range(5):
        trace = root / f"trace_{index}.npz"
        trace.write_bytes(f"trace-{index}".encode())
        centers.append(
            {
                "center_id": f"{index + 1:064x}",
                "trace_path": str(trace.resolve()),
            }
        )
    return SimpleNamespace(
        root=root.resolve(),
        source_authentication_id="a" * 64,
        artifact_paths=(evidence.resolve(),),
        centers=tuple(centers),
        as_mapping=lambda: {
            "root": str(root.resolve()),
            "source_authentication_id": "a" * 64,
            "artifact_paths": [str(evidence.resolve())],
            "centers": copy.deepcopy(centers),
        },
    )


def _api(**overrides) -> campaign._ForceDebiasApi:
    values = {
        "source_authenticator": lambda value: value,
        "jobs_builder": lambda *_args, **_kwargs: (),
        "job_authenticator": lambda *_args, **_kwargs: None,
        "ranker": lambda value: (0 if value.get("full_success") else 1, value["candidate_id"]),
    }
    values.update(overrides)
    return campaign._ForceDebiasApi(**values)


def _job(
    index: int,
    *,
    stage: str,
    physical: str | None = None,
    operation_scale: float | None = None,
) -> dict:
    scale = float((0.70, 1.00)[index % 2] if operation_scale is None else operation_scale)
    config = {
        "schema_version": 14,
        "experiment_id": "left_opposed_face_palm_down_contact_preserving_planned_lift",
        "contact_force_targets_n": {
            "target_id": canonical_sha256({"operation_scale": scale}),
            **({} if scale == 1.0 else {"operation_scale": scale}),
        },
    }
    payload = {
        "kind": "force_debias_candidate",
        "stage": stage,
        "candidate_id": 18_000 + index,
        "local_index": index,
        "projected_parameters": {
            "index": index,
            "feedback:operation_scale": scale,
        },
        "feedback_parameters": {"operation_scale": scale},
        "physical_plan_sha256": physical or canonical_sha256({"physical": index}),
        "full_reset_required": True,
    }
    return {
        **payload,
        "config": config,
        "candidate_payload_sha256": canonical_sha256(payload),
    }


def _record(index: int, *, rank: int | None = None, physical: str | None = None) -> dict:
    return {
        "candidate_id": 18_000 + index,
        "full_success": False,
        "rank": index if rank is None else rank,
        "rescue_job": {
            "physical_plan_sha256": physical or canonical_sha256({"physical": index})
        },
        "summary": {"failed_checks": ["smooth_motion_jerk_within_limit"]},
    }


def test_fixed_budget_matches_declared_topology() -> None:
    assert campaign.DISCOVERY_CENTER_COUNT == 5
    assert campaign.DISCOVERY_CANDIDATES_PER_CENTER == 32
    assert campaign.DISCOVERY_CANDIDATE_COUNT == 160
    assert campaign.REFINEMENT_CENTER_COUNT == 8
    assert campaign.REFINEMENT_CANDIDATES_PER_CENTER == 64
    assert campaign.REFINEMENT_CANDIDATE_COUNT == 512
    assert campaign.PUBLISHED_CANDIDATE_COUNT == 5


def test_normalization_authenticates_and_enforces_global_physical_uniqueness(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    called: list[int] = []
    api = _api(
        job_authenticator=lambda job, *_args, **_kwargs: called.append(
            int(job["candidate_id"])
        )
    )
    jobs = campaign._normalize_jobs(
        (_job(0, stage="discovery"), _job(1, stage="discovery")),
        api=api,
        source=source,
        stage="discovery",
        expected_count=2,
    )
    assert called == [18_000, 18_001]
    assert [value["local_index"] for value in jobs] == [0, 1]
    assert jobs[0]["job_metadata"]["global_exclusion_set_sha256"] == canonical_sha256(())

    duplicate = _job(
        1,
        stage="discovery",
        physical=jobs[0]["physical_plan_sha256"],
    )
    with pytest.raises(RuntimeError, match="duplicate physical plans"):
        campaign._normalize_jobs(
            (_job(0, stage="discovery"), duplicate),
            api=api,
            source=source,
            stage="discovery",
            expected_count=2,
        )
    with pytest.raises(RuntimeError, match="reused an excluded physical plan"):
        campaign._normalize_jobs(
            (_job(0, stage="discovery"),),
            api=api,
            source=source,
            stage="discovery",
            expected_count=1,
            excluded_physical_plan_sha256=(jobs[0]["physical_plan_sha256"],),
        )


def test_operation_scale_is_range_checked_bound_and_discovery_covers_endpoints(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    api = _api()
    jobs = campaign._normalize_jobs(
        (_job(0, stage="discovery", operation_scale=0.70), _job(1, stage="discovery", operation_scale=1.0)),
        api=api,
        source=source,
        stage="discovery",
        expected_count=2,
    )
    assert [campaign._operation_scale(value) for value in jobs] == [0.70, 1.0]
    missing = _job(0, stage="refinement", operation_scale=0.85)
    missing["config"]["contact_force_targets_n"].pop("operation_scale")
    with pytest.raises(RuntimeError, match="absent from config"):
        campaign._normalize_jobs(
            (missing,),
            api=api,
            source=source,
            stage="refinement",
            expected_count=1,
        )
    with pytest.raises(RuntimeError, match="did not cover"):
        campaign._normalize_jobs(
            (_job(0, stage="discovery", operation_scale=0.80), _job(1, stage="discovery", operation_scale=0.95)),
            api=api,
            source=source,
            stage="discovery",
            expected_count=2,
        )


def test_refinement_selects_exactly_eight_ranked_unique_physical_centers() -> None:
    api = _api(ranker=lambda value: (int(value["rank"]), int(value["candidate_id"])))
    duplicate_hash = canonical_sha256({"duplicate": True})
    records = [
        _record(0, rank=0, physical=duplicate_hash),
        _record(1, rank=1, physical=duplicate_hash),
        *(_record(index, rank=index) for index in range(2, 11)),
    ]
    selected = campaign._select_refinement_records(records, api)
    assert len(selected) == 8
    assert [value["candidate_id"] for value in selected[:2]] == [18_000, 18_002]
    assert len({campaign._record_physical_hash(value) for value in selected}) == 8


def test_cross_stage_candidate_and_physical_identity_are_both_global() -> None:
    first = _record(0)
    second = _record(1)
    ids, physical = campaign._assert_global_record_uniqueness((first, second))
    assert ids == (18_000, 18_001)
    assert len(set(physical)) == 2
    duplicate_id = {**copy.deepcopy(second), "candidate_id": first["candidate_id"]}
    with pytest.raises(RuntimeError, match="candidate ID"):
        campaign._assert_global_record_uniqueness((first, duplicate_id))
    duplicate_physical = copy.deepcopy(second)
    duplicate_physical["rescue_job"]["physical_plan_sha256"] = campaign._record_physical_hash(first)
    with pytest.raises(RuntimeError, match="physical plan"):
        campaign._assert_global_record_uniqueness((first, duplicate_physical))


@pytest.mark.parametrize(
    ("hard_ids", "expected_alias"),
    (((), "best_attempt"), ((22,), "best_nominal")),
)
def test_catalog_alias_is_best_nominal_only_for_full_hard_pass(
    tmp_path: Path, hard_ids: tuple[int, ...], expected_alias: str
) -> None:
    path = tmp_path / "catalog.json"
    write_json(
        path,
        {
            "trajectories": [
                {
                    "trajectory_id": "candidate_22",
                    "candidate_id": 22,
                    # A grasp catalog can call this a success even when the
                    # manipulation hard gate failed.  The explicit hard-ID
                    # set, not this label, owns best_nominal publication.
                    "classification": "success",
                },
                {
                    "trajectory_id": "candidate_23",
                    "candidate_id": 23,
                    "classification": "diagnostic",
                },
            ],
            "aliases": {"pair_rank_01": "candidate_22"},
        },
    )
    campaign._restrict_catalog_aliases(path, hard_pass_candidate_ids=hard_ids)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["aliases"] == {expected_alias: "candidate_22"}
    assert payload["trajectories"][0]["aliases"] == [expected_alias]
    assert payload["trajectories"][1]["aliases"] == []
    assert set(payload["best_grasp_object_pairs"]) == {expected_alias}


def test_source_evidence_detects_tamper(tmp_path: Path) -> None:
    source = _source(tmp_path)
    before = campaign._source_evidence_sha256(source)
    source.artifact_paths[0].write_text('{"complete":false}\n', encoding="utf-8")
    after = campaign._source_evidence_sha256(source)
    assert before != after


def test_source_requires_exactly_five_unique_full_reset_traces(tmp_path: Path) -> None:
    source = _source(tmp_path)
    assert len(campaign._source_centers(source)) == 5
    source.centers = source.centers[:4]
    with pytest.raises(RuntimeError, match="exactly five"):
        campaign._source_centers(source)


def test_committed_phase_is_atomic_resumable_and_rejects_changed_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "campaign"
    manifest = {
        "experiment_id": "left_opposed_face_palm_down_contact_preserving_planned_lift",
        "campaign_input_sha256": "f" * 64,
    }
    initialize_or_resume_campaign(workspace, manifest, resume=False)
    job = _job(0, stage="discovery")
    record = {
        **_record(0),
        "artifact_directory": "force_debias_discovery/candidates/candidate_18000",
    }

    def execute(*_args, **_kwargs):
        artifact = workspace / "force_debias_discovery" / "candidates" / "candidate_18000" / "result.json"
        artifact.parent.mkdir(parents=True)
        write_json(artifact, {"complete": True})
        return (copy.deepcopy(record),), (artifact,)

    monkeypatch.setattr(campaign, "_execute_candidate_jobs", execute)
    report, path = campaign._execute_phase(
        workspace,
        stage="force_debias_discovery",
        directory="force_debias_discovery",
        jobs=(job,),
        source_authentication_id="a" * 64,
        workers=1,
        global_rank_offset=0,
    )
    assert report["complete"] is True
    assert path.is_file()
    validate_stage_ledger(workspace)

    resumed, _ = campaign._execute_phase(
        workspace,
        stage="force_debias_discovery",
        directory="force_debias_discovery",
        jobs=(job,),
        source_authentication_id="a" * 64,
        workers=1,
        global_rank_offset=0,
    )
    assert resumed == report

    changed = copy.deepcopy(job)
    changed["candidate_payload_sha256"] = "b" * 64
    with pytest.raises(RuntimeError, match="input changed on resume"):
        campaign._execute_phase(
            workspace,
            stage="force_debias_discovery",
            directory="force_debias_discovery",
            jobs=(changed,),
            source_authentication_id="a" * 64,
            workers=1,
            global_rank_offset=0,
        )


def test_manifest_binds_source_budget_and_does_not_write_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path)
    before = {
        path: path.read_bytes() for path in source.artifact_paths
    }
    monkeypatch.setattr(
        campaign,
        "build_contact_preserving_planned_lift_manifest",
        lambda *_args, **_kwargs: {
            "experiment_id": "left_opposed_face_palm_down_contact_preserving_planned_lift",
            "campaign_input_sha256": "old",
        },
    )
    manifest = campaign._manifest(Path("config.json"), source, seed=campaign.DEFAULT_SEED)
    assert manifest["campaign_kind"] == "contact_preserving_force_debias_rescue"
    assert manifest["force_debias_budget"]["discovery_candidate_count"] == 160
    assert manifest["force_debias_budget"]["refinement_candidate_count"] == 512
    assert manifest["source_adaptive_authentication"]["source_authentication_id"] == "a" * 64
    assert before == {path: path.read_bytes() for path in source.artifact_paths}
