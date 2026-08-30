from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from xhand_grasp.config import load_config
from xhand_grasp.grasp_pose import canonical_sha256
import xhand_grasp.tuning.contact_preserving_adaptive_event_rescue_campaign as campaign


CONFIG = Path(
    "grasp_configs/left_opposed_face_palm_down_contact_preserving_planned_lift.json"
)


def _summary(*failed: str, jerk: float = 4.0) -> dict:
    return {
        "failed_checks": list(failed),
        "metrics": {
            "motion_smoothness": {
                "operation_peak_abs_filtered_jerk_m_s3": jerk,
            }
        },
    }


def _record(candidate_id: int, *failed: str, jerk: float = 4.0, full=False):
    return {
        "candidate_id": candidate_id,
        "full_success": bool(full),
        "summary": _summary(*failed, jerk=jerk),
    }


def _minimal_api(**overrides):
    values = {
        "source_authenticator": lambda value: value,
        "center_discoverer": lambda *_args, **_kwargs: (),
        "center_selector": None,
        "center_bundle_authenticator": lambda *_args: None,
        "source_type": dict,
        "center_type": dict,
        "descriptor_type": dict,
        "polytope_type": dict,
        "descriptor_builder": lambda *_args: {},
        "descriptor_authenticator": lambda *_args: None,
        "polytope_builder": lambda *_args: {},
        "jobs_builder": lambda *_args, **_kwargs: (),
        "job_authenticator": lambda *_args, **_kwargs: None,
        "ranker": lambda value: (int(value["candidate_id"]),),
        "budget_type": object,
    }
    values.update(overrides)
    return campaign._AdaptiveEventApi(**values)


def test_rank_is_jerk_first_only_after_non_jerk_hard_checks() -> None:
    api = _minimal_api()
    records = (
        _record(1, "horizontal_displacement", jerk=0.1),
        _record(2, campaign._JERK_CHECK, jerk=5.0),
        _record(3, campaign._JERK_CHECK, jerk=3.0),
        _record(4, jerk=2.0, full=True),
        {"candidate_id": 5, "full_success": False, "summary": {}},
    )
    ranked = campaign._rank_records(records, api)
    assert [value["candidate_id"] for value in ranked] == [4, 3, 2, 1, 5]


def test_full_success_uses_contact_rank_while_jerk_only_uses_jerk_first() -> None:
    api = _minimal_api(ranker=lambda value: (int(value["quality_rank"]),))
    records = (
        {**_record(10, jerk=2.4, full=True), "quality_rank": 0},
        {**_record(11, jerk=1.0, full=True), "quality_rank": 1},
        {
            **_record(12, campaign._JERK_CHECK, jerk=5.0),
            "quality_rank": 0,
        },
        {
            **_record(13, campaign._JERK_CHECK, jerk=3.0),
            "quality_rank": 1,
        },
    )
    assert [value["candidate_id"] for value in campaign._rank_records(records, api)] == [
        10,
        11,
        13,
        12,
    ]


@dataclass(frozen=True)
class _Center:
    center_id: str
    candidate_id: int
    config: dict
    trace_path: Path
    trace_sha256: str | None = None

    def as_mapping(self):
        return {
            "center_id": self.center_id,
            "candidate_id": self.candidate_id,
            "config": copy.deepcopy(self.config),
            "trace_path": str(self.trace_path),
            "trace_sha256": self.trace_sha256,
        }


@dataclass(frozen=True)
class _Descriptor:
    descriptor_id: str

    def as_mapping(self):
        return {"descriptor_id": self.descriptor_id}

    @classmethod
    def from_mapping(cls, raw):
        return cls(str(raw["descriptor_id"]))


@dataclass(frozen=True)
class _Polytope:
    polytope_id: str

    def as_mapping(self):
        return {"polytope_id": self.polytope_id}

    @classmethod
    def from_mapping(cls, raw):
        return cls(str(raw["polytope_id"]))


def _context(index: int, tmp_path: Path) -> campaign._CenterContext:
    trace = tmp_path / f"trace_{index}.npz"
    np.savez_compressed(trace, time=np.asarray([0.0]))
    center = _Center(
        center_id=f"{index + 1:064x}",
        candidate_id=100 + index,
        config=load_config(CONFIG),
        trace_path=trace,
    )
    return campaign._CenterContext(
        center=center,
        descriptor=_Descriptor(f"{index + 11:064x}"),
        polytope=_Polytope(f"{index + 21:064x}"),
    )


def _job(context: campaign._CenterContext, index: int, stage: str) -> dict:
    config = copy.deepcopy(context.center.config)
    config.setdefault("candidate_metadata", {})["adaptive_test"] = {
        "descriptor_id": context.descriptor_id,
        "index": index,
    }
    payload = {
        "schema_version": 1,
        "kind": "v14_adaptive_contact_event_candidate",
        "stage": stage,
        "candidate_id": 15_000 + index,
        "local_index": index,
        "source_candidate_id": context.center.candidate_id,
        "source_physical_plan_sha256": "a" * 64,
        "descriptor_id": context.descriptor_id,
        "polytope_id": context.polytope_id,
        "requested_normalized_parameters": {"x": 0.0},
        "projected_parameters": {"x": float(index)},
        "projection_distance": 0.0,
        "config_semantic_sha256": canonical_sha256(config),
        "physical_plan_sha256": canonical_sha256(
            {
                "stage": stage,
                "descriptor_id": context.descriptor_id,
                "index": index,
            }
        ),
        "full_reset_required": True,
    }
    return {
        **payload,
        "config": config,
        "candidate_payload_sha256": canonical_sha256(payload),
    }


def test_job_normalization_authenticates_and_rejects_duplicate_global_indices(
    tmp_path: Path,
) -> None:
    contexts = (_context(0, tmp_path), _context(1, tmp_path))
    authenticated: list[int] = []
    api = _minimal_api(
        job_authenticator=lambda job, *_args: authenticated.append(
            int(job["candidate_id"])
        )
    )
    batches = (
        (contexts[0], (_job(contexts[0], 0, "exploration"),)),
        (contexts[1], (_job(contexts[1], 1, "exploration"),)),
    )
    jobs = campaign._normalize_jobs(
        batches,
        api=api,
        expected_count=2,
        stage="exploration",
    )
    assert authenticated == [15_000, 15_001]
    assert [value["job_sequence_index"] for value in jobs] == [0, 1]
    assert jobs[1]["job_metadata"]["source_center_id"] == contexts[1].center_id
    duplicate = _job(contexts[1], 0, "exploration")
    duplicate["candidate_id"] += 100
    with pytest.raises(RuntimeError, match="duplicate local indices"):
        campaign._normalize_jobs(
            ((contexts[0], batches[0][1]), (contexts[1], (duplicate,))),
            api=api,
            expected_count=2,
            stage="exploration",
        )
    duplicate_physical = _job(contexts[1], 1, "exploration")
    duplicate_physical["physical_plan_sha256"] = batches[0][1][0][
        "physical_plan_sha256"
    ]
    duplicate_payload = {
        key: copy.deepcopy(value)
        for key, value in duplicate_physical.items()
        if key not in {"config", "candidate_payload_sha256"}
    }
    duplicate_physical["candidate_payload_sha256"] = canonical_sha256(
        duplicate_payload
    )
    with pytest.raises(RuntimeError, match="duplicate physical plans"):
        campaign._normalize_jobs(
            (
                (contexts[0], batches[0][1]),
                (contexts[1], (duplicate_physical,)),
            ),
            api=api,
            expected_count=2,
            stage="exploration",
        )


def test_refinement_centers_follow_adaptive_rank_and_unique_projection() -> None:
    api = _minimal_api(ranker=lambda value: (float(value["rank"]),))
    records = []
    for index in range(10):
        records.append(
            {
                **_record(index, campaign._JERK_CHECK, jerk=4.0 + index),
                "rank": 10 - index,
                "rescue_job": {
                    "descriptor_id": "d" * 64,
                    "projected_parameters": {"x": index // 2},
                },
            }
        )
    selected = campaign._select_refinement_records(records, api, count=4)
    assert len(selected) == 4
    assert len(
        {
            canonical_sha256(value["rescue_job"]["projected_parameters"])
            for value in selected
        }
    ) == 4


def test_materialized_center_uses_path_bound_authenticator_signature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(CONFIG)
    source_root = tmp_path / "source"
    source_root.mkdir()
    config_path = source_root / "resolved_config.json"
    result_path = source_root / "result.json"
    trace_path = tmp_path / "rerun" / "trace.npz"
    trace_path.parent.mkdir()
    config_path.write_text(json.dumps(config), encoding="utf-8")
    summary = _summary(campaign._JERK_CHECK, jerk=4.0)
    result_path.write_text(json.dumps({"summary": summary}), encoding="utf-8")
    np.savez_compressed(trace_path, time=np.asarray([0.0]))
    record = SimpleNamespace(
        candidate_id=42,
        config_path=config_path,
        result_path=result_path,
    )
    source = SimpleNamespace(source_authentication_id="f" * 64)
    center = _Center(
        center_id="a" * 64,
        candidate_id=42,
        config=config,
        trace_path=trace_path,
    )
    calls = []

    monkeypatch.setattr(
        campaign,
        "authenticate_v14_candidate_artifacts",
        lambda root, **kwargs: SimpleNamespace(
            trace_path=trace_path,
            result={"summary": summary},
            artifact_paths=(config_path, result_path, trace_path),
        ),
    )

    def authenticate(root, *, source_record, source_authentication_id):
        calls.append((root, source_record, source_authentication_id))
        return center

    api = _minimal_api(center_bundle_authenticator=authenticate)
    observed, artifacts = campaign._authenticate_materialized_center(
        api, source, record, trace_path.parent
    )
    assert observed is center
    assert calls == [(trace_path.parent, record, "f" * 64)]
    assert trace_path in artifacts


def test_source_evidence_allows_authenticated_sibling_ancestor(tmp_path: Path) -> None:
    root = tmp_path / "event"
    ancestor = tmp_path / "rescue_v2"
    root.mkdir()
    ancestor.mkdir()
    local = root / "manifest.json"
    external = ancestor / "report.json"
    local.write_text("{}", encoding="utf-8")
    external.write_text("{}", encoding="utf-8")
    source = SimpleNamespace(root=root, artifact_paths=(local, external))
    evidence = campaign._source_evidence_sha256(source)
    assert any(value.startswith("event_source/") for value in evidence)
    assert any(value.startswith("authenticated_ancestor/") for value in evidence)


def test_fourth_stage_runner_commits_and_resumes_atomic_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(CONFIG)
    source_root = tmp_path / "immutable_event_source"
    source_root.mkdir()
    trace_paths = []
    centers = []
    for index in range(2):
        trace = tmp_path / f"source_trace_{index}.npz"
        np.savez_compressed(trace, time=np.asarray([0.0]))
        trace_paths.append(trace)
        centers.append(
            _Center(
                center_id=f"{index + 1:064x}",
                candidate_id=100 + index,
                config=copy.deepcopy(config),
                trace_path=trace,
            )
        )

    class _Source:
        root = source_root
        source_authentication_id = "f" * 64
        artifact_paths = ()
        rerun_required_records = ()

        def as_mapping(self):
            return {
                "root": str(self.root),
                "source_authentication_id": self.source_authentication_id,
                "rerun_required_records": [],
            }

    source = _Source()

    class _Budget:
        def __init__(self, **values):
            self.values = values

    stage_base = {"diagnostic": 0, "exploration": 1_000, "local_refinement": 2_000}

    def jobs_builder(
        center,
        descriptor,
        polytope,
        *,
        budget,
        local_index_offset,
        refinement_centers,
        excluded_physical_plan_sha256,
    ):
        output = []
        count = int(budget.values["total_candidate_count"])
        for local in range(count):
            index = local_index_offset + local
            job = _job(
                campaign._CenterContext(center, descriptor, polytope),
                index,
                budget.values["stage"],
            )
            job["candidate_id"] += stage_base[budget.values["stage"]]
            payload = {
                key: copy.deepcopy(value)
                for key, value in job.items()
                if key not in {"config", "candidate_payload_sha256"}
            }
            job["candidate_payload_sha256"] = canonical_sha256(payload)
            output.append(job)
        return tuple(output)

    api = _minimal_api(
        source_authenticator=lambda _root: source,
        center_discoverer=lambda _source, max_centers: tuple(centers[:max_centers]),
        descriptor_type=_Descriptor,
        polytope_type=_Polytope,
        descriptor_builder=lambda center, _trace: _Descriptor(
            canonical_sha256({"center_id": center.center_id})
        ),
        descriptor_authenticator=lambda *_args: None,
        polytope_builder=lambda _config, descriptor: _Polytope(
            canonical_sha256({"descriptor_id": descriptor.descriptor_id})
        ),
        jobs_builder=jobs_builder,
        job_authenticator=lambda *_args, **_kwargs: None,
        ranker=lambda value: (int(value["candidate_id"]),),
        budget_type=_Budget,
    )

    def execute(jobs, _workspace, *, phase_name, workers, global_rank_offset):
        records = []
        for index, job in enumerate(jobs):
            records.append(
                {
                    "candidate_id": int(job["candidate_id"]),
                    "classification": "near_miss",
                    "full_success": False,
                    "grasp_success": True,
                    "summary": _summary(campaign._JERK_CHECK, jerk=3.0 + 0.01 * index),
                    "rescue_job": copy.deepcopy(job["job_metadata"]),
                    "artifact_directory": f"{phase_name}/candidate_{job['candidate_id']}",
                    "plan_rank": global_rank_offset + index,
                }
            )
        return tuple(records), ()

    def catalog(records, workspace, **kwargs):
        target = int(kwargs["target_success_count"])
        stage = f"adaptive_catalog_target_{target}"
        path = workspace / "catalogs" / f"target_{target}" / "report.json"
        existing = campaign._load_committed_report(workspace, stage, path)
        if existing is not None:
            return existing
        selected = campaign._selected_publication_records(records, kwargs["api"])
        return campaign._commit_report(
            workspace,
            stage,
            path,
            {
                "complete": True,
                "catalogs": {},
                "published_candidate_ids": [int(value["candidate_id"]) for value in selected],
                "records": [],
            },
            stage_input={"selected_sha256": canonical_sha256(selected)},
        )

    monkeypatch.setattr(campaign, "_adaptive_event_api", lambda: api)
    monkeypatch.setattr(campaign, "_execute_candidate_jobs", execute)
    monkeypatch.setattr(campaign, "_catalog_stage", catalog)
    monkeypatch.setattr(campaign, "DIAGNOSTIC_CANDIDATE_COUNT", 2)
    monkeypatch.setattr(campaign, "EXPLORATION_CANDIDATE_COUNT", 4)
    monkeypatch.setattr(campaign, "REFINEMENT_CANDIDATE_COUNT", 4)
    monkeypatch.setattr(
        campaign,
        "_select_refinement_records",
        lambda records, _api: tuple(copy.deepcopy(dict(value)) for value in records[:2]),
    )
    monkeypatch.setattr(
        campaign,
        "_manifest",
        lambda _config, _source, seed: {
            "experiment_id": config["experiment_id"],
            "config_sha256": "a" * 64,
            "model_sha256": "b" * 64,
            "uv_lock_sha256": "c" * 64,
            "actual_qpos_source_manifest_sha256": "d" * 64,
            "source_sha256": "e" * 64,
            "seed": seed,
            "campaign_input_sha256": "1" * 64,
        },
    )

    output = tmp_path / "adaptive_campaign"
    first = campaign.run_contact_preserving_adaptive_event_rescue_campaign(
        CONFIG,
        output,
        source_event_campaign=source_root,
        resume=False,
        target_success_count=1,
        workers=1,
    )
    resumed = campaign.run_contact_preserving_adaptive_event_rescue_campaign(
        CONFIG,
        output,
        source_event_campaign=source_root,
        resume=True,
        target_success_count=1,
        workers=1,
    )
    assert resumed == first
    assert first["full_success_count"] == 0
    assert first["stop_reason"].endswith("exhausted")
    ledger = campaign.validate_stage_ledger(output)
    assert set(ledger["stages"]) == {
        "adaptive_source_materialization",
        "adaptive_source_audit",
        "adaptive_diagnostics",
        "adaptive_projected_exploration",
        "adaptive_local_refinement",
        "adaptive_catalog_target_1",
        "adaptive_result_target_1",
    }
