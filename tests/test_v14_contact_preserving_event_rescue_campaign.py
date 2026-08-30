from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from xhand_grasp.config import load_config
from xhand_grasp.grasp_pose import canonical_sha256
import xhand_grasp.tuning.contact_preserving_event_rescue_campaign as event_campaign
from xhand_grasp.tuning.contact_preserving_event_rescue import EventDetectionSettings


CONFIG = Path(
    "grasp_configs/left_opposed_face_palm_down_contact_preserving_planned_lift.json"
)


def test_default_descriptor_budget_covers_two_events_for_all_three_fingers() -> None:
    settings = EventDetectionSettings()
    assert settings.selected_fingers == ("thumb", "index", "mid")
    assert settings.max_events_per_finger == 2
    assert settings.global_max_events == 6
    assert settings.minimum_peak_abs_jerk_m_s3 == 2.5


class _FakeApi:
    def __init__(self) -> None:
        self.authenticated: list[int] = []

    def authenticate(self, job, _descriptor):
        self.authenticated.append(int(job["candidate_id"]))


def _api(authenticator):
    class _Descriptor:
        @classmethod
        def from_mapping(cls, value):
            return copy.deepcopy(dict(value))

    return event_campaign._EventRescueApi(
        descriptor_type=_Descriptor,
        descriptor_builder=lambda *args, **kwargs: {},
        descriptor_authenticator=lambda *args, **kwargs: None,
        exploration_builder=lambda *args, **kwargs: (),
        refinement_builder=lambda *args, **kwargs: (),
        job_authenticator=authenticator,
        budget_type=object,
    )


def _jobs(count: int):
    base = load_config(CONFIG)
    result = []
    for index in range(count):
        config = copy.deepcopy(base)
        config.setdefault("candidate_metadata", {})["event_test"] = {"index": index}
        payload = {
            "stage": "exploration",
            "candidate_id": 14_200 + index,
            "local_index": index,
            "parameters": {"event_amplitude": 0.001 * index},
            "config_semantic_sha256": canonical_sha256(config),
        }
        result.append(
            {
                **payload,
                "candidate_payload_sha256": canonical_sha256(payload),
                "config": config,
            }
        )
    return result


def test_event_job_adapter_authenticates_and_preserves_runner_evidence() -> None:
    fake = _FakeApi()
    jobs = _jobs(3)
    normalized = event_campaign._normalize_event_jobs(
        jobs,
        descriptor={"descriptor_id": "d" * 64},
        api=_api(fake.authenticate),
        expected_count=3,
        stage="exploration",
    )
    assert fake.authenticated == [14_200, 14_201, 14_202]
    assert [value["job_sequence_index"] for value in normalized] == [0, 1, 2]
    assert [value["candidate_sha256"] for value in normalized] == [
        value["candidate_payload_sha256"] for value in jobs
    ]
    for index, job in enumerate(normalized):
        metadata = job["job_metadata"]
        assert metadata["parameters"] == {"event_amplitude": 0.001 * index}
        assert metadata["full_reset_required"] is True
        assert metadata["runner_stage"] == "exploration"


def test_event_job_adapter_rejects_budget_and_uniqueness_errors() -> None:
    fake = _FakeApi()
    with pytest.raises(RuntimeError, match="exactly 3"):
        event_campaign._normalize_event_jobs(
            _jobs(2),
            descriptor={},
            api=_api(fake.authenticate),
            expected_count=3,
            stage="exploration",
        )
    duplicate = _jobs(3)
    duplicate[2]["candidate_id"] = duplicate[1]["candidate_id"]
    with pytest.raises(RuntimeError, match="unique candidate IDs"):
        event_campaign._normalize_event_jobs(
            duplicate,
            descriptor={},
            api=_api(fake.authenticate),
            expected_count=3,
            stage="exploration",
        )


def test_refinement_centers_use_contact_first_rank_and_are_unique(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        {
            "candidate_id": index,
            "rescue_job": {"parameters": {"center": index}},
        }
        for index in range(10)
    ]
    records.append(
        {
            "candidate_id": 100,
            "rescue_job": {"parameters": {"center": 9}},
        }
    )
    monkeypatch.setattr(
        event_campaign,
        "_rank_phase_one",
        lambda values: tuple(
            sorted((copy.deepcopy(dict(value)) for value in values), key=lambda x: -x["candidate_id"])
        ),
    )
    centers = event_campaign._select_refinement_centers(records, count=8)
    assert centers == tuple({"center": value} for value in (9, 8, 7, 6, 5, 4, 3, 2))


def test_catalog_render_trace_may_only_add_video_frame_steps(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    rendered = tmp_path / "rendered.npz"
    changed = tmp_path / "changed.npz"
    time = np.asarray([0.0, 0.001, 0.002])
    qpos = np.arange(6, dtype=np.float64).reshape(3, 2)
    np.savez_compressed(
        source,
        time=time,
        qpos=qpos,
        video_frame_steps=np.asarray([], dtype=np.int64),
    )
    np.savez_compressed(
        rendered,
        time=time,
        qpos=qpos,
        video_frame_steps=np.asarray([0, 2], dtype=np.int64),
    )
    np.savez_compressed(
        changed,
        time=time,
        qpos=qpos + 1e-9,
        video_frame_steps=np.asarray([0, 2], dtype=np.int64),
    )
    assert event_campaign._catalog_trace_matches_source_dynamics(source, rendered)
    assert not event_campaign._catalog_trace_matches_source_dynamics(source, changed)


def test_phase_report_exposes_fixed_complete_budget_and_centers() -> None:
    records = [
        {"candidate_id": index, "full_success": index == 0} for index in range(4)
    ]
    payload = event_campaign._phase_payload(
        stage="local_refinement",
        records=records,
        descriptor_mapping={"descriptor_id": "d" * 64},
        source_authentication_id="s" * 64,
        centers=({"center": 1}, {"center": 2}),
    )
    assert payload["complete"] is True
    assert payload["declared_candidate_count"] == 4
    assert payload["candidate_count"] == 4
    assert payload["full_success_count"] == 1
    assert payload["refinement_centers"] == [{"center": 1}, {"center": 2}]


def test_third_stage_runner_commits_and_resumes_atomic_phase_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(CONFIG)
    source_root = tmp_path / "immutable_source"
    source_root.mkdir()

    class _Source:
        root = source_root
        parent_candidate_id = 140099
        parent_config = config
        source_authentication_id = "a" * 64
        artifact_paths = ()

        def descriptor(self):
            return {
                "source_authentication_id": self.source_authentication_id,
                "parent_candidate_id": self.parent_candidate_id,
            }

    source = _Source()

    class _Budget:
        def __init__(self, **values):
            self.values = values

    def builder(_config, _descriptor, *centers, budget, validate_configs):
        stage = budget.values["stage"]
        offset = 0 if stage == "exploration" else 100
        jobs = _jobs(2)
        for index, job in enumerate(jobs):
            job["stage"] = stage
            job["candidate_id"] = 14_200 + offset + index
            job["local_index"] = index
            job["parameters"] = {"center": offset + index}
            payload = {
                key: copy.deepcopy(value)
                for key, value in job.items()
                if key not in {"config", "candidate_payload_sha256"}
            }
            job["candidate_payload_sha256"] = canonical_sha256(payload)
        return tuple(jobs)

    api = event_campaign._EventRescueApi(
        descriptor_type=dict,
        descriptor_builder=lambda *args, **kwargs: {},
        descriptor_authenticator=lambda *args, **kwargs: None,
        exploration_builder=builder,
        refinement_builder=builder,
        job_authenticator=lambda *_args: None,
        budget_type=_Budget,
    )

    def execute(jobs, _workspace, *, phase_name, workers, global_rank_offset):
        return (
            tuple(
                {
                    "candidate_id": int(job["candidate_id"]),
                    "full_success": False,
                    "grasp_success": True,
                    "plan_rank": global_rank_offset + index,
                    "rescue_job": copy.deepcopy(job["job_metadata"]),
                    "artifact_directory": (
                        f"{phase_name}/candidate_{job['candidate_id']}"
                    ),
                }
                for index, job in enumerate(jobs)
            ),
            (),
        )

    def catalog(records, workspace, **kwargs):
        path = (
            workspace
            / "catalogs"
            / f"target_{kwargs['target_success_count']}"
            / "report.json"
        )
        existing = event_campaign._load_committed_report(
            workspace,
            f"event_catalog_target_{kwargs['target_success_count']}",
            path,
        )
        if existing is not None:
            return existing
        return event_campaign._commit_report(
            workspace,
            f"event_catalog_target_{kwargs['target_success_count']}",
            path,
            {"complete": True, "catalogs": {}, "records": []},
            stage_input={"records_sha256": canonical_sha256(records)},
        )

    monkeypatch.setattr(
        event_campaign, "authenticate_completed_event_rescue_source", lambda _root: source
    )
    monkeypatch.setattr(event_campaign, "_event_rescue_api", lambda: api)
    monkeypatch.setattr(
        event_campaign,
        "_build_event_descriptor",
        lambda _source, _api: {"descriptor_id": "d" * 64},
    )
    monkeypatch.setattr(
        event_campaign,
        "_event_manifest",
        lambda _config, _source, seed: {
            "experiment_id": config["experiment_id"],
            "campaign_input_sha256": "ignored_before_descriptor_binding",
            "seed": seed,
        },
    )
    monkeypatch.setattr(event_campaign, "_execute_candidate_jobs", execute)
    monkeypatch.setattr(
        event_campaign,
        "_rank_phase_one",
        lambda values: tuple(copy.deepcopy(dict(value)) for value in values),
    )
    monkeypatch.setattr(
        event_campaign,
        "_select_refinement_centers",
        lambda _records: ({"center": 0}, {"center": 1}),
    )
    monkeypatch.setattr(event_campaign, "_catalog_stage", catalog)
    monkeypatch.setattr(event_campaign, "EVENT_EXPLORATION_CANDIDATE_COUNT", 2)
    monkeypatch.setattr(event_campaign, "EVENT_REFINEMENT_CANDIDATE_COUNT", 2)

    output = tmp_path / "event_campaign"
    first = event_campaign.run_contact_preserving_event_rescue_campaign(
        CONFIG,
        output,
        source_rescue_campaign=source_root,
        resume=False,
        target_success_count=1,
        workers=1,
    )
    repeated = event_campaign.run_contact_preserving_event_rescue_campaign(
        CONFIG,
        output,
        source_rescue_campaign=source_root,
        resume=True,
        target_success_count=1,
        workers=1,
    )
    assert repeated == first
    assert first["exploration_candidate_count"] == 2
    assert first["local_refinement_candidate_count"] == 2
    ledger = event_campaign.validate_stage_ledger(output)
    assert set(ledger["stages"]) == {
        "event_source_audit",
        "event_exploration",
        "event_local_refinement",
        "event_catalog_target_1",
        "event_result_target_1",
    }
