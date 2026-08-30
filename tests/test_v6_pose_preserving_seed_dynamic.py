from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

import xhand_grasp.tuning.pose_preserving_seed_dynamic as dynamic
from xhand_grasp.config import ACTIVE_ACTUATORS, load_config, validate_config
from xhand_grasp.experiment import resolve_experiment
from xhand_grasp.scene import rpy_degrees_to_rotation_matrix
from xhand_grasp.tuning.pose_preserving_seed_campaign import (
    load_pose_preserving_seed_sources,
)


ROOT = Path(__file__).resolve().parents[1]
CATALOG = (
    ROOT
    / "artifacts"
    / "left_opposed_face_palm_tilted_down_far_hand_fingertip_grasp_then_lift"
    / "grasp_acquisition_high_thumb"
    / "trajectory_catalog"
    / "catalog.json"
)
TEMPLATE = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_pose_preserving_grasp.json"
)

pytestmark = pytest.mark.skipif(
    not CATALOG.is_file(),
    reason="local six-seed trajectory catalog is unavailable",
)


def _pose_checks(value: bool) -> dict[str, bool]:
    return {
        "object_pose_preserved_until_grasp_acquisition": value,
        "support_retained_until_grasp_acquisition": value,
        "no_hand_cube_contact_during_settle": value,
        "v6_pose_preservation_trace_matches_raw_state": value,
        "v6_close_profile_trace_matches_config": value,
    }


def _fake_summary(config: dict) -> dict:
    local_index = int(config["candidate_metadata"]["local_index"])
    acquired = local_index == 1
    return {
        "passed": False,
        "failed_checks": [] if acquired else ["grasp_not_acquired"],
        "checks": _pose_checks(True),
        "stage_status": {
            "grasp_success": acquired,
            "manipulation_success": False,
            "full_success": False,
        },
        "metrics": {
            "pose_preservation": {
                "max_translation_m": 0.0001,
                "max_orientation_drift_deg": 0.2,
                "translation_limit_m": 0.0005,
                "orientation_limit_deg": 1.0,
                "distal_contact_onset_span_s": 0.003,
            },
            "verify_effective_finger_count": 3 if acquired else 2,
            "verify_max_simultaneous_effective_finger_count": (
                3 if acquired else 2
            ),
            "verify_target_face_effective_duty": {
                finger: 0.9 if acquired else 0.4
                for finger in ("thumb", "index", "mid")
            },
            "verify_max_consecutive_all_gate_steps": 250 if acquired else 0,
            "peak_total_distal_contact_force_n": 1.0,
            "actuator_saturation_fraction": 0.01,
        },
    }


def test_dynamic_generation_is_seeded_and_changes_only_declared_search_fields():
    sources = load_pose_preserving_seed_sources(CATALOG)
    template = load_config(TEMPLATE)
    bounds = resolve_experiment(template).search_bounds
    source = sources[0]
    first = dynamic.generate_dynamic_acquisition_candidates(
        source, template, count=8, seed=20260821
    )
    repeated = dynamic.generate_dynamic_acquisition_candidates(
        source, template, count=8, seed=20260821
    )
    different = dynamic.generate_dynamic_acquisition_candidates(
        source, template, count=8, seed=20260822
    )

    assert first == repeated
    assert first[0] == different[0]
    assert first[1:] != different[1:]
    assert [item["candidate_id"] for item in first] == sorted(
        item["candidate_id"] for item in first
    )
    for record in first:
        config = record["config"]
        validate_config(config)
        dynamic.assert_dynamic_candidate_invariants(config, source, template)
        assert config["cube"] == source["cube"]
        assert config["scene"] == source["scene"]
        assert config["control"]["manipulation_delta_rad"] == {
            name: 0.0 for name in ACTIVE_ACTUATORS
        }
        profile = config["control"]["close_profile"]
        for group in dynamic.CLOSE_GROUP_ORDER:
            starts = {
                profile[name]["start_fraction"]
                for name in dynamic.CLOSE_GROUP_ACTUATORS[group]
            }
            assert len(starts) == 1
            for name in dynamic.CLOSE_GROUP_ACTUATORS[group]:
                assert profile[name]["end_fraction"] == template["control"][
                    "close_profile"
                ][name]["end_fraction"]
        assert bounds.contains_pregrasp_targets(
            config["control"]["pregrasp_targets_rad"]
        )
        for name in ACTIVE_ACTUATORS:
            if name != dynamic.THUMB_BEND_ACTUATOR:
                assert config["control"]["grasp_targets_rad"][name] == source[
                    "grasp_targets_rad"
                ][name]
        root = np.asarray(config["hand_pose"]["translation_m"])
        rotation = rpy_degrees_to_rotation_matrix(config["hand_pose"]["rpy_deg"])
        local = np.asarray(
            config["candidate_metadata"]["candidate_cube_in_root_m"]
        )
        np.testing.assert_allclose(
            root + rotation @ local,
            source["initial_cube_world_pose"]["position_m"],
            rtol=0.0,
            atol=2e-15,
        )


def test_all_six_dynamic_bases_preserve_source_material_and_override_context():
    sources = load_pose_preserving_seed_sources(CATALOG)
    template = load_config(TEMPLATE)
    for source in sources:
        config = dynamic.generate_dynamic_acquisition_candidates(
            source, template, count=1, seed=20260821
        )[0]["config"]
        assert config["cube"] == source["cube"]
        assert config["candidate_metadata"]["seed"] is None
        if source["parameter_override_run"]:
            assert config["run_context"] == {"kind": "parameter_override_run"}
        else:
            assert "run_context" not in config


def test_versioned_evidence_manifest_materializes_all_six_exact_base_candidates():
    sources = load_pose_preserving_seed_sources(CATALOG)
    template = load_config(TEMPLATE)
    evidence, provenance = dynamic.load_evidence_seed_manifest(
        dynamic.DEFAULT_EVIDENCE_SEED_MANIFEST, sources
    )

    assert len(evidence) == 6
    assert provenance["seed_count"] == 6
    binding = provenance["source_catalog_binding"]
    assert binding["mode"] == "source_semantic_rebind"
    assert binding["manifest_sha256"] == json.loads(
        dynamic.DEFAULT_EVIDENCE_SEED_MANIFEST.read_text(encoding="utf-8")
    )["source_catalog_sha256"]
    assert binding["loaded_sha256"] == dynamic.file_sha256(CATALOG)
    for source in sources:
        source_id = source["source_candidate_id"]
        seed = evidence[source_id]
        config = dynamic.generate_dynamic_acquisition_candidates(
            source,
            template,
            count=1,
            seed=20260821,
            evidence_seed=seed,
        )[0]["config"]
        assert config["hand_pose"]["rpy_deg"] == seed["hand_pose"]["rpy_deg"]
        assert config["candidate_metadata"]["candidate_cube_in_root_m"] == seed[
            "hand_pose"
        ]["cube_in_root_m"]
        assert config["control"]["pregrasp_targets_rad"] == seed[
            "pregrasp_targets_rad"
        ]
        assert config["control"]["close_profile"] == seed["close_profile"]
        assert config["candidate_metadata"]["pose_seed_provenance"][
            "kind"
        ] == "versioned_verified_dynamics_seed"
        assert config["candidate_metadata"]["seed"] is None
    assert evidence[110064]["grasp_target_overrides_rad"][
        dynamic.THUMB_BEND_ACTUATOR
    ] == 1.12


def test_catalog_rebind_rejects_a_changed_source_semantic_field():
    sources = list(copy.deepcopy(load_pose_preserving_seed_sources(CATALOG)))
    sources[0]["cube"]["mass_kg"] += 0.001

    with pytest.raises(ValueError, match="catalog hash is stale"):
        dynamic.load_evidence_seed_manifest(
            dynamic.DEFAULT_EVIDENCE_SEED_MANIFEST,
            sources,
        )


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("hand_rpy_radius_deg", (0.5, 3.1, 0.5), "hand_rpy_radius_deg"),
        (
            "cube_in_root_radius_m",
            (0.001, 0.0031, 0.001),
            "cube_in_root_radius_m",
        ),
        ("pregrasp_radius_rad", 0.121, "pregrasp_radius_rad"),
        ("thumb_bend_radius_rad", 0.101, "thumb_bend_radius_rad"),
        (
            "close_group_start_radius",
            (0.10, 0.10, 0.201),
            "close_group_start_radius",
        ),
    ],
)
def test_dynamic_generation_rejects_nonlocal_search_radii(keyword, value, message):
    source = load_pose_preserving_seed_sources(CATALOG)[0]
    template = load_config(TEMPLATE)
    with pytest.raises(ValueError, match=message):
        dynamic.generate_dynamic_acquisition_candidates(
            source,
            template,
            count=2,
            seed=20260821,
            **{keyword: value},
        )


def test_spawn_executor_uses_spawn_context_and_restores_candidate_order(monkeypatch):
    captured = {}

    class FakeExecutor:
        def __init__(self, *, max_workers, mp_context):
            captured["workers"] = max_workers
            captured["start_method"] = mp_context.get_start_method()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def map(self, function, jobs, *, chunksize):
            captured["function"] = function
            captured["chunksize"] = chunksize
            return [
                {"candidate_id": int(job["candidate_id"])} for job in jobs
            ]

    monkeypatch.setattr(dynamic, "ProcessPoolExecutor", FakeExecutor)
    results = dynamic.run_dynamic_candidate_jobs(
        ({"candidate_id": 9}, {"candidate_id": 2}), workers=3
    )

    assert captured == {
        "workers": 3,
        "start_method": "spawn",
        "function": dynamic.execute_dynamic_candidate_job,
        "chunksize": 1,
    }
    assert [result["candidate_id"] for result in results] == [2, 9]


def test_campaign_persists_each_source_and_reuses_hash_verified_candidates(
    tmp_path,
    monkeypatch,
):
    calls = []

    def fake_run(config, *, trace_path=None, video_path=None):
        assert video_path is None
        calls.append(int(config["candidate_metadata"]["candidate_id"]))
        np.savez_compressed(
            trace_path,
            time=np.asarray([0.001]),
            candidate_id=np.asarray(config["candidate_metadata"]["candidate_id"]),
        )
        return _fake_summary(config)

    monkeypatch.setattr(dynamic, "run_simulation", fake_run)
    output = tmp_path / "campaign"
    first = dynamic.run_pose_preserving_seed_dynamic_campaign(
        CATALOG,
        TEMPLATE,
        output,
        count_per_source=2,
        workers=1,
        seed=20260821,
    )

    assert len(calls) == 12
    assert first["source_count"] == 6
    assert first["candidate_count"] == 12
    assert first["executed_candidate_count"] == 12
    assert first["reused_candidate_count"] == 0
    assert first["successful_source_count"] == 6
    assert first["all_sources_acquired"] is True
    for source in first["source_results"]:
        source_dir = output / Path(source["result"]).parent
        assert (source_dir / "source_result.json").is_file()
        assert (source_dir / "best_config.json").is_file()
        assert (source_dir / "best_result.json").is_file()
        with np.load(source_dir / "best_trace.npz", allow_pickle=False) as trace:
            assert trace["time"].shape == (1,)
        persisted = json.loads(
            (source_dir / "source_result.json").read_text(encoding="utf-8")
        )
        assert persisted["best_acquisition_success"] is True
        assert persisted["best_candidate_id"] % 1_000_000 == 1

    def must_not_execute(_jobs, _workers):
        raise AssertionError("resume attempted to rerun a verified candidate")

    resumed = dynamic.run_pose_preserving_seed_dynamic_campaign(
        CATALOG,
        TEMPLATE,
        output,
        count_per_source=2,
        workers=4,
        seed=20260821,
        resume=True,
        executor=must_not_execute,
    )
    assert resumed["executed_candidate_count"] == 0
    assert resumed["reused_candidate_count"] == 12
    assert resumed["successful_source_count"] == 6


def test_resume_rejects_a_changed_candidate_artifact(tmp_path, monkeypatch):
    def fake_run(config, *, trace_path=None, video_path=None):
        np.savez_compressed(trace_path, time=np.asarray([0.001]))
        return _fake_summary(config)

    monkeypatch.setattr(dynamic, "run_simulation", fake_run)
    output = tmp_path / "campaign"
    dynamic.run_pose_preserving_seed_dynamic_campaign(
        CATALOG,
        TEMPLATE,
        output,
        count_per_source=1,
        workers=1,
    )
    config_path = next(output.glob("source_*/candidates/*/resolved_config.json"))
    config_path.write_text(
        config_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )

    try:
        dynamic.run_pose_preserving_seed_dynamic_campaign(
            CATALOG,
            TEMPLATE,
            output,
            count_per_source=1,
            workers=1,
            resume=True,
        )
    except RuntimeError as error:
        assert "config file hash mismatch" in str(error)
    else:  # pragma: no cover - makes an accidental silent reuse explicit.
        raise AssertionError("changed candidate artifact was silently reused")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["seeds"]["117060"]["hand_pose"].update(
                {"verified_translation_m": [0.0, 0.0, 0.0]}
            ),
            "fixed cube pose",
        ),
        (
            lambda payload: payload["seeds"]["117060"]["validation"].update(
                {"object_pose_preserved": False}
            ),
            "object_pose_preserved",
        ),
    ],
)
def test_evidence_manifest_rejects_unverified_or_rebound_pose(
    tmp_path, mutate, message
):
    payload = json.loads(
        dynamic.DEFAULT_EVIDENCE_SEED_MANIFEST.read_text(encoding="utf-8")
    )
    mutate(payload)
    manifest = tmp_path / "evidence.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        dynamic.load_evidence_seed_manifest(
            manifest, load_pose_preserving_seed_sources(CATALOG)
        )


def test_acquisition_qpos_backoff_matches_the_declared_formula():
    source = next(
        item
        for item in load_pose_preserving_seed_sources(CATALOG)
        if item["source_candidate_id"] == 117061
    )
    template = load_config(TEMPLATE)
    actual = source["acquisition"]["active_actuator_qpos_rad"]
    target = source["grasp_targets_rad"]
    result = dynamic.acquisition_qpos_backoff_pregrasp(
        source, template, factor=0.5
    )

    for name in ACTIVE_ACTUATORS:
        assert result[name] == pytest.approx(
            actual[name] - 0.5 * (target[name] - actual[name])
        )


def test_cli_parser_exposes_resume_and_search_radii():
    args = dynamic.build_parser().parse_args(
        [
            "--count-per-source",
            "7",
            "--workers",
            "3",
            "--hand-rpy-radius-deg",
            "0.2",
            "0.3",
            "0.4",
            "--cube-in-root-radius-mm",
            "0.5",
            "0.6",
            "0.7",
            "--pregrasp-radius-rad",
            "0.02",
            "--thumb-bend-radius-rad",
            "0.03",
            "--pregrasp-backoff-factor",
            "0.4",
            "--close-group-start-radius",
            "0.04",
            "0.05",
            "0.06",
            "--evidence-seed-manifest",
            str(dynamic.DEFAULT_EVIDENCE_SEED_MANIFEST),
            "--resume",
        ]
    )
    assert args.count_per_source == 7
    assert args.workers == 3
    assert args.hand_rpy_radius_deg == [0.2, 0.3, 0.4]
    assert args.cube_in_root_radius_mm == [0.5, 0.6, 0.7]
    assert args.pregrasp_radius_rad == 0.02
    assert args.thumb_bend_radius_rad == 0.03
    assert args.pregrasp_backoff_factor == 0.4
    assert args.close_group_start_radius == [0.04, 0.05, 0.06]
    assert args.evidence_seed_manifest == str(
        dynamic.DEFAULT_EVIDENCE_SEED_MANIFEST
    )
    assert args.resume is True
