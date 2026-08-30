from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from xhand_grasp.tuning.pose_preserving_seed_campaign import canonical_sha256
from xhand_grasp.tuning.relative_wrist_pose_dynamic_centroid_recovery import (
    AuthenticatedV3Recovery,
    EXPECTED_PARENT_STAGE_NAMES,
    _commit_stage,
    _load_ledger,
    _source_files_still_match,
    authenticate_completed_v3_recovery,
    build_recovery_manifest,
    build_parser,
    prepare_guidance_evidence,
    recovery_stop_reason,
    result_only_parent_rank,
    select_guided_parent_evidence,
    select_parent_evidence,
    validate_recovery_paths,
)


ROOT = Path(__file__).resolve().parents[1]
REAL_V3 = (
    ROOT
    / "artifacts"
    / "left_opposed_face_palm_down_larger_relative_wrist_pose_actual_contact_"
    "smooth_vertical_lift"
    / "tune"
    / "active_set_coordinate_recovery_v3_from_campaign_6d_r2"
)


def _real_v3_source_binding_is_current() -> bool:
    manifest_path = REAL_V3 / "recovery_manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        _source_files_still_match(manifest)
    except RuntimeError as error:
        if str(error).startswith("v3 bound source changed:"):
            return False
        raise
    return True


def _parent_value(
    identifier: int,
    *,
    retained: bool,
    height_m: float,
    translation_m: float = 0.0002,
) -> dict:
    return {
        "candidate_id": identifier,
        "candidate_sha256": f"sha-{identifier}",
        "artifact_key": f"artifact-{identifier}",
        "trace_retained": retained,
        "result": {
            "grasp_success": False,
            "summary": {
                "metrics": {
                    "verify_max_simultaneous_effective_finger_count": 3,
                    "verify_target_face_simultaneous_duty": 1.0,
                    "verify_max_consecutive_gate_steps": identifier,
                    "verify_gate_component_duty": {
                        "thumb_actual_qpos_within_range": 1.0
                    },
                    "contact_alignment": {
                        "verify": {"height_spread_p95_m": height_m}
                    },
                    "pose_preservation": {
                        "max_translation_m": translation_m,
                        "max_orientation_drift_deg": 0.5,
                    },
                }
            },
        },
    }


@pytest.mark.skipif(
    not (REAL_V3 / "recovery_reports" / "target_1.json").is_file()
    or not _real_v3_source_binding_is_current(),
    reason="sealed v3 recovery evidence is unavailable or bound to older source",
)
def test_real_v3_authenticates_terminal_stage_and_all_dynamic_artifacts() -> None:
    value = authenticate_completed_v3_recovery(REAL_V3)

    assert value.terminal_stage_sha256 == (
        json.loads((REAL_V3 / "recovery_stage_ledger.json").read_text())["stages"][-1][
            "stage_sha256"
        ]
    )
    assert len(value.dynamic_records) == len(value.authenticated_dynamic) == 51
    assert value.retained_trace_count == 30
    assert value.tombstone_count == 21
    assert value.static_source["candidate_id"] == 8110000000200049
    assert value.evidence_snapshot["snapshot_sha256"] == canonical_sha256(
        {
            key: item
            for key, item in value.evidence_snapshot.items()
            if key != "snapshot_sha256"
        }
    )
    selected = select_guided_parent_evidence(
        value.authenticated_dynamic, count=3
    )
    assert [int(item["candidate_id"]) for item in selected] == [
        3232000000000674,
        3232000000001331,
        3232000000001126,
    ]
    assert selected[1]["trace_retained"] is False

    manifest = build_recovery_manifest(value, seed=20260821)
    assert manifest["recovery_input_sha256"] == canonical_sha256(
        {
            key: item
            for key, item in manifest.items()
            if key != "recovery_input_sha256"
        }
    )
    assert (
        "xhand_grasp.tuning.relative_wrist_pose_dynamic_centroid_recovery"
        in manifest["source_code"]
    )
    assert "xhand_grasp.tuning.relative_wrist_pose_dynamic_guided" in manifest[
        "source_code"
    ]


def test_result_only_parent_selection_is_deterministic_and_can_select_tombstone() -> None:
    retained = _parent_value(1, retained=True, height_m=0.009)
    tombstone = _parent_value(2, retained=False, height_m=0.0051)
    unsafe_pose = _parent_value(
        3, retained=True, height_m=0.004, translation_m=0.001
    )

    assert result_only_parent_rank(tombstone) < result_only_parent_rank(retained)
    assert result_only_parent_rank(retained) < result_only_parent_rank(unsafe_pose)
    forward = select_parent_evidence((retained, tombstone, unsafe_pose), count=2)
    reverse = select_parent_evidence(
        (unsafe_pose, tombstone, retained), count=2
    )
    assert [value["candidate_id"] for value in forward] == [2, 1]
    assert [value["candidate_id"] for value in reverse] == [2, 1]
    assert forward[0]["trace_retained"] is False


def test_selected_tombstone_is_exactly_replayed_without_replaying_the_rest(
    tmp_path: Path,
) -> None:
    values = (
        _parent_value(1, retained=True, height_m=0.009),
        _parent_value(2, retained=False, height_m=0.0051),
        _parent_value(3, retained=False, height_m=0.012),
    )
    raw = tuple(
        {
            "candidate_id": value["candidate_id"],
            "candidate_sha256": value["candidate_sha256"],
            "artifact_directory": f"candidates/candidate_{value['candidate_id']}",
        }
        for value in values
    )
    authenticated = AuthenticatedV3Recovery(
        parent=tmp_path / "parent",
        experiment_id="v11",
        recovery_input_sha256="input",
        terminal_stage_sha256="terminal",
        static_source={},
        dynamic_records=raw,
        authenticated_dynamic=values,
        retained_trace_count=1,
        tombstone_count=2,
        evidence_snapshot={},
    )
    calls: list[int] = []

    def replay(value: dict, output: Path) -> tuple[dict, dict]:
        calls.append(int(value["candidate_id"]))
        return (
            {
                "candidate_id": int(value["candidate_id"]),
                "candidate_sha256": value["candidate_sha256"],
                "artifact_directory": str(
                    output / "parent_trace_replays" / f"candidate_{value['candidate_id']}"
                ),
            },
            {"candidate_id": int(value["candidate_id"]), "reused": False},
        )

    guidance, selected, report_path = prepare_guidance_evidence(
        authenticated,
        tmp_path / "output",
        parent_count=2,
        replay=replay,
        selector=select_parent_evidence,
    )

    assert calls == [2]
    assert {int(value["candidate_id"]) for value in guidance} == {1, 2}
    assert [int(value["candidate_id"]) for value in selected] == [2, 1]
    report = json.loads(report_path.read_text())
    assert report["selected_tombstone_count"] == 1
    assert report["exact_replay_count"] == 1


def test_hash_chain_ledger_is_idempotent_and_rejects_artifact_tampering(
    tmp_path: Path,
) -> None:
    output = tmp_path / "recovery"
    artifact = output / "stage" / "report.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"complete": true}\n', encoding="utf-8")
    _commit_stage(
        output,
        "one",
        stage_input={"input": "a", "workers": 4},
        artifacts=(artifact,),
        summary={"count": 1},
    )
    _commit_stage(
        output,
        "one",
        stage_input={"input": "a", "workers": 4},
        artifacts=(artifact,),
        summary={"count": 1},
    )
    ledger = _load_ledger(output)
    assert len(ledger["stages"]) == 1
    assert ledger["stages"][0]["previous_stage_sha256"] is None

    artifact.write_text('{"complete": false}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="artifact changed"):
        _load_ledger(output)


def test_ledger_chain_rejects_stage_digest_tampering(tmp_path: Path) -> None:
    output = tmp_path / "recovery"
    artifact = output / "report.json"
    output.mkdir()
    artifact.write_text("{}\n", encoding="utf-8")
    _commit_stage(
        output,
        "one",
        stage_input={},
        artifacts=(artifact,),
        summary={},
    )
    ledger_path = output / "dynamic_centroid_stage_ledger.json"
    payload = json.loads(ledger_path.read_text())
    payload["stages"][0]["summary"] = {"tampered": True}
    ledger_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash chain"):
        _load_ledger(output)


def test_parent_and_output_must_be_disjoint(tmp_path: Path) -> None:
    parent = tmp_path / "v3"
    parent.mkdir()
    with pytest.raises(ValueError, match="disjoint"):
        validate_recovery_paths(parent, parent)
    with pytest.raises(ValueError, match="disjoint"):
        validate_recovery_paths(parent, parent / "child")
    with pytest.raises(ValueError, match="disjoint"):
        validate_recovery_paths(parent, tmp_path)
    assert validate_recovery_paths(parent, tmp_path / "new") == (
        parent.resolve(),
        (tmp_path / "new").resolve(),
    )


@pytest.mark.parametrize(
    ("counts", "expected"),
    (
        ((0, 0, 0), "dynamic_centroid_grasp_not_verified"),
        ((1, 0, 0), "measured_grasp_pose_not_verified"),
        ((1, 1, 0), "manipulation_full_success_not_verified"),
        ((1, 1, 1), "manipulation_full_success_verified"),
    ),
)
def test_stop_reason_is_stage_specific(
    counts: tuple[int, int, int], expected: str
) -> None:
    assert recovery_stop_reason(
        grasp_count=counts[0], measured_count=counts[1], full_count=counts[2]
    ) == expected


def test_cli_exposes_parent_output_workers_resume_and_target() -> None:
    values = build_parser().parse_args(
        (
            "--parent",
            "v3",
            "--output-dir",
            "v4",
            "--workers",
            "12",
            "--resume",
            "--target-success-count",
            "5",
        )
    )
    assert values.parent == "v3"
    assert values.output_dir == "v4"
    assert values.workers == 12
    assert values.resume is True
    assert values.target_success_count == 5


def test_parent_contract_has_exactly_fifteen_committed_stages() -> None:
    assert len(EXPECTED_PARENT_STAGE_NAMES) == 15
    assert EXPECTED_PARENT_STAGE_NAMES[-1] == "final_report_1"
