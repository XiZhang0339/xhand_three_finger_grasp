from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import xhand_grasp.cli as cli
from xhand_grasp.actual_contact_grasp_pose_catalog import (
    bind_candidate_result_semantic_sha256,
)
from xhand_grasp.grasp_pose import canonical_sha256
from xhand_grasp.viewer import (
    ViewerSource,
    discover_measured_viewer_data,
    resolve_measured_viewer_source,
)


THUMB_BEND = "left_hand_thumb_bend_joint_actuator"
ID_72 = 1_616_000_016_029_744
ID_73_FIRST = 1_616_000_112_009_876
ID_73_SECOND = 4_864_000_014_401_330


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_measured_candidate(
    dynamic_dir: Path,
    *,
    candidate_id: int,
    edge_mm: int,
    actual_thumb_rad: float,
) -> dict:
    directory = dynamic_dir / "measured" / f"candidate_{candidate_id}"
    directory.mkdir(parents=True)
    config_path = directory / "resolved_config.json"
    config = {
        "schema_version": 10,
        "experiment_id": (
            "left_opposed_face_palm_down_larger_actual_contact_"
            "grasp_pose_smooth_vertical_lift"
        ),
        "cube": {
            "edge_m": edge_mm / 1000.0,
            "mass_kg": 0.160,
            "friction": 0.8,
        },
    }
    config_path.write_text(
        json.dumps(
            config,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    trace_path = directory / "trace.npz"
    np.savez_compressed(trace_path, time=np.asarray([0.001]))
    candidate_digest = canonical_sha256(config)
    result_path = directory / "result.json"
    result = bind_candidate_result_semantic_sha256(
        {
            "candidate_id": candidate_id,
            "candidate_sha256": candidate_digest,
            "campaign_kind": "actual_contact_grasp_pose_measured_finalization",
            "complete": True,
            "measured_grasp_pose_success": True,
            "grasp_success": True,
            "actual_grasp_pose_qpos_rad": {
                THUMB_BEND: actual_thumb_rad,
            },
            "artifacts": {
                "resolved_config": config_path.name,
                "trace": trace_path.name,
                "trace_retained": True,
                "sha256": {
                    "resolved_config": _sha256(config_path),
                    "trace": _sha256(trace_path),
                },
            },
        }
    )
    result_path.write_text(
        json.dumps(
            result,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        # The measured-stage report defines this path relative to dynamic/,
        # not relative to dynamic/measured/ where the report itself lives.
        "artifact_directory": f"measured/candidate_{candidate_id}",
        "candidate_id": candidate_id,
        "candidate_sha256": candidate_digest,
        "measured_grasp_pose_success": True,
        "result_semantic_sha256": result["result_semantic_sha256"],
    }


@pytest.fixture
def measured_report(tmp_path: Path) -> Path:
    dynamic = tmp_path / "campaign" / "dynamic"
    measured = dynamic / "measured"
    measured.mkdir(parents=True)
    # Deliberately write a non-canonical order.  The user-visible numbering
    # must not depend on worker completion order or filesystem traversal.
    records = [
        _write_measured_candidate(
            dynamic,
            candidate_id=ID_73_SECOND,
            edge_mm=73,
            actual_thumb_rad=1.51,
        ),
        _write_measured_candidate(
            dynamic,
            candidate_id=ID_72,
            edge_mm=72,
            actual_thumb_rad=1.45,
        ),
        _write_measured_candidate(
            dynamic,
            candidate_id=ID_73_FIRST,
            edge_mm=73,
            actual_thumb_rad=1.49,
        ),
    ]
    report = measured / "expanded_report.json"
    report.write_text(
        json.dumps(
            {
                "actual_contact_measured_grasp_pose_stage_schema_version": 1,
                "complete": True,
                "stage": "expanded",
                "measured_grasp_pose_success_count": len(records),
                "candidate_records": records,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def test_measured_data_listing_is_stable_and_contains_selection_metadata(
    measured_report: Path,
):
    first = discover_measured_viewer_data(measured_report)
    second = discover_measured_viewer_data(measured_report)

    assert first == second
    assert [entry.data_index for entry in first] == [1, 2, 3]
    assert [entry.candidate_id for entry in first] == [
        ID_72,
        ID_73_FIRST,
        ID_73_SECOND,
    ]
    assert [entry.edge_mm for entry in first] == pytest.approx([72.0, 73.0, 73.0])
    assert [entry.edge_rank for entry in first] == [1, 1, 2]
    assert [entry.actual_thumb_bend_rad for entry in first] == pytest.approx(
        [1.45, 1.49, 1.51]
    )
    assert all(entry.measured_grasp_pose_success for entry in first)


def test_measured_data_can_be_selected_by_one_based_index(
    measured_report: Path,
):
    source = resolve_measured_viewer_source(measured_report, data_index=2)

    assert source.trajectory == f"candidate_{ID_73_FIRST}"
    assert source.config_path.parent.name == f"candidate_{ID_73_FIRST}"
    assert source.trace_path is not None
    assert source.trace_path.parent == source.config_path.parent
    assert source.from_catalog

    for invalid in (0, -1, 4):
        with pytest.raises(ValueError, match="data index"):
            resolve_measured_viewer_source(measured_report, data_index=invalid)


def test_measured_data_can_be_selected_by_full_candidate_id_without_float_loss(
    measured_report: Path,
):
    # Production candidate IDs are around 1.6e15.  Selection must retain the
    # complete integer rather than round-tripping through a float.
    source = resolve_measured_viewer_source(
        measured_report,
        candidate_id=str(ID_73_SECOND),
    )

    assert source.config_path.parent.name == f"candidate_{ID_73_SECOND}"
    with pytest.raises(ValueError, match="candidate ID"):
        resolve_measured_viewer_source(
            measured_report,
            candidate_id="1616000112009877.0",
        )
    with pytest.raises(ValueError, match="candidate ID"):
        resolve_measured_viewer_source(measured_report, candidate_id="999")


def test_edge_selection_requires_an_explicit_rank_when_multiple_data_match(
    measured_report: Path,
):
    unique = resolve_measured_viewer_source(measured_report, edge_mm=72.0)
    assert unique.config_path.parent.name == f"candidate_{ID_72}"

    with pytest.raises(ValueError, match="edge.*rank|multiple|ambiguous"):
        resolve_measured_viewer_source(measured_report, edge_mm=73.0)
    first = resolve_measured_viewer_source(
        measured_report,
        edge_mm=73.0,
        edge_rank=1,
    )
    second = resolve_measured_viewer_source(
        measured_report,
        edge_mm=73.0,
        edge_rank=2,
    )
    assert first.config_path.parent.name == f"candidate_{ID_73_FIRST}"
    assert second.config_path.parent.name == f"candidate_{ID_73_SECOND}"
    with pytest.raises(ValueError, match="edge rank"):
        resolve_measured_viewer_source(
            measured_report,
            edge_mm=73.0,
            edge_rank=3,
        )


def test_measured_selectors_are_mutually_exclusive(measured_report: Path):
    with pytest.raises(ValueError, match="exactly one|mutually exclusive"):
        resolve_measured_viewer_source(
            measured_report,
            data_index=1,
            candidate_id=str(ID_72),
        )
    with pytest.raises(ValueError, match="edge rank"):
        resolve_measured_viewer_source(measured_report, edge_rank=1)


@pytest.mark.parametrize("tampered_member", ["resolved_config.json", "trace.npz"])
def test_measured_selection_rejects_tampered_hashed_artifacts(
    measured_report: Path,
    tampered_member: str,
):
    target = (
        measured_report.parent
        / f"candidate_{ID_72}"
        / tampered_member
    )
    target.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        resolve_measured_viewer_source(
            measured_report,
            candidate_id=str(ID_72),
        )


def test_measured_selection_rejects_semantically_tampered_result(
    measured_report: Path,
):
    result_path = (
        measured_report.parent
        / f"candidate_{ID_72}"
        / "result.json"
    )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["actual_grasp_pose_qpos_rad"][THUMB_BEND] = 1.59
    # Preserve the stale result_semantic_sha256 to model an in-place edit.
    result_path.write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="semantic SHA-256 mismatch"):
        resolve_measured_viewer_source(
            measured_report,
            candidate_id=str(ID_72),
        )


def test_view_list_data_prints_entries_without_starting_gui(
    measured_report: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    monkeypatch.setattr(
        cli,
        "simulate_in_viewer",
        lambda *args, **kwargs: pytest.fail("--list-data must not start the Viewer"),
    )
    monkeypatch.setattr(
        cli,
        "replay_in_viewer",
        lambda *args, **kwargs: pytest.fail("--list-data must not replay states"),
    )
    args = cli.build_parser().parse_args(
        [
            "view",
            "--measured-report",
            str(measured_report),
            "--list-data",
        ]
    )

    assert args.func(args) == 0
    output = capsys.readouterr().out
    assert "candidate_id" in output
    assert "edge_mm" in output
    assert "actual_thumb_bend_rad" in output
    assert output.index(str(ID_72)) < output.index(str(ID_73_FIRST))
    assert output.index(str(ID_73_FIRST)) < output.index(str(ID_73_SECOND))


def test_cli_forwards_edge_and_rank_without_using_edge_override(
    measured_report: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    selected = ViewerSource(
        Path(f"candidate_{ID_73_FIRST}/resolved_config.json"),
        Path(f"candidate_{ID_73_FIRST}/trace.npz"),
        f"candidate_{ID_73_FIRST}",
        True,
    )
    seen: dict[str, object] = {}

    def fake_resolve(report_path, **kwargs):
        seen.update(report_path=report_path, **kwargs)
        return selected

    monkeypatch.setattr(cli, "resolve_measured_viewer_source", fake_resolve)
    monkeypatch.setattr(cli, "load_config", lambda path: {"config": True})
    monkeypatch.setattr(
        cli,
        "apply_viewer_overrides",
        lambda config, **kwargs: (
            seen.update(viewer_overrides=kwargs) or config,
            False,
        ),
    )
    monkeypatch.setattr(
        cli,
        "simulate_in_viewer",
        lambda source, config, **kwargs: SimpleNamespace(exit_code=0),
    )
    args = cli.build_parser().parse_args(
        [
            "view",
            "--measured-report",
            str(measured_report),
            "--select-edge-mm",
            "73",
            "--edge-rank",
            "1",
        ]
    )

    assert args.func(args) == 0
    assert seen["edge_mm"] == 73.0
    assert seen["edge_rank"] == 1
    # --select-edge-mm selects existing evidence.  The older --edge-mm flag
    # remains a material/geometry override and must not be populated here.
    assert seen["viewer_overrides"]["edge_mm"] is None
