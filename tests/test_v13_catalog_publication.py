from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from xhand_grasp.actual_contact_grasp_pose_catalog import (
    authenticated_catalog_artifact_paths,
    bind_candidate_result_semantic_sha256,
)
from xhand_grasp.artifacts import file_sha256, write_json
from xhand_grasp.config import load_config
from xhand_grasp.tuning import scaled_contact_downsize_campaign as campaign
from xhand_grasp.tuning.scaled_contact_downsize_catalog import (
    export_scaled_contact_downsize_manipulation_catalog,
    select_scaled_contact_downsize_candidates,
)
from xhand_grasp.viewer import resolve_viewer_source


TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "grasp_configs"
    / "left_opposed_face_palm_down_scaled_centered_spread_actual_grasp_then_lift.json"
)


def _summary() -> dict:
    return {
        "passed": True,
        "failed_checks": [],
        "stage_status": {
            "grasp_success": True,
            "manipulation_success": True,
            "full_success": True,
        },
    }


def _production_candidate(
    root: Path,
    identifier: int,
    edge_mm: int,
    *,
    summary: dict | None = None,
) -> dict:
    member = root / f"source_{identifier}"
    member.mkdir()
    config = load_config(TEMPLATE)
    config["cube"]["edge_m"] = edge_mm / 1000.0
    config.setdefault("candidate_metadata", {})["candidate_id"] = identifier
    config_path = member / "resolved_config.json"
    result_path = member / "result.json"
    trace_path = member / "trace.npz"
    write_json(config_path, config)
    np.savez_compressed(
        trace_path,
        time=np.asarray([0.001, 0.002], dtype=np.float64),
        cube_pos=np.asarray([[0.071, -0.027, 0.08]] * 2, dtype=np.float64),
        video_frame_steps=np.asarray([], dtype=np.int64),
    )
    candidate_summary = copy.deepcopy(summary or _summary())
    result = bind_candidate_result_semantic_sha256(
        {
            "actual_contact_manipulation_candidate_schema_version": 1,
            "complete": True,
            "candidate_id": identifier,
            "summary": candidate_summary,
            "artifacts": {
                "resolved_config": config_path.name,
                "trace": trace_path.name,
                "sha256": {
                    "resolved_config": file_sha256(config_path),
                    "trace": file_sha256(trace_path),
                },
            },
        }
    )
    write_json(result_path, result)
    return {
        "candidate_id": identifier,
        "discovery_index": identifier - 1,
        "config_path": config_path,
        "result_path": result_path,
        "trace_path": trace_path,
        "summary": candidate_summary,
        "downsize_metadata": {
            "edge_m": edge_mm / 1000.0,
            "mapping_mode": (
                "proportional_face_yz" if identifier % 2 else "absolute_face_yz"
            ),
            "source_alias": (
                "best_nominal" if identifier % 2 else "best_pair_center"
            ),
        },
    }


def _slip_summary(*, p95_m: float, maximum_m: float | None = None) -> dict:
    summary = _summary()
    summary["metrics"] = {
        "contact_point_targeting": {
            "contact_slip_from_grasp": {
                "baseline": {
                    finger: {"valid": True}
                    for finger in ("thumb", "index", "mid")
                },
                "operation": {
                    "per_finger": {
                        finger: {
                            "valid_duty": 1.0,
                            "tangent_slip_p95_m": float(p95_m),
                            "tangent_slip_max_m": float(
                                maximum_m
                                if maximum_m is not None
                                else p95_m
                            ),
                        }
                        for finger in ("thumb", "index", "mid")
                    }
                },
            }
        }
    }
    return summary


def _video_evidence() -> dict:
    return {
        "decode_verified": True,
        "codec": "h264",
        "width": 640,
        "height": 480,
        "fps": "30/1",
        "frame_count": 1,
        "duration_s": 1.0 / 30.0,
        "size_bytes": 9,
    }


def test_v13_manipulation_selection_uses_slip_aware_rank():
    config = load_config(TEMPLATE)
    records = [
        {
            "candidate_id": identifier,
            "discovery_index": identifier - 1,
            "config": copy.deepcopy(config),
            "summary": _slip_summary(p95_m=slip),
        }
        for identifier, slip in ((1, 0.004), (2, 0.001))
    ]

    selected = select_scaled_contact_downsize_candidates(
        records,
        kind="manipulation",
        selected_count=2,
    )

    assert [value["candidate_id"] for value in selected.selected] == [2, 1]
    assert (
        selected.metadata["ranking_policy"]
        == "schema_v13_hard_then_soft_contact_slip"
    )


def test_v13_per_edge_manipulation_publication_preserves_slip_rank(tmp_path):
    source_root = tmp_path / "sources"
    source_root.mkdir()
    candidates = [
        _production_candidate(
            source_root,
            identifier,
            88,
            summary=_slip_summary(p95_m=slip),
        )
        for identifier, slip in ((1, 0.004), (2, 0.001))
    ]

    published = campaign._catalog_candidates(
        candidates,
        source_root,
        kind="manipulation",
    )

    assert [value["candidate_id"] for value in published] == [2, 1]


def test_v13_catalog_keeps_all_edges_but_only_renders_key_video_roles(tmp_path):
    source_root = tmp_path / "sources"
    source_root.mkdir()
    candidates = [
        _production_candidate(source_root, identifier, edge_mm)
        for identifier, edge_mm in (
            (1, 80),
            (2, 60),
            (3, 70),
            (4, 88),
            (5, 65),
            (6, 75),
        )
    ]
    source_traces = {
        str(value["candidate_id"]): Path(value["trace_path"])
        for value in candidates
    }
    render_calls: list[int] = []

    def runner(config, *, trace_path, video_path):
        identifier = int(config["candidate_metadata"]["candidate_id"])
        render_calls.append(identifier)
        with np.load(source_traces[str(identifier)], allow_pickle=False) as source:
            np.savez_compressed(
                trace_path,
                time=np.asarray(source["time"]),
                cube_pos=np.asarray(source["cube_pos"]),
                video_frame_steps=np.asarray([1], dtype=np.int64),
            )
        video_path.write_bytes(b"synthetic")
        return {
            **copy.deepcopy(_summary()),
            "video": {**_video_evidence(), "simulation_step_indices": [1]},
        }

    output = tmp_path / "catalog"
    payload = export_scaled_contact_downsize_manipulation_catalog(
        candidates,
        output,
        selected_count=6,
        simulation_runner=runner,
        video_probe=lambda *_args: _video_evidence(),
    )
    campaign._annotate_catalog(output / "catalog.json", candidates, kind="manipulation")

    assert payload["success_count"] == 6
    assert len(payload["trajectories"]) == 6
    assert (
        payload["selection"]["ranking_policy"]
        == "schema_v13_hard_then_soft_contact_slip"
    )
    assert all(
        entry["publication_ranking_policy"]
        == "schema_v13_hard_then_soft_contact_slip"
        and "v13_contact_slip_rank_evidence" in entry
        for entry in payload["trajectories"]
    )
    assert len(render_calls) == 3
    roles = {
        role
        for entry in payload["trajectories"]
        for role in entry["final_video_roles"]
    }
    assert roles == {"best_nominal", "smallest_pass", "hardest_selected_pass"}
    unrendered = [
        entry for entry in payload["trajectories"] if not entry["final_video_required"]
    ]
    assert len(unrendered) == 3
    member_trace = output / unrendered[0]["artifacts"]["trace"]
    source_trace = source_traces[unrendered[0]["candidate_id"]]
    assert member_trace.stat().st_ino == source_trace.stat().st_ino
    assert unrendered[0]["artifacts"]["trace_materialization"] == "hardlink"

    annotated = campaign._annotate_catalog(
        output / "catalog.json", candidates, kind="manipulation"
    )
    assert annotated["successful_edges_mm"] == [60, 65, 70, 75, 80, 88]
    assert annotated["publication_cap_per_edge"] == 3
    assert set(annotated["aliases"]) >= {
        "best_nominal",
        "smallest_lift_pass",
        "edge_60_best",
        "edge_65_best",
        "edge_70_best",
        "edge_75_best",
        "edge_80_best",
        "edge_88_best",
    }
    for edge_mm in annotated["successful_edges_mm"]:
        source = resolve_viewer_source(
            catalog_path=output / "catalog.json",
            trajectory=f"edge_{edge_mm}_best",
        )
        assert source.config_path.is_file()
        assert source.trace_path.is_file()
    authenticated_catalog_artifact_paths(output / "catalog.json")
