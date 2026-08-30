from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from xhand_grasp.actual_contact_grasp_pose_catalog import (
    initialize_or_resume_campaign,
)
from xhand_grasp.grasp_pose import canonical_sha256
from xhand_grasp.tuning.contact_constrained_planner import (
    v14_grasp_object_pair_id,
    v14_grasp_pose_id,
    v14_object_config_id,
)
from xhand_grasp.tuning.contact_preserving_contact_mode_pose_rescue import (
    ContactModePoseRescueBudget,
    ContactModePoseSource,
)
from xhand_grasp.tuning.contact_preserving_contact_mode_pose_rescue_campaign import (
    _manifest,
    _validate_search_report,
)
from xhand_grasp.tuning.contact_preserving_time_warp import (
    _time_warp_controller_id,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "grasp_configs/left_opposed_face_palm_down_contact_preserving_planned_lift.json"


def _source(tmp_path: Path) -> ContactModePoseSource:
    root = tmp_path / "source"
    root.mkdir()
    for name, contents in (
        ("resolved_config.json", b"config"),
        ("result.json", b"result"),
        ("trace.npz", b"trace"),
    ):
        (root / name).write_bytes(contents)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["object_config_id"] = v14_object_config_id(config)
    config["grasp_pose_id"] = v14_grasp_pose_id(config)
    config["grasp_object_pair_id"] = v14_grasp_object_pair_id(config)
    config["planner_id"] = canonical_sha256({"planner": 1})
    config["controller_id"] = _time_warp_controller_id(config)
    return ContactModePoseSource(
        root=root,
        candidate_id=15000000000000001,
        config=config,
        result={"grasp_success": True, "full_success": False},
        config_sha256=canonical_sha256(config),
        result_semantic_sha256="1" * 64,
        trace_sha256="2" * 64,
        source_authentication_id="3" * 64,
        trace_diagnostics={"taxel_count_transition_count": {"thumb": 2}},
    )


def _record(candidate_id: int, physical: str, *, baseline: bool) -> dict:
    return {
        "candidate_id": candidate_id,
        "physical_config_sha256": physical,
        "is_exact_parent_reproduction_baseline": baseline,
        "contact_mode_rank": 0 if baseline else 1,
        "contact_mode_diagnostics": {
            "measurement_available": True,
            "taxel_count_transition_count": {"thumb": 1, "index": 0, "mid": 0},
        },
    }


def test_manifest_binds_budget_source_files_and_implementation(tmp_path: Path) -> None:
    source = _source(tmp_path)
    budget = ContactModePoseRescueBudget(candidate_count=8)
    manifest = _manifest(CONFIG, source, budget)
    assert manifest["campaign_kind"] == "contact_preserving_contact_mode_pose_rescue"
    assert manifest["budget"]["candidate_count"] == 8
    assert manifest["budget"]["root_z_negative_continuation_m"] == [
        -0.0004,
        -0.0006,
        -0.0008,
        -0.001,
    ]
    assert set(manifest["source_evidence_sha256"]) == {
        "resolved_config.json",
        "result.json",
        "trace.npz",
    }
    assert set(manifest["implementation_sha256"]) == {
        "contact_preserving_contact_mode_pose_rescue.py",
        "contact_preserving_contact_mode_pose_rescue_campaign.py",
    }
    assert manifest["campaign_input_sha256"] == canonical_sha256(
        {key: value for key, value in manifest.items() if key != "campaign_input_sha256"}
    )


def test_manifest_change_is_rejected_by_atomic_resume(tmp_path: Path) -> None:
    source = _source(tmp_path)
    budget = ContactModePoseRescueBudget(candidate_count=4)
    first = _manifest(CONFIG, source, budget)
    workspace = tmp_path / "campaign"
    initialize_or_resume_campaign(workspace, first, resume=False)
    (source.root / "trace.npz").write_bytes(b"tampered")
    changed = _manifest(CONFIG, source, budget)
    with pytest.raises(RuntimeError, match="campaign_input_sha256"):
        initialize_or_resume_campaign(workspace, changed, resume=True)


def test_search_report_requires_one_baseline_and_physical_uniqueness() -> None:
    valid = {
        "complete": True,
        "candidate_count": 2,
        "records": [
            _record(1, "1" * 64, baseline=True),
            _record(2, "2" * 64, baseline=False),
        ],
    }
    _validate_search_report(valid, expected_count=2)

    duplicate = copy.deepcopy(valid)
    duplicate["records"][1]["physical_config_sha256"] = "1" * 64
    with pytest.raises(RuntimeError, match="physical uniqueness"):
        _validate_search_report(duplicate, expected_count=2)

    no_baseline = copy.deepcopy(valid)
    no_baseline["records"][0]["is_exact_parent_reproduction_baseline"] = False
    with pytest.raises(RuntimeError, match="exact-parent baseline"):
        _validate_search_report(no_baseline, expected_count=2)
