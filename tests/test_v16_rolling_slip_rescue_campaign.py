from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from xhand_grasp.artifacts import file_sha256
from xhand_grasp.config import load_config
from xhand_grasp.viewer import resolve_viewer_source
from xhand_grasp.tuning.rolling_slip_rescue_campaign import (
    DEFAULT_ALIGNMENT_GAINS,
    DEFAULT_SLIP_RECOVERY_GAINS_RAD_PER_M,
    DEFAULT_SOURCE_CANDIDATE_ID,
    DEFAULT_SOURCE_CONFIG,
    authenticate_source_evidence,
    promote_v15_source_config,
    run_rolling_slip_rescue_campaign,
)


SOURCE = Path(DEFAULT_SOURCE_CONFIG)
SOURCE_TRACE = SOURCE.with_name("trace.npz")
SOURCE_CENTER = 0.9325871364810551


def _summary(config: dict, *, trace_path: Path | None = None) -> dict:
    feedback = config["joint_pair_feedback"]
    alignment = float(feedback["alignment_gain"])
    slip = float(feedback["slip_recovery_gain_rad_per_m"])
    selected = abs(alignment - SOURCE_CENTER) < 1e-14 and slip == 4.0
    contact = 1.0 if selected else 0.95 + 0.01 * (slip / 8.0)
    progress = 1.0 if selected else alignment
    lift = 0.011 if selected else 0.004 + 0.002 * alignment
    if trace_path is not None:
        np.savez_compressed(
            trace_path,
            time=np.asarray((0.0, 0.001)),
            marker=np.asarray((alignment, slip)),
        )
    return {
        "passed": selected,
        "failed_checks": [] if selected else ["median_lift"],
        "stage_status": {
            "grasp_success": True,
            "full_success": selected,
        },
        "metrics": {
            "operation_median_lift_m": lift,
            "operation_minimum_lift_m": lift - 0.001,
            "contact_preserving_planned_lift": {
                "target_face_effective_duty": {
                    "thumb": contact,
                    "index": contact,
                    "mid": contact,
                },
                "simultaneous_target_face_effective_duty": contact,
                "maximum_plan_progress": progress,
                "final_plan_progress": progress,
                "operation_aborted": not selected,
            },
        },
        "rolling_contact_slip": {
            "per_finger": {
                finger: {"maximum_cumulative_irrecoverable_slip_m": 0.0004}
                for finger in ("thumb", "index", "mid")
            }
        },
    }


def test_default_grid_contains_source_center_and_default_slip_gain() -> None:
    assert SOURCE_CENTER in DEFAULT_ALIGNMENT_GAINS
    assert 4.0 in DEFAULT_SLIP_RECOVERY_GAINS_RAD_PER_M
    assert len(DEFAULT_ALIGNMENT_GAINS) * len(
        DEFAULT_SLIP_RECOVERY_GAINS_RAD_PER_M
    ) == 20


def test_source_promotion_is_read_only_and_rebinds_every_v16_identity() -> None:
    source = load_config(SOURCE)
    before = copy.deepcopy(source)
    evidence = authenticate_source_evidence(
        SOURCE,
        SOURCE_TRACE,
        source_candidate_id=DEFAULT_SOURCE_CANDIDATE_ID,
    )
    promoted = promote_v15_source_config(
        source,
        evidence,
        alignment_gain=SOURCE_CENTER,
        slip_recovery_gain_rad_per_m=4.0,
    )

    assert source == before
    assert promoted["schema_version"] == 16
    assert "rolling_slip" in promoted["experiment_id"]
    assert promoted["joint_pair_feedback"]["schema_version"] == 2
    assert promoted["joint_pair_feedback"]["alignment_gain"] == pytest.approx(
        SOURCE_CENTER
    )
    assert promoted["joint_pair_feedback"][
        "slip_recovery_gain_rad_per_m"
    ] == pytest.approx(4.0)
    for field in (
        "object_config_id",
        "grasp_pose_id",
        "grasp_object_pair_id",
        "planner_id",
        "controller_id",
    ):
        assert isinstance(promoted[field], str) and len(promoted[field]) == 64
        assert promoted[field] != source[field]


def test_campaign_is_resumable_selects_best_and_publishes_viewer_catalog(
    tmp_path: Path,
) -> None:
    output = tmp_path / "rescue"
    source_config_hash = file_sha256(SOURCE)
    source_trace_hash = file_sha256(SOURCE_TRACE)
    calls: list[tuple[float, float, bool]] = []

    def runner(config: dict, *, trace_path: Path | None = None, **_: object):
        feedback = config["joint_pair_feedback"]
        calls.append(
            (
                float(feedback["alignment_gain"]),
                float(feedback["slip_recovery_gain_rad_per_m"]),
                trace_path is not None,
            )
        )
        return _summary(config, trace_path=trace_path)

    outcome = run_rolling_slip_rescue_campaign(
        SOURCE,
        output,
        source_trace_path=SOURCE_TRACE,
        alignment_gains=(0.5, SOURCE_CENTER),
        slip_recovery_gains_rad_per_m=(2.0, 4.0),
        simulation_runner=runner,
    )
    report = outcome["report"]
    assert report["candidate_count"] == 4
    assert report["full_success_count"] == 1
    assert report["selected_grid_metrics"]["full_success"] is True
    assert report["selected_final_metrics"]["full_success"] is True
    assert len(calls) == 5
    assert calls[-1] == (SOURCE_CENTER, 4.0, True)
    assert file_sha256(SOURCE) == source_config_hash
    assert file_sha256(SOURCE_TRACE) == source_trace_hash

    catalog = output / "catalog.json"
    selected = resolve_viewer_source(
        catalog_path=catalog,
        trajectory="best_nominal",
    )
    assert selected.config_path == (output / "best/resolved_config.json").resolve()
    assert selected.trace_path == (output / "best/trace.npz").resolve()

    call_count = len(calls)
    resumed = run_rolling_slip_rescue_campaign(
        SOURCE,
        output,
        source_trace_path=SOURCE_TRACE,
        alignment_gains=(SOURCE_CENTER, 0.5),
        slip_recovery_gains_rad_per_m=(4.0, 2.0),
        simulation_runner=runner,
        resume=True,
    )
    assert len(calls) == call_count
    assert resumed["report"] == report

    with pytest.raises(FileExistsError, match="use --resume"):
        run_rolling_slip_rescue_campaign(
            SOURCE,
            output,
            source_trace_path=SOURCE_TRACE,
            alignment_gains=(0.5, SOURCE_CENTER),
            slip_recovery_gains_rad_per_m=(2.0, 4.0),
            simulation_runner=runner,
        )
    with pytest.raises(RuntimeError, match="resume inputs changed"):
        run_rolling_slip_rescue_campaign(
            SOURCE,
            output,
            source_trace_path=SOURCE_TRACE,
            alignment_gains=(SOURCE_CENTER,),
            slip_recovery_gains_rad_per_m=(2.0, 4.0),
            simulation_runner=runner,
            resume=True,
        )
