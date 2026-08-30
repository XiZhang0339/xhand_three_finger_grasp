from __future__ import annotations

import copy
from pathlib import Path

import pytest

from xhand_grasp.config import load_config, validate_config
from xhand_grasp.experiment import resolve_experiment
from xhand_grasp.tuning.actual_contact_grasp_pose_robustness import (
    V9RobustnessSource,
    _derived_seed,
    generate_v9_perturbation_configs,
    run_v9_robustness_campaign,
)
from xhand_grasp.tuning.contact_point_grasp_robustness import (
    V12_CONTACT_POINT_HARD_CHECKS,
    generate_v12_grasp_perturbation_configs,
    run_v12_grasp_perturbation_audit,
)
from xhand_grasp.tuning.scaled_contact_downsize_catalog import (
    select_scaled_contact_downsize_candidates,
)


CONFIG = Path(
    "grasp_configs/"
    "left_opposed_face_palm_down_scaled_centered_spread_actual_grasp_then_lift.json"
)
SOURCE_ID = "13000000000000001"


def _nominal() -> dict:
    value = load_config(CONFIG)
    value["candidate_metadata"] = {"candidate_id": int(SOURCE_ID)}
    validate_config(value)
    return value


def _trial() -> dict:
    base = _nominal()
    seed = _derived_seed(
        resolve_experiment(base).robustness.seed,
        SOURCE_ID,
        16,
    )
    return generate_v9_perturbation_configs(
        base,
        count=1,
        seed=seed,
        source_candidate_id=SOURCE_ID,
        family="per_full_success_local_16",
    )[0]


def test_v13_registered_robustness_trial_is_valid_and_nominal_stays_locked():
    trial = _trial()
    validate_config(trial)

    nominal_override = _nominal()
    nominal_override["cube"]["friction"] += 0.01
    with pytest.raises(ValueError, match="friction must match"):
        validate_config(nominal_override)

    unbound = _nominal()
    unbound["run_context"] = {"kind": "robustness_trial"}
    with pytest.raises(ValueError, match="authenticated robustness metadata"):
        validate_config(unbound)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda value: value["candidate_metadata"]["robustness_trial"].__setitem__(
                "seed",
                value["candidate_metadata"]["robustness_trial"]["seed"] + 1,
            ),
            "seed/source binding",
        ),
        (
            lambda value: value["candidate_metadata"]["robustness_trial"].__setitem__(
                "source_candidate_id", "forged-source"
            ),
            "seed/source binding",
        ),
        (
            lambda value: value["candidate_metadata"]["robustness_trial"].__setitem__(
                "source_config_sha256", "0" * 64
            ),
            "source config binding",
        ),
        (
            lambda value: value["cube"].__setitem__(
                "friction", value["cube"]["friction"] + 0.01
            ),
            "cube state disagrees",
        ),
    ),
)
def test_v13_robustness_trial_rejects_tampered_source_seed_and_physics(
    mutation, message
):
    trial = _trial()
    mutation(trial)
    with pytest.raises(ValueError, match=message):
        validate_config(trial)


def test_v13_robustness_trial_rejects_consistent_but_out_of_envelope_pose():
    trial = _trial()
    metadata = trial["candidate_metadata"]["robustness_trial"]
    previous = float(metadata["resolved_perturbations"]["cube_center_xy_delta_m"][0])
    trial["cube"]["center_xy_m"][0] += 0.010 - previous
    metadata["resolved_perturbations"]["cube_center_xy_delta_m"][0] = 0.010
    with pytest.raises(ValueError, match="XY perturbation is out of range"):
        validate_config(trial)


def test_v13_robustness_source_metadata_presence_is_bound():
    trial = _trial()
    tampered = copy.deepcopy(trial)
    tampered["candidate_metadata"]["robustness_trial"][
        "source_had_candidate_metadata"
    ] = False
    with pytest.raises(ValueError, match="metadata presence binding"):
        validate_config(tampered)


def test_v13_robustness_source_id_and_seed_cannot_be_rebound_together():
    trial = _trial()
    metadata = trial["candidate_metadata"]["robustness_trial"]
    metadata["source_candidate_id"] = "coordinated-forged-source"
    metadata["seed"] = _derived_seed(
        resolve_experiment(trial).robustness.seed,
        metadata["source_candidate_id"],
        16,
    )
    with pytest.raises(ValueError, match="source candidate binding"):
        validate_config(trial)


def test_v13_grasp_only_contact_point_robustness_uses_same_bound_envelope():
    base = _nominal()
    seed = _derived_seed(
        resolve_experiment(base).robustness.seed,
        SOURCE_ID,
        12_016,
    )
    trial = generate_v12_grasp_perturbation_configs(
        base,
        count=1,
        seed=seed,
        source_candidate_id=SOURCE_ID,
        family="v12_grasp_per_nominal_local_16",
    )[0]
    validate_config(trial)
    evidence = trial["candidate_metadata"]["v12_grasp_robustness_trial"]
    assert evidence["point_plan_id"] == trial["contact_point_plan"]["point_plan_id"]
    assert evidence["full_success_required"] is False


def _hard_summary(*, point_checks: bool = True) -> dict:
    return {
        "passed": True,
        "failed_checks": [],
        "checks": {
            name: bool(point_checks) for name in V12_CONTACT_POINT_HARD_CHECKS
        },
        "stage_status": {
            "grasp_success": True,
            "manipulation_success": True,
            "full_success": True,
        },
        "metrics": {},
    }


def test_v13_selection_and_lift_robustness_do_not_truncate_at_five(tmp_path):
    records = [
        {
            "candidate_id": 13_000 + index,
            "discovery_index": index,
            "config": _nominal(),
            "summary": _hard_summary(),
        }
        for index in range(6)
    ]
    selection = select_scaled_contact_downsize_candidates(
        records, kind="manipulation", selected_count=6
    )
    assert len(selection.selected) == 6

    sources = tuple(
        V9RobustnessSource(
            candidate_id=str(value["candidate_id"]),
            config=value["config"],
            summary=value["summary"],
            discovery_index=index,
            best_first=index == 0,
        )
        for index, value in enumerate(records)
    )
    observed = []

    def runner(jobs, _workers):
        observed.extend(jobs)
        return [
            {
                "candidate_id": candidate_id,
                "config": copy.deepcopy(config),
                "summary": _hard_summary(),
            }
            for candidate_id, config in reversed(jobs)
        ]

    report = run_v9_robustness_campaign(
        (tmp_path,),
        tmp_path / "lift.json",
        workers=2,
        local_perturbations=1,
        best_perturbations=1,
        max_nominal_trajectories=6,
        candidate_discoverer=lambda _roots: sources,
        runner=runner,
    )
    assert report["selected_nominal_count"] == 6
    assert len(observed) == 7
    assert len(report["per_nominal"]) == 6


def test_v13_grasp_robustness_audits_every_source_and_rechecks_point_gates(
    tmp_path,
):
    sources = [
        {
            "candidate_id": str(13_100 + index),
            "discovery_index": index,
            "config": _nominal(),
            "summary": _hard_summary(),
            "best": index == 0,
        }
        for index in range(6)
    ]
    observed = []

    def runner(jobs, _workers):
        observed.extend(jobs)
        results = []
        for offset, (candidate_id, config) in enumerate(jobs):
            results.append(
                {
                    "candidate_id": candidate_id,
                    "config": copy.deepcopy(config),
                    # The grasp stage alone is insufficient: the point checks
                    # are deliberately failed for one full-reset perturbation.
                    "summary": _hard_summary(point_checks=offset != 0),
                }
            )
        return results

    report = run_v12_grasp_perturbation_audit(
        sources,
        tmp_path / "grasp.json",
        workers=2,
        local_perturbations=1,
        best_perturbations=1,
        maximum_nominal_grasps=6,
        runner=runner,
    )
    assert report["selected_grasp_count"] == 6
    assert len(observed) == 7
    assert len(report["per_grasp"]) == 6
    assert report["per_grasp"][0]["grasp_passes"] == 0
    assert (
        report["per_grasp"][0]["trials"][0]["grasp_failure_reasons"]
        == list(V12_CONTACT_POINT_HARD_CHECKS)
    )
