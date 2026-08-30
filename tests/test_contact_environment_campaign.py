from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from xhand_grasp.artifacts import file_sha256
from xhand_grasp.contact_environment import (
    ContactEnvironmentSpec,
    requested_environment_snapshot,
)
from xhand_grasp.tuning.contact_environment_campaign import (
    default_contact_environment_cases,
    run_contact_environment_campaign,
)
from xhand_grasp.viewer import resolve_viewer_source


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / (
    "grasp_configs/left_opposed_face_palm_down_joint_pair_near_zero_"
    "contact_preserving_planned_lift.json"
)


def _replace(spec: ContactEnvironmentSpec, **updates: Any) -> ContactEnvironmentSpec:
    values = {
        name: getattr(spec, name)
        for name in ContactEnvironmentSpec.__dataclass_fields__
    }
    values.update(updates)
    return ContactEnvironmentSpec(**values)


class FakeEnvironmentRunner:
    def __init__(self) -> None:
        self.environment_ids: list[str] = []

    def __call__(
        self,
        config: dict[str, Any],
        *,
        trace_path: str | Path,
        contact_environment: ContactEnvironmentSpec,
    ) -> dict[str, Any]:
        del config
        spec = contact_environment
        self.environment_ids.append(spec.environment_id)
        safe = spec.torsional_friction == pytest.approx(0.01)
        slip = 0.0014 if safe else 0.0018
        np.savez_compressed(
            trace_path,
            contact_environment_id=np.asarray(spec.environment_id, dtype=np.str_),
            contact_environment_cube_contact_count=np.asarray(
                [0, 1, 3, 2, 1], dtype=np.int64
            ),
            contact_environment_active_distal_contact_count=np.asarray(
                [0, 1, 2, 2, 1], dtype=np.int64
            ),
        )
        requested = requested_environment_snapshot(spec)
        return {
            "passed": bool(safe),
            "stage_status": {
                "grasp": "acquired",
                "manipulation": "succeeded" if safe else "failed",
                "full_success": bool(safe),
            },
            "metrics": {
                "median_lift_m": 0.0105 if safe else 0.0044,
                "minimum_lift_m": 0.0090 if safe else 0.0043,
                "contact_preserving_planned_lift": {
                    "maximum_plan_progress": 1.0 if safe else 0.44,
                    "final_plan_progress": 1.0 if safe else 0.44,
                },
                "joint_pair_alignment": {
                    "operation_p95_deg": 0.2,
                    "operation_max_deg": 0.25,
                },
                "contact_point_targeting": {
                    "contact_slip_from_grasp": {
                        "operation": {
                            "per_finger": {
                                "thumb": {"tangent_slip_max_m": slip},
                                "index": {"tangent_slip_max_m": 0.0009},
                                "mid": {"tangent_slip_max_m": 0.0010},
                            }
                        }
                    }
                },
            },
            "contact_environment": {
                "requested": requested,
                "compiled": {**requested, "kind": "compiled"},
                "runtime": {"verified_contact_step_count": 4},
            },
        }


def test_default_cases_cover_declared_focused_sweep_without_changing_mu_or_condim() -> None:
    cases = default_contact_environment_cases(SOURCE)
    assert len(cases) == 12
    assert {case.sliding_friction for case in cases} == {0.8}
    assert {case.condim for case in cases} == {4}
    assert {case.impratio for case in cases} >= {10.0, 30.0, 100.0}
    assert {case.iterations for case in cases} >= {100, 200}
    assert {case.tolerance for case in cases} >= {1e-8, 1e-10}
    assert {case.noslip_iterations for case in cases} >= {0, 5, 10, 20}
    assert {case.torsional_friction for case in cases} >= {
        0.0025,
        0.005,
        0.0075,
        0.01,
        0.02,
    }
    assert {(case.solref, case.solimp) for case in cases} == {
        ((0.004, 1.0), (0.9, 0.95, 0.001, 0.5, 2.0))
    }


def test_campaign_is_atomic_ranked_and_catalogs_environment_artifact(
    tmp_path: Path,
) -> None:
    baseline = default_contact_environment_cases(SOURCE)[0]
    material = _replace(baseline, torsional_friction=0.01)
    runner = FakeEnvironmentRunner()
    output = tmp_path / "campaign"
    result = run_contact_environment_campaign(
        SOURCE,
        output,
        [baseline, material],
        simulation_runner=runner,
    )
    assert set(runner.environment_ids) == {
        baseline.environment_id,
        material.environment_id,
    }
    catalog = result["catalog"]
    assert catalog["aliases"]["best_environment"] == catalog["aliases"][
        "best_attempt"
    ]
    best = catalog["trajectories"][0]
    assert best["environment_id"] == material.environment_id
    assert best["classification"] == "material_ablation"
    assert best["metrics"]["tangential_slip_max_m"]["thumb"] == pytest.approx(
        0.0014
    )
    assert best["metrics"]["contact_count"]["cube"] == {
        "minimum": 0,
        "maximum": 3,
        "maximum_step_jump": 2,
        "nonzero_duty": pytest.approx(0.8),
    }
    artifacts = best["artifacts"]
    assert "contact_environment" in artifacts
    assert artifacts["sha256"]["contact_environment"] == file_sha256(
        output / artifacts["contact_environment"]
    )
    assert (output / artifacts["resolved_config"]).is_file()
    assert {"best_attempt", "best_environment"}.issubset(best["aliases"])
    assert "best_same_object" in catalog["aliases"]
    assert "highest_progress" in catalog["aliases"]
    viewer_source = resolve_viewer_source(
        catalog_path=output / "catalog.json", trajectory="best_environment"
    )
    assert viewer_source.config_path == output / artifacts["resolved_config"]
    assert viewer_source.contact_environment_path == (
        output / artifacts["contact_environment"]
    )
    for trajectory in catalog["trajectories"]:
        case_root = output / "cases" / trajectory["trajectory_id"]
        assert {path.name for path in case_root.iterdir()} == {
            "environment.json",
            "result.json",
            "trace.npz",
        }


def test_resume_authenticates_committed_hashes_and_does_not_rerun(
    tmp_path: Path,
) -> None:
    baseline = default_contact_environment_cases(SOURCE)[0]
    output = tmp_path / "campaign"
    first = FakeEnvironmentRunner()
    run_contact_environment_campaign(
        SOURCE, output, [baseline], simulation_runner=first
    )
    assert first.environment_ids == [baseline.environment_id]

    resumed = FakeEnvironmentRunner()
    run_contact_environment_campaign(
        SOURCE,
        output,
        [baseline],
        resume=True,
        simulation_runner=resumed,
    )
    assert resumed.environment_ids == []

    catalog = json.loads((output / "catalog.json").read_text(encoding="utf-8"))
    trace = output / catalog["trajectories"][0]["artifacts"]["trace"]
    trace.write_bytes(trace.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="trace SHA mismatch"):
        run_contact_environment_campaign(
            SOURCE,
            output,
            [baseline],
            resume=True,
            simulation_runner=FakeEnvironmentRunner(),
        )


def test_source_hash_and_case_set_are_resume_inputs(tmp_path: Path) -> None:
    baseline = default_contact_environment_cases(SOURCE)[0]
    with pytest.raises(RuntimeError, match="source config SHA-256"):
        run_contact_environment_campaign(
            SOURCE,
            tmp_path / "wrong_hash",
            [baseline],
            expected_source_sha256="0" * 64,
            simulation_runner=FakeEnvironmentRunner(),
        )

    output = tmp_path / "campaign"
    run_contact_environment_campaign(
        SOURCE, output, [baseline], simulation_runner=FakeEnvironmentRunner()
    )
    changed = _replace(baseline, impratio=30.0)
    with pytest.raises(RuntimeError, match="resume inputs changed"):
        run_contact_environment_campaign(
            SOURCE,
            output,
            [baseline, changed],
            resume=True,
            simulation_runner=FakeEnvironmentRunner(),
        )


def test_case_order_does_not_change_ranking(tmp_path: Path) -> None:
    baseline = default_contact_environment_cases(SOURCE)[0]
    material = _replace(baseline, torsional_friction=0.01)
    left = run_contact_environment_campaign(
        SOURCE,
        tmp_path / "left",
        [baseline, material],
        simulation_runner=FakeEnvironmentRunner(),
    )
    right = run_contact_environment_campaign(
        SOURCE,
        tmp_path / "right",
        [material, baseline],
        simulation_runner=FakeEnvironmentRunner(),
    )
    assert left["report"]["ranked_case_ids"] == right["report"][
        "ranked_case_ids"
    ]
