from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest

import xhand_grasp.trajectory_catalog as catalog
from xhand_grasp.artifacts import write_json
from xhand_grasp.config import ACTIVE_FINGERS, load_config
from xhand_grasp.search import robustness_cases


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_larger_cube_relative_pose_rescue_validated.json"
)
SEED = 20260821


def _report(config: dict, passing: set[int]) -> dict:
    grid, _ = robustness_cases(config, SEED)
    records = []
    for index, case in enumerate(grid):
        passed = index in passing
        records.append(
            {
                "grid_index": index,
                "edge_m": case["cube"]["edge_m"],
                "mass_kg": case["cube"]["mass_kg"],
                "friction": case["cube"]["friction"],
                "case_family": (
                    "constant_density" if index < 75 else "fixed_20g_control"
                ),
                "density_scale": 1.0 if index < 75 else None,
                "passed": passed,
                "failed_checks": [] if passed else ["synthetic_failure"],
                "stage_status": {
                    "grasp_success": passed,
                    "manipulation_success": passed,
                    "full_success": passed,
                },
                "minimum_normalized_acceptance_margin": 0.2 if passed else -0.2,
            }
        )
    return {
        "seed": SEED,
        "config": copy.deepcopy(config),
        "grid_case_count": len(records),
        "grid_passes": len(passing),
        "grid": records,
        "perturbation_trial_count": 0,
        "perturbations": [],
    }


def _successful_trace(path: Path, *, z_offset: float = 0.0) -> None:
    total = 5
    cube_pos = np.asarray(
        [
            [0.071, -0.027, 0.115 + z_offset],
            [0.0711, -0.0270, 0.1152 + z_offset],
            [0.0712, -0.0269, 0.1160 + z_offset],
            [0.0713, -0.0268, 0.1230 + z_offset],
            [0.0714, -0.0267, 0.1260 + z_offset],
        ],
        dtype=np.float64,
    )
    np.savez_compressed(
        path,
        time=np.arange(1, total + 1, dtype=np.float64) * 0.001,
        control_state=np.asarray(
            ["SETTLE", "VERIFY", "MANIPULATE", "MANIPULATE", "HOLD"]
        ),
        cube_pos=cube_pos,
        cube_quat=np.tile(np.asarray([1.0, 0.0, 0.0, 0.0]), (total, 1)),
        cube_velocity=np.arange(total * 6, dtype=np.float64).reshape(total, 6)
        / 1000.0,
        manipulation_progress=np.asarray([0.0, 0.0, 0.25, 1.0, 1.0]),
        support_contact=np.asarray([True, True, False, False, False]),
        floor_contact=np.zeros(total, dtype=bool),
        target_face_effective=np.asarray(
            [[False, False, False], [True, True, True]] + [[True] * 3] * 3,
            dtype=bool,
        ),
        finger_order=np.asarray(ACTIVE_FINGERS),
        grasp_acquisition_step=np.asarray(1, dtype=np.int64),
        manipulation_start_step=np.asarray(2, dtype=np.int64),
        manipulation_end_step=np.asarray(3, dtype=np.int64),
        termination_step=np.asarray(4, dtype=np.int64),
    )


def _success_summary() -> dict:
    return {
        "passed": True,
        "failed_checks": [],
        "checks": {"synthetic": True},
        "stage_status": {
            "grasp_success": True,
            "manipulation_success": True,
            "full_success": True,
        },
        "metrics": {
            "operation_median_lift_m": 0.011,
            "operation_minimum_lift_m": 0.0105,
        },
    }


def test_select_passing_grid_cases_is_sorted_and_deterministic():
    config = load_config(CONFIG_PATH)
    report = _report(config, {2, 7})

    first = catalog.select_passing_grid_cases(config, report, [7, 2])
    second = catalog.select_passing_grid_cases(config, report, [2, 7])

    assert [item["grid_index"] for item in first] == [2, 7]
    assert first == second
    assert first[0]["config"]["cube"]["edge_m"] == pytest.approx(
        report["grid"][2]["edge_m"]
    )


def test_select_passing_grid_cases_rejects_reported_failure():
    config = load_config(CONFIG_PATH)
    report = _report(config, {2})

    with pytest.raises(ValueError, match="grid index 3 is not a passing case"):
        catalog.select_passing_grid_cases(config, report, [3])


def test_catalog_csv_is_a_row_for_row_npz_projection(tmp_path, monkeypatch):
    config = load_config(CONFIG_PATH)
    report_path = tmp_path / "robustness.json"
    write_json(report_path, _report(config, {2}))

    def fake_run(config, *, trace_path, video_path):
        assert video_path is None
        _successful_trace(Path(trace_path), z_offset=config["cube"]["edge_m"])
        return _success_summary()

    monkeypatch.setattr(catalog, "run_simulation", fake_run)
    monkeypatch.setattr(catalog, "run_metadata", lambda path: {"source": str(path)})
    output = tmp_path / "catalog"
    result = catalog.export_trajectory_catalog(
        CONFIG_PATH, report_path, [2], output
    )

    assert result["selected_grid_indices"] == [2]
    assert result["all_reruns_full_success"] is True
    trajectory_dir = output / "grid_002"
    with np.load(trajectory_dir / "trace.npz", allow_pickle=False) as archive:
        time = archive["time"]
        pos = archive["cube_pos"]
        quat = archive["cube_quat"]
        velocity = archive["cube_velocity"]
        progress = archive["manipulation_progress"]
        support = archive["support_contact"]
        floor = archive["floor_contact"]
        effective = archive["target_face_effective"]
        states = archive["control_state"].astype(str)

    with (trajectory_dir / "object_trajectory.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert tuple(rows[0]) == catalog.CSV_COLUMNS
    assert len(rows) == len(time)
    for index, row in enumerate(rows):
        assert float(row["time_s"]) == time[index]
        assert row["control_state"] == states[index]
        assert [float(row[f"cube_pos_{axis}_m"]) for axis in "xyz"] == pytest.approx(
            pos[index]
        )
        assert [float(row[f"cube_quat_{axis}"]) for axis in "wxyz"] == pytest.approx(
            quat[index]
        )
        assert [
            float(row[f"cube_linear_velocity_{axis}_m_s"]) for axis in "xyz"
        ] == pytest.approx(velocity[index, :3])
        assert [
            float(row[f"cube_angular_velocity_{axis}_rad_s"]) for axis in "xyz"
        ] == pytest.approx(velocity[index, 3:])
        assert float(row["manipulation_progress"]) == progress[index]
        assert bool(int(row["support_contact"])) == support[index]
        assert bool(int(row["floor_contact"])) == floor[index]
        assert [
            bool(int(row[f"{finger}_target_face_effective"]))
            for finger in ACTIVE_FINGERS
        ] == effective[index].tolist()

    persisted = json.loads((trajectory_dir / "result.json").read_text())
    assert persisted["summary"]["passed"] is True
    assert persisted["summary"]["stage_status"]["full_success"] is True
    assert list(persisted["keyframes"]) == [
        "final",
        "grasp_acquired",
        "initial",
        "manipulation_end",
        "manipulation_start",
    ]
    assert persisted["keyframes"]["grasp_acquired"][
        "displacement_from_operation_baseline"
    ]["translation_norm_m"] == pytest.approx(0.0)
    assert persisted["keyframes"]["final"][
        "displacement_from_operation_baseline"
    ]["position_xyz_m"] == pytest.approx(pos[-1] - pos[1])

    physics = persisted["physical_parameters"]
    edge = persisted["config"]["cube"]["edge_m"]
    mass = persisted["config"]["cube"]["mass_kg"]
    assert physics["half_size_m"] == pytest.approx([edge / 2.0] * 3)
    assert physics["density_kg_m3"] == pytest.approx(mass / edge**3)
    assert physics["inertia_diagonal_kg_m2"] == pytest.approx(
        [mass * edge**2 / 6.0] * 3
    )
    assert physics["contact"] == {
        "condim": 4,
        "conaffinity": 1,
        "contype": 1,
        "friction_sliding_torsional_rolling": pytest.approx(
            [persisted["config"]["cube"]["friction"], 0.005, 0.0001]
        ),
        "priority": 10,
        "solref": pytest.approx([0.004, 1.0]),
        "solref_timeconst_s": pytest.approx(0.004),
        "solimp": pytest.approx([0.9, 0.95, 0.001, 0.5, 2.0]),
    }
    assert physics["joint"] == "freejoint"
    assert physics["explicit_inertial"] is True
    root_position = np.asarray(physics["fixed_hand_root_pose"]["position_m"])
    relative_position = np.asarray(
        physics["initial_cube_in_hand_root_pose"]["position_m"]
    )
    root_quaternion = np.asarray(
        physics["fixed_hand_root_pose"]["quaternion_wxyz"]
    )
    w, x, y, z = root_quaternion
    rotation = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    reconstructed_cube_position = root_position + rotation @ relative_position
    assert reconstructed_cube_position == pytest.approx(
        physics["initial_cube_pose"]["position_m"]
    )


def test_optional_labels_are_safe_unique_and_do_not_change_directory_ids():
    assert catalog.normalize_trajectory_labels([2, 7], {7: "heavy", 2: "nominal"}) == {
        7: "heavy",
        2: "nominal",
    }
    with pytest.raises(ValueError, match="unselected"):
        catalog.normalize_trajectory_labels([2], {7: "heavy"})
    with pytest.raises(ValueError, match="must match"):
        catalog.normalize_trajectory_labels([2], {2: "../escape"})
    with pytest.raises(ValueError, match="duplicate trajectory label"):
        catalog.normalize_trajectory_labels([2, 7], {2: "same", 7: "same"})


def test_catalog_refuses_overwrite_and_failure_leaves_no_partial_publish(
    tmp_path, monkeypatch
):
    config = load_config(CONFIG_PATH)
    report_path = tmp_path / "robustness.json"
    write_json(report_path, _report(config, {2, 7}))

    existing = tmp_path / "existing"
    existing.mkdir()
    marker = existing / "keep.txt"
    marker.write_text("keep\n", encoding="utf-8")
    monkeypatch.setattr(
        catalog,
        "run_simulation",
        lambda *args, **kwargs: pytest.fail("simulation must not start"),
    )
    with pytest.raises(FileExistsError, match="already exists"):
        catalog.export_trajectory_catalog(
            CONFIG_PATH, report_path, [2], existing
        )
    assert marker.read_text(encoding="utf-8") == "keep\n"

    calls = 0

    def fail_second(config, *, trace_path, video_path):
        nonlocal calls
        del config, video_path
        calls += 1
        if calls == 1:
            _successful_trace(Path(trace_path))
            return _success_summary()
        return {
            "passed": False,
            "failed_checks": ["synthetic_rerun_failure"],
            "stage_status": {
                "grasp_success": True,
                "manipulation_success": False,
                "full_success": False,
            },
        }

    monkeypatch.setattr(catalog, "run_simulation", fail_second)
    monkeypatch.setattr(catalog, "run_metadata", lambda path: {})
    destination = tmp_path / "atomic-catalog"
    with pytest.raises(RuntimeError, match="did not reproduce full success"):
        catalog.export_trajectory_catalog(
            CONFIG_PATH, report_path, [7, 2], destination
        )
    assert calls == 2
    assert not destination.exists()
    assert not list(tmp_path.glob(".atomic-catalog.staging.*"))
