"""Export independently auditable, successful object-trajectory catalogs.

The catalog is deliberately downstream of robustness.  It never invents a
new material case: selected zero-based grid indices are resolved through the
same registered robustness generator and must already be recorded as passing
in a completed robustness report.  Every selected case is then simulated
again, and the complete catalog is published only if every rerun passes both
the hard-check summary and the schema-v3 full-success stage verdict.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .artifacts import (
    file_sha256,
    resolved_run_config,
    run_metadata,
    write_json,
)
from .config import ACTIVE_FINGERS, load_config, validate_config
from .controller import quaternion_drift_deg
from .scene import cube_inertia, rpy_degrees_to_quaternion
from .search import robustness_cases
from .simulation import run_simulation


CATALOG_SCHEMA_VERSION = 1
CSV_COLUMNS = (
    "time_s",
    "control_state",
    "cube_pos_x_m",
    "cube_pos_y_m",
    "cube_pos_z_m",
    "cube_quat_w",
    "cube_quat_x",
    "cube_quat_y",
    "cube_quat_z",
    "cube_linear_velocity_x_m_s",
    "cube_linear_velocity_y_m_s",
    "cube_linear_velocity_z_m_s",
    "cube_angular_velocity_x_rad_s",
    "cube_angular_velocity_y_rad_s",
    "cube_angular_velocity_z_rad_s",
    "manipulation_progress",
    "support_contact",
    "floor_contact",
    "thumb_target_face_effective",
    "index_target_face_effective",
    "mid_target_face_effective",
)
_SAFE_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def _completed_grid(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return a structurally complete, canonically indexed robustness grid."""

    grid = report.get("grid")
    if not isinstance(grid, list):
        raise ValueError("robustness report is missing its grid list")
    try:
        declared_count = int(report["grid_case_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("robustness report is missing grid_case_count") from exc
    if declared_count != len(grid):
        raise ValueError(
            "incomplete robustness grid: "
            f"declared {declared_count}, found {len(grid)} records"
        )
    if any(
        not isinstance(record, dict) or record.get("grid_index") != index
        for index, record in enumerate(grid)
    ):
        raise ValueError("robustness grid indices must be the canonical 0..N-1 order")

    if "grid_passes" not in report:
        raise ValueError("robustness report is missing grid_passes")
    actual_passes = sum(bool(record.get("passed", False)) for record in grid)
    if int(report["grid_passes"]) != actual_passes:
        raise ValueError("robustness grid_passes does not match the grid records")

    perturbations = report.get("perturbations")
    if not isinstance(perturbations, list):
        raise ValueError("robustness report is missing its perturbation list")
    try:
        perturbation_count = int(report["perturbation_trial_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "robustness report is missing perturbation_trial_count"
        ) from exc
    if perturbation_count != len(perturbations):
        raise ValueError(
            "incomplete robustness perturbations: "
            f"declared {perturbation_count}, found {len(perturbations)} records"
        )
    return grid


def _require_validated_v3_config(config: Mapping[str, Any]) -> None:
    if int(config.get("schema_version", 1)) < 3:
        raise ValueError("trajectory catalogs require a schema-v3 experiment")
    status = config.get("experiment_status")
    if not isinstance(status, Mapping) or not bool(status.get("passed", False)):
        raise ValueError("base config must be a validated campaign configuration")
    if not bool(status.get("full_success", False)):
        raise ValueError("base config must record experiment_status.full_success")


def select_passing_grid_cases(
    base_config: dict[str, Any],
    robustness_report: Mapping[str, Any],
    grid_indices: Iterable[int],
) -> list[dict[str, Any]]:
    """Resolve explicitly requested passing grid cases in deterministic order.

    Returned records contain a deep-copied runnable ``config`` and the matching
    report ``record``.  Input ordering has no effect; duplicate indices are
    rejected rather than silently producing two identically named artifacts.
    """

    validate_config(base_config)
    _require_validated_v3_config(base_config)
    report_config = robustness_report.get("config")
    if not isinstance(report_config, dict) or report_config != base_config:
        raise ValueError("robustness report config does not match the base config")
    grid = _completed_grid(robustness_report)

    requested = [int(index) for index in grid_indices]
    if not requested:
        raise ValueError("at least one --grid-index is required")
    if len(requested) != len(set(requested)):
        raise ValueError("duplicate grid indices are not allowed")
    requested.sort()

    try:
        seed = int(robustness_report["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("robustness report is missing a valid seed") from exc
    generated_grid, _ = robustness_cases(base_config, seed)
    if len(generated_grid) != len(grid):
        raise ValueError(
            "registered robustness grid no longer matches the completed report"
        )

    selected: list[dict[str, Any]] = []
    for grid_index in requested:
        if not 0 <= grid_index < len(grid):
            raise ValueError(f"grid index is out of range: {grid_index}")
        record = grid[grid_index]
        stage_status = record.get("stage_status")
        if not bool(record.get("passed", False)):
            raise ValueError(f"grid index {grid_index} is not a passing case")
        if not isinstance(stage_status, Mapping) or not bool(
            stage_status.get("full_success", False)
        ):
            raise ValueError(
                f"grid index {grid_index} does not record stage_status.full_success"
            )

        case_config = generated_grid[grid_index]
        cube = case_config["cube"]
        for key in ("edge_m", "mass_kg", "friction"):
            if key not in record or not math.isclose(
                float(record[key]),
                float(cube[key]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"grid index {grid_index} {key} disagrees with the "
                    "registered robustness generator"
                )
        selected.append(
            {
                "grid_index": grid_index,
                "record": copy.deepcopy(record),
                "config": copy.deepcopy(case_config),
            }
        )
    return selected


def object_physical_parameters(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return explicit model parameters and the compiled initial poses."""

    cube = config["cube"]
    scene = config["scene"]
    edge_m = float(cube["edge_m"])
    mass_kg = float(cube["mass_kg"])
    friction = float(cube["friction"])
    center_x, center_y = [float(value) for value in cube["center_xy_m"]]
    center_z = (
        float(scene["support_top_z_m"])
        + edge_m / 2.0
        + float(cube.get("z_offset_m", 0.0))
    )
    cube_rpy = [float(value) for value in cube.get("rpy_deg", (0.0, 0.0, 0.0))]
    root = config["hand_pose"]
    root_rpy = [float(value) for value in root["rpy_deg"]]
    solref_timeconst = float(cube.get("solref_timeconst_s", 0.004))
    cube_quaternion = rpy_degrees_to_quaternion(cube_rpy)
    root_quaternion = rpy_degrees_to_quaternion(root_rpy)
    root_w, root_x, root_y, root_z = root_quaternion
    root_rotation = np.asarray(
        [
            [
                1.0 - 2.0 * (root_y**2 + root_z**2),
                2.0 * (root_x * root_y - root_z * root_w),
                2.0 * (root_x * root_z + root_y * root_w),
            ],
            [
                2.0 * (root_x * root_y + root_z * root_w),
                1.0 - 2.0 * (root_x**2 + root_z**2),
                2.0 * (root_y * root_z - root_x * root_w),
            ],
            [
                2.0 * (root_x * root_z - root_y * root_w),
                2.0 * (root_y * root_z + root_x * root_w),
                1.0 - 2.0 * (root_x**2 + root_y**2),
            ],
        ],
        dtype=np.float64,
    )
    cube_position = np.asarray([center_x, center_y, center_z], dtype=np.float64)
    root_position = np.asarray(root["translation_m"], dtype=np.float64)
    cube_in_root_position = root_rotation.T @ (cube_position - root_position)
    # Hamilton product conjugate(root) * cube, all persisted as wxyz.
    inverse_root = np.asarray(
        [root_w, -root_x, -root_y, -root_z], dtype=np.float64
    )
    iw, ix, iy, iz = inverse_root
    cw, cx, cy, cz = cube_quaternion
    cube_in_root_quaternion = np.asarray(
        [
            iw * cw - ix * cx - iy * cy - iz * cz,
            iw * cx + ix * cw + iy * cz - iz * cy,
            iw * cy - ix * cz + iy * cw + iz * cx,
            iw * cz + ix * cy - iy * cx + iz * cw,
        ],
        dtype=np.float64,
    )
    cube_in_root_quaternion /= np.linalg.norm(cube_in_root_quaternion)
    return {
        "shape": "cube",
        "edge_m": edge_m,
        "half_size_m": [edge_m / 2.0] * 3,
        "mass_kg": mass_kg,
        "density_kg_m3": mass_kg / edge_m**3,
        "inertia_diagonal_kg_m2": cube_inertia(edge_m, mass_kg).tolist(),
        "contact": {
            "friction_sliding_torsional_rolling": [
                friction,
                0.005,
                0.0001,
            ],
            "condim": 4,
            "contype": 1,
            "conaffinity": 1,
            "priority": 10,
            "solref": [solref_timeconst, 1.0],
            "solref_timeconst_s": solref_timeconst,
            "solimp": [0.9, 0.95, 0.001, 0.5, 2.0],
        },
        "joint": "freejoint",
        "explicit_inertial": True,
        "initial_cube_pose": {
            "position_m": [center_x, center_y, center_z],
            "rpy_deg": cube_rpy,
            "quaternion_wxyz": cube_quaternion.tolist(),
        },
        "fixed_hand_root_pose": {
            "position_m": [float(value) for value in root["translation_m"]],
            "rpy_deg": root_rpy,
            "quaternion_wxyz": root_quaternion.tolist(),
        },
        "initial_cube_in_hand_root_pose": {
            "position_m": cube_in_root_position.tolist(),
            "quaternion_wxyz": cube_in_root_quaternion.tolist(),
        },
    }


def _load_trace(trace_path: Path) -> dict[str, np.ndarray]:
    with np.load(trace_path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def _trace_columns(traces: Mapping[str, np.ndarray]) -> tuple[np.ndarray, ...]:
    time = np.asarray(traces["time"], dtype=np.float64)
    total_steps = int(time.shape[0])
    if time.ndim != 1 or total_steps <= 0:
        raise ValueError("trace time must be a non-empty one-dimensional array")

    cube_pos = np.asarray(traces["cube_pos"], dtype=np.float64)
    cube_quat = np.asarray(traces["cube_quat"], dtype=np.float64)
    cube_velocity = np.asarray(traces["cube_velocity"], dtype=np.float64)
    state = np.asarray(traces["control_state"]).astype(str)
    progress = np.asarray(traces["manipulation_progress"], dtype=np.float64)
    support = np.asarray(traces["support_contact"], dtype=bool)
    floor = np.asarray(traces["floor_contact"], dtype=bool)
    target_effective = np.asarray(traces["target_face_effective"], dtype=bool)
    expected_shapes = {
        "cube_pos": (total_steps, 3),
        "cube_quat": (total_steps, 4),
        "cube_velocity": (total_steps, 6),
        "control_state": (total_steps,),
        "manipulation_progress": (total_steps,),
        "support_contact": (total_steps,),
        "floor_contact": (total_steps,),
        "target_face_effective": (total_steps, 3),
    }
    actual = {
        "cube_pos": cube_pos.shape,
        "cube_quat": cube_quat.shape,
        "cube_velocity": cube_velocity.shape,
        "control_state": state.shape,
        "manipulation_progress": progress.shape,
        "support_contact": support.shape,
        "floor_contact": floor.shape,
        "target_face_effective": target_effective.shape,
    }
    for name, expected in expected_shapes.items():
        if actual[name] != expected:
            raise ValueError(
                f"trace {name} must have shape {expected}, found {actual[name]}"
            )
    if not all(
        np.isfinite(values).all()
        for values in (time, cube_pos, cube_quat, cube_velocity, progress)
    ):
        raise ValueError("trajectory trace contains NaN or Inf")
    finger_order = tuple(str(value) for value in np.asarray(traces["finger_order"]))
    if finger_order != tuple(ACTIVE_FINGERS):
        raise ValueError("trace finger_order is not the canonical active-finger axis")
    return (
        time,
        state,
        cube_pos,
        cube_quat,
        cube_velocity,
        progress,
        support,
        floor,
        target_effective,
    )


def write_object_trajectory_csv(
    output_path: str | Path, traces: Mapping[str, np.ndarray]
) -> None:
    """Write a stable row-for-row projection of the authoritative NPZ trace."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = _trace_columns(traces)
    time, state, pos, quat, velocity, progress, support, floor, effective = columns
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent, prefix=f".{output.name}.", suffix=".tmp", text=True
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(CSV_COLUMNS)
            for step in range(len(time)):
                numeric = (
                    time[step],
                    *pos[step],
                    *quat[step],
                    *velocity[step],
                    progress[step],
                )
                values = [format(float(value), ".17g") for value in numeric]
                writer.writerow(
                    [
                        values[0],
                        state[step],
                        *values[1:4],
                        *values[4:8],
                        *values[8:11],
                        *values[11:14],
                        values[14],
                        int(support[step]),
                        int(floor[step]),
                        int(effective[step, 0]),
                        int(effective[step, 1]),
                        int(effective[step, 2]),
                    ]
                )
        temporary.chmod(0o644)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _trace_scalar_step(
    traces: Mapping[str, np.ndarray], name: str, total_steps: int
) -> int:
    value = np.asarray(traces[name])
    if value.shape != ():
        raise ValueError(f"trace {name} must be a scalar event step")
    step = int(value)
    if not 0 <= step < total_steps:
        raise ValueError(f"successful trace has invalid {name}: {step}")
    return step


def trajectory_keyframes(
    traces: Mapping[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    """Summarize five semantic frames relative to the verified grasp pose."""

    time, _, pos, quat, _, _, _, _, _ = _trace_columns(traces)
    total_steps = len(time)
    acquired = _trace_scalar_step(traces, "grasp_acquisition_step", total_steps)
    manipulation_start = _trace_scalar_step(
        traces, "manipulation_start_step", total_steps
    )
    manipulation_end = _trace_scalar_step(
        traces, "manipulation_end_step", total_steps
    )
    if manipulation_start != acquired + 1:
        raise ValueError("manipulation must start one sample after grasp acquisition")
    if manipulation_end < manipulation_start:
        raise ValueError("manipulation end precedes manipulation start")

    baseline_pos = pos[acquired]
    baseline_quat = quat[acquired]
    events = {
        "initial": 0,
        "grasp_acquired": acquired,
        "manipulation_start": manipulation_start,
        "manipulation_end": manipulation_end,
        "final": total_steps - 1,
    }
    result: dict[str, dict[str, Any]] = {}
    for label, step in events.items():
        delta = pos[step] - baseline_pos
        result[label] = {
            "step": int(step),
            "time_s": float(time[step]),
            "cube_pose": {
                "position_m": pos[step].tolist(),
                "quaternion_wxyz": quat[step].tolist(),
            },
            "displacement_from_operation_baseline": {
                "position_xyz_m": delta.tolist(),
                "translation_norm_m": float(np.linalg.norm(delta)),
                "orientation_deg": quaternion_drift_deg(
                    baseline_quat, quat[step]
                ),
            },
        }
    return result


def _result_success(summary: Mapping[str, Any]) -> bool:
    stage = summary.get("stage_status")
    return bool(
        summary.get("passed", False)
        and isinstance(stage, Mapping)
        and stage.get("full_success", False)
    )


def normalize_trajectory_labels(
    selected_indices: Iterable[int],
    labels: Mapping[int, str] | None,
) -> dict[int, str]:
    """Validate optional display labels without using them as path names."""

    selected = {int(index) for index in selected_indices}
    if labels is None:
        return {}
    normalized: dict[int, str] = {}
    for raw_index, raw_label in labels.items():
        index = int(raw_index)
        label = str(raw_label)
        if index not in selected:
            raise ValueError(f"label refers to unselected grid index {index}")
        if not _SAFE_LABEL.fullmatch(label):
            raise ValueError(
                "trajectory labels must match "
                "[A-Za-z0-9][A-Za-z0-9_-]{0,63}"
            )
        if label in normalized.values():
            raise ValueError(f"duplicate trajectory label: {label}")
        normalized[index] = label
    return normalized


def export_trajectory_catalog(
    base_config_path: str | Path,
    robustness_path: str | Path,
    grid_indices: Sequence[int],
    output_dir: str | Path,
    *,
    video: bool = False,
    labels: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    """Rerun selected passing grid cases and transactionally publish a catalog."""

    base_path = Path(base_config_path).resolve()
    report_path = Path(robustness_path).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(
            f"output directory already exists: {destination}; choose a new output"
        )
    base_config = load_config(base_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("robustness JSON root must be an object")
    selections = select_passing_grid_cases(base_config, report, grid_indices)
    resolved_labels = normalize_trajectory_labels(
        (selection["grid_index"] for selection in selections), labels
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = run_metadata(base_path)
    metadata.update(
        {
            "base_config": str(base_path),
            "base_config_sha256": file_sha256(base_path),
            "robustness_report": str(report_path),
            "robustness_report_sha256": file_sha256(report_path),
        }
    )
    width = max(3, len(str(max(selection["grid_index"] for selection in selections))))
    catalog_entries: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(
        dir=destination.parent, prefix=f".{destination.name}.staging."
    ) as staging_name:
        staging = Path(staging_name)
        for selection in selections:
            grid_index = int(selection["grid_index"])
            trajectory_id = f"grid_{grid_index:0{width}d}"
            display_label = resolved_labels.get(grid_index, trajectory_id)
            trajectory_dir = staging / trajectory_id
            trajectory_dir.mkdir()
            trace_path = trajectory_dir / "trace.npz"
            video_path = trajectory_dir / "trajectory.mp4" if video else None
            summary = run_simulation(
                selection["config"],
                trace_path=trace_path,
                video_path=video_path,
            )
            if not _result_success(summary):
                failed = summary.get("failed_checks", [])
                raise RuntimeError(
                    f"grid index {grid_index} did not reproduce full success; "
                    f"failed_checks={failed!r}"
                )

            traces = _load_trace(trace_path)
            csv_path = trajectory_dir / "object_trajectory.csv"
            write_object_trajectory_csv(csv_path, traces)
            keyframes = trajectory_keyframes(traces)
            resolved = resolved_run_config(selection["config"], summary)
            config_path = trajectory_dir / "resolved_config.json"
            write_json(config_path, resolved)
            physical = object_physical_parameters(resolved)
            artifacts = {
                "resolved_config": config_path.name,
                "trace": trace_path.name,
                "object_trajectory_csv": csv_path.name,
                "video": video_path.name if video_path is not None else None,
                "sha256": {
                    "resolved_config": file_sha256(config_path),
                    "trace": file_sha256(trace_path),
                    "object_trajectory_csv": file_sha256(csv_path),
                },
            }
            if video_path is not None:
                artifacts["sha256"]["video"] = file_sha256(video_path)

            report_record = selection["record"]
            selection_summary = {
                "grid_index": grid_index,
                "label": display_label,
                "case_family": report_record.get("case_family"),
                "density_scale": report_record.get("density_scale"),
                "reported_minimum_normalized_acceptance_margin": (
                    report_record.get("minimum_normalized_acceptance_margin")
                ),
            }
            result = {
                "trajectory_catalog_schema_version": CATALOG_SCHEMA_VERSION,
                "trajectory_id": trajectory_id,
                "metadata": {
                    **copy.deepcopy(metadata),
                    "robustness_grid_index": grid_index,
                },
                "selection": selection_summary,
                "physical_parameters": physical,
                "operation_baseline": keyframes["grasp_acquired"],
                "keyframes": keyframes,
                "config": resolved,
                "summary": summary,
                "experiment_status": resolved["experiment_status"],
                "artifacts": artifacts,
            }
            result_path = trajectory_dir / "result.json"
            write_json(result_path, result)

            catalog_entries.append(
                {
                    "trajectory_id": trajectory_id,
                    **selection_summary,
                    "physical_parameters": physical,
                    "stage_status": copy.deepcopy(summary["stage_status"]),
                    "keyframes": keyframes,
                    "metrics": copy.deepcopy(summary.get("metrics", {})),
                    "artifacts": {
                        "directory": trajectory_id,
                        "resolved_config": f"{trajectory_id}/{config_path.name}",
                        "result": f"{trajectory_id}/{result_path.name}",
                        "trace": f"{trajectory_id}/{trace_path.name}",
                        "object_trajectory_csv": f"{trajectory_id}/{csv_path.name}",
                        "video": (
                            f"{trajectory_id}/{video_path.name}"
                            if video_path is not None
                            else None
                        ),
                        "sha256": {
                            **artifacts["sha256"],
                            "result": file_sha256(result_path),
                        },
                    },
                }
            )

        catalog = {
            "trajectory_catalog_schema_version": CATALOG_SCHEMA_VERSION,
            "metadata": metadata,
            "robustness_seed": int(report["seed"]),
            "selected_grid_indices": [
                int(selection["grid_index"]) for selection in selections
            ],
            "labels": {
                str(index): label for index, label in sorted(resolved_labels.items())
            },
            "trajectory_count": len(catalog_entries),
            "all_reruns_full_success": True,
            "trajectories": catalog_entries,
        }
        write_json(staging / "catalog.json", catalog)
        if destination.exists():
            raise FileExistsError(
                f"output directory appeared during execution: {destination}; "
                "refusing to overwrite"
            )
        staging.rename(destination)
    return catalog


__all__ = [
    "CATALOG_SCHEMA_VERSION",
    "CSV_COLUMNS",
    "export_trajectory_catalog",
    "object_physical_parameters",
    "normalize_trajectory_labels",
    "select_passing_grid_cases",
    "trajectory_keyframes",
    "write_object_trajectory_csv",
]
