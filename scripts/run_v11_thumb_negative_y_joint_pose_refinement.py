#!/usr/bin/env python3
"""Jointly refine wrist pose and thumb shape for a larger -Y contact shift.

This is a deterministic, evidence-producing continuation of the archived
``orbit_neg_1p1_deg`` grasp.  Cube pose and physical properties are copied
byte-for-byte.  Each candidate changes the coupled fixed-root pose and the
thumb rota1 nominal/precontact/preload values, then reruns fresh free-body
dynamics.  A candidate is publishable only when the complete 250 ms measured
grasp window passes the existing schema-v11 hard gates.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xhand_grasp.artifacts import file_sha256, resolved_run_config, write_json
from xhand_grasp.config import ACTIVE_ACTUATORS, load_config, validate_config
from xhand_grasp.simulation import SimulationSession
from xhand_grasp.trajectory import preflight_config
from xhand_grasp.viewer import apply_viewer_overrides

# Reuse the trace-level, force-aware evidence extraction used by the preceding
# fixed-root sweep.  Keeping this in one place prevents subtly different
# definitions of "strict grasp" across the two continuation stages.
from run_v11_thumb_negative_y_refinement import (  # noqa: E402
    FINGERS,
    _candidate_record,
    _cube_signature,
    _local_contact_centroids,
    _mark_override_status,
)


DEFAULT_SOURCE = Path(
    "artifacts/left_opposed_face_palm_down_larger_relative_wrist_pose_"
    "actual_contact_smooth_vertical_lift/tune/"
    "negative_clockwise_orbit_adaptive_fine_sweep_from_4367930796723148114_v1/"
    "orbit_neg_1p1_deg/resolved_config.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/left_opposed_face_palm_down_larger_relative_wrist_pose_"
    "actual_contact_smooth_vertical_lift/tune/"
    "thumb_negative_y_joint_pose_refinement_from_orbit_neg_1p1_v1"
)
THUMB_ROTA1 = "left_hand_thumb_rota_joint1_actuator"


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _candidate_specs() -> tuple[dict[str, Any], ...]:
    """Return a fixed, ordered local design including safe search boundaries."""

    return (
        {
            "label": "joint_pose_shift_9p0mm_rota1_plus_0p02",
            "root_shift_mm": 9.0,
            "orbit_deg": -1.6,
            "wrist_rotvec_deg": (-5.020, 0.508, -0.859),
            "thumb_rota1_offset_rad": 0.02,
            "role": "interior_margin_control",
        },
        {
            "label": "joint_pose_shift_9p5mm_rota1_plus_0p01",
            "root_shift_mm": 9.5,
            "orbit_deg": -1.6,
            "wrist_rotvec_deg": (-5.020, 0.508, -0.859),
            "thumb_rota1_offset_rad": 0.01,
            "role": "joint_lower_probe",
        },
        {
            "label": "joint_pose_shift_9p5mm_rota1_plus_0p02",
            "root_shift_mm": 9.5,
            "orbit_deg": -1.6,
            "wrist_rotvec_deg": (-5.020, 0.508, -0.859),
            "thumb_rota1_offset_rad": 0.02,
            "role": "recommended_balanced",
        },
        {
            "label": "joint_pose_shift_9p5mm_rota1_plus_0p03",
            "root_shift_mm": 9.5,
            "orbit_deg": -1.6,
            "wrist_rotvec_deg": (-5.020, 0.508, -0.859),
            "thumb_rota1_offset_rad": 0.03,
            "role": "pose_preservation_upper_probe",
        },
        {
            "label": "joint_pose_shift_9p9mm_rota1_plus_0p02",
            "root_shift_mm": 9.9,
            "orbit_deg": -1.6,
            "wrist_rotvec_deg": (-5.020, 0.508, -0.859),
            "thumb_rota1_offset_rad": 0.02,
            "role": "root_boundary_probe",
        },
        {
            "label": "joint_pose_optimized_shift_9p9mm",
            "root_shift_mm": 9.9,
            "orbit_deg": -1.1,
            "wrist_rotvec_deg": (-5.020, 0.508, -0.359),
            "role": "joint_pose_optimized",
            # Measured-contact continuation solution.  Unlike a command-only
            # override, nominal qpos, Jacobian-retreated precontact and preload
            # are intentionally distinct.
            "nominal_qpos_rad": (
                1.401393790701948,
                0.10098713651266032,
                0.708461030446209,
                -0.053867881354959464,
                0.5005161830879803,
                1.1228208947676608,
                0.5655724799459076,
                1.1363772258745064,
            ),
            "precontact_targets_rad": (
                1.403042470,
                0.069613252,
                0.699108139,
                -0.063475562,
                0.463468073,
                1.118661636,
                0.521404243,
                1.140068477,
            ),
            "preload_targets_rad": (
                1.404,
                0.10793259692103421,
                0.7159310804392538,
                -0.05226411878261765,
                0.5040363049047765,
                1.1258092296891584,
                0.5681859697561782,
                1.1382134340844432,
            ),
        },
        {
            "label": "joint_wrist_pose_optimized_shift_9p9mm",
            "root_shift_mm": 9.9,
            "root_delta_x_additional_mm": 0.2,
            "orbit_deg": -1.1,
            "wrist_rotvec_deg": (-5.020, 0.508, -1.259),
            "role": "joint_wrist_pose_optimized",
            "nominal_qpos_rad": (
                1.401393790701948,
                0.10098713651266032,
                0.708461030446209,
                -0.053867881354959464,
                0.5005161830879803,
                1.1228208947676608,
                0.5655724799459076,
                1.1363772258745064,
            ),
            "precontact_targets_rad": (
                1.403042470,
                0.069613252,
                0.699108139,
                -0.063475562,
                0.463468073,
                1.118661636,
                0.521404243,
                1.140068477,
            ),
            "preload_targets_rad": (
                1.404,
                0.10793259692103421,
                0.7159310804392538,
                -0.05226411878261765,
                0.5040363049047765,
                1.1258092296891584,
                0.5681859697561782,
                1.1382134340844432,
            ),
        },
    )


def _simulate(
    config: dict[str, Any], trace_path: Path
) -> tuple[dict[str, Any], Mapping[str, np.ndarray]]:
    session = SimulationSession(config)
    try:
        while not session.complete:
            session.advance_one()
        summary = session.finalize(trace_path=trace_path)
        return summary, session.traces
    finally:
        session.close()


def _relative_thumb_shift_mm(record: Mapping[str, Any]) -> float:
    shifts = _mapping(record.get("contact_centroid_shift_cube_local_mm"))
    thumb = float(shifts["thumb"][1])
    opposing_mean = 0.5 * (
        float(shifts["index"][1]) + float(shifts["mid"][1])
    )
    return thumb - opposing_mean


def _selection_rank(record: Mapping[str, Any]) -> tuple[Any, ...]:
    """Prefer a strict grasp, then margin before raw displacement."""

    pose_translation = float(record.get("pose_translation_mm", math.inf))
    root_margin = float(record.get("root_y_boundary_margin_mm", -math.inf))
    relative_shift = float(record.get("thumb_relative_y_shift_mm", math.inf))
    total_shift = float(
        _mapping(record["contact_centroid_shift_cube_local_mm"])["thumb"][1]
    )
    role = str(record.get("role", ""))
    return (
        int(bool(record.get("hard_pass"))),
        int(role == "joint_wrist_pose_optimized"),
        min(root_margin / 0.5, (0.5 - pose_translation) / 0.1),
        -abs(relative_shift),
        -abs(total_shift),
        -pose_translation,
        str(record.get("label", "")),
    )


def _mark_joint_pose_status(
    resolved: dict[str, Any], summary: Mapping[str, Any], record: Mapping[str, Any]
) -> None:
    _mark_override_status(resolved, summary)
    status = resolved["experiment_status"]
    status["classification"] = (
        "validated_fixed_160g_joint_pose_refined_grasp_ablation"
        if bool(record.get("hard_pass"))
        else "joint_pose_refinement_near_miss"
    )
    status["passed"] = bool(record.get("hard_pass"))
    status["hard_constraints_passed"] = bool(record.get("hard_pass"))
    status["note"] = (
        "Fresh schema-v11 free-cube dynamics after coupled hand-root and thumb "
        "joint refinement. This status certifies grasp acquisition only; the "
        "zero manipulation delta is not a lift claim."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    source_path = args.source_config.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(f"output directory already exists: {output}")
    source = load_config(source_path)
    source_trace = source_path.parent / "trace.npz"
    if not source_trace.is_file():
        raise FileNotFoundError(f"source trace does not exist: {source_trace}")
    source_config_sha = file_sha256(source_path)
    source_trace_sha = file_sha256(source_trace)
    source_cube = _cube_signature(source)
    source_relative = _mapping(
        _mapping(source.get("candidate_metadata")).get(
            "relative_wrist_pose_search"
        )
    )
    original_root_delta = np.asarray(
        source_relative["root_delta_cube_m"], dtype=np.float64
    )
    if original_root_delta.shape != (3,):
        raise ValueError("source root delta must contain three values")
    with np.load(source_trace, allow_pickle=False) as trace:
        start = int(np.asarray(trace["grasp_stable_window_start_step"]))
        end = int(np.asarray(trace["grasp_stable_window_end_step"]))
        baseline_centroids, counts = _local_contact_centroids(
            trace, start, end
        )
    if np.any(counts < 250) or not np.isfinite(baseline_centroids).all():
        raise RuntimeError("source lacks an authenticated 250-step contact window")

    output.mkdir(parents=True, exist_ok=args.resume)
    records: list[dict[str, Any]] = []
    catalog_entries: list[dict[str, Any]] = []
    for spec in _candidate_specs():
        label = str(spec["label"])
        directory = output / label
        result_path = directory / "result.json"
        config_path = directory / "resolved_config.json"
        trace_path = directory / "trace.npz"
        if args.resume and all(
            path.is_file() for path in (result_path, config_path, trace_path)
        ):
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            if payload.get("source_config_sha256") != source_config_sha:
                raise RuntimeError(f"resume source config changed: {result_path}")
            if payload.get("source_trace_sha256") != source_trace_sha:
                raise RuntimeError(f"resume source trace changed: {result_path}")
            record = dict(payload["record"])
        else:
            directory.mkdir(exist_ok=args.resume)
            requested_root_delta = original_root_delta.copy()
            requested_root_delta[1] -= float(spec["root_shift_mm"]) / 1000.0
            requested_root_delta[0] += float(
                spec.get("root_delta_x_additional_mm", 0.0)
            ) / 1000.0
            config, changed = apply_viewer_overrides(
                source,
                clockwise_orbit_deg=float(spec["orbit_deg"]),
                root_delta_cube_mm=(requested_root_delta * 1000.0).tolist(),
                wrist_local_rotvec_deg=spec["wrist_rotvec_deg"],
            )
            if not changed:
                raise RuntimeError("joint/pose refinement was not marked as override")
            if "nominal_qpos_rad" in spec:
                nominal_values = tuple(float(value) for value in spec["nominal_qpos_rad"])
                precontact_values = tuple(
                    float(value) for value in spec["precontact_targets_rad"]
                )
                preload_values = tuple(
                    float(value) for value in spec["preload_targets_rad"]
                )
                if not all(
                    len(values) == len(ACTIVE_ACTUATORS)
                    for values in (nominal_values, precontact_values, preload_values)
                ):
                    raise ValueError("explicit joint-pose solution has wrong width")
                config["grasp_pose"]["nominal_joint_qpos_rad"] = dict(
                    zip(ACTIVE_ACTUATORS, nominal_values, strict=True)
                )
                config["control"]["precontact_targets_rad"] = dict(
                    zip(ACTIVE_ACTUATORS, precontact_values, strict=True)
                )
                config["control"]["contact_preload_targets_rad"] = dict(
                    zip(ACTIVE_ACTUATORS, preload_values, strict=True)
                )
                offset = float(
                    nominal_values[ACTIVE_ACTUATORS.index(THUMB_ROTA1)]
                    - source["grasp_pose"]["nominal_joint_qpos_rad"][THUMB_ROTA1]
                )
                joint_update_kind = "measured_contact_multijoint_solution"
            else:
                offset = float(spec["thumb_rota1_offset_rad"])
                config["grasp_pose"]["nominal_joint_qpos_rad"][THUMB_ROTA1] += offset
                for field in (
                    "precontact_targets_rad",
                    "contact_preload_targets_rad",
                ):
                    config["control"][field][THUMB_ROTA1] += offset
                joint_update_kind = "thumb_rota1_three_field_offset"
            metadata = config.setdefault("candidate_metadata", {})
            metadata["thumb_negative_y_joint_pose_refinement"] = {
                "source_config_sha256": source_config_sha,
                "source_trace_sha256": source_trace_sha,
                "cube_pose_sampled": False,
                "hand_root_fixed_during_simulation": True,
                "requested_total_root_shift_cube_y_mm": -float(
                    spec["root_shift_mm"]
                ),
                "thumb_rota1_three_field_offset_rad": offset,
                "joint_update_kind": joint_update_kind,
                "role": str(spec["role"]),
            }
            validate_config(config)
            preflight_config(config)
            if _cube_signature(config) != source_cube:
                raise RuntimeError("candidate changed cube pose or physics")
            summary, traces = _simulate(config, trace_path)
            record = _candidate_record(
                shift_mm=float(spec["root_shift_mm"]),
                summary=summary,
                traces=traces,
                baseline_centroids_m=baseline_centroids,
            )
            record.update(
                {
                    "label": label,
                    "role": str(spec["role"]),
                    "clockwise_orbit_deg": float(spec["orbit_deg"]),
                    "wrist_local_rotvec_deg": list(spec["wrist_rotvec_deg"]),
                    "root_delta_cube_mm": (
                        requested_root_delta * 1000.0
                    ).tolist(),
                    "root_y_boundary_margin_mm": 1000.0
                    * (requested_root_delta[1] - (-0.012)),
                    "thumb_rota1_offset_rad": offset,
                    "joint_update_kind": joint_update_kind,
                    "nominal_joint_qpos_rad": copy.deepcopy(
                        config["grasp_pose"]["nominal_joint_qpos_rad"]
                    ),
                    "precontact_targets_rad": copy.deepcopy(
                        config["control"]["precontact_targets_rad"]
                    ),
                    "contact_preload_targets_rad": copy.deepcopy(
                        config["control"]["contact_preload_targets_rad"]
                    ),
                    "thumb_rota1_nominal_rad": float(
                        config["grasp_pose"]["nominal_joint_qpos_rad"][
                            THUMB_ROTA1
                        ]
                    ),
                    "thumb_relative_y_shift_mm": _relative_thumb_shift_mm(
                        record
                    ),
                    "hand_pose": copy.deepcopy(config["hand_pose"]),
                    "cube_world_pose_size_mass_friction_unchanged": True,
                }
            )
            resolved = resolved_run_config(config, summary)
            _mark_joint_pose_status(resolved, summary, record)
            write_json(config_path, resolved)
            write_json(
                result_path,
                {
                    "diagnostic_schema_version": 1,
                    "diagnostic_kind": "thumb_negative_y_joint_pose_refinement",
                    "source_config": str(source_path),
                    "source_config_sha256": source_config_sha,
                    "source_trace_sha256": source_trace_sha,
                    "record": record,
                    "summary": summary,
                },
            )
        record["artifacts"] = {
            "resolved_config": f"{label}/resolved_config.json",
            "result": f"{label}/result.json",
            "trace": f"{label}/trace.npz",
            "sha256": {
                "resolved_config": file_sha256(config_path),
                "result": file_sha256(result_path),
                "trace": file_sha256(trace_path),
            },
        }
        records.append(record)
        catalog_entries.append(
            {
                "label": label,
                "trajectory_id": label,
                "aliases": [],
                "hard_pass": bool(record["hard_pass"]),
                "artifacts": copy.deepcopy(record["artifacts"]),
            }
        )
        print(
            f"{label}: hard={record['hard_pass']} "
            f"thumb_y={record['contact_centroid_cube_local_mm']['thumb'][1]:.3f} "
            f"relative_shift={record['thumb_relative_y_shift_mm']:.3f} "
            f"pose={record['pose_translation_mm']:.3f}mm/"
            f"{record['pose_orientation_deg']:.3f}deg"
        )

    passing = [record for record in records if bool(record["hard_pass"])]
    if passing:
        recommended = max(passing, key=_selection_rank)
        furthest = min(
            passing,
            key=lambda value: float(
                _mapping(value["contact_centroid_shift_cube_local_mm"])[
                    "thumb"
                ][1]
            ),
        )
    else:
        recommended = max(records, key=_selection_rank)
        furthest = recommended
    recommended_label = str(recommended["label"])
    balanced = max(
        passing or records,
        key=lambda value: (
            int(str(value.get("role")) == "recommended_balanced"),
            _selection_rank(value),
        ),
    )
    balanced_label = str(balanced["label"])
    furthest_label = str(furthest["label"])
    for entry in catalog_entries:
        if entry["label"] == recommended_label:
            entry["aliases"].append("best_nominal")
        if entry["label"] == balanced_label:
            entry["aliases"].append("best_balanced")
        if entry["label"] == furthest_label:
            entry["aliases"].append("furthest_strict_grasp")
        entry["aliases"] = list(dict.fromkeys(entry["aliases"]))

    report = {
        "diagnostic_schema_version": 1,
        "diagnostic_kind": "thumb_negative_y_joint_pose_refinement",
        "status": "complete",
        "source_config": str(source_path),
        "source_config_sha256": source_config_sha,
        "source_trace_sha256": source_trace_sha,
        "cube_world_pose_size_mass_friction_unchanged": True,
        "hand_root_fixed_during_each_simulation": True,
        "runtime_time_varying_actuators": 8,
        "baseline_contact_centroid_cube_local_mm": {
            finger: (baseline_centroids[index] * 1000.0).tolist()
            for index, finger in enumerate(FINGERS)
        },
        "hard_pass_count": len(passing),
        "candidate_count": len(records),
        "best_label": recommended_label,
        "best_balanced_label": balanced_label,
        "furthest_strict_grasp_label": furthest_label,
        "records": records,
    }
    write_json(output / "refinement_report.json", report)
    catalog = {
        "catalog_schema_version": 1,
        "catalog_kind": "thumb_negative_y_joint_pose_refinement_viewer_catalog",
        "aliases": {
            "best_nominal": recommended_label,
            "best_balanced": balanced_label,
            "furthest_strict_grasp": furthest_label,
        },
        "trajectories": catalog_entries,
    }
    write_json(output / "catalog.json", catalog)
    print(f"best={recommended_label}")
    print(f"furthest={furthest_label}")
    print(output / "refinement_report.json")
    return 0 if passing else 2


if __name__ == "__main__":
    raise SystemExit(main())
