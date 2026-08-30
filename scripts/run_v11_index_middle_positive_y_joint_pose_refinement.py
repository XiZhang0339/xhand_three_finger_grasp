#!/usr/bin/env python3
"""Move index/middle contacts along cube-local +Y with a fresh grasp proof.

The source is the measured negative-Y thumb refinement.  The cube is copied
unchanged and remains a free body.  Candidate hand-root poses, wrist residuals
and measured-contact joint states are applied before a complete no-contact to
grasp MuJoCo run.  Static point targets are proposal guidance only; publication
requires the existing schema-v11 250 ms actual-qpos grasp gate.
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
    "thumb_negative_y_joint_pose_refinement_from_orbit_neg_1p1_v1/"
    "joint_wrist_pose_optimized_shift_9p9mm/resolved_config.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/left_opposed_face_palm_down_larger_relative_wrist_pose_"
    "actual_contact_smooth_vertical_lift/tune/"
    "index_middle_positive_y_joint_pose_refinement_from_thumb_negative_y_best_v1"
)
INDEX_BEND = "left_hand_index_bend_joint_actuator"
MIDDLE_JOINT1 = "left_hand_mid_joint1_actuator"


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _candidate_specs() -> tuple[dict[str, Any], ...]:
    common = tuple(
        {
            "label": f"root_common_positive_y_{shift}mm",
            "role": "common_root_control",
            "root_add_mm": (0.0, float(shift), 0.0),
            "wrist_add_deg": (0.0, 0.0, 0.0),
        }
        for shift in (2, 4, 6, 8)
    )
    balanced = {
        "label": "balanced_joint_wrist_positive_y",
        "role": "recommended_interior_joint_pose",
        "root_add_mm": (-0.3, 4.0, 0.0),
        "wrist_add_deg": (-0.5, 0.0, 0.0),
        "nominal_delta_rad": {INDEX_BEND: -0.005, MIDDLE_JOINT1: 0.0025},
        "preload_delta_rad": {INDEX_BEND: -0.005, MIDDLE_JOINT1: 0.0025},
    }
    balanced_multijoint = {
        "label": "balanced_multijoint_positive_y",
        "role": "recommended_interior_joint_pose",
        "root_delta_cube_mm": (
            -6.47542725589706,
            -8.827635908583403,
            3.693364077569605,
        ),
        "wrist_rotvec_deg": (-5.82, 0.53298, -1.80367),
        "nominal_qpos_rad": (
            1.405393790701948,
            0.10258148674266032,
            0.707597246250209,
            -0.05356928540095946,
            0.5121272998879803,
            1.1264255809576609,
            0.5685029369559077,
            1.1373769554775064,
        ),
        "precontact_targets_rad": (
            1.4079836415719376,
            0.06767092497975094,
            0.695696554805127,
            -0.062072664211591536,
            0.4741226972737729,
            1.1241891846998606,
            0.5287051379791479,
            1.1415398724073584,
        ),
        "preload_targets_rad": (
            1.408,
            0.10952694715103421,
            0.7150672962432537,
            -0.051965522828617654,
            0.5156474217047766,
            1.1294139158791587,
            0.5711164267661781,
            1.1392131636874432,
        ),
    }
    opposing_only = {
        "label": "opposing_fingers_positive_y_thumb_fixed",
        "role": "maximum_relative_positive_y",
        # Boundary-safe reparameterization of current orbit -1.1 plus a local
        # -4 degree orbit.  It produces the same world hand pose while keeping
        # the immutable-anchor root residual inside the registered +/-12 mm.
        "clockwise_orbit_deg": -5.5,
        "root_delta_cube_mm": (
            -5.521011145723,
            -11.883485147807,
            3.763261745300,
        ),
        "wrist_rotvec_deg": (
            -4.582148939811,
            0.545297578886,
            -0.889932906412,
        ),
        "natural_composed_orbit_deg": -5.1,
        "equivalent_pose_reparameterization": True,
        "requested_index_middle_target_shift_mm": 4.0,
        "nominal_qpos_rad": (
            1.4266545131,
            0.1076109828,
            0.7044608137,
            -0.0632005397,
            0.5118181760,
            1.1254813858,
            0.5526505765,
            1.1396143851,
        ),
        "precontact_targets_rad": (
            1.4301141097,
            0.0757974884,
            0.6951324101,
            -0.0686518209,
            0.4734138846,
            1.1233624789,
            0.5128782358,
            1.1426487941,
        ),
        "preload_targets_rad": (
            1.4292607223,
            0.1145564432,
            0.7119308636,
            -0.0615967771,
            0.5153382978,
            1.1284697207,
            0.5552640663,
            1.1414505933,
        ),
        "continuation": {
            "target_steps_mm": [2, 4, 6, 8],
            "published_target_mm": 4,
            "reason_larger_not_published": (
                "6mm hand yaw exceeds registered limit; 8mm static hard gate failed"
            ),
            "variables": "8_actual_qpos_plus_root_xyz_plus_wrist_rotvec",
        },
    }
    large_joint_pose = {
        "label": "large_joint_pose_positive_y_14mm_target",
        "role": "largest_joint_pose_positive_y",
        "clockwise_orbit_deg": -2.0868940511102956,
        "root_delta_cube_mm": (
            -5.918026858342059,
            -8.273222968547133,
            4.141555763776043,
        ),
        "wrist_rotvec_deg": (
            -6.0,
            0.6693580682991649,
            -0.5744535864907528,
        ),
        "requested_index_middle_target_shift_mm": 14.0,
        "nominal_qpos_rad": (
            1.4,
            0.09775931932711249,
            0.726059487833427,
            -0.05171254226099409,
            0.5225613120728763,
            1.1302706294125329,
            0.5790757546066274,
            1.1272565056637096,
        ),
        "precontact_targets_rad": (
            1.4026685872899574,
            0.06587201074129445,
            0.7169047130237863,
            -0.05993183719432105,
            0.48405569192461567,
            1.129103051629016,
            0.539083653490847,
            1.132173785812238,
        ),
        "preload_targets_rad": (
            1.4026062092980518,
            0.10470477973548638,
            0.7335295378264718,
            -0.05010877968865227,
            0.5260814338896724,
            1.1332589643340305,
            0.581689244416898,
            1.1290927138736464,
        ),
        "continuation": {
            "target_steps_mm": [2, 4, 6, 8, 10, 12, 14],
            "variables": (
                "8_actual_qpos_plus_root_xyz_plus_wrist_rotvec_plus_orbit"
            ),
            "thumb_bend_lower_boundary_rad": 1.4,
        },
    }
    # Eight 1 mm continuation targets were solved in 14 variables.  The
    # realized free-dynamics displacement is intentionally reported separately
    # from the requested static target.
    dls = {
        "label": "joint_pose_point_target_positive_y_8mm",
        "role": "maximum_requested_point_target",
        "root_delta_cube_mm": (
            -5.970451820271562,
            -9.506359818832568,
            4.12995376078259,
        ),
        "wrist_rotvec_deg": (
            -6.0,
            0.7336507038305524,
            -0.8490896552623508,
        ),
        "requested_index_middle_target_shift_mm": 8.0,
        "nominal_qpos_rad": (
            1.4,
            0.0988447568272304,
            0.7181589571227923,
            -0.05270954174884968,
            0.5167300890488293,
            1.1313461317014846,
            0.5765878971717093,
            1.130496418959616,
        ),
        "precontact_targets_rad": (
            1.402227263595932,
            0.06735800770945638,
            0.7084898964371275,
            -0.061614808817361526,
            0.4784835682979267,
            1.1297107198338496,
            0.5367463156315128,
            1.1351712010224975,
        ),
        "preload_targets_rad": (
            1.4026062092980518,
            0.10579021723560429,
            0.7256290071158371,
            -0.05110577917650787,
            0.5202502108656255,
            1.1343344666229822,
            0.5792013869819799,
            1.1323326271695529,
        ),
        "continuation": {
            "target_steps_mm": list(range(1, 9)),
            "variables": "8_actual_qpos_plus_root_xyz_plus_wrist_rotvec",
            "finite_difference": {
                "joint_rad": 0.0005,
                "root_m": 0.00002,
                "wrist_deg": 0.03,
            },
            "trust_region": {
                "joint_rad": 0.025,
                "root_m": 0.001,
                "wrist_deg": 0.4,
            },
            "damping": 0.025,
            "regularization": 0.01,
        },
    }
    return (
        *common,
        balanced,
        balanced_multijoint,
        opposing_only,
        large_joint_pose,
        dls,
    )


def _simulate(
    config: dict[str, Any], trace_path: Path
) -> tuple[dict[str, Any], Mapping[str, np.ndarray]]:
    session = SimulationSession(config)
    try:
        while not session.complete:
            session.advance_one()
        return session.finalize(trace_path=trace_path), session.traces
    finally:
        session.close()


def _apply_joint_values(
    config: dict[str, Any], spec: Mapping[str, Any]
) -> str:
    if "nominal_qpos_rad" in spec:
        fields = (
            ("nominal_joint_qpos_rad", spec["nominal_qpos_rad"]),
            ("precontact_targets_rad", spec["precontact_targets_rad"]),
            ("contact_preload_targets_rad", spec["preload_targets_rad"]),
        )
        for field, raw in fields:
            values = tuple(float(value) for value in raw)
            if len(values) != len(ACTIVE_ACTUATORS):
                raise ValueError(f"{field} must contain eight values")
            target = (
                config["grasp_pose"]
                if field == "nominal_joint_qpos_rad"
                else config["control"]
            )
            target[field] = dict(zip(ACTIVE_ACTUATORS, values, strict=True))
        return "point_target_dls_measured_contact_solution"
    nominal_delta = _mapping(spec.get("nominal_delta_rad"))
    preload_delta = _mapping(spec.get("preload_delta_rad"))
    precontact_delta = _mapping(spec.get("precontact_delta_rad"))
    for name, value in nominal_delta.items():
        config["grasp_pose"]["nominal_joint_qpos_rad"][name] += float(value)
    for field, changes in (
        ("precontact_targets_rad", precontact_delta),
        ("contact_preload_targets_rad", preload_delta),
    ):
        for name, value in changes.items():
            config["control"][field][name] += float(value)
    return "local_joint_preload_refinement" if nominal_delta else "root_pose_only"


def _record_metrics(
    *,
    spec: Mapping[str, Any],
    summary: Mapping[str, Any],
    traces: Mapping[str, np.ndarray],
    baseline_centroids_m: np.ndarray,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    record = _candidate_record(
        shift_mm=1.0,
        summary=summary,
        traces=traces,
        baseline_centroids_m=baseline_centroids_m,
    )
    record.pop("requested_root_shift_cube_y_mm", None)
    shifts = _mapping(record["contact_centroid_shift_cube_local_mm"])
    thumb_y = float(shifts["thumb"][1])
    index_y = float(shifts["index"][1])
    middle_y = float(shifts["mid"][1])
    record.update(
        {
            "label": str(spec["label"]),
            "role": str(spec["role"]),
            "requested_index_middle_target_shift_mm": spec.get(
                "requested_index_middle_target_shift_mm"
            ),
            "realized_positive_y_shift_mm": {
                "thumb": thumb_y,
                "index": index_y,
                "mid": middle_y,
            },
            "minimum_index_middle_positive_y_shift_mm": min(index_y, middle_y),
            "index_middle_mean_relative_to_thumb_shift_mm": (
                0.5 * (index_y + middle_y) - thumb_y
            ),
            "hand_pose": copy.deepcopy(config["hand_pose"]),
            "relative_wrist_pose": copy.deepcopy(
                _mapping(config.get("candidate_metadata")).get(
                    "relative_wrist_pose_search"
                )
            ),
            "nominal_joint_qpos_rad": copy.deepcopy(
                config["grasp_pose"]["nominal_joint_qpos_rad"]
            ),
            "precontact_targets_rad": copy.deepcopy(
                config["control"]["precontact_targets_rad"]
            ),
            "contact_preload_targets_rad": copy.deepcopy(
                config["control"]["contact_preload_targets_rad"]
            ),
            "cube_world_pose_size_mass_friction_unchanged": True,
        }
    )
    return record


def _rank(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        int(bool(record.get("hard_pass"))),
        int(record.get("role") == "maximum_relative_positive_y"),
        float(record.get("index_middle_mean_relative_to_thumb_shift_mm", -math.inf)),
        float(record.get("minimum_index_middle_positive_y_shift_mm", -math.inf)),
        -float(record.get("pose_translation_mm", math.inf)),
        -float(record.get("pose_orientation_deg", math.inf)),
        str(record.get("label", "")),
    )


def _mark_status(
    resolved: dict[str, Any], summary: Mapping[str, Any], record: Mapping[str, Any]
) -> None:
    _mark_override_status(resolved, summary)
    passed = bool(record.get("hard_pass"))
    resolved["experiment_status"].update(
        {
            "classification": (
                "validated_fixed_160g_index_middle_positive_y_grasp_ablation"
                if passed
                else "index_middle_positive_y_grasp_near_miss"
            ),
            "passed": passed,
            "hard_constraints_passed": passed,
            "note": (
                "Fresh free-cube grasp acquisition after index/middle +Y joint "
                "and fixed-root pose refinement. Zero manipulation delta: this "
                "status is a grasp claim, not a lift claim."
            ),
        }
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
        raise FileNotFoundError(f"source trace is missing: {source_trace}")
    config_sha = file_sha256(source_path)
    trace_sha = file_sha256(source_trace)
    source_cube = _cube_signature(source)
    relative = _mapping(
        _mapping(source.get("candidate_metadata")).get(
            "relative_wrist_pose_search"
        )
    )
    root_base = np.asarray(relative["root_delta_cube_m"], dtype=np.float64)
    wrist_base = np.asarray(relative["wrist_local_rotvec_deg"], dtype=np.float64)
    orbit = float(relative["clockwise_orbit_deg"])
    with np.load(source_trace, allow_pickle=False) as trace:
        start = int(np.asarray(trace["grasp_stable_window_start_step"]))
        end = int(np.asarray(trace["grasp_stable_window_end_step"]))
        baseline, counts = _local_contact_centroids(trace, start, end)
    if np.any(counts < 250) or not np.isfinite(baseline).all():
        raise RuntimeError("source has no authenticated 250-step grasp window")

    output.mkdir(parents=True, exist_ok=args.resume)
    records: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
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
            if payload.get("source_config_sha256") != config_sha:
                raise RuntimeError(f"resume source config changed: {result_path}")
            if payload.get("source_trace_sha256") != trace_sha:
                raise RuntimeError(f"resume source trace changed: {result_path}")
            record = dict(payload["record"])
        else:
            directory.mkdir(exist_ok=args.resume)
            root = (
                np.asarray(spec["root_delta_cube_mm"], dtype=np.float64) / 1000.0
                if "root_delta_cube_mm" in spec
                else root_base
                + np.asarray(spec["root_add_mm"], dtype=np.float64) / 1000.0
            )
            wrist = (
                np.asarray(spec["wrist_rotvec_deg"], dtype=np.float64)
                if "wrist_rotvec_deg" in spec
                else wrist_base
                + np.asarray(spec["wrist_add_deg"], dtype=np.float64)
            )
            effective_orbit = float(spec.get("clockwise_orbit_deg", orbit))
            config, changed = apply_viewer_overrides(
                source,
                clockwise_orbit_deg=effective_orbit,
                root_delta_cube_mm=(root * 1000.0).tolist(),
                wrist_local_rotvec_deg=wrist.tolist(),
            )
            if not changed:
                raise RuntimeError("candidate was not marked as a fresh override")
            update_kind = _apply_joint_values(config, spec)
            metadata = config.setdefault("candidate_metadata", {})
            metadata["index_middle_positive_y_joint_pose_refinement"] = {
                "source_config_sha256": config_sha,
                "source_trace_sha256": trace_sha,
                "cube_pose_sampled": False,
                "hand_root_fixed_during_simulation": True,
                "joint_update_kind": update_kind,
                "requested_index_middle_target_shift_mm": spec.get(
                    "requested_index_middle_target_shift_mm"
                ),
                "continuation": copy.deepcopy(spec.get("continuation")),
                "natural_composed_orbit_deg": spec.get(
                    "natural_composed_orbit_deg"
                ),
                "equivalent_pose_reparameterization": bool(
                    spec.get("equivalent_pose_reparameterization", False)
                ),
            }
            validate_config(config)
            preflight_config(config)
            if _cube_signature(config) != source_cube:
                raise RuntimeError("candidate changed cube pose or physics")
            summary, traces = _simulate(config, trace_path)
            record = _record_metrics(
                spec=spec,
                summary=summary,
                traces=traces,
                baseline_centroids_m=baseline,
                config=config,
            )
            record["joint_update_kind"] = update_kind
            resolved = resolved_run_config(config, summary)
            _mark_status(resolved, summary, record)
            write_json(config_path, resolved)
            write_json(
                result_path,
                {
                    "diagnostic_schema_version": 1,
                    "diagnostic_kind": (
                        "index_middle_positive_y_joint_pose_refinement"
                    ),
                    "source_config": str(source_path),
                    "source_config_sha256": config_sha,
                    "source_trace_sha256": trace_sha,
                    "record": record,
                    "summary": summary,
                },
            )
        artifacts = {
            "resolved_config": f"{label}/resolved_config.json",
            "result": f"{label}/result.json",
            "trace": f"{label}/trace.npz",
            "sha256": {
                "resolved_config": file_sha256(config_path),
                "result": file_sha256(result_path),
                "trace": file_sha256(trace_path),
            },
        }
        record["artifacts"] = copy.deepcopy(artifacts)
        records.append(record)
        entries.append(
            {
                "label": label,
                "trajectory_id": label,
                "aliases": [],
                "hard_pass": bool(record["hard_pass"]),
                "artifacts": artifacts,
            }
        )
        shift = record["realized_positive_y_shift_mm"]
        print(
            f"{label}: hard={record['hard_pass']} "
            f"dy(thumb/index/mid)={shift['thumb']:.3f}/"
            f"{shift['index']:.3f}/{shift['mid']:.3f} mm "
            f"relative={record['index_middle_mean_relative_to_thumb_shift_mm']:.3f} mm"
        )

    passing = [record for record in records if bool(record["hard_pass"])]
    selected = max(passing or records, key=_rank)
    balanced = max(
        passing or records,
        key=lambda value: (
            int(value.get("role") == "recommended_interior_joint_pose"),
            _rank(value),
        ),
    )
    common = max(
        (value for value in (passing or records) if value.get("role") == "common_root_control"),
        key=lambda value: float(value["minimum_index_middle_positive_y_shift_mm"]),
    )
    largest_joint_pose = max(
        (
            value
            for value in (passing or records)
            if value.get("role") == "largest_joint_pose_positive_y"
        ),
        key=lambda value: float(
            value["minimum_index_middle_positive_y_shift_mm"]
        ),
    )
    aliases = {
        "best_nominal": str(selected["label"]),
        "best_relative_positive_y": str(selected["label"]),
        "best_balanced": str(balanced["label"]),
        "largest_common_shift": str(common["label"]),
        "largest_joint_pose_shift": str(largest_joint_pose["label"]),
    }
    for entry in entries:
        entry["aliases"] = [
            alias for alias, target in aliases.items() if target == entry["label"]
        ]
    report = {
        "diagnostic_schema_version": 1,
        "diagnostic_kind": "index_middle_positive_y_joint_pose_refinement",
        "status": "complete",
        "source_config": str(source_path),
        "source_config_sha256": config_sha,
        "source_trace_sha256": trace_sha,
        "cube_world_pose_size_mass_friction_unchanged": True,
        "hand_root_fixed_during_each_simulation": True,
        "runtime_time_varying_actuators": 8,
        "baseline_contact_centroid_cube_local_mm": {
            finger: (baseline[index] * 1000.0).tolist()
            for index, finger in enumerate(FINGERS)
        },
        "candidate_count": len(records),
        "hard_pass_count": len(passing),
        "aliases": aliases,
        "records": records,
    }
    write_json(output / "refinement_report.json", report)
    write_json(
        output / "catalog.json",
        {
            "catalog_schema_version": 1,
            "catalog_kind": (
                "index_middle_positive_y_joint_pose_refinement_viewer_catalog"
            ),
            "aliases": aliases,
            "trajectories": entries,
        },
    )
    print(f"best={aliases['best_nominal']}")
    print(output / "refinement_report.json")
    return 0 if passing else 2


if __name__ == "__main__":
    raise SystemExit(main())
