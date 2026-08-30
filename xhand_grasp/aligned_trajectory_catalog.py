"""Transactional trajectory catalogs for the schema-v4 aligned campaign."""

from __future__ import annotations

import copy
import math
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .aligned_contacts_search import candidate_tilt_band_deg
from .artifacts import file_sha256, resolved_run_config, run_metadata, write_json
from .config import ACTIVE_FINGERS, validate_config
from .experiment import resolve_experiment
from .scene import (
    cube_vertical_half_extent_m,
    rpy_degrees_to_rotation_matrix,
)
from .simulation import run_simulation
from .trajectory_catalog import object_physical_parameters


ALIGNED_TRAJECTORY_CATALOG_SCHEMA_VERSION = 1
_REQUIRED_RUN_METADATA_KEYS = (
    "python",
    "mujoco",
    "numpy",
    "uv",
    "ffmpeg",
    "model_sha256",
    "uv_lock_sha256",
    "pyproject_sha256",
    "implementation_sha256",
    "config_sha256",
)
_SHARED_PROVENANCE_KEYS = tuple(
    key for key in _REQUIRED_RUN_METADATA_KEYS if key != "config_sha256"
)
_SHA256_METADATA_KEYS = (
    "model_sha256",
    "uv_lock_sha256",
    "pyproject_sha256",
    "implementation_sha256",
    "config_sha256",
)
_EVENT_FIELDS = (
    "grasp_acquisition_step",
    "manipulation_start_step",
    "manipulation_end_step",
    "termination_step",
)
_VIEWER_TRACE_SHAPES = {
    "cube_pos": (3,),
    "cube_quat": (4,),
    "cube_velocity": (6,),
    "ctrl": None,
    "joint_qpos": None,
    "joint_qvel": None,
}
_V4_TRACE_SHAPES = {
    "finger_down_tilt_deg": (),
    "distal_face_position_moment_n_m": (3, 8, 3),
    "target_face_contact_centroid_world_m": (3, 3),
    "target_face_contact_centroid_valid": (3,),
    "three_contact_height_spread_m": (),
    "three_contact_height_aligned": (),
}
_V5_TRACE_SHAPES = {
    "root_cube_center_distance_m": (),
    "thumb_bend_command_rad": (),
    "thumb_bend_qpos_rad": (),
    "distal_pad_force_n": (3,),
    "distal_nonpad_force_n": (3,),
    "distal_pad_force_fraction": (3,),
    "distal_active_taxel_count": (3,),
}


def _hard_pass(summary: Mapping[str, Any]) -> bool:
    stage = summary.get("stage_status")
    return bool(
        summary.get("passed", False)
        and isinstance(stage, Mapping)
        and stage.get("full_success", False)
    )


def _tilt_token(value: float) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("tilt band must be finite")
    text = format(number, ".12g")
    if text.startswith("-"):
        text = "m" + text[1:]
    return text.replace(".", "p")


def _declared_band(value: float, declared: tuple[float, ...]) -> float:
    matches = [
        band
        for band in declared
        if math.isclose(float(value), band, rel_tol=0.0, abs_tol=1e-9)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"candidate tilt band {value!r} is not one declared campaign band"
        )
    return matches[0]


def _candidate_id(candidate: Mapping[str, Any]) -> int:
    value = candidate.get("candidate_id")
    if isinstance(value, bool):
        raise ValueError("candidate_id must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("candidate_id must be an integer") from exc
    if result < 0 or result != value:
        raise ValueError("candidate_id must be a non-negative integer")
    return result


def _prepare_candidates(
    candidates: Iterable[Mapping[str, Any]],
    *,
    include_near_misses: bool,
) -> tuple[list[dict[str, Any]], tuple[float, ...], str]:
    materialized = tuple(candidates)
    if not materialized:
        raise ValueError("at least one aligned-contact candidate is required")

    prepared: list[dict[str, Any]] = []
    declared: tuple[float, ...] | None = None
    experiment_id: str | None = None
    seen_bands: set[float] = set()
    seen_ids: set[int] = set()
    for candidate in materialized:
        if not isinstance(candidate, Mapping):
            raise ValueError("each candidate must be a mapping")
        config_value = candidate.get("config")
        summary_value = candidate.get("summary")
        if not isinstance(config_value, dict):
            raise ValueError("each candidate requires a config mapping")
        if not isinstance(summary_value, Mapping):
            raise ValueError("each candidate requires a summary mapping")
        config = copy.deepcopy(config_value)
        if config.get("run_context") is not None:
            raise ValueError(
                "aligned trajectory catalogs require canonical candidates; "
                "override and robustness-trial configs are not publishable"
            )
        validate_config(config)
        schema_version = int(config.get("schema_version", 0))
        if schema_version not in (4, 5):
            raise ValueError(
                "aligned trajectory catalogs require schema version 4 or 5"
            )
        definition = resolve_experiment(config)
        campaign = (
            definition.aligned_contact_campaign
            if definition.aligned_contact_campaign is not None
            else definition.far_hand_campaign
        )
        if campaign is None:
            raise ValueError("candidate is not registered to an aligned campaign")
        candidate_bands = tuple(float(value) for value in campaign.tilt_band_centers_deg)
        if declared is None:
            declared = candidate_bands
            experiment_id = definition.experiment_id
        elif candidate_bands != declared or definition.experiment_id != experiment_id:
            raise ValueError("all candidates must belong to the same aligned campaign")

        band = _declared_band(candidate_tilt_band_deg(candidate), candidate_bands)
        if band in seen_bands:
            raise ValueError(
                f"more than one selected candidate was supplied for tilt band {band:g}"
            )
        identifier = _candidate_id(candidate)
        if identifier in seen_ids:
            raise ValueError("candidate_id values must be unique")
        seen_bands.add(band)
        seen_ids.add(identifier)
        reported_pass = _hard_pass(summary_value)
        prepared.append(
            {
                "candidate_id": identifier,
                "tilt_band_center_deg": band,
                "reported_hard_pass": reported_pass,
                "reported_summary": copy.deepcopy(dict(summary_value)),
                "config": config,
            }
        )

    assert declared is not None
    assert experiment_id is not None
    missing = [band for band in declared if band not in seen_bands]
    if include_near_misses and missing:
        raise ValueError(
            "include_near_misses=True requires one honest result for every "
            "declared tilt band; missing=" + ", ".join(f"{band:g}" for band in missing)
        )
    if not include_near_misses:
        prepared = [item for item in prepared if item["reported_hard_pass"]]
    prepared.sort(key=lambda item: declared.index(item["tilt_band_center_deg"]))
    return prepared, declared, experiment_id


def _load_and_validate_trace(
    trace_path: Path,
    summary: Mapping[str, Any],
    *,
    schema_version: int = 4,
) -> dict[str, np.ndarray]:
    with np.load(trace_path, allow_pickle=False) as archive:
        traces = {name: np.array(archive[name], copy=True) for name in archive.files}
    time = traces.get("time")
    if time is None or time.ndim != 1 or len(time) == 0:
        raise ValueError("rerun trace requires a non-empty one-dimensional time axis")
    total_steps = len(time)
    if not np.issubdtype(time.dtype, np.number) or not np.isfinite(time).all():
        raise ValueError("rerun trace time must be finite and numeric")
    if total_steps > 1 and np.any(np.diff(time) <= 0.0):
        raise ValueError("rerun trace time must be strictly increasing")

    required_shapes = {
        **_VIEWER_TRACE_SHAPES,
        **_V4_TRACE_SHAPES,
    }
    if int(schema_version) >= 5:
        required_shapes.update(_V5_TRACE_SHAPES)
    for name, trailing_shape in required_shapes.items():
        if name not in traces:
            raise ValueError(f"rerun trace is missing {name}")
        values = traces[name]
        if values.shape[:1] != (total_steps,):
            raise ValueError(f"rerun trace {name} has a different frame count")
        if trailing_shape is not None and values.shape[1:] != trailing_shape:
            raise ValueError(
                f"rerun trace {name} must have trailing shape {trailing_shape}"
            )
        if values.dtype.kind not in "bUS" and not np.isfinite(values).all():
            raise ValueError(f"rerun trace {name} contains NaN or Inf")
    widths = {
        int(traces[name].shape[1]) for name in ("ctrl", "joint_qpos", "joint_qvel")
    }
    if len(widths) != 1 or next(iter(widths)) <= 0:
        raise ValueError("rerun trace actuator arrays must share a positive width")
    quaternion_norm = np.linalg.norm(traces["cube_quat"], axis=1)
    if not np.allclose(quaternion_norm, 1.0, rtol=0.0, atol=1e-5):
        raise ValueError("rerun trace cube quaternions are not normalized")

    if "finger_order" not in traces or tuple(
        str(value) for value in traces["finger_order"].tolist()
    ) != tuple(ACTIVE_FINGERS):
        raise ValueError("rerun trace finger_order is not canonical")
    if "grasp_gate_order" not in traces or "contact_height_aligned" not in {
        str(value) for value in traces["grasp_gate_order"].tolist()
    }:
        raise ValueError("rerun trace is not a schema-v4 aligned-contact trace")

    metrics = summary.get("metrics")
    if isinstance(metrics, Mapping):
        for name in _EVENT_FIELDS:
            if name not in metrics:
                continue
            if name not in traces or np.asarray(traces[name]).shape != ():
                raise ValueError(f"rerun trace is missing scalar event {name}")
            if int(np.asarray(traces[name])) != int(metrics[name]):
                raise ValueError(
                    f"rerun summary metric {name} disagrees with trace event"
                )
    return traces


def _v4_physical_parameters(config: Mapping[str, Any]) -> dict[str, Any]:
    """Reuse the established physical report with rotated-support correction."""

    physical = object_physical_parameters(config)
    cube = config["cube"]
    cube_rotation = rpy_degrees_to_rotation_matrix(
        cube.get("rpy_deg", [0.0, 0.0, 0.0])
    )
    center = np.asarray(
        [
            float(cube["center_xy_m"][0]),
            float(cube["center_xy_m"][1]),
            float(config["scene"]["support_top_z_m"])
            + cube_vertical_half_extent_m(float(cube["edge_m"]), cube_rotation)
            + float(cube.get("z_offset_m", 0.0)),
        ],
        dtype=np.float64,
    )
    root = config["hand_pose"]
    root_position = np.asarray(root["translation_m"], dtype=np.float64)
    root_rotation = rpy_degrees_to_rotation_matrix(root["rpy_deg"])
    physical["initial_cube_pose"]["position_m"] = center.tolist()
    physical["initial_cube_in_hand_root_pose"]["position_m"] = (
        root_rotation.T @ (center - root_position)
    ).tolist()
    return physical


def _validated_run_metadata(config_path: Path) -> dict[str, Any]:
    """Collect and bind run provenance to the exact resolved config artifact."""

    value = run_metadata(config_path)
    if not isinstance(value, Mapping):
        raise RuntimeError("run_metadata must return a mapping")
    metadata = copy.deepcopy(dict(value))
    missing = [key for key in _REQUIRED_RUN_METADATA_KEYS if key not in metadata]
    if missing:
        raise RuntimeError(
            "run_metadata is missing required provenance fields: "
            + ", ".join(missing)
        )
    for key in _REQUIRED_RUN_METADATA_KEYS:
        if not isinstance(metadata[key], str) or not metadata[key].strip():
            raise RuntimeError(f"run_metadata field {key} must be a non-empty string")
    for key in _SHA256_METADATA_KEYS:
        digest = metadata[key]
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise RuntimeError(f"run_metadata field {key} is not a lowercase SHA-256")
    expected_config_hash = file_sha256(config_path)
    if metadata["config_sha256"] != expected_config_hash:
        raise RuntimeError(
            "run_metadata config_sha256 does not bind the resolved config artifact"
        )
    return metadata


def _catalog_provenance(
    metadata_by_trajectory: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize immutable provenance while keeping config hashes per trajectory."""

    if not metadata_by_trajectory:
        return {
            "shared_run_metadata": {},
            "resolved_config_sha256_by_trajectory": {},
            "all_required_shared_fields_consistent": True,
        }

    iterator = iter(metadata_by_trajectory.items())
    _, first = next(iterator)
    shared = {
        key: copy.deepcopy(first[key]) for key in _SHARED_PROVENANCE_KEYS
    }
    for trajectory_id, metadata in iterator:
        inconsistent = [
            key for key, expected in shared.items() if metadata[key] != expected
        ]
        if inconsistent:
            raise RuntimeError(
                f"run provenance changed while publishing {trajectory_id}: "
                + ", ".join(inconsistent)
            )
    return {
        "shared_run_metadata": shared,
        "resolved_config_sha256_by_trajectory": {
            trajectory_id: metadata["config_sha256"]
            for trajectory_id, metadata in metadata_by_trajectory.items()
        },
        "all_required_shared_fields_consistent": True,
    }


def export_aligned_trajectory_catalog(
    candidates: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    include_near_misses: bool = True,
    video: bool = True,
    best_candidate_id: int | None = None,
) -> dict[str, Any]:
    """Independently rerun and atomically publish one result per tilt band."""

    if not isinstance(include_near_misses, bool) or not isinstance(video, bool):
        raise ValueError("include_near_misses and video must be booleans")
    if best_candidate_id is not None:
        if (
            isinstance(best_candidate_id, bool)
            or int(best_candidate_id) != best_candidate_id
            or int(best_candidate_id) < 0
        ):
            raise ValueError("best_candidate_id must be a non-negative integer")
        best_candidate_id = int(best_candidate_id)
    selections, declared_bands, experiment_id = _prepare_candidates(
        candidates,
        include_near_misses=include_near_misses,
    )
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(
            f"output directory already exists: {destination}; choose a new output"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    metadata_by_trajectory: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(
        dir=destination.parent,
        prefix=f".{destination.name}.staging.",
    ) as staging_name:
        staging = Path(staging_name)
        for selection in selections:
            band = float(selection["tilt_band_center_deg"])
            token = _tilt_token(band)
            working = staging / f"candidate_tilt_{token}"
            working.mkdir()
            trace_path = working / "trace.npz"
            video_path = working / "trajectory.mp4" if video else None
            summary = run_simulation(
                copy.deepcopy(selection["config"]),
                trace_path=trace_path,
                video_path=video_path,
            )
            if not isinstance(summary, dict):
                raise RuntimeError("run_simulation must return a summary mapping")
            rerun_pass = _hard_pass(summary)
            if selection["reported_hard_pass"] and not rerun_pass:
                raise RuntimeError(
                    f"hard-pass candidate {selection['candidate_id']} in tilt band "
                    f"{band:g} did not reproduce full success; "
                    f"failed_checks={summary.get('failed_checks', [])!r}"
                )
            traces = _load_and_validate_trace(
                trace_path,
                summary,
                schema_version=int(selection["config"].get("schema_version", 4)),
            )
            if video_path is not None and not video_path.is_file():
                raise RuntimeError("run_simulation did not create the requested MP4")

            classification = "pass" if rerun_pass else "near_miss"
            label = (
                f"tilt_{token}"
                if rerun_pass
                else f"near_miss_tilt_{token}"
            )
            trajectory_id = label
            final_directory = staging / trajectory_id
            if final_directory.exists():
                raise ValueError(f"duplicate trajectory identifier: {trajectory_id}")
            working.rename(final_directory)
            trace_path = final_directory / "trace.npz"
            video_path = (
                final_directory / "trajectory.mp4" if video else None
            )

            resolved = resolved_run_config(selection["config"], summary)
            config_path = final_directory / "resolved_config.json"
            write_json(config_path, resolved)
            metadata = _validated_run_metadata(config_path)
            metadata_by_trajectory[trajectory_id] = metadata
            physical = _v4_physical_parameters(resolved)
            local_hashes = {
                "resolved_config": file_sha256(config_path),
                "trace": file_sha256(trace_path),
            }
            if video_path is not None:
                local_hashes["video"] = file_sha256(video_path)
            artifacts = {
                "resolved_config": config_path.name,
                "result": "result.json",
                "trace": trace_path.name,
                "video": video_path.name if video_path is not None else None,
                "sha256": local_hashes,
            }
            result = {
                "aligned_trajectory_catalog_schema_version": (
                    ALIGNED_TRAJECTORY_CATALOG_SCHEMA_VERSION
                ),
                "trajectory_id": trajectory_id,
                "label": label,
                "classification": classification,
                "candidate_id": int(selection["candidate_id"]),
                "tilt_band_center_deg": band,
                "reported_hard_pass": bool(selection["reported_hard_pass"]),
                "rerun_hard_pass": rerun_pass,
                "trace_summary_consistent": True,
                "target_faces": copy.deepcopy(
                    resolved["contact_topology"]["target_faces"]
                ),
                "metadata": metadata,
                "physical_parameters": physical,
                "reported_summary": selection["reported_summary"],
                "summary": summary,
                "config": resolved,
                "experiment_status": resolved["experiment_status"],
                "artifacts": artifacts,
            }
            result_path = final_directory / "result.json"
            write_json(result_path, result)

            catalog_artifacts = {
                "directory": trajectory_id,
                "resolved_config": f"{trajectory_id}/{config_path.name}",
                "result": f"{trajectory_id}/{result_path.name}",
                "trace": f"{trajectory_id}/{trace_path.name}",
                "video": (
                    f"{trajectory_id}/{video_path.name}"
                    if video_path is not None
                    else None
                ),
                "sha256": {
                    **local_hashes,
                    "result": file_sha256(result_path),
                },
            }
            entries.append(
                {
                    "trajectory_id": trajectory_id,
                    "label": label,
                    "classification": classification,
                    "candidate_id": int(selection["candidate_id"]),
                    "tilt_band_center_deg": band,
                    "reported_hard_pass": bool(
                        selection["reported_hard_pass"]
                    ),
                    "rerun_hard_pass": rerun_pass,
                    "trace_summary_consistent": True,
                    "target_faces": copy.deepcopy(
                        resolved["contact_topology"]["target_faces"]
                    ),
                    "physical_parameters": physical,
                    "stage_status": copy.deepcopy(
                        summary.get("stage_status", {})
                    ),
                    "metrics": copy.deepcopy(summary.get("metrics", {})),
                    "failed_checks": copy.deepcopy(
                        summary.get("failed_checks", [])
                    ),
                    "artifacts": catalog_artifacts,
                }
            )

        catalog_aliases: dict[str, str] = {}
        if best_candidate_id is not None:
            matching = [
                entry
                for entry in entries
                if int(entry["candidate_id"]) == best_candidate_id
            ]
            if len(matching) != 1:
                raise ValueError(
                    "best_candidate_id must identify exactly one published "
                    "trajectory"
                )
            best_entry = matching[0]
            best_alias = (
                "best_nominal" if best_entry["rerun_hard_pass"] else "best_attempt"
            )
            best_entry["aliases"] = [best_alias]
            catalog_aliases[best_alias] = str(best_entry["trajectory_id"])

        passing_bands = {
            float(entry["tilt_band_center_deg"])
            for entry in entries
            if entry["rerun_hard_pass"]
        }
        published_bands = {
            float(entry["tilt_band_center_deg"]) for entry in entries
        }
        reported_pass_count = sum(
            bool(entry["reported_hard_pass"]) for entry in entries
        )
        reproduced_reported_passes = sum(
            bool(entry["reported_hard_pass"] and entry["rerun_hard_pass"])
            for entry in entries
        )
        provenance = _catalog_provenance(metadata_by_trajectory)
        catalog = {
            "trajectory_catalog_schema_version": (
                ALIGNED_TRAJECTORY_CATALOG_SCHEMA_VERSION
            ),
            "experiment_id": experiment_id,
            "declared_tilt_bands_deg": list(declared_bands),
            "published_tilt_bands_deg": [
                band for band in declared_bands if band in published_bands
            ],
            "missing_published_tilt_bands_deg": [
                band for band in declared_bands if band not in published_bands
            ],
            "passing_tilt_bands_deg": [
                band for band in declared_bands if band in passing_bands
            ],
            "missing_passing_tilt_bands_deg": [
                band for band in declared_bands if band not in passing_bands
            ],
            "trajectory_count": len(entries),
            "passing_trajectory_count": sum(
                bool(entry["rerun_hard_pass"]) for entry in entries
            ),
            "near_miss_trajectory_count": sum(
                not bool(entry["rerun_hard_pass"]) for entry in entries
            ),
            "reported_hard_pass_count": reported_pass_count,
            "all_selected_passes_reproduced": (
                reproduced_reported_passes == reported_pass_count
            ),
            "all_reruns_full_success": bool(entries)
            and all(bool(entry["rerun_hard_pass"]) for entry in entries),
            "all_declared_bands_passed": all(
                band in passing_bands for band in declared_bands
            ),
            "campaign_has_passing_trajectory": bool(passing_bands),
            "include_near_misses": include_near_misses,
            "aliases": catalog_aliases,
            "provenance": provenance,
            "trajectories": entries,
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
    "ALIGNED_TRAJECTORY_CATALOG_SCHEMA_VERSION",
    "export_aligned_trajectory_catalog",
]
