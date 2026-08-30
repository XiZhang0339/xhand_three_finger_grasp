"""Publish auditable schema-v8 normal-aligned smooth-lift trajectories.

The search stages are allowed to keep thousands of diagnostic candidates.
This module is the much smaller publication boundary: it authenticates source
configs/results, deterministically selects diverse hard passes, and reruns each
published trajectory once while writing its JSON, NPZ and optional MP4 from
that same physical session.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .artifacts import (
    file_sha256,
    json_text,
    resolved_run_config,
    run_metadata,
    write_json,
)
from .config import validate_config
from .evaluation import evaluate_trace
from .rendering import VideoSettings, probe_video
from .scene import build_model
from .simulation import run_simulation
from .tuning.normal_aligned_smooth_lift import (
    CAMPAIGN_KIND,
    CANDIDATE_RESULT_SCHEMA_VERSION,
    canonical_sha256,
    controller_id,
    lift_candidate_evidence,
    lift_candidate_rank,
    pose_id,
)


EXPERIMENT_ID = (
    "left_opposed_face_palm_down_high_thumb_normal_aligned_smooth_vertical_lift"
)
CATALOG_SCHEMA_VERSION = 1
DEFAULT_OUTPUT = Path(
    "artifacts/left_opposed_face_palm_down_high_thumb_normal_aligned_"
    "smooth_vertical_lift/trajectory_catalog"
)
THUMB_BEND_ACTUATOR = "left_hand_thumb_bend_joint_actuator"
_SAFE_TOKEN = re.compile(r"[^A-Za-z0-9_-]+")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REQUIRED_V8_TRACE_FIELDS = (
    "time",
    "cube_pos",
    "cube_quat",
    "cube_velocity",
    "ctrl",
    "joint_qpos",
    "joint_qvel",
    "control_state",
    "closure_command_velocity_world_m_s",
    "closure_cube_outward_normal_world",
    "closure_alignment_angle_deg",
    "closure_inward_speed_m_s",
    "closure_alignment_valid",
    "operation_height_filtered_m",
    "operation_lateral_displacement_m",
    "operation_orientation_drift_deg",
    "motion_filter_valid",
    "grasp_acquisition_step",
    "manipulation_start_step",
    "manipulation_end_step",
    "termination_step",
    "video_frame_steps",
)
_REQUIRED_METADATA_KEYS = (
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


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _member(directory: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"candidate {label} path must be relative")
    path = (directory / value).resolve()
    try:
        path.relative_to(directory.resolve())
    except ValueError as error:
        raise ValueError(f"candidate {label} escapes {directory}") from error
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _required_digest(hashes: object, field: str) -> str:
    if not isinstance(hashes, Mapping):
        raise ValueError("candidate artifact SHA-256 map is missing")
    value = hashes.get(field)
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"candidate {field} SHA-256 is missing or invalid")
    return value


def _require_locked_exact_provenance(
    config: Mapping[str, Any], candidate_number: int
) -> None:
    candidate_metadata = config.get("candidate_metadata")
    if not isinstance(candidate_metadata, Mapping):
        raise ValueError("lift_exact config has no candidate provenance")
    try:
        metadata_candidate_id = int(candidate_metadata.get("candidate_id", -1))
        locked_timestep_s = float(
            candidate_metadata.get("locked_timestep_s", float("nan"))
        )
    except (TypeError, ValueError) as error:
        raise ValueError("lift_exact config has invalid candidate provenance") from error
    if (
        candidate_metadata.get("stage") != "lift_exact"
        or metadata_candidate_id != candidate_number
        or not np.isfinite(locked_timestep_s)
        or not np.isclose(locked_timestep_s, 0.001, rtol=0.0, atol=1e-12)
    ):
        raise ValueError(
            "lift_exact result does not bind locked config candidate provenance"
        )


def _hard_pass(summary: Mapping[str, Any]) -> bool:
    stage = summary.get("stage_status")
    return bool(
        summary.get("passed", False)
        and isinstance(stage, Mapping)
        and stage.get("grasp_success", False)
        and stage.get("manipulation_success", False)
        and stage.get("full_success", False)
    )


def _nested(mapping: Mapping[str, Any], *keys: str) -> object | None:
    value: object = mapping
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


@dataclass(frozen=True, slots=True)
class CatalogCandidate:
    result_path: Path
    config_path: Path
    result: dict[str, Any]
    config: dict[str, Any]
    candidate_id: str
    full_success: bool
    edge_mm: int
    thumb_target_rad: float
    stage: str = "lift_exact"
    pose_id: str = ""
    controller_id: str = ""
    candidate_sha256: str = ""
    trace_path: Path | None = None

    @property
    def thumb_band(self) -> str:
        value = self.thumb_target_rad
        if value <= 1.31 + 1e-12:
            return "low"
        if value <= 1.38 + 1e-12:
            return "middle"
        return "high"

    @property
    def rank(self) -> tuple[Any, ...]:
        return lift_candidate_rank(self.result)

    @property
    def pose_key(self) -> str:
        """Stable identity used to prevent publishing the same pose twice."""

        return self.pose_id or self.candidate_sha256 or (
            f"{self.candidate_id}:{self.result_path}"
        )


def _load_trace_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        traces = {name: np.array(archive[name], copy=True) for name in archive.files}
    missing = [name for name in _REQUIRED_V8_TRACE_FIELDS if name not in traces]
    if missing:
        raise ValueError("schema-v8 trace is missing: " + ", ".join(missing))
    time = np.asarray(traces["time"])
    if (
        time.ndim != 1
        or time.size == 0
        or not np.issubdtype(time.dtype, np.number)
        or not np.isfinite(time).all()
        or (time.size > 1 and np.any(np.diff(time) <= 0.0))
    ):
        raise ValueError("schema-v8 trace has an invalid time axis")
    total_steps = int(time.size)
    for name in _REQUIRED_V8_TRACE_FIELDS:
        if name in {
            "grasp_acquisition_step",
            "manipulation_start_step",
            "manipulation_end_step",
            "termination_step",
            "video_frame_steps",
        }:
            continue
        values = np.asarray(traces[name])
        if values.shape[:1] != (total_steps,):
            raise ValueError(f"schema-v8 trace field {name} has the wrong frame count")
        if values.dtype.kind not in "bUS" and not np.isfinite(values).all():
            raise ValueError(f"schema-v8 trace field {name} contains NaN or Inf")
    quaternion = np.asarray(traces["cube_quat"], dtype=np.float64)
    if quaternion.shape != (total_steps, 4) or not np.allclose(
        np.linalg.norm(quaternion, axis=1), 1.0, rtol=0.0, atol=1e-5
    ):
        raise ValueError("schema-v8 trace cube quaternion is invalid")
    return traces


def _recomputed_trace_summary(
    config: dict[str, Any], traces: dict[str, np.ndarray]
) -> dict[str, Any]:
    # Keep this import local: _phase_steps is an established compatibility
    # helper, but it is intentionally not part of the simulation public API.
    from .simulation import _phase_steps

    model, info = build_model(config)
    return evaluate_trace(model, info, config, _phase_steps(model, config), traces)


def _verify_same_run_trace(
    config: dict[str, Any], trace_path: Path, summary: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    traces = _load_trace_archive(trace_path)
    recomputed = _recomputed_trace_summary(config, copy.deepcopy(traces))
    persisted = copy.deepcopy(dict(summary))
    # Video is probed after evaluate_trace and therefore is the only field not
    # derivable from the NPZ itself.
    persisted.pop("video", None)
    if json_text(recomputed) != json_text(persisted):
        raise ValueError("simulation summary does not recompute from its trace")
    return traces


def _validated_run_metadata(config_path: Path) -> dict[str, Any]:
    value = run_metadata(config_path)
    if not isinstance(value, Mapping):
        raise RuntimeError("run_metadata must return a mapping")
    metadata = copy.deepcopy(dict(value))
    missing = [name for name in _REQUIRED_METADATA_KEYS if name not in metadata]
    if missing:
        raise RuntimeError("run metadata is missing: " + ", ".join(missing))
    for name in _REQUIRED_METADATA_KEYS:
        if not isinstance(metadata[name], str) or not metadata[name]:
            raise RuntimeError(f"run metadata field {name} must be a string")
    for name in (
        "model_sha256",
        "uv_lock_sha256",
        "pyproject_sha256",
        "implementation_sha256",
        "config_sha256",
    ):
        if _SHA256.fullmatch(metadata[name]) is None:
            raise RuntimeError(f"run metadata field {name} is not a SHA-256")
    if metadata["config_sha256"] != file_sha256(config_path):
        raise RuntimeError("run metadata does not bind the resolved config")
    return metadata


def _verify_published_video(
    video_path: Path, summary: Mapping[str, Any], traces: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    if not video_path.is_file() or video_path.stat().st_size <= 0:
        raise RuntimeError("run_simulation did not create the requested MP4")
    expected_steps = np.asarray(traces["video_frame_steps"], dtype=np.int64)
    if expected_steps.ndim != 1 or expected_steps.size == 0:
        raise RuntimeError("video run has no frame-to-simulation-step binding")
    video_summary = summary.get("video")
    if not isinstance(video_summary, Mapping):
        raise RuntimeError("video run summary has no ffprobe evidence")
    declared_steps = video_summary.get("simulation_step_indices")
    if list(expected_steps) != list(declared_steps or []):
        raise RuntimeError("video frame steps disagree with the NPZ trace")
    probed = probe_video(video_path, int(expected_steps.size), VideoSettings())
    for name in (
        "decode_verified",
        "codec",
        "width",
        "height",
        "fps",
        "frame_count",
    ):
        if video_summary.get(name) != probed.get(name):
            raise RuntimeError(f"video summary disagrees with ffprobe field {name}")
    return probed


def load_catalog_candidate(result_path: str | Path) -> CatalogCandidate:
    result_file = Path(result_path).expanduser().resolve()
    result = _load_object(result_file)
    if result.get("complete") is not True:
        raise ValueError(f"candidate result is incomplete: {result_file}")
    if int(result.get("candidate_result_schema_version", 0)) != (
        CANDIDATE_RESULT_SCHEMA_VERSION
    ):
        raise ValueError("candidate result has the wrong schema version")
    if result.get("campaign_kind") != CAMPAIGN_KIND:
        raise ValueError("candidate belongs to a different tuning campaign")
    stage = str(result.get("stage", ""))
    if stage != "lift_exact":
        raise ValueError("only locked-timestep lift_exact candidates are publishable")
    artifacts = result.get("artifacts", {})
    if not isinstance(artifacts, Mapping):
        raise ValueError(f"candidate has no artifact map: {result_file}")
    config_file = _member(
        result_file.parent,
        artifacts.get("resolved_config", "resolved_config.json"),
        "resolved_config",
    )
    trace_file = _member(
        result_file.parent,
        artifacts.get("trace"),
        "trace",
    )
    hashes = artifacts.get("sha256", {})
    config_digest = _required_digest(hashes, "resolved_config")
    trace_digest = _required_digest(hashes, "trace")
    if file_sha256(config_file) != config_digest:
        raise ValueError(f"candidate config SHA-256 mismatch: {config_file}")
    if file_sha256(trace_file) != trace_digest:
        raise ValueError(f"candidate trace SHA-256 mismatch: {trace_file}")
    config = _load_object(config_file)
    validate_config(config)
    if int(config.get("schema_version", 0)) != 8:
        raise ValueError("normal-aligned catalog requires schema version 8")
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("candidate belongs to a different experiment")
    if config.get("run_context") is not None:
        raise ValueError("override and robustness configurations are not publishable")
    semantic_digest = canonical_sha256(config)
    candidate_digest = result.get("candidate_sha256")
    if candidate_digest != semantic_digest:
        raise ValueError("candidate semantic SHA-256 does not bind its config")
    candidate_value = result.get("candidate_id")
    if isinstance(candidate_value, bool):
        raise ValueError("candidate_id must be a non-negative integer")
    try:
        candidate_number = int(candidate_value)
    except (TypeError, ValueError) as error:
        raise ValueError("candidate_id must be a non-negative integer") from error
    if candidate_number < 0 or candidate_number != candidate_value:
        raise ValueError("candidate_id must be a non-negative integer")
    _require_locked_exact_provenance(config, candidate_number)
    expected_pose_id = pose_id(config)
    expected_controller_id = controller_id(config)
    if result.get("pose_id") != expected_pose_id:
        raise ValueError("candidate pose_id does not bind its geometry")
    if result.get("controller_id") != expected_controller_id:
        raise ValueError("candidate controller_id does not bind its control law")
    summary = result.get("summary", {})
    if not isinstance(summary, Mapping):
        raise ValueError("candidate result has no summary")
    _verify_same_run_trace(config, trace_file, summary)
    evidence = lift_candidate_evidence(result)
    if bool(result.get("lift_success", False)) != bool(evidence["lift_success"]):
        raise ValueError("candidate lift_success disagrees with its authenticated evidence")
    reported_full_success = _hard_pass(summary)
    if bool(evidence["lift_success"]) != reported_full_success:
        raise ValueError("candidate hard-pass summary lacks complete lift evidence")
    return CatalogCandidate(
        result_path=result_file,
        config_path=config_file,
        result=result,
        config=config,
        candidate_id=str(candidate_number),
        full_success=reported_full_success,
        edge_mm=int(round(float(config["cube"]["edge_m"]) * 1000.0)),
        thumb_target_rad=float(
            config["control"]["grasp_targets_rad"][THUMB_BEND_ACTUATOR]
        ),
        stage=stage,
        pose_id=expected_pose_id,
        controller_id=expected_controller_id,
        candidate_sha256=semantic_digest,
        trace_path=trace_file,
    )


def discover_catalog_candidates(search_roots: Iterable[str | Path]) -> tuple[CatalogCandidate, ...]:
    found: dict[tuple[str, str], CatalogCandidate] = {}
    for raw in search_roots:
        root = Path(raw).expanduser().resolve()
        paths = (root,) if root.is_file() else sorted(root.rglob("result.json"))
        for path in paths:
            try:
                candidate = load_catalog_candidate(path)
            except (FileNotFoundError, KeyError, TypeError, ValueError):
                continue
            key = (candidate.candidate_id, candidate.candidate_sha256)
            previous = found.get(key)
            if previous is None:
                found[key] = candidate
                continue
            # The tuner's own Viewer catalog contains byte-for-byte copies of
            # exact candidates. Prefer the original candidate directory so
            # provenance never depends on that intermediate publication.
            previous_copy = "trajectory_catalog" in previous.result_path.parts
            candidate_copy = "trajectory_catalog" in candidate.result_path.parts
            if previous_copy and not candidate_copy:
                found[key] = candidate
    return tuple(sorted(found.values(), key=lambda value: value.rank))


def select_diverse_successes(
    candidates: Iterable[CatalogCandidate],
    *,
    count: int = 5,
    minimum_edges: int = 3,
    minimum_thumb_bands: int = 2,
) -> tuple[CatalogCandidate, ...]:
    """Choose the best deterministic combination satisfying diversity if possible."""

    passing = sorted(
        (value for value in candidates if value.full_success),
        key=lambda value: value.rank,
    )
    if count <= 0:
        raise ValueError("count must be positive")
    if minimum_edges <= 0 or minimum_thumb_bands <= 0:
        raise ValueError("diversity minima must be positive")

    # A pose can be represented by several controller candidates. Publishing
    # two of them would overstate trajectory diversity, so retain only its
    # highest-ranked exact rerun source.
    unique: list[CatalogCandidate] = []
    seen_poses: set[str] = set()
    for value in passing:
        if value.pose_key in seen_poses:
            continue
        seen_poses.add(value.pose_key)
        unique.append(value)
    if len(unique) <= count:
        return tuple(unique)

    quota_seed_size = max(minimum_edges, minimum_thumb_bands)
    best_key: tuple[Any, ...] | None = None
    selected: tuple[CatalogCandidate, ...] | None = None
    if quota_seed_size <= count:
        for seed in itertools.combinations(unique, quota_seed_size):
            if (
                len({value.edge_mm for value in seed}) < minimum_edges
                or len({value.thumb_band for value in seed})
                < minimum_thumb_bands
            ):
                continue
            chosen = list(seed)
            chosen_keys = {value.pose_key for value in seed}
            for value in unique:
                if value.pose_key in chosen_keys:
                    continue
                chosen.append(value)
                chosen_keys.add(value.pose_key)
                if len(chosen) == count:
                    break
            if len(chosen) != count:
                continue
            values = tuple(sorted(chosen, key=lambda value: value.rank))
            key = tuple(value.rank for value in values)
            if best_key is None or key < best_key:
                best_key = key
                selected = values
    if selected is not None:
        return selected
    return tuple(unique[:count])


def _token(candidate: CatalogCandidate, index: int) -> str:
    thumb = f"{candidate.thumb_target_rad:.3f}".replace(".", "p")
    raw = f"smooth_lift_{index + 1:02d}_{candidate.edge_mm}mm_thumb_{thumb}"
    return _SAFE_TOKEN.sub("_", raw)


def export_normal_aligned_smooth_lift_catalog(
    candidates: Iterable[CatalogCandidate],
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    selected_count: int = 5,
    include_best_near_miss: bool = True,
    video: bool = True,
) -> dict[str, Any]:
    """Rerun and transactionally publish selected v8 trajectories."""

    source = tuple(candidates)
    if not source:
        raise ValueError("at least one candidate is required")
    if selected_count <= 0:
        raise ValueError("selected_count must be positive")
    for candidate in source:
        if candidate.stage != "lift_exact":
            raise ValueError("only lift_exact candidates are publishable")
        try:
            candidate_number = int(candidate.candidate_id)
        except (TypeError, ValueError) as error:
            raise ValueError("candidate_id must be a non-negative integer") from error
        if candidate_number < 0:
            raise ValueError("candidate_id must be a non-negative integer")
        _require_locked_exact_provenance(candidate.config, candidate_number)
        validate_config(candidate.config)
        if (
            int(candidate.config.get("schema_version", 0)) != 8
            or candidate.config.get("experiment_id") != EXPERIMENT_ID
        ):
            raise ValueError("all catalog candidates must belong to schema-v8")
        if candidate.config.get("run_context") is not None:
            raise ValueError("override and robustness runs cannot be published")
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    successes = list(select_diverse_successes(source, count=selected_count))
    near_misses = sorted(
        (value for value in source if not value.full_success), key=lambda value: value.rank
    )
    selected: list[tuple[str, CatalogCandidate]] = [
        ("success", value) for value in successes
    ]
    if include_best_near_miss and near_misses:
        selected.append(("near_miss", near_misses[0]))
    entries: list[dict[str, Any]] = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output.parent, prefix=f".{output.name}.staging."
    ) as staging_name:
        staging = Path(staging_name)
        for index, (reported_classification, candidate) in enumerate(selected):
            trajectory_id = _token(candidate, index)
            directory = staging / trajectory_id
            directory.mkdir()
            trace_path = directory / "trace.npz"
            video_path = directory / "trajectory.mp4" if video else None
            summary = run_simulation(
                copy.deepcopy(candidate.config),
                trace_path=trace_path,
                video_path=video_path,
            )
            if not isinstance(summary, Mapping):
                raise RuntimeError("run_simulation must return a summary mapping")
            traces = _verify_same_run_trace(candidate.config, trace_path, summary)
            video_evidence = (
                _verify_published_video(video_path, summary, traces)
                if video_path is not None
                else None
            )
            rerun_pass = _hard_pass(summary)
            if reported_classification == "success" and not rerun_pass:
                raise RuntimeError(
                    f"published success failed deterministic rerun: {candidate.candidate_id}"
                )
            classification = "success" if rerun_pass else "near_miss"
            resolved = resolved_run_config(candidate.config, dict(summary))
            config_path = directory / "resolved_config.json"
            write_json(config_path, resolved)
            metadata = _validated_run_metadata(config_path)
            local_hashes = {
                "resolved_config": file_sha256(config_path),
                "trace": file_sha256(trace_path),
            }
            if video_path is not None:
                local_hashes["video"] = file_sha256(video_path)
            result_artifacts = {
                "resolved_config": config_path.name,
                "trace": trace_path.name,
                "video": video_path.name if video_path is not None else None,
                "sha256": copy.deepcopy(local_hashes),
            }
            result_payload = {
                "catalog_result_schema_version": CATALOG_SCHEMA_VERSION,
                "experiment_id": EXPERIMENT_ID,
                "trajectory_id": trajectory_id,
                "classification": classification,
                "reported_classification": reported_classification,
                "reported_full_success": candidate.full_success,
                "rerun_full_success": rerun_pass,
                "trace_summary_consistent": True,
                "video_verified": bool(
                    video_evidence is not None
                    and video_evidence.get("decode_verified", False)
                ),
                "source_candidate_id": candidate.candidate_id,
                "source_stage": candidate.stage,
                "source_pose_id": candidate.pose_id,
                "source_controller_id": candidate.controller_id,
                "source_candidate_sha256": candidate.candidate_sha256,
                "source_provenance": {
                    "result": str(candidate.result_path),
                    "result_sha256": file_sha256(candidate.result_path),
                    "resolved_config": str(candidate.config_path),
                    "resolved_config_sha256": file_sha256(candidate.config_path),
                    "trace": (
                        str(candidate.trace_path)
                        if candidate.trace_path is not None
                        else None
                    ),
                    "trace_sha256": (
                        file_sha256(candidate.trace_path)
                        if candidate.trace_path is not None
                        else None
                    ),
                },
                "summary": copy.deepcopy(dict(summary)),
                "config": resolved,
                "experiment_status": copy.deepcopy(
                    resolved.get("experiment_status", {})
                ),
                "metadata": metadata,
                "artifacts": result_artifacts,
            }
            result_path = directory / "result.json"
            write_json(result_path, result_payload)
            artifacts = {
                "resolved_config": f"{trajectory_id}/resolved_config.json",
                "trace": f"{trajectory_id}/trace.npz",
                "result": f"{trajectory_id}/result.json",
                "video": (
                    f"{trajectory_id}/trajectory.mp4" if video_path is not None else None
                ),
                "sha256": {
                    **local_hashes,
                    "result": file_sha256(result_path),
                },
            }
            entries.append(
                {
                    "trajectory_id": trajectory_id,
                    "label": trajectory_id,
                    "aliases": [],
                    "classification": classification,
                    "full_success": rerun_pass,
                    "reported_full_success": candidate.full_success,
                    "trace_summary_consistent": True,
                    "edge_mm": candidate.edge_mm,
                    "mass_g": float(candidate.config["cube"]["mass_kg"]) * 1000.0,
                    "friction": float(candidate.config["cube"]["friction"]),
                    "thumb_target_rad": candidate.thumb_target_rad,
                    "pose_id": candidate.pose_key,
                    "controller_id": candidate.controller_id,
                    "stage_status": copy.deepcopy(
                        summary.get("stage_status", {})
                    ),
                    "metrics": copy.deepcopy(summary.get("metrics", {})),
                    "failed_checks": copy.deepcopy(
                        summary.get("failed_checks", [])
                    ),
                    "artifacts": artifacts,
                }
            )

        aliases: dict[str, str] = {}
        success_index = 0
        attempt_index = 0
        for entry in entries:
            if entry["classification"] == "success":
                success_index += 1
                names = [f"trajectory_{success_index}"]
                if success_index == 1:
                    names.append("best_nominal")
            else:
                attempt_index += 1
                names = [f"attempt_{attempt_index}"]
                if attempt_index == 1:
                    names.append("best_attempt")
            entry["aliases"] = names
            for alias in names:
                aliases[alias] = entry["trajectory_id"]

        success_entries = [
            entry for entry in entries if entry["classification"] == "success"
        ]
        edge_count = len(
            {entry["edge_mm"] for entry in success_entries}
        )
        band_count = len(
            {
                (
                    "low"
                    if float(entry["thumb_target_rad"]) <= 1.31 + 1e-12
                    else "middle"
                    if float(entry["thumb_target_rad"]) <= 1.38 + 1e-12
                    else "high"
                )
                for entry in success_entries
            }
        )
        pose_count = len({entry["pose_id"] for entry in success_entries})
        catalog = {
            "normal_aligned_smooth_lift_catalog_schema_version": CATALOG_SCHEMA_VERSION,
            "trajectory_catalog_schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "complete": True,
            "success_count": len(success_entries),
            "reported_success_count": len(successes),
            "requested_success_count": selected_count,
            "distinct_success_edges": edge_count,
            "distinct_success_thumb_bands": band_count,
            "distinct_success_poses": pose_count,
            "selection_quota_satisfied": bool(
                len(success_entries) >= selected_count
                and edge_count >= 3
                and band_count >= 2
                and pose_count >= selected_count
            ),
            "aliases": aliases,
            "trajectories": entries,
        }
        write_json(staging / "catalog.json", catalog)
        staging.rename(output)
    return catalog


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish schema-v8 normal-aligned smooth-lift trajectories"
    )
    parser.add_argument("search_root", nargs="+", help="candidate search roots")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--selected-count", type=int, default=5)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-near-miss", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    candidates = discover_catalog_candidates(args.search_root)
    catalog = export_normal_aligned_smooth_lift_catalog(
        candidates,
        args.output_dir,
        selected_count=args.selected_count,
        include_best_near_miss=not args.no_near_miss,
        video=not args.no_video,
    )
    print(json.dumps(catalog, indent=2, sort_keys=True))
    return 0 if catalog["selection_quota_satisfied"] else 2


__all__ = [
    "CATALOG_SCHEMA_VERSION",
    "CatalogCandidate",
    "DEFAULT_OUTPUT",
    "EXPERIMENT_ID",
    "build_parser",
    "discover_catalog_candidates",
    "export_normal_aligned_smooth_lift_catalog",
    "load_catalog_candidate",
    "main",
    "select_diverse_successes",
]
