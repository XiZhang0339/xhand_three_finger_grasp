"""Authenticated resume state and Viewer catalogs for schema-v9 searches.

The actual-contact campaign can take several independent invocations to find
its first trajectory and then extend that set to five.  This module keeps the
filesystem protocol deliberately small: immutable campaign inputs are bound in
``campaign_manifest.json``; stage completion is recorded atomically in
``stage_ledger.json``; and already-produced trajectories are copied into
Viewer-compatible, path-confined catalogs.

Search code is intentionally not imported here.  That keeps resume validation
and publication cheap to test and lets a search process recover its last fully
committed stage without importing MuJoCo worker infrastructure.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

import numpy as np

from .actual_contact_capability import (
    LEGACY_ACTUAL_CONTACT_EXPERIMENT_ID,
    resolve_actual_contact_definition,
)
from .artifacts import (
    REPO_ROOT,
    aggregate_source_sha256,
    file_sha256,
    implementation_paths,
    json_compatible,
    json_text,
    write_json,
)
from .actual_contact_selection import (
    final_candidate_rank,
    final_candidate_rank_evidence,
    select_actual_contact_candidates,
)
from .rendering import VideoSettings, probe_video
from .simulation import run_simulation


EXPERIMENT_ID = LEGACY_ACTUAL_CONTACT_EXPERIMENT_ID
CAMPAIGN_MANIFEST_SCHEMA_VERSION = 1
STAGE_LEDGER_SCHEMA_VERSION = 1
TRAJECTORY_CATALOG_SCHEMA_VERSION = 1
RESULT_SEMANTIC_SHA256_FIELD = "result_semantic_sha256"
_SAFE_TOKEN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


FinalSimulationRunner = Callable[..., Mapping[str, Any]]
FinalVideoProbe = Callable[[Path, int, VideoSettings | None], Mapping[str, Any]]


def _canonical_sha256(value: Any) -> str:
    # MuJoCo/NumPy evaluation summaries legitimately contain NumPy scalar
    # values (notably ``np.bool_``).  Bind the exact strict-JSON form that is
    # persisted by ``write_json`` rather than asking the standard encoder to
    # handle NumPy implementation types directly.
    encoded = json.dumps(
        json_compatible(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def candidate_result_semantic_sha256(payload: Mapping[str, Any]) -> str:
    """Hash candidate evidence independently of mutable artifact storage.

    Trace compaction and final catalog publication legitimately rewrite the
    ``artifacts`` mapping.  Candidate identity, classification, rank evidence
    and the complete simulation summary are semantic evidence and therefore
    remain covered by this digest.
    """

    semantic = copy.deepcopy(dict(payload))
    semantic.pop(RESULT_SEMANTIC_SHA256_FIELD, None)
    semantic.pop("artifacts", None)
    return _canonical_sha256(semantic)


def bind_candidate_result_semantic_sha256(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a copy carrying its canonical candidate-evidence digest."""

    bound = copy.deepcopy(dict(payload))
    bound.pop(RESULT_SEMANTIC_SHA256_FIELD, None)
    bound[RESULT_SEMANTIC_SHA256_FIELD] = candidate_result_semantic_sha256(bound)
    return bound


def authenticate_candidate_result_semantic_sha256(
    payload: Mapping[str, Any],
    *,
    source: str | Path = "candidate result",
) -> str:
    """Reject a candidate result whose semantic evidence was modified."""

    expected = payload.get(RESULT_SEMANTIC_SHA256_FIELD)
    if not isinstance(expected, str) or _SHA256.fullmatch(expected) is None:
        raise RuntimeError(f"{source} has no valid semantic SHA-256")
    actual = candidate_result_semantic_sha256(payload)
    if actual != expected:
        raise RuntimeError(f"{source} semantic SHA-256 mismatch")
    return actual


def build_campaign_manifest(
    config_path: str | Path,
    *,
    seed: int,
    source_paths: Sequence[str | Path] | None = None,
) -> dict[str, Any]:
    """Bind every immutable input required for a safe schema-v9 resume.

    ``target_success_count`` is deliberately absent: increasing the requested
    count from one to five resumes the *same* search rather than changing its
    deterministic candidate stream.
    """

    config = Path(config_path).expanduser().resolve()
    model = (REPO_ROOT / "xhand_left.xml").resolve()
    lock = (REPO_ROOT / "uv.lock").resolve()
    if not config.is_file():
        raise FileNotFoundError(config)
    if int(seed) < 0:
        raise ValueError("seed must be non-negative")
    sources = tuple(
        Path(path).expanduser().resolve()
        for path in (source_paths if source_paths is not None else implementation_paths())
    )
    if not sources or any(not path.is_file() for path in sources):
        raise FileNotFoundError("every resume source path must be an existing file")
    config_payload = json.loads(config.read_text(encoding="utf-8"))
    if not isinstance(config_payload, Mapping):
        raise ValueError("campaign config must be a JSON object")
    definition = resolve_actual_contact_definition(
        config_payload, context="actual-contact campaign manifest"
    )
    campaign = definition.actual_contact_grasp_pose_campaign
    if campaign is None:  # Capability resolution above makes this defensive.
        raise ValueError("actual-contact campaign manifest has no campaign")
    source_manifest = Path(campaign.source_pose_manifest).expanduser()
    if not source_manifest.is_absolute():
        source_manifest = (REPO_ROOT / source_manifest).resolve()
    else:
        source_manifest = source_manifest.resolve()
    if not source_manifest.is_file():
        raise FileNotFoundError(source_manifest)
    bound = {
        "campaign_manifest_schema_version": CAMPAIGN_MANIFEST_SCHEMA_VERSION,
        "experiment_id": definition.experiment_id,
        "seed": int(seed),
        "config_path": str(config),
        "config_sha256": file_sha256(config),
        "model_path": str(model),
        "model_sha256": file_sha256(model),
        "uv_lock_path": str(lock),
        "uv_lock_sha256": file_sha256(lock),
        "actual_qpos_source_manifest_path": (
            str(source_manifest.relative_to(REPO_ROOT))
            if source_manifest.is_relative_to(REPO_ROOT)
            else str(source_manifest)
        ),
        "actual_qpos_source_manifest_sha256": file_sha256(source_manifest),
        "source_sha256": aggregate_source_sha256(sources),
        "source_files": [
            str(path.relative_to(REPO_ROOT))
            if path.is_relative_to(REPO_ROOT)
            else str(path)
            for path in sorted(sources, key=str)
        ],
    }
    return {**bound, "campaign_input_sha256": _canonical_sha256(bound)}


def initialize_or_resume_campaign(
    output_dir: str | Path,
    manifest: Mapping[str, Any],
    *,
    resume: bool,
) -> Path:
    """Create a campaign workspace or authenticate an existing one exactly."""

    output = Path(output_dir).expanduser().resolve()
    manifest_path = output / "campaign_manifest.json"
    ledger_path = output / "stage_ledger.json"
    expected = copy.deepcopy(dict(manifest))
    if output.exists():
        if not resume:
            raise FileExistsError(
                f"output directory already exists: {output}; pass --resume"
            )
        if not manifest_path.is_file() or not ledger_path.is_file():
            raise RuntimeError(
                "resume output is missing campaign_manifest.json or stage_ledger.json"
            )
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        hash_fields = (
            "config_sha256",
            "model_sha256",
            "uv_lock_sha256",
            "actual_qpos_source_manifest_sha256",
            "source_sha256",
            "campaign_input_sha256",
        )
        mismatches = [
            name for name in hash_fields if existing.get(name) != expected.get(name)
        ]
        if mismatches:
            raise RuntimeError(
                "resume inputs do not match the existing schema-v9 campaign: "
                + ", ".join(mismatches)
            )
        validate_stage_ledger(output)
        return output

    output.mkdir(parents=True)
    write_json(manifest_path, expected)
    write_json(
        ledger_path,
        {
            "stage_ledger_schema_version": STAGE_LEDGER_SCHEMA_VERSION,
            "experiment_id": expected.get("experiment_id"),
            "campaign_input_sha256": expected.get("campaign_input_sha256"),
            "stages": {},
        },
    )
    return output


def validate_stage_ledger(output_dir: str | Path) -> dict[str, Any]:
    """Authenticate every artifact of every committed stage."""

    output = Path(output_dir).expanduser().resolve()
    ledger_path = output / "stage_ledger.json"
    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    if payload.get("stage_ledger_schema_version") != STAGE_LEDGER_SCHEMA_VERSION:
        raise RuntimeError("unsupported or malformed schema-v9 stage ledger")
    manifest_path = output / "campaign_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("experiment_id") != manifest.get("experiment_id"):
            raise RuntimeError(
                "stage ledger experiment differs from campaign manifest"
            )
        if payload.get("campaign_input_sha256") != manifest.get(
            "campaign_input_sha256"
        ):
            raise RuntimeError(
                "stage ledger input digest differs from campaign manifest"
            )
    stages = payload.get("stages")
    if not isinstance(stages, dict):
        raise RuntimeError("schema-v9 stage ledger has no stage mapping")
    for stage_name, record in stages.items():
        if _SAFE_TOKEN.fullmatch(str(stage_name)) is None or not isinstance(record, dict):
            raise RuntimeError("schema-v9 stage ledger contains an invalid stage")
        artifacts = record.get("artifacts")
        if not isinstance(artifacts, dict):
            raise RuntimeError(f"committed stage {stage_name} has no artifact map")
        for relative, expected in artifacts.items():
            if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise RuntimeError(f"committed stage {stage_name} has an unsafe artifact path")
            if not isinstance(expected, str) or _SHA256.fullmatch(expected) is None:
                raise RuntimeError(f"committed stage {stage_name} has an invalid digest")
            path = (output / relative).resolve()
            if not path.is_relative_to(output) or not path.is_file():
                raise RuntimeError(f"committed stage artifact is missing: {relative}")
            if file_sha256(path) != expected:
                raise RuntimeError(f"committed stage artifact SHA-256 mismatch: {relative}")
    return payload


def commit_campaign_stage(
    output_dir: str | Path,
    stage: str,
    *,
    stage_input: Mapping[str, Any],
    artifacts: Sequence[str | Path],
    summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically add one complete, hash-bound stage to the recovery ledger."""

    if _SAFE_TOKEN.fullmatch(stage) is None:
        raise ValueError("stage must be a safe lowercase token")
    output = Path(output_dir).expanduser().resolve()
    payload = validate_stage_ledger(output)
    artifact_hashes: dict[str, str] = {}
    for value in artifacts:
        path = Path(value).expanduser().resolve()
        if not path.is_relative_to(output) or not path.is_file():
            raise ValueError("stage artifacts must be existing files inside output_dir")
        relative = str(path.relative_to(output))
        artifact_hashes[relative] = file_sha256(path)
    record = {
        "stage_input_sha256": _canonical_sha256(dict(stage_input)),
        "artifacts": dict(sorted(artifact_hashes.items())),
        "summary": copy.deepcopy(dict(summary or {})),
        "complete": True,
    }
    stages = payload["stages"]
    previous = stages.get(stage)
    if previous is not None and previous != record:
        raise RuntimeError(f"stage {stage!r} is already committed with different evidence")
    stages[stage] = record
    write_json(output / "stage_ledger.json", payload)
    return copy.deepcopy(record)


@dataclass(frozen=True)
class _CatalogSource:
    candidate_id: str
    discovery_index: int
    config_path: Path
    result_path: Path
    trace_path: Path
    video_path: Path | None
    config: dict[str, Any]
    result: dict[str, Any]
    summary: dict[str, Any]
    parameter_override: bool

    @property
    def experiment_id(self) -> str:
        return str(self.config["experiment_id"])

    @property
    def grasp_success(self) -> bool:
        status = self.summary.get("stage_status")
        return bool(isinstance(status, Mapping) and status.get("grasp_success", False))

    @property
    def full_success(self) -> bool:
        status = self.summary.get("stage_status")
        return bool(
            self.summary.get("passed", False)
            and isinstance(status, Mapping)
            and status.get("full_success", False)
        )


class _CompactedFailureCandidate(RuntimeError):
    """Internal signal used to omit an intentionally compacted failure."""


def _confined_source_path(value: Any, field: str) -> Path:
    path = Path(str(value)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"candidate {field} does not exist: {path}")
    return path


def _load_source(
    raw: Mapping[str, Any],
    fallback: int,
    *,
    kind: Literal["grasp_pose", "manipulation"],
) -> _CatalogSource:
    result_path = _confined_source_path(raw.get("result_path"), "result_path")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError("candidate result must be a JSON object")
    production_result = bool(
        int(result.get("candidate_result_schema_version", 0)) == 1
        or int(result.get("actual_contact_manipulation_candidate_schema_version", 0))
        == 1
    )
    if production_result:
        authenticate_candidate_result_semantic_sha256(
            result, source=result_path
        )
    config_path = _confined_source_path(
        raw.get("config_path", result_path.parent / "resolved_config.json"),
        "config_path",
    )
    summary = raw.get("summary", result.get("summary"))
    if not isinstance(summary, Mapping):
        raise ValueError("candidate requires a summary object")
    persisted_summary = result.get("summary")
    if (
        isinstance(persisted_summary, Mapping)
        and _canonical_sha256(dict(summary))
        != _canonical_sha256(dict(persisted_summary))
    ):
        raise ValueError("candidate summary differs from authenticated result")
    artifacts = result.get("artifacts")
    trace_retained = not (
        isinstance(artifacts, Mapping)
        and artifacts.get("trace_retained") is False
    )
    if not trace_retained:
        status = summary.get("stage_status")
        relevant_success = bool(
            isinstance(status, Mapping)
            and (
                status.get("grasp_success")
                if kind == "grasp_pose"
                else summary.get("passed", False)
                and status.get("full_success")
            )
        )
        if relevant_success:
            raise ValueError("a successful candidate may not have a compacted trace")
        raise _CompactedFailureCandidate(str(result_path))
    trace_path = _confined_source_path(
        raw.get("trace_path", result_path.parent / "trace.npz"), "trace_path"
    )
    video_value = raw.get("video_path")
    video_path = (
        _confined_source_path(video_value, "video_path")
        if video_value is not None
        else None
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(summary, Mapping):
        raise ValueError("candidate requires object config and summary records")
    resolve_actual_contact_definition(
        config, context="actual-contact catalog source"
    )
    hashes = artifacts.get("sha256") if isinstance(artifacts, Mapping) else None
    for name, path in (("resolved_config", config_path), ("trace", trace_path)):
        if isinstance(hashes, Mapping) and name in hashes and hashes[name] != file_sha256(path):
            raise ValueError(f"candidate {name} SHA-256 mismatch")
    candidate_value = raw.get("candidate_id", result.get("candidate_id", fallback))
    if isinstance(candidate_value, bool):
        raise ValueError("candidate_id must be an integer or safe token")
    candidate_id = str(candidate_value)
    if _SAFE_TOKEN.fullmatch(candidate_id) is None:
        # Numeric IDs are canonicalized before checking the safe token.
        try:
            candidate_id = str(int(candidate_value))
        except (TypeError, ValueError) as exc:
            raise ValueError("candidate_id must be an integer or safe token") from exc
    if _SAFE_TOKEN.fullmatch(candidate_id) is None:
        raise ValueError("candidate_id must be an integer or safe token")
    run_context = config.get("run_context")
    # Only canonical full reruns may become campaign evidence.  In particular,
    # the old Viewer command override is diagnostic, and robustness trials must
    # not accidentally become a nominal ``best_first`` either.
    parameter_override = run_context is not None
    return _CatalogSource(
        candidate_id=candidate_id,
        discovery_index=int(raw.get("discovery_index", result.get("discovery_index", fallback))),
        config_path=config_path,
        result_path=result_path,
        trace_path=trace_path,
        video_path=video_path,
        config=copy.deepcopy(config),
        result=copy.deepcopy(result),
        summary=copy.deepcopy(dict(summary)),
        parameter_override=parameter_override,
    )


def _requires_final_video_rerun(source: _CatalogSource) -> bool:
    """Recognize real v9 search evidence without changing legacy test fixtures.

    The production dynamic and manipulation writers both emit an immutable
    result schema marker.  Older/synthetic catalog fixtures deliberately lack
    these markers and retain the historical copy-only behavior.
    """

    result = source.result
    return bool(
        result.get("complete") is True
        and (
            int(result.get("candidate_result_schema_version", 0)) == 1
            or int(
                result.get(
                    "actual_contact_manipulation_candidate_schema_version", 0
                )
            )
            == 1
        )
    )


def _array_equal_exact(left: np.ndarray, right: np.ndarray) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if np.issubdtype(left.dtype, np.inexact):
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def _verify_rerun_trace_matches_source(
    source_path: Path, rerun_path: Path
) -> dict[str, Any]:
    """Prove rendering did not change any recorded physics sample."""

    ignored = {"video_frame_steps"}
    with np.load(source_path, allow_pickle=False) as source, np.load(
        rerun_path, allow_pickle=False
    ) as rerun:
        source_fields = set(source.files) - ignored
        rerun_fields = set(rerun.files) - ignored
        if source_fields != rerun_fields:
            missing = sorted(source_fields - rerun_fields)
            added = sorted(rerun_fields - source_fields)
            raise RuntimeError(
                "final video rerun trace schema changed: "
                f"missing={missing}, added={added}"
            )
        mismatched = [
            name
            for name in sorted(source_fields)
            if not _array_equal_exact(
                np.asarray(source[name]), np.asarray(rerun[name])
            )
        ]
        if mismatched:
            raise RuntimeError(
                "final video rerun changed physical trajectory fields: "
                + ", ".join(mismatched)
            )
        frame_steps = np.asarray(
            rerun["video_frame_steps"] if "video_frame_steps" in rerun else (),
            dtype=np.int64,
        )
        if frame_steps.ndim != 1 or frame_steps.size == 0:
            raise RuntimeError("final video rerun has no frame-to-step binding")
    return {
        "source_trace_sha256": file_sha256(source_path),
        "rerun_trace_sha256": file_sha256(rerun_path),
        "compared_field_count": len(source_fields),
        "ignored_fields": sorted(ignored),
        "physical_fields_exact": True,
        "video_frame_steps": frame_steps.tolist(),
    }


def _summary_without_video(summary: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(summary))
    value.pop("video", None)
    return value


def _verify_final_video(
    video_path: Path,
    trace_path: Path,
    summary: Mapping[str, Any],
    video_probe: FinalVideoProbe,
) -> dict[str, Any]:
    if not video_path.is_file() or video_path.stat().st_size <= 0:
        raise RuntimeError("final simulation did not create trajectory.mp4")
    with np.load(trace_path, allow_pickle=False) as trace:
        frame_steps = np.asarray(trace["video_frame_steps"], dtype=np.int64)
    declared = summary.get("video")
    if not isinstance(declared, Mapping):
        raise RuntimeError("final simulation summary has no video evidence")
    if list(declared.get("simulation_step_indices", ())) != frame_steps.tolist():
        raise RuntimeError("final video summary and NPZ frame steps disagree")
    settings = VideoSettings()
    probed = copy.deepcopy(
        dict(video_probe(video_path, int(frame_steps.size), settings))
    )
    for field in (
        "decode_verified",
        "codec",
        "width",
        "height",
        "fps",
        "frame_count",
    ):
        if declared.get(field) != probed.get(field):
            raise RuntimeError(
                f"final video summary disagrees with ffprobe field {field}"
            )
    if probed.get("decode_verified") is not True:
        raise RuntimeError("final MP4 did not pass a complete decode")
    return probed


def _rerun_final_candidate_with_video(
    source: _CatalogSource,
    *,
    trace_path: Path,
    video_path: Path,
    simulation_runner: FinalSimulationRunner,
    video_probe: FinalVideoProbe,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = simulation_runner(
        copy.deepcopy(source.config),
        trace_path=trace_path,
        video_path=video_path,
    )
    if not isinstance(summary, Mapping):
        raise RuntimeError("final simulation runner must return a summary mapping")
    rerun_summary = copy.deepcopy(dict(summary))
    # Persisted JSON turns tuples into lists; compare their canonical JSON
    # representations so that representation-only changes do not masquerade
    # as a dynamics difference.
    if json_text(_summary_without_video(rerun_summary)) != json_text(
        _summary_without_video(source.summary)
    ):
        raise RuntimeError("final video rerun changed the deterministic result summary")
    trace_evidence = _verify_rerun_trace_matches_source(
        source.trace_path, trace_path
    )
    video_evidence = _verify_final_video(
        video_path, trace_path, rerun_summary, video_probe
    )
    return rerun_summary, {
        "rendered_from_initial_no_contact_state": True,
        "checkpoint_used": False,
        "trace_reproduction": trace_evidence,
        "ffprobe_and_full_decode": video_evidence,
    }


def _publish_catalog(
    candidates: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    kind: Literal["grasp_pose", "manipulation"],
    selected_count: int = 5,
    simulation_runner: FinalSimulationRunner | None = None,
    video_probe: FinalVideoProbe | None = None,
) -> dict[str, Any]:
    if not 1 <= int(selected_count) <= 5:
        raise ValueError("selected_count must be between 1 and 5")
    sources: list[_CatalogSource] = []
    for index, value in enumerate(candidates):
        try:
            sources.append(_load_source(value, index, kind=kind))
        except _CompactedFailureCandidate:
            # Global trace compaction is allowed only for failed candidates;
            # such a record cannot become a success or Viewer diagnostic.
            continue
    if not sources:
        raise ValueError("at least one trajectory candidate is required")
    experiment_ids = {value.experiment_id for value in sources}
    if len(experiment_ids) != 1:
        raise ValueError("catalog candidates must belong to one experiment")
    experiment_id = experiment_ids.pop()
    definition = resolve_actual_contact_definition(
        sources[0].config, context="actual-contact catalog publication"
    )
    campaign = definition.actual_contact_grasp_pose_campaign
    validation_labels = (
        None if campaign is None else campaign.validation_labels
    )
    sources.sort(key=lambda value: (value.discovery_index, value.candidate_id))
    source_by_id = {value.candidate_id: value for value in sources}
    if len(source_by_id) != len(sources):
        raise ValueError("catalog candidates must have unique candidate IDs")
    selection = select_actual_contact_candidates(
        (
            {
                "candidate_id": value.candidate_id,
                "discovery_index": value.discovery_index,
                "config": value.config,
                "summary": value.summary,
                "parameter_override": value.parameter_override,
            }
            for value in sources
        ),
        kind=kind,
        selected_count=int(selected_count),
    )
    successes = [
        source_by_id[str(value["candidate_id"])] for value in selection.selected
    ]
    selected_ids = {value.candidate_id for value in successes}
    diagnostic_pool = sorted(
        (value for value in sources if value.candidate_id not in selected_ids),
        key=lambda value: final_candidate_rank(
            {
                "candidate_id": value.candidate_id,
                "discovery_index": value.discovery_index,
                "config": value.config,
                "summary": value.summary,
                "parameter_override": value.parameter_override,
            }
        ),
    )
    selected = [*(('success', value) for value in successes)]
    if diagnostic_pool:
        selected.append(("diagnostic", diagnostic_pool[0]))

    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    aliases: dict[str, str] = {}
    execute = run_simulation if simulation_runner is None else simulation_runner
    inspect_video = probe_video if video_probe is None else video_probe
    with tempfile.TemporaryDirectory(
        dir=output.parent, prefix=f".{output.name}.staging."
    ) as staging_name:
        staging = Path(staging_name)
        success_alias_index = 0
        best_first_candidate_id = selection.metadata["best_first_candidate_id"]
        for index, (classification, source) in enumerate(selected):
            trajectory_id = f"{kind}_{index + 1:02d}_{source.candidate_id}"
            member = staging / trajectory_id
            member.mkdir()
            destinations = {
                "resolved_config": member / "resolved_config.json",
                "result": member / "result.json",
                "trace": member / "trace.npz",
            }
            shutil.copy2(source.config_path, destinations["resolved_config"])
            published_summary = copy.deepcopy(source.summary)
            final_video_evidence: dict[str, Any] | None = None
            video_destination: Path | None = None
            production_source = _requires_final_video_rerun(source)
            if production_source:
                video_destination = member / "trajectory.mp4"
                published_summary, final_video_evidence = (
                    _rerun_final_candidate_with_video(
                        source,
                        trace_path=destinations["trace"],
                        video_path=video_destination,
                        simulation_runner=execute,
                        video_probe=inspect_video,
                    )
                )
                result_payload = copy.deepcopy(source.result)
                result_payload["summary"] = copy.deepcopy(published_summary)
                result_payload["final_video_publication"] = copy.deepcopy(
                    final_video_evidence
                )
                result_payload["artifacts"] = {
                    "resolved_config": destinations["resolved_config"].name,
                    "trace": destinations["trace"].name,
                    "video": video_destination.name,
                    "sha256": {
                        "resolved_config": file_sha256(
                            destinations["resolved_config"]
                        ),
                        "trace": file_sha256(destinations["trace"]),
                        "video": file_sha256(video_destination),
                    },
                }
                result_payload = bind_candidate_result_semantic_sha256(
                    result_payload
                )
                write_json(destinations["result"], result_payload)
            else:
                shutil.copy2(source.result_path, destinations["result"])
                shutil.copy2(source.trace_path, destinations["trace"])
                if source.video_path is not None:
                    video_destination = member / "trajectory.mp4"
                    shutil.copy2(source.video_path, video_destination)
            hashes = {
                name: file_sha256(path) for name, path in destinations.items()
            }
            if video_destination is not None:
                hashes["video"] = file_sha256(video_destination)
            entry_aliases: list[str] = []
            if classification == "success":
                success_alias_index += 1
                alias = f"{kind}_{success_alias_index}"
                entry_aliases.append(alias)
                aliases[alias] = trajectory_id
                if success_alias_index == 1:
                    entry_aliases.append("best_nominal")
                    aliases["best_nominal"] = trajectory_id
                if source.candidate_id == best_first_candidate_id:
                    entry_aliases.append("best_first")
                    aliases["best_first"] = trajectory_id
            elif not successes:
                entry_aliases.append("best_attempt")
                aliases["best_attempt"] = trajectory_id
            entry = {
                "trajectory_id": trajectory_id,
                "label": trajectory_id,
                "aliases": entry_aliases,
                "classification": (
                    "diagnostic_override"
                    if source.parameter_override
                    else classification
                ),
                "grasp_success": source.grasp_success,
                "full_success": source.full_success,
                "candidate_id": source.candidate_id,
                "discovery_index": source.discovery_index,
                "final_rank_evidence": final_candidate_rank_evidence(
                    {
                        "candidate_id": source.candidate_id,
                        "discovery_index": source.discovery_index,
                        "config": source.config,
                        "summary": published_summary,
                        "parameter_override": source.parameter_override,
                    }
                ),
                "stage_status": copy.deepcopy(
                    published_summary.get("stage_status", {})
                ),
                "failed_checks": copy.deepcopy(
                    published_summary.get("failed_checks", [])
                ),
                "final_video_required": production_source,
                "final_video_verified": bool(
                    final_video_evidence is not None
                    and final_video_evidence.get("ffprobe_and_full_decode", {}).get(
                        "decode_verified", False
                    )
                ),
                "final_video_evidence": copy.deepcopy(final_video_evidence),
                "artifacts": {
                    "resolved_config": f"{trajectory_id}/resolved_config.json",
                    "result": f"{trajectory_id}/result.json",
                    "trace": f"{trajectory_id}/trace.npz",
                    "video": (
                        f"{trajectory_id}/trajectory.mp4"
                        if video_destination is not None
                        else None
                    ),
                    "sha256": hashes,
                },
            }
            if classification == "success" and validation_labels is not None:
                entry["validation_label"] = validation_labels[
                    "grasp" if kind == "grasp_pose" else "manipulation"
                ]
            entries.append(entry)
        catalog = {
            "actual_contact_grasp_pose_catalog_schema_version": (
                TRAJECTORY_CATALOG_SCHEMA_VERSION
            ),
            "trajectory_catalog_schema_version": 1,
            "experiment_id": experiment_id,
            "catalog_kind": kind,
            "complete": True,
            "requested_success_count": int(selected_count),
            "success_count": len(successes),
            "eligible_success_count": selection.metadata[
                "eligible_success_count"
            ],
            "target_reached": selection.metadata["target_reached"],
            "selection": copy.deepcopy(selection.metadata),
            "diversity": copy.deepcopy(selection.metadata["diversity"]),
            "production_trajectory_video_policy": (
                "deterministic_full_reset_rerun_ffprobe_and_full_decode"
            ),
            "aliases": aliases,
            "trajectories": entries,
        }
        if validation_labels is not None:
            catalog["validation_label"] = validation_labels[
                "grasp" if kind == "grasp_pose" else "manipulation"
            ]
        write_json(staging / "catalog.json", catalog)
        staging.rename(output)
    return catalog


def authenticated_catalog_artifact_paths(
    catalog_path: str | Path,
) -> tuple[Path, ...]:
    """Return and authenticate every file referenced by one v9 catalog.

    Campaign stage ledgers use this expansion so a valid ``catalog.json`` can
    never mask a deleted or modified member config, result, trace or video.
    """

    catalog = Path(catalog_path).expanduser().resolve()
    if not catalog.is_file():
        raise FileNotFoundError(catalog)
    payload = json.loads(catalog.read_text(encoding="utf-8"))
    experiment_id = payload.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise RuntimeError("catalog has no experiment_id")
    trajectories = payload.get("trajectories")
    if not isinstance(trajectories, list):
        raise RuntimeError("catalog has no trajectory list")
    root = catalog.parent
    paths: set[Path] = {catalog}
    for entry in trajectories:
        if not isinstance(entry, Mapping):
            raise RuntimeError("catalog trajectory entry must be an object")
        artifacts = entry.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise RuntimeError("catalog trajectory has no artifact mapping")
        hashes = artifacts.get("sha256")
        if not isinstance(hashes, Mapping):
            raise RuntimeError("catalog trajectory has no artifact hash mapping")
        for name in ("resolved_config", "result", "trace", "video"):
            relative = artifacts.get(name)
            if relative is None:
                continue
            if not isinstance(relative, str) or Path(relative).is_absolute():
                raise RuntimeError("catalog contains an unsafe artifact path")
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise RuntimeError(f"catalog member is missing: {relative}")
            expected = hashes.get(name)
            if (
                not isinstance(expected, str)
                or _SHA256.fullmatch(expected) is None
                or file_sha256(path) != expected
            ):
                raise RuntimeError(f"catalog member SHA-256 mismatch: {relative}")
            if name == "result":
                result = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(result, Mapping):
                    raise RuntimeError("catalog result member must be an object")
                contact_preserving_result = (
                    int(
                        result.get(
                            "contact_preserving_candidate_schema_version",
                            0,
                        )
                    )
                    == 1
                )
                # Synthetic backends used by catalog unit tests predate sealed
                # v14 results and explicitly advertise that no production video
                # was generated.  Keep that narrow compatibility path while
                # requiring every production (or otherwise unlabelled) v14
                # catalog member to carry and authenticate its semantic digest.
                contact_preserving_semantics_required = bool(
                    contact_preserving_result
                    and payload.get("production_trajectory_video_policy")
                    != "disabled_for_injected_test_backend"
                )
                if (
                    int(result.get("candidate_result_schema_version", 0)) == 1
                    or int(
                        result.get(
                            "actual_contact_manipulation_candidate_schema_version",
                            0,
                        )
                    )
                    == 1
                    or contact_preserving_semantics_required
                ):
                    authenticate_candidate_result_semantic_sha256(
                        result, source=path
                    )
            elif name == "resolved_config":
                config = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(config, Mapping):
                    raise RuntimeError("catalog resolved config is malformed")
                definition = resolve_actual_contact_definition(
                    config, context="authenticated actual-contact catalog"
                )
                if definition.experiment_id != experiment_id:
                    raise RuntimeError("catalog member changed experiment")
            paths.add(path)
    return tuple(sorted(paths, key=str))


def export_actual_contact_grasp_pose_catalog(
    candidates: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    selected_count: int = 5,
    simulation_runner: FinalSimulationRunner | None = None,
    video_probe: FinalVideoProbe | None = None,
) -> dict[str, Any]:
    return _publish_catalog(
        candidates,
        output_dir,
        kind="grasp_pose",
        selected_count=selected_count,
        simulation_runner=simulation_runner,
        video_probe=video_probe,
    )


def export_actual_contact_manipulation_catalog(
    candidates: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    selected_count: int = 5,
    simulation_runner: FinalSimulationRunner | None = None,
    video_probe: FinalVideoProbe | None = None,
) -> dict[str, Any]:
    return _publish_catalog(
        candidates,
        output_dir,
        kind="manipulation",
        selected_count=selected_count,
        simulation_runner=simulation_runner,
        video_probe=video_probe,
    )


__all__ = [
    "CAMPAIGN_MANIFEST_SCHEMA_VERSION",
    "EXPERIMENT_ID",
    "RESULT_SEMANTIC_SHA256_FIELD",
    "STAGE_LEDGER_SCHEMA_VERSION",
    "TRAJECTORY_CATALOG_SCHEMA_VERSION",
    "authenticate_candidate_result_semantic_sha256",
    "authenticated_catalog_artifact_paths",
    "bind_candidate_result_semantic_sha256",
    "build_campaign_manifest",
    "candidate_result_semantic_sha256",
    "commit_campaign_stage",
    "export_actual_contact_grasp_pose_catalog",
    "export_actual_contact_manipulation_catalog",
    "initialize_or_resume_campaign",
    "validate_stage_ledger",
]
