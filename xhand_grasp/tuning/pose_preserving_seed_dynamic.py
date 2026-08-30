"""Resumable MuJoCo tuning for the six pose-preserving grasp seeds.

The campaign deliberately keeps every source object's world pose, size, mass
and friction fixed.  Its only search variables are the hand root pose, the
eight pregrasp targets and a small adjustment to the thumb-bend terminal
target.  Every completed simulation is committed atomically, so an interrupted
spawn campaign can reuse verified candidate artifacts on its next invocation.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import multiprocessing
import os
import shutil
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts import file_sha256, json_text, write_json
from ..config import ACTIVE_ACTUATORS, load_config, resolved_pose_constraint_values, validate_config
from ..experiment import resolve_experiment
from ..scene import rpy_degrees_to_rotation_matrix
from ..simulation import run_simulation
from .pose_preserving_grasp import (
    acquisition_succeeded,
    deterministic_rank_pose_preserving_results,
    grasp_succeeded,
    pose_preservation_succeeded,
)
from .pose_preserving_seed_campaign import (
    CAMPAIGN_SCHEMA_VERSION,
    EXPECTED_SOURCE_COUNT,
    RETARGET_METHOD,
    TARGET_EXPERIMENT_ID,
    _registered_v6_canonical_status,
    _seed_hand_relation,
    _target_mapping,
    _v6_template,
    canonical_sha256,
    generate_hand_pose_only_candidates,
    initial_cube_world_pose,
    load_pose_preserving_seed_sources,
)


DYNAMIC_CAMPAIGN_SCHEMA_VERSION = 1
CANDIDATE_RESULT_SCHEMA_VERSION = 1
CAMPAIGN_KIND = "pose_preserving_six_seed_dynamic_acquisition"
THUMB_BEND_ACTUATOR = "left_hand_thumb_bend_joint_actuator"
DEFAULT_HAND_RPY_RADIUS_DEG = (0.50, 0.75, 0.50)
DEFAULT_CUBE_IN_ROOT_RADIUS_M = (0.0010, 0.0010, 0.0010)
DEFAULT_PREGRASP_RADIUS_RAD = 0.035
DEFAULT_THUMB_BEND_RADIUS_RAD = 0.035
DEFAULT_PREGRASP_BACKOFF_FACTOR = 0.50
DEFAULT_CLOSE_GROUP_START_FRACTIONS = (0.20, 0.0, 0.0)
DEFAULT_CLOSE_GROUP_START_RADIUS = (0.08, 0.06, 0.06)
MAX_HAND_RPY_RADIUS_DEG = (2.0, 3.0, 2.0)
MAX_CUBE_IN_ROOT_RADIUS_M = (0.003, 0.003, 0.003)
MAX_PREGRASP_RADIUS_RAD = 0.12
MAX_THUMB_BEND_RADIUS_RAD = 0.10
MAX_CLOSE_GROUP_START_RADIUS = (0.20, 0.20, 0.20)
DEFAULT_COUNT_PER_SOURCE = 32
DEFAULT_SEED = 20260821
DEFAULT_OUTPUT = Path(
    "artifacts/left_opposed_face_palm_down_pose_preserving_grasp/"
    "six_seed_dynamic_tune"
)
DEFAULT_CATALOG = Path(
    "artifacts/"
    "left_opposed_face_palm_tilted_down_far_hand_fingertip_grasp_then_lift/"
    "grasp_acquisition_high_thumb/trajectory_catalog/catalog.json"
)
DEFAULT_TEMPLATE = Path(
    "grasp_configs/left_opposed_face_palm_down_pose_preserving_grasp.json"
)
DEFAULT_EVIDENCE_SEED_MANIFEST = Path(
    "grasp_configs/pose_preserving_six_seed_hand_pose_seeds.json"
)
_CANDIDATE_ID_STRIDE = 1_000_000

_SOURCE_FINGERPRINT_FIELDS = (
    "source_candidate_id",
    "trajectory_id",
    "parameter_override_run",
    "cube",
    "scene",
    "contact_topology",
    "grasp_targets_rad",
    "initial_cube_world_pose",
    "acquisition",
    "terminal_cube_in_root",
    "fixed_object_sha256",
    "terminal_relation_sha256",
    "provenance",
)

CLOSE_GROUP_ORDER = ("thumb", "index", "mid")
CLOSE_GROUP_ACTUATORS = {
    "thumb": (
        "left_hand_thumb_bend_joint_actuator",
        "left_hand_thumb_rota_joint1_actuator",
        "left_hand_thumb_rota_joint2_actuator",
    ),
    "index": (
        "left_hand_index_bend_joint_actuator",
        "left_hand_index_joint1_actuator",
        "left_hand_index_joint2_actuator",
    ),
    "mid": (
        "left_hand_mid_joint1_actuator",
        "left_hand_mid_joint2_actuator",
    ),
}

CandidateExecutor = Callable[
    [Sequence[Mapping[str, Any]], int], Sequence[Mapping[str, Any]]
]


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _finite_vector(values: Sequence[Any], length: int, label: str) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain {length} finite values") from error
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must contain {length} finite values")
    return result.copy()


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _latin_hypercube(
    samples: int,
    dimensions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if samples <= 0 or dimensions <= 0:
        raise ValueError("Latin-hypercube dimensions must be positive")
    values = np.empty((samples, dimensions), dtype=np.float64)
    for dimension in range(dimensions):
        values[:, dimension] = (
            rng.permutation(samples) + rng.random(samples)
        ) / samples
    return values


def acquisition_qpos_backoff_pregrasp(
    source: Mapping[str, Any],
    template: Mapping[str, Any],
    *,
    factor: float = DEFAULT_PREGRASP_BACKOFF_FACTOR,
) -> dict[str, float]:
    """Back away from the measured acquisition qpos along servo error.

    For each active actuator this computes
    ``q_acq - factor * (q_target - q_acq)`` and clips only at the registered
    schema-v6 pregrasp bounds.  A factor of 0.50 is the independently verified
    61/62 mm evidence seed.
    """

    backoff = _finite(factor, "pregrasp backoff factor")
    if not 0.0 <= backoff <= 1.0:
        raise ValueError("pregrasp backoff factor must be within [0, 1]")
    resolved_template = _v6_template(template)
    bounds = resolve_experiment(resolved_template).search_bounds.pregrasp_targets_rad
    if bounds is None:
        raise ValueError("schema-v6 experiment has no pregrasp bounds")
    actual = _target_mapping(
        source["acquisition"]["active_actuator_qpos_rad"],
        "source acquisition qpos",
    )
    target = _target_mapping(
        source["grasp_targets_rad"], "source grasp targets"
    )
    result: dict[str, float] = {}
    for name in ACTIVE_ACTUATORS:
        lower, upper = bounds[name]
        proposed = actual[name] - backoff * (target[name] - actual[name])
        result[name] = float(np.clip(proposed, lower, upper))
    return result


def _source_sha256_bound_to_catalog(
    source: Mapping[str, Any],
    catalog_sha256: str,
) -> str:
    """Recompute a source fingerprint against a specific catalog digest.

    A catalog can be republished with different aliases or report metadata
    while still naming the exact same authenticated config and trace files.
    The source fingerprint includes the catalog digest as provenance, so a
    byte-only catalog rewrite would otherwise invalidate independently sealed
    dynamics seeds.  Rebinding only that one provenance field lets the caller
    prove that every simulation-relevant source field and artifact digest is
    unchanged.
    """

    if not isinstance(catalog_sha256, str) or len(catalog_sha256) != 64:
        raise ValueError("evidence seed manifest catalog hash is invalid")
    rebound = copy.deepcopy(dict(source))
    provenance = rebound.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("source provenance is missing")
    rebound["provenance"] = copy.deepcopy(dict(provenance))
    rebound["provenance"]["catalog_sha256"] = catalog_sha256
    try:
        payload = {
            key: copy.deepcopy(rebound[key]) for key in _SOURCE_FINGERPRINT_FIELDS
        }
    except KeyError as error:
        raise ValueError(
            f"source fingerprint field is missing: {error.args[0]}"
        ) from error
    return canonical_sha256(payload)


def load_evidence_seed_manifest(
    path: str | Path,
    sources: Sequence[Mapping[str, Any]],
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Load and authenticate the optional versioned dynamics evidence seeds."""

    manifest_path = Path(path).expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("evidence seed manifest must be a mapping")
    if payload.get("evidence_seed_manifest_schema_version") != 1:
        raise ValueError("unsupported evidence seed manifest schema")
    if payload.get("campaign_kind") != CAMPAIGN_KIND:
        raise ValueError("evidence seed manifest has the wrong campaign kind")
    if payload.get("target_experiment_id") != TARGET_EXPERIMENT_ID:
        raise ValueError("evidence seed manifest has the wrong experiment")
    source_by_id = {
        int(source["source_candidate_id"]): source for source in sources
    }
    raw_seeds = payload.get("seeds")
    if not isinstance(raw_seeds, Mapping):
        raise ValueError("evidence seed manifest must contain a seeds mapping")
    catalog_hashes = {
        str(source["provenance"]["catalog_sha256"]) for source in sources
    }
    manifest_catalog_sha256 = payload.get("source_catalog_sha256")
    if len(catalog_hashes) != 1:
        raise ValueError("evidence seed manifest catalog hash is stale")
    loaded_catalog_sha256 = next(iter(catalog_hashes))
    exact_catalog_binding = manifest_catalog_sha256 == loaded_catalog_sha256
    if exact_catalog_binding:
        bound_source_hashes = {
            source_id: str(source["source_sha256"])
            for source_id, source in source_by_id.items()
        }
    else:
        if not isinstance(manifest_catalog_sha256, str):
            raise ValueError("evidence seed manifest catalog hash is stale")
        bound_source_hashes = {
            source_id: _source_sha256_bound_to_catalog(
                source, manifest_catalog_sha256
            )
            for source_id, source in source_by_id.items()
        }
    seeds: dict[int, dict[str, Any]] = {}
    for raw_id, raw_seed in raw_seeds.items():
        try:
            source_id = int(raw_id)
        except (TypeError, ValueError) as error:
            raise ValueError("evidence source IDs must be integers") from error
        if source_id not in source_by_id or not isinstance(raw_seed, Mapping):
            raise ValueError("evidence seed references an unknown source")
        seed = copy.deepcopy(dict(raw_seed))
        source = source_by_id[source_id]
        if int(seed.get("source_candidate_id", -1)) != source_id:
            raise ValueError("evidence seed source ID is inconsistent")
        if seed.get("source_sha256") != bound_source_hashes[source_id]:
            message = (
                "evidence seed source hash is stale"
                if exact_catalog_binding
                else "evidence seed manifest catalog hash is stale"
            )
            raise ValueError(message)
        hand_pose = seed.get("hand_pose")
        if not isinstance(hand_pose, Mapping):
            raise ValueError("evidence seed has no hand_pose")
        hand_rpy = _finite_vector(
            hand_pose.get("rpy_deg"), 3, "evidence hand RPY"
        )
        cube_in_root = _finite_vector(
            hand_pose.get("cube_in_root_m"), 3, "evidence cube-in-root"
        )
        verified_translation = _finite_vector(
            hand_pose.get("verified_translation_m"),
            3,
            "evidence verified translation",
        )
        cube_world = _finite_vector(
            source["initial_cube_world_pose"]["position_m"],
            3,
            "source initial cube position",
        )
        expected_translation = cube_world - (
            rpy_degrees_to_rotation_matrix(hand_rpy) @ cube_in_root
        )
        if not np.allclose(
            verified_translation,
            expected_translation,
            rtol=0.0,
            atol=2e-12,
        ):
            raise ValueError("evidence hand translation violates the fixed cube pose")
        _target_mapping(
            seed.get("pregrasp_targets_rad", {}),
            "evidence pregrasp targets",
        )
        overrides = seed.get("grasp_target_overrides_rad", {})
        if not isinstance(overrides, Mapping) or set(overrides) - {
            THUMB_BEND_ACTUATOR
        }:
            raise ValueError("evidence may override only the thumb-bend target")
        if THUMB_BEND_ACTUATOR in overrides:
            _finite(overrides[THUMB_BEND_ACTUATOR], "evidence thumb target")
        profile = seed.get("close_profile")
        if not isinstance(profile, Mapping) or set(profile) != set(ACTIVE_ACTUATORS):
            raise ValueError("evidence seed close_profile is incomplete")
        for group in CLOSE_GROUP_ORDER:
            starts = set()
            for name in CLOSE_GROUP_ACTUATORS[group]:
                interval = profile[name]
                if not isinstance(interval, Mapping) or set(interval) != {
                    "start_fraction",
                    "end_fraction",
                }:
                    raise ValueError("evidence close interval is invalid")
                start = _finite(
                    interval["start_fraction"], "evidence close start"
                )
                end = _finite(interval["end_fraction"], "evidence close end")
                if not 0.0 <= start < end <= 1.0:
                    raise ValueError("evidence close interval is out of range")
                starts.add(start)
            if len(starts) != 1:
                raise ValueError("evidence close starts are not grouped")
        validation = seed.get("validation")
        if not isinstance(validation, Mapping):
            raise ValueError("evidence seed has no validation record")
        for field in (
            "grasp_success",
            "object_pose_preserved",
            "settle_contact_free",
            "support_retained",
        ):
            if validation.get(field) is not True:
                raise ValueError(f"evidence validation field {field} did not pass")
        if _finite(
            validation.get("max_translation_m"),
            "evidence max translation",
        ) > 0.0005 + 1e-12:
            raise ValueError("evidence translation exceeds the v6 limit")
        if _finite(
            validation.get("max_orientation_drift_deg"),
            "evidence max orientation drift",
        ) > 1.0 + 1e-12:
            raise ValueError("evidence orientation exceeds the v6 limit")
        seeds[source_id] = seed
    return seeds, {
        "path": str(manifest_path),
        "file_sha256": file_sha256(manifest_path),
        "semantic_sha256": canonical_sha256(payload),
        "seed_count": len(seeds),
        "source_catalog_binding": {
            "mode": (
                "exact_file_sha256"
                if exact_catalog_binding
                else "source_semantic_rebind"
            ),
            "manifest_sha256": manifest_catalog_sha256,
            "loaded_sha256": loaded_catalog_sha256,
        },
    }


def _base_dynamic_pose(
    source: Mapping[str, Any],
    template: Mapping[str, Any],
    evidence_seed: Mapping[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if evidence_seed is None:
        rpy, local = _seed_hand_relation(source, template)
        return rpy, local, {"kind": "terminal_relation_seed"}
    evidence = evidence_seed
    hand_pose = evidence["hand_pose"]
    return (
        _finite_vector(hand_pose["rpy_deg"], 3, "evidence hand RPY"),
        _finite_vector(
            hand_pose["cube_in_root_m"], 3, "evidence cube-in-root"
        ),
        {
            "kind": "versioned_verified_dynamics_seed",
            "provenance": copy.deepcopy(evidence["provenance"]),
            "validation": copy.deepcopy(evidence["validation"]),
        },
    )


def close_profile_from_group_starts(
    template: Mapping[str, Any],
    starts: Mapping[str, Any],
) -> dict[str, dict[str, float]]:
    """Set one start fraction per thumb/index/middle actuator group."""

    if set(starts) != set(CLOSE_GROUP_ORDER):
        raise ValueError("close group starts must contain thumb, index and mid")
    profile = copy.deepcopy(template["control"]["close_profile"])
    for group in CLOSE_GROUP_ORDER:
        start = _finite(starts[group], f"close group {group} start")
        for name in CLOSE_GROUP_ACTUATORS[group]:
            end = float(profile[name]["end_fraction"])
            if not 0.0 <= start < end:
                raise ValueError(
                    f"close group {group} start must satisfy 0 <= start < {end}"
                )
            profile[name]["start_fraction"] = start
    return profile


def _dynamic_materialization_sha256(
    config: Mapping[str, Any], source: Mapping[str, Any]
) -> str:
    return canonical_sha256(
        {
            "dynamic_campaign_schema_version": DYNAMIC_CAMPAIGN_SCHEMA_VERSION,
            "campaign_kind": CAMPAIGN_KIND,
            "retarget_method": RETARGET_METHOD,
            "source_sha256": source["source_sha256"],
            "fixed_object_sha256": source["fixed_object_sha256"],
            "hand_pose": config["hand_pose"],
            "cube": config["cube"],
            "scene": config["scene"],
            "contact_topology": config["contact_topology"],
            "control": config["control"],
            "run_context": config.get("run_context"),
        }
    )


def assert_dynamic_candidate_invariants(
    config: Mapping[str, Any],
    source: Mapping[str, Any],
    template: Mapping[str, Any],
) -> None:
    """Enforce the campaign's fixed-object and narrow-control search scope."""

    if config["cube"] != source["cube"]:
        raise ValueError("dynamic candidate changed its source cube block")
    if config["scene"] != source["scene"]:
        raise ValueError("dynamic candidate changed its source scene block")
    if config["contact_topology"] != source["contact_topology"]:
        raise ValueError("dynamic candidate changed its source contact topology")
    if initial_cube_world_pose(config) != source["initial_cube_world_pose"]:
        raise ValueError("dynamic candidate changed the configured cube world pose")
    control = config["control"]
    if control["manipulation_delta_rad"] != {
        name: 0.0 for name in ACTIVE_ACTUATORS
    }:
        raise ValueError("dynamic candidate manipulation delta must remain zero")
    template_profile = template["control"]["close_profile"]
    profile = control["close_profile"]
    if set(profile) != set(ACTIVE_ACTUATORS):
        raise ValueError("dynamic candidate close profile is incomplete")
    for group in CLOSE_GROUP_ORDER:
        group_starts = set()
        for name in CLOSE_GROUP_ACTUATORS[group]:
            interval = profile[name]
            if set(interval) != {"start_fraction", "end_fraction"}:
                raise ValueError("dynamic close intervals have invalid fields")
            start = float(interval["start_fraction"])
            end = float(interval["end_fraction"])
            if end != float(template_profile[name]["end_fraction"]):
                raise ValueError("dynamic close end fractions must remain at the template")
            if not 0.0 <= start < end:
                raise ValueError("dynamic close interval is invalid")
            group_starts.add(start)
        if len(group_starts) != 1:
            raise ValueError("dynamic close start fractions must be shared by finger group")
    source_targets = source["grasp_targets_rad"]
    for name in ACTIVE_ACTUATORS:
        if name == THUMB_BEND_ACTUATOR:
            continue
        if control["grasp_targets_rad"][name] != source_targets[name]:
            raise ValueError(
                "only the thumb-bend terminal grasp target may be adjusted"
            )
    metadata = config.get("candidate_metadata", {})
    allowed = {
        "hand_pose.translation_m",
        "hand_pose.rpy_deg",
        "control.pregrasp_targets_rad",
        f"control.grasp_targets_rad.{THUMB_BEND_ACTUATOR}",
        "control.close_profile.group_start_fraction",
    }
    if set(metadata.get("sampled_fields", ())) != allowed:
        raise ValueError("dynamic candidate sampled_fields do not match its scope")
    resolved_local = np.asarray(
        resolved_pose_constraint_values(dict(config))["cube_position_in_root_m"],
        dtype=np.float64,
    )
    recorded_local = _finite_vector(
        metadata["candidate_cube_in_root_m"],
        3,
        "candidate_metadata.candidate_cube_in_root_m",
    )
    if not np.allclose(resolved_local, recorded_local, rtol=0.0, atol=2e-12):
        raise ValueError("dynamic candidate hand pose/local relation is inconsistent")


def materialize_dynamic_acquisition_candidate(
    source: Mapping[str, Any],
    template: Mapping[str, Any],
    *,
    hand_rpy_deg: Sequence[float],
    cube_in_root_m: Sequence[float],
    pregrasp_targets_rad: Mapping[str, Any],
    thumb_bend_target_rad: float,
    close_group_start_fractions: Mapping[str, Any],
    candidate_id: int,
    local_index: int,
    seed: int,
    rpy_delta_deg: Sequence[float],
    cube_in_root_delta_m: Sequence[float],
    pregrasp_delta_rad: Mapping[str, Any],
    thumb_bend_delta_rad: float,
    close_group_start_delta: Mapping[str, Any],
    pose_seed_provenance: Mapping[str, Any],
    pregrasp_seed_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one validated schema-v6 dynamic-acquisition candidate."""

    resolved_template = _v6_template(template)
    base_record = generate_hand_pose_only_candidates(
        source,
        resolved_template,
        count=1,
        seed=seed,
        hand_rpy_radius_deg=(0.0, 0.0, 0.0),
        cube_in_root_radius_m=(0.0, 0.0, 0.0),
    )[0]
    config = copy.deepcopy(base_record["config"])
    rpy = _finite_vector(hand_rpy_deg, 3, "hand_rpy_deg")
    local = _finite_vector(cube_in_root_m, 3, "cube_in_root_m")
    cube_world = _finite_vector(
        source["initial_cube_world_pose"]["position_m"],
        3,
        "source initial cube position",
    )
    root_rotation = rpy_degrees_to_rotation_matrix(rpy)
    config["hand_pose"] = {
        "translation_m": (cube_world - root_rotation @ local).tolist(),
        "rpy_deg": rpy.tolist(),
    }
    pregrasp = _target_mapping(pregrasp_targets_rad, "pregrasp_targets_rad")
    pregrasp_delta = _target_mapping(
        pregrasp_delta_rad, "pregrasp_delta_rad"
    )
    if set(close_group_start_delta) != set(CLOSE_GROUP_ORDER):
        raise ValueError("close_group_start_delta has invalid groups")
    close_delta = {
        group: _finite(
            close_group_start_delta[group],
            f"close group {group} start delta",
        )
        for group in CLOSE_GROUP_ORDER
    }
    rpy_delta = _finite_vector(rpy_delta_deg, 3, "rpy_delta_deg")
    local_delta = _finite_vector(
        cube_in_root_delta_m, 3, "cube_in_root_delta_m"
    )
    thumb_delta = _finite(thumb_bend_delta_rad, "thumb_bend_delta_rad")
    if np.any(np.abs(rpy_delta) > np.asarray(MAX_HAND_RPY_RADIUS_DEG) + 1e-12):
        raise ValueError("hand RPY adjustment exceeds the campaign safety radius")
    if np.any(
        np.abs(local_delta) > np.asarray(MAX_CUBE_IN_ROOT_RADIUS_M) + 1e-12
    ):
        raise ValueError("cube-in-root adjustment exceeds the campaign safety radius")
    if any(
        abs(value) > MAX_PREGRASP_RADIUS_RAD + 1e-12
        for value in pregrasp_delta.values()
    ):
        raise ValueError("pregrasp adjustment exceeds the campaign safety radius")
    if abs(thumb_delta) > MAX_THUMB_BEND_RADIUS_RAD + 1e-12:
        raise ValueError("thumb-bend adjustment exceeds the campaign safety radius")
    if any(
        abs(close_delta[group])
        > MAX_CLOSE_GROUP_START_RADIUS[index] + 1e-12
        for index, group in enumerate(CLOSE_GROUP_ORDER)
    ):
        raise ValueError("close-group adjustment exceeds the campaign safety radius")
    grasp_targets = copy.deepcopy(source["grasp_targets_rad"])
    grasp_targets[THUMB_BEND_ACTUATOR] = _finite(
        thumb_bend_target_rad, "thumb_bend_target_rad"
    )
    close_profile = close_profile_from_group_starts(
        resolved_template, close_group_start_fractions
    )
    config["control"] = {
        "pregrasp_targets_rad": pregrasp,
        "grasp_targets_rad": grasp_targets,
        "manipulation_delta_rad": {
            name: 0.0 for name in ACTIVE_ACTUATORS
        },
        "close_profile": close_profile,
    }
    canonical = _registered_v6_canonical_status(config)
    config.pop("run_context", None)
    if bool(source["parameter_override_run"]) or not canonical["all"]:
        config["run_context"] = {"kind": "parameter_override_run"}
    metadata = copy.deepcopy(config.get("candidate_metadata", {}))
    metadata.update(
        {
            "campaign_kind": CAMPAIGN_KIND,
            "candidate_id": int(candidate_id),
            "local_index": int(local_index),
            "seed": None if local_index == 0 else int(seed),
            "candidate_cube_in_root_m": local.tolist(),
            "hand_rpy_delta_deg": rpy_delta.tolist(),
            "cube_in_root_delta_m": local_delta.tolist(),
            "pregrasp_delta_rad": pregrasp_delta,
            "thumb_bend_delta_rad": thumb_delta,
            "close_group_start_fractions": {
                group: float(close_group_start_fractions[group])
                for group in CLOSE_GROUP_ORDER
            },
            "close_group_start_delta": close_delta,
            "pose_seed_provenance": copy.deepcopy(dict(pose_seed_provenance)),
            "pregrasp_seed_provenance": copy.deepcopy(
                dict(pregrasp_seed_provenance)
            ),
            "canonical_v6_material": canonical["material"],
            "canonical_v6_pose_envelope": canonical["pose_envelope"],
            "parameter_override_run": "run_context" in config,
            "sampled_fields": [
                "hand_pose.translation_m",
                "hand_pose.rpy_deg",
                "control.pregrasp_targets_rad",
                f"control.grasp_targets_rad.{THUMB_BEND_ACTUATOR}",
                "control.close_profile.group_start_fraction",
            ],
        }
    )
    config["candidate_metadata"] = metadata
    metadata["materialization_sha256"] = _dynamic_materialization_sha256(
        config, source
    )
    config["candidate_metadata"] = metadata
    assert_dynamic_candidate_invariants(config, source, resolved_template)
    validate_config(config)
    return config


def generate_dynamic_acquisition_candidates(
    source: Mapping[str, Any],
    template: Mapping[str, Any],
    *,
    count: int,
    seed: int,
    hand_rpy_radius_deg: Sequence[float] = DEFAULT_HAND_RPY_RADIUS_DEG,
    cube_in_root_radius_m: Sequence[float] = DEFAULT_CUBE_IN_ROOT_RADIUS_M,
    pregrasp_radius_rad: float = DEFAULT_PREGRASP_RADIUS_RAD,
    thumb_bend_radius_rad: float = DEFAULT_THUMB_BEND_RADIUS_RAD,
    pregrasp_backoff_factor: float = DEFAULT_PREGRASP_BACKOFF_FACTOR,
    close_group_start_radius: Sequence[float] = DEFAULT_CLOSE_GROUP_START_RADIUS,
    evidence_seed: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Generate stable-order local candidates, with the exact seed first."""

    candidate_count = _positive_int(count, "count")
    if candidate_count >= _CANDIDATE_ID_STRIDE:
        raise ValueError("count is too large for stable candidate IDs")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rpy_radius = _finite_vector(
        hand_rpy_radius_deg, 3, "hand_rpy_radius_deg"
    )
    local_radius = _finite_vector(
        cube_in_root_radius_m, 3, "cube_in_root_radius_m"
    )
    pregrasp_radius = _finite(pregrasp_radius_rad, "pregrasp_radius_rad")
    thumb_radius = _finite(thumb_bend_radius_rad, "thumb_bend_radius_rad")
    close_radius = _finite_vector(
        close_group_start_radius, 3, "close_group_start_radius"
    )
    backoff_factor = _finite(
        pregrasp_backoff_factor, "pregrasp_backoff_factor"
    )
    if (
        np.any(rpy_radius < 0.0)
        or np.any(local_radius < 0.0)
        or pregrasp_radius < 0.0
        or thumb_radius < 0.0
        or np.any(close_radius < 0.0)
    ):
        raise ValueError("dynamic search radii must be non-negative")
    if not 0.0 <= backoff_factor <= 1.0:
        raise ValueError("pregrasp_backoff_factor must be within [0, 1]")
    if np.any(rpy_radius > np.asarray(MAX_HAND_RPY_RADIUS_DEG) + 1e-12):
        raise ValueError("hand_rpy_radius_deg exceeds the campaign safety radius")
    if np.any(
        local_radius > np.asarray(MAX_CUBE_IN_ROOT_RADIUS_M) + 1e-12
    ):
        raise ValueError("cube_in_root_radius_m exceeds the campaign safety radius")
    if pregrasp_radius > MAX_PREGRASP_RADIUS_RAD + 1e-12:
        raise ValueError("pregrasp_radius_rad exceeds the campaign safety radius")
    if thumb_radius > MAX_THUMB_BEND_RADIUS_RAD + 1e-12:
        raise ValueError("thumb_bend_radius_rad exceeds the campaign safety radius")
    if np.any(
        close_radius > np.asarray(MAX_CLOSE_GROUP_START_RADIUS) + 1e-12
    ):
        raise ValueError("close_group_start_radius exceeds the campaign safety radius")
    resolved_template = _v6_template(template)
    definition = resolve_experiment(resolved_template)
    pregrasp_bounds = definition.search_bounds.pregrasp_targets_rad
    if pregrasp_bounds is None:
        raise ValueError("schema-v6 experiment has no pregrasp bounds")
    thumb_bounds = definition.search_bounds.actuator_targets_rad[
        THUMB_BEND_ACTUATOR
    ]
    base_rpy, base_local, pose_seed_provenance = _base_dynamic_pose(
        source, resolved_template, evidence_seed
    )
    if evidence_seed is None:
        base_pregrasp = acquisition_qpos_backoff_pregrasp(
            source, resolved_template, factor=backoff_factor
        )
        pregrasp_seed_provenance = {
            "kind": "acquisition_qpos_servo_error_backoff",
            "factor": backoff_factor,
        }
        base_close_starts = {
            group: float(DEFAULT_CLOSE_GROUP_START_FRACTIONS[index])
            for index, group in enumerate(CLOSE_GROUP_ORDER)
        }
        base_thumb = float(source["grasp_targets_rad"][THUMB_BEND_ACTUATOR])
    else:
        base_pregrasp = _target_mapping(
            evidence_seed["pregrasp_targets_rad"],
            "evidence pregrasp targets",
        )
        evidence_profile = evidence_seed["close_profile"]
        base_close_starts = {}
        for group in CLOSE_GROUP_ORDER:
            values = {
                float(evidence_profile[name]["start_fraction"])
                for name in CLOSE_GROUP_ACTUATORS[group]
            }
            if len(values) != 1:
                raise ValueError("evidence close starts are not grouped")
            base_close_starts[group] = values.pop()
            for name in CLOSE_GROUP_ACTUATORS[group]:
                if float(evidence_profile[name]["end_fraction"]) != float(
                    resolved_template["control"]["close_profile"][name][
                        "end_fraction"
                    ]
                ):
                    raise ValueError("evidence close end differs from template")
        base_thumb = float(
            evidence_seed.get("grasp_target_overrides_rad", {}).get(
                THUMB_BEND_ACTUATOR,
                source["grasp_targets_rad"][THUMB_BEND_ACTUATOR],
            )
        )
        pregrasp_seed_provenance = copy.deepcopy(
            evidence_seed.get(
                "pregrasp_strategy",
                {
                    "kind": "versioned_evidence_values",
                    "factor": evidence_seed.get("pregrasp_backoff_factor"),
                },
            )
        )
    offsets = np.zeros((candidate_count, 18), dtype=np.float64)
    if candidate_count > 1:
        random = np.random.default_rng(
            np.random.SeedSequence(
                [seed, int(source["source_candidate_id"]), 6_600_001]
            )
        )
        offsets[1:] = 2.0 * _latin_hypercube(
            candidate_count - 1, 18, random
        ) - 1.0
    generated: list[dict[str, Any]] = []
    source_candidate_id = int(source["source_candidate_id"])
    for local_index, unit in enumerate(offsets):
        rpy_delta = unit[:3] * rpy_radius
        local_delta = unit[3:6] * local_radius
        pregrasp_delta: dict[str, float] = {}
        pregrasp: dict[str, float] = {}
        for actuator_index, name in enumerate(ACTIVE_ACTUATORS):
            proposed_delta = float(unit[6 + actuator_index] * pregrasp_radius)
            lower, upper = pregrasp_bounds[name]
            value = float(
                np.clip(base_pregrasp[name] + proposed_delta, lower, upper)
            )
            pregrasp[name] = value
            pregrasp_delta[name] = value - base_pregrasp[name]
        proposed_thumb_delta = float(unit[14] * thumb_radius)
        thumb = float(
            np.clip(base_thumb + proposed_thumb_delta, *thumb_bounds)
        )
        thumb_delta = thumb - base_thumb
        close_starts: dict[str, float] = {}
        close_delta: dict[str, float] = {}
        for group_index, group in enumerate(CLOSE_GROUP_ORDER):
            base_start = base_close_starts[group]
            template_group_end = min(
                float(
                    resolved_template["control"]["close_profile"][name][
                        "end_fraction"
                    ]
                )
                for name in CLOSE_GROUP_ACTUATORS[group]
            )
            proposed = base_start + float(
                unit[15 + group_index] * close_radius[group_index]
            )
            start = float(np.clip(proposed, 0.0, template_group_end - 0.02))
            close_starts[group] = start
            close_delta[group] = start - base_start
        candidate_id = source_candidate_id * _CANDIDATE_ID_STRIDE + local_index
        config = materialize_dynamic_acquisition_candidate(
            source,
            resolved_template,
            hand_rpy_deg=base_rpy + rpy_delta,
            cube_in_root_m=base_local + local_delta,
            pregrasp_targets_rad=pregrasp,
            thumb_bend_target_rad=thumb,
            close_group_start_fractions=close_starts,
            candidate_id=candidate_id,
            local_index=local_index,
            seed=seed,
            rpy_delta_deg=rpy_delta,
            cube_in_root_delta_m=local_delta,
            pregrasp_delta_rad=pregrasp_delta,
            thumb_bend_delta_rad=thumb_delta,
            close_group_start_delta=close_delta,
            pose_seed_provenance=pose_seed_provenance,
            pregrasp_seed_provenance=pregrasp_seed_provenance,
        )
        generated.append(
            {
                "candidate_id": candidate_id,
                "source_order": int(source["source_order"]),
                "source_candidate_id": source_candidate_id,
                "source_trajectory_id": str(source["trajectory_id"]),
                "local_index": local_index,
                "config": config,
                "candidate_sha256": canonical_sha256(config),
            }
        )
    generated.sort(key=lambda item: int(item["candidate_id"]))
    return tuple(generated)


def _candidate_classification(result: Mapping[str, Any]) -> str:
    if acquisition_succeeded(result):
        return "pose_preserving_grasp_acquired"
    if grasp_succeeded(result):
        return "grasp_acquired_pose_preservation_failed"
    if pose_preservation_succeeded(result):
        return "pose_preserved_grasp_not_acquired"
    return "pose_preserving_grasp_near_miss"


def _candidate_result_payload(
    job: Mapping[str, Any],
    summary: Mapping[str, Any],
    *,
    config_file_sha256: str,
    trace_file_sha256: str,
) -> dict[str, Any]:
    ranking_record = {
        "candidate_id": int(job["candidate_id"]),
        "config": job["config"],
        "summary": summary,
    }
    return {
        "candidate_result_schema_version": CANDIDATE_RESULT_SCHEMA_VERSION,
        "complete": True,
        "campaign_kind": CAMPAIGN_KIND,
        "source_order": int(job["source_order"]),
        "source_candidate_id": int(job["source_candidate_id"]),
        "source_trajectory_id": str(job["source_trajectory_id"]),
        "candidate_id": int(job["candidate_id"]),
        "local_index": int(job["local_index"]),
        "candidate_sha256": str(job["candidate_sha256"]),
        "classification": _candidate_classification(ranking_record),
        "grasp_success": grasp_succeeded(ranking_record),
        "pose_preservation_success": pose_preservation_succeeded(ranking_record),
        "acquisition_success": acquisition_succeeded(ranking_record),
        "summary": copy.deepcopy(dict(summary)),
        "artifacts": {
            "resolved_config": "resolved_config.json",
            "trace": "trace.npz",
            "sha256": {
                "resolved_config": config_file_sha256,
                "trace": trace_file_sha256,
            },
        },
    }


def execute_dynamic_candidate_job(job: Mapping[str, Any]) -> dict[str, Any]:
    """Run and atomically commit one complete candidate artifact directory."""

    output = Path(str(job["output_directory"]))
    if output.exists():
        raise FileExistsError(f"candidate output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output.parent, prefix=f".{output.name}.staging."
    ) as staging_name:
        staging = Path(staging_name)
        config_path = staging / "resolved_config.json"
        trace_path = staging / "trace.npz"
        write_json(config_path, job["config"])
        summary = run_simulation(
            copy.deepcopy(dict(job["config"])), trace_path=trace_path
        )
        if not trace_path.is_file():
            raise RuntimeError("run_simulation did not create the requested trace")
        payload = _candidate_result_payload(
            job,
            summary,
            config_file_sha256=file_sha256(config_path),
            trace_file_sha256=file_sha256(trace_path),
        )
        write_json(staging / "result.json", payload)
        staging.rename(output)
    return {
        **copy.deepcopy(payload),
        "config": copy.deepcopy(dict(job["config"])),
        "artifact_directory": str(job["artifact_directory"]),
        "reused": False,
    }


def run_dynamic_candidate_jobs(
    jobs: Sequence[Mapping[str, Any]], workers: int
) -> tuple[dict[str, Any], ...]:
    """Execute jobs sequentially or with a spawn-based process pool."""

    worker_count = _positive_int(workers, "workers")
    materialized = tuple(copy.deepcopy(dict(job)) for job in jobs)
    if not materialized:
        return ()
    if worker_count == 1:
        results = [execute_dynamic_candidate_job(job) for job in materialized]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=worker_count, mp_context=context
        ) as executor:
            results = list(
                executor.map(
                    execute_dynamic_candidate_job,
                    materialized,
                    chunksize=1,
                )
            )
    results.sort(key=lambda item: int(item["candidate_id"]))
    return tuple(results)


def _load_reusable_candidate(job: Mapping[str, Any]) -> dict[str, Any] | None:
    directory = Path(str(job["output_directory"]))
    if not directory.exists():
        return None
    result_path = directory / "result.json"
    config_path = directory / "resolved_config.json"
    trace_path = directory / "trace.npz"
    if not all(path.is_file() for path in (result_path, config_path, trace_path)):
        raise RuntimeError(f"incomplete candidate artifact directory: {directory}")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("complete") is not True:
        raise RuntimeError(f"candidate result is not complete: {result_path}")
    if int(payload.get("candidate_id", -1)) != int(job["candidate_id"]):
        raise RuntimeError(f"candidate ID mismatch in reusable result: {result_path}")
    if payload.get("candidate_sha256") != job["candidate_sha256"]:
        raise RuntimeError(f"candidate config digest mismatch: {result_path}")
    persisted_config = load_config(config_path)
    if canonical_sha256(persisted_config) != job["candidate_sha256"]:
        raise RuntimeError(f"persisted candidate config changed: {config_path}")
    artifacts = payload.get("artifacts", {})
    hashes = artifacts.get("sha256", {}) if isinstance(artifacts, Mapping) else {}
    if hashes.get("resolved_config") != file_sha256(config_path):
        raise RuntimeError(f"persisted config file hash mismatch: {config_path}")
    if hashes.get("trace") != file_sha256(trace_path):
        raise RuntimeError(f"persisted trace file hash mismatch: {trace_path}")
    try:
        with np.load(trace_path, allow_pickle=False) as archive:
            if not archive.files:
                raise RuntimeError(f"persisted trace is empty: {trace_path}")
    except (OSError, ValueError) as error:
        raise RuntimeError(f"persisted trace is unreadable: {trace_path}") from error
    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        raise RuntimeError(f"persisted candidate has no summary: {result_path}")
    return {
        **copy.deepcopy(dict(payload)),
        "config": copy.deepcopy(dict(job["config"])),
        "artifact_directory": str(job["artifact_directory"]),
        "reused": True,
    }


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _public_candidate_record(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in result.items()
        if key not in {"config"}
    }


def _write_source_outputs(
    output_root: Path,
    source: Mapping[str, Any],
    ranked: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    source_name = (
        f"source_{int(source['source_order']):02d}_"
        f"{int(source['source_candidate_id'])}"
    )
    source_dir = output_root / source_name
    best = ranked[0]
    candidate_dir = output_root / str(best["artifact_directory"])
    best_config_path = source_dir / "best_config.json"
    best_trace_path = source_dir / "best_trace.npz"
    best_result_path = source_dir / "best_result.json"
    _atomic_copy(candidate_dir / "resolved_config.json", best_config_path)
    _atomic_copy(candidate_dir / "trace.npz", best_trace_path)
    best_payload = _public_candidate_record(best)
    best_payload["selected_as_source_best"] = True
    best_payload["artifacts"] = {
        "resolved_config": best_config_path.name,
        "trace": best_trace_path.name,
        "sha256": {
            "resolved_config": file_sha256(best_config_path),
            "trace": file_sha256(best_trace_path),
        },
    }
    write_json(best_result_path, best_payload)
    candidate_records = [_public_candidate_record(result) for result in ranked]
    source_payload = {
        "dynamic_campaign_schema_version": DYNAMIC_CAMPAIGN_SCHEMA_VERSION,
        "campaign_kind": CAMPAIGN_KIND,
        "source_order": int(source["source_order"]),
        "source_candidate_id": int(source["source_candidate_id"]),
        "source_trajectory_id": str(source["trajectory_id"]),
        "source_sha256": str(source["source_sha256"]),
        "fixed_object_sha256": str(source["fixed_object_sha256"]),
        "configured_cube_pose": copy.deepcopy(source["initial_cube_world_pose"]),
        "cube": copy.deepcopy(source["cube"]),
        "candidate_count": len(ranked),
        "grasp_success_count": sum(grasp_succeeded(result) for result in ranked),
        "pose_preservation_success_count": sum(
            pose_preservation_succeeded(result) for result in ranked
        ),
        "acquisition_success_count": sum(
            acquisition_succeeded(result) for result in ranked
        ),
        "best_candidate_id": int(best["candidate_id"]),
        "best_classification": _candidate_classification(best),
        "best_grasp_success": grasp_succeeded(best),
        "best_pose_preservation_success": pose_preservation_succeeded(best),
        "best_acquisition_success": acquisition_succeeded(best),
        "best_artifacts": {
            "resolved_config": best_config_path.name,
            "trace": best_trace_path.name,
            "result": best_result_path.name,
            "sha256": {
                "resolved_config": file_sha256(best_config_path),
                "trace": file_sha256(best_trace_path),
                "result": file_sha256(best_result_path),
            },
        },
        "ranked_candidates": candidate_records,
    }
    write_json(source_dir / "source_result.json", source_payload)
    return source_payload


def _campaign_input_payload(
    *,
    sources: Sequence[Mapping[str, Any]],
    template: Mapping[str, Any],
    catalog_sha256: str,
    template_file_sha256: str,
    candidate_records: Sequence[Mapping[str, Any]],
    count_per_source: int,
    seed: int,
    hand_rpy_radius_deg: Sequence[float],
    cube_in_root_radius_m: Sequence[float],
    pregrasp_radius_rad: float,
    thumb_bend_radius_rad: float,
    pregrasp_backoff_factor: float,
    close_group_start_radius: Sequence[float],
    evidence_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "dynamic_campaign_schema_version": DYNAMIC_CAMPAIGN_SCHEMA_VERSION,
        "seed_campaign_schema_version": CAMPAIGN_SCHEMA_VERSION,
        "campaign_kind": CAMPAIGN_KIND,
        "target_experiment_id": TARGET_EXPERIMENT_ID,
        "catalog_sha256": catalog_sha256,
        "template_file_sha256": template_file_sha256,
        "template_semantic_sha256": canonical_sha256(template),
        "source_sha256": [str(source["source_sha256"]) for source in sources],
        "source_candidate_ids": [
            int(source["source_candidate_id"]) for source in sources
        ],
        "count_per_source": int(count_per_source),
        "seed": int(seed),
        "hand_rpy_radius_deg": list(hand_rpy_radius_deg),
        "cube_in_root_radius_m": list(cube_in_root_radius_m),
        "pregrasp_radius_rad": float(pregrasp_radius_rad),
        "thumb_bend_radius_rad": float(thumb_bend_radius_rad),
        "pregrasp_backoff_factor": float(pregrasp_backoff_factor),
        "close_group_start_radius": list(close_group_start_radius),
        "evidence_seed_manifest_file_sha256": evidence_manifest[
            "file_sha256"
        ],
        "evidence_seed_manifest_semantic_sha256": evidence_manifest[
            "semantic_sha256"
        ],
        "evidence_seed_count": int(evidence_manifest["seed_count"]),
        "candidates": [
            {
                "candidate_id": int(candidate["candidate_id"]),
                "source_order": int(candidate["source_order"]),
                "candidate_sha256": str(candidate["candidate_sha256"]),
            }
            for candidate in candidate_records
        ],
    }


def run_pose_preserving_seed_dynamic_campaign(
    catalog_path: str | Path,
    template_path: str | Path,
    output_dir: str | Path,
    *,
    evidence_seed_manifest_path: str | Path = DEFAULT_EVIDENCE_SEED_MANIFEST,
    count_per_source: int = DEFAULT_COUNT_PER_SOURCE,
    workers: int = 1,
    seed: int = DEFAULT_SEED,
    hand_rpy_radius_deg: Sequence[float] = DEFAULT_HAND_RPY_RADIUS_DEG,
    cube_in_root_radius_m: Sequence[float] = DEFAULT_CUBE_IN_ROOT_RADIUS_M,
    pregrasp_radius_rad: float = DEFAULT_PREGRASP_RADIUS_RAD,
    thumb_bend_radius_rad: float = DEFAULT_THUMB_BEND_RADIUS_RAD,
    pregrasp_backoff_factor: float = DEFAULT_PREGRASP_BACKOFF_FACTOR,
    close_group_start_radius: Sequence[float] = DEFAULT_CLOSE_GROUP_START_RADIUS,
    resume: bool = False,
    executor: CandidateExecutor = run_dynamic_candidate_jobs,
) -> dict[str, Any]:
    """Run or resume all six source-local MuJoCo acquisition searches."""

    candidate_count = _positive_int(count_per_source, "count_per_source")
    worker_count = _positive_int(workers, "workers")
    catalog = Path(catalog_path).expanduser().resolve()
    template_file = Path(template_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    sources = load_pose_preserving_seed_sources(catalog)
    if len(sources) != EXPECTED_SOURCE_COUNT:
        raise ValueError("dynamic campaign requires all six source trajectories")
    template = _v6_template(load_config(template_file))
    evidence_seeds, evidence_manifest = load_evidence_seed_manifest(
        evidence_seed_manifest_path, sources
    )
    candidates: list[dict[str, Any]] = []
    for source in sources:
        candidates.extend(
            generate_dynamic_acquisition_candidates(
                source,
                template,
                count=candidate_count,
                seed=seed,
                hand_rpy_radius_deg=hand_rpy_radius_deg,
                cube_in_root_radius_m=cube_in_root_radius_m,
                pregrasp_radius_rad=pregrasp_radius_rad,
                thumb_bend_radius_rad=thumb_bend_radius_rad,
                pregrasp_backoff_factor=pregrasp_backoff_factor,
                close_group_start_radius=close_group_start_radius,
                evidence_seed=evidence_seeds.get(
                    int(source["source_candidate_id"])
                ),
            )
        )
    candidates.sort(
        key=lambda item: (int(item["source_order"]), int(item["candidate_id"]))
    )
    input_payload = _campaign_input_payload(
        sources=sources,
        template=template,
        catalog_sha256=file_sha256(catalog),
        template_file_sha256=file_sha256(template_file),
        candidate_records=candidates,
        count_per_source=candidate_count,
        seed=seed,
        hand_rpy_radius_deg=hand_rpy_radius_deg,
        cube_in_root_radius_m=cube_in_root_radius_m,
        pregrasp_radius_rad=pregrasp_radius_rad,
        thumb_bend_radius_rad=thumb_bend_radius_rad,
        pregrasp_backoff_factor=pregrasp_backoff_factor,
        close_group_start_radius=close_group_start_radius,
        evidence_manifest=evidence_manifest,
    )
    input_sha256 = canonical_sha256(input_payload)
    manifest = {
        **input_payload,
        "campaign_input_sha256": input_sha256,
        "catalog_path": str(catalog),
        "template_path": str(template_file),
        "evidence_seed_manifest_path": evidence_manifest["path"],
        "output_directory": str(output),
    }
    manifest_path = output / "campaign_manifest.json"
    if output.exists():
        if not resume:
            raise FileExistsError(
                f"output directory already exists: {output}; pass --resume to reuse it"
            )
        if not manifest_path.is_file():
            raise RuntimeError("resume output has no campaign_manifest.json")
        persisted_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if persisted_manifest.get("campaign_input_sha256") != input_sha256:
            raise RuntimeError("resume campaign inputs do not match the existing manifest")
    else:
        output.mkdir(parents=True)
        write_json(manifest_path, manifest)

    jobs: list[dict[str, Any]] = []
    for candidate in candidates:
        source_name = (
            f"source_{int(candidate['source_order']):02d}_"
            f"{int(candidate['source_candidate_id'])}"
        )
        relative = (
            Path(source_name)
            / "candidates"
            / f"candidate_{int(candidate['candidate_id'])}"
        )
        jobs.append(
            {
                **copy.deepcopy(candidate),
                "artifact_directory": str(relative),
                "output_directory": str(output / relative),
            }
        )

    results: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for job in jobs:
        reusable = _load_reusable_candidate(job) if resume else None
        if reusable is None:
            pending.append(job)
        else:
            results.append(reusable)
    executed = tuple(executor(tuple(pending), worker_count)) if pending else ()
    submitted = {int(job["candidate_id"]): job for job in pending}
    received = [int(result.get("candidate_id", -1)) for result in executed]
    if len(received) != len(submitted) or set(received) != set(submitted):
        raise RuntimeError("dynamic candidate executor did not preserve candidate IDs")
    if len(received) != len(set(received)):
        raise RuntimeError("dynamic candidate executor returned duplicate candidate IDs")
    for result_value in executed:
        result = copy.deepcopy(dict(result_value))
        job = submitted[int(result["candidate_id"])]
        if result.get("candidate_sha256") != job["candidate_sha256"]:
            raise RuntimeError("dynamic candidate executor rebound candidate config")
        if result.get("config") != job["config"]:
            raise RuntimeError("dynamic candidate executor returned a different config")
        results.append(result)
    results.sort(
        key=lambda item: (int(item["source_order"]), int(item["candidate_id"]))
    )
    if len(results) != len(jobs):
        raise RuntimeError("dynamic campaign did not obtain every candidate result")

    source_results: list[dict[str, Any]] = []
    for source in sources:
        source_group = [
            result
            for result in results
            if int(result["source_order"]) == int(source["source_order"])
        ]
        ranked = deterministic_rank_pose_preserving_results(source_group)
        if len(ranked) != candidate_count:
            raise RuntimeError("dynamic campaign lost a source candidate result")
        source_results.append(_write_source_outputs(output, source, ranked))

    successful_sources = sum(
        bool(result["best_acquisition_success"]) for result in source_results
    )
    campaign_result = {
        "dynamic_campaign_schema_version": DYNAMIC_CAMPAIGN_SCHEMA_VERSION,
        "campaign_kind": CAMPAIGN_KIND,
        "campaign_input_sha256": input_sha256,
        "seed": int(seed),
        "workers": worker_count,
        "source_count": len(source_results),
        "candidate_count": len(results),
        "executed_candidate_count": len(executed),
        "reused_candidate_count": sum(bool(result["reused"]) for result in results),
        "successful_source_count": successful_sources,
        "all_sources_acquired": successful_sources == EXPECTED_SOURCE_COUNT,
        "source_results": [
            {
                "source_order": result["source_order"],
                "source_candidate_id": result["source_candidate_id"],
                "best_candidate_id": result["best_candidate_id"],
                "best_grasp_success": result["best_grasp_success"],
                "best_pose_preservation_success": result[
                    "best_pose_preservation_success"
                ],
                "best_acquisition_success": result["best_acquisition_success"],
                "result": (
                    f"source_{int(result['source_order']):02d}_"
                    f"{int(result['source_candidate_id'])}/source_result.json"
                ),
            }
            for result in source_results
        ],
    }
    write_json(output / "campaign_results.json", campaign_result)
    return campaign_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the six-source schema-v6 hand-pose/pregrasp/thumb-bend "
            "MuJoCo acquisition search."
        )
    )
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    parser.add_argument(
        "--evidence-seed-manifest",
        default=str(DEFAULT_EVIDENCE_SEED_MANIFEST),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--count-per-source", type=int, default=DEFAULT_COUNT_PER_SOURCE)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--hand-rpy-radius-deg",
        type=float,
        nargs=3,
        metavar=("ROLL", "PITCH", "YAW"),
        default=DEFAULT_HAND_RPY_RADIUS_DEG,
    )
    parser.add_argument(
        "--cube-in-root-radius-mm",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=tuple(value * 1000.0 for value in DEFAULT_CUBE_IN_ROOT_RADIUS_M),
    )
    parser.add_argument(
        "--pregrasp-radius-rad", type=float, default=DEFAULT_PREGRASP_RADIUS_RAD
    )
    parser.add_argument(
        "--thumb-bend-radius-rad",
        type=float,
        default=DEFAULT_THUMB_BEND_RADIUS_RAD,
    )
    parser.add_argument(
        "--pregrasp-backoff-factor",
        type=float,
        default=DEFAULT_PREGRASP_BACKOFF_FACTOR,
    )
    parser.add_argument(
        "--close-group-start-radius",
        type=float,
        nargs=3,
        metavar=("THUMB", "INDEX", "MID"),
        default=DEFAULT_CLOSE_GROUP_START_RADIUS,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse complete, hash-verified candidate artifacts in output-dir",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_pose_preserving_seed_dynamic_campaign(
        args.catalog,
        args.template,
        args.output_dir,
        evidence_seed_manifest_path=args.evidence_seed_manifest,
        count_per_source=args.count_per_source,
        workers=args.workers,
        seed=args.seed,
        hand_rpy_radius_deg=args.hand_rpy_radius_deg,
        cube_in_root_radius_m=tuple(
            float(value) / 1000.0 for value in args.cube_in_root_radius_mm
        ),
        pregrasp_radius_rad=args.pregrasp_radius_rad,
        thumb_bend_radius_rad=args.thumb_bend_radius_rad,
        pregrasp_backoff_factor=args.pregrasp_backoff_factor,
        close_group_start_radius=args.close_group_start_radius,
        resume=args.resume,
    )
    print(json_text(result))
    return 0 if result["all_sources_acquired"] else 2


__all__ = [
    "CAMPAIGN_KIND",
    "DEFAULT_CATALOG",
    "DEFAULT_COUNT_PER_SOURCE",
    "DEFAULT_CUBE_IN_ROOT_RADIUS_M",
    "DEFAULT_HAND_RPY_RADIUS_DEG",
    "DEFAULT_EVIDENCE_SEED_MANIFEST",
    "DEFAULT_CLOSE_GROUP_START_FRACTIONS",
    "DEFAULT_CLOSE_GROUP_START_RADIUS",
    "DEFAULT_OUTPUT",
    "DEFAULT_PREGRASP_RADIUS_RAD",
    "DEFAULT_PREGRASP_BACKOFF_FACTOR",
    "DEFAULT_SEED",
    "DEFAULT_TEMPLATE",
    "DEFAULT_THUMB_BEND_RADIUS_RAD",
    "DYNAMIC_CAMPAIGN_SCHEMA_VERSION",
    "MAX_CUBE_IN_ROOT_RADIUS_M",
    "MAX_CLOSE_GROUP_START_RADIUS",
    "MAX_HAND_RPY_RADIUS_DEG",
    "MAX_PREGRASP_RADIUS_RAD",
    "MAX_THUMB_BEND_RADIUS_RAD",
    "THUMB_BEND_ACTUATOR",
    "assert_dynamic_candidate_invariants",
    "acquisition_qpos_backoff_pregrasp",
    "build_parser",
    "execute_dynamic_candidate_job",
    "generate_dynamic_acquisition_candidates",
    "load_evidence_seed_manifest",
    "main",
    "materialize_dynamic_acquisition_candidate",
    "run_dynamic_candidate_jobs",
    "run_pose_preserving_seed_dynamic_campaign",
]
