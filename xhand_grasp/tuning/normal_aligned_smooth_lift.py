"""Schema-v8 control rescue and response-guided smooth vertical lifting.

This module implements the first, independently resumable stage of the v8
campaign.  It authenticates the useful dynamic poses from the completed v7
campaign, keeps every hand/object geometric field fixed, and searches only the
closure controller.  New static-pose discovery and manipulation-Jacobian
search intentionally lives outside this module.  The manipulation stage is
implemented here because it must consume the *resolved* rescue controller,
probe the real MuJoCo response, and keep pose/controller identity auditable.

The public helpers are deliberately data-oriented.  ``pose_id`` hashes only
``cube``, ``hand_pose`` and ``contact_topology`` while ``controller_id`` hashes
only the resolved control law and close duration.  A failed controller can
therefore never mark a pose itself as failed.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import multiprocessing
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts import file_sha256, json_text, write_json
from ..config import ACTIVE_ACTUATORS, load_config, validate_config
from .pose_preserving_grasp import (
    acquisition_succeeded,
    pose_preservation_succeeded,
)
from .pose_preserving_seed_campaign import canonical_sha256
from .pose_preserving_seed_dynamic import (
    CLOSE_GROUP_ACTUATORS,
    CLOSE_GROUP_ORDER,
)


V7_EXPERIMENT_ID = (
    "left_opposed_face_palm_down_high_thumb_variable_size_"
    "pose_preserving_grasp_then_lift"
)
EXPERIMENT_ID = (
    "left_opposed_face_palm_down_high_thumb_normal_aligned_smooth_vertical_lift"
)
CAMPAIGN_KIND = "normal_aligned_smooth_vertical_lift"
CAMPAIGN_SCHEMA_VERSION = 1
POSE_MANIFEST_SCHEMA_VERSION = 1
CANDIDATE_RESULT_SCHEMA_VERSION = 1
DEFAULT_SEED = 20260821

# The v8 manipulation search operates on *relative* actuator targets, while
# MuJoCo validates the resulting absolute position command.  Keep the pinned
# xhand_left.xml position limits here so a valid relative search bound cannot
# push a source grasp beyond its real joint/ctrl range.  The model uses the
# same values for both ranges; a regression test checks this table against the
# compiled model.
ACTIVE_ACTUATOR_POSITION_LIMITS_RAD = {
    "left_hand_thumb_bend_joint_actuator": (0.0, 1.832),
    "left_hand_thumb_rota_joint1_actuator": (-0.698, 1.57),
    "left_hand_thumb_rota_joint2_actuator": (0.0, 1.57),
    "left_hand_index_bend_joint_actuator": (-0.174, 0.174),
    "left_hand_index_joint1_actuator": (0.0, 1.919),
    "left_hand_index_joint2_actuator": (0.0, 1.919),
    "left_hand_mid_joint1_actuator": (0.0, 1.919),
    "left_hand_mid_joint2_actuator": (0.0, 1.919),
}

DEFAULT_V7_CAMPAIGN_RESULTS = Path(
    "artifacts/"
    "left_opposed_face_palm_down_high_thumb_variable_size_pose_preserving/"
    "campaign_results.json"
)
DEFAULT_V8_TEMPLATE = Path(
    "grasp_configs/"
    "left_opposed_face_palm_down_high_thumb_normal_aligned_"
    "smooth_vertical_lift.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/"
    "left_opposed_face_palm_down_high_thumb_normal_aligned_smooth_vertical_lift/"
    "tune/rescue"
)
DEFAULT_LIFT_OUTPUT = Path(
    "artifacts/"
    "left_opposed_face_palm_down_high_thumb_normal_aligned_smooth_vertical_lift/"
    "tune/lift"
)

THUMB_BEND_ACTUATOR = "left_hand_thumb_bend_joint_actuator"
NON_THUMB_TARGET_ACTUATORS = tuple(
    name for name in ACTIVE_ACTUATORS if name != THUMB_BEND_ACTUATOR
)
POSE_CONTEXT_FIELDS = ("side", "scene", "hand_pose", "cube", "contact_topology")
CLOSE_DURATION_OPTIONS_S = (1.0, 1.25, 1.5)

TIER_A_MIN_GATE_STEPS = 500
RESCUE_MIN_GATE_STEPS = 250
RESCUE_EDGE_RANGE_M = (0.060, 0.070)

_TIER_A_COARSE_BASE = 81_000_000_000_000
_TIER_A_LOCAL_BASE = 82_000_000_000_000
_TIER_B_COARSE_BASE = 83_000_000_000_000
_TIER_B_LOCAL_BASE = 84_000_000_000_000
_EXACT_BASE = 85_000_000_000_000
_LIFT_PROBE_BASE = 86_000_000_000_000
_LIFT_TRUST_BASE = 87_000_000_000_000
_LIFT_REFINE_BASE = 88_000_000_000_000
_LIFT_EXACT_BASE = 89_000_000_000_000
_ID_STRIDE = 1_000_000

_LIFT_RESPONSE_TARGET = (0.0, 0.0, 0.011, 0.0, 0.0, 0.0)
_HISTORICAL_LIFT_DELTA_RAD = {
    "left_hand_thumb_bend_joint_actuator": -0.19913,
    "left_hand_thumb_rota_joint1_actuator": 0.09008,
    "left_hand_thumb_rota_joint2_actuator": 0.36111,
    "left_hand_index_bend_joint_actuator": -0.00074,
    "left_hand_index_joint1_actuator": -0.12366,
    "left_hand_index_joint2_actuator": 0.13474,
    "left_hand_mid_joint1_actuator": 0.03705,
    "left_hand_mid_joint2_actuator": 0.41102,
}

_VALIDATED_67_NON_THUMB_TARGETS = {
    "left_hand_thumb_rota_joint1_actuator": 0.3139796524041977,
    "left_hand_thumb_rota_joint2_actuator": 0.8842179936059648,
    "left_hand_index_joint1_actuator": 0.6810364882476693,
    "left_hand_index_joint2_actuator": 1.267336875450936,
    "left_hand_mid_joint1_actuator": 0.8554395853489825,
    "left_hand_mid_joint2_actuator": 1.0934414295482138,
}
_VALIDATED_67_STARTS = {"thumb": 0.15, "index": 0.13, "mid": 0.04}
_VALIDATED_67_AMPLITUDES = {"thumb": 0.50, "index": 0.675, "mid": 0.80}


CandidateExecutor = Callable[
    [Sequence[Mapping[str, Any]], int], Sequence[Mapping[str, Any]]
]


@dataclass(frozen=True, slots=True)
class RescueBudget:
    """The fixed, decision-complete 1,616-run v8 pose-rescue budget."""

    tier_a_pose_count: int = 11
    tier_b_pose_count: int = 10
    tier_a_coarse_per_pose: int = 32
    tier_a_local_seed_count_per_pose: int = 2
    tier_a_local_per_seed: int = 32
    tier_a_exact_count: int = 16
    tier_b_coarse_per_pose: int = 16
    tier_b_global_local_seed_count: int = 8
    tier_b_local_per_seed: int = 48

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"budget.{name} must be a positive integer")

    @property
    def tier_a_coarse_count(self) -> int:
        return self.tier_a_pose_count * self.tier_a_coarse_per_pose

    @property
    def tier_a_local_count(self) -> int:
        return (
            self.tier_a_pose_count
            * self.tier_a_local_seed_count_per_pose
            * self.tier_a_local_per_seed
        )

    @property
    def tier_a_total(self) -> int:
        return self.tier_a_coarse_count + self.tier_a_local_count + self.tier_a_exact_count

    @property
    def tier_b_coarse_count(self) -> int:
        return self.tier_b_pose_count * self.tier_b_coarse_per_pose

    @property
    def tier_b_local_count(self) -> int:
        return self.tier_b_global_local_seed_count * self.tier_b_local_per_seed

    @property
    def tier_b_total(self) -> int:
        return self.tier_b_coarse_count + self.tier_b_local_count

    @property
    def total(self) -> int:
        return self.tier_a_total + self.tier_b_total


FIXED_RESCUE_BUDGET = RescueBudget()


@dataclass(frozen=True, slots=True)
class LiftBudget:
    """Fixed response-probe and deterministic trust-region lift budget."""

    pose_count: int = 24
    probe_count_per_pose: int = 17
    trust_candidates_per_pose: int = 64
    refine_pose_count: int = 8
    refine_per_pose: int = 128
    exact_count: int = 5

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"lift_budget.{name} must be a positive integer")
        if self.probe_count_per_pose != 1 + 2 * len(ACTIVE_ACTUATORS):
            raise ValueError("lift probe budget must be zero plus +/- for eight actuators")

    @property
    def maximum_probe_count(self) -> int:
        return self.pose_count * self.probe_count_per_pose

    @property
    def maximum_trust_count(self) -> int:
        return self.pose_count * self.trust_candidates_per_pose

    @property
    def maximum_refine_count(self) -> int:
        return self.refine_pose_count * self.refine_per_pose


FIXED_LIFT_BUDGET = LiftBudget()


@dataclass(frozen=True, slots=True)
class RescueControlSpec:
    """One controller before registered actuator-bound clipping."""

    pregrasp_residual_rad: tuple[float, ...]
    non_thumb_target_residual_rad: tuple[float, ...]
    group_start_fraction: tuple[float, float, float]
    group_end_fraction: tuple[float, float, float]
    group_amplitude_scale: tuple[float, float, float]
    close_s: float
    anchor_kind: str

    def __post_init__(self) -> None:
        expected = (
            (self.pregrasp_residual_rad, len(ACTIVE_ACTUATORS), "pregrasp residual"),
            (
                self.non_thumb_target_residual_rad,
                len(NON_THUMB_TARGET_ACTUATORS),
                "non-thumb target residual",
            ),
            (self.group_start_fraction, len(CLOSE_GROUP_ORDER), "group start"),
            (self.group_end_fraction, len(CLOSE_GROUP_ORDER), "group end"),
            (self.group_amplitude_scale, len(CLOSE_GROUP_ORDER), "group amplitude"),
        )
        for values, length, label in expected:
            if len(values) != length or not np.isfinite(values).all():
                raise ValueError(f"{label} must contain {length} finite values")
        for start, end in zip(self.group_start_fraction, self.group_end_fraction):
            if not 0.0 <= float(start) < float(end) <= 1.0:
                raise ValueError("group profile must satisfy 0 <= start < end <= 1")
        if any(not 0.25 <= float(value) <= 1.50 for value in self.group_amplitude_scale):
            raise ValueError("group amplitude must stay within [0.25, 1.50]")
        if not any(math.isclose(self.close_s, option, abs_tol=1e-12) for option in CLOSE_DURATION_OPTIONS_S):
            raise ValueError(
                f"close_s must be one of {CLOSE_DURATION_OPTIONS_S}, got {self.close_s}"
            )
        if not self.anchor_kind:
            raise ValueError("anchor_kind must not be empty")

    def as_dict(self) -> dict[str, Any]:
        return {
            "pregrasp_residual_rad": {
                name: float(value)
                for name, value in zip(ACTIVE_ACTUATORS, self.pregrasp_residual_rad)
            },
            "non_thumb_target_residual_rad": {
                name: float(value)
                for name, value in zip(
                    NON_THUMB_TARGET_ACTUATORS,
                    self.non_thumb_target_residual_rad,
                )
            },
            "group_start_fraction": {
                group: float(value)
                for group, value in zip(CLOSE_GROUP_ORDER, self.group_start_fraction)
            },
            "group_end_fraction": {
                group: float(value)
                for group, value in zip(CLOSE_GROUP_ORDER, self.group_end_fraction)
            },
            "group_amplitude_scale": {
                group: float(value)
                for group, value in zip(CLOSE_GROUP_ORDER, self.group_amplitude_scale)
            },
            "close_s": float(self.close_s),
            "anchor_kind": self.anchor_kind,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RescueControlSpec":
        return cls(
            pregrasp_residual_rad=tuple(
                float(value["pregrasp_residual_rad"][name])
                for name in ACTIVE_ACTUATORS
            ),
            non_thumb_target_residual_rad=tuple(
                float(value["non_thumb_target_residual_rad"][name])
                for name in NON_THUMB_TARGET_ACTUATORS
            ),
            group_start_fraction=tuple(
                float(value["group_start_fraction"][group])
                for group in CLOSE_GROUP_ORDER
            ),
            group_end_fraction=tuple(
                float(value["group_end_fraction"][group])
                for group in CLOSE_GROUP_ORDER
            ),
            group_amplitude_scale=tuple(
                float(value["group_amplitude_scale"][group])
                for group in CLOSE_GROUP_ORDER
            ),
            close_s=float(value["close_s"]),
            anchor_kind=str(value["anchor_kind"]),
        )


def _safe_member(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} must be a non-empty relative path")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes its campaign directory") from error
    return candidate


def pose_context(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact geometry used to identify an independently tunable pose."""

    return {
        key: copy.deepcopy(config[key])
        for key in ("cube", "hand_pose", "contact_topology")
    }


def pose_id(config: Mapping[str, Any]) -> str:
    return canonical_sha256(pose_context(config))


def controller_context(config: Mapping[str, Any]) -> dict[str, Any]:
    protocol = config.get("control_protocol", {})
    return {
        "control": copy.deepcopy(config["control"]),
        "close_s": float(protocol.get("close_s", 0.0)),
        "manipulation_profile": protocol.get("manipulation_profile"),
    }


def controller_id(config: Mapping[str, Any]) -> str:
    return canonical_sha256(controller_context(config))


def _all_state_gate_steps(result: Mapping[str, Any]) -> int:
    declared = result.get("rank_metrics", {})
    if isinstance(declared, Mapping) and declared.get("all_state_gate_steps") is not None:
        return max(0, int(float(declared["all_state_gate_steps"])))
    summary = result.get("summary", {})
    metrics = summary.get("metrics", {}) if isinstance(summary, Mapping) else {}
    for name in (
        "verify_max_consecutive_all_gate_steps",
        "grasp_gate_final_consecutive_steps",
    ):
        if isinstance(metrics, Mapping) and metrics.get(name) is not None:
            return max(0, int(float(metrics[name])))
    return 0


def _pose_manifest_record(
    campaign_root: Path,
    binding: Mapping[str, Any],
) -> dict[str, Any] | None:
    result_path = _safe_member(campaign_root, binding.get("result"), "candidate result")
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    expected_result_sha = binding.get("result_sha256")
    if expected_result_sha is not None and file_sha256(result_path) != expected_result_sha:
        raise ValueError(f"candidate result hash mismatch: {result_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    candidate_id_value = int(result.get("candidate_id", -1))
    if candidate_id_value != int(binding.get("candidate_id", -2)):
        raise ValueError(f"candidate ID binding mismatch: {result_path}")
    gate_steps = _all_state_gate_steps(result)
    edge_m = float(result.get("edge_m", math.nan))
    if gate_steps < RESCUE_MIN_GATE_STEPS or not (
        RESCUE_EDGE_RANGE_M[0] - 1e-12
        <= edge_m
        <= RESCUE_EDGE_RANGE_M[1] + 1e-12
    ):
        return None
    artifacts = result.get("artifacts", {})
    if not isinstance(artifacts, Mapping):
        raise ValueError(f"candidate has no artifact map: {result_path}")
    config_name = artifacts.get("resolved_config", "resolved_config.json")
    config_path = _safe_member(result_path.parent, config_name, "resolved config")
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    hashes = artifacts.get("sha256", {})
    if isinstance(hashes, Mapping) and hashes.get("resolved_config") is not None:
        if file_sha256(config_path) != hashes["resolved_config"]:
            raise ValueError(f"candidate config file hash mismatch: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if int(config.get("schema_version", 0)) != 7 or config.get("experiment_id") != V7_EXPERIMENT_ID:
        raise ValueError(f"rescue source is not the registered v7 experiment: {config_path}")
    semantic_sha = canonical_sha256(config)
    for expected in (binding.get("candidate_sha256"), result.get("candidate_sha256")):
        if expected is not None and expected != semantic_sha:
            raise ValueError(f"candidate semantic hash mismatch: {config_path}")
    metrics = result.get("summary", {}).get("metrics", {})
    pose_metrics = metrics.get("pose_preservation", {}) if isinstance(metrics, Mapping) else {}
    identifier = pose_id(config)
    tier = "A" if gate_steps >= TIER_A_MIN_GATE_STEPS else "B"
    return {
        "pose_id": identifier,
        "tier": tier,
        "source_candidate_id": candidate_id_value,
        "source_family_id": str(result.get("source_family_id", "unknown")),
        "source_trajectory_id": str(result.get("source_trajectory_id", "unknown")),
        "edge_m": edge_m,
        "thumb_target_rad": float(result.get("thumb_target_rad", math.nan)),
        "all_state_gate_steps": gate_steps,
        "first_distal_contact_step": copy.deepcopy(
            pose_metrics.get("first_distal_contact_step", {})
            if isinstance(pose_metrics, Mapping)
            else {}
        ),
        "source_result": str(result_path),
        "source_config": str(config_path),
        "source_result_sha256": file_sha256(result_path),
        "source_config_file_sha256": file_sha256(config_path),
        "source_config_sha256": semantic_sha,
    }


def build_pose_rescue_manifest(
    campaign_results_path: str | Path = DEFAULT_V7_CAMPAIGN_RESULTS,
    *,
    expected_count: int | None = 21,
) -> dict[str, Any]:
    """Authenticate, filter and pose-deduplicate the completed v7 campaign."""

    campaign_path = Path(campaign_results_path).expanduser().resolve()
    payload = json.loads(campaign_path.read_text(encoding="utf-8"))
    if payload.get("complete") is not True:
        raise ValueError("v7 campaign_results.json must declare complete=true")
    if payload.get("experiment_id") != V7_EXPERIMENT_ID:
        raise ValueError("pose rescue source belongs to the wrong experiment")
    bindings = payload.get("candidate_results")
    if not isinstance(bindings, list):
        raise ValueError("v7 campaign_results.json has no candidate_results list")
    campaign_root = campaign_path.parent
    raw: list[dict[str, Any]] = []
    for binding in bindings:
        if not isinstance(binding, Mapping):
            raise ValueError("candidate result binding must be an object")
        record = _pose_manifest_record(campaign_root, binding)
        if record is not None:
            raw.append(record)
    best_by_pose: dict[str, dict[str, Any]] = {}
    for record in raw:
        identifier = str(record["pose_id"])
        previous = best_by_pose.get(identifier)
        selection_key = (
            -int(record["all_state_gate_steps"]),
            int(record["source_candidate_id"]),
        )
        if previous is None or selection_key < (
            -int(previous["all_state_gate_steps"]),
            int(previous["source_candidate_id"]),
        ):
            best_by_pose[identifier] = record
    poses = sorted(
        best_by_pose.values(),
        key=lambda value: (
            value["tier"] != "A",
            -int(value["all_state_gate_steps"]),
            int(value["source_candidate_id"]),
        ),
    )
    if expected_count is not None and len(poses) != int(expected_count):
        raise RuntimeError(
            f"authenticated pose rescue manifest has {len(poses)} poses; "
            f"expected {expected_count}"
        )
    tier_counts = {
        tier: sum(record["tier"] == tier for record in poses) for tier in ("A", "B")
    }
    return {
        "pose_rescue_manifest_schema_version": POSE_MANIFEST_SCHEMA_VERSION,
        "campaign_kind": CAMPAIGN_KIND,
        "source_experiment_id": V7_EXPERIMENT_ID,
        "target_experiment_id": EXPERIMENT_ID,
        "source_campaign_results": str(campaign_path),
        "source_campaign_results_sha256": file_sha256(campaign_path),
        "selection": {
            "minimum_all_state_gate_steps": RESCUE_MIN_GATE_STEPS,
            "tier_a_minimum_steps": TIER_A_MIN_GATE_STEPS,
            "edge_range_m": list(RESCUE_EDGE_RANGE_M),
            "pose_hash_projection": ["cube", "hand_pose", "contact_topology"],
            "raw_eligible_count": len(raw),
            "deduplicated_count": len(poses),
            "tier_counts": tier_counts,
        },
        "poses": poses,
    }


def _profile_groups(config: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    profile = config["control"]["close_profile"]
    starts: dict[str, float] = {}
    ends: dict[str, float] = {}
    for group in CLOSE_GROUP_ORDER:
        group_starts = {float(profile[name]["start_fraction"]) for name in CLOSE_GROUP_ACTUATORS[group]}
        group_ends = {float(profile[name]["end_fraction"]) for name in CLOSE_GROUP_ACTUATORS[group]}
        if len(group_starts) != 1 or len(group_ends) != 1:
            raise ValueError(f"source close profile group {group!r} is not synchronized")
        starts[group] = group_starts.pop()
        ends[group] = group_ends.pop()
    return starts, ends


def _zeros(length: int) -> tuple[float, ...]:
    return tuple(0.0 for _ in range(length))


def original_control_spec(source_config: Mapping[str, Any]) -> RescueControlSpec:
    starts, ends = _profile_groups(source_config)
    source_close = float(source_config.get("control_protocol", {}).get("close_s", 1.0))
    close_s = min(CLOSE_DURATION_OPTIONS_S, key=lambda value: abs(value - source_close))
    return RescueControlSpec(
        pregrasp_residual_rad=_zeros(len(ACTIVE_ACTUATORS)),
        non_thumb_target_residual_rad=_zeros(len(NON_THUMB_TARGET_ACTUATORS)),
        group_start_fraction=tuple(starts[group] for group in CLOSE_GROUP_ORDER),
        group_end_fraction=tuple(ends[group] for group in CLOSE_GROUP_ORDER),
        group_amplitude_scale=(1.0, 1.0, 1.0),
        close_s=close_s,
        anchor_kind="original_control",
    )


def validated_67_control_spec(source_config: Mapping[str, Any]) -> RescueControlSpec:
    targets = source_config["control"]["grasp_targets_rad"]
    residuals = tuple(
        float(_VALIDATED_67_NON_THUMB_TARGETS.get(name, targets[name]))
        - float(targets[name])
        for name in NON_THUMB_TARGET_ACTUATORS
    )
    _, source_ends = _profile_groups(source_config)
    return RescueControlSpec(
        pregrasp_residual_rad=_zeros(len(ACTIVE_ACTUATORS)),
        non_thumb_target_residual_rad=residuals,
        group_start_fraction=tuple(
            _VALIDATED_67_STARTS[group] for group in CLOSE_GROUP_ORDER
        ),
        group_end_fraction=tuple(source_ends[group] for group in CLOSE_GROUP_ORDER),
        group_amplitude_scale=tuple(
            _VALIDATED_67_AMPLITUDES[group] for group in CLOSE_GROUP_ORDER
        ),
        close_s=1.0,
        anchor_kind="validated_67_control",
    )


def analytic_synchronization_spec(
    source_config: Mapping[str, Any], pose_record: Mapping[str, Any]
) -> RescueControlSpec:
    """Delay earlier first contacts so all three distal contacts arrive together."""

    base = original_control_spec(source_config)
    starts = dict(zip(CLOSE_GROUP_ORDER, base.group_start_fraction))
    ends = dict(zip(CLOSE_GROUP_ORDER, base.group_end_fraction))
    onsets = pose_record.get("first_distal_contact_step", {})
    if isinstance(onsets, Mapping) and all(
        onsets.get(group) is not None for group in CLOSE_GROUP_ORDER
    ):
        latest = max(float(onsets[group]) for group in CLOSE_GROUP_ORDER)
        close_steps = max(1.0, base.close_s / 0.001)
        for group in CLOSE_GROUP_ORDER:
            delay = (latest - float(onsets[group])) / close_steps
            starts[group] = float(
                np.clip(starts[group] + delay, 0.0, ends[group] - 0.05)
            )
    else:
        pregrasp = source_config["control"]["pregrasp_targets_rad"]
        targets = source_config["control"]["grasp_targets_rad"]
        travel = {
            group: float(
                np.linalg.norm(
                    [
                        float(targets[name]) - float(pregrasp[name])
                        for name in CLOSE_GROUP_ACTUATORS[group]
                    ]
                )
            )
            for group in CLOSE_GROUP_ORDER
        }
        maximum = max(max(travel.values()), 1e-12)
        for group in CLOSE_GROUP_ORDER:
            duration = max(0.05, travel[group] / maximum)
            starts[group] = max(0.0, ends[group] - duration)
    return RescueControlSpec(
        pregrasp_residual_rad=base.pregrasp_residual_rad,
        non_thumb_target_residual_rad=base.non_thumb_target_residual_rad,
        group_start_fraction=tuple(starts[group] for group in CLOSE_GROUP_ORDER),
        group_end_fraction=base.group_end_fraction,
        group_amplitude_scale=base.group_amplitude_scale,
        close_s=base.close_s,
        anchor_kind="analytic_contact_synchronization",
    )


def _latin_hypercube(count: int, dimensions: int, seed: int) -> np.ndarray:
    if count <= 0:
        return np.empty((0, dimensions), dtype=np.float64)
    rng = np.random.default_rng(seed)
    result = np.empty((count, dimensions), dtype=np.float64)
    for dimension in range(dimensions):
        values = (np.arange(count, dtype=np.float64) + rng.random(count)) / count
        rng.shuffle(values)
        result[:, dimension] = 2.0 * values - 1.0
    return result


def _clip_profile(start: float, end: float) -> tuple[float, float]:
    clipped_end = float(np.clip(end, 0.10, 1.0))
    clipped_start = float(np.clip(start, 0.0, clipped_end - 0.05))
    return clipped_start, clipped_end


def generate_control_specs(
    source_config: Mapping[str, Any],
    pose_record: Mapping[str, Any],
    *,
    count: int,
    seed: int,
) -> tuple[RescueControlSpec, ...]:
    """Return the three evidence anchors followed by deterministic LHS controls."""

    if count <= 0:
        raise ValueError("control candidate count must be positive")
    anchors = (
        original_control_spec(source_config),
        validated_67_control_spec(source_config),
        analytic_synchronization_spec(source_config, pose_record),
    )
    if count <= len(anchors):
        return anchors[:count]
    remaining = count - len(anchors)
    # 8 pregrasp residuals, 7 terminal residuals, 3 starts, 3 ends,
    # 3 group amplitudes and one categorical-close selector.
    units = _latin_hypercube(remaining, 25, seed)
    generated: list[RescueControlSpec] = list(anchors)
    centre = anchors[2]
    for index, unit in enumerate(units):
        pregrasp = tuple(float(value * 0.08) for value in unit[:8])
        targets = tuple(float(value * 0.08) for value in unit[8:15])
        starts: list[float] = []
        ends: list[float] = []
        for group_index in range(3):
            start, end = _clip_profile(
                centre.group_start_fraction[group_index] + unit[15 + group_index] * 0.12,
                centre.group_end_fraction[group_index] + unit[18 + group_index] * 0.10,
            )
            starts.append(start)
            ends.append(end)
        amplitudes = tuple(
            float(np.clip(0.80 + unit[21 + group_index] * 0.40, 0.40, 1.20))
            for group_index in range(3)
        )
        close_bin = min(2, int((unit[24] + 1.0) * 1.5))
        generated.append(
            RescueControlSpec(
                pregrasp_residual_rad=pregrasp,
                non_thumb_target_residual_rad=targets,
                group_start_fraction=tuple(starts),
                group_end_fraction=tuple(ends),
                group_amplitude_scale=amplitudes,
                close_s=CLOSE_DURATION_OPTIONS_S[close_bin],
                anchor_kind="latin_hypercube",
            )
        )
    return tuple(generated)


def _control_bounds(
    template: Mapping[str, Any], field: str
) -> Mapping[str, Sequence[float]] | None:
    for block_name in (
        "normal_aligned_smooth_lift_campaign",
        "normal_aligned_smooth_vertical_lift_campaign",
        "high_thumb_size_campaign",
    ):
        block = template.get(block_name)
        if not isinstance(block, Mapping):
            continue
        search = block.get("control_search", block)
        if isinstance(search, Mapping) and isinstance(search.get(field), Mapping):
            return search[field]
    try:
        from ..experiment import resolve_experiment

        bounds = resolve_experiment(dict(template)).search_bounds
        return (
            bounds.pregrasp_targets_rad
            if field == "pregrasp_target_bounds_rad"
            else bounds.actuator_targets_rad
        )
    except (AttributeError, TypeError, ValueError):
        return None


def _bounded(value: float, bounds: Mapping[str, Sequence[float]] | None, name: str) -> float:
    if bounds is None or name not in bounds:
        return float(value)
    return float(np.clip(value, float(bounds[name][0]), float(bounds[name][1])))


def assert_pose_context_preserved(
    source_v7: Mapping[str, Any], candidate_v8: Mapping[str, Any]
) -> None:
    for field in POSE_CONTEXT_FIELDS:
        if candidate_v8.get(field) != source_v7.get(field):
            raise ValueError(f"v8 rescue candidate changed frozen source field {field}")
    expected_pose_id = pose_id(source_v7)
    metadata = candidate_v8.get("candidate_metadata", {})
    if metadata.get("pose_id") != expected_pose_id:
        raise ValueError("v8 rescue candidate pose_id does not bind the source pose")
    delta = candidate_v8["control"]["manipulation_delta_rad"]
    if set(delta) != set(ACTIVE_ACTUATORS) or any(float(value) != 0.0 for value in delta.values()):
        raise ValueError("control rescue must keep manipulation delta exactly zero")


def materialize_v8_rescue_candidate(
    source_v7: Mapping[str, Any],
    v8_template: Mapping[str, Any],
    pose_record: Mapping[str, Any],
    spec: RescueControlSpec,
    *,
    candidate_id_value: int,
    stage: str,
    parent_candidate_id: int | None = None,
    validator: Callable[[dict[str, Any]], None] | None = validate_config,
) -> dict[str, Any]:
    """Put one v7 pose into v8 while changing only its closure controller."""

    if int(source_v7.get("schema_version", 0)) != 7:
        raise ValueError("rescue source must use schema version 7")
    if int(v8_template.get("schema_version", 0)) != 8:
        raise ValueError("rescue template must use schema version 8")
    if v8_template.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("rescue template has the wrong experiment_id")
    candidate = copy.deepcopy(dict(v8_template))
    for field in POSE_CONTEXT_FIELDS:
        candidate[field] = copy.deepcopy(source_v7[field])

    source_control = source_v7["control"]
    source_pregrasp = source_control["pregrasp_targets_rad"]
    source_targets = source_control["grasp_targets_rad"]
    pregrasp_bounds = _control_bounds(v8_template, "pregrasp_target_bounds_rad")
    target_bounds = _control_bounds(v8_template, "grasp_target_bounds_rad")
    target_residual = dict(
        zip(NON_THUMB_TARGET_ACTUATORS, spec.non_thumb_target_residual_rad)
    )
    targets = {
        name: _bounded(
            float(source_targets[name]) + float(target_residual.get(name, 0.0)),
            target_bounds,
            name,
        )
        for name in ACTIVE_ACTUATORS
    }
    # The source pose's commanded thumb band remains an invariant.  Group
    # amplitude modifies travel by moving the pregrasp, never by relabelling
    # the pose with a different terminal thumb command.
    targets[THUMB_BEND_ACTUATOR] = float(source_targets[THUMB_BEND_ACTUATOR])
    proposed_pregrasp = {
        name: float(source_pregrasp[name]) + float(spec.pregrasp_residual_rad[index])
        for index, name in enumerate(ACTIVE_ACTUATORS)
    }
    amplitudes = dict(zip(CLOSE_GROUP_ORDER, spec.group_amplitude_scale))
    pregrasp: dict[str, float] = {}
    for group in CLOSE_GROUP_ORDER:
        amplitude = float(amplitudes[group])
        for name in CLOSE_GROUP_ACTUATORS[group]:
            value = targets[name] - amplitude * (targets[name] - proposed_pregrasp[name])
            pregrasp[name] = _bounded(value, pregrasp_bounds, name)
    starts = dict(zip(CLOSE_GROUP_ORDER, spec.group_start_fraction))
    ends = dict(zip(CLOSE_GROUP_ORDER, spec.group_end_fraction))
    close_profile = {
        name: {
            "start_fraction": float(starts[group]),
            "end_fraction": float(ends[group]),
        }
        for group in CLOSE_GROUP_ORDER
        for name in CLOSE_GROUP_ACTUATORS[group]
    }
    candidate["control"] = {
        "pregrasp_targets_rad": pregrasp,
        "grasp_targets_rad": targets,
        "manipulation_delta_rad": {name: 0.0 for name in ACTIVE_ACTUATORS},
        "close_profile": close_profile,
    }
    candidate["control_protocol"]["close_s"] = float(spec.close_s)
    candidate.pop("run_context", None)
    source_identifier = pose_id(source_v7)
    metadata = {
        "campaign_kind": CAMPAIGN_KIND,
        "candidate_id": int(candidate_id_value),
        "stage": str(stage),
        "pose_id": source_identifier,
        "source_candidate_id": int(pose_record["source_candidate_id"]),
        "source_family_id": str(pose_record.get("source_family_id", "unknown")),
        "source_trajectory_id": str(
            pose_record.get("source_trajectory_id", "unknown")
        ),
        "source_config_sha256": str(pose_record["source_config_sha256"]),
        "control_rescue": True,
        "cube_pose_sampled": False,
        "free_cube_pose_reset_during_run": False,
        "rescue_control_spec": spec.as_dict(),
    }
    if parent_candidate_id is not None:
        metadata["parent_candidate_id"] = int(parent_candidate_id)
    candidate["candidate_metadata"] = metadata
    metadata["controller_id"] = controller_id(candidate)
    assert_pose_context_preserved(source_v7, candidate)
    if validator is not None:
        validator(candidate)
    return candidate


def _candidate_record(
    config: Mapping[str, Any], pose_record: Mapping[str, Any]
) -> dict[str, Any]:
    metadata = config["candidate_metadata"]
    return {
        "campaign_kind": CAMPAIGN_KIND,
        "candidate_id": int(metadata["candidate_id"]),
        "candidate_sha256": canonical_sha256(config),
        "stage": str(metadata["stage"]),
        "pose_id": str(metadata["pose_id"]),
        "controller_id": str(metadata["controller_id"]),
        "tier": str(pose_record["tier"]),
        "source_candidate_id": int(pose_record["source_candidate_id"]),
        "source_family_id": str(pose_record.get("source_family_id", "unknown")),
        "source_trajectory_id": str(
            pose_record.get("source_trajectory_id", "unknown")
        ),
        "edge_m": float(config["cube"]["edge_m"]),
        "thumb_target_rad": float(
            config["control"]["grasp_targets_rad"][THUMB_BEND_ACTUATOR]
        ),
        "config": copy.deepcopy(dict(config)),
    }


def generate_pose_coarse_candidates(
    source_v7: Mapping[str, Any],
    v8_template: Mapping[str, Any],
    pose_record: Mapping[str, Any],
    *,
    pose_index: int,
    count: int,
    seed: int = DEFAULT_SEED,
    validator: Callable[[dict[str, Any]], None] | None = validate_config,
) -> tuple[dict[str, Any], ...]:
    tier = str(pose_record["tier"])
    if tier not in ("A", "B"):
        raise ValueError("pose tier must be A or B")
    stage = f"rescue_tier_{tier.lower()}_coarse"
    base = _TIER_A_COARSE_BASE if tier == "A" else _TIER_B_COARSE_BASE
    specs = generate_control_specs(
        source_v7,
        pose_record,
        count=count,
        seed=int(
            np.random.SeedSequence(
                [seed, int(pose_record["source_candidate_id"]), 8_100_001]
            ).generate_state(1)[0]
        ),
    )
    records = []
    for local_index, spec in enumerate(specs):
        identifier = base + int(pose_index) * _ID_STRIDE + local_index
        config = materialize_v8_rescue_candidate(
            source_v7,
            v8_template,
            pose_record,
            spec,
            candidate_id_value=identifier,
            stage=stage,
            validator=validator,
        )
        records.append(_candidate_record(config, pose_record))
    return tuple(records)


def _local_control_specs(
    parent: Mapping[str, Any], *, count: int, seed: int
) -> tuple[RescueControlSpec, ...]:
    metadata = parent["config"]["candidate_metadata"]
    centre = RescueControlSpec.from_dict(metadata["rescue_control_spec"])
    if count <= 0:
        raise ValueError("local control count must be positive")
    units = _latin_hypercube(max(0, count - 1), 25, seed)
    result = [
        RescueControlSpec(
            **{
                **asdict(centre),
                "anchor_kind": "local_parent_exact",
            }
        )
    ]
    for unit in units:
        starts: list[float] = []
        ends: list[float] = []
        for group_index in range(3):
            start, end = _clip_profile(
                centre.group_start_fraction[group_index] + unit[15 + group_index] * 0.03,
                centre.group_end_fraction[group_index] + unit[18 + group_index] * 0.03,
            )
            starts.append(start)
            ends.append(end)
        close_index = CLOSE_DURATION_OPTIONS_S.index(centre.close_s)
        if unit[24] < -0.50:
            close_index = max(0, close_index - 1)
        elif unit[24] > 0.50:
            close_index = min(len(CLOSE_DURATION_OPTIONS_S) - 1, close_index + 1)
        result.append(
            RescueControlSpec(
                pregrasp_residual_rad=tuple(
                    centre.pregrasp_residual_rad[index] + unit[index] * 0.02
                    for index in range(8)
                ),
                non_thumb_target_residual_rad=tuple(
                    centre.non_thumb_target_residual_rad[index] + unit[8 + index] * 0.025
                    for index in range(7)
                ),
                group_start_fraction=tuple(starts),
                group_end_fraction=tuple(ends),
                group_amplitude_scale=tuple(
                    float(
                        np.clip(
                            centre.group_amplitude_scale[index] + unit[21 + index] * 0.08,
                            0.25,
                            1.50,
                        )
                    )
                    for index in range(3)
                ),
                close_s=CLOSE_DURATION_OPTIONS_S[close_index],
                anchor_kind="local_latin_hypercube",
            )
        )
    return tuple(result)


def generate_local_candidates(
    parents: Sequence[Mapping[str, Any]],
    source_by_pose: Mapping[str, tuple[Mapping[str, Any], Mapping[str, Any]]],
    v8_template: Mapping[str, Any],
    *,
    count_per_parent: int,
    tier: str,
    seed: int = DEFAULT_SEED,
    validator: Callable[[dict[str, Any]], None] | None = validate_config,
) -> tuple[dict[str, Any], ...]:
    if tier not in ("A", "B"):
        raise ValueError("local tier must be A or B")
    base = _TIER_A_LOCAL_BASE if tier == "A" else _TIER_B_LOCAL_BASE
    stage = f"rescue_tier_{tier.lower()}_local"
    records: list[dict[str, Any]] = []
    for parent_index, parent in enumerate(parents):
        identifier = str(parent["pose_id"])
        source, pose_record = source_by_pose[identifier]
        specs = _local_control_specs(
            parent,
            count=count_per_parent,
            seed=int(
                np.random.SeedSequence(
                    [seed, int(parent["candidate_id"]), 8_200_001]
                ).generate_state(1)[0]
            ),
        )
        for local_index, spec in enumerate(specs):
            candidate_id_value = base + parent_index * _ID_STRIDE + local_index
            config = materialize_v8_rescue_candidate(
                source,
                v8_template,
                pose_record,
                spec,
                candidate_id_value=candidate_id_value,
                stage=stage,
                parent_candidate_id=int(parent["candidate_id"]),
                validator=validator,
            )
            records.append(_candidate_record(config, pose_record))
    return tuple(records)


def generate_exact_candidates(
    ranked: Sequence[Mapping[str, Any]],
    *,
    count: int,
    validator: Callable[[dict[str, Any]], None] | None = validate_config,
) -> tuple[dict[str, Any], ...]:
    generated = []
    for index, parent in enumerate(ranked[:count]):
        config = copy.deepcopy(dict(parent["config"]))
        metadata = config["candidate_metadata"]
        metadata.update(
            {
                "candidate_id": _EXACT_BASE + index,
                "stage": "rescue_exact",
                "parent_candidate_id": int(parent["candidate_id"]),
                "locked_timestep_s": 0.001,
            }
        )
        # An exact replay has the same controller_id by design.
        if metadata["controller_id"] != controller_id(config):
            raise ValueError("exact replay changed its controller")
        if validator is not None:
            validator(config)
        record = {
            key: copy.deepcopy(parent[key])
            for key in (
                "pose_id",
                "controller_id",
                "tier",
                "source_candidate_id",
                "source_family_id",
                "source_trajectory_id",
                "edge_m",
                "thumb_target_rad",
            )
        }
        record.update(
            {
                "campaign_kind": CAMPAIGN_KIND,
                "candidate_id": _EXACT_BASE + index,
                "candidate_sha256": canonical_sha256(config),
                "stage": "rescue_exact",
                "config": config,
            }
        )
        generated.append(record)
    return tuple(generated)


def _finite_metric(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _closure_rank_metrics(
    metrics: Mapping[str, Any], checks: Mapping[str, Any] | None = None
) -> tuple[bool, bool, float, float]:
    block = metrics.get("closure_alignment")
    if not isinstance(block, Mapping) or not block:
        return False, False, math.inf, -math.inf
    close = block.get("close", block)
    if not isinstance(close, Mapping):
        return False, False, math.inf, -math.inf
    angles: list[float] = []
    inward: list[float] = []
    per_finger = close.get("per_finger")
    if isinstance(per_finger, Mapping):
        for finger in ("thumb", "index", "mid"):
            value = per_finger.get(finger, {})
            if isinstance(value, Mapping):
                angles.append(
                    _finite_metric(
                        value.get(
                            "angle_p95_deg",
                            value.get(
                                "p95_angle_deg",
                                value.get("angle_max_deg", value.get("max_angle_deg")),
                            ),
                        ),
                        math.inf,
                    )
                )
                inward.append(
                    _finite_metric(
                        value.get("minimum_inward_speed_m_s", value.get("inward_speed_m_s")),
                        -math.inf,
                    )
                )
    maximum = _finite_metric(
        close.get(
            "max_p95_angle_deg",
            close.get("worst_p95_angle_deg", close.get("p95_angle_deg")),
        ),
        max(angles, default=math.inf),
    )
    minimum_inward = _finite_metric(
        close.get("worst_minimum_inward_speed_m_s"),
        min(inward, default=-math.inf),
    )
    available = math.isfinite(maximum) and math.isfinite(minimum_inward)
    closure_checks = (
        "v8_closure_alignment_trace_matches_vectors",
        "closure_alignment_valid_for_all_fingers",
        "closure_alignment_p95_within_limit",
        "closure_inward_speed_positive",
    )
    if isinstance(checks, Mapping) and all(name in checks for name in closure_checks):
        passed = all(bool(checks[name]) for name in closure_checks)
    else:
        passed = bool(
            close.get(
                "passed",
                available and maximum <= 30.0 + 1e-12 and minimum_inward > 0.0,
            )
        )
    return available, passed, maximum, minimum_inward


def _motion_rank_metrics(
    metrics: Mapping[str, Any], checks: Mapping[str, Any] | None = None
) -> tuple[bool, bool, float]:
    block = metrics.get("motion_smoothness")
    if not isinstance(block, Mapping) or not block:
        return False, False, math.inf
    values = {
        "lateral": _finite_metric(
            block.get(
                "operation_max_lateral_displacement_m",
                block.get("lateral_displacement_m", block.get("max_lateral_displacement_m")),
            ),
            math.inf,
        ),
        "orientation": _finite_metric(
            block.get("operation_max_orientation_drift_deg"), math.inf
        ),
        "backtrack": _finite_metric(
            block.get(
                "operation_cumulative_height_backtrack_m",
                block.get("cumulative_height_backtrack_m"),
            ),
            math.inf,
        ),
        "downward_duty": _finite_metric(
            block.get("operation_downward_speed_duty"), math.inf
        ),
        "speed": _finite_metric(
            block.get(
                "operation_peak_filtered_upward_speed_m_s",
                block.get("peak_filtered_upward_speed_m_s"),
            ),
            math.inf,
        ),
        "acceleration": _finite_metric(
            block.get(
                "operation_peak_abs_filtered_acceleration_m_s2",
                block.get("peak_abs_filtered_acceleration_m_s2"),
            ),
            math.inf,
        ),
        "jerk": _finite_metric(
            block.get(
                "operation_peak_abs_filtered_jerk_m_s3",
                block.get("peak_abs_filtered_jerk_m_s3"),
            ),
            math.inf,
        ),
        "hold_entry_speed": _finite_metric(
            block.get("operation_hold_entry_linear_speed_m_s"), math.inf
        ),
    }
    available = all(math.isfinite(value) for value in values.values())
    normalized = max(
        values["lateral"] / 0.002,
        values["orientation"] / 10.0,
        values["backtrack"] / 0.0002,
        values["downward_duty"] / 0.02,
        values["speed"] / 0.020,
        values["acceleration"] / 0.12,
        values["jerk"] / 2.5,
        values["hold_entry_speed"] / 0.005,
    )
    smooth_checks = (
        "smooth_motion_event_sequence_valid",
        "smooth_motion_filter_window_available",
        "smooth_motion_cumulative_backtrack_within_limit",
        "smooth_motion_downward_speed_duty_within_limit",
        "smooth_motion_peak_upward_speed_within_limit",
        "smooth_motion_acceleration_within_limit",
        "smooth_motion_jerk_within_limit",
        "smooth_motion_hold_entry_speed_within_limit",
        "smooth_motion_lateral_displacement_within_limit",
        "smooth_motion_orientation_drift_within_limit",
    )
    if isinstance(checks, Mapping) and all(name in checks for name in smooth_checks):
        passed = all(bool(checks[name]) for name in smooth_checks)
    else:
        passed = bool(
            block.get(
                "passed",
                block.get("hard_pass", available and normalized <= 1.0 + 1e-12),
            )
        )
    return available, passed, normalized


def candidate_rank_evidence(record: Mapping[str, Any]) -> dict[str, Any]:
    summary = record.get("summary", {})
    metrics = summary.get("metrics", {}) if isinstance(summary, Mapping) else {}
    checks = summary.get("checks", {}) if isinstance(summary, Mapping) else {}
    if not isinstance(metrics, Mapping):
        metrics = {}
    if not isinstance(checks, Mapping):
        checks = {}
    closure_available, closure_passed, closure_angle, inward_speed = (
        _closure_rank_metrics(metrics, checks)
    )
    motion_available, motion_passed, motion_score = _motion_rank_metrics(metrics, checks)
    pose = metrics.get("pose_preservation", {})
    if not isinstance(pose, Mapping):
        pose = {}
    translation = _finite_metric(pose.get("max_translation_m"), math.inf)
    orientation = _finite_metric(pose.get("max_orientation_drift_deg"), math.inf)
    pose_margin = min(
        (0.0005 - translation) / 0.0005,
        (1.0 - orientation) / 1.0,
    )
    gate_steps = int(
        _finite_metric(
            metrics.get("verify_max_consecutive_all_gate_steps", record.get("all_state_gate_steps")),
            0.0,
        )
    )
    # Rescue intentionally has zero manipulation delta.  Its job is only to
    # establish a pose-preserving, normal-aligned grasp; zero-operation jitter
    # must not turn a valid grasp into a rescue failure.
    evidence_complete = closure_available
    acquisition = bool(record.get("acquisition_success", False))
    preservation = bool(record.get("pose_preservation_success", False))
    rescue_success = bool(
        acquisition
        and preservation
        and evidence_complete
        and closure_passed
    )
    return {
        "evidence_complete": evidence_complete,
        "closure_alignment_available": closure_available,
        "motion_smoothness_available": motion_available,
        "closure_alignment_passed": closure_passed,
        "motion_smoothness_passed": motion_passed,
        "max_closure_p95_angle_deg": closure_angle,
        "minimum_closure_inward_speed_m_s": inward_speed,
        "motion_normalized_worst": motion_score,
        "pose_min_normalized_margin": pose_margin,
        "verify_gate_steps": gate_steps,
        "rescue_success": rescue_success,
    }


def rescue_candidate_rank(record: Mapping[str, Any]) -> tuple[Any, ...]:
    evidence = candidate_rank_evidence(record)
    summary = record.get("summary", {})
    metrics = summary.get("metrics", {}) if isinstance(summary, Mapping) else {}
    force = _finite_metric(
        metrics.get("peak_total_distal_contact_force_n")
        if isinstance(metrics, Mapping)
        else None,
        math.inf,
    )
    saturation = _finite_metric(
        metrics.get("actuator_saturation_fraction")
        if isinstance(metrics, Mapping)
        else None,
        math.inf,
    )
    return (
        not evidence["rescue_success"],
        not evidence["evidence_complete"],
        not bool(record.get("acquisition_success", False)),
        not bool(record.get("pose_preservation_success", False)),
        float(evidence["max_closure_p95_angle_deg"]),
        -int(evidence["verify_gate_steps"]),
        -float(evidence["pose_min_normalized_margin"]),
        force,
        saturation,
        int(record.get("candidate_id", 2**63 - 1)),
    )


def rank_rescue_results(
    records: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    materialized = [copy.deepcopy(dict(record)) for record in records]
    materialized.sort(key=rescue_candidate_rank)
    return tuple(materialized)


def manipulation_delta_bounds(
    config: Mapping[str, Any],
) -> dict[str, tuple[float, float]]:
    """Return the registered eight-actuator manipulation search bounds."""

    from ..experiment import resolve_experiment

    raw = resolve_experiment(config).search_bounds.manipulation_delta_rad
    if raw is None or set(raw) != set(ACTIVE_ACTUATORS):
        raise ValueError("v8 experiment has no complete manipulation delta bounds")
    return {
        name: (float(raw[name][0]), float(raw[name][1]))
        for name in ACTIVE_ACTUATORS
    }


def _ctrlrange_safe_manipulation_delta_bounds(
    config: Mapping[str, Any],
) -> dict[str, tuple[float, float]]:
    """Intersect registered relative deltas with absolute model limits."""

    raw = manipulation_delta_bounds(config)
    control = config.get("control", {})
    targets = (
        control.get("grasp_targets_rad", {})
        if isinstance(control, Mapping)
        else {}
    )
    if not isinstance(targets, Mapping) or set(targets) != set(ACTIVE_ACTUATORS):
        raise ValueError("v8 grasp target must contain exactly eight actuators")

    result: dict[str, tuple[float, float]] = {}
    for name in ACTIVE_ACTUATORS:
        target = float(targets[name])
        absolute_lower, absolute_upper = ACTIVE_ACTUATOR_POSITION_LIMITS_RAD[name]
        if not absolute_lower <= target <= absolute_upper:
            raise ValueError(
                f"{name} grasp target {target} is outside model position limits "
                f"[{absolute_lower}, {absolute_upper}]"
            )
        lower = max(raw[name][0], absolute_lower - target)
        upper = min(raw[name][1], absolute_upper - target)
        if lower > upper + 1e-12:
            raise ValueError(f"{name} grasp target leaves no feasible manipulation delta")
        result[name] = (lower, upper)
    return result


def _clip_delta(
    values: Mapping[str, Any], bounds: Mapping[str, Sequence[float]]
) -> dict[str, float]:
    if set(values) != set(ACTIVE_ACTUATORS):
        raise ValueError("manipulation delta must contain exactly eight actuators")
    return {
        name: float(
            np.clip(
                _finite_metric(values[name], 0.0),
                float(bounds[name][0]),
                float(bounds[name][1]),
            )
        )
        for name in ACTIVE_ACTUATORS
    }


def _lift_source_record(
    config: Mapping[str, Any],
    *,
    source_candidate_id: int,
    source_family_id: str = "direct_config",
    source_trajectory_id: str = "direct_config",
    tier: str = "lift",
    source_path: str | None = None,
) -> dict[str, Any]:
    if int(config.get("schema_version", 0)) != 8 or config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("lift source must be a registered schema-v8 configuration")
    validate_config(dict(config))
    return {
        "pose_id": pose_id(config),
        "controller_id": controller_id(config),
        "tier": str(tier),
        "source_candidate_id": int(source_candidate_id),
        "source_family_id": str(source_family_id),
        "source_trajectory_id": str(source_trajectory_id),
        "source_config": source_path,
        "source_config_sha256": canonical_sha256(config),
        "edge_m": float(config["cube"]["edge_m"]),
        "thumb_target_rad": float(
            config["control"]["grasp_targets_rad"][THUMB_BEND_ACTUATOR]
        ),
        "config": copy.deepcopy(dict(config)),
    }


def materialize_lift_candidate(
    base_config: Mapping[str, Any],
    source_record: Mapping[str, Any],
    delta_rad: Mapping[str, Any],
    *,
    candidate_id_value: int,
    stage: str,
    parent_candidate_id: int | None = None,
    search_metadata: Mapping[str, Any] | None = None,
    validator: Callable[[dict[str, Any]], None] | None = validate_config,
) -> dict[str, Any]:
    """Change only a resolved v8 grasp's manipulation delta and metadata."""

    if int(base_config.get("schema_version", 0)) != 8 or base_config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("lift candidate base must be the registered schema-v8 experiment")
    bounds = _ctrlrange_safe_manipulation_delta_bounds(base_config)
    candidate = copy.deepcopy(dict(base_config))
    candidate["control"]["manipulation_delta_rad"] = _clip_delta(delta_rad, bounds)
    candidate.pop("run_context", None)
    previous_metadata = candidate.get("candidate_metadata", {})
    metadata: dict[str, Any] = {
        "campaign_kind": CAMPAIGN_KIND,
        "candidate_id": int(candidate_id_value),
        "stage": str(stage),
        "pose_id": str(source_record.get("pose_id", pose_id(base_config))),
        "source_candidate_id": int(source_record.get("source_candidate_id", -1)),
        "source_family_id": str(source_record.get("source_family_id", "unknown")),
        "source_trajectory_id": str(
            source_record.get("source_trajectory_id", "unknown")
        ),
        "source_config_sha256": str(
            source_record.get("source_config_sha256", canonical_sha256(base_config))
        ),
        "manipulation_response_search": True,
        "cube_pose_sampled": False,
        "free_cube_pose_reset_during_run": False,
    }
    if isinstance(previous_metadata, Mapping):
        metadata["parent_controller_id"] = str(
            previous_metadata.get("controller_id", controller_id(base_config))
        )
    if parent_candidate_id is not None:
        metadata["parent_candidate_id"] = int(parent_candidate_id)
    if search_metadata:
        metadata["lift_search"] = copy.deepcopy(dict(search_metadata))
    candidate["candidate_metadata"] = metadata
    metadata["controller_id"] = controller_id(candidate)
    for field in POSE_CONTEXT_FIELDS:
        if candidate.get(field) != base_config.get(field):
            raise ValueError(f"lift candidate changed frozen source field {field}")
    if validator is not None:
        validator(candidate)
    return candidate


def _lift_candidate_record(
    config: Mapping[str, Any], source_record: Mapping[str, Any]
) -> dict[str, Any]:
    metadata = config["candidate_metadata"]
    return {
        "campaign_kind": CAMPAIGN_KIND,
        "candidate_id": int(metadata["candidate_id"]),
        "candidate_sha256": canonical_sha256(config),
        "stage": str(metadata["stage"]),
        "pose_id": str(metadata["pose_id"]),
        "controller_id": str(metadata["controller_id"]),
        "tier": str(source_record.get("tier", "lift")),
        "source_candidate_id": int(source_record.get("source_candidate_id", -1)),
        "source_family_id": str(source_record.get("source_family_id", "unknown")),
        "source_trajectory_id": str(
            source_record.get("source_trajectory_id", "unknown")
        ),
        "edge_m": float(config["cube"]["edge_m"]),
        "thumb_target_rad": float(
            config["control"]["grasp_targets_rad"][THUMB_BEND_ACTUATOR]
        ),
        "config": copy.deepcopy(dict(config)),
    }


def generate_lift_probe_candidates(
    base_config: Mapping[str, Any],
    source_record: Mapping[str, Any] | None = None,
    *,
    pose_index: int,
    epsilon_rad: float = 0.02,
    validator: Callable[[dict[str, Any]], None] | None = validate_config,
) -> tuple[dict[str, Any], ...]:
    """Generate zero and +/- single-actuator probes (always exactly 17)."""

    if not math.isfinite(epsilon_rad) or epsilon_rad <= 0.0:
        raise ValueError("epsilon_rad must be positive and finite")
    source = (
        copy.deepcopy(dict(source_record))
        if source_record is not None
        else _lift_source_record(base_config, source_candidate_id=-1)
    )
    bounds = _ctrlrange_safe_manipulation_delta_bounds(base_config)
    zero = {name: 0.0 for name in ACTIVE_ACTUATORS}
    specifications: list[tuple[str, str | None, int, dict[str, float]]] = [
        ("zero", None, 0, zero)
    ]
    for name in ACTIVE_ACTUATORS:
        for direction in (-1, 1):
            values = dict(zero)
            values[name] = float(
                np.clip(direction * epsilon_rad, bounds[name][0], bounds[name][1])
            )
            specifications.append(("single_actuator", name, direction, values))
    records = []
    for local_index, (kind, actuator, direction, values) in enumerate(specifications):
        identifier = _LIFT_PROBE_BASE + int(pose_index) * _ID_STRIDE + local_index
        applied = 0.0 if actuator is None else float(values[actuator])
        config = materialize_lift_candidate(
            base_config,
            source,
            values,
            candidate_id_value=identifier,
            stage="lift_probe",
            parent_candidate_id=int(source.get("source_candidate_id", -1)),
            search_metadata={
                "kind": kind,
                "actuator": actuator,
                "direction": int(direction),
                "requested_step_rad": float(direction * epsilon_rad),
                "applied_step_rad": applied,
            },
            validator=validator,
        )
        records.append(_lift_candidate_record(config, source))
    return tuple(records)


def _trace_scalar_int(trace: Mapping[str, Any], key: str) -> int:
    value = np.asarray(trace[key])
    if value.size != 1:
        raise ValueError(f"{key} must be scalar")
    return int(value.reshape(-1)[0])


def _quat_rotation_vector_wxyz(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    if start.shape != (4,) or end.shape != (4,):
        raise ValueError("cube quaternions must have shape (4,)")
    start = start / np.linalg.norm(start)
    end = end / np.linalg.norm(end)
    sw, sx, sy, sz = start
    inverse = np.asarray((sw, -sx, -sy, -sz), dtype=np.float64)
    ew, ex, ey, ez = end
    iw, ix, iy, iz = inverse
    relative = np.asarray(
        (
            ew * iw - ex * ix - ey * iy - ez * iz,
            ew * ix + ex * iw + ey * iz - ez * iy,
            ew * iy - ex * iz + ey * iw + ez * ix,
            ew * iz + ex * iy - ey * ix + ez * iw,
        ),
        dtype=np.float64,
    )
    relative /= np.linalg.norm(relative)
    if relative[0] < 0.0:
        relative *= -1.0
    vector_norm = float(np.linalg.norm(relative[1:]))
    if vector_norm <= 1e-15:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(vector_norm, float(np.clip(relative[0], -1.0, 1.0)))
    return relative[1:] * (angle / vector_norm)


def manipulation_response_from_trace(
    trace_or_path: Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    """Measure the 6-D object response over the commanded MANIPULATE phase."""

    close_archive = False
    if isinstance(trace_or_path, Mapping):
        trace = trace_or_path
    else:
        trace = np.load(Path(trace_or_path), allow_pickle=False)
        close_archive = True
    try:
        positions = np.asarray(trace["cube_pos"], dtype=np.float64)
        quaternions = np.asarray(trace["cube_quat"], dtype=np.float64)
        start = _trace_scalar_int(trace, "manipulation_start_step")
        end = _trace_scalar_int(trace, "manipulation_end_step")
        valid = bool(
            positions.ndim == 2
            and positions.shape[1] == 3
            and quaternions.shape == (positions.shape[0], 4)
            and 0 <= start <= end < positions.shape[0]
            and np.isfinite(positions).all()
            and np.isfinite(quaternions).all()
        )
        if not valid:
            return {
                "available": False,
                "reason": "manipulation_not_executed_or_trace_invalid",
                "manipulation_start_step": int(start),
                "manipulation_end_step": int(end),
            }
        baseline = max(0, start - 1)
        translation = positions[end] - positions[baseline]
        rotation = _quat_rotation_vector_wxyz(
            quaternions[baseline], quaternions[end]
        )
        response = np.concatenate((translation, rotation))
        return {
            "available": True,
            "baseline_step": int(baseline),
            "manipulation_start_step": int(start),
            "manipulation_end_step": int(end),
            "translation_world_m": translation.tolist(),
            "rotation_vector_world_rad": rotation.tolist(),
            "response_6d": response.tolist(),
            "upward_displacement_m": float(translation[2]),
            "lateral_displacement_m": float(np.linalg.norm(translation[:2])),
            "orientation_change_deg": float(np.degrees(np.linalg.norm(rotation))),
        }
    finally:
        if close_archive:
            trace.close()


def _probe_metadata(record: Mapping[str, Any]) -> Mapping[str, Any]:
    config = record.get("config", {})
    metadata = config.get("candidate_metadata", {}) if isinstance(config, Mapping) else {}
    lift = metadata.get("lift_search", {}) if isinstance(metadata, Mapping) else {}
    return lift if isinstance(lift, Mapping) else {}


def fit_manipulation_response_model(
    probe_results: Sequence[Mapping[str, Any]],
    bounds: Mapping[str, Sequence[float]],
    *,
    target_response_6d: Sequence[float] = _LIFT_RESPONSE_TARGET,
    ridge: float = 1e-4,
) -> dict[str, Any]:
    """Fit a 6x8 finite-difference response and solve a bounded ridge target."""

    if set(bounds) != set(ACTIVE_ACTUATORS):
        raise ValueError("response model bounds must contain eight actuators")
    target = np.asarray(target_response_6d, dtype=np.float64)
    if target.shape != (6,) or not np.isfinite(target).all():
        raise ValueError("target_response_6d must contain six finite values")
    zero_record = next(
        (value for value in probe_results if _probe_metadata(value).get("kind") == "zero"),
        None,
    )
    if zero_record is None:
        raise ValueError("probe set has no zero response")
    zero_response = zero_record.get("manipulation_response", {})
    if not isinstance(zero_response, Mapping) or not zero_response.get("available", False):
        bias = np.zeros(6, dtype=np.float64)
        zero_available = False
    else:
        bias = np.asarray(zero_response["response_6d"], dtype=np.float64)
        zero_available = bias.shape == (6,) and bool(np.isfinite(bias).all())
        if not zero_available:
            bias = np.zeros(6, dtype=np.float64)
    matrix = np.zeros((6, len(ACTIVE_ACTUATORS)), dtype=np.float64)
    columns = []
    for column, name in enumerate(ACTIVE_ACTUATORS):
        directional: dict[int, tuple[float, np.ndarray]] = {}
        for result in probe_results:
            metadata = _probe_metadata(result)
            if metadata.get("actuator") != name:
                continue
            response = result.get("manipulation_response", {})
            if not isinstance(response, Mapping) or not response.get("available", False):
                continue
            vector = np.asarray(response.get("response_6d"), dtype=np.float64)
            if vector.shape != (6,) or not np.isfinite(vector).all():
                continue
            direction = int(metadata.get("direction", 0))
            step = float(metadata.get("applied_step_rad", 0.0))
            if direction in (-1, 1) and abs(step) > 1e-15:
                directional[direction] = (step, vector)
        method = "missing"
        if -1 in directional and 1 in directional:
            negative_step, negative = directional[-1]
            positive_step, positive = directional[1]
            denominator = positive_step - negative_step
            if abs(denominator) > 1e-15:
                matrix[:, column] = (positive - negative) / denominator
                method = "central"
        if method == "missing" and directional:
            direction = 1 if 1 in directional else -1
            step, response = directional[direction]
            matrix[:, column] = (response - bias) / step
            method = "forward" if direction == 1 else "backward"
        columns.append(
            {
                "actuator": name,
                "method": method,
                "available": method != "missing",
                "column_norm": float(np.linalg.norm(matrix[:, column])),
            }
        )
    scales = np.asarray(
        (0.002, 0.002, 0.011, math.radians(10.0), math.radians(10.0), math.radians(10.0)),
        dtype=np.float64,
    )
    weighted = matrix / scales[:, None]
    residual_target = (target - bias) / scales
    lower = np.asarray([float(bounds[name][0]) for name in ACTIVE_ACTUATORS])
    upper = np.asarray([float(bounds[name][1]) for name in ACTIVE_ACTUATORS])
    hessian = weighted.T @ weighted + float(ridge) * np.eye(len(ACTIVE_ACTUATORS))
    gradient_offset = weighted.T @ residual_target
    try:
        solution = np.linalg.solve(hessian, gradient_offset)
    except np.linalg.LinAlgError:
        solution = np.linalg.lstsq(hessian, gradient_offset, rcond=None)[0]
    solution = np.clip(solution, lower, upper)
    lipschitz = max(float(np.linalg.eigvalsh(hessian)[-1]), 1e-12)
    for _ in range(256):
        gradient = hessian @ solution - gradient_offset
        updated = np.clip(solution - gradient / lipschitz, lower, upper)
        if float(np.max(np.abs(updated - solution))) <= 1e-12:
            solution = updated
            break
        solution = updated
    predicted = bias + matrix @ solution
    available_columns = sum(item["available"] for item in columns)
    condition_number = (
        float(np.linalg.cond(weighted)) if np.any(weighted) else math.inf
    )
    return {
        "available": bool(zero_available and available_columns > 0),
        "zero_response_available": bool(zero_available),
        "available_column_count": int(available_columns),
        "columns": columns,
        "jacobian_6x8": matrix.tolist(),
        "bias_response_6d": bias.tolist(),
        "target_response_6d": target.tolist(),
        "solution_delta_rad": {
            name: float(solution[index]) for index, name in enumerate(ACTIVE_ACTUATORS)
        },
        "predicted_response_6d": predicted.tolist(),
        "matrix_rank": int(np.linalg.matrix_rank(weighted)),
        "condition_number": condition_number if math.isfinite(condition_number) else None,
        "weighted_residual_norm": float(np.linalg.norm((predicted - target) / scales)),
    }


def generate_lift_trust_candidates(
    base_config: Mapping[str, Any],
    source_record: Mapping[str, Any],
    response_model: Mapping[str, Any],
    *,
    pose_index: int,
    count: int = 64,
    seed: int = DEFAULT_SEED,
    validator: Callable[[dict[str, Any]], None] | None = validate_config,
) -> tuple[dict[str, Any], ...]:
    """Generate deterministic bounded response-solution/trust-region candidates."""

    if count <= 0:
        raise ValueError("trust candidate count must be positive")
    bounds = _ctrlrange_safe_manipulation_delta_bounds(base_config)
    zero = {name: 0.0 for name in ACTIVE_ACTUATORS}
    solved = _clip_delta(
        response_model.get("solution_delta_rad", zero), bounds
    )
    historical = _clip_delta(_HISTORICAL_LIFT_DELTA_RAD, bounds)
    anchors: list[tuple[str, dict[str, float]]] = [
        ("response_solution", solved),
        ("validated_historical", historical),
        ("zero_control", zero),
    ]
    for fraction in (0.25, 0.50, 0.75):
        anchors.append(
            (
                f"response_historical_blend_{fraction:.2f}",
                {
                    name: (1.0 - fraction) * solved[name] + fraction * historical[name]
                    for name in ACTIVE_ACTUATORS
                },
            )
        )
    selected = anchors[:count]
    remaining = count - len(selected)
    units = _latin_hypercube(
        remaining,
        len(ACTIVE_ACTUATORS),
        int(np.random.SeedSequence([seed, pose_index, 8_700_001]).generate_state(1)[0]),
    )
    for unit in units:
        selected.append(
            (
                "response_trust_latin_hypercube",
                {
                    name: float(
                        np.clip(
                            solved[name]
                            + float(unit[index])
                            * 0.25
                            * (float(bounds[name][1]) - float(bounds[name][0])),
                            bounds[name][0],
                            bounds[name][1],
                        )
                    )
                    for index, name in enumerate(ACTIVE_ACTUATORS)
                },
            )
        )
    records = []
    for local_index, (anchor_kind, delta) in enumerate(selected):
        identifier = _LIFT_TRUST_BASE + int(pose_index) * _ID_STRIDE + local_index
        config = materialize_lift_candidate(
            base_config,
            source_record,
            delta,
            candidate_id_value=identifier,
            stage="lift_trust",
            parent_candidate_id=int(source_record.get("source_candidate_id", -1)),
            search_metadata={
                "kind": "trust_region",
                "anchor_kind": anchor_kind,
                "response_model_available": bool(response_model.get("available", False)),
            },
            validator=validator,
        )
        records.append(_lift_candidate_record(config, source_record))
    return tuple(records)


def lift_candidate_evidence(record: Mapping[str, Any]) -> dict[str, Any]:
    summary = record.get("summary", {})
    metrics = summary.get("metrics", {}) if isinstance(summary, Mapping) else {}
    checks = summary.get("checks", {}) if isinstance(summary, Mapping) else {}
    stage = summary.get("stage_status", {}) if isinstance(summary, Mapping) else {}
    if not isinstance(metrics, Mapping):
        metrics = {}
    if not isinstance(checks, Mapping):
        checks = {}
    if not isinstance(stage, Mapping):
        stage = {}
    closure_available, closure_passed, angle, inward = _closure_rank_metrics(metrics, checks)
    motion_available, motion_passed, motion_score = _motion_rank_metrics(metrics, checks)
    median_lift = _finite_metric(
        metrics.get("operation_median_lift_m", metrics.get("median_lift_m")),
        -math.inf,
    )
    minimum_lift = _finite_metric(
        metrics.get("operation_minimum_lift_m", metrics.get("minimum_lift_m")),
        -math.inf,
    )
    response = record.get("manipulation_response", {})
    response_available = bool(
        isinstance(response, Mapping) and response.get("available", False)
    )
    full = bool(stage.get("full_success", summary.get("passed", False)))
    evidence_complete = bool(
        closure_available and motion_available and response_available
    )
    lift_success = bool(
        full
        and evidence_complete
        and closure_passed
        and motion_passed
        and median_lift >= 0.010 - 1e-12
        and minimum_lift >= 0.008 - 1e-12
    )
    failed_checks = summary.get("failed_checks", [])
    return {
        "evidence_complete": evidence_complete,
        "response_available": response_available,
        "closure_alignment_available": closure_available,
        "closure_alignment_passed": closure_passed,
        "motion_smoothness_available": motion_available,
        "motion_smoothness_passed": motion_passed,
        "max_closure_p95_angle_deg": angle,
        "minimum_closure_inward_speed_m_s": inward,
        "motion_normalized_worst": motion_score,
        "median_lift_m": median_lift,
        "minimum_lift_m": minimum_lift,
        "median_lift_margin_m": median_lift - 0.010,
        "minimum_lift_margin_m": minimum_lift - 0.008,
        "grasp_success": bool(stage.get("grasp_success", False)),
        "manipulation_success": bool(stage.get("manipulation_success", False)),
        "full_success": full,
        "failed_check_count": len(failed_checks) if isinstance(failed_checks, list) else 10**6,
        "lift_success": lift_success,
    }


def lift_candidate_rank(record: Mapping[str, Any]) -> tuple[Any, ...]:
    evidence = lift_candidate_evidence(record)
    response = record.get("manipulation_response", {})
    lateral = _finite_metric(
        response.get("lateral_displacement_m") if isinstance(response, Mapping) else None,
        math.inf,
    )
    orientation = _finite_metric(
        response.get("orientation_change_deg") if isinstance(response, Mapping) else None,
        math.inf,
    )
    return (
        not evidence["lift_success"],
        not evidence["evidence_complete"],
        not evidence["grasp_success"],
        not evidence["manipulation_success"],
        -min(
            float(evidence["median_lift_margin_m"]) / 0.010,
            float(evidence["minimum_lift_margin_m"]) / 0.008,
        ),
        int(evidence["failed_check_count"]),
        float(evidence["motion_normalized_worst"]),
        lateral / 0.002,
        orientation / 10.0,
        float(evidence["max_closure_p95_angle_deg"]),
        int(record.get("candidate_id", 2**63 - 1)),
    )


def rank_lift_results(
    records: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    materialized = [copy.deepcopy(dict(record)) for record in records]
    materialized.sort(key=lift_candidate_rank)
    return tuple(materialized)


def _candidate_classification(
    stage: str, *, rescue_success: bool, lift_success: bool
) -> str:
    if stage.startswith("lift_"):
        if lift_success:
            return (
                "validated_normal_aligned_smooth_vertical_lift"
                if stage == "lift_exact"
                else "normal_aligned_smooth_vertical_lift_search_pass"
            )
        return "normal_aligned_smooth_vertical_lift_near_miss"
    return (
        "validated_normal_aligned_control_rescue"
        if rescue_success
        else "normal_aligned_control_rescue_near_miss"
    )


def _evaluate_candidate_job(job: Mapping[str, Any]) -> dict[str, Any]:
    from ..simulation import run_simulation

    output = Path(str(job["output_directory"]))
    if output.exists():
        raise FileExistsError(f"candidate output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=f".{output.name}.") as staging:
        staging_path = Path(staging)
        config_path = staging_path / "resolved_config.json"
        trace_path = staging_path / "trace.npz"
        write_json(config_path, job["config"])
        summary = run_simulation(copy.deepcopy(dict(job["config"])), trace_path=trace_path)
        manipulation_response = manipulation_response_from_trace(trace_path)
        check_record = {"candidate_id": int(job["candidate_id"]), "summary": summary}
        provisional = {
            **copy.deepcopy(dict(job)),
            "summary": summary,
            "acquisition_success": acquisition_succeeded(check_record),
            "pose_preservation_success": pose_preservation_succeeded(check_record),
        }
        provisional.pop("config", None)
        provisional.pop("output_directory", None)
        provisional["manipulation_response"] = manipulation_response
        evidence = candidate_rank_evidence(provisional)
        lift_evidence = lift_candidate_evidence(provisional)
        stage = str(job["stage"])
        payload = {
            "candidate_result_schema_version": CANDIDATE_RESULT_SCHEMA_VERSION,
            "complete": True,
            "campaign_kind": CAMPAIGN_KIND,
            "stage": stage,
            "candidate_id": int(job["candidate_id"]),
            "candidate_sha256": str(job["candidate_sha256"]),
            "pose_id": str(job["pose_id"]),
            "controller_id": str(job["controller_id"]),
            "tier": str(job["tier"]),
            "source_candidate_id": int(job["source_candidate_id"]),
            "source_family_id": str(job["source_family_id"]),
            "source_trajectory_id": str(job["source_trajectory_id"]),
            "edge_m": float(job["edge_m"]),
            "thumb_target_rad": float(job["thumb_target_rad"]),
            "acquisition_success": bool(provisional["acquisition_success"]),
            "pose_preservation_success": bool(provisional["pose_preservation_success"]),
            "rescue_success": bool(evidence["rescue_success"]),
            "lift_success": bool(lift_evidence["lift_success"]),
            "classification": _candidate_classification(
                stage,
                rescue_success=bool(evidence["rescue_success"]),
                lift_success=bool(lift_evidence["lift_success"]),
            ),
            "rank_evidence": evidence,
            "lift_rank_evidence": lift_evidence,
            "manipulation_response": manipulation_response,
            "summary": summary,
            "artifacts": {
                "resolved_config": "resolved_config.json",
                "trace": "trace.npz",
                "sha256": {
                    "resolved_config": file_sha256(config_path),
                    "trace": file_sha256(trace_path),
                },
            },
        }
        write_json(staging_path / "result.json", payload)
        staging_path.rename(output)
    return {
        **payload,
        "config": copy.deepcopy(dict(job["config"])),
        "artifact_directory": str(job["artifact_directory"]),
        "reused": False,
    }


def run_candidate_jobs(
    jobs: Sequence[Mapping[str, Any]], workers: int
) -> tuple[dict[str, Any], ...]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    if not jobs:
        return ()
    if workers == 1:
        results = [_evaluate_candidate_job(job) for job in jobs]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            results = list(executor.map(_evaluate_candidate_job, jobs, chunksize=1))
    results.sort(key=lambda value: int(value["candidate_id"]))
    return tuple(results)


def _load_reusable_candidate(job: Mapping[str, Any]) -> dict[str, Any] | None:
    directory = Path(str(job["output_directory"]))
    if not directory.exists():
        return None
    result_path = directory / "result.json"
    config_path = directory / "resolved_config.json"
    trace_path = directory / "trace.npz"
    if not all(path.is_file() for path in (result_path, config_path, trace_path)):
        raise RuntimeError(f"incomplete rescue candidate directory: {directory}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("complete") is not True:
        raise RuntimeError(f"rescue result is incomplete: {result_path}")
    if int(result.get("candidate_id", -1)) != int(job["candidate_id"]):
        raise RuntimeError(f"rescue candidate ID mismatch: {result_path}")
    if result.get("candidate_sha256") != job["candidate_sha256"]:
        raise RuntimeError(f"rescue semantic hash mismatch: {result_path}")
    persisted = json.loads(config_path.read_text(encoding="utf-8"))
    if canonical_sha256(persisted) != job["candidate_sha256"]:
        raise RuntimeError(f"persisted rescue config changed: {config_path}")
    hashes = result.get("artifacts", {}).get("sha256", {})
    if hashes.get("resolved_config") != file_sha256(config_path):
        raise RuntimeError(f"rescue config file hash mismatch: {config_path}")
    if hashes.get("trace") != file_sha256(trace_path):
        raise RuntimeError(f"rescue trace hash mismatch: {trace_path}")
    return {
        **result,
        "config": copy.deepcopy(dict(job["config"])),
        "artifact_directory": str(job["artifact_directory"]),
        "reused": True,
    }


def _run_or_resume(
    candidates: Sequence[Mapping[str, Any]],
    output: Path,
    *,
    workers: int,
    resume: bool,
    executor: CandidateExecutor,
) -> tuple[dict[str, Any], ...]:
    jobs = []
    for candidate in candidates:
        relative = Path("candidates") / f"candidate_{int(candidate['candidate_id'])}"
        jobs.append(
            {
                **copy.deepcopy(dict(candidate)),
                "artifact_directory": str(relative),
                "output_directory": str(output / relative),
            }
        )
    complete: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for job in jobs:
        reusable = _load_reusable_candidate(job) if resume else None
        (complete if reusable is not None else pending).append(reusable or job)
    executed = tuple(executor(tuple(pending), workers)) if pending else ()
    expected = {int(job["candidate_id"]): job for job in pending}
    if {int(value.get("candidate_id", -1)) for value in executed} != set(expected):
        raise RuntimeError("rescue executor did not preserve candidate IDs")
    for value in executed:
        result = copy.deepcopy(dict(value))
        job = expected[int(result["candidate_id"])]
        if result.get("candidate_sha256") != job["candidate_sha256"]:
            raise RuntimeError("rescue executor rebound a different candidate config")
        result.setdefault("config", copy.deepcopy(job["config"]))
        complete.append(result)
    complete.sort(key=lambda value: int(value["candidate_id"]))
    return tuple(complete)


def _source_by_pose(
    pose_manifest: Mapping[str, Any],
) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    result = {}
    for raw in pose_manifest["poses"]:
        record = copy.deepcopy(dict(raw))
        config = load_config(record["source_config"])
        if pose_id(config) != record["pose_id"]:
            raise ValueError("pose manifest no longer matches its source config")
        result[str(record["pose_id"])] = (config, record)
    return result


def _select_local_parents(
    coarse: Sequence[Mapping[str, Any]], budget: RescueBudget
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    tier_a_by_pose: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    tier_b: list[Mapping[str, Any]] = []
    for value in coarse:
        if value["tier"] == "A":
            tier_a_by_pose[str(value["pose_id"])].append(value)
        else:
            tier_b.append(value)
    tier_a_parents: list[dict[str, Any]] = []
    for identifier in sorted(tier_a_by_pose):
        ranked = rank_rescue_results(tier_a_by_pose[identifier])
        tier_a_parents.extend(ranked[: budget.tier_a_local_seed_count_per_pose])
    tier_b_parents = rank_rescue_results(tier_b)[: budget.tier_b_global_local_seed_count]
    return tuple(tier_a_parents), tuple(tier_b_parents)


def _compact_result(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value.get(key))
        for key in (
            "candidate_id",
            "candidate_sha256",
            "stage",
            "pose_id",
            "controller_id",
            "tier",
            "source_candidate_id",
            "edge_m",
            "thumb_target_rad",
            "acquisition_success",
            "pose_preservation_success",
            "rescue_success",
            "classification",
            "rank_evidence",
            "artifact_directory",
        )
    }


def run_pose_rescue_campaign(
    campaign_results_path: str | Path = DEFAULT_V7_CAMPAIGN_RESULTS,
    template_path: str | Path = DEFAULT_V8_TEMPLATE,
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    workers: int = 1,
    resume: bool = False,
    seed: int = DEFAULT_SEED,
    executor: CandidateExecutor = run_candidate_jobs,
) -> dict[str, Any]:
    """Run the complete fixed-budget old-pose controller rescue stage."""

    if workers <= 0:
        raise ValueError("workers must be positive")
    output = Path(output_dir).expanduser().resolve()
    template_file = Path(template_path).expanduser().resolve()
    template = load_config(template_file)
    if int(template.get("schema_version", 0)) != 8 or template.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("--template must be the registered schema-v8 experiment")
    pose_manifest = build_pose_rescue_manifest(campaign_results_path)
    budget = FIXED_RESCUE_BUDGET
    if pose_manifest["selection"]["tier_counts"] != {
        "A": budget.tier_a_pose_count,
        "B": budget.tier_b_pose_count,
    }:
        raise RuntimeError("authenticated v7 tiers no longer match the fixed rescue budget")
    input_payload = {
        "campaign_schema_version": CAMPAIGN_SCHEMA_VERSION,
        "campaign_kind": CAMPAIGN_KIND,
        "experiment_id": EXPERIMENT_ID,
        "stage": "rescue",
        "seed": int(seed),
        "budget": asdict(budget),
        "pose_manifest_sha256": canonical_sha256(pose_manifest),
        "template_file_sha256": file_sha256(template_file),
        "template_semantic_sha256": canonical_sha256(template),
        "tuner_source_sha256": file_sha256(Path(__file__)),
    }
    input_sha = canonical_sha256(input_payload)
    campaign_manifest_path = output / "campaign_manifest.json"
    if output.exists():
        if not resume:
            raise FileExistsError(f"output directory already exists: {output}; pass --resume")
        if not campaign_manifest_path.is_file():
            raise RuntimeError("resume output has no campaign_manifest.json")
        existing = json.loads(campaign_manifest_path.read_text(encoding="utf-8"))
        if existing.get("campaign_input_sha256") != input_sha:
            raise RuntimeError("resume inputs do not match the existing rescue manifest")
    else:
        output.mkdir(parents=True)
        write_json(output / "pose_manifest.json", pose_manifest)
        write_json(
            campaign_manifest_path,
            {
                **input_payload,
                "campaign_input_sha256": input_sha,
                "output_directory": str(output),
                "complete": False,
            },
        )
    source_by_pose = _source_by_pose(pose_manifest)
    coarse_candidates: list[dict[str, Any]] = []
    for pose_index, pose_record in enumerate(pose_manifest["poses"]):
        source, record = source_by_pose[str(pose_record["pose_id"])]
        count = (
            budget.tier_a_coarse_per_pose
            if record["tier"] == "A"
            else budget.tier_b_coarse_per_pose
        )
        coarse_candidates.extend(
            generate_pose_coarse_candidates(
                source,
                template,
                record,
                pose_index=pose_index,
                count=count,
                seed=seed,
            )
        )
    coarse_results = _run_or_resume(
        coarse_candidates,
        output,
        workers=workers,
        resume=resume,
        executor=executor,
    )
    tier_a_parents, tier_b_parents = _select_local_parents(coarse_results, budget)
    tier_a_local = generate_local_candidates(
        tier_a_parents,
        source_by_pose,
        template,
        count_per_parent=budget.tier_a_local_per_seed,
        tier="A",
        seed=seed,
    )
    tier_b_local = generate_local_candidates(
        tier_b_parents,
        source_by_pose,
        template,
        count_per_parent=budget.tier_b_local_per_seed,
        tier="B",
        seed=seed,
    )
    local_results = _run_or_resume(
        (*tier_a_local, *tier_b_local),
        output,
        workers=workers,
        resume=resume,
        executor=executor,
    )
    tier_a_pool = [
        value for value in (*coarse_results, *local_results) if value["tier"] == "A"
    ]
    exact_candidates = generate_exact_candidates(
        rank_rescue_results(tier_a_pool), count=budget.tier_a_exact_count
    )
    exact_results = _run_or_resume(
        exact_candidates,
        output,
        workers=workers,
        resume=resume,
        executor=executor,
    )
    all_results = tuple((*coarse_results, *local_results, *exact_results))
    if len(all_results) != budget.total:
        raise RuntimeError(
            f"rescue executed {len(all_results)} candidates; expected {budget.total}"
        )
    ranked = rank_rescue_results(all_results)
    by_pose: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for result in all_results:
        by_pose[str(result["pose_id"])].append(result)
    pose_reports = []
    for pose_record in pose_manifest["poses"]:
        values = rank_rescue_results(by_pose[str(pose_record["pose_id"])])
        pose_reports.append(
            {
                "pose_id": pose_record["pose_id"],
                "tier": pose_record["tier"],
                "source_candidate_id": pose_record["source_candidate_id"],
                "candidate_count": len(values),
                "rescue_pass_count": sum(
                    bool(candidate_rank_evidence(value)["rescue_success"])
                    for value in values
                ),
                "best_candidate": _compact_result(values[0]),
            }
        )
    rescue_passes = [
        value for value in ranked if candidate_rank_evidence(value)["rescue_success"]
    ]
    report = {
        "campaign_schema_version": CAMPAIGN_SCHEMA_VERSION,
        "campaign_kind": CAMPAIGN_KIND,
        "experiment_id": EXPERIMENT_ID,
        "stage": "rescue",
        "complete": True,
        "campaign_input_sha256": input_sha,
        "workers": int(workers),
        "seed": int(seed),
        "budget": {
            **asdict(budget),
            "tier_a_total": budget.tier_a_total,
            "tier_b_total": budget.tier_b_total,
            "total": budget.total,
        },
        "candidate_count": len(all_results),
        "rescue_pass_count": len(rescue_passes),
        "rescued_pose_count": sum(
            any(candidate_rank_evidence(value)["rescue_success"] for value in values)
            for values in by_pose.values()
        ),
        "best_candidate": _compact_result(ranked[0]),
        "pose_results": pose_reports,
        "next_stage_required": True,
        "next_stage_reason": (
            "the authenticated old poses cover only the 1.25--1.30 rad thumb band; "
            "new-pose and manipulation-Jacobian stages remain required"
        ),
        "results": [_compact_result(value) for value in ranked],
    }
    write_json(output / "search_report.json", report)
    write_json(
        campaign_manifest_path,
        {
            **input_payload,
            "campaign_input_sha256": input_sha,
            "output_directory": str(output),
            "complete": True,
            "candidate_count": len(all_results),
            "search_report": "search_report.json",
            "search_report_sha256": file_sha256(output / "search_report.json"),
        },
    )
    return report


def load_lift_sources(
    *,
    config_paths: Sequence[str | Path] = (),
    rescue_report_path: str | Path | None = None,
    maximum_count: int = FIXED_LIFT_BUDGET.pose_count,
) -> tuple[dict[str, Any], ...]:
    """Load authenticated rescue passes or explicit schema-v8 configurations."""

    if maximum_count <= 0:
        raise ValueError("maximum_count must be positive")
    if config_paths and rescue_report_path is not None:
        raise ValueError("explicit configs and --rescue-report are mutually exclusive")
    sources: list[dict[str, Any]] = []
    if rescue_report_path is not None:
        report_path = Path(rescue_report_path).expanduser().resolve()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("complete") is not True or report.get("stage") != "rescue":
            raise ValueError("lift input report must be a complete rescue report")
        results = report.get("results", [])
        if not isinstance(results, list):
            raise ValueError("rescue report has no results list")
        seen_poses: set[str] = set()
        for compact in results:
            if not isinstance(compact, Mapping) or not compact.get("rescue_success", False):
                continue
            identifier = str(compact.get("pose_id", ""))
            if not identifier or identifier in seen_poses:
                continue
            relative = compact.get("artifact_directory")
            directory = _safe_member(report_path.parent, relative, "rescue artifact directory")
            result_path = directory / "result.json"
            config_path = directory / "resolved_config.json"
            if not result_path.is_file() or not config_path.is_file():
                raise FileNotFoundError(f"incomplete rescue pass artifact: {directory}")
            persisted_result = json.loads(result_path.read_text(encoding="utf-8"))
            if persisted_result.get("rescue_success") is not True:
                raise ValueError(f"reported rescue pass is not authenticated: {result_path}")
            config = load_config(config_path)
            semantic = canonical_sha256(config)
            if compact.get("candidate_sha256") != semantic:
                raise ValueError(f"rescue config semantic hash mismatch: {config_path}")
            hashes = persisted_result.get("artifacts", {}).get("sha256", {})
            if hashes.get("resolved_config") != file_sha256(config_path):
                raise ValueError(f"rescue config file hash mismatch: {config_path}")
            metadata = config.get("candidate_metadata", {})
            source = _lift_source_record(
                config,
                source_candidate_id=int(compact["candidate_id"]),
                source_family_id=str(metadata.get("source_family_id", "rescue")),
                source_trajectory_id=str(metadata.get("source_trajectory_id", "rescue")),
                tier=str(compact.get("tier", "rescue")),
                source_path=str(config_path),
            )
            if source["pose_id"] != identifier:
                raise ValueError("rescue report rebound a different pose")
            source["rescue_report"] = str(report_path)
            source["rescue_result"] = str(result_path)
            sources.append(source)
            seen_poses.add(identifier)
            if len(sources) >= maximum_count:
                break
    else:
        seen_poses: set[str] = set()
        for index, raw_path in enumerate(config_paths):
            config_path = Path(raw_path).expanduser().resolve()
            config = load_config(config_path)
            source = _lift_source_record(
                config,
                source_candidate_id=-(index + 1),
                source_path=str(config_path),
            )
            if source["pose_id"] in seen_poses:
                continue
            sources.append(source)
            seen_poses.add(str(source["pose_id"]))
            if len(sources) >= maximum_count:
                break
    return tuple(sources)


def generate_lift_refine_candidates(
    parents: Sequence[Mapping[str, Any]],
    source_by_pose: Mapping[str, Mapping[str, Any]],
    *,
    count_per_parent: int,
    seed: int = DEFAULT_SEED,
    validator: Callable[[dict[str, Any]], None] | None = validate_config,
) -> tuple[dict[str, Any], ...]:
    if count_per_parent <= 0:
        raise ValueError("count_per_parent must be positive")
    records: list[dict[str, Any]] = []
    for parent_index, parent in enumerate(parents):
        identifier = str(parent["pose_id"])
        source = source_by_pose[identifier]
        base = copy.deepcopy(dict(parent["config"]))
        bounds = _ctrlrange_safe_manipulation_delta_bounds(base)
        centre = _clip_delta(base["control"]["manipulation_delta_rad"], bounds)
        units = _latin_hypercube(
            max(0, count_per_parent - 1),
            len(ACTIVE_ACTUATORS),
            int(
                np.random.SeedSequence(
                    [seed, int(parent["candidate_id"]), 8_800_001]
                ).generate_state(1)[0]
            ),
        )
        deltas: list[tuple[str, dict[str, float]]] = [("parent_exact", centre)]
        for unit in units:
            deltas.append(
                (
                    "local_latin_hypercube",
                    {
                        name: float(
                            np.clip(
                                centre[name]
                                + float(unit[index])
                                * 0.10
                                * (bounds[name][1] - bounds[name][0]),
                                bounds[name][0],
                                bounds[name][1],
                            )
                        )
                        for index, name in enumerate(ACTIVE_ACTUATORS)
                    },
                )
            )
        for local_index, (kind, delta) in enumerate(deltas):
            candidate_id_value = (
                _LIFT_REFINE_BASE + parent_index * _ID_STRIDE + local_index
            )
            config = materialize_lift_candidate(
                base,
                source,
                delta,
                candidate_id_value=candidate_id_value,
                stage="lift_refine",
                parent_candidate_id=int(parent["candidate_id"]),
                search_metadata={"kind": kind, "trust_radius_fraction": 0.10},
                validator=validator,
            )
            records.append(_lift_candidate_record(config, source))
    return tuple(records)


def generate_lift_exact_candidates(
    ranked: Sequence[Mapping[str, Any]],
    source_by_pose: Mapping[str, Mapping[str, Any]],
    *,
    count: int,
    validator: Callable[[dict[str, Any]], None] | None = validate_config,
) -> tuple[dict[str, Any], ...]:
    selected: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for candidate in ranked:
        identifier = str(candidate["pose_id"])
        if identifier in seen:
            continue
        selected.append(candidate)
        seen.add(identifier)
        if len(selected) >= count:
            break
    records = []
    for index, parent in enumerate(selected):
        source = source_by_pose[str(parent["pose_id"])]
        config = materialize_lift_candidate(
            parent["config"],
            source,
            parent["config"]["control"]["manipulation_delta_rad"],
            candidate_id_value=_LIFT_EXACT_BASE + index,
            stage="lift_exact",
            parent_candidate_id=int(parent["candidate_id"]),
            search_metadata={"kind": "locked_timestep_replay", "timestep_s": 0.001},
            validator=validator,
        )
        config["candidate_metadata"]["locked_timestep_s"] = 0.001
        if pose_id(config) != str(parent["pose_id"]):
            raise RuntimeError("exact lift replay changed its pose")
        if controller_id(config) != str(parent["controller_id"]):
            raise RuntimeError("exact lift replay changed its controller")
        if validator is not None:
            validator(config)
        records.append(_lift_candidate_record(config, source))
    return tuple(records)


def _compact_lift_result(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value.get(key))
        for key in (
            "candidate_id",
            "candidate_sha256",
            "stage",
            "pose_id",
            "controller_id",
            "source_candidate_id",
            "edge_m",
            "thumb_target_rad",
            "acquisition_success",
            "pose_preservation_success",
            "rescue_success",
            "lift_success",
            "classification",
            "lift_rank_evidence",
            "manipulation_response",
            "artifact_directory",
        )
    }


def _require_exact_lift_candidate(value: Mapping[str, Any]) -> None:
    if value.get("stage") != "lift_exact":
        raise ValueError("only lift_exact candidates may enter the lift catalog")
    config = value.get("config")
    metadata = config.get("candidate_metadata") if isinstance(config, Mapping) else None
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("stage") != "lift_exact"
        or int(metadata.get("candidate_id", -1)) != int(value.get("candidate_id", -2))
        or not math.isclose(
            _finite_metric(metadata.get("locked_timestep_s"), math.inf),
            0.001,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(
            "lift catalog candidates must carry locked lift_exact provenance"
        )


def publish_lift_trajectory_catalog(
    output: str | Path,
    ranked: Sequence[Mapping[str, Any]],
    *,
    maximum_count: int = 5,
) -> dict[str, Any]:
    """Publish locked exact Viewer members without escaping catalog root."""

    if maximum_count <= 0:
        raise ValueError("maximum_count must be positive")
    for value in ranked:
        _require_exact_lift_candidate(value)
    root = Path(output).expanduser().resolve()
    catalog_root = root / "trajectory_catalog"
    catalog_root.mkdir(parents=True, exist_ok=True)
    selected: list[Mapping[str, Any]] = []
    seen_poses: set[str] = set()
    for value in ranked:
        identifier = str(value["pose_id"])
        if identifier in seen_poses:
            continue
        selected.append(value)
        seen_poses.add(identifier)
        if len(selected) >= maximum_count:
            break
    trajectories = []
    aliases: dict[str, str] = {}
    for index, value in enumerate(selected, start=1):
        trajectory_id = f"trajectory_{index:02d}"
        source = _safe_member(
            root, value.get("artifact_directory"), "lift candidate artifact"
        )
        member = catalog_root / trajectory_id
        member.mkdir(parents=True, exist_ok=True)
        copied: dict[str, Path] = {}
        for key, filename in (
            ("resolved_config", "resolved_config.json"),
            ("trace", "trace.npz"),
            ("result", "result.json"),
        ):
            source_path = source / filename
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            destination = member / filename
            shutil.copy2(source_path, destination)
            copied[key] = destination
        passed = bool(lift_candidate_evidence(value)["lift_success"])
        entry_aliases: list[str] = []
        if index == 1:
            best_alias = "best_nominal" if passed else "best_attempt"
            aliases[best_alias] = trajectory_id
            entry_aliases.append(best_alias)
            fixed_name = (
                "best_nominal_config.json" if passed else "best_attempt_config.json"
            )
            shutil.copy2(copied["resolved_config"], root / fixed_name)
        trajectories.append(
            {
                "trajectory_id": trajectory_id,
                "label": trajectory_id,
                "aliases": entry_aliases,
                "candidate_id": int(value["candidate_id"]),
                "pose_id": str(value["pose_id"]),
                "classification": "validated_lift" if passed else "near_miss",
                "artifacts": {
                    key: f"{trajectory_id}/{path.name}" for key, path in copied.items()
                }
                | {
                    "sha256": {
                        key: file_sha256(path) for key, path in copied.items()
                    }
                },
            }
        )
    catalog = {
        "trajectory_catalog_schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "campaign_kind": CAMPAIGN_KIND,
        "campaign_has_passing_trajectory": any(
            entry["classification"] == "validated_lift" for entry in trajectories
        ),
        "aliases": aliases,
        "trajectories": trajectories,
    }
    write_json(catalog_root / "catalog.json", catalog)
    return {
        "path": "trajectory_catalog/catalog.json",
        "sha256": file_sha256(catalog_root / "catalog.json"),
        "trajectory_count": len(trajectories),
        "aliases": aliases,
        "best_config": (
            "best_nominal_config.json"
            if "best_nominal" in aliases
            else "best_attempt_config.json" if trajectories else None
        ),
    }


def run_lift_campaign(
    output_dir: str | Path = DEFAULT_LIFT_OUTPUT,
    *,
    config_paths: Sequence[str | Path] = (),
    rescue_report_path: str | Path | None = None,
    workers: int = 1,
    resume: bool = False,
    seed: int = DEFAULT_SEED,
    budget: LiftBudget = FIXED_LIFT_BUDGET,
    executor: CandidateExecutor = run_candidate_jobs,
) -> dict[str, Any]:
    """Probe real object response, fit it, and search bounded lift deltas."""

    if workers <= 0:
        raise ValueError("workers must be positive")
    sources = load_lift_sources(
        config_paths=config_paths,
        rescue_report_path=rescue_report_path,
        maximum_count=budget.pose_count,
    )
    output = Path(output_dir).expanduser().resolve()
    source_manifest = [
        {
            key: copy.deepcopy(source.get(key))
            for key in (
                "pose_id",
                "controller_id",
                "source_candidate_id",
                "source_config",
                "source_config_sha256",
                "edge_m",
                "thumb_target_rad",
            )
        }
        for source in sources
    ]
    input_payload = {
        "campaign_schema_version": CAMPAIGN_SCHEMA_VERSION,
        "campaign_kind": CAMPAIGN_KIND,
        "experiment_id": EXPERIMENT_ID,
        "stage": "lift",
        "seed": int(seed),
        "budget": asdict(budget),
        "sources_sha256": canonical_sha256(source_manifest),
        "rescue_report_file_sha256": (
            file_sha256(Path(rescue_report_path).expanduser().resolve())
            if rescue_report_path is not None
            else None
        ),
        "tuner_source_sha256": file_sha256(Path(__file__)),
    }
    input_sha = canonical_sha256(input_payload)
    campaign_manifest_path = output / "campaign_manifest.json"
    if output.exists():
        if not resume:
            raise FileExistsError(f"output directory already exists: {output}; pass --resume")
        if not campaign_manifest_path.is_file():
            raise RuntimeError("resume output has no campaign_manifest.json")
        existing = json.loads(campaign_manifest_path.read_text(encoding="utf-8"))
        if existing.get("campaign_input_sha256") != input_sha:
            raise RuntimeError("resume inputs do not match the existing lift manifest")
    else:
        output.mkdir(parents=True)
        write_json(output / "source_manifest.json", source_manifest)
        write_json(
            campaign_manifest_path,
            {
                **input_payload,
                "campaign_input_sha256": input_sha,
                "output_directory": str(output),
                "complete": False,
            },
        )
    if not sources:
        report = {
            "campaign_schema_version": CAMPAIGN_SCHEMA_VERSION,
            "campaign_kind": CAMPAIGN_KIND,
            "experiment_id": EXPERIMENT_ID,
            "stage": "lift",
            "complete": True,
            "campaign_input_sha256": input_sha,
            "source_count": 0,
            "candidate_count": 0,
            "provisional_lift_pass_count": 0,
            "lift_pass_count": 0,
            "validated_pose_count": 0,
            "best_search_candidate": None,
            "best_candidate": None,
            "stop_reason": "no_authenticated_rescue_pass_or_explicit_config",
            "search_results": [],
            "results": [],
        }
        write_json(output / "search_report.json", report)
        write_json(
            campaign_manifest_path,
            {
                **input_payload,
                "campaign_input_sha256": input_sha,
                "output_directory": str(output),
                "complete": True,
                "candidate_count": 0,
                "search_report": "search_report.json",
                "search_report_sha256": file_sha256(output / "search_report.json"),
            },
        )
        return report

    source_by_pose = {str(source["pose_id"]): source for source in sources}
    probes = tuple(
        candidate
        for pose_index, source in enumerate(sources)
        for candidate in generate_lift_probe_candidates(
            source["config"], source, pose_index=pose_index
        )
    )
    probe_results = _run_or_resume(
        probes, output, workers=workers, resume=resume, executor=executor
    )
    probe_by_pose: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for result in probe_results:
        probe_by_pose[str(result["pose_id"])].append(result)
    response_models: dict[str, dict[str, Any]] = {}
    trust_candidates: list[dict[str, Any]] = []
    for pose_index, source in enumerate(sources):
        identifier = str(source["pose_id"])
        bounds = _ctrlrange_safe_manipulation_delta_bounds(source["config"])
        model = fit_manipulation_response_model(probe_by_pose[identifier], bounds)
        response_models[identifier] = model
        trust_candidates.extend(
            generate_lift_trust_candidates(
                source["config"],
                source,
                model,
                pose_index=pose_index,
                count=budget.trust_candidates_per_pose,
                seed=seed,
            )
        )
    write_json(output / "response_models.json", response_models)
    trust_results = _run_or_resume(
        trust_candidates, output, workers=workers, resume=resume, executor=executor
    )
    best_by_pose: list[dict[str, Any]] = []
    grouped_trust: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for result in trust_results:
        grouped_trust[str(result["pose_id"])].append(result)
    for identifier in sorted(grouped_trust):
        best_by_pose.append(rank_lift_results(grouped_trust[identifier])[0])
    refine_parents = rank_lift_results(best_by_pose)[: budget.refine_pose_count]
    refine_candidates = generate_lift_refine_candidates(
        refine_parents,
        source_by_pose,
        count_per_parent=budget.refine_per_pose,
        seed=seed,
    )
    refine_results = _run_or_resume(
        refine_candidates, output, workers=workers, resume=resume, executor=executor
    )
    search_ranked = rank_lift_results((*trust_results, *refine_results))
    exact_candidates = generate_lift_exact_candidates(
        search_ranked,
        source_by_pose,
        count=min(budget.exact_count, len(sources)),
    )
    exact_results = _run_or_resume(
        exact_candidates, output, workers=workers, resume=resume, executor=executor
    )
    all_results = tuple((*probe_results, *trust_results, *refine_results, *exact_results))
    exact_ranked = rank_lift_results(exact_results)
    provisional_passes = [
        result
        for result in search_ranked
        if lift_candidate_evidence(result)["lift_success"]
    ]
    exact_passes = [
        result
        for result in exact_ranked
        if lift_candidate_evidence(result)["lift_success"]
    ]
    trajectory_catalog = publish_lift_trajectory_catalog(
        output, exact_ranked, maximum_count=budget.exact_count
    )
    source_reports = []
    for source in sources:
        identifier = str(source["pose_id"])
        search_values = [
            value for value in search_ranked if str(value["pose_id"]) == identifier
        ]
        exact_values = [
            value for value in exact_ranked if str(value["pose_id"]) == identifier
        ]
        source_reports.append(
            {
                "pose_id": identifier,
                "source_candidate_id": source["source_candidate_id"],
                "edge_m": source["edge_m"],
                "thumb_target_rad": source["thumb_target_rad"],
                "probe_count": len(probe_by_pose[identifier]),
                "response_model": response_models[identifier],
                "optimization_candidate_count": len(search_values),
                "provisional_lift_pass_count": sum(
                    lift_candidate_evidence(value)["lift_success"]
                    for value in search_values
                ),
                "best_search_candidate": (
                    _compact_lift_result(search_values[0]) if search_values else None
                ),
                "exact_candidate_count": len(exact_values),
                "lift_pass_count": sum(
                    lift_candidate_evidence(value)["lift_success"]
                    for value in exact_values
                ),
                "best_candidate": (
                    _compact_lift_result(exact_values[0]) if exact_values else None
                ),
            }
        )
    report = {
        "campaign_schema_version": CAMPAIGN_SCHEMA_VERSION,
        "campaign_kind": CAMPAIGN_KIND,
        "experiment_id": EXPERIMENT_ID,
        "stage": "lift",
        "complete": True,
        "campaign_input_sha256": input_sha,
        "workers": int(workers),
        "seed": int(seed),
        "budget": asdict(budget),
        "source_count": len(sources),
        "candidate_count": len(all_results),
        "probe_candidate_count": len(probe_results),
        "trust_candidate_count": len(trust_results),
        "refine_candidate_count": len(refine_results),
        "exact_candidate_count": len(exact_results),
        "provisional_lift_pass_count": len(provisional_passes),
        "lift_pass_count": len(exact_passes),
        "validated_pose_count": len({value["pose_id"] for value in exact_passes}),
        "best_search_candidate": (
            _compact_lift_result(search_ranked[0]) if search_ranked else None
        ),
        "best_candidate": (
            _compact_lift_result(exact_ranked[0]) if exact_ranked else None
        ),
        "trajectory_catalog": trajectory_catalog,
        "source_results": source_reports,
        "search_results": [_compact_lift_result(value) for value in search_ranked],
        "results": [_compact_lift_result(value) for value in exact_ranked],
    }
    write_json(output / "search_report.json", report)
    write_json(
        campaign_manifest_path,
        {
            **input_payload,
            "campaign_input_sha256": input_sha,
            "output_directory": str(output),
            "complete": True,
            "candidate_count": len(all_results),
            "search_report": "search_report.json",
            "search_report_sha256": file_sha256(output / "search_report.json"),
        },
    )
    return report


def run_all_campaign(
    output_dir: str | Path,
    *,
    campaign_results_path: str | Path = DEFAULT_V7_CAMPAIGN_RESULTS,
    template_path: str | Path = DEFAULT_V8_TEMPLATE,
    workers: int = 1,
    resume: bool = False,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Run rescue then feed only authenticated rescue passes into lift."""

    root = Path(output_dir).expanduser().resolve()
    rescue_output = root / "rescue"
    lift_output = root / "lift"
    rescue = run_pose_rescue_campaign(
        campaign_results_path,
        template_path,
        rescue_output,
        workers=workers,
        resume=resume,
        seed=seed,
    )
    lift = run_lift_campaign(
        lift_output,
        rescue_report_path=rescue_output / "search_report.json",
        workers=workers,
        resume=resume,
        seed=seed,
    )
    return {
        "campaign_kind": CAMPAIGN_KIND,
        "experiment_id": EXPERIMENT_ID,
        "stage": "all",
        "complete": True,
        "output_directory": str(root),
        "rescue": {key: value for key, value in rescue.items() if key != "results"},
        "lift": {
            key: value
            for key, value in lift.items()
            if key not in {"results", "search_results"}
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run schema-v8 normal-aligned grasp rescue, response-guided lift, "
            "or both stages."
        )
    )
    parser.add_argument("--stage", choices=("rescue", "lift", "all"), default="rescue")
    parser.add_argument("--campaign-results", default=str(DEFAULT_V7_CAMPAIGN_RESULTS))
    parser.add_argument("--template", default=str(DEFAULT_V8_TEMPLATE))
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "stage output directory (defaults to tune/rescue, tune/lift, or tune for all)"
        ),
    )
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument(
        "--rescue-report",
        help="complete rescue search_report.json consumed by --stage lift",
    )
    inputs.add_argument(
        "--config",
        action="append",
        default=[],
        help="resolved schema-v8 grasp config; repeat to tune several poses",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage != "lift" and (args.rescue_report or args.config):
        raise SystemExit("--rescue-report/--config are only valid with --stage lift")
    if args.stage == "rescue":
        output = args.output_dir or str(DEFAULT_OUTPUT)
        report = run_pose_rescue_campaign(
            args.campaign_results,
            args.template,
            output,
            workers=args.workers,
            resume=args.resume,
            seed=args.seed,
        )
        success_count = int(report["rescue_pass_count"])
    elif args.stage == "lift":
        output = args.output_dir or str(DEFAULT_LIFT_OUTPUT)
        rescue_report = args.rescue_report
        if not args.config and rescue_report is None:
            rescue_report = str(DEFAULT_OUTPUT / "search_report.json")
        report = run_lift_campaign(
            output,
            config_paths=args.config,
            rescue_report_path=rescue_report,
            workers=args.workers,
            resume=args.resume,
            seed=args.seed,
        )
        success_count = int(report["lift_pass_count"])
    else:
        output = args.output_dir or str(DEFAULT_OUTPUT.parent)
        report = run_all_campaign(
            output,
            campaign_results_path=args.campaign_results,
            template_path=args.template,
            workers=args.workers,
            resume=args.resume,
            seed=args.seed,
        )
        success_count = int(report["lift"]["lift_pass_count"])
    print(
        json_text(
            {
                key: value
                for key, value in report.items()
                if key not in {"results", "search_results"}
            }
        )
    )
    return 0 if success_count > 0 else 2


__all__ = [
    "ACTIVE_ACTUATOR_POSITION_LIMITS_RAD",
    "CAMPAIGN_KIND",
    "CLOSE_DURATION_OPTIONS_S",
    "DEFAULT_LIFT_OUTPUT",
    "DEFAULT_OUTPUT",
    "DEFAULT_V7_CAMPAIGN_RESULTS",
    "DEFAULT_V8_TEMPLATE",
    "EXPERIMENT_ID",
    "FIXED_RESCUE_BUDGET",
    "FIXED_LIFT_BUDGET",
    "LiftBudget",
    "RescueBudget",
    "RescueControlSpec",
    "analytic_synchronization_spec",
    "assert_pose_context_preserved",
    "build_parser",
    "build_pose_rescue_manifest",
    "candidate_rank_evidence",
    "controller_id",
    "generate_control_specs",
    "generate_exact_candidates",
    "generate_local_candidates",
    "generate_lift_exact_candidates",
    "generate_lift_probe_candidates",
    "generate_lift_refine_candidates",
    "generate_lift_trust_candidates",
    "generate_pose_coarse_candidates",
    "main",
    "fit_manipulation_response_model",
    "lift_candidate_evidence",
    "lift_candidate_rank",
    "load_lift_sources",
    "manipulation_delta_bounds",
    "manipulation_response_from_trace",
    "materialize_lift_candidate",
    "materialize_v8_rescue_candidate",
    "original_control_spec",
    "pose_context",
    "pose_id",
    "publish_lift_trajectory_catalog",
    "rank_rescue_results",
    "rank_lift_results",
    "rescue_candidate_rank",
    "run_pose_rescue_campaign",
    "run_lift_campaign",
    "run_all_campaign",
    "validated_67_control_spec",
]
