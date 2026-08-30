"""Versioned manifests for reproducible schema-v5 grasp-only catalogs."""

from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

from .config import ACTIVE_ACTUATORS, load_config
from .tuning.high_thumb_grasp import (
    THUMB_BEND_ACTUATOR,
    materialize_high_thumb_candidate,
)


GRASP_CAMPAIGN_MANIFEST_SCHEMA_VERSION = 1
_SAFE_LABEL = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _candidate_id(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("candidate_id must be a non-negative integer")
    try:
        identifier = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("candidate_id must be a non-negative integer") from error
    if identifier < 0 or identifier != value:
        raise ValueError("candidate_id must be a non-negative integer")
    return identifier


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def load_grasp_campaign_manifest(
    path: str | Path,
) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
    """Load and materialize one compact high-thumb grasp campaign manifest.

    The manifest stores relative-pose/search evidence without duplicating the
    full versioned experiment config.  Every returned candidate is a complete,
    independently valid schema-v5 config ready for a real MuJoCo rerun.
    """

    manifest_path = Path(path).expanduser().resolve()
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = _mapping(raw, "manifest")
    if manifest.get("manifest_schema_version") != (
        GRASP_CAMPAIGN_MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError(
            "manifest_schema_version must be "
            f"{GRASP_CAMPAIGN_MANIFEST_SCHEMA_VERSION}"
        )
    if manifest.get("validation_scope") != "grasp_acquisition":
        raise ValueError("manifest validation_scope must be grasp_acquisition")

    base_value = manifest.get("base_config")
    if not isinstance(base_value, str) or not base_value:
        raise ValueError("manifest base_config must be a non-empty path")
    base_path = (manifest_path.parent / base_value).resolve()
    base = load_config(base_path)
    shared = _mapping(manifest.get("shared"), "manifest shared")
    shared_targets = _mapping(
        shared.get("grasp_targets_rad"), "shared.grasp_targets_rad"
    )
    non_thumb = set(ACTIVE_ACTUATORS) - {THUMB_BEND_ACTUATOR}
    if set(shared_targets) != non_thumb:
        raise ValueError(
            "shared.grasp_targets_rad must contain exactly the seven active "
            "actuators other than thumb bend"
        )

    raw_candidates = manifest.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("manifest candidates must be a non-empty list")
    prepared: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    seen_labels: set[str] = set()
    for index, raw_candidate in enumerate(raw_candidates):
        spec = _mapping(raw_candidate, f"candidates[{index}]")
        identifier = _candidate_id(spec.get("candidate_id"))
        if identifier in seen_ids:
            raise ValueError("manifest candidate_id values must be unique")
        seen_ids.add(identifier)
        label = spec.get("label")
        if not isinstance(label, str) or _SAFE_LABEL.fullmatch(label) is None:
            raise ValueError(
                "candidate label must use lowercase letters, digits, '_' or '-'"
            )
        if label in seen_labels:
            raise ValueError("manifest candidate labels must be unique")
        seen_labels.add(label)

        relative_mm = spec.get("cube_in_root_mm")
        if not isinstance(relative_mm, list) or len(relative_mm) != 3:
            raise ValueError("candidate cube_in_root_mm must contain three values")
        relative_m = [
            _finite(value, "candidate cube_in_root_mm") / 1000.0
            for value in relative_mm
        ]
        declared_distance_mm = _finite(
            spec.get("root_cube_distance_mm"),
            "candidate root_cube_distance_mm",
        )
        actual_distance_mm = 1000.0 * math.sqrt(
            sum(value * value for value in relative_m)
        )
        if not math.isclose(
            actual_distance_mm,
            declared_distance_mm,
            rel_tol=0.0,
            abs_tol=0.02,
        ):
            raise ValueError(
                "candidate root_cube_distance_mm disagrees with cube_in_root_mm"
            )

        targets = {
            name: _finite(shared_targets[name], f"shared target {name}")
            for name in non_thumb
        }
        targets[THUMB_BEND_ACTUATOR] = _finite(
            spec.get("thumb_bend_target_rad"),
            "candidate thumb_bend_target_rad",
        )
        target_overrides = spec.get("grasp_target_overrides_rad", {})
        target_overrides = _mapping(
            target_overrides, "candidate grasp_target_overrides_rad"
        )
        if THUMB_BEND_ACTUATOR in target_overrides:
            raise ValueError(
                "thumb bend must be set with candidate thumb_bend_target_rad"
            )
        unknown_overrides = set(target_overrides) - set(ACTIVE_ACTUATORS)
        if unknown_overrides:
            raise ValueError(
                "candidate grasp_target_overrides_rad contains unknown actuators"
            )
        for name, value in target_overrides.items():
            targets[name] = _finite(value, f"candidate target override {name}")
        config = materialize_high_thumb_candidate(
            base,
            edge_m=_finite(spec.get("edge_mm"), "candidate edge_mm") / 1000.0,
            cube_in_root_m=relative_m,
            cube_yaw_deg=_finite(
                spec.get("cube_yaw_deg"), "candidate cube_yaw_deg"
            ),
            grasp_targets_rad=targets,
            finger_down_tilt_deg=_finite(
                shared.get("finger_down_tilt_deg"),
                "shared finger_down_tilt_deg",
            ),
            tilt_band_center_deg=_finite(
                shared.get("tilt_band_center_deg"),
                "shared tilt_band_center_deg",
            ),
            hand_roll_deg=_finite(
                shared.get("hand_roll_deg"), "shared hand_roll_deg"
            ),
            hand_yaw_deg=_finite(
                shared.get("hand_yaw_deg"), "shared hand_yaw_deg"
            ),
        )
        config["candidate_metadata"].update(
            {
                "manifest_schema_version": (
                    GRASP_CAMPAIGN_MANIFEST_SCHEMA_VERSION
                ),
                "manifest_label": label,
                "manifest_candidate_id": identifier,
                "declared_root_cube_distance_m": declared_distance_mm / 1000.0,
            }
        )
        prepared.append(
            {
                "candidate_id": identifier,
                "label": label,
                "config": config,
            }
        )

    best_candidate_id = _candidate_id(manifest.get("best_candidate_id"))
    if best_candidate_id not in seen_ids:
        raise ValueError("best_candidate_id is not present in candidates")
    metadata = {
        "manifest_schema_version": GRASP_CAMPAIGN_MANIFEST_SCHEMA_VERSION,
        "validation_scope": "grasp_acquisition",
        "manifest_path": str(manifest_path),
        "base_config_path": str(base_path),
        "candidate_count": len(prepared),
        "best_candidate_id": best_candidate_id,
        "search_seed": manifest.get("search_seed"),
        "search_note": copy.deepcopy(manifest.get("search_note")),
    }
    return prepared, best_candidate_id, metadata


__all__ = [
    "GRASP_CAMPAIGN_MANIFEST_SCHEMA_VERSION",
    "load_grasp_campaign_manifest",
]
