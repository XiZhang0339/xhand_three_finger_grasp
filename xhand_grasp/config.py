"""Configuration schema and shared names for the XHAND cube-lift task."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


# Keep this pointing at the repository root when the implementation lives in a
# package.  ``grasp_cube.SCRIPT_DIR`` will continue to expose the same path.
SCRIPT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = SCRIPT_DIR / "grasp_configs" / "left_three_finger_cube.json"

ACTIVE_ACTUATORS = (
    "left_hand_thumb_bend_joint_actuator",
    "left_hand_thumb_rota_joint1_actuator",
    "left_hand_thumb_rota_joint2_actuator",
    "left_hand_index_bend_joint_actuator",
    "left_hand_index_joint1_actuator",
    "left_hand_index_joint2_actuator",
    "left_hand_mid_joint1_actuator",
    "left_hand_mid_joint2_actuator",
)
INACTIVE_ACTUATORS = (
    "left_hand_ring_joint1_actuator",
    "left_hand_ring_joint2_actuator",
    "left_hand_pinky_joint1_actuator",
    "left_hand_pinky_joint2_actuator",
)
ACTIVE_FINGERS = ("thumb", "index", "mid")
ALL_FINGERS = ("thumb", "index", "mid", "ring", "pinky")
DISTAL_BODY_NAMES = {
    "thumb": "left_hand_thumb_rota_link2",
    "index": "left_hand_index_rota_link2",
    "mid": "left_hand_mid_link2",
}

SEARCH_TARGET_BOUNDS = {
    "left_hand_thumb_bend_joint_actuator": (0.35, 1.05),
    "left_hand_thumb_rota_joint1_actuator": (0.55, 1.10),
    "left_hand_thumb_rota_joint2_actuator": (0.75, 1.40),
    "left_hand_index_bend_joint_actuator": (-0.12, 0.12),
    "left_hand_index_joint1_actuator": (0.75, 1.50),
    "left_hand_index_joint2_actuator": (0.65, 1.25),
    "left_hand_mid_joint1_actuator": (0.75, 1.50),
    "left_hand_mid_joint2_actuator": (0.65, 1.25),
}


def precontact_targets(config: dict[str, Any]) -> dict[str, Any]:
    """Return the command pose used before contact across all schema versions."""

    control = config["control"]
    if int(config.get("schema_version", 1)) >= 9:
        return control["precontact_targets_rad"]
    if "pregrasp_targets_rad" in control:
        return control["pregrasp_targets_rad"]
    # Schema v3--v5 historically has no separately persisted precontact pose.
    return control["grasp_targets_rad"]


def contact_preload_targets(config: dict[str, Any]) -> dict[str, Any]:
    """Return the controller preload without treating it as measured qpos."""

    control = config["control"]
    if int(config.get("schema_version", 1)) >= 9:
        return control["contact_preload_targets_rad"]
    if "grasp_targets_rad" in control:
        return control["grasp_targets_rad"]
    return control["final_targets_rad"]


def _finite_sequence(values: Iterable[float], length: int, label: str) -> list[float]:
    result = [float(value) for value in values]
    if len(result) != length or not np.isfinite(result).all():
        raise ValueError(f"{label} must contain {length} finite values")
    return result


def _rpy_rotation_matrix(rpy_deg: Iterable[float]) -> np.ndarray:
    """Small config-local XYZ Euler helper; avoids a config/scene import cycle."""

    roll, pitch, yaw = np.radians(
        np.asarray(_finite_sequence(rpy_deg, 3, "hand_pose.rpy_deg"))
    )
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def resolved_pose_constraint_values(config: dict[str, Any]) -> dict[str, Any]:
    """Recompute schema-v4 and later pose values in a resolved config."""

    pose = config["pose_constraints"]
    hand = config["hand_pose"]
    rotation = _rpy_rotation_matrix(hand["rpy_deg"])
    gravity = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
    finger_axis = rotation[:, 2]
    gravity_component = float(finger_axis @ gravity)
    horizontal_component = float(
        np.linalg.norm(finger_axis - gravity_component * gravity)
    )
    finger_tilt = math.degrees(
        math.atan2(gravity_component, horizontal_component)
    )
    palm_cosine = float(np.clip(rotation[:, 0] @ gravity, -1.0, 1.0))
    palm_angle = math.degrees(math.acos(palm_cosine))
    translation = _finite_sequence(
        hand["translation_m"], 3, "hand_pose.translation_m"
    )

    cube = config["cube"]
    cube_rotation = _rpy_rotation_matrix(cube.get("rpy_deg", [0.0, 0.0, 0.0]))
    half_extent_z = float(cube["edge_m"]) / 2.0 * float(
        np.sum(np.abs(cube_rotation[2]))
    )
    cube_world = np.asarray(
        [
            float(cube["center_xy_m"][0]),
            float(cube["center_xy_m"][1]),
            float(config["scene"]["support_top_z_m"])
            + half_extent_z
            + float(cube.get("z_offset_m", 0.0)),
        ],
        dtype=np.float64,
    )
    cube_in_root = rotation.T @ (cube_world - np.asarray(translation))
    common = {
        "finger_down_tilt_deg": float(finger_tilt),
        "palm_plane_ground_angle_deg": float(palm_angle),
        "cube_position_in_root_m": tuple(float(value) for value in cube_in_root),
    }
    if int(config.get("schema_version", 1)) in (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16):
        reference = _finite_sequence(
            pose["legacy_press_reference_translation_m"],
            3,
            "pose_constraints.legacy_press_reference_translation_m",
        )
        common.update(
            {
                "root_cube_distance_m": float(np.linalg.norm(cube_in_root)),
                "legacy_palm_press_depth_m": float(
                    reference[2] - translation[2]
                ),
            }
        )
        return common
    reference = _finite_sequence(
        pose["reference_hand_translation_m"],
        3,
        "pose_constraints.reference_hand_translation_m",
    )
    common["palm_press_depth_m"] = float(reference[2] - translation[2])
    return common


def _v4_run_context_kind(config: dict[str, Any]) -> str | None:
    context = config.get("run_context")
    if context is None:
        return None
    if not isinstance(context, dict) or set(context) != {"kind"}:
        raise ValueError("run_context must contain exactly kind")
    kind = context["kind"]
    if kind not in ("parameter_override_run", "robustness_trial"):
        raise ValueError(
            "run_context.kind must be parameter_override_run or robustness_trial"
        )
    return str(kind)


_V13_ROBUSTNESS_FAMILIES = {
    "per_full_success_local_16": ("robustness_trial", 16, 16),
    "best_first_pose_material_50": ("robustness_trial", 50, 50),
    "v12_grasp_per_nominal_local_16": (
        "v12_grasp_robustness_trial",
        12_016,
        16,
    ),
    "v12_grasp_best_pose_material_50": (
        "v12_grasp_robustness_trial",
        12_050,
        50,
    ),
}


def _canonical_config_sha256(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _v13_expected_robustness_seed(
    campaign_seed: int, source_candidate_id: str, family_salt: int
) -> int:
    digest = hashlib.sha256(source_candidate_id.encode("utf-8")).digest()
    identifier = int.from_bytes(digest[:4], "little")
    state = np.random.SeedSequence(
        [int(campaign_seed), identifier, int(family_salt)]
    ).generate_state(1, dtype=np.uint32)
    return int(state[0])


def _validate_v13_robustness_trial(
    config: dict[str, Any], definition: Any
) -> None:
    """Authenticate and bound a schema-v13 perturbation config.

    Nominal schema-v13 configs remain locked to the fixed-160-g campaign.
    A robustness trial is accepted only when its declared perturbations can be
    reversed to one valid nominal source config whose semantic hash and grasp /
    controller identities are explicitly bound in the trial metadata.
    """

    metadata = config.get("candidate_metadata")
    if not isinstance(metadata, dict):
        raise ValueError(
            "schema-v13 robustness_trial requires candidate_metadata"
        )
    present = [
        name
        for name in ("robustness_trial", "v12_grasp_robustness_trial")
        if name in metadata
    ]
    if len(present) != 1:
        raise ValueError(
            "schema-v13 robustness_trial requires exactly one authenticated "
            "robustness metadata block"
        )
    metadata_key = present[0]
    trial = metadata[metadata_key]
    if not isinstance(trial, dict):
        raise ValueError("schema-v13 robustness metadata must be an object")
    family = trial.get("family")
    if family not in _V13_ROBUSTNESS_FAMILIES:
        raise ValueError("schema-v13 robustness family is not registered")
    expected_key, family_salt, maximum_count = _V13_ROBUSTNESS_FAMILIES[family]
    if metadata_key != expected_key:
        raise ValueError("schema-v13 robustness family uses the wrong metadata block")

    required = {
        "family",
        "seed",
        "trial",
        "source_candidate_id",
        "source_config_sha256",
        "source_grasp_pose_id",
        "source_controller_id",
        "source_candidate_binding_sha256",
        "source_had_candidate_metadata",
        "source_cube",
        "resolved_perturbations",
        "full_reset_rerun",
        "initial_state_source",
        "checkpoint_used",
    }
    if metadata_key == "v12_grasp_robustness_trial":
        required.update(
            {
                "point_plan_id",
                "success_scope",
                "manipulation_success_required",
                "full_success_required",
            }
        )
    if set(trial) != required:
        raise ValueError(
            "schema-v13 robustness metadata has unexpected or missing fields"
        )
    source_candidate_id = trial["source_candidate_id"]
    if not isinstance(source_candidate_id, str) or not source_candidate_id:
        raise ValueError(
            "schema-v13 robustness source_candidate_id must be non-empty"
        )
    for name in (
        "source_config_sha256",
        "source_grasp_pose_id",
        "source_controller_id",
        "source_candidate_binding_sha256",
    ):
        value = trial[name]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"schema-v13 robustness {name} must be SHA-256")
    if not isinstance(trial["source_had_candidate_metadata"], bool):
        raise ValueError(
            "schema-v13 robustness source_had_candidate_metadata must be boolean"
        )
    trial_index = trial["trial"]
    if (
        not isinstance(trial_index, int)
        or isinstance(trial_index, bool)
        or not 0 <= trial_index < maximum_count
    ):
        raise ValueError("schema-v13 robustness trial index is outside its budget")
    if (
        not isinstance(trial["seed"], int)
        or isinstance(trial["seed"], bool)
        or trial["seed"]
        != _v13_expected_robustness_seed(
            definition.robustness.seed, source_candidate_id, family_salt
        )
    ):
        raise ValueError("schema-v13 robustness seed/source binding is invalid")
    if (
        trial["full_reset_rerun"] is not True
        or trial["initial_state_source"] != "configured_no_contact_reset"
        or trial["checkpoint_used"] is not False
    ):
        raise ValueError(
            "schema-v13 robustness trials require a full no-contact reset"
        )
    if metadata_key == "v12_grasp_robustness_trial":
        if trial["point_plan_id"] != config["contact_point_plan"]["point_plan_id"]:
            raise ValueError("schema-v13 robustness point-plan binding changed")
        if (
            trial["success_scope"]
            != "grasp_only_contact_point_hard_checks"
            or trial["manipulation_success_required"] is not False
            or trial["full_success_required"] is not False
        ):
            raise ValueError("schema-v13 grasp robustness scope is invalid")

    resolved = trial["resolved_perturbations"]
    expected_resolved_fields = {
        "cube_center_xy_delta_m",
        "cube_gap_delta_m",
        "cube_rpy_delta_deg",
        "mass_scale",
        "friction_delta",
    }
    if not isinstance(resolved, dict) or set(resolved) != expected_resolved_fields:
        raise ValueError(
            "schema-v13 resolved_perturbations has unexpected or missing fields"
        )
    xy_delta = np.asarray(
        _finite_sequence(
            resolved["cube_center_xy_delta_m"],
            2,
            "resolved cube_center_xy_delta_m",
        ),
        dtype=np.float64,
    )
    rpy_delta = np.asarray(
        _finite_sequence(
            resolved["cube_rpy_delta_deg"],
            3,
            "resolved cube_rpy_delta_deg",
        ),
        dtype=np.float64,
    )
    gap_delta = float(resolved["cube_gap_delta_m"])
    mass_scale = float(resolved["mass_scale"])
    friction_delta = float(resolved["friction_delta"])
    if not all(
        math.isfinite(value)
        for value in (gap_delta, mass_scale, friction_delta)
    ):
        raise ValueError("schema-v13 resolved perturbations must be finite")

    nominal_cube = trial["source_cube"]
    if not isinstance(nominal_cube, dict):
        raise ValueError("schema-v13 robustness source_cube must be an object")
    expected_xy = (
        np.asarray(nominal_cube["center_xy_m"], dtype=np.float64) + xy_delta
    )
    expected_rpy = (
        np.asarray(nominal_cube.get("rpy_deg", (0.0, 0.0, 0.0)), dtype=np.float64)
        + rpy_delta
    )
    physics_matches = bool(
        np.allclose(
            np.asarray(config["cube"]["center_xy_m"], dtype=np.float64),
            expected_xy,
            rtol=0.0,
            atol=1e-12,
        )
        and np.allclose(
            np.asarray(config["cube"].get("rpy_deg", (0.0, 0.0, 0.0)), dtype=np.float64),
            expected_rpy,
            rtol=0.0,
            atol=1e-12,
        )
        and math.isclose(
            float(config["cube"].get("z_offset_m", 0.0)),
            float(nominal_cube.get("z_offset_m", 0.0)) + gap_delta,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(config["cube"]["mass_kg"]),
            float(nominal_cube["mass_kg"]) * mass_scale,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(config["cube"]["friction"]),
            float(nominal_cube["friction"]) + friction_delta,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(config["cube"]["edge_m"]),
            float(nominal_cube["edge_m"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )
    if not physics_matches:
        raise ValueError(
            "schema-v13 robustness cube state disagrees with resolved perturbations"
        )
    reconstructed = copy.deepcopy(config)
    reconstructed.pop("run_context", None)
    reconstructed_metadata = reconstructed.get("candidate_metadata")
    assert isinstance(reconstructed_metadata, dict)
    reconstructed_metadata.pop(metadata_key)
    if not trial["source_had_candidate_metadata"]:
        if reconstructed_metadata:
            raise ValueError(
                "schema-v13 robustness source metadata presence binding changed"
            )
        reconstructed.pop("candidate_metadata", None)
    reconstructed["cube"] = copy.deepcopy(nominal_cube)
    # Recursive nominal validation re-applies all fixed cube, contact-plan and
    # experiment registry checks without the robustness exception.
    validate_config(reconstructed)
    if _canonical_config_sha256(reconstructed) != trial["source_config_sha256"]:
        raise ValueError("schema-v13 robustness source config binding changed")
    from .grasp_pose import controller_id, grasp_pose_id

    if grasp_pose_id(reconstructed) != trial["source_grasp_pose_id"]:
        raise ValueError("schema-v13 robustness source grasp-pose binding changed")
    if controller_id(reconstructed) != trial["source_controller_id"]:
        raise ValueError("schema-v13 robustness source controller binding changed")
    expected_source_binding = _canonical_config_sha256(
        {
            "source_candidate_id": source_candidate_id,
            "source_config_sha256": trial["source_config_sha256"],
            "source_grasp_pose_id": trial["source_grasp_pose_id"],
            "source_controller_id": trial["source_controller_id"],
        }
    )
    if trial["source_candidate_binding_sha256"] != expected_source_binding:
        raise ValueError(
            "schema-v13 robustness source candidate binding changed"
        )
    parameters = definition.robustness

    def within(value: float, bounds: tuple[float, float]) -> bool:
        return float(bounds[0]) - 1e-12 <= value <= float(bounds[1]) + 1e-12

    if not all(within(float(value), parameters.position_xy_delta_m) for value in xy_delta):
        raise ValueError("schema-v13 robustness XY perturbation is out of range")
    if not within(gap_delta, parameters.z_offset_delta_m):
        raise ValueError("schema-v13 robustness gap perturbation is out of range")
    if not all(within(float(value), parameters.rpy_delta_deg) for value in rpy_delta):
        raise ValueError("schema-v13 robustness RPY perturbation is out of range")
    if not within(mass_scale, parameters.mass_scale):
        raise ValueError("schema-v13 robustness mass scale is out of range")
    if not within(friction_delta, parameters.friction_delta):
        raise ValueError("schema-v13 robustness friction perturbation is out of range")


def validate_config(config: dict[str, Any]) -> None:
    """Validate the legacy schemas and the feedback-gated schema-v3 extension.

    Version 1 retains its existing validation and scene behavior.  Version 2
    adds a required maximum palm-down angle; the model-owned local palm normal
    is deliberately not configurable, so callers cannot redefine the frame to
    bypass the acceptance check.  Version 3 replaces the two absolute open-loop
    poses with a grasp target and a bounded relative manipulation command.
    """

    schema_version = config.get("schema_version")
    if schema_version not in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16):
        raise ValueError(
            "schema_version must be 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 or 16"
        )
    if schema_version not in (6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16) and "pose_preservation" in config:
        raise ValueError("pose_preservation is only allowed in schema v6 and later")
    campaign_run_context = (
        _v4_run_context_kind(config)
        if schema_version in (4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        else None
    )
    if config.get("side") != "left":
        raise ValueError("this experiment supports only the left hand")

    hand_pose = config["hand_pose"]
    _finite_sequence(hand_pose["translation_m"], 3, "hand_pose.translation_m")
    _finite_sequence(hand_pose["rpy_deg"], 3, "hand_pose.rpy_deg")

    cube = config["cube"]
    _finite_sequence(cube["center_xy_m"], 2, "cube.center_xy_m")
    _finite_sequence(cube.get("rpy_deg", [0.0, 0.0, 0.0]), 3, "cube.rpy_deg")
    for key in ("edge_m", "mass_kg", "friction"):
        if not math.isfinite(float(cube[key])) or float(cube[key]) <= 0:
            raise ValueError(f"cube.{key} must be positive and finite")
    if not math.isfinite(float(cube.get("z_offset_m", 0.0))):
        raise ValueError("cube.z_offset_m must be finite")

    scene = config["scene"]
    for key in ("support_top_z_m", "support_radius_m"):
        if not math.isfinite(float(scene[key])) or float(scene[key]) <= 0:
            raise ValueError(f"scene.{key} must be positive and finite")
    floor_z = float(scene.get("floor_z_m", -0.015))
    if not math.isfinite(floor_z) or floor_z >= float(scene["support_top_z_m"]):
        raise ValueError("scene.floor_z_m must be finite and below support_top_z_m")
    solref_timeconst = float(cube.get("solref_timeconst_s", 0.004))
    if not math.isfinite(solref_timeconst) or solref_timeconst <= 0:
        raise ValueError("cube.solref_timeconst_s must be positive and finite")

    control = config["control"]
    feedback_gated = schema_version in (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
    pose_preserving_control = schema_version in (6, 7, 8)
    actual_contact_control = schema_version in (9, 10, 11, 12, 13, 14, 15, 16)
    if actual_contact_control:
        control_fields = (
            "precontact_targets_rad",
            "contact_preload_targets_rad",
            "manipulation_delta_rad",
        )
        expected_control_fields = set(control_fields) | {"close_profile"}
    elif pose_preserving_control:
        control_fields = (
            "pregrasp_targets_rad",
            "grasp_targets_rad",
            "manipulation_delta_rad",
        )
        expected_control_fields = set(control_fields) | {"close_profile"}
    elif feedback_gated:
        control_fields = ("grasp_targets_rad", "manipulation_delta_rad")
        expected_control_fields = set(control_fields)
    else:
        control_fields = ("pregrasp_targets_rad", "final_targets_rad")
        expected_control_fields = set(control_fields)
    if actual_contact_control and set(control) != expected_control_fields:
        raise ValueError(
            f"schema v{schema_version} control must contain exactly precontact_targets_rad, "
            "contact_preload_targets_rad, manipulation_delta_rad and close_profile"
        )
    if pose_preserving_control and set(control) != expected_control_fields:
        raise ValueError(
            f"schema v{schema_version} control must contain exactly "
            "pregrasp_targets_rad, "
            "grasp_targets_rad, manipulation_delta_rad and close_profile"
        )
    if (
        feedback_gated
        and not pose_preserving_control
        and not actual_contact_control
        and set(control) != set(control_fields)
    ):
        raise ValueError(
            f"schema v{schema_version} control must contain exactly "
            "grasp_targets_rad and "
            "manipulation_delta_rad"
        )
    for field in control_fields:
        targets = control[field]
        if set(targets) != set(ACTIVE_ACTUATORS):
            raise ValueError(f"{field} must contain exactly the eight active actuators")
        for name, value in targets.items():
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"{field}.{name} must be finite")

    if pose_preserving_control or actual_contact_control:
        close_profile = control["close_profile"]
        if not isinstance(close_profile, dict) or set(close_profile) != set(
            ACTIVE_ACTUATORS
        ):
            raise ValueError(
                "control.close_profile must contain exactly the eight active "
                "actuators"
            )
        for name, interval in close_profile.items():
            if not isinstance(interval, dict) or set(interval) != {
                "start_fraction",
                "end_fraction",
            }:
                raise ValueError(
                    f"control.close_profile.{name} must contain exactly "
                    "start_fraction and end_fraction"
                )
            start = float(interval["start_fraction"])
            end = float(interval["end_fraction"])
            if (
                not math.isfinite(start)
                or not math.isfinite(end)
                or not 0.0 <= start < end <= 1.0
            ):
                raise ValueError(
                    f"control.close_profile.{name} must satisfy "
                    "0 <= start_fraction < end_fraction <= 1"
                )

    if feedback_gated:
        if "timing" in config:
            raise ValueError(
                f"schema v{schema_version} timing is versioned inside "
                "control_protocol"
            )
    else:
        timing = config["timing"]
        for key in ("settle_s", "pregrasp_s", "lift_s", "hold_s"):
            if not math.isfinite(float(timing[key])) or float(timing[key]) <= 0:
                raise ValueError(f"timing.{key} must be positive and finite")

    acceptance = config["acceptance"]
    for key in ("finger_contact_duty", "simultaneous_contact_duty"):
        value = float(acceptance[key])
        if not 0 <= value <= 1:
            raise ValueError(f"acceptance.{key} must be within [0, 1]")
    positive_thresholds = (
        "median_lift_m",
        "minimum_lift_m",
        "height_window_s",
        "max_height_span_m",
        "max_orientation_drift_deg",
        "max_end_linear_speed_m_s",
        "max_penetration_m",
        "contact_force_min_n",
        "touch_force_min_n",
        "inactive_joint_abs_max_rad",
    )
    for key in positive_thresholds:
        if not math.isfinite(float(acceptance[key])) or float(acceptance[key]) <= 0:
            raise ValueError(f"acceptance.{key} must be positive and finite")

    if schema_version in (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16):
        experiment_id = config.get("experiment_id")
        if not isinstance(experiment_id, str) or not experiment_id:
            raise ValueError(
                f"schema v{schema_version} requires a non-empty experiment_id"
            )
        if "max_palm_down_angle_deg" not in acceptance:
            raise ValueError(
                "schema v2 requires acceptance.max_palm_down_angle_deg"
            )
        maximum_angle = float(acceptance["max_palm_down_angle_deg"])
        if not math.isfinite(maximum_angle) or not 0.0 <= maximum_angle <= 180.0:
            raise ValueError(
                "acceptance.max_palm_down_angle_deg must be finite and within [0, 180]"
            )

        topology = config.get("contact_topology")
        if not isinstance(topology, dict):
            raise ValueError("schema v2 requires contact_topology")
        required_topology = {
            "target_faces",
            "surface_tolerance_m",
            "edge_margin_m",
            "min_normal_alignment",
            "target_force_fraction",
            "max_off_target_force_fraction",
            "max_material_off_target_duty",
            "max_material_off_target_run_s",
            "forbid_active_nondistal",
        }
        missing_topology = sorted(required_topology - set(topology))
        if missing_topology:
            raise ValueError(
                "contact_topology is missing required fields: "
                + ", ".join(missing_topology)
            )
        target_faces = topology["target_faces"]
        if not isinstance(target_faces, dict) or set(target_faces) != set(
            ACTIVE_FINGERS
        ):
            raise ValueError(
                "contact_topology.target_faces must contain exactly thumb, index and mid"
            )
        physical_faces = {"+X", "-X", "+Y", "-Y", "+Z", "-Z"}
        if any(str(face) not in physical_faces for face in target_faces.values()):
            raise ValueError("contact_topology.target_faces contains an unknown face")
        opposite = {
            "+X": "-X",
            "-X": "+X",
            "+Y": "-Y",
            "-Y": "+Y",
            "+Z": "-Z",
            "-Z": "+Z",
        }
        if target_faces["index"] != target_faces["mid"]:
            raise ValueError("index and mid target faces must be identical")
        if target_faces["thumb"] != opposite[target_faces["index"]]:
            raise ValueError("thumb target face must oppose index and mid")

        surface_tolerance = float(topology["surface_tolerance_m"])
        edge_margin = float(topology["edge_margin_m"])
        if (
            not math.isfinite(surface_tolerance)
            or surface_tolerance <= 0
            or not math.isfinite(edge_margin)
            or edge_margin <= surface_tolerance
        ):
            raise ValueError(
                "contact surface tolerance must be positive and smaller than edge margin"
            )
        for key in (
            "min_normal_alignment",
            "target_force_fraction",
            "max_off_target_force_fraction",
            "max_material_off_target_duty",
        ):
            value = float(topology[key])
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"contact_topology.{key} must be within [0, 1]")
        if float(topology["min_normal_alignment"]) <= 0.0:
            raise ValueError("contact_topology.min_normal_alignment must be positive")
        if float(topology["target_force_fraction"]) <= 0.0:
            raise ValueError("contact_topology.target_force_fraction must be positive")
        if not math.isclose(
            float(topology["max_off_target_force_fraction"]),
            1.0 - float(topology["target_force_fraction"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "max_off_target_force_fraction must equal 1 - target_force_fraction"
            )
        maximum_run = float(topology["max_material_off_target_run_s"])
        if not math.isfinite(maximum_run) or maximum_run < 0.0:
            raise ValueError(
                "contact_topology.max_material_off_target_run_s must be non-negative"
            )
        if not isinstance(topology["forbid_active_nondistal"], bool):
            raise ValueError("contact_topology.forbid_active_nondistal must be boolean")

        search = config.get("search")
        if not isinstance(search, dict):
            raise ValueError("schema v2 requires search settings")
        shared_faces = search.get("shared_faces")
        expected_shared_faces = (
            ["+X"]
            if schema_version in (12, 13, 14, 15, 16)
            else (
                ["+X", "+Y"]
                if schema_version in (5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16)
                else ["+X", "-X", "+Y", "-Y"]
            )
        )
        if shared_faces != expected_shared_faces:
            raise ValueError(
                "search.shared_faces must match the registered ordered face policy"
            )

        # Resolve last so registry-level semantics cannot be bypassed by a
        # syntactically valid but mismatched experiment identifier.
        from .experiment import resolve_experiment

        definition = resolve_experiment(config)
        if feedback_gated:
            protocol = config.get("control_protocol")
            if not isinstance(protocol, dict):
                raise ValueError(
                    f"schema v{schema_version} requires control_protocol"
                )
            if definition.control_protocol is None:
                raise ValueError(
                    f"schema v{schema_version} experiment definition requires "
                    "control_protocol"
                )
            expected_protocol = definition.control_protocol.as_config()
            protocol_matches = protocol == expected_protocol
            if schema_version in (8, 9, 10, 11, 12, 13, 14, 15, 16):
                candidate_protocol = dict(protocol)
                registered_protocol = dict(expected_protocol)
                candidate_close_s = float(candidate_protocol.pop("close_s"))
                registered_protocol.pop("close_s")
                options = definition.control_protocol.close_duration_options_s
                protocol_matches = bool(
                    options is not None
                    and any(
                        math.isclose(candidate_close_s, value, abs_tol=1e-12)
                        for value in options
                    )
                    and candidate_protocol == registered_protocol
                )
            if not protocol_matches:
                raise ValueError(
                    "control_protocol must match the versioned experiment definition"
                )
            bounds = definition.search_bounds
            grasp_targets = contact_preload_targets(config)
            manipulation_delta = control["manipulation_delta_rad"]
            boundary_metadata = config.get("candidate_metadata", {}).get(
                "boundary_expansion", {}
            )
            expanded_thumb = bool(
                isinstance(boundary_metadata, dict)
                and boundary_metadata.get("applied") is True
                and boundary_metadata.get("count") == 1
                and boundary_metadata.get("thumb_bend_expanded") is True
            )
            targets_in_bounds = bounds.contains_targets(grasp_targets)
            if schema_version in (5, 6) and not targets_in_bounds and expanded_thumb:
                thumb_name = "left_hand_thumb_bend_joint_actuator"
                targets_in_bounds = all(
                    (
                        bounds.actuator_targets_rad[name][0]
                        <= float(value)
                        <= (
                            bounds.actuator_targets_rad[name][1] + 0.10
                            if name == thumb_name
                            else bounds.actuator_targets_rad[name][1]
                        )
                    )
                    for name, value in grasp_targets.items()
                )
            if not targets_in_bounds:
                raise ValueError(
                    "control.grasp_targets_rad must stay inside registered search bounds"
                )
            if not bounds.contains_manipulation_delta(manipulation_delta):
                raise ValueError(
                    "control.manipulation_delta_rad must stay inside registered search bounds"
                )
            if (
                (pose_preserving_control or actual_contact_control)
                and not bounds.contains_pregrasp_targets(
                    precontact_targets(config)
                )
            ):
                precontact_label = (
                    "control.precontact_targets_rad"
                    if actual_contact_control
                    else "control.pregrasp_targets_rad"
                )
                raise ValueError(
                    f"{precontact_label} must stay inside registered search bounds"
                )
        elif "control_protocol" in config:
            raise ValueError(
                "control_protocol is only allowed in schema v3 and later"
            )
        if (
            definition.size_campaign is not None
            or definition.aligned_contact_campaign is not None
            or definition.far_hand_campaign is not None
            or definition.high_thumb_size_campaign is not None
            or definition.normal_aligned_smooth_lift_campaign is not None
            or definition.actual_contact_grasp_pose_campaign is not None
        ):
            expected_acceptance = definition.evaluation.acceptance_config()
            if acceptance != expected_acceptance:
                raise ValueError(
                    "acceptance must match the versioned experiment definition"
                )
            expected_topology = definition.evaluation.contact_topology_config()
            expected_topology.pop("target_faces", None)
            declared_topology = dict(topology)
            declared_topology.pop("target_faces", None)
            if declared_topology != expected_topology:
                raise ValueError(
                    "contact_topology thresholds must match the versioned "
                    "experiment definition"
                )
        from .experiment import OpposedFaceAssignment

        assignment = OpposedFaceAssignment.from_mapping(target_faces)
        if assignment not in definition.candidate_faces:
            raise ValueError(
                "contact_topology.target_faces is not registered for the experiment"
            )
        bounds = definition.search_bounds
        budget = search.get("budget")
        if not isinstance(budget, dict):
            raise ValueError("schema v2 requires search.budget")
        if definition.contact_preserving_planned_lift_campaign is not None:
            expected_budget = (
                definition.contact_preserving_planned_lift_campaign.budget_config()
            )
        elif definition.scaled_contact_downsize_campaign is not None:
            expected_budget = (
                definition.scaled_contact_downsize_campaign.budget_config()
            )
        elif definition.contact_point_search is not None:
            expected_budget = definition.contact_point_search.as_config()["budget"]
        elif definition.actual_contact_grasp_pose_campaign is not None:
            expected_budget = (
                definition.actual_contact_grasp_pose_campaign.budget_config()
            )
        elif definition.normal_aligned_smooth_lift_campaign is not None:
            expected_budget = (
                definition.normal_aligned_smooth_lift_campaign.budget_config()
            )
        elif definition.high_thumb_size_campaign is not None:
            expected_budget = (
                definition.high_thumb_size_campaign.budget_config()
            )
        elif definition.far_hand_campaign is not None:
            expected_budget = definition.far_hand_campaign.budget_config()
        elif definition.aligned_contact_campaign is not None:
            expected_budget = definition.aligned_contact_campaign.budget_config()
        else:
            expected_budget = {
                "palm_pitch_values_deg": list(bounds.palm_pitch_values_deg),
                "kinematic_samples_per_pitch": bounds.kinematic_samples_per_pitch,
                "dynamic_candidate_count": bounds.dynamic_candidate_count,
                "local_refine_seed_count": bounds.local_refine_seed_count,
                "local_refine_per_seed": bounds.local_refine_per_seed,
                "final_candidate_count": bounds.final_candidate_count,
                "perturbations_per_final_candidate": (
                    bounds.perturbations_per_final_candidate
                ),
                "fallback_kinematic_samples_per_pitch": (
                    bounds.fallback_kinematic_samples_per_pitch
                ),
                "fallback_candidate_count": bounds.fallback_candidate_count,
            }
        if search.get("seed") != bounds.seed:
            raise ValueError(
                "search.seed must match the versioned experiment definition"
            )
        if budget != expected_budget:
            raise ValueError(
                "search.budget must match the versioned experiment definition"
            )

        declared_campaign = config.get("size_campaign")
        if definition.size_campaign is None:
            if declared_campaign is not None:
                raise ValueError(
                    "size_campaign is not allowed for the selected experiment"
                )
        elif declared_campaign != definition.size_campaign.as_config():
            raise ValueError(
                "size_campaign must match the versioned experiment definition"
            )
        if definition.size_campaign is not None:
            edge_m = float(cube["edge_m"])
            lower_edge = definition.size_campaign.coarse_edges_m[0]
            upper_edge = definition.size_campaign.coarse_edges_m[-1]
            if not lower_edge <= edge_m <= upper_edge:
                raise ValueError(
                    "cube.edge_m must stay inside the registered size_campaign range"
                )

        declared_aligned_campaign = config.get("aligned_contact_campaign")
        if definition.aligned_contact_campaign is None:
            if declared_aligned_campaign is not None:
                raise ValueError(
                    "aligned_contact_campaign is not allowed for the selected "
                    "experiment"
                )
        else:
            if (
                declared_aligned_campaign
                != definition.aligned_contact_campaign.as_config()
            ):
                raise ValueError(
                    "aligned_contact_campaign must match the versioned "
                    "experiment definition"
                )
            campaign = definition.aligned_contact_campaign
            edge_m = float(cube["edge_m"])
            if campaign_run_context is None:
                if not any(
                    math.isclose(edge_m, edge, rel_tol=0.0, abs_tol=1e-12)
                    for edge in campaign.edges_m
                ):
                    raise ValueError(
                        "cube.edge_m must be one of the aligned-contact campaign "
                        "edges"
                    )
                expected_mass = campaign.constant_density_mass_kg(edge_m)
                if not math.isclose(
                    float(cube["mass_kg"]),
                    expected_mass,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                ):
                    raise ValueError(
                        "cube.mass_kg must match the aligned-contact "
                        "constant-density policy"
                    )
                if not math.isclose(
                    float(cube["friction"]),
                    campaign.friction,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        "cube.friction must match the aligned-contact material "
                        "policy"
                    )

        declared_far_campaign = config.get("far_hand_campaign")
        if definition.far_hand_campaign is None:
            if declared_far_campaign is not None:
                raise ValueError(
                    "far_hand_campaign is not allowed for the selected experiment"
                )
        else:
            campaign = definition.far_hand_campaign
            if declared_far_campaign != campaign.as_config():
                raise ValueError(
                    "far_hand_campaign must match the versioned experiment definition"
                )
            if campaign_run_context is None:
                for actual, expected, label in (
                    (float(cube["edge_m"]), campaign.nominal_edge_m, "edge"),
                    (float(cube["mass_kg"]), campaign.nominal_mass_kg, "mass"),
                    (float(cube["friction"]), campaign.friction, "friction"),
                ):
                    if not math.isclose(
                        actual, expected, rel_tol=1e-12, abs_tol=1e-15
                    ):
                        raise ValueError(
                            f"schema-v5 nominal cube {label} must match the "
                            "registered far-hand campaign"
                        )

        declared_high_thumb_campaign = config.get("high_thumb_size_campaign")
        if definition.high_thumb_size_campaign is None:
            if declared_high_thumb_campaign is not None:
                raise ValueError(
                    "high_thumb_size_campaign is not allowed for the selected "
                    "experiment"
                )
        else:
            campaign = definition.high_thumb_size_campaign
            if declared_high_thumb_campaign != campaign.as_config():
                raise ValueError(
                    "high_thumb_size_campaign must match the versioned "
                    "experiment definition"
                )
            if campaign_run_context is None:
                edge_m = float(cube["edge_m"])
                if not campaign.contains_edge(edge_m):
                    raise ValueError(
                        "cube.edge_m must stay inside the registered high-thumb "
                        "size range"
                    )
                fine_index = (
                    edge_m - campaign.coarse_edges_m[0]
                ) / campaign.edge_fine_step_m
                if not math.isclose(
                    fine_index, round(fine_index), rel_tol=0.0, abs_tol=1e-9
                ):
                    raise ValueError(
                        "cube.edge_m must lie on the registered fine edge grid"
                    )
                for actual, expected, label in (
                    (float(cube["mass_kg"]), campaign.fixed_mass_kg, "mass"),
                    (float(cube["friction"]), campaign.friction, "friction"),
                ):
                    if not math.isclose(
                        actual, expected, rel_tol=1e-12, abs_tol=1e-15
                    ):
                        raise ValueError(
                            f"schema-v7 cube {label} must match the registered "
                            "high-thumb campaign"
                        )
                centre = _finite_sequence(
                    cube["center_xy_m"], 2, "cube.center_xy_m"
                )
                if any(
                    not math.isclose(
                        actual, expected, rel_tol=0.0, abs_tol=1e-12
                    )
                    for actual, expected in zip(
                        centre, campaign.cube_center_xy_m
                    )
                ):
                    raise ValueError(
                        "schema-v7 cube center_xy_m must remain fixed by the "
                        "campaign"
                    )
                if not math.isclose(
                    float(cube.get("z_offset_m", 0.0)),
                    0.0,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        "schema-v7 cube z_offset_m must remain zero during tuning"
                    )

        declared_normal_campaign = config.get(
            "normal_aligned_smooth_lift_campaign"
        )
        normal_campaign = definition.normal_aligned_smooth_lift_campaign
        if normal_campaign is None:
            if declared_normal_campaign is not None:
                raise ValueError(
                    "normal_aligned_smooth_lift_campaign is not allowed for "
                    "the selected experiment"
                )
        else:
            if declared_normal_campaign != normal_campaign.as_config():
                raise ValueError(
                    "normal_aligned_smooth_lift_campaign must match the "
                    "versioned experiment definition"
                )
            if campaign_run_context is None:
                edge_m = float(cube["edge_m"])
                if not any(
                    math.isclose(edge_m, edge, abs_tol=1e-12)
                    for edge in normal_campaign.edges_m
                ):
                    raise ValueError(
                        "schema-v8 cube edge must lie on the registered 60--70 mm grid"
                    )
                for actual, expected, label in (
                    (float(cube["mass_kg"]), normal_campaign.fixed_mass_kg, "mass"),
                    (float(cube["friction"]), normal_campaign.friction, "friction"),
                ):
                    if not math.isclose(actual, expected, abs_tol=1e-12):
                        raise ValueError(
                            f"schema-v8 cube {label} must match the campaign"
                        )
                centre = _finite_sequence(
                    cube["center_xy_m"], 2, "cube.center_xy_m"
                )
                if any(
                    not math.isclose(actual, expected, abs_tol=1e-12)
                    for actual, expected in zip(
                        centre, normal_campaign.cube_center_xy_m
                    )
                ):
                    raise ValueError(
                        "schema-v8 cube center_xy_m must remain fixed"
                    )
                if not math.isclose(
                    float(cube.get("z_offset_m", 0.0)), 0.0, abs_tol=1e-12
                ):
                    raise ValueError("schema-v8 cube z_offset_m must remain zero")

        declared_actual_contact_campaign = config.get(
            "actual_contact_grasp_pose_campaign"
        )
        actual_contact_campaign = (
            definition.actual_contact_grasp_pose_campaign
        )
        if actual_contact_campaign is None:
            if declared_actual_contact_campaign is not None:
                raise ValueError(
                    "actual_contact_grasp_pose_campaign is not allowed for "
                    "the selected experiment"
                )
        else:
            if declared_actual_contact_campaign != actual_contact_campaign.as_config():
                raise ValueError(
                    "actual_contact_grasp_pose_campaign must match the "
                    "versioned experiment definition"
                )
            if campaign_run_context is None:
                edge_m = float(cube["edge_m"])
                if not any(
                    math.isclose(edge_m, edge, abs_tol=1e-12)
                    for edge in actual_contact_campaign.edges_m
                ):
                    raise ValueError(
                        f"schema-v{schema_version} cube edge must lie on the registered campaign grid"
                    )
                for actual, expected, label in (
                    (
                        float(cube["mass_kg"]),
                        actual_contact_campaign.fixed_mass_kg,
                        "mass",
                    ),
                    (
                        float(cube["friction"]),
                        actual_contact_campaign.friction,
                        "friction",
                    ),
                ):
                    if not math.isclose(actual, expected, abs_tol=1e-12):
                        raise ValueError(
                            f"schema-v{schema_version} cube {label} must match the campaign"
                        )
                centre = _finite_sequence(
                    cube["center_xy_m"], 2, "cube.center_xy_m"
                )
                if any(
                    not math.isclose(actual, expected, abs_tol=1e-12)
                    for actual, expected in zip(
                        centre, actual_contact_campaign.cube_center_xy_m
                    )
                ):
                    raise ValueError(
                        f"schema-v{schema_version} cube center_xy_m must remain fixed"
                    )
                cube_yaw = float(cube.get("rpy_deg", [0.0, 0.0, 0.0])[2])
                if not math.isclose(
                    cube_yaw,
                    actual_contact_campaign.cube_yaw_deg,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        f"schema-v{schema_version} cube yaw must remain fixed"
                    )
                if not math.isclose(
                    float(cube.get("z_offset_m", 0.0)), 0.0, abs_tol=1e-12
                ):
                    raise ValueError(
                        f"schema-v{schema_version} cube z_offset_m must remain zero"
                    )

        if schema_version == 4:
            if definition.pose_constraints is None:
                raise ValueError(
                    "schema v4 experiment definition requires pose_constraints"
                )
            if config.get("pose_constraints") != (
                definition.pose_constraints.as_config()
            ):
                raise ValueError(
                    "pose_constraints must match the versioned experiment definition"
                )
            if definition.contact_alignment is None:
                raise ValueError(
                    "schema v4 experiment definition requires contact_alignment"
                )
            if config.get("contact_alignment") != (
                definition.contact_alignment.as_config()
            ):
                raise ValueError(
                    "contact_alignment must match the versioned experiment definition"
                )
            resolved_pose = resolved_pose_constraint_values(config)
            if campaign_run_context is None:
                constraints = definition.pose_constraints
                for label in (
                    "finger_down_tilt_deg",
                    "palm_plane_ground_angle_deg",
                    "palm_press_depth_m",
                ):
                    lower, upper = getattr(constraints, label)
                    value = float(resolved_pose[label])
                    if not lower - 1e-12 <= value <= upper + 1e-12:
                        raise ValueError(
                            f"resolved {label} must stay inside pose_constraints"
                        )
                hand_rpy = _finite_sequence(
                    config["hand_pose"]["rpy_deg"], 3, "hand_pose.rpy_deg"
                )
                cube_rpy = _finite_sequence(
                    cube.get("rpy_deg", [0.0, 0.0, 0.0]), 3, "cube.rpy_deg"
                )
                for value, declared, label in (
                    (hand_rpy[0], bounds.hand_roll_deg, "hand roll"),
                    (hand_rpy[2], bounds.hand_yaw_deg, "hand yaw"),
                    (cube_rpy[2], bounds.cube_yaw_deg, "cube yaw"),
                ):
                    if not declared[0] - 1e-12 <= value <= declared[1] + 1e-12:
                        raise ValueError(
                            f"resolved {label} must stay inside registered search bounds"
                        )
                if not bounds.contains_cube_position(
                    resolved_pose["cube_position_in_root_m"]
                ):
                    raise ValueError(
                        "resolved cube_position_in_root_m must stay inside "
                        "registered search bounds"
                    )
        elif schema_version in (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16):
            constraints = definition.far_hand_pose_constraints
            preferences = definition.fingertip_contact_preferences
            campaign = definition.far_hand_campaign
            high_thumb_campaign = definition.high_thumb_size_campaign
            normal_campaign = definition.normal_aligned_smooth_lift_campaign
            actual_contact_campaign = (
                definition.actual_contact_grasp_pose_campaign
            )
            if (
                constraints is None
                or preferences is None
                or (schema_version in (5, 6) and campaign is None)
                or (schema_version == 7 and high_thumb_campaign is None)
                or (schema_version == 8 and normal_campaign is None)
                or (schema_version in (9, 10, 11, 12, 13, 14, 15, 16) and actual_contact_campaign is None)
            ):
                raise ValueError(
                    f"schema v{schema_version} experiment definition is incomplete"
                )
            if config.get("pose_constraints") != constraints.as_config():
                raise ValueError(
                    "pose_constraints must match the far-hand experiment definition"
                )
            if config.get("contact_alignment") != (
                definition.contact_alignment.as_config()
                if definition.contact_alignment is not None
                else None
            ):
                raise ValueError(
                    "contact_alignment must match the far-hand experiment definition"
                )
            if config.get("fingertip_contact_preferences") != (
                preferences.as_config()
            ):
                raise ValueError(
                    "fingertip_contact_preferences must match the far-hand definition"
                )
            resolved_pose = resolved_pose_constraint_values(config)
            thumb_name = "left_hand_thumb_bend_joint_actuator"
            grasp_thumb = float(contact_preload_targets(config)[thumb_name])
            manipulated_thumb = grasp_thumb + float(
                control["manipulation_delta_rad"][thumb_name]
            )
            if grasp_thumb < bounds.actuator_targets_rad[thumb_name][0] - 1e-12:
                raise ValueError(
                    f"schema-v{schema_version} thumb bend grasp target is too small"
                )
            if campaign is not None and manipulated_thumb < (
                campaign.minimum_manipulated_thumb_bend_rad - 1e-12
            ):
                raise ValueError(
                    f"schema-v{schema_version} manipulated thumb bend target must remain at least "
                    f"{campaign.minimum_manipulated_thumb_bend_rad} rad"
                )
            if campaign_run_context is None:
                boundary = config.get("candidate_metadata", {}).get(
                    "boundary_expansion", {}
                )
                expanded_distance = bool(
                    campaign is not None
                    and isinstance(boundary, dict)
                    and boundary.get("applied") is True
                    and boundary.get("count") == 1
                    and boundary.get("distance_expanded") is True
                )
                for label in (
                    "finger_down_tilt_deg",
                    "palm_plane_ground_angle_deg",
                    "root_cube_distance_m",
                ):
                    lower, upper = getattr(constraints, label)
                    if (
                        label == "root_cube_distance_m"
                        and expanded_distance
                        and campaign is not None
                    ):
                        upper = (
                            campaign.boundary_expansion
                            .expanded_root_cube_distance_max_m
                        )
                    value = float(resolved_pose[label])
                    if not lower - 1e-12 <= value <= upper + 1e-12:
                        raise ValueError(
                            f"resolved {label} must stay inside far-hand pose constraints"
                        )
                hand_rpy = _finite_sequence(
                    config["hand_pose"]["rpy_deg"], 3, "hand_pose.rpy_deg"
                )
                cube_rpy = _finite_sequence(
                    cube.get("rpy_deg", [0.0, 0.0, 0.0]), 3, "cube.rpy_deg"
                )
                for value, declared, label in (
                    (hand_rpy[0], bounds.hand_roll_deg, "hand roll"),
                    (hand_rpy[2], bounds.hand_yaw_deg, "hand yaw"),
                    (cube_rpy[2], bounds.cube_yaw_deg, "cube yaw"),
                ):
                    if not declared[0] - 1e-12 <= value <= declared[1] + 1e-12:
                        raise ValueError(
                            f"resolved {label} must stay inside registered search bounds"
                        )
                cube_in_root = resolved_pose["cube_position_in_root_m"]
                normal_pose = constraints.contains_cube_position(cube_in_root)
                if not normal_pose and expanded_distance and campaign is not None:
                    vector = tuple(float(value) for value in cube_in_root)
                    distance = math.sqrt(sum(value * value for value in vector))
                    expansion = campaign.boundary_expansion
                    normal_pose = bool(
                        constraints.root_cube_distance_m[0] - 1e-12
                        <= distance
                        <= expansion.expanded_root_cube_distance_max_m + 1e-12
                        and constraints.cube_position_in_root_m["x"][0] - 1e-12
                        <= vector[0]
                        <= expansion.expanded_cube_in_root_x_max_m + 1e-12
                        and constraints.cube_position_in_root_m["y"][0] - 1e-12
                        <= vector[1]
                        <= constraints.cube_position_in_root_m["y"][1] + 1e-12
                        and constraints.cube_position_in_root_m["z"][0] - 1e-12
                        <= vector[2]
                        <= constraints.cube_position_in_root_m["z"][1] + 1e-12
                    )
                if not normal_pose:
                    raise ValueError(
                        "resolved cube_position_in_root_m or root distance is "
                        "outside the far-hand search envelope"
                    )
            if schema_version in (6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16):
                if definition.pose_preservation is None:
                    raise ValueError(
                        f"schema v{schema_version} experiment definition requires "
                        "pose_preservation"
                    )
                if config.get("pose_preservation") != (
                    definition.pose_preservation.as_config()
                ):
                    raise ValueError(
                        "pose_preservation must match the versioned experiment "
                        "definition"
                    )
            if schema_version in (8, 9, 10, 11, 12, 13, 14, 15, 16):
                if definition.closure_alignment is None:
                    raise ValueError(
                        f"schema v{schema_version} experiment definition requires closure_alignment"
                    )
                if config.get("closure_alignment") != (
                    definition.closure_alignment.as_config()
                ):
                    raise ValueError(
                        "closure_alignment must match the versioned experiment definition"
                    )
                if definition.motion_smoothness is None:
                    raise ValueError(
                        f"schema v{schema_version} experiment definition requires motion_smoothness"
                    )
                if config.get("motion_smoothness") != (
                    definition.motion_smoothness.as_config()
                ):
                    raise ValueError(
                        "motion_smoothness must match the versioned experiment definition"
                    )
            if schema_version in (9, 10, 11, 12, 13, 14, 15, 16):
                grasp_pose_settings = definition.actual_contact_grasp_pose
                if grasp_pose_settings is None:
                    raise ValueError(
                        f"schema v{schema_version} experiment definition requires actual grasp-pose settings"
                    )
                grasp_pose = config.get("grasp_pose")
                if not isinstance(grasp_pose, dict):
                    raise ValueError(f"schema v{schema_version} requires grasp_pose")
                expected_fields = {
                    "nominal_joint_qpos_rad",
                    *grasp_pose_settings.as_config().keys(),
                }
                if set(grasp_pose) != expected_fields:
                    raise ValueError(
                        "grasp_pose must contain nominal_joint_qpos_rad and the "
                        "registered actual-contact acceptance fields"
                    )
                grasp_pose_settings.validate_nominal_joint_qpos_rad(
                    grasp_pose["nominal_joint_qpos_rad"]
                )
                declared_settings = dict(grasp_pose)
                declared_settings.pop("nominal_joint_qpos_rad")
                if declared_settings != grasp_pose_settings.as_config():
                    raise ValueError(
                        "grasp_pose thresholds must match the versioned experiment definition"
                    )
        elif "pose_constraints" in config or "contact_alignment" in config:
            raise ValueError(
                "pose_constraints and contact_alignment are only allowed in "
                "schema v4 and later"
            )
        if schema_version not in (5, 6) and "far_hand_campaign" in config:
            raise ValueError(
                "far_hand_campaign is only allowed in schema v5 or v6"
            )
        if schema_version not in (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16) and (
            "fingertip_contact_preferences" in config
        ):
            raise ValueError(
                "fingertip contact preferences are only allowed in schema v5 "
                "and later"
            )
        if schema_version not in (6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16) and "pose_preservation" in config:
            raise ValueError(
                "pose_preservation is only allowed in schema v6 and later"
            )
        if schema_version not in (8, 9, 10, 11, 12, 13, 14, 15, 16) and (
            "closure_alignment" in config or "motion_smoothness" in config
        ):
            raise ValueError(
                "closure_alignment and motion_smoothness are only allowed in schema v8 and later"
            )
        if schema_version not in (9, 10, 11, 12, 13, 14, 15, 16) and "grasp_pose" in config:
            raise ValueError("grasp_pose is only allowed in actual-contact schemas")

        if schema_version == 10:
            # Keep the schema-v10 size bridge versioned and fail closed before
            # creating a campaign workspace.  The import is deliberately
            # local: core configuration remains independent of tuning at
            # module-import time.
            from .tuning.actual_contact_size_continuation import (
                SizeContinuationPolicy,
            )

            SizeContinuationPolicy.from_config(config)
        elif "size_continuation" in config:
            raise ValueError(
                "size_continuation is only allowed in schema v10"
            )

        relative_search = definition.relative_wrist_pose_search
        declared_relative_search = config.get("relative_wrist_pose_search")
        if schema_version == 11:
            if relative_search is None:
                raise ValueError(
                    "schema v11 experiment definition requires "
                    "relative_wrist_pose_search"
                )
            campaign = definition.actual_contact_grasp_pose_campaign
            assert campaign is not None
            expected_relative_search = relative_search.as_config(
                edge_count=len(campaign.edges_m),
                thumb_band_count=len(campaign.thumb_actual_centers_rad),
            )
            if declared_relative_search != expected_relative_search:
                raise ValueError(
                    "relative_wrist_pose_search must match the versioned "
                    "experiment definition"
                )
        elif declared_relative_search is not None:
            raise ValueError(
                "relative_wrist_pose_search is only allowed in schema v11"
            )

        registered_contact_search = definition.contact_point_search
        declared_contact_search = config.get("contact_point_search")
        declared_contact_plan = config.get("contact_point_plan")
        if schema_version == 12:
            if registered_contact_search is None:
                raise ValueError(
                    "schema v12 experiment definition requires contact_point_search"
                )
            if declared_contact_search != registered_contact_search.as_config():
                raise ValueError(
                    "contact_point_search must match the versioned experiment definition"
                )
            if not isinstance(declared_contact_plan, dict):
                raise ValueError("schema v12 requires contact_point_plan")
            from .experiment import ContactPointPlanParameters

            selected_plan = ContactPointPlanParameters.from_config(
                declared_contact_plan
            )
            registered_contact_search.validate_selected_plan(selected_plan)
            if not math.isclose(
                selected_plan.cube_edge_m, float(cube["edge_m"]), abs_tol=1e-12
            ):
                raise ValueError(
                    "contact_point_plan cube edge must match the configured cube"
                )
            if selected_plan.target_faces.as_dict() != target_faces:
                raise ValueError(
                    "contact_point_plan faces must match contact_topology.target_faces"
                )
        elif schema_version in (13, 14, 15, 16):
            if registered_contact_search is not None:
                raise ValueError(
                    f"schema v{schema_version} must use a frozen point plan without "
                    "contact_point_search"
                )
            if declared_contact_search is not None:
                raise ValueError("contact_point_search is only allowed in schema v12")
            if not isinstance(declared_contact_plan, dict):
                raise ValueError(
                    f"schema v{schema_version} requires contact_point_plan"
                )
            from .experiment import ContactPointPlanParameters

            selected_plan = ContactPointPlanParameters.from_config(
                declared_contact_plan
            )
            if not math.isclose(
                selected_plan.cube_edge_m,
                float(cube["edge_m"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "contact_point_plan cube edge must match the configured cube"
                )
            if selected_plan.target_faces.as_dict() != target_faces:
                raise ValueError(
                    "contact_point_plan faces must match contact_topology.target_faces"
                )
        elif declared_contact_search is not None or declared_contact_plan is not None:
            raise ValueError(
                "contact_point_search and contact_point_plan are only allowed in schema v12"
            )

        registered_downsize = definition.scaled_contact_downsize_campaign
        declared_downsize = config.get("scaled_contact_downsize_campaign")
        if schema_version == 13:
            if registered_downsize is None:
                raise ValueError(
                    "schema v13 experiment definition requires "
                    "scaled_contact_downsize_campaign"
                )
            if declared_downsize != registered_downsize.as_config():
                raise ValueError(
                    "scaled_contact_downsize_campaign must match the versioned "
                    "experiment definition"
                )
            declared_mapping = config.get("scaled_contact_mapping")
            if not isinstance(declared_mapping, dict):
                raise ValueError("schema v13 requires scaled_contact_mapping")
            from .experiment import ScaledContactMappingParameters

            mapping = ScaledContactMappingParameters.from_config(
                declared_mapping
            )
            mapping.validate_for_campaign(registered_downsize, selected_plan)
            edge_m = float(cube["edge_m"])
            if not any(
                math.isclose(edge_m, edge, rel_tol=0.0, abs_tol=1e-12)
                for edge in registered_downsize.edges_m
            ):
                raise ValueError(
                    "schema-v13 cube edge must lie on the registered downsize grid"
                )
            if campaign_run_context == "robustness_trial":
                _validate_v13_robustness_trial(config, definition)
            else:
                for actual, expected, label in (
                    (
                        float(cube["mass_kg"]),
                        registered_downsize.fixed_mass_kg,
                        "mass",
                    ),
                    (
                        float(cube["friction"]),
                        registered_downsize.friction,
                        "friction",
                    ),
                ):
                    if not math.isclose(
                        actual, expected, rel_tol=0.0, abs_tol=1e-12
                    ):
                        raise ValueError(
                            f"schema-v13 cube {label} must match the downsize campaign"
                        )
                if not math.isclose(
                    float(cube.get("z_offset_m", 0.0)),
                    0.0,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError("schema-v13 cube z_offset_m must remain zero")
        elif declared_downsize is not None:
            raise ValueError(
                "scaled_contact_downsize_campaign is only allowed in schema v13"
            )
        elif "scaled_contact_mapping" in config:
            raise ValueError("scaled_contact_mapping is only allowed in schema v13")

        registered_planned_lift = (
            definition.contact_preserving_planned_lift_campaign
        )
        declared_planned_lift = config.get(
            "contact_preserving_planned_lift_campaign"
        )
        planned_blocks = (
            "manipulation_plan",
            "contact_force_targets_n",
            "contact_feedback",
        )
        joint_pair_blocks = (
            "joint_pair_alignment",
            "joint_pair_feedback",
        )
        if schema_version in (14, 15, 16):
            if registered_planned_lift is None:
                raise ValueError(
                    f"schema v{schema_version} experiment definition requires "
                    "contact_preserving_planned_lift_campaign"
                )
            if declared_planned_lift != registered_planned_lift.as_config():
                raise ValueError(
                    "contact_preserving_planned_lift_campaign must match the "
                    "versioned experiment definition"
                )
            from .experiment import (
                ContactFeedbackParameters,
                ContactForceTargets,
                ManipulationPlanParameters,
            )

            plan = ManipulationPlanParameters.from_config(
                config.get("manipulation_plan", {})
            )
            force_targets = ContactForceTargets.from_config(
                config.get("contact_force_targets_n", {})
            )
            feedback = ContactFeedbackParameters.from_config(
                config.get("contact_feedback", {})
            )
            registered_planned_lift.validate_candidate(
                plan, force_targets, feedback
            )
            if schema_version == 14:
                unexpected_joint_pair = [
                    name for name in joint_pair_blocks if name in config
                ]
                if unexpected_joint_pair:
                    raise ValueError(
                        "joint-pair controller blocks are only allowed in schema v15 or v16"
                    )
            else:
                from .experiment import (
                    JointPairAlignmentSettings,
                    JointPairFeedbackParameters,
                )

                registered_alignment = definition.joint_pair_alignment
                registered_pair_feedback = definition.joint_pair_feedback
                if registered_alignment is None or registered_pair_feedback is None:
                    raise ValueError(
                        f"schema v{schema_version} experiment definition requires "
                        "joint-pair settings"
                    )
                alignment = JointPairAlignmentSettings.from_config(
                    config.get("joint_pair_alignment", {})
                )
                if alignment != registered_alignment:
                    raise ValueError(
                        "joint_pair_alignment must match the versioned experiment definition"
                    )
                pair_feedback = JointPairFeedbackParameters.from_config(
                    config.get("joint_pair_feedback", {})
                )
                expected_feedback_schema = 1 if schema_version == 15 else 2
                if pair_feedback.schema_version != expected_feedback_schema:
                    raise ValueError(
                        f"schema v{schema_version} requires joint-pair feedback "
                        f"schema {expected_feedback_schema}"
                    )
                candidate_fixed = pair_feedback.as_config()
                registered_fixed = registered_pair_feedback.as_config()
                for key in (
                    "feedback_id",
                    "alignment_gain",
                    "slip_recovery_gain_rad_per_m",
                ):
                    candidate_fixed.pop(key)
                    registered_fixed.pop(key)
                if candidate_fixed != registered_fixed:
                    raise ValueError(
                        "joint_pair_feedback fixed contract must match the "
                        "versioned experiment definition"
                    )
                alignment_options = (
                    registered_planned_lift.alignment_gain_options
                )
                slip_options = (
                    registered_planned_lift.slip_recovery_gain_options_rad_per_m
                )
                assert alignment_options is not None and slip_options is not None
                if not (
                    alignment_options[0] - 1e-12
                    <= pair_feedback.alignment_gain
                    <= alignment_options[-1] + 1e-12
                ):
                    raise ValueError(
                        "joint_pair_feedback.alignment_gain is outside the registered range"
                    )
                if not (
                    slip_options[0] - 1e-12
                    <= pair_feedback.slip_recovery_gain_rad_per_m
                    <= slip_options[-1] + 1e-12
                ):
                    raise ValueError(
                        "joint_pair_feedback slip recovery gain is outside the registered range"
                    )
                if not math.isclose(
                    pair_feedback.freeze_threshold_deg,
                    alignment.operation_p95_max_deg,
                    abs_tol=1e-12,
                ) or not math.isclose(
                    pair_feedback.abort_threshold_deg,
                    alignment.operation_max_deg,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        "joint-pair feedback angles must match alignment acceptance"
                    )
            protocol = definition.control_protocol
            assert protocol is not None
            if not math.isclose(
                plan.duration_s,
                protocol.manipulate_s,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "manipulation_plan duration must match control_protocol.manipulate_s"
                )
            for actuator in ACTIVE_ACTUATORS:
                if not math.isclose(
                    plan.actuator_waypoints_rad[actuator][-1],
                    float(control["manipulation_delta_rad"][actuator]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        "manipulation_plan terminal waypoint must match "
                        "control.manipulation_delta_rad"
                    )
            edge_m = float(cube["edge_m"])
            if not any(
                math.isclose(edge_m, edge, rel_tol=0.0, abs_tol=1e-12)
                for edge in registered_planned_lift.edges_m
            ):
                raise ValueError(
                    f"schema-v{schema_version} cube edge must lie on the registered grid"
                )
            if campaign_run_context is None:
                for actual, expected, label in (
                    (
                        float(cube["mass_kg"]),
                        registered_planned_lift.fixed_mass_kg,
                        "mass",
                    ),
                    (
                        float(cube["friction"]),
                        registered_planned_lift.friction,
                        "friction",
                    ),
                ):
                    if not math.isclose(
                        actual, expected, rel_tol=0.0, abs_tol=1e-12
                    ):
                        raise ValueError(
                            f"schema-v{schema_version} cube {label} must match the planned-lift campaign"
                        )
            if force_targets.minimum_n < float(acceptance["contact_force_min_n"]):
                raise ValueError(
                    "contact force target clamp cannot be below hard contact force acceptance"
                )
            # The tune template intentionally carries no top-level IDs, and
            # pre-planning source-pair intermediates carry only the three
            # directly recomputable pair IDs.  Any runnable planned config
            # must carry the complete five-ID chain.  Import locally to keep
            # legacy schema import/validation paths independent of v14.
            if schema_version == 14:
                from .v14_identity import validate_v14_top_level_identities

                validate_v14_top_level_identities(config)
            elif schema_version == 15:
                from .v15_identity import validate_v15_top_level_identities

                validate_v15_top_level_identities(config)
            else:
                from .v16_identity import validate_v16_top_level_identities

                validate_v16_top_level_identities(config)
        else:
            unexpected = [name for name in planned_blocks if name in config]
            unexpected.extend(
                name for name in joint_pair_blocks if name in config
            )
            if declared_planned_lift is not None or unexpected:
                raise ValueError(
                    "planned-lift campaign and controller blocks are only allowed "
                    "in schema v14, v15 or v16"
                )

        if config.get("robustness") != definition.robustness.as_config():
            raise ValueError(
                "robustness must match the versioned experiment definition"
            )


def load_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_config(config)
    return config
