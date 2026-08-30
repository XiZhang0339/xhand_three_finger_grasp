"""Trace evaluation shared by legacy and versioned grasp experiments."""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np

from .config import (
    ACTIVE_ACTUATORS,
    ACTIVE_FINGERS,
    contact_preload_targets,
    precontact_targets,
    resolved_pose_constraint_values,
)
from .grasp_pose import (
    controller_id,
    evaluate_actual_grasp_pose_trace,
    grasp_pose_id,
)
from .contacts import (
    FACE_ORDER,
    Face,
    target_face_contact_centroids,
    three_finger_height_spread,
)
from .contact_point_targeting import (
    contact_point_observation,
    contact_point_plan_from_config,
)
from .contact_slip import contact_tangent_slip_from_grasp
from .joint_pair_geometry import joint_pair_telemetry, resolve_joint_pair
from .scene import ModelInfo
from .trajectory import interpolate_quintic_c2, quintic_c2_knot_derivatives
from .v14_identity import (
    V14_TOP_LEVEL_ID_FIELDS,
    validate_v14_top_level_identities,
)


_FACE_LABELS = {
    "+x": Face.X_POS,
    "x+": Face.X_POS,
    "x_pos": Face.X_POS,
    "-x": Face.X_NEG,
    "x-": Face.X_NEG,
    "x_neg": Face.X_NEG,
    "+y": Face.Y_POS,
    "y+": Face.Y_POS,
    "y_pos": Face.Y_POS,
    "-y": Face.Y_NEG,
    "y-": Face.Y_NEG,
    "y_neg": Face.Y_NEG,
    "+z": Face.Z_POS,
    "z+": Face.Z_POS,
    "z_pos": Face.Z_POS,
    "-z": Face.Z_NEG,
    "z-": Face.Z_NEG,
    "z_neg": Face.Z_NEG,
}


def face_from_label(value: str | Face) -> Face:
    if isinstance(value, Face):
        return value
    key = str(value).strip().lower()
    try:
        return _FACE_LABELS[key]
    except KeyError as error:
        raise ValueError(f"unknown cube face label: {value!r}") from error


def orientation_angles(reference: np.ndarray, quaternions: np.ndarray) -> np.ndarray:
    reference = reference / np.linalg.norm(reference)
    normalized = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)
    dots = np.clip(np.abs(normalized @ reference), 0.0, 1.0)
    return 2.0 * np.arccos(dots)


def _longest_true_run_steps(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    longest = current = 0
    for value in mask:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _longest_true_run_seconds(mask: np.ndarray, timestep_s: float) -> float:
    return float(_longest_true_run_steps(mask) * timestep_s)


def _longest_false_run_steps(mask: np.ndarray) -> int:
    """Return the longest consecutive loss in an effective-contact mask."""

    return _longest_true_run_steps(~np.asarray(mask, dtype=bool))


def _v16_rolling_aware_target_face_evidence(
    config: dict[str, Any],
    traces: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Recompute schema-v16 contact evidence from physical trace fields.

    A tactile taxel remains mandatory while acquiring the grasp.  Once the
    controller enters MANIPULATE/HOLD, a real distal collision patch on the
    requested cube face may bridge a taxel dropout.  The replacement evidence
    is deliberately strict: target-face force and purity must still pass, the
    rolling estimator must see sufficient real normal force, and material
    off-target or active non-distal contact always rejects the sample.

    The returned ``effective`` mask is state-aware (native before operation,
    rolling-aware during operation).  ``native`` and ``rolling_physical`` are
    exposed separately so persisted caches cannot become acceptance authority.
    """

    if int(config.get("schema_version", 1)) < 16:
        raise ValueError("rolling-aware contact evidence requires schema_version >= 16")

    total_steps = int(np.asarray(traces["time"]).shape[0])
    states = np.asarray(traces["control_state"]).astype(str)
    face_force = np.asarray(traces["distal_face_force_n"], dtype=np.float64)
    nondistal = np.asarray(
        traces["active_nondistal_force_n"], dtype=np.float64
    )
    tactile = np.asarray(traces["tactile_max"], dtype=np.float64)
    rolling_valid = np.asarray(traces["rolling_contact_valid"], dtype=bool)
    rolling_normal_force = np.asarray(
        traces["rolling_contact_normal_force_n"], dtype=np.float64
    )
    expected_face_shape = (total_steps, 3, len(FACE_ORDER))
    expected_finger_shape = (total_steps, 3)
    if states.shape != (total_steps,):
        raise ValueError("schema-v16 control state trace has invalid shape")
    if face_force.shape != expected_face_shape:
        raise ValueError("schema-v16 distal face force trace has invalid shape")
    if nondistal.shape != expected_finger_shape:
        raise ValueError("schema-v16 active non-distal force trace has invalid shape")
    if tactile.shape != (total_steps, 5):
        raise ValueError("schema-v16 tactile trace has invalid shape")
    if rolling_valid.shape != expected_finger_shape:
        raise ValueError("schema-v16 rolling contact valid trace has invalid shape")
    if rolling_normal_force.shape != expected_finger_shape:
        raise ValueError("schema-v16 rolling contact force trace has invalid shape")
    if (
        not np.isfinite(face_force).all()
        or not np.isfinite(nondistal).all()
        or not np.isfinite(tactile).all()
        or not np.isfinite(rolling_normal_force).all()
        or np.any(face_force < 0.0)
        or np.any(nondistal < 0.0)
        or np.any(rolling_normal_force < 0.0)
    ):
        raise ValueError(
            "schema-v16 contact evidence must be finite and non-negative"
        )

    topology = config["contact_topology"]
    gate = config["control_protocol"]["grasp_gate"]
    target_indices = np.asarray(
        [
            FACE_ORDER.index(face_from_label(topology["target_faces"][finger]))
            for finger in ACTIVE_FINGERS
        ],
        dtype=np.int64,
    )
    target_force = face_force[:, np.arange(3), target_indices]
    total_distal = np.sum(face_force, axis=2)
    purity = np.divide(
        target_force,
        total_distal,
        out=np.zeros_like(target_force),
        where=total_distal > 0.0,
    )
    force_min_n = float(gate["min_target_face_force_n"])
    purity_min = float(gate["min_target_force_fraction"])
    force_and_purity = (target_force >= force_min_n) & (
        purity >= purity_min
    )
    native = force_and_purity.copy()
    if bool(gate["require_touch"]):
        native &= tactile[:, :3] >= float(
            config["acceptance"]["touch_force_min_n"]
        )

    off_target = np.maximum(0.0, total_distal - target_force)
    off_fraction = np.divide(
        off_target,
        total_distal,
        out=np.zeros_like(off_target),
        where=total_distal > 0.0,
    )
    maximum_off_fraction = 1.0 - purity_min
    material_off_target = (off_target >= force_min_n) & (
        off_fraction > maximum_off_fraction + 1e-12
    )
    combined = total_distal + nondistal
    nondistal_fraction = np.divide(
        nondistal,
        combined,
        out=np.zeros_like(nondistal),
        where=combined > 0.0,
    )
    material_active_nondistal = (nondistal >= force_min_n) & (
        nondistal_fraction > maximum_off_fraction + 1e-12
    )
    safe_material = ~material_off_target & ~material_active_nondistal
    rolling_physical = (
        force_and_purity
        & rolling_valid
        & (rolling_normal_force >= force_min_n)
        & safe_material
    )
    # A native taxel is itself valid physical pad evidence.  During operation
    # it may be ORed with the real distal rolling patch, but both paths remain
    # subject to the same force/purity and material-exclusion predicates.
    operation_effective = (
        force_and_purity & safe_material & (native | rolling_physical)
    )
    operation_mask = (states == "MANIPULATE") | (states == "HOLD")
    effective = native.copy()
    effective[operation_mask] = operation_effective[operation_mask]
    return {
        "native": native,
        "rolling_physical": rolling_physical,
        "operation_effective": operation_effective,
        "effective": effective,
        "operation_mask": operation_mask,
        "target_force_n": target_force,
        "target_force_purity": purity,
        "material_off_target": material_off_target,
        "material_active_nondistal": material_active_nondistal,
    }


def _planned_command_total_correction_rad(
    schema_version: int,
    traces: dict[str, np.ndarray],
    expected_shape: tuple[int, int],
) -> np.ndarray:
    """Compose versioned controller corrections without changing old paths."""

    correction = np.asarray(traces["feedback_correction_rad"], dtype=np.float64)
    if correction.shape != expected_shape:
        raise ValueError("schema-v14 force correction trace has invalid shape")
    total = correction
    if int(schema_version) >= 15:
        pair = np.asarray(
            traces["joint_pair_feedback_correction_rad"], dtype=np.float64
        )
        if pair.shape != expected_shape:
            raise ValueError("schema-v15 joint-pair correction trace has invalid shape")
        total = total + pair
    if int(schema_version) >= 16:
        rolling = np.asarray(
            traces["rolling_slip_correction_rad"], dtype=np.float64
        )
        if rolling.shape != expected_shape:
            raise ValueError("schema-v16 rolling-slip correction trace has invalid shape")
        total = total + rolling
    return total


def _versioned_operation_feedback_risk(
    schema_version: int,
    contact_risk: np.ndarray,
    *,
    pair_risk: np.ndarray,
    legacy_slip_risk: np.ndarray,
    pair_legacy_slip_risk: np.ndarray,
    rolling_slip_freeze_active: np.ndarray,
) -> np.ndarray:
    """Compose the exact risk sources used by each controller generation."""

    base = np.asarray(contact_risk, dtype=bool)
    expected_shape = base.shape
    sources = {
        "pair_risk": pair_risk,
        "legacy_slip_risk": legacy_slip_risk,
        "pair_legacy_slip_risk": pair_legacy_slip_risk,
        "rolling_slip_freeze_active": rolling_slip_freeze_active,
    }
    normalized: dict[str, np.ndarray] = {}
    for name, value in sources.items():
        array = np.asarray(value, dtype=bool)
        if array.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")
        normalized[name] = array

    risk = base | normalized["pair_risk"]
    if int(schema_version) >= 16:
        # v16 explicitly disables both legacy centroid-slip branches.
        return risk | normalized["rolling_slip_freeze_active"]
    risk |= normalized["legacy_slip_risk"]
    if int(schema_version) >= 15:
        risk |= normalized["pair_legacy_slip_risk"]
    return risk


def _v2_face_metrics(
    model: mujoco.MjModel,
    config: dict[str, Any],
    phase_steps: dict[str, int],
    traces: dict[str, np.ndarray],
    *,
    hold_start_override: int | None = None,
    material_start_override: int | None = None,
) -> tuple[dict[str, Any], dict[str, bool]]:
    acceptance = config["acceptance"]
    topology = config["contact_topology"]
    total_steps = traces["time"].shape[0]
    hold_steps = phase_steps["hold"]
    hold_start = (
        total_steps - hold_steps
        if hold_start_override is None
        else int(hold_start_override)
    )
    if not 0 <= hold_start < total_steps:
        raise ValueError("hold_start must identify a non-empty trace interval")
    hold_slice = slice(hold_start, total_steps)

    target_faces = topology["target_faces"]
    target_indices = np.asarray(
        [FACE_ORDER.index(face_from_label(target_faces[finger])) for finger in ACTIVE_FINGERS],
        dtype=int,
    )
    all_face_force = np.asarray(traces["distal_face_force_n"], dtype=np.float64)
    if all_face_force.shape != (total_steps, 3, len(FACE_ORDER)):
        raise ValueError(
            "distal_face_force_n must have shape "
            f"({total_steps}, 3, {len(FACE_ORDER)})"
        )
    finger_index = np.arange(3)
    target_force = all_face_force[:, finger_index, target_indices]
    total_distal_force = np.sum(all_face_force, axis=2)
    off_target_force = np.maximum(0.0, total_distal_force - target_force)
    purity = np.divide(
        target_force,
        total_distal_force,
        out=np.zeros_like(target_force),
        where=total_distal_force > 0.0,
    )

    force_min = float(acceptance["contact_force_min_n"])
    touch_min = float(acceptance["touch_force_min_n"])
    purity_min = float(topology["target_force_fraction"])
    tactile = np.asarray(traces["tactile_max"], dtype=np.float64)[:, :3]
    target_effective = (
        (target_force >= force_min)
        & (tactile >= touch_min)
        & (purity >= purity_min)
    )
    if int(config.get("schema_version", 1)) >= 16:
        # VERIFY remains native-touch gated inside the state-aware helper;
        # only MANIPULATE/HOLD may use a real rolling distal patch to bridge
        # a taxel dropout.
        target_effective = _v16_rolling_aware_target_face_evidence(
            config, traces
        )["effective"]
    hold_effective = target_effective[hold_slice]
    target_duty = np.mean(hold_effective, axis=0)
    simultaneous_duty = float(np.mean(np.all(hold_effective, axis=1)))

    support_or_floor = np.asarray(traces["support_contact"], dtype=bool) | np.asarray(
        traces["floor_contact"], dtype=bool
    )
    support_indices = np.flatnonzero(support_or_floor)
    unsupported_start = int(support_indices[-1] + 1) if support_indices.size else 0
    unsupported_start = min(unsupported_start, total_steps)
    material_start = (
        unsupported_start
        if material_start_override is None
        else int(material_start_override)
    )
    if not 0 <= material_start <= total_steps:
        raise ValueError(
            "material_start_override must identify a valid trace boundary"
        )
    material_interval = slice(material_start, total_steps)

    off_fraction = np.divide(
        off_target_force,
        total_distal_force,
        out=np.zeros_like(off_target_force),
        where=total_distal_force > 0.0,
    )
    material_off = (off_target_force >= force_min) & (
        off_fraction > float(topology["max_off_target_force_fraction"])
    )
    active_nondistal = np.asarray(
        traces["active_nondistal_force_n"], dtype=np.float64
    )
    nondistal_fraction = np.divide(
        active_nondistal,
        total_distal_force + active_nondistal,
        out=np.zeros_like(active_nondistal),
        where=(total_distal_force + active_nondistal) > 0.0,
    )
    material_nondistal = (active_nondistal >= force_min) & (
        nondistal_fraction > float(topology["max_off_target_force_fraction"])
    )

    interval_off = material_off[material_interval]
    interval_nondistal = material_nondistal[material_interval]
    any_off = (
        np.any(interval_off, axis=1)
        if interval_off.shape[0]
        else np.zeros(0, dtype=bool)
    )
    any_nondistal = (
        np.any(interval_nondistal, axis=1)
        if interval_nondistal.shape[0]
        else np.zeros(0, dtype=bool)
    )
    timestep = float(model.opt.timestep)
    max_duty = float(topology["max_material_off_target_duty"])
    max_run = float(topology["max_material_off_target_run_s"])
    palm_angles = np.asarray(traces["palm_down_angle_deg"], dtype=np.float64)

    on_contact_purity_min: dict[str, float] = {}
    for index, finger in enumerate(ACTIVE_FINGERS):
        active_mask = total_distal_force[hold_slice, index] >= force_min
        values = purity[hold_slice, index][active_mask]
        on_contact_purity_min[finger] = float(np.min(values)) if values.size else 0.0

    metrics = {
        "target_faces": {finger: str(target_faces[finger]) for finger in ACTIVE_FINGERS},
        "max_palm_down_angle_deg": float(np.max(palm_angles)),
        "target_face_contact_duty": {
            finger: float(target_duty[index])
            for index, finger in enumerate(ACTIVE_FINGERS)
        },
        "target_face_simultaneous_duty": simultaneous_duty,
        "target_force_purity_min_on_contact": on_contact_purity_min,
        "peak_target_face_force_n": {
            finger: float(np.max(target_force[hold_slice, index]))
            for index, finger in enumerate(ACTIVE_FINGERS)
        },
        "unsupported_interval_start_step": unsupported_start,
        "support_cleared_before_hold": unsupported_start <= hold_start,
        "material_off_target_duty": float(np.mean(any_off)) if any_off.size else 0.0,
        "material_off_target_longest_run_s": _longest_true_run_seconds(
            any_off, timestep
        ),
        "material_active_nondistal_duty": (
            float(np.mean(any_nondistal)) if any_nondistal.size else 0.0
        ),
        "material_active_nondistal_longest_run_s": _longest_true_run_seconds(
            any_nondistal, timestep
        ),
    }
    if material_start_override is not None:
        metrics.update(
            {
                "material_contact_interval_start_step": material_start,
                "material_contact_interval_scope": "operation_and_hold",
            }
        )
    checks = {
        "palm_faces_down": metrics["max_palm_down_angle_deg"]
        <= float(acceptance["max_palm_down_angle_deg"]) + 1e-12,
        "support_cleared_before_hold": bool(metrics["support_cleared_before_hold"]),
        "thumb_target_face_contact_duty": target_duty[0] + 1e-12
        >= float(acceptance["finger_contact_duty"]),
        "index_target_face_contact_duty": target_duty[1] + 1e-12
        >= float(acceptance["finger_contact_duty"]),
        "middle_target_face_contact_duty": target_duty[2] + 1e-12
        >= float(acceptance["finger_contact_duty"]),
        "simultaneous_target_face_topology": simultaneous_duty + 1e-12
        >= float(acceptance["simultaneous_contact_duty"]),
        "off_target_contacts_within_limit": metrics["material_off_target_duty"]
        <= max_duty + 1e-12
        and metrics["material_off_target_longest_run_s"] <= max_run + 1e-12,
        "active_nondistal_contacts_within_limit": (
            not bool(topology["forbid_active_nondistal"])
            or (
                metrics["material_active_nondistal_duty"] <= max_duty + 1e-12
                and metrics["material_active_nondistal_longest_run_s"]
                <= max_run + 1e-12
            )
        ),
    }
    return metrics, checks


def _trace_scalar_int(traces: dict[str, np.ndarray], name: str) -> int:
    """Read one persisted event index without allowing an ambiguous vector."""

    value = np.asarray(traces[name])
    if value.size != 1:
        raise ValueError(f"{name} must contain exactly one scalar event index")
    return int(value.reshape(()))


def _protocol_steps(model: mujoco.MjModel, config: dict[str, Any]) -> dict[str, int]:
    """Resolve the fixed v3 protocol without importing trajectory internals."""

    protocol = config["control_protocol"]
    timestep = float(model.opt.timestep)
    fields = (
        ("settle", "settle_s"),
        ("close", "close_s"),
        ("verify", "verify_timeout_s"),
        ("manipulate", "manipulate_s"),
        ("hold", "min_hold_s"),
    )
    result: dict[str, int] = {}
    for phase, field in fields:
        seconds = float(protocol[field])
        count = int(round(seconds / timestep))
        if count <= 0 or abs(count * timestep - seconds) > 0.5 * timestep + 1e-12:
            raise ValueError(f"control_protocol.{field} does not align with timestep")
        result[phase] = count
    return result


def _reconstruct_v3_controller_trace(
    model: mujoco.MjModel,
    config: dict[str, Any],
    gate: np.ndarray,
    gate_order: tuple[str, ...],
    *,
    forced_acquisition_step: int | None = None,
) -> dict[str, Any]:
    """Reconstruct the controller state solely from persisted gate evidence."""

    from .controller import grasp_gate_order

    expected_gate_order = grasp_gate_order(int(config["schema_version"]))
    if gate_order != expected_gate_order:
        if int(config["schema_version"]) == 3:
            raise ValueError(
                "grasp_gate_order must equal the canonical v3 gate axis"
            )
        raise ValueError(
            "grasp_gate_order must equal the canonical schema-v"
            f"{config['schema_version']} gate axis"
        )
    phase = _protocol_steps(model, config)
    total_steps = sum(phase.values())
    if gate.shape != (total_steps, len(expected_gate_order)):
        raise ValueError(
            "grasp_gate must have shape "
            f"({total_steps}, {len(expected_gate_order)})"
        )
    stable_steps = int(
        round(float(config["control_protocol"]["stable_window_s"]) / model.opt.timestep)
    )
    settle_end = phase["settle"]
    close_end = settle_end + phase["close"]
    verify_end = close_end + phase["verify"]
    hard_abort_indices = np.asarray(
        [
            expected_gate_order.index(name)
            for name in (
                "no_forbidden_contact",
                "palm_down",
                "penetration_within_limit",
                "finite",
                "joint_limits_respected",
                "inactive_controls_zero",
            )
        ],
        dtype=int,
    )

    states = np.empty(total_steps, dtype="<U10")
    consecutive = np.zeros(total_steps, dtype=np.int64)
    acquired_latch = np.zeros(total_steps, dtype=bool)
    progress = np.zeros(total_steps, dtype=np.float64)
    counter = 0
    acquired = False
    aborted = False
    acquisition_step = -1
    manipulation_start = -1
    manipulation_end = -1
    termination_step = -1

    for step in range(total_steps):
        if aborted:
            state = "ABORT"
        elif step < settle_end:
            state = "SETTLE"
        elif step < close_end:
            state = "CLOSE"
        elif not acquired:
            state = "VERIFY"
        else:
            offset = step - manipulation_start
            if offset < phase["manipulate"]:
                state = "MANIPULATE"
                alpha = min(1.0, max(0.0, (offset + 1) / phase["manipulate"]))
                if (
                    int(config["schema_version"]) >= 8
                    and config["control_protocol"].get(
                        "manipulation_profile"
                    )
                    == "minimum_jerk_quintic"
                ):
                    progress[step] = alpha**3 * (
                        10.0 - 15.0 * alpha + 6.0 * alpha**2
                    )
                else:
                    progress[step] = alpha * alpha * (3.0 - 2.0 * alpha)
            else:
                state = "HOLD"
                progress[step] = 1.0
        states[step] = state

        hard_abort = not bool(np.all(gate[step, hard_abort_indices]))
        if hard_abort and not aborted:
            aborted = True
            termination_step = step
            counter = 0
        else:
            if state == "VERIFY":
                counter = counter + 1 if bool(np.all(gate[step])) else 0
                may_acquire = (
                    counter >= stable_steps
                    if forced_acquisition_step is None
                    else (
                        step == forced_acquisition_step
                        and counter >= stable_steps
                    )
                )
                if may_acquire:
                    acquired = True
                    acquisition_step = step
                    manipulation_start = step + 1
                elif step >= verify_end - 1:
                    aborted = True
                    termination_step = step
            if (
                state == "MANIPULATE"
                and manipulation_start >= 0
                and step == manipulation_start + phase["manipulate"] - 1
            ):
                manipulation_end = step

        consecutive[step] = counter
        acquired_latch[step] = acquired

    if termination_step < 0:
        termination_step = total_steps - 1
    return {
        "states": states,
        "consecutive": consecutive,
        "acquired_latch": acquired_latch,
        "manipulation_progress": progress,
        "grasp_acquisition_step": acquisition_step,
        "manipulation_start_step": manipulation_start,
        "manipulation_end_step": manipulation_end,
        "termination_step": termination_step,
    }


def _v14_progress_trace_is_consistent(
    model: mujoco.MjModel,
    config: dict[str, Any],
    states: np.ndarray,
    manipulation_progress: np.ndarray,
    traces: dict[str, np.ndarray],
) -> bool:
    """Recompute schema-v14 plan time, including causal progress freezes.

    Schema v14 deliberately decouples plan time from wall-clock protocol time:
    a risky contact observation freezes the next command's plan increment.  It
    therefore cannot be checked against the legacy smoothstep reconstruction.
    This routine treats the persisted freeze bit only as an input to the plan
    clock; :func:`_v14_contact_preservation_metrics` independently verifies
    that the bit came from the preceding raw contact observation.
    """

    total_steps = states.shape[0]
    frozen = np.asarray(traces["contact_progress_frozen"], dtype=bool)
    if frozen.shape != (total_steps,):
        raise ValueError(
            f"contact_progress_frozen must have shape ({total_steps},)"
        )
    duration_s = float(config["manipulation_plan"]["duration_s"])
    if not np.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("manipulation_plan.duration_s must be positive")

    elapsed_s = 0.0
    expected = np.zeros(total_steps, dtype=np.float64)
    comparable = states != "ABORT"
    for step, state in enumerate(states):
        if state == "MANIPULATE":
            if not frozen[step]:
                elapsed_s = min(
                    duration_s,
                    elapsed_s + float(model.opt.timestep),
                )
            expected[step] = elapsed_s / duration_s
        elif state == "HOLD":
            if frozen[step]:
                return False
            expected[step] = elapsed_s / duration_s
        else:
            if frozen[step]:
                return False
            # SETTLE/CLOSE/VERIFY expose no authorized manipulation progress.
            # Multiple schema-v14 safety-abort paths intentionally preserve
            # different diagnostic progress values; ABORT is a separate hard
            # failure and is excluded from this clock comparison.
            expected[step] = 0.0
    return bool(
        np.isfinite(manipulation_progress).all()
        and np.all(
            (manipulation_progress >= -1e-12)
            & (manipulation_progress <= 1.0 + 1e-12)
        )
        and np.allclose(
            manipulation_progress[comparable],
            expected[comparable],
            rtol=0.0,
            atol=2e-13,
        )
    )


def _v3_stage_metrics(
    model: mujoco.MjModel,
    config: dict[str, Any],
    traces: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, bool]]:
    """Evaluate the grasp latch and the manipulation that it authorizes.

    The controller records the state that produced each sample.  In
    particular, an acquisition on sample ``t`` may only authorize a
    MANIPULATE command on ``t + 1``.  Recomputing these invariants from the
    persisted trace makes the JSON verdict independently auditable.
    """

    protocol = config["control_protocol"]
    acceptance = config["acceptance"]
    total_steps = int(np.asarray(traces["time"]).shape[0])
    timestep = float(model.opt.timestep)
    phase = _protocol_steps(model, config)
    if total_steps != sum(phase.values()):
        raise ValueError("schema-v3 trace length must equal the fixed protocol duration")
    stable_steps = int(round(float(protocol["stable_window_s"]) / timestep))
    if stable_steps <= 0:
        raise ValueError("control_protocol.stable_window_s must contain a step")

    acquired_step = _trace_scalar_int(traces, "grasp_acquisition_step")
    manipulation_start = _trace_scalar_int(traces, "manipulation_start_step")
    manipulation_end = _trace_scalar_int(traces, "manipulation_end_step")
    termination_step = _trace_scalar_int(traces, "termination_step")

    states = np.asarray(traces["control_state"]).astype(str)
    acquired_latch = np.asarray(traces["grasp_acquired"], dtype=bool)
    manipulation_progress = np.asarray(
        traces["manipulation_progress"], dtype=np.float64
    )
    persisted_target_effective = np.asarray(
        traces["target_face_effective"], dtype=bool
    )
    gate = np.asarray(traces["grasp_gate"], dtype=bool)
    gate_axis = np.asarray(traces["grasp_gate_order"])
    if gate_axis.ndim != 1:
        raise ValueError("grasp_gate_order must be a one-dimensional axis")
    gate_order = tuple(str(value) for value in gate_axis)
    consecutive = np.asarray(
        traces["grasp_gate_consecutive_steps"], dtype=np.int64
    )
    if states.shape != (total_steps,):
        raise ValueError(f"control_state must have shape ({total_steps},)")
    if acquired_latch.shape != (total_steps,):
        raise ValueError(f"grasp_acquired must have shape ({total_steps},)")
    if manipulation_progress.shape != (total_steps,):
        raise ValueError(f"manipulation_progress must have shape ({total_steps},)")
    if persisted_target_effective.shape != (total_steps, 3):
        raise ValueError(
            f"target_face_effective must have shape ({total_steps}, 3)"
        )
    if consecutive.shape != (total_steps,):
        raise ValueError(
            f"grasp_gate_consecutive_steps must have shape ({total_steps},)"
        )

    forced_acquisition_step = None
    if int(config["schema_version"]) >= 9:
        forced_acquisition_step = _trace_scalar_int(traces, "grasp_lock_step")
    reconstructed = _reconstruct_v3_controller_trace(
        model,
        config,
        gate,
        gate_order,
        forced_acquisition_step=forced_acquisition_step,
    )

    # Recompute the three target-face predicates from the raw force and tactile
    # arrays.  Persisted derived booleans are evidence caches, never authority.
    topology = config["contact_topology"]
    face_force = np.asarray(traces["distal_face_force_n"], dtype=np.float64)
    tactile = np.asarray(traces["tactile_max"], dtype=np.float64)
    if face_force.shape != (total_steps, 3, len(FACE_ORDER)):
        raise ValueError(
            "distal_face_force_n must have shape "
            f"({total_steps}, 3, {len(FACE_ORDER)})"
        )
    if tactile.shape != (total_steps, 5):
        raise ValueError(f"tactile_max must have shape ({total_steps}, 5)")
    target_indices = np.asarray(
        [
            FACE_ORDER.index(face_from_label(topology["target_faces"][finger]))
            for finger in ACTIVE_FINGERS
        ],
        dtype=int,
    )
    target_force = face_force[:, np.arange(3), target_indices]
    total_distal_force = np.sum(face_force, axis=2)
    purity = np.divide(
        target_force,
        total_distal_force,
        out=np.zeros_like(target_force),
        where=total_distal_force > 0.0,
    )
    gate_config = protocol["grasp_gate"]
    target_effective = (
        target_force >= float(gate_config["min_target_face_force_n"])
    ) & (purity >= float(gate_config["min_target_force_fraction"]))
    if bool(gate_config["require_touch"]):
        target_effective &= tactile[:, :3] >= float(
            acceptance["touch_force_min_n"]
        )
    gate_index = {name: index for index, name in enumerate(gate_order)}
    target_gate_consistent = all(
        np.array_equal(
            gate[:, gate_index[f"{finger}_target_face_effective"]],
            target_effective[:, index],
        )
        for index, finger in enumerate(ACTIVE_FINGERS)
    ) and np.array_equal(
        gate[:, gate_index["target_face_topology"]],
        np.all(target_effective, axis=1),
    )
    persisted_topology = np.asarray(traces["target_face_topology"], dtype=bool)
    persisted_purity = np.asarray(
        traces["target_face_force_purity"], dtype=np.float64
    )
    target_evidence_consistent = bool(
        np.array_equal(persisted_target_effective, target_effective)
        and target_gate_consistent
        and np.array_equal(persisted_topology, np.all(target_effective, axis=1))
        and np.array_equal(persisted_purity, purity)
    )

    # Schema-v5 persists the exclusion gate as an auditable cache, not as
    # authority.  Recompute off-target and active-finger non-distal predicates
    # from the raw force arrays so a stale/edited gate cannot turn proximal-link
    # support into a verified three-distal-contact grasp.  Keep v3/v4 verdicts
    # byte-for-byte compatible by making this an explicit v5 integrity check.
    contact_exclusion_gate_consistent = True
    if int(config["schema_version"]) >= 5:
        active_nondistal = np.asarray(
            traces["active_nondistal_force_n"], dtype=np.float64
        )
        if active_nondistal.shape != (total_steps, 3):
            raise ValueError(
                "active_nondistal_force_n must have shape "
                f"({total_steps}, 3)"
            )
        if (
            not np.isfinite(active_nondistal).all()
            or np.any(active_nondistal < 0.0)
        ):
            raise ValueError(
                "active_nondistal_force_n must be finite and non-negative"
            )
        off_target = np.maximum(0.0, total_distal_force - target_force)
        off_fraction = np.divide(
            off_target,
            total_distal_force,
            out=np.zeros_like(off_target),
            where=total_distal_force > 0.0,
        )
        force_min = float(gate_config["min_target_face_force_n"])
        max_off_fraction = 1.0 - float(
            gate_config["min_target_force_fraction"]
        )
        material_off = (off_target >= force_min) & (
            off_fraction > max_off_fraction + 1e-12
        )
        combined_force = total_distal_force + active_nondistal
        nondistal_fraction = np.divide(
            active_nondistal,
            combined_force,
            out=np.zeros_like(active_nondistal),
            where=combined_force > 0.0,
        )
        material_nondistal = (active_nondistal >= force_min) & (
            nondistal_fraction > max_off_fraction + 1e-12
        )
        contact_exclusion_gate_consistent = bool(
            np.array_equal(
                gate[:, gate_index["no_off_target_contact"]],
                ~np.any(material_off, axis=1),
            )
            and np.array_equal(
                gate[:, gate_index["no_active_nondistal_contact"]],
                ~np.any(material_nondistal, axis=1),
            )
        )

    # Near-miss evidence is measured only while the reconstructed controller is
    # issuing VERIFY commands.  Reconstructing the interval avoids crediting
    # MANIPULATE/HOLD samples after an early acquisition, while deriving the
    # predicates again from force and tactile arrays prevents persisted helper
    # booleans from becoming ranking authority.
    verify_mask = reconstructed["states"] == "VERIFY"
    verify_gate = gate[verify_mask]
    verify_effective = target_effective[verify_mask]
    verify_target_force = target_force[verify_mask]
    verify_tactile = tactile[verify_mask, :3]
    verify_sample_count = int(np.count_nonzero(verify_mask))
    if verify_sample_count:
        verify_gate_component_duty = np.mean(verify_gate, axis=0)
        verify_effective_duty = np.mean(verify_effective, axis=0)
        verify_simultaneous_duty = float(
            np.mean(np.all(verify_effective, axis=1))
        )
        verify_peak_target_force = np.max(verify_target_force, axis=0)
        verify_peak_tactile = np.max(verify_tactile, axis=0)
        verify_all_gate = np.all(verify_gate, axis=1)
        verify_all_gate_duty = float(np.mean(verify_all_gate))
        verify_max_all_gate_steps = _longest_true_run_steps(verify_all_gate)
        verify_effective_finger_count = int(
            np.count_nonzero(np.any(verify_effective, axis=0))
        )
        verify_max_simultaneous_fingers = int(
            np.max(np.count_nonzero(verify_effective, axis=1))
        )
    else:
        verify_gate_component_duty = np.zeros(len(gate_order), dtype=np.float64)
        verify_effective_duty = np.zeros(3, dtype=np.float64)
        verify_simultaneous_duty = 0.0
        verify_peak_target_force = np.zeros(3, dtype=np.float64)
        verify_peak_tactile = np.zeros(3, dtype=np.float64)
        verify_all_gate_duty = 0.0
        verify_max_all_gate_steps = 0
        verify_effective_finger_count = 0
        verify_max_simultaneous_fingers = 0

    state_sequence_consistent = bool(
        np.array_equal(states, reconstructed["states"])
    )
    latch_consistent = bool(
        np.array_equal(acquired_latch, reconstructed["acquired_latch"])
    )
    if int(config["schema_version"]) >= 14:
        # A schema-v14 operation safety abort clears the online VERIFY
        # counter.  That post-lock recovery state cannot invalidate the
        # already authenticated grasp window, so compare the counter only
        # through the acquisition sample.  Operation abort consistency is
        # checked separately and remains a hard manipulation failure.
        counter_end = acquired_step + 1 if acquired_step >= 0 else total_steps
        counter_consistent = bool(
            np.array_equal(
                consecutive[:counter_end],
                reconstructed["consecutive"][:counter_end],
            )
        )
    else:
        counter_consistent = bool(
            np.array_equal(consecutive, reconstructed["consecutive"])
        )
    if int(config["schema_version"]) >= 14:
        progress_consistent = _v14_progress_trace_is_consistent(
            model,
            config,
            states,
            manipulation_progress,
            traces,
        )
    else:
        progress_consistent = bool(
            np.array_equal(
                manipulation_progress, reconstructed["manipulation_progress"]
            )
        )
    acquisition_event_consistent = bool(
        acquired_step == reconstructed["grasp_acquisition_step"]
        and latch_consistent
        and counter_consistent
    )
    if int(config["schema_version"]) >= 14 and acquired_step >= 0:
        expected_start = acquired_step + 1
        full_progress_steps = np.flatnonzero(
            (np.arange(total_steps) >= expected_start)
            & (manipulation_progress >= 1.0 - 1e-12)
        )
        expected_end = (
            int(full_progress_steps[0]) if full_progress_steps.size else -1
        )
        expected_states = reconstructed["states"].copy()
        no_operation_abort = not bool(np.any(states == "ABORT"))
        if expected_end >= expected_start:
            expected_states[expected_start : expected_end + 1] = "MANIPULATE"
            expected_states[expected_end + 1 :] = "HOLD"
        state_sequence_consistent = bool(
            no_operation_abort and np.array_equal(states, expected_states)
        )
        operation_events_consistent = bool(
            manipulation_start == expected_start
            and manipulation_end == expected_end
            and expected_end >= expected_start
            and states[expected_end] == "MANIPULATE"
        )
        termination_event_consistent = bool(
            no_operation_abort and termination_step == total_steps - 1
        )
    else:
        expected_end = reconstructed["manipulation_end_step"]
        operation_events_consistent = bool(
            manipulation_start == reconstructed["manipulation_start_step"]
            and manipulation_end == expected_end
        )
        termination_event_consistent = bool(
            termination_step == reconstructed["termination_step"]
        )

    abort_mask = states == "ABORT"
    grasp_target = np.zeros(model.nu, dtype=np.float64)
    for name, value in contact_preload_targets(config).items():
        grasp_target[model.actuator(name).id] = float(value)
    controls = np.asarray(traces["ctrl"], dtype=np.float64)
    if controls.shape != (total_steps, model.nu):
        raise ValueError(f"ctrl must have shape ({total_steps}, {model.nu})")
    if int(config["schema_version"]) >= 6 and np.any(abort_mask):
        first_abort = int(np.flatnonzero(abort_mask)[0])
        previous_target = (
            controls[first_abort - 1]
            if first_abort > 0
            else np.zeros(model.nu, dtype=np.float64)
        )
        abort_target_ok = bool(
            np.all(controls[abort_mask] == previous_target)
            and np.all(manipulation_progress[abort_mask] == 0.0)
        )
    else:
        abort_target_ok = bool(
            not np.any(abort_mask)
            or (
                np.all(controls[abort_mask] == grasp_target)
                and np.all(manipulation_progress[abort_mask] == 0.0)
            )
        )

    acquired = 0 <= acquired_step < total_steps
    window_start = acquired_step - stable_steps + 1 if acquired else 0
    window_valid = acquired and window_start >= 0
    acquisition_slice = (
        slice(window_start, acquired_step + 1) if window_valid else slice(0, 0)
    )
    contiguous_gate = bool(
        window_valid
        and target_evidence_consistent
        and np.all(gate[acquisition_slice])
        and np.all(target_effective[acquisition_slice])
        and (
            consecutive[acquired_step] >= stable_steps
            if int(config["schema_version"]) >= 9
            else consecutive[acquired_step] == stable_steps
        )
    )

    operation_started = 0 <= manipulation_start < total_steps
    start_after_acquisition = bool(
        operation_started
        and acquired
        and manipulation_start == acquired_step + 1
        and operation_events_consistent
    ) or bool(not acquired and not operation_started and operation_events_consistent)
    no_unauthorized_progress = bool(
        progress_consistent
        and np.all(manipulation_progress[: max(0, manipulation_start)] == 0.0)
        and (acquired or np.all(manipulation_progress == 0.0))
    )
    operation_command_seen = bool(
        operation_started
        and state_sequence_consistent
        and states[manipulation_start] == "MANIPULATE"
    )
    completion_progress_ok = bool(
        manipulation_end >= 0
        and (
            manipulation_progress[manipulation_end] >= 1.0 - 1e-12
            if int(config["schema_version"]) >= 14
            else manipulation_progress[manipulation_end] == 1.0
        )
    )
    if int(config["schema_version"]) >= 14:
        actual_hold_steps = int(np.count_nonzero(states == "HOLD"))
        completion_deadline = total_steps - phase["hold"] - 1
        completion_within_slack = bool(
            manipulation_end >= manipulation_start
            and manipulation_end <= completion_deadline
        )
        minimum_hold_preserved = actual_hold_steps >= phase["hold"]
        manipulation_completed = bool(
            operation_command_seen
            and operation_events_consistent
            and completion_progress_ok
            and completion_within_slack
            and minimum_hold_preserved
        )
    else:
        actual_hold_steps = int(np.count_nonzero(states == "HOLD"))
        completion_within_slack = True
        minimum_hold_preserved = True
        manipulation_completed = bool(
            operation_command_seen
            and operation_events_consistent
            and manipulation_end
            == manipulation_start + phase["manipulate"] - 1
            and completion_progress_ok
        )

    operation_slice = (
        slice(manipulation_start, total_steps)
        if operation_started
        else slice(0, 0)
    )
    if int(config["schema_version"]) >= 16:
        rolling_evidence = _v16_rolling_aware_target_face_evidence(
            config, traces
        )
        operation_effective = rolling_evidence["effective"][
            rolling_evidence["operation_mask"]
        ]
    else:
        operation_effective = target_effective[operation_slice]
    if operation_effective.size:
        operation_duty = np.mean(operation_effective, axis=0)
        operation_simultaneous = float(
            np.mean(np.all(operation_effective, axis=1))
        )
    else:
        operation_duty = np.zeros(3, dtype=np.float64)
        operation_simultaneous = 0.0

    # Lift is measured from the verified grasp, not from the initial support
    # pose, so motion during CLOSE cannot be credited as manipulation.
    baseline_z = float(
        traces["cube_pos"][acquired_step if acquired else 0, 2]
    )
    hold_steps = int(round(float(protocol["min_hold_s"]) / timestep))
    height_count = max(
        1, int(round(float(acceptance["height_window_s"]) / timestep))
    )
    final_count = min(total_steps, hold_steps, height_count)
    final_heights = np.asarray(traces["cube_pos"][-final_count:, 2], dtype=np.float64)
    final_gain = final_heights - baseline_z

    def gate_component_all(name: str) -> bool:
        if not window_valid or name not in gate_index:
            return False
        return bool(np.all(gate[acquisition_slice, gate_index[name]]))

    stable_translation = gate_component_all("cube_translation_stable")
    stable_orientation = gate_component_all("cube_orientation_stable")
    early_lift_ok = gate_component_all("no_early_lift")
    support_retained = gate_component_all("support_contact")

    metrics = {
        "grasp_acquisition_step": acquired_step,
        "manipulation_start_step": manipulation_start,
        "manipulation_end_step": manipulation_end,
        "termination_step": termination_step,
        "grasp_stable_window_steps": stable_steps,
        "grasp_stable_window_s": stable_steps * timestep,
        "grasp_gate_component_order": gate_order,
        "grasp_gate_final_consecutive_steps": (
            int(consecutive[acquired_step]) if acquired else int(np.max(consecutive))
        ),
        "verify_sample_count": verify_sample_count,
        "verify_gate_component_duty": {
            name: float(verify_gate_component_duty[index])
            for index, name in enumerate(gate_order)
        },
        "verify_target_face_effective_duty": {
            finger: float(verify_effective_duty[index])
            for index, finger in enumerate(ACTIVE_FINGERS)
        },
        "verify_peak_target_face_force_n": {
            finger: float(verify_peak_target_force[index])
            for index, finger in enumerate(ACTIVE_FINGERS)
        },
        "verify_peak_tactile_n": {
            finger: float(verify_peak_tactile[index])
            for index, finger in enumerate(ACTIVE_FINGERS)
        },
        "verify_target_face_simultaneous_duty": verify_simultaneous_duty,
        "verify_all_gate_duty": verify_all_gate_duty,
        "verify_max_consecutive_gate_steps": verify_max_all_gate_steps,
        "verify_max_consecutive_all_gate_steps": verify_max_all_gate_steps,
        "verify_effective_finger_count": verify_effective_finger_count,
        "verify_max_simultaneous_effective_finger_count": (
            verify_max_simultaneous_fingers
        ),
        "operation_target_face_contact_duty": {
            finger: float(operation_duty[index])
            for index, finger in enumerate(ACTIVE_FINGERS)
        },
        "operation_target_face_simultaneous_duty": operation_simultaneous,
        "operation_baseline_cube_z_m": baseline_z,
        "operation_median_lift_m": float(np.median(final_gain)),
        "operation_minimum_lift_m": float(np.min(final_gain)),
        "operation_peak_lift_m": float(
            np.max(np.asarray(traces["cube_pos"][:, 2]) - baseline_z)
        ),
        "controller_aborted": bool(np.any(states == "ABORT")),
        "actual_hold_start_step": (
            int(np.flatnonzero(states == "HOLD")[0])
            if np.any(states == "HOLD")
            else -1
        ),
        "actual_hold_steps": int(np.count_nonzero(states == "HOLD")),
        "grasp_support_retained": support_retained,
        "grasp_translation_stable": stable_translation,
        "grasp_orientation_stable": stable_orientation,
        "no_early_lift": early_lift_ok,
    }
    duty_threshold = float(acceptance["finger_contact_duty"])
    simultaneous_threshold = float(acceptance["simultaneous_contact_duty"])
    checks = {
        "stable_grasp_acquired": acquired,
        "grasp_acquisition_event_consistent": acquisition_event_consistent,
        "grasp_gate_contiguous_window": contiguous_gate,
        "target_face_evidence_matches_raw_trace": target_evidence_consistent,
        **(
            {
                "v5_contact_exclusion_gate_matches_raw_trace": (
                    contact_exclusion_gate_consistent
                )
            }
            if int(config["schema_version"]) >= 5
            else {}
        ),
        "grasp_latch_remains_set": latch_consistent,
        "grasp_gate_counter_is_consistent": counter_consistent,
        "controller_state_sequence_is_consistent": state_sequence_consistent,
        "controller_operation_events_are_consistent": operation_events_consistent,
        "controller_termination_event_is_consistent": termination_event_consistent,
        "manipulation_progress_is_consistent": progress_consistent,
        # The public key is retained for schema-v3-v5 compatibility.  In v6
        # its stricter meaning is "abort holds the last issued safe target".
        "abort_holds_grasp_pose": abort_target_ok,
        "grasp_support_retained": support_retained,
        "grasp_pose_is_stable": stable_translation and stable_orientation,
        "no_early_object_lift": early_lift_ok,
        "operation_started_only_after_acquisition": start_after_acquisition,
        "no_operation_target_without_grasp": no_unauthorized_progress,
        "operation_executed": operation_command_seen,
        "manipulation_completed": manipulation_completed,
        "operation_thumb_target_face_contact_duty": operation_duty[0] + 1e-12
        >= duty_threshold,
        "operation_index_target_face_contact_duty": operation_duty[1] + 1e-12
        >= duty_threshold,
        "operation_middle_target_face_contact_duty": operation_duty[2] + 1e-12
        >= duty_threshold,
        "operation_simultaneous_target_face_topology": operation_simultaneous
        + 1e-12
        >= simultaneous_threshold,
        "operation_median_lift_reached": metrics["operation_median_lift_m"]
        + 1e-12
        >= float(acceptance["median_lift_m"]),
        "operation_minimum_lift_reached": metrics["operation_minimum_lift_m"]
        + 1e-12
        >= float(acceptance["minimum_lift_m"]),
        **(
            {
                "v14_manipulation_completed_within_saved_verify_slack": (
                    completion_within_slack
                ),
                "v14_minimum_hold_duration_preserved": minimum_hold_preserved,
                "v14_manipulation_end_is_first_full_progress_sample": bool(
                    manipulation_end == expected_end
                ),
            }
            if int(config["schema_version"]) >= 14
            else {}
        ),
    }
    return metrics, checks


def _v4_alignment_metrics(
    model: mujoco.MjModel,
    config: dict[str, Any],
    traces: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, bool]]:
    """Recompute schema-v4 contact centroids, alignment and pose constraints.

    Persisted centroids and booleans are caches for visualization.  Raw
    per-face forces and force-position moments remain the authority used for
    acceptance so editing an NPZ-derived boolean cannot create a pass.
    """

    total_steps = int(np.asarray(traces["time"]).shape[0])
    face_force = np.asarray(traces["distal_face_force_n"], dtype=np.float64)
    face_moment = np.asarray(
        traces["distal_face_position_moment_n_m"], dtype=np.float64
    )
    tactile = np.asarray(traces["tactile_max"], dtype=np.float64)
    if face_force.shape != (total_steps, 3, len(FACE_ORDER)):
        raise ValueError(
            "distal_face_force_n must have shape "
            f"({total_steps}, 3, {len(FACE_ORDER)})"
        )
    if face_moment.shape != (total_steps, 3, len(FACE_ORDER), 3):
        raise ValueError(
            "distal_face_position_moment_n_m must have shape "
            f"({total_steps}, 3, {len(FACE_ORDER)}, 3)"
        )
    if tactile.shape != (total_steps, 5):
        raise ValueError(f"tactile_max must have shape ({total_steps}, 5)")

    target_faces = tuple(
        face_from_label(config["contact_topology"]["target_faces"][finger])
        for finger in ACTIVE_FINGERS
    )
    centroids = np.zeros((total_steps, 3, 3), dtype=np.float64)
    centroid_force_valid = np.zeros((total_steps, 3), dtype=bool)
    geometric_spread = np.zeros(total_steps, dtype=np.float64)
    geometric_spread_valid = np.zeros(total_steps, dtype=bool)
    gravity = np.asarray(model.opt.gravity, dtype=np.float64)
    for step in range(total_steps):
        centroid, valid = target_face_contact_centroids(
            face_force[step], face_moment[step], target_faces
        )
        spread, spread_valid = three_finger_height_spread(
            centroid, valid, gravity
        )
        centroids[step] = centroid
        centroid_force_valid[step] = valid
        geometric_spread[step] = spread
        geometric_spread_valid[step] = spread_valid

    target_indices = np.asarray(
        [FACE_ORDER.index(face) for face in target_faces], dtype=int
    )
    target_force = face_force[:, np.arange(3), target_indices]
    total_distal_force = np.sum(face_force, axis=2)
    purity = np.divide(
        target_force,
        total_distal_force,
        out=np.zeros_like(target_force),
        where=total_distal_force > 0.0,
    )
    gate_config = config["control_protocol"]["grasp_gate"]
    target_effective = (
        target_force >= float(gate_config["min_target_face_force_n"])
    ) & (purity >= float(gate_config["min_target_force_fraction"]))
    if bool(gate_config["require_touch"]):
        target_effective &= tactile[:, :3] >= float(
            config["acceptance"]["touch_force_min_n"]
        )
    native_all_target_effective = np.all(target_effective, axis=1)

    alignment = config["contact_alignment"]
    maximum_spread = float(alignment["max_height_spread_m"])
    native_aligned = (
        geometric_spread_valid
        & native_all_target_effective
        & (geometric_spread <= maximum_spread + 1e-12)
    )

    # Persisted v4 alignment fields and the grasp gate retain their historical
    # native-tactile meaning.  Schema v16 substitutes strict rolling-aware pad
    # evidence only for MANIPULATE/HOLD acceptance statistics.
    evaluation_target_effective = target_effective
    if int(config.get("schema_version", 4)) >= 16:
        evaluation_target_effective = (
            _v16_rolling_aware_target_face_evidence(config, traces)["effective"]
        )
    all_target_effective = np.all(evaluation_target_effective, axis=1)
    aligned = (
        geometric_spread_valid
        & all_target_effective
        & (geometric_spread <= maximum_spread + 1e-12)
    )

    persisted_centroids = np.asarray(
        traces["target_face_contact_centroid_world_m"], dtype=np.float64
    )
    persisted_centroid_valid = np.asarray(
        traces["target_face_contact_centroid_valid"], dtype=bool
    )
    persisted_spread = np.asarray(
        traces["three_contact_height_spread_m"], dtype=np.float64
    )
    persisted_aligned = np.asarray(
        traces["three_contact_height_aligned"], dtype=bool
    )
    if persisted_centroids.shape != centroids.shape:
        raise ValueError(
            "target_face_contact_centroid_world_m must have shape "
            f"{centroids.shape}"
        )
    if persisted_centroid_valid.shape != centroid_force_valid.shape:
        raise ValueError(
            "target_face_contact_centroid_valid must have shape "
            f"{centroid_force_valid.shape}"
        )
    if persisted_spread.shape != (total_steps,):
        raise ValueError(
            f"three_contact_height_spread_m must have shape ({total_steps},)"
        )
    if persisted_aligned.shape != (total_steps,):
        raise ValueError(
            f"three_contact_height_aligned must have shape ({total_steps},)"
        )

    gate_axis = tuple(str(value) for value in np.asarray(traces["grasp_gate_order"]))
    try:
        alignment_gate = np.asarray(traces["grasp_gate"], dtype=bool)[
            :, gate_axis.index("contact_height_aligned")
        ]
    except ValueError as error:
        raise ValueError(
            "schema-v4 grasp_gate_order is missing contact_height_aligned"
        ) from error
    trace_recomputes = bool(
        np.allclose(persisted_centroids, centroids, rtol=0.0, atol=1e-15)
        and np.array_equal(persisted_centroid_valid, centroid_force_valid)
        and np.allclose(persisted_spread, geometric_spread, rtol=0.0, atol=1e-15)
        and np.array_equal(persisted_aligned, native_aligned)
        and np.array_equal(alignment_gate, native_aligned)
    )

    states = np.asarray(traces["control_state"]).astype(str)
    if states.shape != (total_steps,):
        raise ValueError(f"control_state must have shape ({total_steps},)")

    def stage_statistics(mask: np.ndarray) -> dict[str, Any]:
        count = int(np.count_nonzero(mask))
        if count == 0:
            return {
                "sample_count": 0,
                "effective_duty": 0.0,
                "aligned_duty": 0.0,
                "height_spread_p50_m": None,
                "height_spread_p95_m": None,
                "height_spread_max_m": None,
            }
        stage_effective = all_target_effective[mask]
        stage_aligned = aligned[mask]
        values = geometric_spread[mask][stage_effective]
        return {
            "sample_count": count,
            "effective_duty": float(np.mean(stage_effective)),
            "aligned_duty": float(np.mean(stage_aligned)),
            "height_spread_p50_m": (
                float(np.percentile(values, 50.0)) if values.size else None
            ),
            "height_spread_p95_m": (
                float(np.percentile(values, 95.0)) if values.size else None
            ),
            "height_spread_max_m": float(np.max(values)) if values.size else None,
        }

    verify_mask = states == "VERIFY"
    manipulate_mask = states == "MANIPULATE"
    hold_mask = states == "HOLD"
    operation_mask = manipulate_mask | hold_mask
    stage_alignment = {
        "verify": stage_statistics(verify_mask),
        "manipulate": stage_statistics(manipulate_mask),
        "hold": stage_statistics(hold_mask),
        "operation": stage_statistics(operation_mask),
    }

    acquisition_step = _trace_scalar_int(traces, "grasp_acquisition_step")
    verify_seconds = float(alignment["verify_continuous_s"])
    verify_steps = int(round(verify_seconds / float(model.opt.timestep)))
    acquisition_start = acquisition_step - verify_steps + 1
    acquisition_aligned = bool(
        acquisition_step >= 0
        and acquisition_start >= 0
        and np.all(native_aligned[acquisition_start : acquisition_step + 1])
    )

    finger_tilt = np.asarray(traces["finger_down_tilt_deg"], dtype=np.float64)
    palm_angle = np.asarray(traces["palm_down_angle_deg"], dtype=np.float64)
    root_position = np.asarray(traces["root_pos"], dtype=np.float64)
    if finger_tilt.shape != (total_steps,):
        raise ValueError(f"finger_down_tilt_deg must have shape ({total_steps},)")
    if palm_angle.shape != (total_steps,):
        raise ValueError(f"palm_down_angle_deg must have shape ({total_steps},)")
    if root_position.shape != (total_steps, 3):
        raise ValueError(f"root_pos must have shape ({total_steps}, 3)")
    schema_version = int(config.get("schema_version", 4))
    pose = config["pose_constraints"]
    tilt_limits = tuple(float(value) for value in pose["finger_down_tilt_deg"])
    palm_limits = tuple(
        float(value) for value in pose["palm_plane_ground_angle_deg"]
    )

    def all_in_range(values: np.ndarray, limits: tuple[float, float]) -> bool:
        return bool(
            np.isfinite(values).all()
            and np.all(values >= limits[0] - 1e-12)
            and np.all(values <= limits[1] + 1e-12)
        )

    metrics: dict[str, Any] = {
        "contact_alignment": stage_alignment,
        "contact_alignment_max_height_spread_m": maximum_spread,
        "contact_alignment_verify_continuous_s": verify_seconds,
        "contact_alignment_operation_required_duty": float(
            alignment["operation_aligned_duty"]
        ),
        "finger_down_tilt_deg": {
            "min": float(np.min(finger_tilt)),
            "max": float(np.max(finger_tilt)),
        },
        "palm_plane_ground_angle_deg": {
            "min": float(np.min(palm_angle)),
            "max": float(np.max(palm_angle)),
        },
    }
    checks = {
        "v4_alignment_trace_matches_raw_contacts": trace_recomputes,
        "grasp_contact_height_alignment_contiguous": acquisition_aligned,
        "operation_contact_height_aligned_duty": (
            stage_alignment["operation"]["aligned_duty"] + 1e-12
            >= float(alignment["operation_aligned_duty"])
        ),
        "finger_down_tilt_within_range": all_in_range(
            finger_tilt, tilt_limits
        ),
        "palm_plane_ground_angle_within_range": all_in_range(
            palm_angle, palm_limits
        ),
    }
    if schema_version >= 5:
        cube_position = np.asarray(traces["cube_pos"], dtype=np.float64)
        if cube_position.shape != (total_steps, 3):
            raise ValueError(f"cube_pos must have shape ({total_steps}, 3)")
        root_cube_distance = np.linalg.norm(cube_position - root_position, axis=1)
        reference_root = np.asarray(
            pose["legacy_press_reference_translation_m"], dtype=np.float64
        ).reshape(3)
        legacy_press_depth = reference_root[2] - root_position[:, 2]
        resolved_pose = resolved_pose_constraint_values(config)
        initial_distance = float(resolved_pose["root_cube_distance_m"])
        distance_limits = tuple(
            float(value) for value in pose["root_cube_distance_m"]
        )
        effective_distance_limits = list(distance_limits)
        boundary = config.get("candidate_metadata", {}).get(
            "boundary_expansion", {}
        )
        distance_expanded = bool(
            isinstance(boundary, dict)
            and boundary.get("applied") is True
            and boundary.get("count") == 1
            and boundary.get("distance_expanded") is True
        )
        if distance_expanded:
            effective_distance_limits[1] = float(
                config["far_hand_campaign"]["boundary_expansion"]
                ["expanded_root_cube_distance_max_m"]
            )
        metrics.update(
            {
                "root_cube_center_distance_m": {
                    "initial": initial_distance,
                    "min": float(np.min(root_cube_distance)),
                    "max": float(np.max(root_cube_distance)),
                    "acceptance_range": effective_distance_limits,
                    "upper_boundary_expanded": distance_expanded,
                },
                "legacy_palm_press_depth_m": {
                    "initial": float(
                        resolved_pose["legacy_palm_press_depth_m"]
                    ),
                    "min": float(np.min(legacy_press_depth)),
                    "max": float(np.max(legacy_press_depth)),
                    "acceptance_role": "diagnostic_only",
                },
            }
        )
        checks["initial_root_cube_distance_within_range"] = bool(
            effective_distance_limits[0] - 1e-12
            <= initial_distance
            <= effective_distance_limits[1] + 1e-12
        )
    else:
        reference_root = np.asarray(
            pose["reference_hand_translation_m"], dtype=np.float64
        ).reshape(3)
        press_depth = reference_root[2] - root_position[:, 2]
        press_limits = tuple(float(value) for value in pose["palm_press_depth_m"])
        metrics["palm_press_depth_m"] = {
            "min": float(np.min(press_depth)),
            "max": float(np.max(press_depth)),
        }
        checks["palm_press_depth_within_range"] = all_in_range(
            press_depth, press_limits
        )
    return metrics, checks


def _v5_fingertip_contact_metrics(
    model: mujoco.MjModel,
    info: ModelInfo,
    config: dict[str, Any],
    traces: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Aggregate schema-v5 pad preference and trace-integrity evidence.

    Pad quality remains a soft ranking signal.  Equality between persisted
    helper arrays and their raw force/state/control sources is a hard artifact
    integrity requirement applied by :func:`evaluate_trace`.
    """

    total_steps = int(np.asarray(traces["time"]).shape[0])
    expected = (total_steps, len(ACTIVE_FINGERS))
    pad = np.asarray(traces["distal_pad_force_n"], dtype=np.float64)
    nonpad = np.asarray(traces["distal_nonpad_force_n"], dtype=np.float64)
    persisted_fraction = np.asarray(
        traces["distal_pad_force_fraction"], dtype=np.float64
    )
    active_taxels = np.asarray(traces["distal_active_taxel_count"], dtype=np.int64)
    for name, values in (
        ("distal_pad_force_n", pad),
        ("distal_nonpad_force_n", nonpad),
        ("distal_pad_force_fraction", persisted_fraction),
        ("distal_active_taxel_count", active_taxels),
    ):
        if values.shape != expected:
            raise ValueError(f"{name} must have shape {expected}")
    if (
        not np.isfinite(pad).all()
        or not np.isfinite(nonpad).all()
        or not np.isfinite(persisted_fraction).all()
        or np.any(pad < 0.0)
        or np.any(nonpad < 0.0)
        or np.any(active_taxels < 0)
    ):
        raise ValueError("schema-v5 fingertip traces must be finite and non-negative")

    total_distal = pad + nonpad
    recomputed_fraction = np.divide(
        pad,
        total_distal,
        out=np.zeros_like(pad),
        where=total_distal > 0.0,
    )
    trace_fraction_consistent = bool(
        np.array_equal(persisted_fraction, recomputed_fraction)
    )

    root_distance = np.asarray(
        traces["root_cube_center_distance_m"], dtype=np.float64
    )
    thumb_command = np.asarray(traces["thumb_bend_command_rad"], dtype=np.float64)
    thumb_qpos = np.asarray(traces["thumb_bend_qpos_rad"], dtype=np.float64)
    for name, values in (
        ("root_cube_center_distance_m", root_distance),
        ("thumb_bend_command_rad", thumb_command),
        ("thumb_bend_qpos_rad", thumb_qpos),
    ):
        if values.shape != (total_steps,) or not np.isfinite(values).all():
            raise ValueError(f"{name} must have shape ({total_steps},) and be finite")
    recomputed_root_distance = np.linalg.norm(
        np.asarray(traces["cube_pos"], dtype=np.float64)
        - np.asarray(traces["root_pos"], dtype=np.float64),
        axis=1,
    )
    thumb_actuator_id = model.actuator(
        "left_hand_thumb_bend_joint_actuator"
    ).id
    trace_state_consistent = bool(
        np.array_equal(root_distance, recomputed_root_distance)
        and np.array_equal(
            thumb_command,
            np.asarray(traces["ctrl"], dtype=np.float64)[:, thumb_actuator_id],
        )
        and np.array_equal(
            thumb_qpos,
            np.asarray(traces["joint_qpos"], dtype=np.float64)[
                :, thumb_actuator_id
            ],
        )
    )

    states = np.asarray(traces["control_state"]).astype(str)
    if states.shape != (total_steps,):
        raise ValueError(f"control_state must have shape ({total_steps},)")

    thumb_protocol_consistent: bool | None = None
    if "control" in config and "control_protocol" in config:
        grasp_thumb = float(
            contact_preload_targets(config)[
                "left_hand_thumb_bend_joint_actuator"
            ]
        )
        thumb_delta = float(
            config["control"]["manipulation_delta_rad"]
            ["left_hand_thumb_bend_joint_actuator"]
        )
        manipulated_thumb = grasp_thumb + thumb_delta
        protocol_steps = _protocol_steps(model, config)
        settle_end = protocol_steps["settle"]
        expected_thumb = np.empty(total_steps, dtype=np.float64)
        schema_version = int(config.get("schema_version", 5))
        pregrasp_thumb = 0.0
        close_start = 0.0
        close_end = 1.0
        if schema_version >= 6:
            pregrasp_thumb = float(
                precontact_targets(config)[
                    "left_hand_thumb_bend_joint_actuator"
                ]
            )
            profile = config["control"]["close_profile"][
                "left_hand_thumb_bend_joint_actuator"
            ]
            close_start = float(profile["start_fraction"])
            close_end = float(profile["end_fraction"])
        for step, state in enumerate(states):
            if state == "SETTLE":
                if schema_version >= 6:
                    if bool(
                        config["pose_preservation"][
                            "initialize_active_joints_at_pregrasp"
                        ]
                    ):
                        expected_thumb[step] = pregrasp_thumb
                    else:
                        raw_alpha = (step + 1) / protocol_steps["settle"]
                        alpha = raw_alpha * raw_alpha * (
                            3.0 - 2.0 * raw_alpha
                        )
                        expected_thumb[step] = alpha * pregrasp_thumb
                else:
                    expected_thumb[step] = 0.0
            elif state == "CLOSE":
                raw_fraction = (
                    step - settle_end + 1
                ) / protocol_steps["close"]
                if schema_version >= 6:
                    raw_alpha = (raw_fraction - close_start) / (
                        close_end - close_start
                    )
                    raw_alpha = min(1.0, max(0.0, raw_alpha))
                    alpha = raw_alpha * raw_alpha * (3.0 - 2.0 * raw_alpha)
                    expected_thumb[step] = pregrasp_thumb + alpha * (
                        grasp_thumb - pregrasp_thumb
                    )
                else:
                    alpha = raw_fraction * raw_fraction * (
                        3.0 - 2.0 * raw_fraction
                    )
                    expected_thumb[step] = alpha * grasp_thumb
            elif state == "VERIFY":
                expected_thumb[step] = grasp_thumb
            elif state == "ABORT":
                if schema_version >= 6:
                    expected_thumb[step] = (
                        expected_thumb[step - 1] if step > 0 else 0.0
                    )
                else:
                    expected_thumb[step] = grasp_thumb
            elif state == "MANIPULATE":
                progress = float(
                    np.asarray(
                        traces["manipulation_progress"], dtype=np.float64
                    )[step]
                )
                # Mirror the controller's arithmetic, including the rounded
                # absolute manipulation target, for exact trace comparison.
                expected_thumb[step] = grasp_thumb + progress * (
                    manipulated_thumb - grasp_thumb
                )
            elif state == "HOLD":
                expected_thumb[step] = manipulated_thumb
            else:
                expected_thumb[step] = np.nan
        thumb_protocol_consistent = bool(
            np.array_equal(thumb_command, expected_thumb)
        )

    def aggregate(mask: np.ndarray) -> dict[str, Any]:
        sample_count = int(np.count_nonzero(mask))
        if sample_count:
            stage_pad = pad[mask]
            stage_nonpad = nonpad[mask]
            stage_total = stage_pad + stage_nonpad
            summed_pad = np.sum(stage_pad, axis=0)
            summed_total = np.sum(stage_total, axis=0)
            force_weighted = np.divide(
                summed_pad,
                summed_total,
                out=np.zeros_like(summed_pad),
                where=summed_total > 0.0,
            )
            contact_mask = stage_total > 0.0
            mean_on_contact = np.divide(
                np.sum(recomputed_fraction[mask] * contact_mask, axis=0),
                np.count_nonzero(contact_mask, axis=0),
                out=np.zeros(len(ACTIVE_FINGERS), dtype=np.float64),
                where=np.count_nonzero(contact_mask, axis=0) > 0,
            )
            peak_pad = np.max(stage_pad, axis=0)
            peak_nonpad = np.max(stage_nonpad, axis=0)
            maximum_taxels = np.max(active_taxels[mask], axis=0)
            pad_duty = np.mean(stage_pad > 0.0, axis=0)
        else:
            force_weighted = np.zeros(len(ACTIVE_FINGERS), dtype=np.float64)
            mean_on_contact = np.zeros(len(ACTIVE_FINGERS), dtype=np.float64)
            peak_pad = np.zeros(len(ACTIVE_FINGERS), dtype=np.float64)
            peak_nonpad = np.zeros(len(ACTIVE_FINGERS), dtype=np.float64)
            maximum_taxels = np.zeros(len(ACTIVE_FINGERS), dtype=np.int64)
            pad_duty = np.zeros(len(ACTIVE_FINGERS), dtype=np.float64)

        def per_finger(values: np.ndarray, cast: type = float) -> dict[str, Any]:
            return {
                finger: cast(values[index])
                for index, finger in enumerate(ACTIVE_FINGERS)
            }

        return {
            "sample_count": sample_count,
            "force_weighted_pad_fraction": per_finger(force_weighted),
            "mean_pad_fraction_on_contact": per_finger(mean_on_contact),
            "peak_pad_force_n": per_finger(peak_pad),
            "peak_nonpad_force_n": per_finger(peak_nonpad),
            "max_active_taxel_count": per_finger(maximum_taxels, int),
            "pad_contact_duty": per_finger(pad_duty),
        }

    fingertip_config = config.get("fingertip_contact_preferences", {})
    return {
        "max_taxel_assignment_distance_m": float(
            fingertip_config.get("taxel_assignment_max_distance_m", 0.006)
        ),
        "trace_fraction_matches_raw_forces": trace_fraction_consistent,
        "trace_pose_and_thumb_bend_match_raw_state": trace_state_consistent,
        "thumb_bend_command_matches_control_protocol": (
            thumb_protocol_consistent
        ),
        "initial_root_cube_center_distance_m": float(root_distance[0]),
        "root_cube_center_distance_range_m": [
            float(np.min(root_distance)),
            float(np.max(root_distance)),
        ],
        "thumb_bend_command_range_rad": [
            float(np.min(thumb_command)),
            float(np.max(thumb_command)),
        ],
        "thumb_bend_qpos_range_rad": [
            float(np.min(thumb_qpos)),
            float(np.max(thumb_qpos)),
        ],
        "all": aggregate(np.ones(total_steps, dtype=bool)),
        "verify": aggregate(states == "VERIFY"),
        "operation": aggregate(np.isin(states, ("MANIPULATE", "HOLD"))),
    }


def _v6_pose_preservation_metrics(
    model: mujoco.MjModel,
    info: ModelInfo,
    config: dict[str, Any],
    traces: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, bool]]:
    """Recompute the schema-v6 pre-grasp pose contract from raw state.

    The immutable baseline is deliberately not ``cube_pos[0]``: that sample
    follows one integration step.  Every latch is recomputed from the raw
    position, quaternion and contact arrays, so edited helper booleans cannot
    turn a trajectory that first pushes the cube into a valid grasp.
    """

    total_steps = int(np.asarray(traces["time"]).shape[0])
    initial_position = np.asarray(
        traces["initial_cube_pos_m"], dtype=np.float64
    )
    initial_quaternion = np.asarray(
        traces["initial_cube_quat"], dtype=np.float64
    )
    cube_position = np.asarray(traces["cube_pos"], dtype=np.float64)
    cube_quaternion = np.asarray(traces["cube_quat"], dtype=np.float64)
    initial_joint_qpos = np.asarray(
        traces["initial_joint_qpos_rad"], dtype=np.float64
    )
    initialized_at_pregrasp = bool(
        np.asarray(traces["initialized_at_pregrasp"]).reshape(())
    )
    if initial_position.shape != (3,) or not np.isfinite(initial_position).all():
        raise ValueError("initial_cube_pos_m must have shape (3,) and be finite")
    if initial_quaternion.shape != (4,) or not np.isfinite(
        initial_quaternion
    ).all():
        raise ValueError("initial_cube_quat must have shape (4,) and be finite")
    if cube_position.shape != (total_steps, 3):
        raise ValueError(f"cube_pos must have shape ({total_steps}, 3)")
    if cube_quaternion.shape != (total_steps, 4):
        raise ValueError(f"cube_quat must have shape ({total_steps}, 4)")
    if initial_joint_qpos.shape != (model.nu,) or not np.isfinite(
        initial_joint_qpos
    ).all():
        raise ValueError(
            f"initial_joint_qpos_rad must have shape ({model.nu},) and be finite"
        )
    settings = config["pose_preservation"]
    expected_initial_qpos = np.zeros(model.nu, dtype=np.float64)
    if bool(settings["initialize_active_joints_at_pregrasp"]):
        for name, value in precontact_targets(config).items():
            expected_initial_qpos[model.actuator(name).id] = float(value)
    initial_joint_state_matches = bool(
        initialized_at_pregrasp
        == bool(settings["initialize_active_joints_at_pregrasp"])
        and np.array_equal(initial_joint_qpos, expected_initial_qpos)
    )

    translation = np.linalg.norm(cube_position - initial_position, axis=1)
    orientation = np.degrees(
        orientation_angles(initial_quaternion, cube_quaternion)
    )
    translation_limit = float(settings["max_translation_m"])
    orientation_limit = float(settings["max_orientation_drift_deg"])
    translation_ok = translation <= translation_limit + 1e-12
    orientation_ok = orientation <= orientation_limit + 1e-12
    current_pose_ok = translation_ok & orientation_ok

    acquisition_step = _trace_scalar_int(traces, "grasp_acquisition_step")
    scope_last = acquisition_step if acquisition_step >= 0 else total_steps - 1
    scope_last = min(total_steps - 1, scope_last)
    active_scope = np.arange(total_steps) <= scope_last

    translation_latch = np.ones(total_steps, dtype=bool)
    orientation_latch = np.ones(total_steps, dtype=bool)
    translation_latch[: scope_last + 1] = np.logical_and.accumulate(
        translation_ok[: scope_last + 1]
    )
    orientation_latch[: scope_last + 1] = np.logical_and.accumulate(
        orientation_ok[: scope_last + 1]
    )
    if scope_last + 1 < total_steps:
        translation_latch[scope_last + 1 :] = translation_latch[scope_last]
        orientation_latch[scope_last + 1 :] = orientation_latch[scope_last]
    pose_latch = translation_latch & orientation_latch

    support = np.asarray(traces["support_contact"], dtype=bool)
    states = np.asarray(traces["control_state"]).astype(str)
    forbidden = np.asarray(traces["forbidden_contact"], dtype=bool)
    distal_force = np.asarray(traces["finger_contact_force"], dtype=np.float64)
    nondistal_force = np.asarray(
        traces["active_nondistal_force_n"], dtype=np.float64
    )
    if support.shape != (total_steps,):
        raise ValueError(f"support_contact must have shape ({total_steps},)")
    if states.shape != (total_steps,):
        raise ValueError(f"control_state must have shape ({total_steps},)")
    if forbidden.shape != (total_steps,):
        raise ValueError(f"forbidden_contact must have shape ({total_steps},)")
    if distal_force.shape != (total_steps, len(ACTIVE_FINGERS)):
        raise ValueError(
            "finger_contact_force must have shape "
            f"({total_steps}, {len(ACTIVE_FINGERS)})"
        )
    if nondistal_force.shape != (total_steps, len(ACTIVE_FINGERS)):
        raise ValueError(
            "active_nondistal_force_n must have shape "
            f"({total_steps}, {len(ACTIVE_FINGERS)})"
        )

    support_latch = np.ones(total_steps, dtype=bool)
    if bool(settings["require_support_contact"]):
        support_latch[: scope_last + 1] = np.logical_and.accumulate(
            support[: scope_last + 1]
        )
        if scope_last + 1 < total_steps:
            support_latch[scope_last + 1 :] = support_latch[scope_last]

    hand_contact = np.asarray(traces["hand_cube_contact"], dtype=bool)
    if hand_contact.shape != (total_steps,):
        raise ValueError(f"hand_cube_contact must have shape ({total_steps},)")
    # A force-bearing contact must also be represented by the stricter
    # geometric-touching cache.  The converse need not hold: a zero-force
    # touching collision is still forbidden during SETTLE in schema v6.
    force_implies_touch = bool(
        np.all(
            ~(
                forbidden
                | (np.sum(distal_force, axis=1) > 0.0)
                | (np.sum(nondistal_force, axis=1) > 0.0)
            )
            | hand_contact
        )
    )
    settle_contact_free_latch = np.ones(total_steps, dtype=bool)
    if bool(settings["require_no_hand_cube_contact_during_settle"]):
        running = True
        for step in range(total_steps):
            if active_scope[step] and states[step] == "SETTLE":
                running = bool(running and not hand_contact[step])
            settle_contact_free_latch[step] = running

    combined_latch = pose_latch & support_latch & settle_contact_free_latch
    persisted_translation = np.asarray(
        traces["cube_translation_from_initial_m"], dtype=np.float64
    )
    persisted_orientation = np.asarray(
        traces["cube_orientation_from_initial_deg"], dtype=np.float64
    )
    persisted_translation_latch = np.asarray(
        traces["initial_pose_translation_history_stable"], dtype=bool
    )
    persisted_orientation_latch = np.asarray(
        traces["initial_pose_orientation_history_stable"], dtype=bool
    )
    persisted_current_ok = np.asarray(
        traces["pregrasp_pose_within_limit"], dtype=bool
    )
    persisted_combined = np.asarray(
        traces["pregrasp_pose_preserved_latched"], dtype=bool
    )
    persisted_support = np.asarray(
        traces["pregrasp_support_retained_latched"], dtype=bool
    )
    persisted_settle = np.asarray(
        traces["settle_hand_contact_free_latched"], dtype=bool
    )
    for name, values in (
        ("cube_translation_from_initial_m", persisted_translation),
        ("cube_orientation_from_initial_deg", persisted_orientation),
        ("initial_pose_translation_history_stable", persisted_translation_latch),
        ("initial_pose_orientation_history_stable", persisted_orientation_latch),
        ("pregrasp_pose_within_limit", persisted_current_ok),
        ("pregrasp_pose_preserved_latched", persisted_combined),
        ("pregrasp_support_retained_latched", persisted_support),
        ("settle_hand_contact_free_latched", persisted_settle),
    ):
        if values.shape != (total_steps,):
            raise ValueError(f"{name} must have shape ({total_steps},)")

    gate_axis = tuple(
        str(value) for value in np.asarray(traces["grasp_gate_order"])
    )
    try:
        history_gate = np.asarray(traces["grasp_gate"], dtype=bool)[
            :, gate_axis.index("initial_pose_history_stable")
        ]
    except ValueError as error:
        raise ValueError(
            "schema-v6 grasp_gate_order is missing "
            "initial_pose_history_stable"
        ) from error

    history_trace_matches = bool(
        np.allclose(persisted_translation, translation, rtol=0.0, atol=1e-15)
        and np.allclose(
            persisted_orientation, orientation, rtol=0.0, atol=1e-12
        )
        and np.array_equal(persisted_translation_latch, translation_latch)
        and np.array_equal(persisted_orientation_latch, orientation_latch)
        and np.array_equal(persisted_current_ok, current_pose_ok)
        and np.array_equal(persisted_combined, combined_latch)
        and np.array_equal(persisted_support, support_latch)
        and np.array_equal(persisted_settle, settle_contact_free_latch)
        and np.array_equal(history_gate, combined_latch)
        and force_implies_touch
    )

    close_progress = np.asarray(traces["close_progress"], dtype=np.float64)
    if close_progress.shape != (total_steps, model.nu):
        raise ValueError(
            f"close_progress must have shape ({total_steps}, {model.nu})"
        )
    expected_progress = np.zeros_like(close_progress)
    protocol_steps = _protocol_steps(model, config)
    settle_end = protocol_steps["settle"]
    for step, state in enumerate(states):
        if state == "CLOSE":
            fraction = (step - settle_end + 1) / protocol_steps["close"]
            for name, profile in config["control"]["close_profile"].items():
                actuator_id = model.actuator(name).id
                raw = (fraction - float(profile["start_fraction"])) / (
                    float(profile["end_fraction"])
                    - float(profile["start_fraction"])
                )
                raw = min(1.0, max(0.0, raw))
                expected_progress[step, actuator_id] = raw * raw * (
                    3.0 - 2.0 * raw
                )
        elif state == "ABORT":
            if step > 0:
                expected_progress[step] = expected_progress[step - 1]
        elif state not in ("SETTLE",):
            expected_progress[step, info.active_actuator_ids] = 1.0

    pregrasp_target = np.asarray(
        traces["pregrasp_target_rad"], dtype=np.float64
    )
    close_start = np.asarray(
        traces["close_start_fraction"], dtype=np.float64
    )
    close_end = np.asarray(traces["close_end_fraction"], dtype=np.float64)
    expected_pregrasp = np.zeros(model.nu, dtype=np.float64)
    expected_start = np.zeros(model.nu, dtype=np.float64)
    expected_end = np.ones(model.nu, dtype=np.float64)
    for name, value in precontact_targets(config).items():
        actuator_id = model.actuator(name).id
        expected_pregrasp[actuator_id] = float(value)
        expected_start[actuator_id] = float(
            config["control"]["close_profile"][name]["start_fraction"]
        )
        expected_end[actuator_id] = float(
            config["control"]["close_profile"][name]["end_fraction"]
        )
    close_trace_matches = bool(
        np.allclose(close_progress, expected_progress, rtol=0.0, atol=1e-15)
        and np.array_equal(pregrasp_target, expected_pregrasp)
        and np.array_equal(close_start, expected_start)
        and np.array_equal(close_end, expected_end)
    )

    first_distal = np.asarray(
        traces["first_distal_contact_step"], dtype=np.int64
    )
    if first_distal.shape != (len(ACTIVE_FINGERS),):
        raise ValueError(
            "first_distal_contact_step must have shape "
            f"({len(ACTIVE_FINGERS)},)"
        )
    expected_first = np.full(len(ACTIVE_FINGERS), -1, dtype=np.int64)
    for finger_index in range(len(ACTIVE_FINGERS)):
        steps = np.flatnonzero(distal_force[:, finger_index] > 0.0)
        if steps.size:
            expected_first[finger_index] = int(steps[0])
    valid_first = expected_first >= 0
    onset_span_steps = (
        int(np.ptp(expected_first[valid_first]))
        if np.all(valid_first)
        else None
    )

    acquisition_exists = 0 <= acquisition_step < total_steps
    max_translation = float(np.max(translation[: scope_last + 1]))
    max_orientation = float(np.max(orientation[: scope_last + 1]))
    final_delta = (
        cube_position[scope_last] - initial_position
        if scope_last >= 0
        else np.zeros(3, dtype=np.float64)
    )
    metrics = {
        "pose_preservation": {
            "reference": str(settings["reference"]),
            "scope": str(settings["scope"]),
            "scope_last_step": int(scope_last),
            "max_translation_m": max_translation,
            "max_orientation_drift_deg": max_orientation,
            "translation_limit_m": translation_limit,
            "orientation_limit_deg": orientation_limit,
            "translation_at_scope_end_m": float(translation[scope_last]),
            "orientation_at_scope_end_deg": float(orientation[scope_last]),
            "position_delta_at_scope_end_m": [
                float(value) for value in final_delta
            ],
            "support_retained": bool(support_latch[scope_last]),
            "settle_hand_contact_free": bool(
                settle_contact_free_latch[scope_last]
            ),
            "first_distal_contact_step": {
                finger: int(expected_first[index])
                for index, finger in enumerate(ACTIVE_FINGERS)
            },
            "distal_contact_onset_span_steps": onset_span_steps,
            "distal_contact_onset_span_s": (
                None
                if onset_span_steps is None
                else float(onset_span_steps * model.opt.timestep)
            ),
            "initialized_at_pregrasp": initialized_at_pregrasp,
        }
    }
    checks = {
        "v6_pose_preservation_trace_matches_raw_state": history_trace_matches,
        "v6_close_profile_trace_matches_config": close_trace_matches,
        "v6_first_distal_contact_steps_match_raw_trace": bool(
            np.array_equal(first_distal, expected_first)
        ),
        "v6_initial_joint_state_matches_pregrasp_config": (
            initial_joint_state_matches
        ),
        "object_pose_preserved_until_grasp_acquisition": bool(
            acquisition_exists and pose_latch[scope_last]
        ),
        "support_retained_until_grasp_acquisition": bool(
            acquisition_exists and support_latch[scope_last]
        ),
        "no_hand_cube_contact_during_settle": bool(
            acquisition_exists and settle_contact_free_latch[scope_last]
        ),
    }
    return metrics, checks


def _v8_closure_alignment_metrics(
    config: dict[str, Any], traces: dict[str, np.ndarray]
) -> tuple[dict[str, Any], dict[str, bool]]:
    """Recompute command/normal closure alignment from persisted vectors."""

    from .closure_alignment import closure_alignment_from_velocity

    states = np.asarray(traces["control_state"]).astype(str)
    total_steps = states.shape[0]
    velocity = np.asarray(
        traces["closure_command_velocity_world_m_s"], dtype=np.float64
    )
    normal = np.asarray(
        traces["closure_cube_outward_normal_world"], dtype=np.float64
    )
    force = np.asarray(
        traces["closure_target_contact_force_n"], dtype=np.float64
    )
    persisted_cosine = np.asarray(
        traces["closure_alignment_cosine"], dtype=np.float64
    )
    persisted_angle = np.asarray(
        traces["closure_alignment_angle_deg"], dtype=np.float64
    )
    persisted_inward = np.asarray(
        traces["closure_inward_speed_m_s"], dtype=np.float64
    )
    persisted_tangent = np.asarray(
        traces["closure_tangent_speed_m_s"], dtype=np.float64
    )
    persisted_valid = np.asarray(
        traces["closure_alignment_valid"], dtype=bool
    )
    vector_shape = (total_steps, len(ACTIVE_FINGERS), 3)
    scalar_shape = (total_steps, len(ACTIVE_FINGERS))
    if velocity.shape != vector_shape or normal.shape != vector_shape:
        raise ValueError("schema-v8 closure vectors have an invalid shape")
    for values, label in (
        (force, "closure_target_contact_force_n"),
        (persisted_cosine, "closure_alignment_cosine"),
        (persisted_angle, "closure_alignment_angle_deg"),
        (persisted_inward, "closure_inward_speed_m_s"),
        (persisted_tangent, "closure_tangent_speed_m_s"),
        (persisted_valid, "closure_alignment_valid"),
    ):
        if values.shape != scalar_shape:
            raise ValueError(f"{label} has an invalid shape")

    minimum_force = float(config["closure_alignment"]["min_contact_force_n"])
    cosine = np.full(scalar_shape, -1.0, dtype=np.float64)
    angle = np.full(scalar_shape, 180.0, dtype=np.float64)
    inward = np.zeros(scalar_shape, dtype=np.float64)
    tangent = np.zeros(scalar_shape, dtype=np.float64)
    valid = np.zeros(scalar_shape, dtype=bool)
    close_mask = states == "CLOSE"
    for step in np.flatnonzero(close_mask):
        for finger_index in range(len(ACTIVE_FINGERS)):
            sample = closure_alignment_from_velocity(
                velocity[step, finger_index],
                normal[step, finger_index],
                normal_force_n=force[step, finger_index],
                minimum_normal_force_n=minimum_force,
            )
            cosine[step, finger_index] = sample.cosine
            angle[step, finger_index] = sample.angle_deg
            inward[step, finger_index] = sample.inward_speed_m_s
            tangent[step, finger_index] = sample.tangent_speed_m_s
            valid[step, finger_index] = sample.valid
    trace_matches = bool(
        np.array_equal(valid, persisted_valid)
        and np.allclose(cosine, persisted_cosine, rtol=0.0, atol=1e-12)
        and np.allclose(angle, persisted_angle, rtol=0.0, atol=1e-10)
        and np.allclose(inward, persisted_inward, rtol=0.0, atol=1e-12)
        and np.allclose(tangent, persisted_tangent, rtol=0.0, atol=1e-12)
    )
    closure_settings = config["closure_alignment"]
    limit = float(closure_settings["dynamic_p95_max_angle_deg"])
    minimum_inward_speed = float(
        closure_settings["min_inward_speed_m_s"]
    )
    require_positive_inward = bool(
        closure_settings["require_positive_inward_speed"]
    )
    per_finger: dict[str, Any] = {}
    valid_each = []
    p95_each = []
    inward_each = []
    for index, finger in enumerate(ACTIVE_FINGERS):
        mask = close_mask & valid[:, index]
        count = int(np.count_nonzero(mask))
        angles = angle[mask, index]
        inward_values = inward[mask, index]
        tangent_values = tangent[mask, index]
        p50 = float(np.percentile(angles, 50)) if count else 180.0
        p95 = float(np.percentile(angles, 95)) if count else 180.0
        maximum = float(np.max(angles)) if count else 180.0
        min_inward = float(np.min(inward_values)) if count else 0.0
        p95_tangent = (
            float(np.percentile(tangent_values, 95)) if count else 0.0
        )
        per_finger[finger] = {
            "valid_count": count,
            "angle_p50_deg": p50,
            "angle_p95_deg": p95,
            "angle_max_deg": maximum,
            "minimum_inward_speed_m_s": min_inward,
            "tangent_speed_p95_m_s": p95_tangent,
        }
        valid_each.append(count > 0)
        p95_each.append(p95 <= limit + 1e-12)
        inward_each.append(
            count > 0
            and min_inward + 1e-12 >= minimum_inward_speed
            and (not require_positive_inward or min_inward > 0.0)
        )
    worst_finger = max(
        ACTIVE_FINGERS,
        key=lambda finger: per_finger[finger]["angle_p95_deg"],
    )
    metrics = {
        "closure_alignment": {
            "measurement_phase": "CLOSE",
            "dynamic_p95_limit_deg": limit,
            "minimum_inward_speed_threshold_m_s": minimum_inward_speed,
            "require_positive_inward_speed": require_positive_inward,
            "optimization_target_max_angle_deg": float(
                config["closure_alignment"][
                    "optimization_target_max_angle_deg"
                ]
            ),
            "per_finger": per_finger,
            "worst_finger": worst_finger,
            "worst_p95_angle_deg": float(
                per_finger[worst_finger]["angle_p95_deg"]
            ),
            "worst_max_angle_deg": float(
                max(value["angle_max_deg"] for value in per_finger.values())
            ),
            "worst_minimum_inward_speed_m_s": float(
                min(
                    value["minimum_inward_speed_m_s"]
                    for value in per_finger.values()
                )
            ),
        }
    }
    checks = {
        "v8_closure_alignment_trace_matches_vectors": trace_matches,
        "closure_alignment_valid_for_all_fingers": bool(all(valid_each)),
        "closure_alignment_p95_within_limit": bool(all(p95_each)),
        "closure_inward_speed_positive": bool(all(inward_each)),
    }
    return metrics, checks


def _v8_motion_smoothness_metrics(
    model: mujoco.MjModel,
    config: dict[str, Any],
    traces: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, bool]]:
    """Evaluate and persist gravity-relative smooth near-vertical motion."""

    from .motion_smoothness import (
        MotionSmoothnessThresholds,
        evaluate_smooth_motion,
    )

    acquired = _trace_scalar_int(traces, "grasp_acquisition_step")
    manipulation_start = _trace_scalar_int(traces, "manipulation_start_step")
    manipulation_end = _trace_scalar_int(traces, "manipulation_end_step")
    states = np.asarray(traces["control_state"]).astype(str)
    hold_steps = np.flatnonzero(states == "HOLD")
    hold_start = int(hold_steps[0]) if hold_steps.size else -1
    thresholds = MotionSmoothnessThresholds.from_mapping(
        config["motion_smoothness"]
    )
    smooth = evaluate_smooth_motion(
        traces["cube_pos"],
        traces["cube_quat"],
        np.asarray(traces["cube_velocity"], dtype=np.float64)[:, :3],
        model.opt.gravity,
        timestep_s=float(model.opt.timestep),
        baseline_step=acquired,
        manipulation_start_step=manipulation_start,
        manipulation_end_step=manipulation_end,
        hold_start_step=hold_start,
        thresholds=thresholds,
    )
    total_steps = states.shape[0]
    traces["operation_lateral_displacement_m"][:] = (
        smooth.relative_motion.lateral_distance_m
    )
    traces["operation_orientation_drift_deg"][:] = (
        smooth.relative_motion.orientation_drift_deg
    )
    centres = smooth.filtered_center_steps
    if centres.size:
        traces["operation_height_filtered_m"][centres] = (
            smooth.filtered_height_m
        )
        traces["motion_filter_valid"][centres] = True
    if smooth.vertical_speed_m_s.size:
        traces["operation_vertical_velocity_filtered_m_s"][centres[1:]] = (
            smooth.vertical_speed_m_s
        )
    if smooth.vertical_acceleration_m_s2.size:
        traces[
            "operation_vertical_acceleration_filtered_m_s2"
        ][centres[2:]] = smooth.vertical_acceleration_m_s2
    if smooth.vertical_jerk_m_s3.size:
        traces["operation_vertical_jerk_filtered_m_s3"][centres[3:]] = (
            smooth.vertical_jerk_m_s3
        )
    if any(
        np.asarray(traces[name]).shape != (total_steps,)
        for name in (
            "operation_height_filtered_m",
            "operation_lateral_displacement_m",
            "operation_orientation_drift_deg",
            "motion_filter_valid",
        )
    ):
        raise ValueError("schema-v8 derived motion traces have an invalid shape")
    return {"motion_smoothness": dict(smooth.metrics)}, dict(smooth.checks)


def _v9_actual_grasp_pose_metrics(
    model: mujoco.MjModel,
    config: dict[str, Any],
    traces: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, bool]]:
    """Recompute the measured contact pose without trusting commands/caches."""

    total_steps = int(np.asarray(traces["time"]).shape[0])
    states = np.asarray(traces["control_state"]).astype(str)
    gate = np.asarray(traces["grasp_gate"], dtype=bool)
    if states.shape != (total_steps,) or gate.shape[0] != total_steps:
        raise ValueError("schema-v9 state/gate trace length mismatch")
    recomputed_base_gate = np.all(gate, axis=1) & (states == "VERIFY")
    persisted_base_gate = np.asarray(
        traces["grasp_pose_base_gate"], dtype=bool
    )
    if persisted_base_gate.shape != (total_steps,):
        raise ValueError(
            f"grasp_pose_base_gate must have shape ({total_steps},)"
        )

    target_effective = np.asarray(
        traces["target_face_effective"], dtype=bool
    )
    if target_effective.shape != (total_steps, 3):
        raise ValueError(
            f"target_face_effective must have shape ({total_steps}, 3)"
        )
    result = evaluate_actual_grasp_pose_trace(
        model,
        config,
        traces,
        base_gate_mask=recomputed_base_gate,
        contact_mask=np.any(
            np.asarray(traces["finger_contact_force"], dtype=np.float64)
            > 1e-8,
            axis=1,
        ),
    )

    active_ids = np.asarray(
        [model.actuator(name).id for name in result.actuator_names],
        dtype=np.int64,
    )
    joint_qpos = np.asarray(traces["joint_qpos"], dtype=np.float64)
    raw_actual = joint_qpos[:, active_ids]
    persisted_actual = np.asarray(
        traces["grasp_pose_actual_joint_qpos_rad"], dtype=np.float64
    )
    nominal = np.asarray(
        [
            float(config["grasp_pose"]["nominal_joint_qpos_rad"][name])
            for name in result.actuator_names
        ],
        dtype=np.float64,
    )
    persisted_nominal = np.asarray(
        traces["grasp_pose_nominal_joint_qpos_rad"], dtype=np.float64
    )
    expected_precontact = np.zeros(model.nu, dtype=np.float64)
    for name, value in precontact_targets(config).items():
        expected_precontact[model.actuator(name).id] = float(value)
    persisted_precontact = np.asarray(
        traces["precontact_target_rad"], dtype=np.float64
    )
    thumb_id = model.actuator(
        "left_hand_thumb_bend_joint_actuator"
    ).id
    thumb_lower, thumb_upper = (
        float(value)
        for value in config["grasp_pose"]["thumb_actual_range_rad"]
    )
    thumb_qpos = joint_qpos[:, thumb_id]
    raw_thumb_range = (thumb_qpos >= thumb_lower - 1e-12) & (
        thumb_qpos <= thumb_upper + 1e-12
    )
    persisted_thumb_range = np.asarray(
        traces["grasp_pose_thumb_actual_within_range"], dtype=bool
    )
    per_step_trace_matches = bool(
        persisted_actual.shape == raw_actual.shape
        and np.array_equal(persisted_actual, raw_actual)
        and persisted_nominal.shape == nominal.shape
        and np.array_equal(persisted_nominal, nominal)
        and persisted_precontact.shape == expected_precontact.shape
        and np.array_equal(persisted_precontact, expected_precontact)
        and persisted_thumb_range.shape == raw_thumb_range.shape
        and np.array_equal(persisted_thumb_range, raw_thumb_range)
        and np.array_equal(persisted_base_gate, recomputed_base_gate)
    )

    expected_fields = result.as_trace_fields()
    summary_trace_matches = all(
        name in traces
        and np.array_equal(np.asarray(traces[name]), np.asarray(expected))
        for name, expected in expected_fields.items()
    )
    lock_step = result.events.grasp_lock_step
    online_offline_match = bool(
        _trace_scalar_int(traces, "grasp_acquisition_step") == lock_step
    )
    controller_fields = [
        "grasp_stable_window_start_step",
        "grasp_stable_window_end_step",
        "grasp_lock_step",
    ]
    # An unlocked controller has no authoritative pose summary; the offline
    # evaluator deliberately retains a near-miss diagnostic window.  Compare
    # qpos summaries only after a real lock, while event frames must always
    # agree (including the all-minus-one failure case).
    if lock_step >= 0:
        controller_fields.extend(
            [
                "grasp_pose_actual_qpos_rad",
                "grasp_pose_nominal_error_rad",
                "grasp_pose_joint_stability_span_rad",
            ]
        )
    for name in controller_fields:
        controller_name = f"controller_{name}"
        if controller_name in traces:
            online_offline_match = bool(
                online_offline_match
                and np.array_equal(
                    np.asarray(traces[controller_name]),
                    np.asarray(expected_fields[name]),
                )
            )

    metrics = {
        "actual_grasp_pose": result.as_summary(),
        "grasp_pose_id": grasp_pose_id(config),
        "controller_id": controller_id(config),
    }
    checks = {
        **dict(result.checks),
        "v9_actual_qpos_trace_matches_raw_state": per_step_trace_matches,
        "v9_actual_grasp_pose_summary_matches_raw_state": summary_trace_matches,
        "v9_online_grasp_lock_matches_offline_recomputation": (
            online_offline_match
        ),
        "v9_preload_command_is_not_grasp_pose_evidence": bool(
            not result.metrics["preload_command_used_as_acceptance_evidence"]
        ),
    }
    return metrics, checks


def _v12_contact_point_metrics(
    model: mujoco.MjModel,
    config: dict[str, Any],
    traces: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, bool]]:
    """Recompute schema-v12 cube-local point targeting from raw evidence."""

    from .controller import compute_target_face_evidence

    plan = contact_point_plan_from_config(config)
    total_steps = int(np.asarray(traces["time"]).shape[0])
    face_force = np.asarray(traces["distal_face_force_n"], dtype=np.float64)
    face_moment = np.asarray(
        traces["distal_face_position_moment_n_m"], dtype=np.float64
    )
    nondistal = np.asarray(
        traces["active_nondistal_force_n"], dtype=np.float64
    )
    tactile = np.asarray(traces["tactile_max"], dtype=np.float64)
    cube_position = np.asarray(traces["cube_pos"], dtype=np.float64)
    cube_quaternion = np.asarray(traces["cube_quat"], dtype=np.float64)
    if face_force.shape != (total_steps, 3, len(FACE_ORDER)):
        raise ValueError("schema-v12 distal face force trace has invalid shape")
    if face_moment.shape != (total_steps, 3, len(FACE_ORDER), 3):
        raise ValueError("schema-v12 distal face moment trace has invalid shape")
    if nondistal.shape != (total_steps, 3) or tactile.shape != (total_steps, 5):
        raise ValueError("schema-v12 contact/tactile trace has invalid shape")
    if cube_position.shape != (total_steps, 3) or cube_quaternion.shape != (
        total_steps,
        4,
    ):
        raise ValueError("schema-v12 cube pose trace has invalid shape")

    target_faces = tuple(face_from_label(value) for value in plan.target_faces)
    recomputed_local = np.zeros((total_steps, 3, 3), dtype=np.float64)
    recomputed_error = np.zeros((total_steps, 3), dtype=np.float64)
    recomputed_within = np.zeros((total_steps, 3), dtype=bool)
    centroid_valid = np.zeros((total_steps, 3), dtype=bool)
    target_face_effective = np.zeros((total_steps, 3), dtype=bool)
    target_face_force = np.zeros((total_steps, 3), dtype=np.float64)
    for step in range(total_steps):
        centroid_world, valid = target_face_contact_centroids(
            face_force[step], face_moment[step], target_faces
        )
        target = compute_target_face_evidence(
            config,
            face_force[step],
            nondistal[step],
            tactile[step, :3],
        )
        rotation_flat = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rotation_flat, cube_quaternion[step])
        observation = contact_point_observation(
            plan,
            centroid_world,
            valid,
            target.target_face_effective,
            cube_position[step],
            rotation_flat.reshape(3, 3),
        )
        centroid_valid[step] = valid
        target_face_effective[step] = target.target_face_effective
        target_face_force[step] = target.target_force_n
        recomputed_local[step] = observation.centroid_cube_local_m
        recomputed_error[step] = observation.tangent_error_m
        recomputed_within[step] = observation.within_radius

    persisted_target = np.asarray(
        traces["target_contact_points_cube_local_m"], dtype=np.float64
    )
    persisted_local = np.asarray(
        traces["target_face_contact_centroid_cube_local_m"], dtype=np.float64
    )
    persisted_error = np.asarray(
        traces["target_contact_point_tangent_error_m"], dtype=np.float64
    )
    persisted_within = np.asarray(
        traces["target_contact_point_within_radius"], dtype=bool
    )
    persisted_radius = float(
        np.asarray(traces["target_contact_point_radius_m"]).reshape(())
    )
    persisted_plan_id = str(
        np.asarray(traces["contact_point_plan_id"]).reshape(())
    )
    if persisted_target.shape != (3, 3):
        raise ValueError("target_contact_points_cube_local_m must have shape (3, 3)")
    if persisted_local.shape != recomputed_local.shape:
        raise ValueError(
            "target_face_contact_centroid_cube_local_m must have shape "
            f"{recomputed_local.shape}"
        )
    if persisted_error.shape != recomputed_error.shape:
        raise ValueError(
            "target_contact_point_tangent_error_m must have shape "
            f"{recomputed_error.shape}"
        )
    if persisted_within.shape != recomputed_within.shape:
        raise ValueError(
            "target_contact_point_within_radius must have shape "
            f"{recomputed_within.shape}"
        )

    gate_order = tuple(
        str(value) for value in np.asarray(traces["grasp_gate_order"])
    )
    try:
        gate_column = gate_order.index("contact_points_within_target_regions")
    except ValueError as error:
        raise ValueError(
            "schema-v12 grasp gate is missing contact point targeting"
        ) from error
    persisted_gate = np.asarray(traces["grasp_gate"], dtype=bool)[:, gate_column]
    recomputed_gate = np.all(recomputed_within, axis=1)

    trace_matches = bool(
        persisted_plan_id == plan.point_plan_id
        and persisted_radius == plan.target_radius_m
        and np.array_equal(persisted_target, plan.target_points_cube_local_m)
        and np.allclose(persisted_local, recomputed_local, rtol=0.0, atol=1e-14)
        and np.allclose(persisted_error, recomputed_error, rtol=0.0, atol=1e-14)
        and np.array_equal(persisted_within, recomputed_within)
    )
    gate_matches = bool(np.array_equal(persisted_gate, recomputed_gate))

    states = np.asarray(traces["control_state"]).astype(str)
    if states.shape != (total_steps,):
        raise ValueError("schema-v12 control state trace has invalid shape")

    def stage_statistics(mask: np.ndarray) -> dict[str, Any]:
        sample_count = int(np.count_nonzero(mask))
        per_finger: dict[str, Any] = {}
        for finger_index, finger in enumerate(ACTIVE_FINGERS):
            valid_mask = mask & centroid_valid[:, finger_index]
            values = recomputed_error[valid_mask, finger_index]
            per_finger[finger] = {
                "valid_sample_count": int(values.size),
                "within_radius_duty": (
                    float(np.mean(recomputed_within[mask, finger_index]))
                    if sample_count
                    else 0.0
                ),
                "tangent_error_p50_m": (
                    float(np.percentile(values, 50.0)) if values.size else None
                ),
                "tangent_error_p95_m": (
                    float(np.percentile(values, 95.0)) if values.size else None
                ),
                "tangent_error_max_m": (
                    float(np.max(values)) if values.size else None
                ),
            }
        return {
            "sample_count": sample_count,
            "all_points_within_radius_duty": (
                float(np.mean(recomputed_gate[mask])) if sample_count else 0.0
            ),
            "per_finger": per_finger,
        }

    acquisition_step = _trace_scalar_int(traces, "grasp_acquisition_step")
    stable_steps = int(
        round(
            float(config["control_protocol"]["stable_window_s"])
            / float(model.opt.timestep)
        )
    )
    acquisition_start = acquisition_step - stable_steps + 1
    acquisition_contiguous = bool(
        acquisition_step >= 0
        and acquisition_start >= 0
        and np.all(recomputed_gate[acquisition_start : acquisition_step + 1])
    )
    verify_mask = states == "VERIFY"
    manipulate_mask = states == "MANIPULATE"
    hold_mask = states == "HOLD"
    operation_mask = manipulate_mask | hold_mask
    acquisition_mask = np.zeros(total_steps, dtype=bool)
    if acquisition_step >= 0 and acquisition_start >= 0:
        acquisition_mask[acquisition_start : acquisition_step + 1] = True

    metrics = {
        "contact_point_targeting": {
            "point_plan_id": plan.point_plan_id,
            "coordinate_frame": "cube_local",
            "target_radius_m": plan.target_radius_m,
            "target_points_cube_local_m": {
                finger: plan.target_points_cube_local_m[index].tolist()
                for index, finger in enumerate(ACTIVE_FINGERS)
            },
            "verify": stage_statistics(verify_mask),
            "acquisition_window": stage_statistics(acquisition_mask),
        }
    }
    checks = {
        "v12_contact_point_trace_matches_raw_contacts": trace_matches,
        "v12_contact_point_gate_matches_raw_contacts": gate_matches,
        "grasp_contact_points_contiguous": acquisition_contiguous,
    }
    if int(config.get("schema_version", 1)) >= 13:
        slip = contact_tangent_slip_from_grasp(
            recomputed_local,
            centroid_valid,
            target_face_effective,
            target_face_force,
            plan.target_faces,
            acquisition_start_step=acquisition_start,
            acquisition_end_step=acquisition_step,
        )

        persisted_baseline = np.asarray(
            traces["grasp_contact_centroid_baseline_cube_local_m"],
            dtype=np.float64,
        )
        persisted_baseline_valid = np.asarray(
            traces["grasp_contact_centroid_baseline_valid"], dtype=bool
        )
        persisted_force_sum = np.asarray(
            traces["grasp_contact_centroid_baseline_force_sum_n"],
            dtype=np.float64,
        )
        persisted_slip = np.asarray(
            traces["target_contact_tangent_slip_from_grasp_m"],
            dtype=np.float64,
        )
        persisted_slip_valid = np.asarray(
            traces["target_contact_tangent_slip_from_grasp_valid"], dtype=bool
        )
        if persisted_baseline.shape != (3, 3):
            raise ValueError(
                "grasp_contact_centroid_baseline_cube_local_m must have "
                "shape (3, 3)"
            )
        if persisted_baseline_valid.shape != (3,):
            raise ValueError(
                "grasp_contact_centroid_baseline_valid must have shape (3,)"
            )
        if persisted_force_sum.shape != (3,):
            raise ValueError(
                "grasp_contact_centroid_baseline_force_sum_n must have "
                "shape (3,)"
            )
        if persisted_slip.shape != (total_steps, 3):
            raise ValueError(
                "target_contact_tangent_slip_from_grasp_m must have shape "
                f"({total_steps}, 3)"
            )
        if persisted_slip_valid.shape != (total_steps, 3):
            raise ValueError(
                "target_contact_tangent_slip_from_grasp_valid must have shape "
                f"({total_steps}, 3)"
            )

        slip_trace_matches = bool(
            np.allclose(
                persisted_baseline,
                slip.baseline_centroid_cube_local_m,
                rtol=0.0,
                atol=1e-14,
            )
            and np.array_equal(
                persisted_baseline_valid, slip.baseline_valid
            )
            and np.allclose(
                persisted_force_sum,
                slip.baseline_force_sum_n,
                rtol=0.0,
                atol=1e-14,
            )
            and np.allclose(
                persisted_slip,
                slip.tangent_slip_from_grasp_m,
                rtol=0.0,
                atol=1e-14,
            )
            and np.array_equal(
                persisted_slip_valid, slip.tangent_slip_valid
            )
        )

        def slip_stage_statistics(mask: np.ndarray) -> dict[str, Any]:
            sample_count = int(np.count_nonzero(mask))
            per_finger: dict[str, Any] = {}
            simultaneous = mask & np.all(slip.tangent_slip_valid, axis=1)
            for finger_index, finger in enumerate(ACTIVE_FINGERS):
                valid_mask = mask & slip.tangent_slip_valid[:, finger_index]
                values = slip.tangent_slip_from_grasp_m[
                    valid_mask, finger_index
                ]
                per_finger[finger] = {
                    "valid_sample_count": int(values.size),
                    "valid_duty": (
                        float(values.size / sample_count) if sample_count else 0.0
                    ),
                    "tangent_slip_p50_m": (
                        float(np.percentile(values, 50.0))
                        if values.size
                        else None
                    ),
                    "tangent_slip_p95_m": (
                        float(np.percentile(values, 95.0))
                        if values.size
                        else None
                    ),
                    "tangent_slip_max_m": (
                        float(np.max(values)) if values.size else None
                    ),
                }
            return {
                "sample_count": sample_count,
                "simultaneous_valid_duty": (
                    float(np.count_nonzero(simultaneous) / sample_count)
                    if sample_count
                    else 0.0
                ),
                "per_finger": per_finger,
            }

        targeting = metrics["contact_point_targeting"]
        targeting.update(
            {
                "manipulate": stage_statistics(manipulate_mask),
                "hold": stage_statistics(hold_mask),
                "operation": stage_statistics(operation_mask),
                "contact_slip_from_grasp": {
                    "baseline_method": (
                        "acquisition_window_target_normal_force_weighted_"
                        "cube_local_centroid"
                    ),
                    "soft_ranking_only": True,
                    "baseline": {
                        finger: {
                            "valid": bool(slip.baseline_valid[index]),
                            "centroid_cube_local_m": (
                                slip.baseline_centroid_cube_local_m[index].tolist()
                            ),
                            "target_normal_force_sum_n": float(
                                slip.baseline_force_sum_n[index]
                            ),
                        }
                        for index, finger in enumerate(ACTIVE_FINGERS)
                    },
                    "manipulate": slip_stage_statistics(manipulate_mask),
                    "hold": slip_stage_statistics(hold_mask),
                    "operation": slip_stage_statistics(operation_mask),
                },
            }
        )
        checks["v13_contact_slip_trace_matches_raw_contacts"] = (
            slip_trace_matches
        )
    return metrics, checks


def _v14_contact_preservation_metrics(
    model: mujoco.MjModel,
    config: dict[str, Any],
    traces: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, bool]]:
    """Audit contact-preserving planning from raw force and command traces.

    The online controller is allowed to bridge isolated solver-noise samples,
    but published manipulation evidence must retain each target-face contact
    for at least 99% of MANIPULATE+HOLD and may never contain a loss longer
    than the configured 10 ms allowance.  Derived force/error/loss arrays are
    visualization caches; the verdict below is recomputed from raw per-face
    forces, tactile samples and the resolved configuration.
    """

    schema_version = int(config.get("schema_version", 14))
    identity_audit = validate_v14_top_level_identities(config)
    expected_identity = identity_audit["values"]
    persisted_identity: dict[str, str | None] = {}
    identity_trace_matches = True
    if schema_version == 14:
        for name in V14_TOP_LEVEL_ID_FIELDS:
            raw = traces.get(name)
            if raw is None or np.asarray(raw).shape != ():
                persisted_identity[name] = None
                identity_trace_matches = False
                continue
            value = str(np.asarray(raw).reshape(()))
            persisted_identity[name] = value
            identity_trace_matches = bool(
                identity_trace_matches and value == expected_identity[name]
            )
    else:
        # v15 has its own object/grasp/pair/planner/controller identity
        # contract in config validation.  Do not reinterpret those hashes as
        # one of the sealed historical v14 controller formats.
        identity_audit = {
            "state": "superseded_by_schema_v15",
            "values": {},
            "planner_verification": "not_applicable",
            "controller_format": "not_applicable",
        }
        expected_identity = {}
        persisted_identity = {}

    total_steps = int(np.asarray(traces["time"]).shape[0])
    timestep_s = float(model.opt.timestep)
    states = np.asarray(traces["control_state"]).astype(str)
    if states.shape != (total_steps,):
        raise ValueError("schema-v14 control state trace has invalid shape")
    operation_mask = (states == "MANIPULATE") | (states == "HOLD")
    operation_indices = np.flatnonzero(operation_mask)

    face_force = np.asarray(traces["distal_face_force_n"], dtype=np.float64)
    nondistal = np.asarray(
        traces["active_nondistal_force_n"], dtype=np.float64
    )
    tactile = np.asarray(traces["tactile_max"], dtype=np.float64)
    if face_force.shape != (total_steps, 3, len(FACE_ORDER)):
        raise ValueError("schema-v14 distal face force trace has invalid shape")
    if nondistal.shape != (total_steps, 3):
        raise ValueError("schema-v14 active non-distal force trace has invalid shape")
    if tactile.shape != (total_steps, 5):
        raise ValueError("schema-v14 tactile trace has invalid shape")
    if (
        not np.isfinite(face_force).all()
        or not np.isfinite(nondistal).all()
        or not np.isfinite(tactile).all()
        or np.any(face_force < 0.0)
        or np.any(nondistal < 0.0)
    ):
        raise ValueError("schema-v14 contact evidence must be finite and non-negative")

    topology = config["contact_topology"]
    gate_config = config["control_protocol"]["grasp_gate"]
    target_indices = np.asarray(
        [
            FACE_ORDER.index(face_from_label(topology["target_faces"][finger]))
            for finger in ACTIVE_FINGERS
        ],
        dtype=np.int64,
    )
    target_force = face_force[:, np.arange(3), target_indices]
    total_distal = np.sum(face_force, axis=2)
    purity = np.divide(
        target_force,
        total_distal,
        out=np.zeros_like(target_force),
        where=total_distal > 0.0,
    )
    force_min_n = float(gate_config["min_target_face_force_n"])
    purity_min = float(gate_config["min_target_force_fraction"])
    target_effective = (target_force >= force_min_n) & (purity >= purity_min)
    if bool(gate_config["require_touch"]):
        target_effective &= tactile[:, :3] >= float(
            config["acceptance"]["touch_force_min_n"]
        )

    off_target = np.maximum(0.0, total_distal - target_force)
    off_fraction = np.divide(
        off_target,
        total_distal,
        out=np.zeros_like(off_target),
        where=total_distal > 0.0,
    )
    max_off_fraction = 1.0 - purity_min
    material_off = (off_target >= force_min_n) & (
        off_fraction > max_off_fraction + 1e-12
    )
    combined = total_distal + nondistal
    nondistal_fraction = np.divide(
        nondistal,
        combined,
        out=np.zeros_like(nondistal),
        where=combined > 0.0,
    )
    material_nondistal = (nondistal >= force_min_n) & (
        nondistal_fraction > max_off_fraction + 1e-12
    )

    native_target_effective = target_effective
    controller_target_effective = native_target_effective
    v16_native_trace_matches = True
    v16_rolling_trace_matches = True
    if schema_version >= 16:
        rolling_evidence = _v16_rolling_aware_target_face_evidence(
            config, traces
        )
        controller_target_effective = rolling_evidence["effective"]
        persisted_native = np.asarray(
            traces["native_tactile_target_face_effective"], dtype=bool
        )
        persisted_rolling = np.asarray(
            traces["rolling_aware_target_face_effective"], dtype=bool
        )
        if (
            persisted_native.shape != native_target_effective.shape
            or persisted_rolling.shape != controller_target_effective.shape
        ):
            raise ValueError(
                "schema-v16 persisted target-face evidence has invalid shape"
            )
        v16_native_trace_matches = bool(
            np.array_equal(persisted_native, native_target_effective)
        )
        v16_rolling_trace_matches = bool(
            np.array_equal(persisted_rolling, controller_target_effective)
        )

    operation_effective = controller_target_effective[operation_mask]
    if operation_effective.shape[0]:
        per_finger_duty = np.mean(operation_effective, axis=0)
        simultaneous_effective = np.all(operation_effective, axis=1)
        simultaneous_duty = float(np.mean(simultaneous_effective))
        longest_loss_steps = np.asarray(
            [
                _longest_false_run_steps(operation_effective[:, index])
                for index in range(3)
            ],
            dtype=np.int64,
        )
        simultaneous_longest_loss_steps = _longest_false_run_steps(
            simultaneous_effective
        )
    else:
        per_finger_duty = np.zeros(3, dtype=np.float64)
        simultaneous_duty = 0.0
        longest_loss_steps = np.zeros(3, dtype=np.int64)
        simultaneous_longest_loss_steps = 0

    feedback_config = config["contact_feedback"]
    duty_min = float(feedback_config["operation_contact_duty_min"])
    max_loss_s = float(feedback_config["max_loss_s"])
    allowed_loss_steps = int(round(max_loss_s / timestep_s))
    if (
        allowed_loss_steps <= 0
        or abs(allowed_loss_steps * timestep_s - max_loss_s)
        > 0.5 * timestep_s + 1e-12
    ):
        raise ValueError("contact_feedback.max_loss_s must align with timestep")

    # Recompute the controller's online consecutive-loss state.  The state is
    # updated after the physics sample that used the current command and is
    # retained unchanged once ABORT has latched.
    recomputed_loss_run = np.zeros((total_steps, 3), dtype=np.int64)
    current_loss = np.zeros(3, dtype=np.int64)
    for step in range(total_steps):
        if operation_mask[step]:
            current_loss = np.where(
                controller_target_effective[step], 0, current_loss + 1
            )
        recomputed_loss_run[step] = current_loss
    persisted_loss_run = np.asarray(
        traces["contact_loss_run_steps"], dtype=np.int64
    )
    if persisted_loss_run.shape != recomputed_loss_run.shape:
        raise ValueError(
            "contact_loss_run_steps must have shape "
            f"{recomputed_loss_run.shape}"
        )
    persisted_maximum_loss = np.asarray(
        traces["maximum_contact_loss_run_steps"], dtype=np.int64
    )
    if persisted_maximum_loss.shape != (3,):
        raise ValueError("maximum_contact_loss_run_steps must have shape (3,)")
    loss_trace_matches = bool(
        np.array_equal(persisted_loss_run, recomputed_loss_run)
        and np.array_equal(
            persisted_maximum_loss,
            np.max(recomputed_loss_run, axis=0, initial=0),
        )
    )

    # Audit the one-sample causal feedback contract.  A command at t may only
    # react to contact evidence observed after step t-1.
    feedback_source = np.asarray(
        traces["operation_feedback_source_step"], dtype=np.int64
    )
    if feedback_source.shape != (total_steps,):
        raise ValueError(
            f"operation_feedback_source_step must have shape ({total_steps},)"
        )
    expected_feedback_source = np.arange(total_steps, dtype=np.int64) - 1
    causal_source_matches = bool(
        np.array_equal(feedback_source, expected_feedback_source)
    )

    risk_force_n = float(feedback_config["force_risk_n"])
    contact_risk = (
        np.any(target_force < risk_force_n, axis=1)
        | np.any(purity < purity_min, axis=1)
        | ~np.all(controller_target_effective, axis=1)
        | np.any(material_off, axis=1)
        | np.any(material_nondistal, axis=1)
    )
    slip_feedback_enabled = int(feedback_config.get("schema_version", 1)) >= 2
    # Schema v16 records the legacy centroid proxy for diagnostics only.  Its
    # controller sanitizes that proxy before the inherited v14/v15 risk path,
    # then adds rolling material-slip risk explicitly.
    legacy_slip_control_enabled = slip_feedback_enabled and schema_version < 16
    slip_trace_matches = True
    slip_risk_trace_matches = True
    slip_abort_trace_matches = True
    slip_freeze_observation = np.zeros((total_steps, 3), dtype=bool)
    slip_abort_observation = np.zeros((total_steps, 3), dtype=bool)
    pair_freeze_observation = np.zeros(total_steps, dtype=bool)
    pair_slip_freeze = np.zeros((total_steps, 3), dtype=bool)
    rolling_freeze_observation = np.zeros((total_steps, 3), dtype=bool)
    if slip_feedback_enabled:
        online_slip = np.asarray(
            traces["online_contact_tangent_slip_m"], dtype=np.float64
        )
        online_slip_valid = np.asarray(
            traces["online_contact_tangent_slip_valid"], dtype=bool
        )
        derived_slip = np.asarray(
            traces["target_contact_tangent_slip_from_grasp_m"],
            dtype=np.float64,
        )
        derived_slip_valid = np.asarray(
            traces["target_contact_tangent_slip_from_grasp_valid"], dtype=bool
        )
        expected_shape = (total_steps, 3)
        if any(
            value.shape != expected_shape
            for value in (
                online_slip,
                online_slip_valid,
                derived_slip,
                derived_slip_valid,
            )
        ):
            raise ValueError("schema-v14 online tangent-slip traces have invalid shape")
        online_baseline = np.asarray(
            traces["online_grasp_contact_centroid_baseline_cube_local_m"],
            dtype=np.float64,
        )
        online_baseline_valid = np.asarray(
            traces["online_grasp_contact_centroid_baseline_valid"], dtype=bool
        )
        online_baseline_force = np.asarray(
            traces["online_grasp_contact_centroid_baseline_force_sum_n"],
            dtype=np.float64,
        )
        online_acquisition_step = _trace_scalar_int(
            traces, "grasp_acquisition_step"
        )
        online_available = np.arange(total_steps) >= online_acquisition_step
        if online_acquisition_step < 0:
            online_available[:] = False
        expected_online_valid = derived_slip_valid & online_available[:, None]
        expected_online_slip = np.where(
            expected_online_valid, derived_slip, 0.0
        )
        slip_trace_matches = bool(
            np.allclose(
                online_slip, expected_online_slip, rtol=0.0, atol=1e-14
            )
            and np.array_equal(online_slip_valid, expected_online_valid)
            and np.allclose(
                online_baseline,
                np.asarray(
                    traces["grasp_contact_centroid_baseline_cube_local_m"],
                    dtype=np.float64,
                ),
                rtol=0.0,
                atol=1e-14,
            )
            and np.array_equal(
                online_baseline_valid,
                np.asarray(
                    traces["grasp_contact_centroid_baseline_valid"], dtype=bool
                ),
            )
            and np.allclose(
                online_baseline_force,
                np.asarray(
                    traces["grasp_contact_centroid_baseline_force_sum_n"],
                    dtype=np.float64,
                ),
                rtol=0.0,
                atol=1e-14,
            )
        )
        if legacy_slip_control_enabled:
            slip_freeze_observation = online_slip_valid & (
                online_slip
                > float(feedback_config["tangent_slip_freeze_threshold_m"])
            )
            slip_abort_observation = (
                operation_mask[:, None]
                & online_slip_valid
                & (
                    online_slip
                    > float(feedback_config["tangent_slip_abort_threshold_m"])
                )
            )
    if schema_version >= 15:
        pair_feedback = config["joint_pair_feedback"]
        pair_alignment = config["joint_pair_alignment"]
        pair_angle = np.asarray(traces["joint_pair_angle_deg"], dtype=np.float64)
        pair_length = np.asarray(traces["joint_pair_length_m"], dtype=np.float64)
        pair_positive = np.asarray(traces["joint_pair_positive_y"], dtype=bool)
        pair_valid = np.asarray(traces["joint_pair_valid"], dtype=bool)
        if any(
            value.shape != (total_steps,)
            for value in (pair_angle, pair_length, pair_positive, pair_valid)
        ):
            raise ValueError("schema-v15 joint-pair risk traces have invalid shape")
        pair_freeze_observation = (
            ~pair_valid
            | ~pair_positive
            | (
                pair_length
                < float(pair_alignment["minimum_length_m"]) - 1e-12
            )
            | (
                pair_angle
                > float(pair_feedback["freeze_threshold_deg"]) + 1e-12
            )
        )
        if schema_version < 16:
            pair_slip_freeze = np.asarray(
                traces["online_contact_tangent_slip_valid"], dtype=bool
            ) & (
                np.asarray(
                    traces["online_contact_tangent_slip_m"], dtype=np.float64
                )
                >= float(pair_feedback["slip_freeze_threshold_m"]) - 1e-12
            )
    if schema_version >= 16:
        rolling_freeze_observation = np.asarray(
            traces["rolling_slip_freeze_active"], dtype=bool
        )
        if rolling_freeze_observation.shape != (total_steps, 3):
            raise ValueError(
                "schema-v16 rolling-slip freeze trace has invalid shape"
            )
    raw_risk = _versioned_operation_feedback_risk(
        schema_version,
        contact_risk,
        pair_risk=pair_freeze_observation,
        legacy_slip_risk=np.any(slip_freeze_observation, axis=1),
        pair_legacy_slip_risk=np.any(pair_slip_freeze, axis=1),
        rolling_slip_freeze_active=np.any(
            rolling_freeze_observation, axis=1
        ),
    )
    expected_frozen = np.zeros(total_steps, dtype=bool)
    expected_recovery = np.zeros(total_steps, dtype=bool)
    for step in operation_indices:
        source_step = step - 1
        at_risk = source_step < 0 or bool(raw_risk[source_step])
        expected_recovery[step] = at_risk
        expected_frozen[step] = states[step] == "MANIPULATE" and at_risk
    if slip_feedback_enabled:
        expected_slip_risk = np.zeros((total_steps, 3), dtype=bool)
        for step in operation_indices:
            source_step = step - 1
            if source_step >= 0:
                expected_slip_risk[step] = slip_freeze_observation[source_step]
        persisted_slip_risk = np.asarray(
            traces["contact_tangent_slip_freeze_risk"], dtype=bool
        )
        persisted_slip_abort = np.asarray(
            traces["contact_tangent_slip_abort_risk"], dtype=bool
        )
        if (
            persisted_slip_risk.shape != (total_steps, 3)
            or persisted_slip_abort.shape != (total_steps, 3)
        ):
            raise ValueError("schema-v14 tangent-slip risk traces have invalid shape")
        slip_risk_trace_matches = bool(
            np.array_equal(persisted_slip_risk, expected_slip_risk)
        )
        slip_abort_trace_matches = bool(
            np.array_equal(persisted_slip_abort, slip_abort_observation)
        )
    persisted_frozen = np.asarray(
        traces["contact_progress_frozen"], dtype=bool
    )
    persisted_recovery = np.asarray(
        traces["contact_recovery_active"], dtype=bool
    )
    if persisted_frozen.shape != (total_steps,) or persisted_recovery.shape != (
        total_steps,
    ):
        raise ValueError("schema-v14 freeze/recovery traces have invalid shape")
    causal_risk_matches = bool(
        np.array_equal(persisted_frozen, expected_frozen)
        and np.array_equal(persisted_recovery, expected_recovery)
    )

    # Check that the immutable plan persisted in the NPZ is exactly the plan
    # named by the resolved config, then independently interpolate all
    # operation feed-forward/path samples from the recorded causal progress.
    plan = config["manipulation_plan"]
    knot_times = np.asarray(plan["knot_times_s"], dtype=np.float64)
    knot_count = knot_times.size
    config_waypoints = np.zeros((knot_count, model.nu), dtype=np.float64)
    for actuator in ACTIVE_ACTUATORS:
        config_waypoints[:, model.actuator(actuator).id] = np.asarray(
            plan["actuator_waypoints_rad"][actuator], dtype=np.float64
        )
    config_desired_position = np.asarray(
        plan["desired_cube_position_delta_m"], dtype=np.float64
    )
    config_desired_rotation = np.asarray(
        plan["desired_cube_rotation_vector_rad"], dtype=np.float64
    )
    persisted_knots = np.asarray(
        traces["manipulation_plan_knot_times_s"], dtype=np.float64
    )
    persisted_waypoints = np.asarray(
        traces["manipulation_plan_waypoints_rad"], dtype=np.float64
    )
    persisted_desired_knots = np.asarray(
        traces["manipulation_plan_desired_cube_position_delta_m"],
        dtype=np.float64,
    )
    persisted_rotation_knots = np.asarray(
        traces["manipulation_plan_desired_cube_rotation_vector_rad"],
        dtype=np.float64,
    )
    static_plan_matches = bool(
        persisted_knots.shape == knot_times.shape
        and persisted_waypoints.shape == config_waypoints.shape
        and persisted_desired_knots.shape == config_desired_position.shape
        and persisted_rotation_knots.shape == config_desired_rotation.shape
        and np.array_equal(persisted_knots, knot_times)
        and np.array_equal(persisted_waypoints, config_waypoints)
        and np.array_equal(persisted_desired_knots, config_desired_position)
        and np.array_equal(persisted_rotation_knots, config_desired_rotation)
        and str(np.asarray(traces["manipulation_plan_id"]).reshape(()))
        == str(plan["plan_id"])
        and str(np.asarray(traces["contact_feedback_id"]).reshape(()))
        == str(feedback_config["feedback_id"])
    )

    progress = np.asarray(traces["manipulation_progress"], dtype=np.float64)
    feedforward = np.asarray(
        traces["planned_feedforward_target_rad"], dtype=np.float64
    )
    desired_position = np.asarray(
        traces["desired_cube_position_delta_m"], dtype=np.float64
    )
    desired_rotation = np.asarray(
        traces["desired_cube_rotation_vector_rad"], dtype=np.float64
    )
    knot_index = np.asarray(traces["planned_knot_index"], dtype=np.int64)
    correction = np.asarray(traces["feedback_correction_rad"], dtype=np.float64)
    controls = np.asarray(traces["ctrl"], dtype=np.float64)
    expected_matrix_shape = (total_steps, model.nu)
    if (
        progress.shape != (total_steps,)
        or feedforward.shape != expected_matrix_shape
        or correction.shape != expected_matrix_shape
        or controls.shape != expected_matrix_shape
        or desired_position.shape != (total_steps, 3)
        or desired_rotation.shape != (total_steps, 3)
        or knot_index.shape != (total_steps,)
    ):
        raise ValueError("schema-v14 planned command trace has invalid shape")

    preload = np.zeros(model.nu, dtype=np.float64)
    for actuator, value in contact_preload_targets(config).items():
        preload[model.actuator(actuator).id] = float(value)
    expected_feedforward = np.zeros_like(feedforward)
    expected_desired_position = np.zeros_like(desired_position)
    expected_desired_rotation = np.zeros_like(desired_rotation)
    expected_knot_index = np.zeros_like(knot_index)
    duration_s = float(plan["duration_s"])
    waypoint_velocity, waypoint_acceleration = quintic_c2_knot_derivatives(
        knot_times, config_waypoints
    )
    position_velocity, position_acceleration = quintic_c2_knot_derivatives(
        knot_times, config_desired_position
    )
    rotation_velocity, rotation_acceleration = quintic_c2_knot_derivatives(
        knot_times, config_desired_rotation
    )
    for step in operation_indices:
        elapsed_s = float(np.clip(progress[step], 0.0, 1.0) * duration_s)
        waypoint, _, _, index = interpolate_quintic_c2(
            knot_times,
            config_waypoints,
            elapsed_s,
            knot_velocities=waypoint_velocity,
            knot_accelerations=waypoint_acceleration,
        )
        position, _, _, position_index = interpolate_quintic_c2(
            knot_times,
            config_desired_position,
            elapsed_s,
            knot_velocities=position_velocity,
            knot_accelerations=position_acceleration,
        )
        rotation, _, _, rotation_index = interpolate_quintic_c2(
            knot_times,
            config_desired_rotation,
            elapsed_s,
            knot_velocities=rotation_velocity,
            knot_accelerations=rotation_acceleration,
        )
        if position_index != index or rotation_index != index:
            raise ValueError("schema-v14 interpolated plan axes diverged")
        expected_feedforward[step] = preload + waypoint
        expected_desired_position[step] = position
        expected_desired_rotation[step] = rotation
        expected_knot_index[step] = index

    knot_index_matches = np.ones(total_steps, dtype=bool)
    for step in operation_indices:
        if knot_index[step] == expected_knot_index[step]:
            continue
        elapsed_s = float(np.clip(progress[step], 0.0, 1.0) * duration_s)
        boundary_distance = float(np.min(np.abs(knot_times - elapsed_s)))
        knot_index_matches[step] = bool(
            boundary_distance <= 2e-12
            and abs(int(knot_index[step]) - int(expected_knot_index[step])) == 1
        )
    dynamic_plan_matches = bool(
        np.allclose(
            feedforward[operation_mask],
            expected_feedforward[operation_mask],
            rtol=0.0,
            atol=2e-12,
        )
        and np.allclose(
            desired_position[operation_mask],
            expected_desired_position[operation_mask],
            rtol=0.0,
            atol=2e-14,
        )
        and np.allclose(
            desired_rotation[operation_mask],
            expected_desired_rotation[operation_mask],
            rtol=0.0,
            atol=2e-14,
        )
        and np.all(knot_index_matches[operation_mask])
    )

    ctrl_limited = np.asarray(model.actuator_ctrllimited, dtype=bool)
    ctrl_lower = np.where(ctrl_limited, model.actuator_ctrlrange[:, 0], -np.inf)
    ctrl_upper = np.where(ctrl_limited, model.actuator_ctrlrange[:, 1], np.inf)
    total_correction = _planned_command_total_correction_rad(
        schema_version, traces, expected_matrix_shape
    )
    reconstructed_ctrl = np.clip(
        feedforward + total_correction, ctrl_lower, ctrl_upper
    )
    command_composition_matches = bool(
        np.allclose(
            controls[operation_mask],
            reconstructed_ctrl[operation_mask],
            rtol=0.0,
            atol=2e-14,
        )
    )

    persisted_force_target = np.asarray(
        traces["contact_force_target_n"], dtype=np.float64
    )
    persisted_filtered_force = np.asarray(
        traces["contact_force_filtered_n"], dtype=np.float64
    )
    persisted_integral = np.asarray(
        traces["contact_force_integral_n_s"], dtype=np.float64
    )
    for value, label in (
        (persisted_force_target, "contact_force_target_n"),
        (persisted_filtered_force, "contact_force_filtered_n"),
        (persisted_integral, "contact_force_integral_n_s"),
    ):
        if value.shape != (total_steps, 3):
            raise ValueError(f"{label} must have shape ({total_steps}, 3)")
    filter_alpha = timestep_s / (
        float(feedback_config["filter_time_constant_s"]) + timestep_s
    )
    expected_filtered = np.zeros_like(persisted_filtered_force)
    if total_steps:
        expected_filtered[0] = target_force[0]
        for step in range(1, total_steps):
            expected_filtered[step] = expected_filtered[step - 1] + filter_alpha * (
                target_force[step] - expected_filtered[step - 1]
            )
    resolved_targets = np.asarray(
        traces["resolved_contact_force_targets_n"], dtype=np.float64
    )
    force_minimum = float(config["contact_force_targets_n"]["minimum_n"])
    force_maximum = float(config["contact_force_targets_n"]["maximum_n"])
    configured_targets = np.asarray(
        [
            float(config["contact_force_targets_n"]["per_finger_n"][finger])
            for finger in ACTIVE_FINGERS
        ],
        dtype=np.float64,
    )
    expected_resolved_targets = configured_targets.copy()
    expected_target_trace = np.broadcast_to(
        configured_targets, persisted_force_target.shape
    ).copy()
    if "grasp_acquisition_step" in traces:
        acquisition_step = _trace_scalar_int(traces, "grasp_acquisition_step")
        stable_steps = int(
            round(
                float(config["control_protocol"]["stable_window_s"])
                / timestep_s
            )
        )
        acquisition_start = acquisition_step - stable_steps + 1
        if acquisition_step >= 0:
            if acquisition_start < 0:
                raise ValueError(
                    "schema-v14 grasp acquisition lacks a complete force window"
                )
            measured_targets = np.median(
                target_force[acquisition_start : acquisition_step + 1], axis=0
            )
            operation_scale = float(
                config["contact_force_targets_n"].get("operation_scale", 1.0)
            )
            if operation_scale == 1.0:
                # Preserve the exact legacy recomputation path for every
                # schema-v14 configuration that omits the optional scale.
                expected_resolved_targets = np.clip(
                    np.maximum(configured_targets, measured_targets),
                    force_minimum,
                    force_maximum,
                )
            else:
                expected_resolved_targets = np.clip(
                    operation_scale
                    * np.maximum(configured_targets, measured_targets),
                    force_minimum,
                    force_maximum,
                )
            # Simulation stores controller state after observing each sample,
            # so the resolved median first appears on the lock sample itself.
            expected_target_trace[acquisition_step:] = (
                expected_resolved_targets
            )
    force_trace_matches = bool(
        resolved_targets.shape == (3,)
        and np.allclose(
            persisted_filtered_force,
            expected_filtered,
            rtol=0.0,
            atol=2e-14,
        )
        and np.allclose(
            resolved_targets,
            expected_resolved_targets,
            rtol=0.0,
            atol=2e-14,
        )
        and np.allclose(
            persisted_force_target,
            expected_target_trace,
            rtol=0.0,
            atol=2e-14,
        )
        and np.all(persisted_force_target >= force_minimum - 1e-12)
        and np.all(persisted_force_target <= force_maximum + 1e-12)
        and np.all(
            np.abs(persisted_integral)
            <= float(feedback_config["integral_limit_n_s"]) + 1e-12
        )
        and np.all(
            np.abs(correction)
            <= float(feedback_config["correction_limit_rad"]) + 1e-12
        )
    )

    operation_abort = (
        bool(np.any(states[operation_indices[0] :] == "ABORT"))
        if operation_indices.size
        else bool(np.any(states == "ABORT"))
    )
    operation_progress = progress[operation_mask]
    progress_reached_one = bool(
        operation_progress.size
        and np.max(operation_progress) >= 1.0 - 1e-12
        and operation_progress[-1] >= 1.0 - 1e-12
    )
    progress_monotonic = bool(
        not operation_progress.size
        or np.all(np.diff(operation_progress) >= -1e-14)
    )

    metrics = {
        "contact_preserving_planned_lift": {
            "top_level_identity": {
                "state": identity_audit["state"],
                "planner_verification": identity_audit[
                    "planner_verification"
                ],
                "controller_format": identity_audit["controller_format"],
                "expected": expected_identity,
                "persisted": persisted_identity,
            },
            "operation_sample_count": int(operation_effective.shape[0]),
            "target_face_effective_duty": {
                finger: float(per_finger_duty[index])
                for index, finger in enumerate(ACTIVE_FINGERS)
            },
            "simultaneous_target_face_effective_duty": simultaneous_duty,
            "longest_contact_loss_steps": {
                finger: int(longest_loss_steps[index])
                for index, finger in enumerate(ACTIVE_FINGERS)
            },
            "longest_contact_loss_s": {
                finger: float(longest_loss_steps[index] * timestep_s)
                for index, finger in enumerate(ACTIVE_FINGERS)
            },
            "simultaneous_longest_contact_loss_steps": int(
                simultaneous_longest_loss_steps
            ),
            "simultaneous_longest_contact_loss_s": float(
                simultaneous_longest_loss_steps * timestep_s
            ),
            "allowed_contact_loss_steps": allowed_loss_steps,
            "allowed_contact_loss_s": max_loss_s,
            "required_contact_duty": duty_min,
            "operation_aborted": operation_abort,
            "maximum_plan_progress": (
                float(np.max(operation_progress))
                if operation_progress.size
                else 0.0
            ),
            "final_plan_progress": (
                float(operation_progress[-1])
                if operation_progress.size
                else 0.0
            ),
        }
    }
    if slip_feedback_enabled:
        metrics["contact_preserving_planned_lift"]["online_tangent_slip"] = {
            "freeze_threshold_m": float(
                feedback_config["tangent_slip_freeze_threshold_m"]
            ),
            "abort_threshold_m": float(
                feedback_config["tangent_slip_abort_threshold_m"]
            ),
            "maximum_m": {
                finger: (
                    float(np.max(
                        np.asarray(
                            traces["online_contact_tangent_slip_m"],
                            dtype=np.float64,
                        )[operation_mask, index],
                        initial=0.0,
                    ))
                )
                for index, finger in enumerate(ACTIVE_FINGERS)
            },
        }
    checks = {
        "v14_top_level_identity_trace_matches_recomputed_config": (
            identity_trace_matches
        ),
        "v14_target_face_effective_matches_raw_trace": bool(
            np.array_equal(
                np.asarray(traces["target_face_effective"], dtype=bool),
                native_target_effective,
            )
            and v16_native_trace_matches
            and v16_rolling_trace_matches
        ),
        "v14_contact_loss_trace_matches_raw_contacts": loss_trace_matches,
        "v14_feedback_uses_previous_observation": causal_source_matches,
        "v14_freeze_and_recovery_match_previous_contact_risk": (
            causal_risk_matches
        ),
        "v14_static_plan_trace_matches_config": static_plan_matches,
        "v14_dynamic_plan_trace_matches_progress": dynamic_plan_matches,
        "v14_command_composition_matches_plan_and_feedback": (
            command_composition_matches
        ),
        "v14_force_feedback_trace_is_bounded_and_recomputable": (
            force_trace_matches
        ),
        "v14_operation_did_not_abort": not operation_abort,
        "v14_plan_progress_is_monotonic": progress_monotonic,
        "v14_plan_progress_reached_one": progress_reached_one,
        "v14_thumb_contact_duty_at_least_99_percent": bool(
            per_finger_duty[0] + 1e-12 >= duty_min
        ),
        "v14_index_contact_duty_at_least_99_percent": bool(
            per_finger_duty[1] + 1e-12 >= duty_min
        ),
        "v14_middle_contact_duty_at_least_99_percent": bool(
            per_finger_duty[2] + 1e-12 >= duty_min
        ),
        "v14_simultaneous_contact_duty_at_least_99_percent": bool(
            simultaneous_duty + 1e-12 >= duty_min
        ),
        "v14_thumb_contact_loss_within_limit": bool(
            longest_loss_steps[0] <= allowed_loss_steps
        ),
        "v14_index_contact_loss_within_limit": bool(
            longest_loss_steps[1] <= allowed_loss_steps
        ),
        "v14_middle_contact_loss_within_limit": bool(
            longest_loss_steps[2] <= allowed_loss_steps
        ),
        "v14_simultaneous_contact_loss_within_limit": bool(
            simultaneous_longest_loss_steps <= allowed_loss_steps
        ),
    }
    if schema_version >= 16:
        metrics["contact_preserving_planned_lift"].update(
            {
                "operation_contact_evidence": "rolling_aware_physical_pad",
                "native_tactile_target_face_effective_duty": {
                    finger: float(
                        np.mean(native_target_effective[operation_mask, index])
                    )
                    if np.any(operation_mask)
                    else 0.0
                    for index, finger in enumerate(ACTIVE_FINGERS)
                },
            }
        )
        checks.update(
            {
                "v16_native_tactile_target_face_effective_matches_raw_trace": (
                    v16_native_trace_matches
                ),
                "v16_rolling_aware_target_face_effective_matches_raw_trace": (
                    v16_rolling_trace_matches
                ),
            }
        )
    if slip_feedback_enabled:
        checks.update(
            {
                "v14_online_tangent_slip_matches_raw_contacts": (
                    slip_trace_matches
                ),
                "v14_tangent_slip_freeze_uses_previous_observation": (
                    slip_risk_trace_matches
                ),
                "v14_tangent_slip_abort_matches_current_safety_observation": (
                    slip_abort_trace_matches
                ),
            }
        )
    return metrics, checks


def _v15_joint_pair_alignment_metrics(
    model: mujoco.MjModel,
    info: ModelInfo,
    config: dict[str, Any],
    traces: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, bool]]:
    """Recompute the directed pair geometry and causal v15 feedback audit."""

    total = int(np.asarray(traces["time"]).shape[0])
    timestep = float(model.opt.timestep)
    alignment = config["joint_pair_alignment"]
    feedback = config["joint_pair_feedback"]
    binding = resolve_joint_pair(model, alignment["joint_names"])
    if binding is None:
        raise ValueError("schema-v15 joint pair is missing")

    vector = np.asarray(traces["joint_pair_vector_cube_m"], dtype=np.float64)
    residual = np.asarray(traces["joint_pair_residual"], dtype=np.float64)
    angle = np.asarray(traces["joint_pair_angle_deg"], dtype=np.float64)
    length = np.asarray(traces["joint_pair_length_m"], dtype=np.float64)
    positive = np.asarray(traces["joint_pair_positive_y"], dtype=bool)
    valid = np.asarray(traces["joint_pair_valid"], dtype=bool)
    if (
        vector.shape != (total, 3)
        or residual.shape != (total, 2)
        or any(value.shape != (total,) for value in (angle, length, positive, valid))
    ):
        raise ValueError("schema-v15 joint-pair geometry traces have invalid shape")

    # Rebuild kinematics from authoritative qpos/cube pose samples.  This is
    # intentionally independent of the online observation cache.
    audit_data = mujoco.MjData(model)
    joint_qpos = np.asarray(traces["joint_qpos"], dtype=np.float64)
    joint_qvel = np.asarray(traces["joint_qvel"], dtype=np.float64)
    cube_pos = np.asarray(traces["cube_pos"], dtype=np.float64)
    cube_quat = np.asarray(traces["cube_quat"], dtype=np.float64)
    cube_velocity = np.asarray(traces["cube_velocity"], dtype=np.float64)
    controls = np.asarray(traces["ctrl"], dtype=np.float64)
    expected_vector = np.zeros_like(vector)
    expected_residual = np.zeros_like(residual)
    expected_angle = np.full(total, 180.0, dtype=np.float64)
    expected_length = np.zeros(total, dtype=np.float64)
    expected_positive = np.zeros(total, dtype=bool)
    expected_valid = np.zeros(total, dtype=bool)
    expected_self_collision = np.zeros(total, dtype=bool)
    expected_self_collision_count = np.zeros(total, dtype=np.int64)
    expected_self_collision_pairs = np.full(total, "", dtype="<U512")
    contact_force = np.zeros(6, dtype=np.float64)
    for step in range(total):
        mujoco.mj_resetData(model, audit_data)
        audit_data.qpos[info.actuator_qpos_adrs] = joint_qpos[step]
        audit_data.qvel[info.actuator_dof_adrs] = joint_qvel[step]
        cube_adr = int(info.cube_qpos_adr)
        audit_data.qpos[cube_adr : cube_adr + 3] = cube_pos[step]
        audit_data.qpos[cube_adr + 3 : cube_adr + 7] = cube_quat[step]
        cube_dof = int(info.cube_dof_adr)
        audit_data.qvel[cube_dof : cube_dof + 6] = cube_velocity[step]
        audit_data.ctrl[:] = controls[step]
        mujoco.mj_forward(model, audit_data)
        collision_pairs: list[str] = []
        for contact_index, contact in enumerate(
            audit_data.contact[: audit_data.ncon]
        ):
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            finger1 = info.hand_body_parts.get(int(model.geom_bodyid[geom1]))
            finger2 = info.hand_body_parts.get(int(model.geom_bodyid[geom2]))
            if (
                finger1 not in ACTIVE_FINGERS
                or finger2 not in ACTIVE_FINGERS
                or finger1 == finger2
            ):
                continue
            normal_force = 0.0
            if int(contact.efc_address) >= 0:
                contact_force[:] = 0.0
                mujoco.mj_contactForce(
                    model, audit_data, contact_index, contact_force
                )
                normal_force = max(0.0, float(contact_force[0]))
            penetration = max(0.0, -float(contact.dist))
            if normal_force <= 1e-8 and penetration <= 0.0:
                continue
            name1 = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, geom1
            ) or f"geom_{geom1}"
            name2 = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, geom2
            ) or f"geom_{geom2}"
            first, second = sorted((str(name1), str(name2)))
            collision_pairs.append(f"{first}|{second}")
        collision_pairs.sort()
        expected_self_collision[step] = bool(collision_pairs)
        expected_self_collision_count[step] = len(collision_pairs)
        expected_self_collision_pairs[step] = ";".join(collision_pairs)
        try:
            measured = joint_pair_telemetry(audit_data, binding)
        except ValueError:
            continue
        current_vector = np.asarray(measured["vector_cube_m"], dtype=np.float64)
        current_length = float(measured["length_m"])
        current_y = float(current_vector[1])
        expected_vector[step] = current_vector
        expected_length[step] = current_length
        if abs(current_y) <= 1e-12:
            continue
        expected_residual[step] = (
            current_vector[0] / current_y,
            current_vector[2] / current_y,
        )
        expected_angle[step] = float(
            np.degrees(
                np.arccos(np.clip(current_y / current_length, -1.0, 1.0))
            )
        )
        expected_positive[step] = current_y > 0.0
        expected_valid[step] = True
    geometry_matches = bool(
        np.allclose(vector, expected_vector, rtol=0.0, atol=2e-12)
        and np.allclose(residual, expected_residual, rtol=0.0, atol=2e-12)
        and np.allclose(angle, expected_angle, rtol=0.0, atol=2e-10)
        and np.allclose(length, expected_length, rtol=0.0, atol=2e-12)
        and np.array_equal(positive, expected_positive)
        and np.array_equal(valid, expected_valid)
    )

    persisted_self_collision = np.asarray(
        traces["active_finger_self_collision"], dtype=bool
    )
    persisted_self_collision_count = np.asarray(
        traces["active_finger_self_collision_contact_count"], dtype=np.int64
    )
    persisted_self_collision_force = np.asarray(
        traces["active_finger_self_collision_normal_force_n"], dtype=np.float64
    )
    persisted_self_collision_penetration = np.asarray(
        traces["active_finger_self_collision_max_penetration_m"], dtype=np.float64
    )
    persisted_self_collision_pairs = np.asarray(
        traces["active_finger_self_collision_pairs"]
    ).astype(str)
    self_collision_trace_matches = bool(
        all(
            value.shape == (total,)
            for value in (
                persisted_self_collision,
                persisted_self_collision_count,
                persisted_self_collision_force,
                persisted_self_collision_penetration,
                persisted_self_collision_pairs,
            )
        )
        and np.array_equal(persisted_self_collision, expected_self_collision)
        and np.array_equal(
            persisted_self_collision_count, expected_self_collision_count
        )
        and np.array_equal(
            persisted_self_collision_pairs, expected_self_collision_pairs
        )
        and np.isfinite(persisted_self_collision_force).all()
        and np.isfinite(persisted_self_collision_penetration).all()
        and np.all(persisted_self_collision_force >= 0.0)
        and np.all(persisted_self_collision_penetration >= 0.0)
        and np.array_equal(
            persisted_self_collision,
            persisted_self_collision_count > 0,
        )
    )

    minimum_length = float(alignment["minimum_length_m"])
    grasp_max = float(alignment["grasp_max_deg"])
    expected_safe = (
        expected_valid
        & expected_positive
        & (expected_length >= minimum_length - 1e-12)
        & (expected_angle <= grasp_max + 1e-12)
    )
    persisted_safe = np.asarray(
        traces["joint_pair_alignment_safe"], dtype=bool
    )
    gate_order = tuple(
        str(value) for value in np.asarray(traces["grasp_gate_order"])
    )
    pair_gate_index = gate_order.index("joint_pair_alignment_safe")
    self_collision_gate_index = gate_order.index(
        "no_active_finger_self_collision"
    )
    gate = np.asarray(traces["grasp_gate"], dtype=bool)
    grasp_gate_matches = bool(
        np.array_equal(persisted_safe, expected_safe)
        and np.array_equal(gate[:, pair_gate_index], expected_safe)
        and np.array_equal(
            gate[:, self_collision_gate_index], ~persisted_self_collision
        )
    )

    acquisition = _trace_scalar_int(traces, "grasp_acquisition_step")
    stable_steps = int(
        round(float(config["control_protocol"]["stable_window_s"]) / timestep)
    )
    grasp_angles = np.zeros(0, dtype=np.float64)
    if acquisition >= 0:
        start = acquisition - stable_steps + 1
        if start < 0:
            raise ValueError("schema-v15 grasp has an incomplete alignment window")
        grasp_angles = expected_angle[start : acquisition + 1]
    grasp_p95 = (
        float(np.percentile(grasp_angles, 95.0)) if grasp_angles.size else 180.0
    )
    grasp_maximum = float(np.max(grasp_angles)) if grasp_angles.size else 180.0
    persisted_grasp_p95 = float(
        np.asarray(traces["joint_pair_grasp_p95_deg"]).reshape(())
    )
    persisted_grasp_max = float(
        np.asarray(traces["joint_pair_grasp_max_deg"]).reshape(())
    )

    states = np.asarray(traces["control_state"]).astype(str)
    operation_mask = (states == "MANIPULATE") | (states == "HOLD")
    operation_angles = expected_angle[operation_mask]
    operation_safe = expected_safe[operation_mask]
    operation_geometry_valid = (
        expected_valid[operation_mask]
        & expected_positive[operation_mask]
        & (expected_length[operation_mask] >= minimum_length - 1e-12)
    )
    within_p95_limit = operation_safe & (
        operation_angles <= float(alignment["operation_p95_max_deg"]) + 1e-12
    )
    operation_p95 = (
        float(np.percentile(operation_angles, 95.0))
        if operation_angles.size
        else 180.0
    )
    operation_maximum = (
        float(np.max(operation_angles)) if operation_angles.size else 180.0
    )
    operation_duty = (
        float(np.mean(within_p95_limit)) if within_p95_limit.size else 0.0
    )
    violation = ~expected_safe | (
        expected_angle > float(feedback["freeze_threshold_deg"]) + 1e-12
    )
    current_run = 0
    recomputed_run = np.zeros(total, dtype=np.int64)
    for step in range(total):
        if operation_mask[step]:
            current_run = current_run + 1 if violation[step] else 0
        recomputed_run[step] = current_run
    persisted_run = np.asarray(
        traces["joint_pair_violation_run_steps"], dtype=np.int64
    )
    maximum_run = int(np.max(recomputed_run, initial=0))
    persisted_maximum_run = int(
        np.asarray(traces["joint_pair_maximum_violation_run_steps"]).reshape(())
    )
    run_trace_matches = bool(
        np.array_equal(persisted_run, recomputed_run)
        and persisted_maximum_run == maximum_run
    )

    source = np.asarray(traces["joint_pair_feedback_source_step"], dtype=np.int64)
    source_matches = bool(
        np.array_equal(source, np.arange(total, dtype=np.int64) - 1)
    )
    expected_freeze_risk = np.zeros(total, dtype=bool)
    expected_slip_recovery = np.zeros((total, len(ACTIVE_FINGERS)), dtype=bool)
    online_slip = np.asarray(
        traces["online_contact_tangent_slip_m"], dtype=np.float64
    )
    online_slip_valid = np.asarray(
        traces["online_contact_tangent_slip_valid"], dtype=bool
    )
    for step in np.flatnonzero(operation_mask):
        previous = step - 1
        if previous >= 0:
            expected_freeze_risk[step] = violation[previous]
            expected_slip_recovery[step] = online_slip_valid[previous] & (
                online_slip[previous]
                >= float(feedback["slip_freeze_threshold_m"]) - 1e-12
            )
    causal_risk_matches = bool(
        np.array_equal(
            np.asarray(traces["joint_pair_freeze_risk"], dtype=bool),
            expected_freeze_risk,
        )
        and np.array_equal(
            np.asarray(
                traces["joint_pair_slip_recovery_active"], dtype=bool
            ),
            expected_slip_recovery,
        )
    )

    plan = config["manipulation_plan"]
    config_pair_jacobian = np.asarray(
        plan["joint_pair_residual_jacobian_2x8"], dtype=np.float64
    )
    config_object_jacobian = np.asarray(
        plan["object_response_jacobian_6x8"], dtype=np.float64
    )
    config_force_jacobian = np.asarray(
        plan["target_force_jacobian_3x8"], dtype=np.float64
    )
    static_jacobians_match = bool(
        np.array_equal(
            np.asarray(
                traces["joint_pair_plan_residual_jacobian_2x8"],
                dtype=np.float64,
            ),
            config_pair_jacobian,
        )
        and np.array_equal(
            np.asarray(
                traces["joint_pair_plan_object_jacobian_6x8"],
                dtype=np.float64,
            ),
            config_object_jacobian,
        )
        and np.array_equal(
            np.asarray(
                traces["joint_pair_plan_force_jacobian_3x8"],
                dtype=np.float64,
            ),
            config_force_jacobian,
        )
    )
    knot_index = np.asarray(traces["planned_knot_index"], dtype=np.int64)
    expected_active_jacobian = np.zeros(
        (total, 2, len(ACTIVE_ACTUATORS)), dtype=np.float64
    )
    expected_alignment_request = np.zeros(
        (total, len(ACTIVE_ACTUATORS)), dtype=np.float64
    )
    expected_slip_request = np.zeros_like(expected_alignment_request)
    preload = np.zeros(model.nu, dtype=np.float64)
    precontact = np.zeros(model.nu, dtype=np.float64)
    for actuator, value in contact_preload_targets(config).items():
        preload[model.actuator(actuator).id] = float(value)
    for actuator, value in precontact_targets(config).items():
        precontact[model.actuator(actuator).id] = float(value)
    active_ids = np.asarray(
        [model.actuator(name).id for name in ACTIVE_ACTUATORS], dtype=np.int64
    )
    inward_active = np.zeros((3, len(ACTIVE_ACTUATORS)), dtype=np.float64)
    groups = (ACTIVE_ACTUATORS[:3], ACTIVE_ACTUATORS[3:6], ACTIVE_ACTUATORS[6:])
    for finger, names in enumerate(groups):
        local = np.asarray([model.actuator(name).id for name in names], dtype=int)
        direction = preload[local] - precontact[local]
        direction /= float(np.max(np.abs(direction)))
        active_columns = [ACTIVE_ACTUATORS.index(name) for name in names]
        inward_active[finger, active_columns] = direction
    damping = float(feedback.get("damping", 1e-8))
    for step in np.flatnonzero(operation_mask):
        previous = step - 1
        knot = int(np.clip(knot_index[step], 0, config_pair_jacobian.shape[0] - 1))
        if (
            previous >= 0
            and expected_valid[previous]
            and expected_positive[previous]
            and expected_length[previous] >= minimum_length
        ):
            pair_jacobian = config_pair_jacobian[knot]
            expected_active_jacobian[step] = pair_jacobian
            protected = np.vstack(
                (
                    float(feedback.get("vertical_response_weight", 1.0))
                    * config_object_jacobian[knot, 2:3],
                    float(feedback.get("force_response_weight", 1.0))
                    * config_force_jacobian[knot],
                )
            )
            projector = np.eye(len(ACTIVE_ACTUATORS)) - protected.T @ np.linalg.solve(
                protected @ protected.T + damping * np.eye(protected.shape[0]),
                protected,
            )
            effective = pair_jacobian @ projector
            expected_alignment_request[step] = -float(
                feedback["alignment_gain"]
            ) * (
                projector
                @ effective.T
                @ np.linalg.solve(
                    effective @ effective.T + damping * np.eye(2),
                    expected_residual[previous],
                )
            )
        if previous >= 0:
            scalars = np.where(
                expected_slip_recovery[step],
                float(feedback["slip_recovery_gain_rad_per_m"])
                * online_slip[previous],
                0.0,
            )
            expected_slip_request[step] = np.sum(
                scalars[:, None] * inward_active, axis=0
            )
    persisted_active_jacobian = np.asarray(
        traces["joint_pair_active_residual_jacobian_2x8"], dtype=np.float64
    )
    persisted_alignment_request = np.asarray(
        traces["joint_pair_alignment_request_rad"], dtype=np.float64
    )[:, active_ids]
    persisted_slip_request = np.asarray(
        traces["joint_pair_slip_recovery_correction_rad"], dtype=np.float64
    )[:, active_ids]
    request_matches = bool(
        np.allclose(
            persisted_active_jacobian,
            expected_active_jacobian,
            rtol=0.0,
            atol=2e-12,
        )
        and np.allclose(
            persisted_alignment_request,
            expected_alignment_request,
            rtol=0.0,
            atol=2e-12,
        )
        and np.allclose(
            persisted_slip_request,
            expected_slip_request,
            rtol=0.0,
            atol=2e-12,
        )
    )

    correction = np.asarray(
        traces["joint_pair_feedback_correction_rad"], dtype=np.float64
    )
    velocity = np.asarray(
        traces["joint_pair_feedback_velocity_rad_s"], dtype=np.float64
    )
    inactive = info.inactive_actuator_ids
    bounded = bool(
        correction.shape == (total, model.nu)
        and velocity.shape == (total, model.nu)
        and np.all(
            np.abs(correction)
            <= float(feedback["correction_limit_rad"]) + 1e-12
        )
        and np.all(
            np.abs(velocity[operation_mask])
            <= float(feedback["rate_limit_rad_s"]) + 1e-12
        )
        and np.all(correction[:, inactive] == 0.0)
        and np.all(velocity[:, inactive] == 0.0)
    )
    operation_velocity = velocity[operation_mask]
    if operation_velocity.shape[0] > 1:
        bounded = bool(
            bounded
            and np.all(
                np.abs(np.diff(operation_velocity, axis=0))
                <= float(feedback["acceleration_limit_rad_s2"]) * timestep
                + 1e-12
            )
        )

    allowed_run_steps = int(
        round(float(alignment["max_continuous_violation_s"]) / timestep)
    )
    collision_pair_counts: dict[str, int] = {}
    for encoded in persisted_self_collision_pairs:
        for pair in filter(None, str(encoded).split(";")):
            collision_pair_counts[pair] = collision_pair_counts.get(pair, 0) + 1
    collision_longest_steps = _longest_true_run_steps(
        persisted_self_collision
    )
    grasp_state_mask = np.isin(states, ("SETTLE", "CLOSE", "VERIFY"))
    operation_state_mask = np.isin(
        states, ("MANIPULATE", "HOLD", "ABORT")
    )
    metrics = {
        "joint_pair_alignment": {
            "joint_names": list(alignment["joint_names"]),
            "grasp_window_p95_deg": grasp_p95,
            "grasp_window_max_deg": grasp_maximum,
            "operation_p95_deg": operation_p95,
            "operation_max_deg": operation_maximum,
            "operation_within_limit_duty": operation_duty,
            "operation_longest_violation_steps": maximum_run,
            "operation_longest_violation_s": maximum_run * timestep,
            "maximum_length_m": float(np.max(expected_length, initial=0.0)),
            "minimum_length_m": (
                float(np.min(expected_length)) if expected_length.size else 0.0
            ),
            "abort_reason": str(
                np.asarray(traces["joint_pair_abort_reason"]).reshape(())
            ),
        },
        "active_finger_self_collision": {
            "collision_free": not bool(np.any(persisted_self_collision)),
            "collision_frame_count": int(
                np.count_nonzero(persisted_self_collision)
            ),
            "collision_duty": float(np.mean(persisted_self_collision)),
            "first_collision_step": (
                int(np.flatnonzero(persisted_self_collision)[0])
                if np.any(persisted_self_collision)
                else -1
            ),
            "longest_consecutive_collision_steps": collision_longest_steps,
            "longest_consecutive_collision_s": (
                collision_longest_steps * timestep
            ),
            "maximum_total_normal_force_n": float(
                np.max(persisted_self_collision_force, initial=0.0)
            ),
            "maximum_penetration_m": float(
                np.max(persisted_self_collision_penetration, initial=0.0)
            ),
            "pair_frame_counts": dict(sorted(collision_pair_counts.items())),
        },
    }
    checks = {
        "v15_joint_pair_trace_matches_raw_state": geometry_matches,
        "v15_joint_pair_grasp_gate_matches_raw_state": grasp_gate_matches,
        "v15_active_finger_self_collision_trace_matches_raw_state": (
            self_collision_trace_matches
        ),
        "grasp_no_active_finger_self_collision": not bool(
            np.any(persisted_self_collision[grasp_state_mask])
        ),
        "operation_no_active_finger_self_collision": not bool(
            np.any(persisted_self_collision[operation_state_mask])
        ),
        "no_active_finger_self_collision": not bool(
            np.any(persisted_self_collision)
        ),
        "v15_joint_pair_grasp_window_p95_within_limit": bool(
            grasp_angles.size
            and grasp_p95 <= float(alignment["grasp_p95_max_deg"]) + 1e-12
        ),
        "v15_joint_pair_grasp_window_max_within_limit": bool(
            grasp_angles.size and grasp_maximum <= grasp_max + 1e-12
        ),
        "v15_joint_pair_direction_and_length_valid_at_grasp": bool(
            grasp_angles.size
            and np.all(expected_positive[acquisition - stable_steps + 1 : acquisition + 1])
            and np.all(expected_length[acquisition - stable_steps + 1 : acquisition + 1] >= minimum_length - 1e-12)
        ),
        "v15_joint_pair_grasp_summary_matches_trace": bool(
            math.isclose(persisted_grasp_p95, grasp_p95, abs_tol=2e-10)
            and math.isclose(persisted_grasp_max, grasp_maximum, abs_tol=2e-10)
        ),
        "v15_joint_pair_feedback_uses_previous_observation": source_matches,
        "v15_joint_pair_freeze_uses_previous_observation": causal_risk_matches,
        "v15_joint_pair_violation_trace_matches_raw_state": run_trace_matches,
        "v15_joint_pair_static_jacobians_match_config": static_jacobians_match,
        "v15_joint_pair_requests_use_previous_observation": request_matches,
        "v15_joint_pair_feedback_is_bounded": bounded,
        "v15_joint_pair_operation_p95_within_limit": bool(
            operation_angles.size
            and operation_p95
            <= float(alignment["operation_p95_max_deg"]) + 1e-12
        ),
        "v15_joint_pair_operation_max_within_limit": bool(
            operation_angles.size
            and operation_maximum
            <= float(alignment["operation_max_deg"]) + 1e-12
        ),
        "v15_joint_pair_operation_duty_at_least_99_percent": bool(
            operation_duty + 1e-12
            >= float(alignment["operation_within_p95_limit_duty_min"])
        ),
        "v15_joint_pair_continuous_violation_within_limit": bool(
            maximum_run <= allowed_run_steps
        ),
        "v15_joint_pair_direction_and_length_preserved": bool(
            operation_angles.size and np.all(operation_geometry_valid)
        ),
        "v15_joint_pair_no_alignment_abort_risk": not bool(
            np.any(np.asarray(traces["joint_pair_abort_risk"], dtype=bool))
        ),
    }
    return metrics, checks


def _stage_manipulation_label(
    *,
    schema_version: int,
    manipulation_success: bool,
    legacy_operation_executed: bool,
    manipulation_start_step: int,
    operation_sample_count: int,
) -> str:
    """Return the public stage label without changing legacy hard checks.

    Through schema v14, ``checks.operation_executed`` means that the complete
    controller event sequence was valid.  A safety abort therefore makes that
    check false even after real MANIPULATE commands were issued.  Schema v15
    reports the observed attempt separately for the human-readable label while
    retaining the historical check and every success/failure verdict.
    """

    if manipulation_success:
        return "passed"
    attempted = bool(legacy_operation_executed)
    if int(schema_version) >= 15:
        attempted = bool(
            int(manipulation_start_step) >= 0
            and int(operation_sample_count) > 0
        )
    return "failed" if attempted else "not_run"


def evaluate_trace(
    model: mujoco.MjModel,
    info: ModelInfo,
    config: dict[str, Any],
    phase_steps: dict[str, int],
    traces: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Evaluate a complete trace; schema v1 keeps its historical semantics."""

    inclusive_tolerance = 1e-12
    acceptance = config["acceptance"]
    schema_version = int(config.get("schema_version", 1))
    settle_steps = phase_steps["settle"]
    total_steps = traces["time"].shape[0]
    hold_start = total_steps - phase_steps["hold"]
    if schema_version >= 3:
        states = np.asarray(traces["control_state"]).astype(str)
        if states.shape != (total_steps,):
            raise ValueError(f"control_state must have shape ({total_steps},)")
        hold_indices = np.flatnonzero(states == "HOLD")
        if hold_indices.size:
            hold_start = int(hold_indices[0])
    hold_steps = total_steps - hold_start
    hold_slice = slice(hold_start, total_steps)
    baseline_count = min(settle_steps, max(1, int(round(0.1 / model.opt.timestep))))
    baseline_slice = slice(settle_steps - baseline_count, settle_steps)
    height_count = max(
        1, int(round(float(acceptance["height_window_s"]) / model.opt.timestep))
    )
    height_slice = slice(total_steps - min(height_count, hold_steps), total_steps)

    baseline_z = float(np.median(traces["cube_pos"][baseline_slice, 2]))
    height_gain = traces["cube_pos"][:, 2] - baseline_z
    height_window = height_gain[height_slice]
    hold_heights = traces["cube_pos"][hold_slice, 2]
    contact_force = traces["finger_contact_force"][hold_slice]
    tactile = traces["tactile_max"][hold_slice, :3]
    effective = (contact_force >= float(acceptance["contact_force_min_n"])) & (
        tactile >= float(acceptance["touch_force_min_n"])
    )
    final_physical_pad_evidence: np.ndarray | None = None
    if schema_version >= 16:
        final_physical_pad_evidence = (
            _v16_rolling_aware_target_face_evidence(config, traces)[
                "effective"
            ][hold_slice]
        )
        effective = final_physical_pad_evidence
    contact_duty = effective.mean(axis=0)
    simultaneous_duty = float(np.all(effective, axis=1).mean())

    hold_quaternions = traces["cube_quat"][hold_slice]
    orientation_drift_deg = float(
        np.degrees(orientation_angles(hold_quaternions[0], hold_quaternions)).max()
    )
    root_positions = np.asarray(traces["root_pos"], dtype=np.float64)
    root_quaternions = np.asarray(traces["root_quat"], dtype=np.float64)
    root_position_unchanged = bool(
        np.array_equal(root_positions, np.broadcast_to(root_positions[0], root_positions.shape))
    )
    root_orientation_unchanged = bool(
        np.array_equal(
            root_quaternions,
            np.broadcast_to(root_quaternions[0], root_quaternions.shape),
        )
    )
    root_position_drift = (
        0.0
        if root_position_unchanged
        else float(
            np.linalg.norm(root_positions - root_positions[0], axis=1).max()
        )
    )
    # ``acos(dot(q, q))`` can report roughly 3e-8 rad for bit-identical,
    # non-axis-aligned unit quaternions because normalization rounds the dot
    # just below one.  A fixed MuJoCo body has an exactly repeated trace, so
    # preserve that stronger evidence instead of turning round-off into a
    # spurious root-motion failure.
    root_orientation_drift = (
        0.0
        if root_orientation_unchanged
        else float(orientation_angles(root_quaternions[0], root_quaternions).max())
    )

    inactive_ctrl = traces["ctrl"][:, info.inactive_actuator_ids]
    inactive_qpos = traces["joint_qpos"][:, info.inactive_actuator_ids]
    limited_qpos = traces["joint_qpos"][:, info.joint_limited]
    limited_ranges = info.joint_ranges[info.joint_limited]
    joint_limits_ok = bool(
        np.all(limited_qpos >= limited_ranges[:, 0] - 2e-3)
        and np.all(limited_qpos <= limited_ranges[:, 1] + 2e-3)
    )
    active_forces = np.abs(traces["actuator_force"][:, info.active_actuator_ids])
    active_limits = info.force_limits[info.active_actuator_ids]
    saturation_fraction = float(np.mean(active_forces >= 0.98 * active_limits))

    metrics: dict[str, Any] = {
        "baseline_cube_z_m": baseline_z,
        "median_lift_m": float(np.median(height_window)),
        "minimum_lift_m": float(np.min(height_window)),
        "peak_lift_m": float(np.max(height_gain)),
        "hold_height_span_m": float(np.ptp(hold_heights)),
        "orientation_drift_deg": orientation_drift_deg,
        "end_linear_speed_m_s": float(np.linalg.norm(traces["cube_velocity"][-1, :3])),
        "contact_duty": {
            finger: float(contact_duty[index])
            for index, finger in enumerate(ACTIVE_FINGERS)
        },
        "simultaneous_contact_duty": simultaneous_duty,
        "peak_tactile_n": {
            finger: float(np.max(tactile[:, index]))
            for index, finger in enumerate(ACTIVE_FINGERS)
        },
        "peak_distal_contact_force_n": {
            finger: float(np.max(contact_force[:, index]))
            for index, finger in enumerate(ACTIVE_FINGERS)
        },
        "peak_total_distal_contact_force_n": float(
            np.max(np.sum(contact_force, axis=1))
        ),
        "forbidden_contact_steps": int(
            np.count_nonzero(traces["forbidden_contact"])
        ),
        "support_contact_steps_final_window": int(
            np.count_nonzero(traces["support_contact"][height_slice])
        ),
        "floor_contact_steps_final_window": int(
            np.count_nonzero(traces["floor_contact"][height_slice])
        ),
        "root_position_drift_m": root_position_drift,
        "root_orientation_drift_rad": root_orientation_drift,
        "inactive_ctrl_max_abs": float(np.max(np.abs(inactive_ctrl), initial=0.0)),
        "inactive_joint_max_abs_rad": float(
            np.max(np.abs(inactive_qpos), initial=0.0)
        ),
        "max_penetration_m": float(np.max(traces["max_penetration"])),
        "max_runtime_friction_error": float(np.max(traces["friction_error"])),
        "actuator_saturation_fraction": saturation_fraction,
        "requested_sliding_friction": info.requested_friction,
    }
    if final_physical_pad_evidence is not None:
        metrics["final_physical_pad_contact_duty"] = {
            finger: float(np.mean(final_physical_pad_evidence[:, index]))
            for index, finger in enumerate(ACTIVE_FINGERS)
        }

    checks: dict[str, bool] = {
        "hand_root_is_structurally_fixed": bool(
            model.body_jntnum[info.root_body_id] == 0
            and model.body_parentid[info.root_body_id] == 0
            and model.body_mocapid[info.root_body_id] == -1
        ),
        "hand_root_pose_did_not_move": root_position_unchanged
        and root_orientation_unchanged,
        "inactive_controls_are_exactly_zero": bool(np.all(inactive_ctrl == 0.0)),
        "inactive_joints_remain_open": metrics["inactive_joint_max_abs_rad"]
        <= float(acceptance["inactive_joint_abs_max_rad"]) + inclusive_tolerance,
        "all_state_is_finite": bool(np.all(traces["finite"])),
        "joint_limits_respected": joint_limits_ok,
        "penetration_within_limit": metrics["max_penetration_m"]
        <= float(acceptance["max_penetration_m"]) + inclusive_tolerance,
        "runtime_contact_friction_matches_cube": bool(
            np.any(traces["cube_contact_seen"])
        )
        and metrics["max_runtime_friction_error"] <= 1e-9
        and bool(np.all(traces["contact_dim_ok"][traces["cube_contact_seen"]])),
        "median_lift_reached": metrics["median_lift_m"] + inclusive_tolerance
        >= float(acceptance["median_lift_m"]),
        "minimum_lift_reached": metrics["minimum_lift_m"] + inclusive_tolerance
        >= float(acceptance["minimum_lift_m"]),
        "cube_cleared_support_and_floor": metrics[
            "support_contact_steps_final_window"
        ]
        == 0
        and metrics["floor_contact_steps_final_window"] == 0,
        "hold_height_is_stable": metrics["hold_height_span_m"]
        <= float(acceptance["max_height_span_m"]) + inclusive_tolerance,
        "hold_orientation_is_stable": orientation_drift_deg
        <= float(acceptance["max_orientation_drift_deg"]) + inclusive_tolerance,
        "end_linear_speed_is_low": metrics["end_linear_speed_m_s"]
        < float(acceptance["max_end_linear_speed_m_s"]),
        "thumb_contact_duty": contact_duty[0] + inclusive_tolerance
        >= float(acceptance["finger_contact_duty"]),
        "index_contact_duty": contact_duty[1] + inclusive_tolerance
        >= float(acceptance["finger_contact_duty"]),
        "middle_contact_duty": contact_duty[2] + inclusive_tolerance
        >= float(acceptance["finger_contact_duty"]),
        "simultaneous_three_finger_contact": simultaneous_duty + inclusive_tolerance
        >= float(acceptance["simultaneous_contact_duty"]),
        "thumb_tactile_nonzero": (
            bool(np.any(final_physical_pad_evidence[:, 0]))
            if final_physical_pad_evidence is not None
            else metrics["peak_tactile_n"]["thumb"]
            >= float(acceptance["touch_force_min_n"])
        ),
        "index_tactile_nonzero": (
            bool(np.any(final_physical_pad_evidence[:, 1]))
            if final_physical_pad_evidence is not None
            else metrics["peak_tactile_n"]["index"]
            >= float(acceptance["touch_force_min_n"])
        ),
        "middle_tactile_nonzero": (
            bool(np.any(final_physical_pad_evidence[:, 2]))
            if final_physical_pad_evidence is not None
            else metrics["peak_tactile_n"]["mid"]
            >= float(acceptance["touch_force_min_n"])
        ),
        "no_palm_ring_or_pinky_contact": metrics["forbidden_contact_steps"] == 0,
    }

    if schema_version >= 2:
        material_start_override = None
        if schema_version >= 5:
            manipulation_start = _trace_scalar_int(
                traces, "manipulation_start_step"
            )
            material_start_override = (
                manipulation_start if manipulation_start >= 0 else total_steps
            )
        v2_metrics, v2_checks = _v2_face_metrics(
            model,
            config,
            phase_steps,
            traces,
            hold_start_override=hold_start if schema_version >= 3 else None,
            material_start_override=material_start_override,
        )
        metrics.update(v2_metrics)
        # The historical duty fields become strict target-face duty for v2.
        metrics["contact_duty"] = v2_metrics["target_face_contact_duty"]
        metrics["simultaneous_contact_duty"] = v2_metrics[
            "target_face_simultaneous_duty"
        ]
        checks.update(v2_checks)
        checks["thumb_contact_duty"] = v2_checks["thumb_target_face_contact_duty"]
        checks["index_contact_duty"] = v2_checks["index_target_face_contact_duty"]
        checks["middle_contact_duty"] = v2_checks["middle_target_face_contact_duty"]
        checks["simultaneous_three_finger_contact"] = v2_checks[
            "simultaneous_target_face_topology"
        ]

    stage_status: dict[str, Any] | None = None
    if schema_version >= 3:
        v3_metrics, v3_checks = _v3_stage_metrics(model, config, traces)
        metrics.update(v3_metrics)
        # Preserve the established public lift keys while changing their v3
        # reference to the verified grasp immediately before manipulation.
        metrics["baseline_cube_z_m"] = v3_metrics["operation_baseline_cube_z_m"]
        metrics["median_lift_m"] = v3_metrics["operation_median_lift_m"]
        metrics["minimum_lift_m"] = v3_metrics["operation_minimum_lift_m"]
        metrics["peak_lift_m"] = v3_metrics["operation_peak_lift_m"]
        checks.update(v3_checks)
        checks["median_lift_reached"] = v3_checks[
            "operation_median_lift_reached"
        ]
        checks["minimum_lift_reached"] = v3_checks[
            "operation_minimum_lift_reached"
        ]

        if schema_version >= 4:
            v4_metrics, v4_checks = _v4_alignment_metrics(
                model, config, traces
            )
            metrics.update(v4_metrics)
            checks.update(v4_checks)
        if schema_version >= 5:
            fingertip_metrics = _v5_fingertip_contact_metrics(
                model, info, config, traces
            )
            metrics["fingertip_contact"] = fingertip_metrics
            checks.update(
                {
                    "v5_pad_fraction_trace_matches_raw_forces": bool(
                        fingertip_metrics[
                            "trace_fraction_matches_raw_forces"
                        ]
                    ),
                    "v5_pose_and_thumb_bend_trace_matches_raw_state": bool(
                        fingertip_metrics[
                            "trace_pose_and_thumb_bend_match_raw_state"
                        ]
                    ),
                    **(
                        {
                            "v5_thumb_bend_command_matches_control_protocol": bool(
                                fingertip_metrics[
                                    "thumb_bend_command_matches_control_protocol"
                                ]
                            )
                        }
                        if schema_version <= 13
                        else {}
                    ),
                }
            )
        if schema_version >= 6:
            v6_metrics, v6_checks = _v6_pose_preservation_metrics(
                model, info, config, traces
            )
            metrics.update(v6_metrics)
            checks.update(v6_checks)
        if schema_version >= 8:
            v8_closure_metrics, v8_closure_checks = (
                _v8_closure_alignment_metrics(config, traces)
            )
            metrics.update(v8_closure_metrics)
            checks.update(v8_closure_checks)
            v8_motion_metrics, v8_motion_checks = (
                _v8_motion_smoothness_metrics(model, config, traces)
            )
            metrics.update(v8_motion_metrics)
            checks.update(v8_motion_checks)
        if schema_version >= 9:
            v9_metrics, v9_checks = _v9_actual_grasp_pose_metrics(
                model, config, traces
            )
            metrics.update(v9_metrics)
            checks.update(v9_checks)
        if schema_version >= 12:
            v12_metrics, v12_checks = _v12_contact_point_metrics(
                model, config, traces
            )
            metrics.update(v12_metrics)
            checks.update(v12_checks)
        if schema_version >= 14:
            v14_metrics, v14_checks = _v14_contact_preservation_metrics(
                model, config, traces
            )
            metrics.update(v14_metrics)
            checks.update(v14_checks)
        if schema_version >= 15:
            v15_metrics, v15_checks = _v15_joint_pair_alignment_metrics(
                model, info, config, traces
            )
            metrics.update(v15_metrics)
            checks.update(v15_checks)

        grasp_check_names = [
            "stable_grasp_acquired",
            "grasp_acquisition_event_consistent",
            "grasp_gate_contiguous_window",
            "target_face_evidence_matches_raw_trace",
            "grasp_latch_remains_set",
            "grasp_gate_counter_is_consistent",
            "grasp_support_retained",
            "grasp_pose_is_stable",
            "no_early_object_lift",
        ]
        operation_check_names = [
            "operation_started_only_after_acquisition",
            "no_operation_target_without_grasp",
            "controller_state_sequence_is_consistent",
            "controller_operation_events_are_consistent",
            "controller_termination_event_is_consistent",
            "manipulation_progress_is_consistent",
            "abort_holds_grasp_pose",
            "operation_executed",
            "manipulation_completed",
            "operation_thumb_target_face_contact_duty",
            "operation_index_target_face_contact_duty",
            "operation_middle_target_face_contact_duty",
            "operation_simultaneous_target_face_topology",
            "operation_median_lift_reached",
            "operation_minimum_lift_reached",
        ]
        if schema_version >= 4:
            grasp_check_names.extend(
                [
                    "v4_alignment_trace_matches_raw_contacts",
                    "grasp_contact_height_alignment_contiguous",
                    "finger_down_tilt_within_range",
                    "palm_plane_ground_angle_within_range",
                ]
            )
            grasp_check_names.append(
                "initial_root_cube_distance_within_range"
                if schema_version >= 5
                else "palm_press_depth_within_range"
            )
            operation_check_names.append(
                "operation_contact_height_aligned_duty"
            )
        if schema_version >= 5:
            grasp_check_names.extend(
                [
                    "v5_contact_exclusion_gate_matches_raw_trace",
                    "v5_pad_fraction_trace_matches_raw_forces",
                    "v5_pose_and_thumb_bend_trace_matches_raw_state",
                ]
            )
            if schema_version <= 13:
                grasp_check_names.append(
                    "v5_thumb_bend_command_matches_control_protocol"
                )
        if schema_version >= 6:
            grasp_check_names.extend(
                [
                    "v6_pose_preservation_trace_matches_raw_state",
                    "v6_close_profile_trace_matches_config",
                    "v6_first_distal_contact_steps_match_raw_trace",
                    "v6_initial_joint_state_matches_pregrasp_config",
                    "object_pose_preserved_until_grasp_acquisition",
                    "support_retained_until_grasp_acquisition",
                    "no_hand_cube_contact_during_settle",
                ]
            )
        if schema_version >= 8:
            grasp_check_names.extend(
                [
                    "v8_closure_alignment_trace_matches_vectors",
                    "closure_alignment_valid_for_all_fingers",
                    "closure_alignment_p95_within_limit",
                    "closure_inward_speed_positive",
                ]
            )
            operation_check_names.extend(
                [
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
                ]
            )
            if int(config["contact_feedback"].get("schema_version", 1)) >= 2:
                operation_check_names.extend(
                    [
                        "v14_online_tangent_slip_matches_raw_contacts",
                        "v14_tangent_slip_freeze_uses_previous_observation",
                        "v14_tangent_slip_abort_matches_current_safety_observation",
                    ]
                )
        if schema_version >= 9:
            grasp_check_names.extend(
                [
                    "grasp_pose_base_gate_contiguous",
                    "thumb_actual_qpos_within_range",
                    "actual_joint_median_matches_nominal",
                    "actual_joint_window_stable",
                    "actual_grasp_pose_locked",
                    "v9_actual_qpos_trace_matches_raw_state",
                    "v9_actual_grasp_pose_summary_matches_raw_state",
                    "v9_online_grasp_lock_matches_offline_recomputation",
                    "v9_preload_command_is_not_grasp_pose_evidence",
                ]
            )
        if schema_version >= 12:
            grasp_check_names.extend(
                [
                    "v12_contact_point_trace_matches_raw_contacts",
                    "v12_contact_point_gate_matches_raw_contacts",
                    "grasp_contact_points_contiguous",
                ]
            )
        if schema_version >= 14:
            operation_check_names.extend(
                [
                    "v14_top_level_identity_trace_matches_recomputed_config",
                    "v14_manipulation_completed_within_saved_verify_slack",
                    "v14_minimum_hold_duration_preserved",
                    "v14_manipulation_end_is_first_full_progress_sample",
                    "v14_target_face_effective_matches_raw_trace",
                    "v14_contact_loss_trace_matches_raw_contacts",
                    "v14_feedback_uses_previous_observation",
                    "v14_freeze_and_recovery_match_previous_contact_risk",
                    "v14_static_plan_trace_matches_config",
                    "v14_dynamic_plan_trace_matches_progress",
                    "v14_command_composition_matches_plan_and_feedback",
                    "v14_force_feedback_trace_is_bounded_and_recomputable",
                    "v14_operation_did_not_abort",
                    "v14_plan_progress_is_monotonic",
                    "v14_plan_progress_reached_one",
                    "v14_thumb_contact_duty_at_least_99_percent",
                    "v14_index_contact_duty_at_least_99_percent",
                    "v14_middle_contact_duty_at_least_99_percent",
                    "v14_simultaneous_contact_duty_at_least_99_percent",
                    "v14_thumb_contact_loss_within_limit",
                    "v14_index_contact_loss_within_limit",
                    "v14_middle_contact_loss_within_limit",
                    "v14_simultaneous_contact_loss_within_limit",
                ]
            )
        if schema_version >= 15:
            grasp_check_names.extend(
                [
                    "v15_joint_pair_trace_matches_raw_state",
                    "v15_joint_pair_grasp_gate_matches_raw_state",
                    "v15_active_finger_self_collision_trace_matches_raw_state",
                    "grasp_no_active_finger_self_collision",
                    "v15_joint_pair_grasp_window_p95_within_limit",
                    "v15_joint_pair_grasp_window_max_within_limit",
                    "v15_joint_pair_direction_and_length_valid_at_grasp",
                    "v15_joint_pair_grasp_summary_matches_trace",
                ]
            )
            operation_check_names.extend(
                [
                    "v15_joint_pair_feedback_uses_previous_observation",
                    "v15_joint_pair_freeze_uses_previous_observation",
                    "v15_joint_pair_violation_trace_matches_raw_state",
                    "v15_joint_pair_static_jacobians_match_config",
                    "v15_joint_pair_requests_use_previous_observation",
                    "v15_joint_pair_feedback_is_bounded",
                    "v15_joint_pair_operation_p95_within_limit",
                    "v15_joint_pair_operation_max_within_limit",
                    "v15_joint_pair_operation_duty_at_least_99_percent",
                    "v15_joint_pair_continuous_violation_within_limit",
                    "v15_joint_pair_direction_and_length_preserved",
                    "v15_joint_pair_no_alignment_abort_risk",
                    "operation_no_active_finger_self_collision",
                ]
            )
        if schema_version >= 16:
            operation_check_names.extend(
                [
                    "v16_native_tactile_target_face_effective_matches_raw_trace",
                    "v16_rolling_aware_target_face_effective_matches_raw_trace",
                ]
            )
        grasp_success = all(bool(checks[name]) for name in grasp_check_names)
        operation_executed = bool(checks["operation_executed"])
        manipulation_success = bool(
            grasp_success
            and operation_executed
            and all(bool(checks[name]) for name in operation_check_names)
        )
        manipulation_label = _stage_manipulation_label(
            schema_version=schema_version,
            manipulation_success=manipulation_success,
            legacy_operation_executed=operation_executed,
            manipulation_start_step=int(metrics["manipulation_start_step"]),
            operation_sample_count=(
                int(
                    metrics["contact_preserving_planned_lift"][
                        "operation_sample_count"
                    ]
                )
                if schema_version >= 15
                else 0
            ),
        )
        stage_status = {
            "grasp_success": grasp_success,
            "grasp": "acquired" if grasp_success else "failed",
            "manipulation_success": manipulation_success,
            "manipulation": manipulation_label,
        }

    failed_checks = [name for name, passed in checks.items() if not passed]
    result = {
        "passed": not failed_checks,
        "failed_checks": failed_checks,
        "checks": checks,
        "metrics": metrics,
        "phase_steps": phase_steps,
    }
    if stage_status is not None:
        stage_status["full_success"] = not failed_checks
        result["stage_status"] = stage_status
    return result
