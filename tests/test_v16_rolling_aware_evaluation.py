from __future__ import annotations

import numpy as np

from xhand_grasp.contacts import FACE_ORDER, Face
from xhand_grasp.evaluation import (
    _planned_command_total_correction_rad,
    _v16_rolling_aware_target_face_evidence,
    _versioned_operation_feedback_risk,
)


def _config() -> dict:
    return {
        "schema_version": 16,
        "contact_topology": {
            "target_faces": {
                "thumb": "-X",
                "index": "+X",
                "mid": "+X",
            }
        },
        "control_protocol": {
            "grasp_gate": {
                "min_target_face_force_n": 0.05,
                "min_target_force_fraction": 0.95,
                "require_touch": True,
            }
        },
        "acceptance": {"touch_force_min_n": 0.05},
    }


def _traces() -> dict[str, np.ndarray]:
    total = 4
    face_force = np.zeros((total, 3, len(FACE_ORDER)), dtype=np.float64)
    face_force[:, 0, FACE_ORDER.index(Face.X_NEG)] = 0.2
    face_force[:, 1:, FACE_ORDER.index(Face.X_POS)] = 0.2
    return {
        "time": np.arange(total, dtype=np.float64) * 0.001,
        "control_state": np.asarray(
            ("VERIFY", "MANIPULATE", "HOLD", "HOLD")
        ),
        "distal_face_force_n": face_force,
        "active_nondistal_force_n": np.zeros((total, 3)),
        "tactile_max": np.zeros((total, 5)),
        "rolling_contact_valid": np.ones((total, 3), dtype=bool),
        "rolling_contact_normal_force_n": np.full((total, 3), 0.2),
    }


def test_v16_verify_stays_native_but_operation_accepts_real_rolling_pad() -> None:
    evidence = _v16_rolling_aware_target_face_evidence(_config(), _traces())

    assert not np.any(evidence["native"])
    assert not np.any(evidence["effective"][0])
    assert np.all(evidence["effective"][1:])
    assert np.all(evidence["rolling_physical"])


def test_v16_rolling_pad_cannot_bypass_offtarget_or_nondistal_contact() -> None:
    traces = _traces()
    traces["distal_face_force_n"][2, 0, FACE_ORDER.index(Face.Y_POS)] = 0.2
    traces["active_nondistal_force_n"][3, 1] = 0.2

    evidence = _v16_rolling_aware_target_face_evidence(_config(), traces)

    assert not evidence["effective"][2, 0]
    assert evidence["material_off_target"][2, 0]
    assert not evidence["effective"][3, 1]
    assert evidence["material_active_nondistal"][3, 1]


def test_versioned_command_composition_adds_rolling_only_for_v16() -> None:
    shape = (2, 4)
    traces = {
        "feedback_correction_rad": np.full(shape, 1.0),
        "joint_pair_feedback_correction_rad": np.full(shape, 2.0),
        "rolling_slip_correction_rad": np.full(shape, 4.0),
    }

    np.testing.assert_array_equal(
        _planned_command_total_correction_rad(14, traces, shape), 1.0
    )
    np.testing.assert_array_equal(
        _planned_command_total_correction_rad(15, traces, shape), 3.0
    )
    np.testing.assert_array_equal(
        _planned_command_total_correction_rad(16, traces, shape), 7.0
    )


def test_v16_freeze_risk_uses_rolling_and_ignores_legacy_centroid_slip() -> None:
    contact = np.asarray((False, True, False, False))
    pair = np.asarray((False, False, True, False))
    legacy = np.asarray((True, False, False, False))
    pair_legacy = np.asarray((False, False, False, True))
    rolling = np.asarray((False, False, False, True))

    v14 = _versioned_operation_feedback_risk(
        14,
        contact,
        pair_risk=np.zeros_like(pair),
        legacy_slip_risk=legacy,
        pair_legacy_slip_risk=pair_legacy,
        rolling_slip_freeze_active=rolling,
    )
    v15 = _versioned_operation_feedback_risk(
        15,
        contact,
        pair_risk=pair,
        legacy_slip_risk=legacy,
        pair_legacy_slip_risk=pair_legacy,
        rolling_slip_freeze_active=rolling,
    )
    v16 = _versioned_operation_feedback_risk(
        16,
        contact,
        pair_risk=pair,
        legacy_slip_risk=legacy,
        pair_legacy_slip_risk=pair_legacy,
        rolling_slip_freeze_active=rolling,
    )

    np.testing.assert_array_equal(v14, (True, True, False, False))
    np.testing.assert_array_equal(v15, (True, True, True, True))
    # The legacy-only source at sample 0 is ignored; contact, pair and rolling
    # remain authoritative at samples 1, 2 and 3 respectively.
    np.testing.assert_array_equal(v16, (False, True, True, True))
