from __future__ import annotations

import copy
from pathlib import Path

import mujoco
import numpy as np
import pytest

from xhand_grasp.config import load_config
from xhand_grasp.contact_environment import (
    ContactEnvironmentSpec,
    apply_to_model,
    audit_allowed_model_changes,
    canonical_environment_id,
    compiled_environment_snapshot,
    requested_environment_snapshot,
    runtime_cube_contact_snapshot,
    verify_compiled_environment,
    verify_runtime_cube_contacts,
)
from xhand_grasp.scene import build_model


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / (
    "grasp_configs/left_opposed_face_palm_down_joint_pair_near_zero_"
    "contact_preserving_planned_lift.json"
)


def _model():
    return build_model(load_config(CONFIG))


def test_environment_spec_is_canonical_round_trippable_and_tamper_evident() -> None:
    spec = ContactEnvironmentSpec(
        sliding_friction=1.2,
        impratio=100.0,
        tolerance=1e-10,
        noslip_iterations=50,
    )
    config = spec.as_config()
    assert ContactEnvironmentSpec.from_config(config) == spec
    assert requested_environment_snapshot(spec)["environment_id"] == (
        spec.environment_id
    )
    assert canonical_environment_id(requested_environment_snapshot(spec)) == (
        spec.environment_id
    )

    reordered = {
        "required_model": copy.deepcopy(config["required_model"]),
        "solver": copy.deepcopy(config["solver"]),
        "contact": copy.deepcopy(config["contact"]),
        "schema_version": config["schema_version"],
        "environment_id": config["environment_id"],
    }
    assert ContactEnvironmentSpec.from_config(reordered).environment_id == (
        spec.environment_id
    )

    tampered = copy.deepcopy(config)
    tampered["solver"]["impratio"] = 30.0
    with pytest.raises(ValueError, match="environment_id does not match"):
        ContactEnvironmentSpec.from_config(tampered)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"schema_version": True}, "schema_version"),
        ({"schema_version": 2}, "schema_version"),
        ({"sliding_friction": 0.0}, "sliding_friction"),
        ({"condim": 5}, "condim"),
        ({"solref": (0.0, 1.0)}, "solref"),
        ({"solimp": (0.9, 0.95, 0.0, 0.5, 2.0)}, "solimp"),
        ({"cone": "unknown"}, "cone"),
        ({"solver": "unknown"}, "solver"),
        ({"iterations": 0}, "iterations"),
        ({"noslip_iterations": -1}, "noslip_iterations"),
    ],
)
def test_environment_spec_rejects_malformed_values(
    updates: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ContactEnvironmentSpec(**updates)


def test_apply_records_actual_compiled_parameters_without_changing_invariants() -> None:
    model, info = _model()
    baseline_timestep = float(model.opt.timestep)
    baseline_integrator = int(model.opt.integrator)
    baseline_gravity = model.opt.gravity.copy()
    spec = ContactEnvironmentSpec(
        sliding_friction=1.2,
        torsional_friction=0.007,
        rolling_friction=0.0002,
        condim=6,
        solref=(0.003, 1.2),
        solimp=(0.88, 0.97, 0.0008, 0.45, 3.0),
        impratio=100.0,
        tolerance=1e-10,
        iterations=200,
        ls_iterations=80,
        noslip_iterations=50,
        noslip_tolerance=1e-8,
    )

    compiled = apply_to_model(model, info.cube_geom_id, spec)
    verify_compiled_environment(spec, compiled)
    assert compiled["environment_id"] == spec.environment_id
    assert compiled["contact"] == {
        "sliding_friction": pytest.approx(1.2),
        "torsional_friction": pytest.approx(0.007),
        "rolling_friction": pytest.approx(0.0002),
        "condim": 6,
        "priority": 10,
        "solref": pytest.approx([0.003, 1.2]),
        "solimp": pytest.approx([0.88, 0.97, 0.0008, 0.45, 3.0]),
    }
    assert compiled["solver"] == {
        "cone": "elliptic",
        "solver": "newton",
        "impratio": pytest.approx(100.0),
        "tolerance": pytest.approx(1e-10),
        "iterations": 200,
        "ls_iterations": 80,
        "noslip_iterations": 50,
        "noslip_tolerance": pytest.approx(1e-8),
    }
    assert float(model.opt.timestep) == baseline_timestep
    assert int(model.opt.integrator) == baseline_integrator
    np.testing.assert_array_equal(model.opt.gravity, baseline_gravity)


def test_apply_rejects_invariant_mismatch_before_mutation() -> None:
    model, info = _model()
    before = compiled_environment_snapshot(model, info.cube_geom_id)
    spec = ContactEnvironmentSpec(
        sliding_friction=1.2,
        required_gravity_m_s2=(0.0, 0.0, -1.0),
    )
    with pytest.raises(ValueError, match="gravity"):
        apply_to_model(model, info.cube_geom_id, spec)
    assert compiled_environment_snapshot(model, info.cube_geom_id) == before


def test_priority_produces_requested_runtime_contact_parameters() -> None:
    config = load_config(CONFIG)
    model, info = build_model(config)
    spec = ContactEnvironmentSpec(
        sliding_friction=1.2,
        torsional_friction=0.007,
        rolling_friction=0.0002,
        condim=4,
        solref=(0.003, 1.2),
        solimp=(0.88, 0.97, 0.0008, 0.45, 3.0),
        impratio=30.0,
        tolerance=1e-10,
        noslip_iterations=10,
    )
    apply_to_model(model, info.cube_geom_id, spec)
    data = mujoco.MjData(model)
    # Make the support contact unambiguous through the freejoint state.  The
    # versioned config and compiled freejoint remain untouched.
    data.qpos[info.cube_qpos_adr + 2] -= 0.0005
    mujoco.mj_forward(model, data)
    snapshot = runtime_cube_contact_snapshot(model, data, info.cube_geom_id)
    verify_runtime_cube_contacts(spec, snapshot)
    support = [
        contact
        for contact in snapshot["contacts"]
        if contact["other_geom_id"] == info.support_geom_id
    ]
    assert support
    for contact in support:
        assert contact["dim"] == 4
        assert contact["friction"] == pytest.approx(
            [1.2, 1.2, 0.007, 0.0002, 0.0002]
        )
        assert contact["solref"] == pytest.approx([0.003, 1.2])
        assert contact["solimp"] == pytest.approx(
            [0.88, 0.97, 0.0008, 0.45, 3.0]
        )

    tampered = copy.deepcopy(snapshot)
    tampered["contacts"][0]["dim"] = 6
    with pytest.raises(RuntimeError, match="snapshot_id"):
        verify_runtime_cube_contacts(spec, tampered)


def test_allowed_change_audit_accepts_only_explicit_environment_fields() -> None:
    reference, reference_info = _model()
    candidate, candidate_info = _model()
    spec = ContactEnvironmentSpec(
        sliding_friction=1.2,
        impratio=100.0,
        tolerance=1e-10,
        noslip_iterations=50,
    )
    apply_to_model(candidate, candidate_info.cube_geom_id, spec)
    allowed = {
        "contact.sliding_friction",
        "solver.impratio",
        "solver.tolerance",
        "solver.noslip_iterations",
    }
    audit = audit_allowed_model_changes(
        reference,
        candidate,
        reference_info.cube_geom_id,
        candidate_cube_geom_id=candidate_info.cube_geom_id,
        allowed_environment_fields=allowed,
    )
    audit.assert_passed()
    assert audit.passed
    assert set(audit.changed_environment_fields) == allowed
    assert audit.unexpected_environment_fields == ()
    assert audit.unexpected_model_fields == ()
    assert (
        audit.reference_immutable_model_sha256
        == audit.candidate_immutable_model_sha256
    )
    mapping = audit.as_mapping()
    assert mapping["passed"] is True
    assert len(mapping["audit_id"]) == 64


def test_allowed_change_audit_rejects_unlisted_and_hidden_physics_changes() -> None:
    reference, reference_info = _model()
    candidate, candidate_info = _model()
    apply_to_model(
        candidate,
        candidate_info.cube_geom_id,
        ContactEnvironmentSpec(sliding_friction=1.2, noslip_iterations=10),
    )
    candidate.body_mass[candidate_info.cube_body_id] += 0.001
    audit = audit_allowed_model_changes(
        reference,
        candidate,
        reference_info.cube_geom_id,
        candidate_cube_geom_id=candidate_info.cube_geom_id,
        allowed_environment_fields={"contact.sliding_friction"},
    )
    assert not audit.passed
    assert audit.unexpected_environment_fields == ("solver.noslip_iterations",)
    assert "model.body_mass" in audit.unexpected_model_fields
    with pytest.raises(RuntimeError, match="noslip_iterations"):
        audit.assert_passed()


def test_unknown_allowed_change_name_is_rejected() -> None:
    reference, info = _model()
    candidate, candidate_info = _model()
    with pytest.raises(ValueError, match="unknown allowed"):
        audit_allowed_model_changes(
            reference,
            candidate,
            info.cube_geom_id,
            candidate_cube_geom_id=candidate_info.cube_geom_id,
            allowed_environment_fields={"required_model.gravity_m_s2"},
        )
