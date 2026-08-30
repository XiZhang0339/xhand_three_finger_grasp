"""Auditable MuJoCo contact-environment overrides.

The established XHAND experiments compile their scene from ``xhand_left.xml``
and must remain byte-for-byte and numerically unchanged.  This module is an
opt-in sidecar for later experiments which need to vary contact/solver
parameters while keeping the hand, object, controller and trajectory fixed.

The contract deliberately separates three pieces of evidence:

* :class:`ContactEnvironmentSpec` is the requested, versioned configuration;
* :func:`compiled_environment_snapshot` records what an ``MjModel`` actually
  contains after applying that request;
* :func:`audit_allowed_model_changes` fingerprints the rest of both models and
  fails closed when anything outside an explicit allow-list changed.

Importing this module has no side effects.  In particular, no existing scene
builder or simulation path calls :func:`apply_to_model` implicitly.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import mujoco
import numpy as np

from .grasp_pose import canonical_sha256


CONTACT_ENVIRONMENT_SCHEMA_VERSION = 1
CONTACT_ENVIRONMENT_SNAPSHOT_SCHEMA_VERSION = 1
MODEL_CHANGE_AUDIT_SCHEMA_VERSION = 1


_CONE_TO_VALUE = {
    "pyramidal": int(mujoco.mjtCone.mjCONE_PYRAMIDAL),
    "elliptic": int(mujoco.mjtCone.mjCONE_ELLIPTIC),
}
_SOLVER_TO_VALUE = {
    "pgs": int(mujoco.mjtSolver.mjSOL_PGS),
    "cg": int(mujoco.mjtSolver.mjSOL_CG),
    "newton": int(mujoco.mjtSolver.mjSOL_NEWTON),
}
_INTEGRATOR_TO_VALUE = {
    "euler": int(mujoco.mjtIntegrator.mjINT_EULER),
    "rk4": int(mujoco.mjtIntegrator.mjINT_RK4),
    "implicit": int(mujoco.mjtIntegrator.mjINT_IMPLICIT),
    "implicitfast": int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST),
}
_VALUE_TO_CONE = {value: key for key, value in _CONE_TO_VALUE.items()}
_VALUE_TO_SOLVER = {value: key for key, value in _SOLVER_TO_VALUE.items()}
_VALUE_TO_INTEGRATOR = {
    value: key for key, value in _INTEGRATOR_TO_VALUE.items()
}


CONTACT_ENVIRONMENT_FIELD_PATHS = frozenset(
    {
        "contact.sliding_friction",
        "contact.torsional_friction",
        "contact.rolling_friction",
        "contact.condim",
        "contact.priority",
        "contact.solref",
        "contact.solimp",
        "solver.cone",
        "solver.solver",
        "solver.impratio",
        "solver.tolerance",
        "solver.iterations",
        "solver.ls_iterations",
        "solver.noslip_iterations",
        "solver.noslip_tolerance",
    }
)

# These fields are removed from the immutable-model fingerprint and compared
# individually through CONTACT_ENVIRONMENT_FIELD_PATHS instead.
_CONTACT_ARRAY_NAMES = frozenset(
    {
        "geom_friction",
        "geom_condim",
        "geom_priority",
        "geom_solref",
        "geom_solimp",
    }
)
_SOLVER_OPTION_NAMES = frozenset(
    {
        "cone",
        "solver",
        "impratio",
        "tolerance",
        "iterations",
        "ls_iterations",
        "noslip_iterations",
        "noslip_tolerance",
    }
)


def _finite(value: Any, label: str) -> float:
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{label} must be finite")
    return resolved


def _positive(value: Any, label: str) -> float:
    resolved = _finite(value, label)
    if resolved <= 0.0:
        raise ValueError(f"{label} must be positive")
    return resolved


def _nonnegative(value: Any, label: str) -> float:
    resolved = _finite(value, label)
    if resolved < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    return resolved


def _integer(value: Any, label: str, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be an integer")
    try:
        resolved = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be an integer") from error
    if resolved != value or resolved < minimum:
        qualifier = "nonnegative" if minimum == 0 else f">= {minimum}"
        raise ValueError(f"{label} must be an integer {qualifier}")
    return resolved


def _finite_tuple(
    value: Iterable[Any], length: int, label: str
) -> tuple[float, ...]:
    try:
        result = tuple(_finite(item, label) for item in value)
    except TypeError as error:
        raise ValueError(f"{label} must contain {length} finite values") from error
    if len(result) != length:
        raise ValueError(f"{label} must contain {length} finite values")
    return result


def _enum_name(value: Any, inverse: Mapping[int, str], label: str) -> str:
    resolved = int(value)
    try:
        return inverse[resolved]
    except KeyError as error:
        raise ValueError(f"unsupported compiled MuJoCo {label}: {resolved}") from error


@dataclass(frozen=True, slots=True)
class ContactEnvironmentSpec:
    """Versioned contact and solver request for one cube collision geom.

    Timestep, gravity and integrator are invariants rather than sweep knobs.
    ``apply_to_model`` verifies them before mutating the model and never writes
    them.  This prevents an environment campaign from silently making the task
    easier through a smaller timestep, weaker gravity or different integrator.
    """

    schema_version: int = CONTACT_ENVIRONMENT_SCHEMA_VERSION
    sliding_friction: float = 0.8
    torsional_friction: float = 0.005
    rolling_friction: float = 0.0001
    condim: int = 4
    priority: int = 10
    solref: tuple[float, float] = (0.004, 1.0)
    solimp: tuple[float, float, float, float, float] = (
        0.9,
        0.95,
        0.001,
        0.5,
        2.0,
    )
    cone: str = "elliptic"
    solver: str = "newton"
    impratio: float = 10.0
    tolerance: float = 1e-8
    iterations: int = 100
    ls_iterations: int = 50
    noslip_iterations: int = 0
    noslip_tolerance: float = 1e-6
    required_timestep_s: float = 0.001
    required_integrator: str = "implicitfast"
    required_gravity_m_s2: tuple[float, float, float] = (0.0, 0.0, -9.81)

    def __post_init__(self) -> None:
        schema_version = _integer(
            self.schema_version, "schema_version", minimum=1
        )
        if schema_version != CONTACT_ENVIRONMENT_SCHEMA_VERSION:
            raise ValueError(
                "unsupported contact-environment schema_version: "
                f"{self.schema_version!r}"
            )
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(
            self,
            "sliding_friction",
            _positive(self.sliding_friction, "contact.sliding_friction"),
        )
        object.__setattr__(
            self,
            "torsional_friction",
            _nonnegative(self.torsional_friction, "contact.torsional_friction"),
        )
        object.__setattr__(
            self,
            "rolling_friction",
            _nonnegative(self.rolling_friction, "contact.rolling_friction"),
        )
        condim = _integer(self.condim, "contact.condim", minimum=1)
        if condim not in (1, 3, 4, 6):
            raise ValueError("contact.condim must be one of 1, 3, 4 or 6")
        object.__setattr__(self, "condim", condim)
        object.__setattr__(
            self,
            "priority",
            _integer(self.priority, "contact.priority", minimum=0),
        )
        solref = _finite_tuple(self.solref, 2, "contact.solref")
        if solref[0] <= 0.0 or solref[1] <= 0.0:
            raise ValueError("contact.solref time constant and damping must be positive")
        object.__setattr__(self, "solref", solref)
        solimp = _finite_tuple(self.solimp, 5, "contact.solimp")
        if not (0.0 < solimp[0] <= 1.0 and 0.0 < solimp[1] <= 1.0):
            raise ValueError("contact.solimp d0 and dwidth must lie in (0, 1]")
        if solimp[2] <= 0.0:
            raise ValueError("contact.solimp width must be positive")
        if not 0.0 <= solimp[3] <= 1.0:
            raise ValueError("contact.solimp midpoint must lie in [0, 1]")
        if solimp[4] < 1.0:
            raise ValueError("contact.solimp power must be at least 1")
        object.__setattr__(self, "solimp", solimp)

        cone = str(self.cone).strip().lower()
        solver = str(self.solver).strip().lower()
        integrator = str(self.required_integrator).strip().lower()
        if cone not in _CONE_TO_VALUE:
            raise ValueError(f"unsupported solver.cone: {self.cone!r}")
        if solver not in _SOLVER_TO_VALUE:
            raise ValueError(f"unsupported solver.solver: {self.solver!r}")
        if integrator not in _INTEGRATOR_TO_VALUE:
            raise ValueError(
                f"unsupported required_model.integrator: {self.required_integrator!r}"
            )
        object.__setattr__(self, "cone", cone)
        object.__setattr__(self, "solver", solver)
        object.__setattr__(self, "required_integrator", integrator)
        object.__setattr__(
            self, "impratio", _positive(self.impratio, "solver.impratio")
        )
        object.__setattr__(
            self, "tolerance", _positive(self.tolerance, "solver.tolerance")
        )
        object.__setattr__(
            self,
            "iterations",
            _integer(self.iterations, "solver.iterations", minimum=1),
        )
        object.__setattr__(
            self,
            "ls_iterations",
            _integer(self.ls_iterations, "solver.ls_iterations", minimum=1),
        )
        object.__setattr__(
            self,
            "noslip_iterations",
            _integer(
                self.noslip_iterations, "solver.noslip_iterations", minimum=0
            ),
        )
        object.__setattr__(
            self,
            "noslip_tolerance",
            _positive(self.noslip_tolerance, "solver.noslip_tolerance"),
        )
        object.__setattr__(
            self,
            "required_timestep_s",
            _positive(self.required_timestep_s, "required_model.timestep_s"),
        )
        object.__setattr__(
            self,
            "required_gravity_m_s2",
            _finite_tuple(
                self.required_gravity_m_s2, 3, "required_model.gravity_m_s2"
            ),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "contact": {
                "sliding_friction": self.sliding_friction,
                "torsional_friction": self.torsional_friction,
                "rolling_friction": self.rolling_friction,
                "condim": self.condim,
                "priority": self.priority,
                "solref": list(self.solref),
                "solimp": list(self.solimp),
            },
            "solver": {
                "cone": self.cone,
                "solver": self.solver,
                "impratio": self.impratio,
                "tolerance": self.tolerance,
                "iterations": self.iterations,
                "ls_iterations": self.ls_iterations,
                "noslip_iterations": self.noslip_iterations,
                "noslip_tolerance": self.noslip_tolerance,
            },
            "required_model": {
                "timestep_s": self.required_timestep_s,
                "integrator": self.required_integrator,
                "gravity_m_s2": list(self.required_gravity_m_s2),
            },
        }

    @property
    def environment_id(self) -> str:
        return canonical_environment_id(self._payload())

    def as_config(self) -> dict[str, Any]:
        result = self._payload()
        result["environment_id"] = self.environment_id
        return result

    @classmethod
    def from_config(cls, value: Mapping[str, Any]) -> "ContactEnvironmentSpec":
        if not isinstance(value, Mapping):
            raise ValueError("contact environment must be a mapping")
        expected_top = {
            "schema_version",
            "contact",
            "solver",
            "required_model",
        }
        actual_top = set(value) - {"environment_id"}
        if actual_top != expected_top:
            raise ValueError(
                "contact environment top-level keys must be exactly "
                f"{sorted(expected_top)}"
            )
        contact = value.get("contact")
        solver = value.get("solver")
        required = value.get("required_model")
        if not all(isinstance(item, Mapping) for item in (contact, solver, required)):
            raise ValueError("contact, solver and required_model must be mappings")
        assert isinstance(contact, Mapping)
        assert isinstance(solver, Mapping)
        assert isinstance(required, Mapping)
        expected_contact = {
            "sliding_friction",
            "torsional_friction",
            "rolling_friction",
            "condim",
            "priority",
            "solref",
            "solimp",
        }
        expected_solver = {
            "cone",
            "solver",
            "impratio",
            "tolerance",
            "iterations",
            "ls_iterations",
            "noslip_iterations",
            "noslip_tolerance",
        }
        expected_required = {"timestep_s", "integrator", "gravity_m_s2"}
        for block, expected, label in (
            (contact, expected_contact, "contact"),
            (solver, expected_solver, "solver"),
            (required, expected_required, "required_model"),
        ):
            if set(block) != expected:
                raise ValueError(f"{label} keys must be exactly {sorted(expected)}")
        result = cls(
            schema_version=value["schema_version"],
            sliding_friction=contact["sliding_friction"],
            torsional_friction=contact["torsional_friction"],
            rolling_friction=contact["rolling_friction"],
            condim=contact["condim"],
            priority=contact["priority"],
            solref=contact["solref"],
            solimp=contact["solimp"],
            cone=solver["cone"],
            solver=solver["solver"],
            impratio=solver["impratio"],
            tolerance=solver["tolerance"],
            iterations=solver["iterations"],
            ls_iterations=solver["ls_iterations"],
            noslip_iterations=solver["noslip_iterations"],
            noslip_tolerance=solver["noslip_tolerance"],
            required_timestep_s=required["timestep_s"],
            required_integrator=required["integrator"],
            required_gravity_m_s2=required["gravity_m_s2"],
        )
        declared_id = value.get("environment_id")
        if declared_id is not None and declared_id != result.environment_id:
            raise ValueError("contact environment_id does not match its content")
        return result


def canonical_environment_id(value: Mapping[str, Any] | ContactEnvironmentSpec) -> str:
    """Return a domain-separated canonical identity for an environment."""

    if isinstance(value, ContactEnvironmentSpec):
        payload = value._payload()
    else:
        try:
            payload = {
                key: value[key]
                for key in (
                    "schema_version",
                    "contact",
                    "solver",
                    "required_model",
                )
            }
        except KeyError as error:
            raise ValueError(
                "environment identity requires schema_version, contact, solver "
                "and required_model"
            ) from error
    return canonical_sha256(
        {
            "identity_schema_version": 1,
            "domain": "xhand_mujoco_contact_environment",
            "payload": payload,
        }
    )


def requested_environment_snapshot(spec: ContactEnvironmentSpec) -> dict[str, Any]:
    """Return the canonical requested-parameter artifact block."""

    return {
        "snapshot_schema_version": CONTACT_ENVIRONMENT_SNAPSHOT_SCHEMA_VERSION,
        "kind": "requested",
        **spec.as_config(),
    }


def _validate_geom_id(model: mujoco.MjModel, geom_id: int) -> int:
    if isinstance(geom_id, (bool, np.bool_)):
        raise ValueError("cube_geom_id must be an integer")
    resolved = int(geom_id)
    if resolved != geom_id or not 0 <= resolved < model.ngeom:
        raise ValueError("cube_geom_id is outside the compiled model")
    return resolved


def compiled_environment_snapshot(
    model: mujoco.MjModel, cube_geom_id: int
) -> dict[str, Any]:
    """Record actual compiled contact and solver values from ``model``."""

    geom_id = _validate_geom_id(model, cube_geom_id)
    payload = {
        "schema_version": CONTACT_ENVIRONMENT_SCHEMA_VERSION,
        "contact": {
            "sliding_friction": float(model.geom_friction[geom_id, 0]),
            "torsional_friction": float(model.geom_friction[geom_id, 1]),
            "rolling_friction": float(model.geom_friction[geom_id, 2]),
            "condim": int(model.geom_condim[geom_id]),
            "priority": int(model.geom_priority[geom_id]),
            "solref": model.geom_solref[geom_id].astype(float).tolist(),
            "solimp": model.geom_solimp[geom_id].astype(float).tolist(),
        },
        "solver": {
            "cone": _enum_name(model.opt.cone, _VALUE_TO_CONE, "cone"),
            "solver": _enum_name(model.opt.solver, _VALUE_TO_SOLVER, "solver"),
            "impratio": float(model.opt.impratio),
            "tolerance": float(model.opt.tolerance),
            "iterations": int(model.opt.iterations),
            "ls_iterations": int(model.opt.ls_iterations),
            "noslip_iterations": int(model.opt.noslip_iterations),
            "noslip_tolerance": float(model.opt.noslip_tolerance),
        },
        "required_model": {
            "timestep_s": float(model.opt.timestep),
            "integrator": _enum_name(
                model.opt.integrator, _VALUE_TO_INTEGRATOR, "integrator"
            ),
            "gravity_m_s2": np.asarray(model.opt.gravity, dtype=float).tolist(),
        },
    }
    return {
        "snapshot_schema_version": CONTACT_ENVIRONMENT_SNAPSHOT_SCHEMA_VERSION,
        "kind": "compiled",
        "geom_id": geom_id,
        "geom_name": model.geom(geom_id).name,
        **payload,
        "environment_id": canonical_environment_id(payload),
    }


def _environment_payload_from_snapshot(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value[key]
        for key in ("schema_version", "contact", "solver", "required_model")
    }


def verify_compiled_environment(
    spec: ContactEnvironmentSpec, compiled: Mapping[str, Any]
) -> None:
    """Fail closed unless a compiled snapshot exactly realizes ``spec``."""

    if compiled.get("snapshot_schema_version") != (
        CONTACT_ENVIRONMENT_SNAPSHOT_SCHEMA_VERSION
    ) or compiled.get("kind") != "compiled":
        raise RuntimeError("invalid compiled MuJoCo environment snapshot contract")
    actual = _environment_payload_from_snapshot(compiled)
    expected = spec._payload()
    mismatches = _mapping_differences(expected, actual)
    if mismatches:
        raise RuntimeError(
            "compiled MuJoCo environment differs from request: "
            + ", ".join(mismatches)
        )
    if compiled.get("environment_id") != spec.environment_id:
        raise RuntimeError("compiled MuJoCo environment_id differs from request")


def _verify_required_model(model: mujoco.MjModel, spec: ContactEnvironmentSpec) -> None:
    actual_integrator = _enum_name(
        model.opt.integrator, _VALUE_TO_INTEGRATOR, "integrator"
    )
    errors: list[str] = []
    if not math.isclose(
        float(model.opt.timestep), spec.required_timestep_s, rel_tol=0.0, abs_tol=1e-15
    ):
        errors.append("timestep")
    if actual_integrator != spec.required_integrator:
        errors.append("integrator")
    if not np.array_equal(
        np.asarray(model.opt.gravity, dtype=np.float64),
        np.asarray(spec.required_gravity_m_s2, dtype=np.float64),
    ):
        errors.append("gravity")
    if errors:
        raise ValueError(
            "compiled model violates required contact-environment invariants: "
            + ", ".join(errors)
        )


def apply_to_model(
    model: mujoco.MjModel,
    cube_geom_id: int,
    spec: ContactEnvironmentSpec,
) -> dict[str, Any]:
    """Apply only declared contact/solver fields and return compiled evidence."""

    if not isinstance(spec, ContactEnvironmentSpec):
        raise TypeError("spec must be a ContactEnvironmentSpec")
    geom_id = _validate_geom_id(model, cube_geom_id)
    _verify_required_model(model, spec)

    model.geom_friction[geom_id] = np.asarray(
        [spec.sliding_friction, spec.torsional_friction, spec.rolling_friction],
        dtype=np.float64,
    )
    model.geom_condim[geom_id] = spec.condim
    model.geom_priority[geom_id] = spec.priority
    model.geom_solref[geom_id] = np.asarray(spec.solref, dtype=np.float64)
    model.geom_solimp[geom_id] = np.asarray(spec.solimp, dtype=np.float64)
    model.opt.cone = _CONE_TO_VALUE[spec.cone]
    model.opt.solver = _SOLVER_TO_VALUE[spec.solver]
    model.opt.impratio = spec.impratio
    model.opt.tolerance = spec.tolerance
    model.opt.iterations = spec.iterations
    model.opt.ls_iterations = spec.ls_iterations
    model.opt.noslip_iterations = spec.noslip_iterations
    model.opt.noslip_tolerance = spec.noslip_tolerance

    compiled = compiled_environment_snapshot(model, geom_id)
    verify_compiled_environment(spec, compiled)
    return compiled


def runtime_cube_contact_snapshot(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cube_geom_id: int,
) -> dict[str, Any]:
    """Capture effective ``mjContact`` parameters involving the cube geom."""

    geom_id = _validate_geom_id(model, cube_geom_id)
    records: list[dict[str, Any]] = []
    for contact_index, contact in enumerate(data.contact[: data.ncon]):
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if geom_id not in (geom1, geom2):
            continue
        other = geom2 if geom1 == geom_id else geom1
        records.append(
            {
                "contact_index": contact_index,
                "geom1": geom1,
                "geom2": geom2,
                "other_geom_id": other,
                "other_geom_name": model.geom(other).name,
                "active": int(contact.efc_address) >= 0,
                "distance_m": float(contact.dist),
                "dim": int(contact.dim),
                "friction": np.asarray(contact.friction, dtype=float).tolist(),
                "solref": np.asarray(contact.solref, dtype=float).tolist(),
                "solimp": np.asarray(contact.solimp, dtype=float).tolist(),
            }
        )
    payload = {
        "runtime_contact_snapshot_schema_version": 1,
        "cube_geom_id": geom_id,
        "cube_geom_name": model.geom(geom_id).name,
        "contact_count": len(records),
        "contacts": records,
    }
    payload["snapshot_id"] = canonical_sha256(payload)
    return payload


def verify_runtime_cube_contacts(
    spec: ContactEnvironmentSpec,
    snapshot: Mapping[str, Any],
    *,
    require_contacts: bool = True,
) -> None:
    """Verify effective contact parameters, including MuJoCo's 3-to-5 map."""

    payload = dict(snapshot)
    declared_snapshot_id = payload.pop("snapshot_id", None)
    if declared_snapshot_id != canonical_sha256(payload):
        raise RuntimeError("runtime contact snapshot_id does not match its content")
    contacts = snapshot.get("contacts")
    if not isinstance(contacts, list):
        raise RuntimeError("runtime contact snapshot is missing contacts")
    if snapshot.get("contact_count") != len(contacts):
        raise RuntimeError("runtime contact snapshot count is inconsistent")
    if require_contacts and not contacts:
        raise RuntimeError("runtime contact verification requires cube contacts")
    expected_friction = np.asarray(
        [
            spec.sliding_friction,
            spec.sliding_friction,
            spec.torsional_friction,
            spec.rolling_friction,
            spec.rolling_friction,
        ],
        dtype=np.float64,
    )
    for index, contact in enumerate(contacts):
        if int(contact["dim"]) != spec.condim:
            raise RuntimeError(f"runtime cube contact {index} has wrong condim")
        if not np.allclose(
            np.asarray(contact["friction"], dtype=np.float64),
            expected_friction,
            rtol=0.0,
            atol=1e-15,
        ):
            raise RuntimeError(f"runtime cube contact {index} has wrong friction")
        if not np.allclose(
            np.asarray(contact["solref"], dtype=np.float64),
            np.asarray(spec.solref, dtype=np.float64),
            rtol=0.0,
            atol=1e-15,
        ):
            raise RuntimeError(f"runtime cube contact {index} has wrong solref")
        if not np.allclose(
            np.asarray(contact["solimp"], dtype=np.float64),
            np.asarray(spec.solimp, dtype=np.float64),
            rtol=0.0,
            atol=1e-15,
        ):
            raise RuntimeError(f"runtime cube contact {index} has wrong solimp")


def _fingerprint_array(name: str, value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(
        (
            f"{name}\0{array.dtype.str}\0"
            + ",".join(str(item) for item in array.shape)
            + "\0"
        ).encode("utf-8")
    )
    if array.dtype.hasobject:
        digest.update(repr(array.tolist()).encode("utf-8"))
    else:
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _fingerprint_scalar(name: str, value: Any) -> str:
    if isinstance(value, bytes):
        payload: Any = {"type": "bytes", "hex": value.hex()}
    elif isinstance(value, np.generic):
        payload = value.item()
    else:
        payload = value
    return canonical_sha256({"field": name, "value": payload})


def immutable_model_field_fingerprints(
    model: mujoco.MjModel, cube_geom_id: int
) -> dict[str, str]:
    """Hash all compiled model arrays after masking declared environment rows.

    This intentionally covers mesh/assets, bodies, inertias, joints, actuators,
    sensors and derived constants, rather than maintaining a fragile short
    whitelist of fields that are assumed to matter.
    """

    geom_id = _validate_geom_id(model, cube_geom_id)
    result: dict[str, str] = {}
    for name in sorted(dir(model)):
        if name.startswith("_"):
            continue
        try:
            value = getattr(model, name)
        except (AttributeError, RuntimeError):
            continue
        if not isinstance(value, np.ndarray):
            continue
        array = np.asarray(value)
        if name in _CONTACT_ARRAY_NAMES:
            array = np.array(array, copy=True)
            array[geom_id] = 0
        result[f"model.{name}"] = _fingerprint_array(name, array)

    # Stable structural scalars and name/path buffers are part of the model
    # identity.  Memory sizes and the compiler's aggregate signature are not:
    # they may depend on allocation details or on an allowed contact field.
    for name in sorted(dir(model)):
        if not name.startswith("n") or name in {"narena", "nbuffer"}:
            continue
        try:
            value = getattr(model, name)
        except (AttributeError, RuntimeError):
            continue
        if isinstance(value, (int, np.integer)):
            result[f"model.{name}"] = _fingerprint_scalar(name, int(value))
    for name in ("names", "paths", "text_data"):
        value = getattr(model, name, None)
        if isinstance(value, bytes):
            result[f"model.{name}"] = _fingerprint_scalar(name, value)

    for name in sorted(dir(model.opt)):
        if name.startswith("_") or name in _SOLVER_OPTION_NAMES:
            continue
        try:
            value = getattr(model.opt, name)
        except (AttributeError, RuntimeError):
            continue
        if isinstance(value, np.ndarray):
            result[f"option.{name}"] = _fingerprint_array(name, value)
        elif isinstance(value, (bool, int, float, np.generic)):
            result[f"option.{name}"] = _fingerprint_scalar(name, value)
    return result


def _mapping_differences(expected: Any, actual: Any, prefix: str = "") -> list[str]:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        result: list[str] = []
        keys = sorted(set(expected) | set(actual))
        for key in keys:
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in expected or key not in actual:
                result.append(path)
            else:
                result.extend(_mapping_differences(expected[key], actual[key], path))
        return result
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        if len(expected) != len(actual):
            return [prefix]
        result = []
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            result.extend(_mapping_differences(left, right, f"{prefix}[{index}]"))
        return result
    if isinstance(expected, (float, np.floating)) or isinstance(
        actual, (float, np.floating)
    ):
        try:
            equal = math.isclose(
                float(expected), float(actual), rel_tol=0.0, abs_tol=1e-15
            )
        except (TypeError, ValueError):
            equal = False
    else:
        equal = expected == actual
    return [] if equal else [prefix]


def _environment_field_values(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    contact = snapshot["contact"]
    solver = snapshot["solver"]
    assert isinstance(contact, Mapping) and isinstance(solver, Mapping)
    return {
        "contact.sliding_friction": contact["sliding_friction"],
        "contact.torsional_friction": contact["torsional_friction"],
        "contact.rolling_friction": contact["rolling_friction"],
        "contact.condim": contact["condim"],
        "contact.priority": contact["priority"],
        "contact.solref": contact["solref"],
        "contact.solimp": contact["solimp"],
        "solver.cone": solver["cone"],
        "solver.solver": solver["solver"],
        "solver.impratio": solver["impratio"],
        "solver.tolerance": solver["tolerance"],
        "solver.iterations": solver["iterations"],
        "solver.ls_iterations": solver["ls_iterations"],
        "solver.noslip_iterations": solver["noslip_iterations"],
        "solver.noslip_tolerance": solver["noslip_tolerance"],
    }


@dataclass(frozen=True, slots=True)
class AllowedModelChangeAudit:
    passed: bool
    allowed_environment_fields: tuple[str, ...]
    changed_environment_fields: tuple[str, ...]
    unexpected_environment_fields: tuple[str, ...]
    unexpected_model_fields: tuple[str, ...]
    reference_environment_id: str
    candidate_environment_id: str
    reference_immutable_model_sha256: str
    candidate_immutable_model_sha256: str

    def as_mapping(self) -> dict[str, Any]:
        payload = {
            "model_change_audit_schema_version": MODEL_CHANGE_AUDIT_SCHEMA_VERSION,
            "passed": self.passed,
            "allowed_environment_fields": list(self.allowed_environment_fields),
            "changed_environment_fields": list(self.changed_environment_fields),
            "unexpected_environment_fields": list(
                self.unexpected_environment_fields
            ),
            "unexpected_model_fields": list(self.unexpected_model_fields),
            "reference_environment_id": self.reference_environment_id,
            "candidate_environment_id": self.candidate_environment_id,
            "reference_immutable_model_sha256": (
                self.reference_immutable_model_sha256
            ),
            "candidate_immutable_model_sha256": (
                self.candidate_immutable_model_sha256
            ),
        }
        payload["audit_id"] = canonical_sha256(payload)
        return payload

    def assert_passed(self) -> None:
        if not self.passed:
            details = [
                *self.unexpected_environment_fields,
                *self.unexpected_model_fields,
            ]
            raise RuntimeError(
                "MuJoCo environment changed outside the declared allow-list: "
                + ", ".join(details)
            )


def audit_allowed_model_changes(
    reference_model: mujoco.MjModel,
    candidate_model: mujoco.MjModel,
    reference_cube_geom_id: int,
    *,
    candidate_cube_geom_id: int | None = None,
    allowed_environment_fields: Iterable[str] = (),
) -> AllowedModelChangeAudit:
    """Audit a candidate model against a reference with an explicit allow-list."""

    reference_geom = _validate_geom_id(reference_model, reference_cube_geom_id)
    candidate_geom = _validate_geom_id(
        candidate_model,
        reference_geom if candidate_cube_geom_id is None else candidate_cube_geom_id,
    )
    allowed = tuple(sorted(set(allowed_environment_fields)))
    unknown = set(allowed) - CONTACT_ENVIRONMENT_FIELD_PATHS
    if unknown:
        raise ValueError(
            "unknown allowed contact-environment fields: " + ", ".join(sorted(unknown))
        )

    reference_environment = compiled_environment_snapshot(
        reference_model, reference_geom
    )
    candidate_environment = compiled_environment_snapshot(candidate_model, candidate_geom)
    reference_values = _environment_field_values(reference_environment)
    candidate_values = _environment_field_values(candidate_environment)
    changed = tuple(
        sorted(
            key
            for key in CONTACT_ENVIRONMENT_FIELD_PATHS
            if _mapping_differences(reference_values[key], candidate_values[key])
        )
    )
    unexpected_environment = tuple(sorted(set(changed) - set(allowed)))

    reference_fields = immutable_model_field_fingerprints(
        reference_model, reference_geom
    )
    candidate_fields = immutable_model_field_fingerprints(
        candidate_model, candidate_geom
    )
    unexpected_model = tuple(
        sorted(
            name
            for name in set(reference_fields) | set(candidate_fields)
            if reference_fields.get(name) != candidate_fields.get(name)
        )
    )
    reference_digest = canonical_sha256(reference_fields)
    candidate_digest = canonical_sha256(candidate_fields)
    return AllowedModelChangeAudit(
        passed=not unexpected_environment and not unexpected_model,
        allowed_environment_fields=allowed,
        changed_environment_fields=changed,
        unexpected_environment_fields=unexpected_environment,
        unexpected_model_fields=unexpected_model,
        reference_environment_id=str(reference_environment["environment_id"]),
        candidate_environment_id=str(candidate_environment["environment_id"]),
        reference_immutable_model_sha256=reference_digest,
        candidate_immutable_model_sha256=candidate_digest,
    )


__all__ = [
    "CONTACT_ENVIRONMENT_FIELD_PATHS",
    "CONTACT_ENVIRONMENT_SCHEMA_VERSION",
    "AllowedModelChangeAudit",
    "ContactEnvironmentSpec",
    "apply_to_model",
    "audit_allowed_model_changes",
    "canonical_environment_id",
    "compiled_environment_snapshot",
    "immutable_model_field_fingerprints",
    "requested_environment_snapshot",
    "runtime_cube_contact_snapshot",
    "verify_compiled_environment",
    "verify_runtime_cube_contacts",
]
