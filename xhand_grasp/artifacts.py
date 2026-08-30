"""Deterministic artifact serialization and provenance helpers."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Literal

import mujoco
import numpy as np

from .experiment import resolve_experiment


REPO_ROOT = Path(__file__).resolve().parent.parent


def default_artifact_path(
    config: dict[str, Any], command: Literal["run", "tune", "robustness"]
) -> Path:
    """Resolve one CLI's default output from the registered experiment.

    Schema-v1 historically calls its single-run directory ``run``.  Versioned
    experiments call the corresponding directory ``nominal``; retaining that
    small distinction keeps both public layouts stable while allowing new
    experiments to select an independent root in their registry definition.
    """

    definition = resolve_experiment(config)
    if definition.artifact_root is None:
        raise ValueError(
            f"experiment {definition.experiment_id!r} has no default artifact root"
        )
    root = Path(definition.artifact_root)
    if command == "run":
        leaf = "run" if int(config.get("schema_version", 1)) == 1 else "nominal"
    elif command == "tune":
        leaf = "tune"
    elif command == "robustness":
        leaf = "robustness.json"
    else:  # pragma: no cover - protected by the Literal annotation for callers.
        raise ValueError(f"unknown artifact command: {command!r}")
    return root / leaf


def resolved_run_config(
    config: dict[str, Any], summary: dict[str, Any]
) -> dict[str, Any]:
    """Return the configuration as resolved by an observed simulation run.

    A schema-v2 input may intentionally describe an initial or best near miss.
    Copying that status into a run artifact would make the embedded config
    disagree with the adjacent summary.  The physical inputs remain unchanged,
    while the status is derived solely from the just-computed hard checks.
    """

    resolved = copy.deepcopy(config)
    if int(resolved.get("schema_version", 1)) < 2:
        return resolved

    hard_constraints_passed = bool(summary["passed"])
    failed_checks = [str(name) for name in summary.get("failed_checks", [])]
    input_status = config.get("experiment_status")
    classification = "validated_run" if hard_constraints_passed else "failed_run"
    note = (
        "This run passed all declared hard constraints."
        if hard_constraints_passed
        else "This run did not pass all declared hard constraints."
    )
    definition = resolve_experiment(config)
    campaign = definition.size_campaign
    aligned_campaign = definition.aligned_contact_campaign
    far_hand_campaign = definition.far_hand_campaign
    normal_aligned_campaign = definition.normal_aligned_smooth_lift_campaign
    contact_point_campaign = definition.contact_point_search
    run_context = config.get("run_context")
    run_context_kind = (
        str(run_context.get("kind")) if isinstance(run_context, dict) else None
    )
    constant_density_passed = False
    fixed_mass_discovery_passed = False
    campaign_validated = hard_constraints_passed
    if campaign is not None:
        edge_m = float(config["cube"]["edge_m"])
        mass_kg = float(config["cube"]["mass_kg"])
        density_mass = campaign.constant_density_mass_kg(edge_m)
        is_constant_density = math.isclose(
            mass_kg, density_mass, rel_tol=0.0, abs_tol=1e-12
        )
        is_fixed_mass = math.isclose(
            mass_kg,
            campaign.discovery_mass_kg,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        is_nominal_friction = math.isclose(
            float(config["cube"]["friction"]),
            campaign.discovery_friction,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        constant_density_passed = bool(
            hard_constraints_passed and is_constant_density and is_nominal_friction
        )
        fixed_mass_discovery_passed = bool(
            hard_constraints_passed and is_fixed_mass and is_nominal_friction
        )
        campaign_validated = constant_density_passed
        if hard_constraints_passed:
            if is_constant_density and is_nominal_friction:
                classification = "validated_constant_density"
                note = (
                    "This run passed all declared hard constraints at the "
                    "campaign's reference density."
                )
            elif is_fixed_mass and is_nominal_friction:
                classification = "validated_fixed_mass_ablation"
                note = (
                    "This fixed-mass geometry ablation passed all hard constraints; "
                    "campaign validation still requires a constant-density pass."
                )
            else:
                classification = "validated_noncanonical_material"
                note = (
                    "This noncanonical material run passed all hard constraints but "
                    "does not establish constant-density campaign validation."
                )
    if aligned_campaign is not None:
        edge_m = float(config["cube"]["edge_m"])
        mass_kg = float(config["cube"]["mass_kg"])
        edge_is_declared = any(
            math.isclose(edge_m, declared, rel_tol=0.0, abs_tol=1e-12)
            for declared in aligned_campaign.edges_m
        )
        mass_is_constant_density = math.isclose(
            mass_kg,
            aligned_campaign.constant_density_mass_kg(edge_m),
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
        friction_is_nominal = math.isclose(
            float(config["cube"]["friction"]),
            aligned_campaign.friction,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        canonical_material = bool(
            edge_is_declared
            and mass_is_constant_density
            and friction_is_nominal
        )
        constant_density_passed = bool(
            hard_constraints_passed and canonical_material
        )
        campaign_validated = bool(
            constant_density_passed and run_context_kind is None
        )
        if run_context_kind == "parameter_override_run":
            classification = "parameter_override_run"
            note = (
                "This parameter-override result was recomputed from the actual "
                "simulation and inherits no catalog validation claim."
            )
        elif run_context_kind == "robustness_trial":
            classification = "robustness_trial"
            note = (
                "This is one independently evaluated aligned-contact robustness "
                "trial, not a nominal campaign-validation claim."
            )
        elif hard_constraints_passed and canonical_material:
            classification = "validated_aligned_contacts_nominal"
            note = (
                "This canonical aligned-contact run passed all declared hard "
                "constraints; robustness remains a separate 50-case result."
            )
        elif hard_constraints_passed:
            classification = "validated_noncanonical_aligned_contact_run"
            note = (
                "This run passed the hard constraints but is outside the "
                "registered aligned-contact material campaign."
            )
    far_hand_nominal_material = False
    far_hand_nominal_passed = False
    if far_hand_campaign is not None:
        edge_is_nominal = math.isclose(
            float(config["cube"]["edge_m"]),
            far_hand_campaign.nominal_edge_m,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        mass_is_nominal = math.isclose(
            float(config["cube"]["mass_kg"]),
            far_hand_campaign.nominal_mass_kg,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        friction_is_nominal = math.isclose(
            float(config["cube"]["friction"]),
            far_hand_campaign.friction,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        far_hand_nominal_material = bool(
            edge_is_nominal and mass_is_nominal and friction_is_nominal
        )
        far_hand_nominal_passed = bool(
            hard_constraints_passed
            and far_hand_nominal_material
            and run_context_kind is None
        )
        campaign_validated = far_hand_nominal_passed
        if run_context_kind == "parameter_override_run":
            classification = "parameter_override_run"
            note = (
                "This parameter-override result was recomputed from the actual "
                "simulation and inherits no far-hand fingertip validation claim."
            )
        elif run_context_kind == "robustness_trial":
            classification = "robustness_trial"
            note = (
                "This is one independently evaluated far-hand fingertip robustness "
                "trial, not a nominal campaign-validation claim."
            )
        elif hard_constraints_passed and far_hand_nominal_material:
            classification = "validated_far_hand_fingertip_nominal"
            note = (
                "This canonical 60 mm, 160 g, friction-0.8 far-hand fingertip "
                "run passed all declared hard constraints; robustness remains "
                "a separate 50-case result."
            )
        elif hard_constraints_passed:
            classification = "validated_noncanonical_far_hand_fingertip_run"
            note = (
                "This far-hand fingertip run passed the hard constraints but is "
                "outside the registered 60 mm, 160 g, friction-0.8 nominal case."
            )
    normal_aligned_nominal_material = False
    normal_aligned_nominal_passed = False
    if normal_aligned_campaign is not None:
        edge_is_declared = any(
            math.isclose(
                float(config["cube"]["edge_m"]),
                edge,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for edge in normal_aligned_campaign.edges_m
        )
        mass_is_nominal = math.isclose(
            float(config["cube"]["mass_kg"]),
            normal_aligned_campaign.fixed_mass_kg,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        friction_is_nominal = math.isclose(
            float(config["cube"]["friction"]),
            normal_aligned_campaign.friction,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        normal_aligned_nominal_material = bool(
            edge_is_declared and mass_is_nominal and friction_is_nominal
        )
        normal_aligned_nominal_passed = bool(
            hard_constraints_passed
            and normal_aligned_nominal_material
            and run_context_kind is None
        )
        campaign_validated = normal_aligned_nominal_passed
        if run_context_kind == "parameter_override_run":
            classification = "parameter_override_run"
            note = (
                "This parameter-override result was recomputed from the actual "
                "simulation and inherits no normal-aligned smooth-lift claim."
            )
        elif run_context_kind == "robustness_trial":
            classification = "robustness_trial"
            note = (
                "This is one independently evaluated normal-aligned smooth-lift "
                "robustness trial, not a nominal campaign-validation claim."
            )
        elif hard_constraints_passed and normal_aligned_nominal_material:
            classification = "validated_normal_aligned_smooth_lift_nominal"
            note = (
                "This declared 60--70 mm, 160 g, friction-0.8 run passed the "
                "normal-alignment and smooth near-vertical lift constraints."
            )
        elif hard_constraints_passed:
            classification = "validated_noncanonical_normal_aligned_smooth_lift"
            note = (
                "This run passed all hard constraints but uses material or size "
                "parameters outside the registered schema-v8 campaign."
            )
    stage_status = summary.get("stage_status")
    grasp_success = False
    manipulation_success = False
    if int(resolved.get("schema_version", 1)) >= 3 and isinstance(
        stage_status, dict
    ):
        grasp_success = bool(stage_status.get("grasp_success", False))
        manipulation_success = bool(
            stage_status.get("manipulation_success", False)
        )
        preserve_context_classification = bool(
            (
                aligned_campaign is not None
                or far_hand_campaign is not None
                or normal_aligned_campaign is not None
            )
            and run_context_kind is not None
        )
        if not grasp_success and not preserve_context_classification:
            classification = "failed_grasp_acquisition"
            note = (
                "The controller did not acquire the declared continuous stable "
                "grasp, so no manipulation target was authorized."
            )
        elif (
            not manipulation_success
            and contact_point_campaign is None
            and not preserve_context_classification
        ):
            classification = "grasp_acquired_manipulation_failed"
            note = (
                "A stable grasp was acquired, but the authorized manipulation "
                "did not pass all declared hard constraints."
            )
        if contact_point_campaign is not None:
            actual_campaign = definition.actual_contact_grasp_pose_campaign
            assert actual_campaign is not None
            canonical_material = bool(
                any(
                    math.isclose(
                        float(config["cube"]["edge_m"]),
                        edge,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    for edge in actual_campaign.edges_m
                )
                and math.isclose(
                    float(config["cube"]["mass_kg"]),
                    actual_campaign.fixed_mass_kg,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and math.isclose(
                    float(config["cube"]["friction"]),
                    actual_campaign.friction,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )
            campaign_validated = bool(
                grasp_success
                and canonical_material
                and run_context_kind is None
            )
            labels = actual_campaign.validation_labels
            if run_context_kind == "parameter_override_run":
                classification = "parameter_override_run"
                note = (
                    "This parameter-override contact-point grasp was rerun and "
                    "re-evaluated; it inherits no catalog success claim."
                )
            elif run_context_kind == "robustness_trial":
                classification = "robustness_trial"
                note = (
                    "This is one independently evaluated contact-point grasp "
                    "robustness trial, not a nominal validation claim."
                )
            elif grasp_success and canonical_material:
                classification = (
                    labels["grasp"]
                    if labels is not None
                    else "validated_contact_point_grasp"
                )
                note = (
                    "The fixed-160-g contact-point grasp passed every registered "
                    "grasp hard check. Manipulation is diagnostic-only for this "
                    "experiment and is not required for success."
                )
            elif grasp_success:
                classification = "validated_noncanonical_contact_point_grasp"
                note = (
                    "The contact-point grasp passed every registered grasp hard "
                    "check outside the canonical fixed-160-g material setting."
                )
    status: dict[str, Any] = {
        "classification": classification,
        "passed": (
            grasp_success
            if contact_point_campaign is not None and run_context_kind is not None
            else hard_constraints_passed
            if (
                aligned_campaign is not None
                or far_hand_campaign is not None
                or normal_aligned_campaign is not None
            )
            and run_context_kind is not None
            else campaign_validated
        ),
        "hard_constraints_passed": (
            grasp_success
            if contact_point_campaign is not None
            else hard_constraints_passed
        ),
        "failed_checks": failed_checks,
        "note": note,
    }
    if campaign is not None:
        status.update(
            {
                "campaign_validated": campaign_validated,
                "constant_density_passed": constant_density_passed,
                "fixed_mass_discovery_passed": fixed_mass_discovery_passed,
            }
        )
    if aligned_campaign is not None:
        status.update(
            {
                "campaign_validated": campaign_validated,
                "constant_density_passed": constant_density_passed,
                "robustness_passed": False,
                "run_context": run_context_kind,
            }
        )
    if far_hand_campaign is not None:
        status.update(
            {
                "campaign_validated": campaign_validated,
                "canonical_nominal_material": far_hand_nominal_material,
                "canonical_nominal_passed": far_hand_nominal_passed,
                "robustness_passed": False,
                "run_context": run_context_kind,
            }
        )
    if normal_aligned_campaign is not None:
        status.update(
            {
                "campaign_validated": campaign_validated,
                "canonical_nominal_material": normal_aligned_nominal_material,
                "canonical_nominal_passed": normal_aligned_nominal_passed,
                "robustness_passed": False,
                "run_context": run_context_kind,
            }
        )
    if contact_point_campaign is not None:
        status.update(
            {
                "campaign_validated": campaign_validated,
                "success_scope": "grasp_only_contact_point_hard_checks",
                "manipulation_success_required": False,
                "full_hard_constraints_passed": hard_constraints_passed,
                "robustness_passed": False,
                "run_context": run_context_kind,
            }
        )
    if isinstance(stage_status, dict):
        status["grasp_success"] = bool(stage_status.get("grasp_success", False))
        status["manipulation_success"] = bool(
            stage_status.get("manipulation_success", False)
        )
        status["full_success"] = bool(stage_status.get("full_success", False))
    if isinstance(input_status, dict) and input_status.get("classification"):
        status["input_classification"] = str(input_status["classification"])
    resolved["experiment_status"] = status
    return resolved


def json_compatible(value: Any) -> Any:
    """Return a strict-JSON representation, replacing non-finite floats by null."""

    if isinstance(value, np.generic):
        return json_compatible(value.item())
    if isinstance(value, np.ndarray):
        return json_compatible(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    return value


def json_text(value: Any) -> str:
    return json.dumps(
        json_compatible(value),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )


def write_json(path: str | Path, value: Any) -> None:
    """Atomically write strict, stable JSON with mode 0644."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    content = json_text(value) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent, prefix=f".{output.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.chmod(0o644)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_source_sha256(paths: Iterable[str | Path]) -> str:
    """Hash sorted path names and contents so provenance covers all modules."""

    digest = hashlib.sha256()
    resolved = sorted((Path(path).resolve() for path in paths), key=lambda path: str(path))
    for path in resolved:
        try:
            relative = path.relative_to(REPO_ROOT)
        except ValueError:
            relative = path
        digest.update(str(relative).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def implementation_paths() -> list[Path]:
    return sorted((REPO_ROOT / "xhand_grasp").rglob("*.py"))


def run_metadata(config_path: Path) -> dict[str, Any]:
    """Collect reproducibility metadata without mutating repository state."""

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status_lines = subprocess.run(
            ["git", "status", "--short", "--untracked-files=all"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
        branch = "unknown"
        status_lines = []

    uv_version = subprocess.run(
        [str(REPO_ROOT / ".tools" / "uv-0.12.5" / "uv"), "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    ffmpeg_version = subprocess.run(
        ["ffmpeg", "-version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()[0]
    facade = REPO_ROOT / "grasp_cube.py"
    sources = implementation_paths()
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "uv": uv_version,
        "ffmpeg": ffmpeg_version,
        "platform": platform.platform(),
        "repo_commit": commit,
        "repo_branch": branch,
        "repo_dirty": bool(status_lines),
        "repo_status_short": status_lines,
        # Keep the historical key while also covering the modular implementation.
        "experiment_sha256": file_sha256(facade),
        "implementation_sha256": aggregate_source_sha256(sources),
        "implementation_files": [str(path.relative_to(REPO_ROOT)) for path in sources],
        "model_sha256": file_sha256(REPO_ROOT / "xhand_left.xml"),
        "pyproject_sha256": file_sha256(REPO_ROOT / "pyproject.toml"),
        "verify_xhand_sha256": file_sha256(REPO_ROOT / "verify_xhand.py"),
        "uv_lock_sha256": file_sha256(REPO_ROOT / "uv.lock"),
        "config_sha256": file_sha256(config_path),
        "mujoco_gl": os.environ.get("MUJOCO_GL", "default"),
    }
