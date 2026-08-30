"""Deterministic left-hand, three-finger cube-lift experiment for XHAND1."""

from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
from pathlib import Path
from typing import Literal

from .actual_contact_capability import (
    LEGACY_ACTUAL_CONTACT_EXPERIMENT_ID,
    is_actual_contact_definition,
    is_contact_preserving_planned_lift_definition,
    is_joint_pair_near_zero_contact_preserving_planned_lift_definition,
    resolve_joint_pair_near_zero_contact_preserving_planned_lift_definition,
)
from .aligned_trajectory_catalog import export_aligned_trajectory_catalog
from .artifacts import (
    REPO_ROOT,
    default_artifact_path,
    file_sha256,
    json_text,
    resolved_run_config,
    run_metadata,
    write_json,
)
from .config import DEFAULT_CONFIG, load_config, validate_config
from .contact_environment import ContactEnvironmentSpec
from .experiment import resolve_experiment
from .grasp_campaign_manifest import load_grasp_campaign_manifest
from .grasp_trajectory_catalog import export_grasp_trajectory_catalog
from .search import (
    normalized_acceptance_margins,
    robustness,
    robustness_cases,
    tune,
)
from .simulation import run_simulation
from .trajectory import preflight_config
from .trajectory_catalog import export_trajectory_catalog
from .viewer import (
    apply_viewer_overrides,
    discover_measured_viewer_data,
    format_measured_viewer_data,
    parse_actuator_overrides,
    replay_in_viewer,
    ReplaySource,
    resolve_measured_viewer_source,
    resolve_replay_source,
    resolve_viewer_source,
    simulate_in_viewer,
)


LEGACY_RUN_OUTPUT = "artifacts/left_three_finger_cube/run"
LEGACY_TUNE_OUTPUT = "artifacts/left_three_finger_cube/tune"
LEGACY_ROBUSTNESS_OUTPUT = "artifacts/left_three_finger_cube/robustness.json"

_ALIGNED_CATALOG_SUMMARY_FIELDS = (
    "trajectory_catalog_schema_version",
    "experiment_id",
    "declared_tilt_bands_deg",
    "published_tilt_bands_deg",
    "missing_published_tilt_bands_deg",
    "passing_tilt_bands_deg",
    "missing_passing_tilt_bands_deg",
    "trajectory_count",
    "passing_trajectory_count",
    "near_miss_trajectory_count",
    "reported_hard_pass_count",
    "all_selected_passes_reproduced",
    "all_reruns_full_success",
    "all_declared_bands_passed",
    "campaign_has_passing_trajectory",
)


def _resolved_default_output(
    config: dict,
    output_value: str | None,
    legacy_default: str,
    command: Literal["run", "tune", "robustness"],
) -> str:
    """Resolve an omitted output while preserving every explicit path."""

    del legacy_default  # Retained in the private signature for caller compatibility.
    if output_value is not None:
        return output_value
    return str(default_artifact_path(config, command))


def _load_run_config(args: argparse.Namespace) -> tuple[Path, dict, str | None]:
    """Load either a config file or a robustness result's hardest config."""

    hardest_from = getattr(args, "hardest_from", None)
    if hardest_from is None:
        source_path = Path(args.config).resolve()
        return source_path, load_config(source_path), None

    source_path = Path(hardest_from).resolve()
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    config = (
        payload.get("hardest_passing_config")
        if isinstance(payload, dict)
        else None
    )
    if not isinstance(config, dict):
        raise ValueError(
            f"robustness result has no usable hardest_passing_config: {source_path}"
        )
    return source_path, copy.deepcopy(config), "hardest_passing_config"


def _hardest_passing_config(config: dict, result: dict) -> dict | None:
    """Materialize the full runnable config for a reported grid boundary."""

    hardest = result.get("hardest_passing_grid_case")
    if not isinstance(hardest, dict):
        return None
    grid, _ = robustness_cases(config, int(result["seed"]))
    grid_index = int(hardest["grid_index"])
    if not 0 <= grid_index < len(grid):
        raise ValueError(f"hardest grid_index is out of range: {grid_index}")
    # Select the exact generated case rather than reconstructing it from the
    # human-readable, rounded millimetre/gram summary fields.
    resolved = grid[grid_index]
    runnable = resolved_run_config(
        resolved,
        {
            "passed": bool(hardest["passed"]),
            "failed_checks": hardest.get("failed_checks", []),
        },
    )
    status = runnable.get("experiment_status")
    if isinstance(status, dict) and hardest.get("case_family") is not None:
        status["robustness_case_family"] = str(hardest["case_family"])
        status["density_scale"] = hardest.get("density_scale")
        status["grid_index"] = grid_index
    return runnable


def _normalized_margin_fields(config: dict, summary: dict) -> dict:
    """Derive a complete quantitative margin report from one actual summary."""

    # Worker failures intentionally carry only a small diagnostic metric set.
    # They must remain serializable robustness failures instead of turning the
    # reporting pass into a second exception while looking up absent metrics.
    # ``None`` is preferable to an arbitrary finite sentinel: no quantitative
    # acceptance distance exists when the simulation itself did not complete.
    if "simulation_error" in summary.get("failed_checks", ()):
        return {
            "normalized_acceptance_margins": {},
            "minimum_normalized_acceptance_margin": None,
            "limiting_metric": "simulation_error",
        }

    margin_kwargs = {"contact_topology": config.get("contact_topology")}
    if config.get("contact_alignment") is not None:
        margin_kwargs["contact_alignment"] = config["contact_alignment"]
    if config.get("pose_constraints") is not None:
        margin_kwargs["pose_constraints"] = config["pose_constraints"]
    margins = normalized_acceptance_margins(
        summary["metrics"],
        config["acceptance"],
        **margin_kwargs,
    )
    limiting_metric = min(margins, key=margins.get)
    return {
        "normalized_acceptance_margins": margins,
        "minimum_normalized_acceptance_margin": margins[limiting_metric],
        "limiting_metric": limiting_metric,
    }


def _add_v2_robustness_margin_fields(config: dict, result: dict) -> None:
    """Attach full margins to every emitted v2 robustness observation."""

    nominal_fields = _normalized_margin_fields(config, result["nominal_summary"])
    result["nominal_normalized_acceptance_margins"] = nominal_fields[
        "normalized_acceptance_margins"
    ]
    result["nominal_minimum_normalized_acceptance_margin"] = nominal_fields[
        "minimum_normalized_acceptance_margin"
    ]
    result["nominal_limiting_metric"] = nominal_fields["limiting_metric"]
    for record in result["grid"]:
        fields = _normalized_margin_fields(config, record)
        record.update(fields)
        # Retain the historical singular field as an alias.  It is now also
        # populated for failed cases instead of being left null.
        record["normalized_acceptance_margin"] = fields[
            "minimum_normalized_acceptance_margin"
        ]
    for record in result["perturbations"]:
        record.update(_normalized_margin_fields(config, record))


def _aligned_catalog_summary(catalog: dict) -> dict:
    """Return the bounded catalog status embedded in a tune artifact."""

    missing = [
        key for key in _ALIGNED_CATALOG_SUMMARY_FIELDS if key not in catalog
    ]
    if missing:
        raise RuntimeError(
            "aligned trajectory publisher returned an incomplete catalog: "
            + ", ".join(missing)
        )
    return {
        key: copy.deepcopy(catalog[key])
        for key in _ALIGNED_CATALOG_SUMMARY_FIELDS
    }


def command_run(args: argparse.Namespace) -> int:
    source_config_path, config, source_config_selector = _load_run_config(args)
    source_config_sha256 = file_sha256(source_config_path)
    parameter_overridden = any(
        value is not None for value in (args.edge_mm, args.mass_g, args.friction)
    )
    if args.edge_mm is not None:
        config["cube"]["edge_m"] = float(args.edge_mm) / 1000.0
    if args.mass_g is not None:
        config["cube"]["mass_kg"] = float(args.mass_g) / 1000.0
    if args.friction is not None:
        config["cube"]["friction"] = float(args.friction)
    if int(config.get("schema_version", 1)) >= 4 and parameter_overridden:
        # A v4 override is a fresh experiment, not evidence inherited from a
        # canonical search/catalog config.  Validation permits the physical
        # change while the evaluator still enforces every hard criterion.
        config.pop("experiment_status", None)
        config.pop("candidate_metadata", None)
        config["run_context"] = {"kind": "parameter_override_run"}
    validate_config(config)
    output_value = _resolved_default_output(
        config, args.output_dir, LEGACY_RUN_OUTPUT, "run"
    )
    output_dir = Path(output_value).resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"output directory already exists: {output_dir}; choose a new --output-dir"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    video_filename = Path(args.video_filename)
    if (
        video_filename.name != args.video_filename
        or video_filename.suffix.lower() != ".mp4"
    ):
        raise ValueError("--video-filename must be a plain .mp4 filename")
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent, prefix=f".{output_dir.name}.staging."
    ) as staging_name:
        staging_dir = Path(staging_name)
        staged_config_path = staging_dir / "resolved_config.json"
        staged_video_path = staging_dir / video_filename if args.video else None
        staged_trace_path = staging_dir / "trace.npz" if not args.no_trace else None
        summary = run_simulation(
            config, trace_path=staged_trace_path, video_path=staged_video_path
        )
        resolved_config = resolved_run_config(config, summary)
        write_json(staged_config_path, resolved_config)
        metadata = run_metadata(staged_config_path)
        metadata["source_config"] = str(source_config_path)
        metadata["source_config_sha256"] = source_config_sha256
        if source_config_selector is not None:
            metadata["source_config_selector"] = source_config_selector
        final_config_path = output_dir / "resolved_config.json"
        final_video_path = output_dir / video_filename if args.video else None
        final_trace_path = output_dir / "trace.npz" if not args.no_trace else None
        artifact_hashes = {"resolved_config": file_sha256(staged_config_path)}
        if staged_trace_path is not None:
            artifact_hashes["trace"] = file_sha256(staged_trace_path)
        if staged_video_path is not None:
            artifact_hashes["video"] = file_sha256(staged_video_path)
        result = {
            "metadata": metadata,
            "config": resolved_config,
            "summary": summary,
            "artifacts": {
                "resolved_config": final_config_path.name,
                "trace": final_trace_path.name if final_trace_path else None,
                "video": final_video_path.name if final_video_path else None,
                "sha256": artifact_hashes,
            },
        }
        if int(config.get("schema_version", 1)) >= 2:
            result["experiment_status"] = resolved_config["experiment_status"]
            result.update(_normalized_margin_fields(config, summary))
        write_json(staging_dir / "result.json", result)
        staging_dir.rename(output_dir)
    console_summary = copy.deepcopy(summary)
    if int(config.get("schema_version", 1)) >= 2:
        console_summary["experiment_status"] = result["experiment_status"]
    print(json_text(console_summary))
    return 0 if result.get("experiment_status", {}).get(
        "passed", summary["passed"]
    ) else 2


def command_tune(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    preflight_config(config)
    output_value = _resolved_default_output(
        config, args.output_dir, LEGACY_TUNE_OUTPUT, "tune"
    )
    output_dir = Path(output_value).resolve()
    definition = resolve_experiment(config)
    is_v14_planned_lift = is_contact_preserving_planned_lift_definition(
        definition
    )
    rescue_from = getattr(args, "rescue_from", None)
    event_rescue_from = getattr(args, "event_rescue_from", None)
    adaptive_event_rescue_from = getattr(args, "adaptive_event_rescue_from", None)
    force_debias_rescue_from = getattr(args, "force_debias_rescue_from", None)
    micro_jerk_rescue_from = getattr(args, "micro_jerk_rescue_from", None)
    contact_mode_pose_rescue_from = getattr(
        args, "contact_mode_pose_rescue_from", None
    )
    adaptive_pose_followup_from = getattr(
        args, "adaptive_pose_followup_from", None
    )
    reuse_refinement_from = getattr(args, "reuse_refinement_from", None)
    rescue_sources = {
        "--rescue-from": rescue_from,
        "--event-rescue-from": event_rescue_from,
        "--adaptive-event-rescue-from": adaptive_event_rescue_from,
        "--force-debias-rescue-from": force_debias_rescue_from,
        "--micro-jerk-rescue-from": micro_jerk_rescue_from,
        "--contact-mode-pose-rescue-from": contact_mode_pose_rescue_from,
        "--adaptive-pose-followup-from": adaptive_pose_followup_from,
    }
    selected_rescue_sources = [
        name for name, value in rescue_sources.items() if value is not None
    ]
    if len(selected_rescue_sources) > 1:
        raise ValueError(
            "--rescue-from, --event-rescue-from, --adaptive-event-rescue-from "
            "--force-debias-rescue-from, --micro-jerk-rescue-from and "
            "--contact-mode-pose-rescue-from and --adaptive-pose-followup-from are "
            "mutually exclusive"
        )
    if adaptive_pose_followup_from is not None:
        if not is_v14_planned_lift:
            raise ValueError(
                "--adaptive-pose-followup-from is available only for the "
                "registered schema-v14 contact-preserving planned-lift experiment"
            )
        if args.output_dir is None:
            raise ValueError(
                "--adaptive-pose-followup-from requires an explicit new --output-dir"
            )
        if reuse_refinement_from is not None:
            raise ValueError(
                "--reuse-refinement-from cannot be combined with "
                "--adaptive-pose-followup-from"
            )
        if bool(getattr(args, "evidence_grasp_anchor", ())):
            raise ValueError(
                "--adaptive-pose-followup-from cannot be combined with "
                "--evidence-grasp-anchor"
            )
        return _command_tune_contact_preserving_adaptive_pose_followup(
            args,
            config_path=config_path,
            output_dir=output_dir,
            experiment_id=definition.experiment_id,
        )
    if contact_mode_pose_rescue_from is not None:
        if not is_v14_planned_lift:
            raise ValueError(
                "--contact-mode-pose-rescue-from is available only for the "
                "registered schema-v14 contact-preserving planned-lift experiment"
            )
        if args.output_dir is None:
            raise ValueError(
                "--contact-mode-pose-rescue-from requires an explicit new --output-dir"
            )
        if reuse_refinement_from is not None:
            raise ValueError(
                "--reuse-refinement-from cannot be combined with "
                "--contact-mode-pose-rescue-from"
            )
        if bool(getattr(args, "evidence_grasp_anchor", ())):
            raise ValueError(
                "--contact-mode-pose-rescue-from cannot be combined with "
                "--evidence-grasp-anchor"
            )
        return _command_tune_contact_mode_pose_rescue(
            args,
            config_path=config_path,
            output_dir=output_dir,
            experiment_id=definition.experiment_id,
        )
    if micro_jerk_rescue_from is not None:
        if not is_v14_planned_lift:
            raise ValueError(
                "--micro-jerk-rescue-from is available only for the registered "
                "schema-v14 contact-preserving planned-lift experiment"
            )
        if args.output_dir is None:
            raise ValueError(
                "--micro-jerk-rescue-from requires an explicit new --output-dir"
            )
        if reuse_refinement_from is not None:
            raise ValueError(
                "--reuse-refinement-from cannot be combined with "
                "--micro-jerk-rescue-from"
            )
        if bool(getattr(args, "evidence_grasp_anchor", ())):
            raise ValueError(
                "--micro-jerk-rescue-from cannot be combined with "
                "--evidence-grasp-anchor"
            )
        return _command_tune_contact_preserving_micro_jerk_rescue(
            args,
            config_path=config_path,
            output_dir=output_dir,
            experiment_id=definition.experiment_id,
        )
    if force_debias_rescue_from is not None:
        if not is_v14_planned_lift:
            raise ValueError(
                "--force-debias-rescue-from is available only for the registered "
                "schema-v14 contact-preserving planned-lift experiment"
            )
        if args.output_dir is None:
            raise ValueError(
                "--force-debias-rescue-from requires an explicit new --output-dir"
            )
        if reuse_refinement_from is not None:
            raise ValueError(
                "--reuse-refinement-from cannot be combined with "
                "--force-debias-rescue-from"
            )
        if bool(getattr(args, "evidence_grasp_anchor", ())):
            raise ValueError(
                "--force-debias-rescue-from cannot be combined with "
                "--evidence-grasp-anchor"
            )
        return _command_tune_contact_preserving_force_debias_rescue(
            args,
            config_path=config_path,
            output_dir=output_dir,
            experiment_id=definition.experiment_id,
        )
    if adaptive_event_rescue_from is not None:
        if not is_v14_planned_lift:
            raise ValueError(
                "--adaptive-event-rescue-from is available only for the registered "
                "schema-v14 contact-preserving planned-lift experiment"
            )
        if args.output_dir is None:
            raise ValueError(
                "--adaptive-event-rescue-from requires an explicit new --output-dir"
            )
        if reuse_refinement_from is not None:
            raise ValueError(
                "--reuse-refinement-from cannot be combined with "
                "--adaptive-event-rescue-from"
            )
        if bool(getattr(args, "evidence_grasp_anchor", ())):
            raise ValueError(
                "--adaptive-event-rescue-from cannot be combined with "
                "--evidence-grasp-anchor"
            )
        return _command_tune_contact_preserving_adaptive_event_rescue(
            args,
            config_path=config_path,
            output_dir=output_dir,
            experiment_id=definition.experiment_id,
        )
    if event_rescue_from is not None:
        if not is_v14_planned_lift:
            raise ValueError(
                "--event-rescue-from is available only for the registered schema-v14 "
                "contact-preserving planned-lift experiment"
            )
        if args.output_dir is None:
            raise ValueError(
                "--event-rescue-from requires an explicit new --output-dir"
            )
        if reuse_refinement_from is not None:
            raise ValueError(
                "--reuse-refinement-from cannot be combined with --event-rescue-from"
            )
        if bool(getattr(args, "evidence_grasp_anchor", ())):
            raise ValueError(
                "--event-rescue-from cannot be combined with --evidence-grasp-anchor"
            )
        return _command_tune_contact_preserving_event_rescue(
            args,
            config_path=config_path,
            output_dir=output_dir,
            experiment_id=definition.experiment_id,
        )
    if rescue_from is not None:
        if not is_v14_planned_lift:
            raise ValueError(
                "--rescue-from is available only for the registered schema-v14 "
                "contact-preserving planned-lift experiment"
            )
        if args.output_dir is None:
            raise ValueError(
                "--rescue-from requires an explicit new --output-dir"
            )
        if bool(getattr(args, "evidence_grasp_anchor", ())):
            raise ValueError(
                "--rescue-from cannot be combined with --evidence-grasp-anchor"
            )
        return _command_tune_contact_preserving_rescue(
            args,
            config_path=config_path,
            output_dir=output_dir,
            experiment_id=definition.experiment_id,
        )
    if reuse_refinement_from is not None:
        raise ValueError("--reuse-refinement-from requires --rescue-from")
    is_actual_contact_campaign = is_actual_contact_definition(definition)
    if is_actual_contact_campaign:
        return _command_tune_actual_contact_grasp_pose(
            args,
            config_path=config_path,
            output_dir=output_dir,
            experiment_id=definition.experiment_id,
        )
    if (
        bool(getattr(args, "resume", False))
        or getattr(args, "target_success_count", None) is not None
        or bool(getattr(args, "evidence_grasp_anchor", ()))
    ):
        raise ValueError(
            "--resume, --target-success-count and --evidence-grasp-anchor are "
            "available only for the "
            "registered schema-v9-or-later actual-contact grasp-pose campaign"
        )
    if output_dir.exists():
        raise FileExistsError(
            f"output directory already exists: {output_dir}; choose a new --output-dir"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    metadata = run_metadata(config_path)
    result = tune(
        config,
        samples=args.samples,
        refine_top=args.refine_top,
        refine_per=args.refine_per,
        workers=args.workers,
        seed=args.seed,
        kinematic_samples_per_pitch=args.kinematic_samples_per_pitch,
        dynamic_candidate_count=args.dynamic_candidates,
        local_refine_seed_count=args.local_refine_seeds,
        local_refine_per_seed=args.local_refine_per_seed,
        final_candidate_count=args.final_candidates,
        perturbations_per_final=args.perturbations_per_final,
        fallback_physics_count=args.fallback_physics_candidates,
        fallback_kinematic_samples_per_pitch=(
            args.fallback_kinematic_samples_per_pitch
        ),
    )
    result["metadata"] = metadata
    result["input_config"] = config
    artifact_names = {
        "tune_results": "tune_results.json",
        "best_config": "best_config.json",
        "best_fixed_mass_config": None,
        "best_constant_density_config": None,
    }
    best_fixed_mass = result.get("best_fixed_mass")
    if isinstance(best_fixed_mass, dict) and isinstance(
        best_fixed_mass.get("config"), dict
    ):
        artifact_names["best_fixed_mass_config"] = "best_fixed_mass_config.json"
    if result["best"].get("material_policy") == "constant_density":
        artifact_names["best_constant_density_config"] = (
            "best_constant_density_config.json"
        )
    result["artifacts"] = artifact_names
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent, prefix=f".{output_dir.name}.staging."
    ) as staging_name:
        staging_dir = Path(staging_name)
        write_json(staging_dir / "best_config.json", result["best"]["config"])
        if artifact_names["best_fixed_mass_config"] is not None:
            write_json(
                staging_dir / artifact_names["best_fixed_mass_config"],
                best_fixed_mass["config"],
            )
        if artifact_names["best_constant_density_config"] is not None:
            write_json(
                staging_dir / artifact_names["best_constant_density_config"],
                result["best"]["config"],
            )
        if int(config.get("schema_version", 1)) >= 4:
            selected_band_candidates = result.get("selected_band_candidates")
            if not isinstance(selected_band_candidates, (list, tuple)):
                raise RuntimeError(
                    "schema-v4 tuning must return selected_band_candidates"
                )
            catalog_directory = staging_dir / "trajectory_catalog"
            catalog_kwargs = {
                "include_near_misses": True,
                "video": not bool(getattr(args, "no_catalog_video", False)),
            }
            if int(config.get("schema_version", 1)) >= 5:
                catalog_kwargs["best_candidate_id"] = int(
                    result["best"]["candidate_id"]
                )
            catalog = export_aligned_trajectory_catalog(
                selected_band_candidates,
                catalog_directory,
                **catalog_kwargs,
            )
            catalog_path = catalog_directory / "catalog.json"
            if not catalog_path.is_file():
                raise RuntimeError(
                    "aligned trajectory publisher did not create catalog.json"
                )
            artifact_names.update(
                {
                    "trajectory_catalog": "trajectory_catalog/catalog.json",
                    "trajectory_catalog_sha256": file_sha256(catalog_path),
                    "trajectory_catalog_summary": _aligned_catalog_summary(catalog),
                }
            )
        write_json(staging_dir / "tune_results.json", result)
        staging_dir.rename(output_dir)
    hard_constraints_passed = bool(result["best"]["summary"]["passed"])
    campaign_passed = bool(
        result["best"]["config"].get("experiment_status", {}).get(
            "passed", hard_constraints_passed
        )
    )
    summary = {
        "candidate_count": result["candidate_count"],
        "simulation_count": result["simulation_count"],
        "perturbation_probe_count": result["perturbation_probe_count"],
        "passing_candidates": result["passing_candidates"],
        "best_passed": campaign_passed,
        "best_hard_constraints_passed": hard_constraints_passed,
        "best_perturbation_passes": result["best"]["local_perturbation_probe"][
            "passes"
        ],
        "best_perturbation_trials": result["best"]["local_perturbation_probe"][
            "trial_count"
        ],
        "best_failed_checks": result["best"]["summary"]["failed_checks"],
        "best_metrics": result["best"]["summary"]["metrics"],
    }
    print(json_text(summary))
    return 0 if campaign_passed else 2


def _load_contact_preserving_rescue_runner():
    """Load the v14 post-campaign rescue implementation only when requested."""

    from .tuning.contact_preserving_lift_rescue_campaign import (
        run_contact_preserving_lift_rescue_campaign,
    )

    return run_contact_preserving_lift_rescue_campaign


def _load_contact_preserving_event_rescue_runner():
    """Load the v14 event-aware third-stage rescue only when requested."""

    from .tuning.contact_preserving_event_rescue_campaign import (
        run_contact_preserving_event_rescue_campaign,
    )

    return run_contact_preserving_event_rescue_campaign


def _load_contact_preserving_adaptive_event_rescue_runner():
    """Load the v14 projected, jerk-first fourth-stage rescue on demand."""

    from .tuning.contact_preserving_adaptive_event_rescue_campaign import (
        run_contact_preserving_adaptive_event_rescue_campaign,
    )

    return run_contact_preserving_adaptive_event_rescue_campaign


def _load_contact_preserving_force_debias_rescue_runner():
    """Load the v14 broad force-debias fifth-stage rescue on demand."""

    from .tuning.contact_preserving_force_debias_rescue_campaign import (
        run_contact_preserving_force_debias_rescue_campaign,
    )

    return run_contact_preserving_force_debias_rescue_campaign


def _load_contact_preserving_micro_jerk_rescue_runner():
    """Load the final bounded v14 micro-jerk rescue only when requested."""

    from .tuning.contact_preserving_micro_jerk_rescue_campaign import (
        run_contact_preserving_micro_jerk_rescue_campaign,
    )

    return run_contact_preserving_micro_jerk_rescue_campaign


def _load_contact_mode_pose_rescue_runner():
    """Load the bounded v14 contact-mode grasp-pose rescue on demand."""

    from .tuning.contact_preserving_contact_mode_pose_rescue_campaign import (
        run_contact_mode_pose_rescue_campaign,
    )

    return run_contact_mode_pose_rescue_campaign


def _load_contact_preserving_adaptive_pose_followup_runner():
    """Load the conditional 64 + 64 v14 grasp-pose follow-up on demand."""

    from .tuning.contact_preserving_adaptive_pose_followup_campaign import (
        run_contact_preserving_adaptive_pose_followup_campaign,
    )

    return run_contact_preserving_adaptive_pose_followup_campaign


def _command_tune_contact_preserving_adaptive_pose_followup(
    args: argparse.Namespace,
    *,
    config_path: Path,
    output_dir: Path,
    experiment_id: str,
) -> int:
    """Run/resume the finite-difference then safe-sparse pose follow-up."""

    source = Path(str(args.adaptive_pose_followup_from)).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output_dir == source or output_dir.is_relative_to(source):
        raise ValueError(
            "the adaptive-pose-followup --output-dir must be outside the "
            "immutable source"
        )
    resume = bool(getattr(args, "resume", False))
    if output_dir.exists() and not resume:
        raise FileExistsError(
            f"output directory already exists: {output_dir}; pass --resume"
        )
    target_success_count = int(getattr(args, "target_success_count", None) or 1)
    runner = _load_contact_preserving_adaptive_pose_followup_runner()
    result = runner(
        config_path,
        output_dir,
        source_contact_mode_campaign=source,
        resume=resume,
        target_success_count=target_success_count,
        workers=int(args.workers),
        seed=int(args.seed),
    )
    if not isinstance(result, dict):
        raise RuntimeError("schema-v14 adaptive pose runner must return a mapping")
    full_success_count = int(result.get("full_success_count", 0))
    console = {
        "experiment_id": experiment_id,
        "campaign_kind": "contact_preserving_adaptive_pose_followup",
        "source_contact_mode_campaign": str(source),
        "resume": resume,
        "target_success_count": target_success_count,
        "probe_candidate_count": int(result.get("probe_candidate_count", 0)),
        "probe_full_success_count": int(
            result.get("probe_full_success_count", 0)
        ),
        "sparse_stage_executed": bool(
            result.get("sparse_stage_executed", False)
        ),
        "sparse_candidate_count": int(result.get("sparse_candidate_count", 0)),
        "physical_unique_candidate_count": int(
            result.get("physical_unique_candidate_count", 0)
        ),
        "full_success_count": full_success_count,
        "target_reached": full_success_count >= target_success_count,
        "catalogs": copy.deepcopy(result.get("catalogs", {})),
        "output_dir": str(output_dir),
    }
    print(json_text(console))
    return 0 if console["target_reached"] else 2


def _command_tune_contact_mode_pose_rescue(
    args: argparse.Namespace,
    *,
    config_path: Path,
    output_dir: Path,
    experiment_id: str,
) -> int:
    """Run/resume the immutable-source, trace-ranked grasp-pose rescue."""

    source = Path(str(args.contact_mode_pose_rescue_from)).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output_dir == source or output_dir.is_relative_to(source):
        raise ValueError(
            "the contact-mode-pose-rescue --output-dir must be outside the "
            "immutable source"
        )
    resume = bool(getattr(args, "resume", False))
    if output_dir.exists() and not resume:
        raise FileExistsError(
            f"output directory already exists: {output_dir}; pass --resume"
        )
    target_success_count = int(getattr(args, "target_success_count", None) or 1)
    runner = _load_contact_mode_pose_rescue_runner()
    result = runner(
        config_path,
        output_dir,
        source_candidate_directory=source,
        resume=resume,
        target_success_count=target_success_count,
        workers=int(args.workers),
        seed=int(args.seed),
    )
    if not isinstance(result, dict):
        raise RuntimeError("schema-v14 contact-mode rescue runner must return a mapping")
    full_success_count = int(result.get("full_success_count", 0))
    console = {
        "experiment_id": experiment_id,
        "campaign_kind": "contact_preserving_contact_mode_pose_rescue",
        "source_candidate_directory": str(source),
        "resume": resume,
        "target_success_count": target_success_count,
        "candidate_count": int(result.get("candidate_count", 0)),
        "source_reproduction_candidate_count": int(
            result.get("source_reproduction_candidate_count", 0)
        ),
        "new_unique_candidate_count": int(
            result.get("new_unique_candidate_count", 0)
        ),
        "full_success_count": full_success_count,
        "target_reached": full_success_count >= target_success_count,
        "catalogs": copy.deepcopy(result.get("catalogs", {})),
        "output_dir": str(output_dir),
    }
    print(json_text(console))
    return 0 if console["target_reached"] else 2


def _command_tune_contact_preserving_micro_jerk_rescue(
    args: argparse.Namespace,
    *,
    config_path: Path,
    output_dir: Path,
    experiment_id: str,
) -> int:
    """Run/resume the immutable-source, bounded micro-jerk rescue."""

    source = Path(str(args.micro_jerk_rescue_from)).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output_dir == source or output_dir.is_relative_to(source):
        raise ValueError(
            "the micro-jerk-rescue --output-dir must be outside the immutable source"
        )
    resume = bool(getattr(args, "resume", False))
    if output_dir.exists() and not resume:
        raise FileExistsError(
            f"output directory already exists: {output_dir}; pass --resume"
        )
    target_success_count = int(getattr(args, "target_success_count", None) or 1)
    runner = _load_contact_preserving_micro_jerk_rescue_runner()
    result = runner(
        config_path,
        output_dir,
        source_force_debias_campaign=source,
        resume=resume,
        target_success_count=target_success_count,
        workers=int(args.workers),
        seed=int(args.seed),
    )
    if not isinstance(result, dict):
        raise RuntimeError("schema-v14 micro-jerk rescue runner must return a mapping")
    full_success_count = int(result.get("full_success_count", 0))
    console = {
        "experiment_id": experiment_id,
        "campaign_kind": "contact_preserving_micro_jerk_rescue",
        "source_force_debias_campaign": str(source),
        "resume": resume,
        "target_success_count": target_success_count,
        "micro_candidate_count": int(
            result.get("micro_candidate_count", result.get("candidate_count", 0))
        ),
        "physical_unique_candidate_count": int(
            result.get("physical_unique_candidate_count", 0)
        ),
        "full_success_count": full_success_count,
        "target_reached": full_success_count >= target_success_count,
        "catalogs": copy.deepcopy(result.get("catalogs", {})),
        "output_dir": str(output_dir),
    }
    print(json_text(console))
    return 0 if console["target_reached"] else 2


def _command_tune_contact_preserving_force_debias_rescue(
    args: argparse.Namespace,
    *,
    config_path: Path,
    output_dir: Path,
    experiment_id: str,
) -> int:
    """Run/resume the immutable-source broad force-debias rescue."""

    source = Path(str(args.force_debias_rescue_from)).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output_dir == source or output_dir.is_relative_to(source):
        raise ValueError(
            "the force-debias-rescue --output-dir must be outside the immutable source"
        )
    resume = bool(getattr(args, "resume", False))
    if output_dir.exists() and not resume:
        raise FileExistsError(
            f"output directory already exists: {output_dir}; pass --resume"
        )
    target_success_count = int(getattr(args, "target_success_count", None) or 1)
    runner = _load_contact_preserving_force_debias_rescue_runner()
    result = runner(
        config_path,
        output_dir,
        source_adaptive_campaign=source,
        resume=resume,
        target_success_count=target_success_count,
        workers=int(args.workers),
        seed=int(args.seed),
    )
    if not isinstance(result, dict):
        raise RuntimeError("schema-v14 force-debias rescue runner must return a mapping")
    full_success_count = int(result.get("full_success_count", 0))
    console = {
        "experiment_id": experiment_id,
        "campaign_kind": "contact_preserving_force_debias_rescue",
        "source_adaptive_campaign": str(source),
        "resume": resume,
        "target_success_count": target_success_count,
        "discovery_candidate_count": int(
            result.get("discovery_candidate_count", 0)
        ),
        "refinement_candidate_count": int(
            result.get("refinement_candidate_count", 0)
        ),
        "physical_unique_candidate_count": int(
            result.get("physical_unique_candidate_count", 0)
        ),
        "full_success_count": full_success_count,
        "target_reached": full_success_count >= target_success_count,
        "catalogs": copy.deepcopy(result.get("catalogs", {})),
        "output_dir": str(output_dir),
    }
    print(json_text(console))
    return 0 if console["target_reached"] else 2


def _command_tune_contact_preserving_adaptive_event_rescue(
    args: argparse.Namespace,
    *,
    config_path: Path,
    output_dir: Path,
    experiment_id: str,
) -> int:
    """Run/resume the physical-plan-deduplicated adaptive event rescue."""

    source = Path(str(args.adaptive_event_rescue_from)).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output_dir == source or output_dir.is_relative_to(source):
        raise ValueError(
            "the adaptive-event-rescue --output-dir must be outside the immutable "
            "source"
        )
    resume = bool(getattr(args, "resume", False))
    if output_dir.exists() and not resume:
        raise FileExistsError(
            f"output directory already exists: {output_dir}; pass --resume"
        )
    target_success_count = int(getattr(args, "target_success_count", None) or 1)
    runner = _load_contact_preserving_adaptive_event_rescue_runner()
    result = runner(
        config_path,
        output_dir,
        source_event_campaign=source,
        resume=resume,
        target_success_count=target_success_count,
        workers=int(args.workers),
        seed=int(args.seed),
    )
    if not isinstance(result, dict):
        raise RuntimeError("schema-v14 adaptive event-rescue runner must return a mapping")
    full_success_count = int(result.get("full_success_count", 0))
    console = {
        "experiment_id": experiment_id,
        "campaign_kind": "contact_witness_adaptive_event_rescue",
        "source_event_campaign": str(source),
        "resume": resume,
        "target_success_count": target_success_count,
        "diagnostic_candidate_count": int(
            result.get("diagnostic_candidate_count", 0)
        ),
        "exploration_candidate_count": int(
            result.get("exploration_candidate_count", 0)
        ),
        "local_refinement_candidate_count": int(
            result.get("local_refinement_candidate_count", 0)
        ),
        "physical_unique_candidate_count": int(
            result.get("physical_unique_candidate_count", 0)
        ),
        "full_success_count": full_success_count,
        "target_reached": full_success_count >= target_success_count,
        "catalogs": copy.deepcopy(result.get("catalogs", {})),
        "output_dir": str(output_dir),
    }
    print(json_text(console))
    return 0 if console["target_reached"] else 2


def _command_tune_contact_preserving_event_rescue(
    args: argparse.Namespace,
    *,
    config_path: Path,
    output_dir: Path,
    experiment_id: str,
) -> int:
    """Run/resume the hash-bound contact-witness event rescue."""

    source = Path(str(args.event_rescue_from)).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output_dir == source or output_dir.is_relative_to(source):
        raise ValueError(
            "the event-rescue --output-dir must be outside the immutable source"
        )
    resume = bool(getattr(args, "resume", False))
    if output_dir.exists() and not resume:
        raise FileExistsError(
            f"output directory already exists: {output_dir}; pass --resume"
        )
    target_success_count = int(getattr(args, "target_success_count", None) or 1)
    runner = _load_contact_preserving_event_rescue_runner()
    result = runner(
        config_path,
        output_dir,
        source_rescue_campaign=source,
        resume=resume,
        target_success_count=target_success_count,
        workers=int(args.workers),
        seed=int(args.seed),
    )
    if not isinstance(result, dict):
        raise RuntimeError("schema-v14 event-rescue runner must return a mapping")
    full_success_count = int(result.get("full_success_count", 0))
    console = {
        "experiment_id": experiment_id,
        "campaign_kind": "contact_witness_event_rescue",
        "source_rescue_campaign": str(source),
        "resume": resume,
        "target_success_count": target_success_count,
        "exploration_candidate_count": int(
            result.get("exploration_candidate_count", 0)
        ),
        "local_refinement_candidate_count": int(
            result.get("local_refinement_candidate_count", 0)
        ),
        "full_success_count": full_success_count,
        "target_reached": full_success_count >= target_success_count,
        "catalogs": copy.deepcopy(result.get("catalogs", {})),
        "output_dir": str(output_dir),
    }
    print(json_text(console))
    return 0 if console["target_reached"] else 2


def _command_tune_contact_preserving_rescue(
    args: argparse.Namespace,
    *,
    config_path: Path,
    output_dir: Path,
    experiment_id: str,
) -> int:
    """Run/resume a hash-bound rescue without mutating its source campaign."""

    source = Path(str(args.rescue_from)).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output_dir == source or output_dir.is_relative_to(source):
        raise ValueError(
            "the rescue --output-dir must be outside the immutable source campaign"
        )
    reuse_refinement_value = getattr(args, "reuse_refinement_from", None)
    reuse_refinement_from = (
        None
        if reuse_refinement_value is None
        else Path(str(reuse_refinement_value)).expanduser().resolve()
    )
    if reuse_refinement_from is not None:
        if not reuse_refinement_from.is_dir():
            raise FileNotFoundError(reuse_refinement_from)
        if output_dir == reuse_refinement_from or output_dir.is_relative_to(
            reuse_refinement_from
        ):
            raise ValueError(
                "the rescue --output-dir must be outside the immutable refinement source"
            )
    resume = bool(getattr(args, "resume", False))
    if output_dir.exists() and not resume:
        raise FileExistsError(
            f"output directory already exists: {output_dir}; pass --resume"
        )
    target_success_count = int(
        getattr(args, "target_success_count", None) or 1
    )
    runner = _load_contact_preserving_rescue_runner()
    result = runner(
        config_path,
        output_dir,
        source_campaign=source,
        reuse_refinement_from=reuse_refinement_from,
        resume=resume,
        target_success_count=target_success_count,
        workers=int(args.workers),
        seed=int(args.seed),
    )
    if not isinstance(result, dict):
        raise RuntimeError("schema-v14 rescue runner must return a mapping")
    full_success_count = int(result.get("full_success_count", 0))
    console = {
        "experiment_id": experiment_id,
        "campaign_kind": "contact_preserving_post_campaign_rescue",
        "source_campaign": str(source),
        "resume": resume,
        "target_success_count": target_success_count,
        "phase_one_candidate_count": int(
            result.get("phase_one_candidate_count", 0)
        ),
        "time_warp_candidate_count": int(
            result.get("time_warp_candidate_count", 0)
        ),
        "full_success_count": full_success_count,
        "target_reached": full_success_count >= target_success_count,
        "catalogs": copy.deepcopy(result.get("catalogs", {})),
        "output_dir": str(output_dir),
    }
    print(json_text(console))
    return 0 if console["target_reached"] else 2


def _load_v9_tune_runner():
    """Compatibility hook used by the schema-v9--v11 CLI tests."""

    from .tuning.actual_contact_grasp_pose import (
        run_actual_contact_grasp_pose_campaign,
    )

    return run_actual_contact_grasp_pose_campaign


def _load_v15_campaign_backend_factory():
    """Load the schema-v15 MuJoCo backend only when ``tune`` needs it."""

    from .tuning.joint_pair_near_zero_physics_backend import (
        create_joint_pair_near_zero_campaign_backend,
    )

    return create_joint_pair_near_zero_campaign_backend


def _run_v15_joint_pair_near_zero_tune_campaign(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    resume: bool,
    target_success_count: int,
    workers: int,
    seed: int,
    evidence_anchor_paths: tuple[str, ...] = (),
) -> dict:
    """Adapt the common CLI contract to the injected schema-v15 runner.

    The expensive physics implementation stays behind a delayed factory.  In
    particular, importing ``grasp_cube.py`` or running an older experiment
    does not import MuJoCo campaign workers or construct a process pool.
    """

    if evidence_anchor_paths:
        raise ValueError(
            "schema-v15 uses its registered hash-bound near-zero source; "
            "--evidence-grasp-anchor is not accepted"
        )
    config = load_config(config_path)
    resolve_joint_pair_near_zero_contact_preserving_planned_lift_definition(
        config,
        context="schema-v15 tune",
    )
    from .tuning.joint_pair_near_zero_campaign import JointPairNearZeroBudget
    from .tuning.joint_pair_near_zero_campaign_runner import (
        run_joint_pair_near_zero_campaign,
    )

    budget = JointPairNearZeroBudget(seed=int(seed))
    backend_factory = _load_v15_campaign_backend_factory()
    backend = backend_factory(workers=int(workers), seed=int(seed))
    result = run_joint_pair_near_zero_campaign(
        config_path,
        output_dir,
        resume=bool(resume),
        target_success_count=int(target_success_count),
        backend=backend,
        budget=budget,
    )
    catalog_path = result.get("catalog_path")
    if catalog_path is not None and "catalogs" not in result:
        result["catalogs"] = {"manipulation": str(catalog_path)}
    return result


def _load_actual_contact_tune_runner(definition):
    """Load the registered actual-contact campaign implementation lazily."""

    if is_joint_pair_near_zero_contact_preserving_planned_lift_definition(
        definition
    ):
        return _run_v15_joint_pair_near_zero_tune_campaign
    if (
        getattr(definition, "contact_preserving_planned_lift_campaign", None)
        is not None
    ):
        from .tuning.contact_preserving_planned_lift_campaign import (
            run_contact_preserving_planned_lift_campaign,
        )

        return run_contact_preserving_planned_lift_campaign
    if getattr(definition, "scaled_contact_downsize_campaign", None) is not None:
        from .tuning.scaled_contact_downsize_campaign import (
            run_scaled_contact_downsize_campaign,
        )

        return run_scaled_contact_downsize_campaign
    if getattr(definition, "contact_point_search", None) is not None:
        from .tuning.contact_point_targeted_campaign import (
            run_contact_point_targeted_campaign,
        )

        return run_contact_point_targeted_campaign

    return _load_v9_tune_runner()


def _build_actual_contact_campaign_manifest(
    definition,
    config_path: Path,
    *,
    seed: int,
) -> dict:
    """Build the capability-specific immutable resume manifest."""

    if is_joint_pair_near_zero_contact_preserving_planned_lift_definition(
        definition
    ):
        from .tuning.joint_pair_near_zero_campaign import JointPairNearZeroBudget
        from .tuning.joint_pair_near_zero_campaign_runner import (
            build_v15_campaign_manifest,
        )

        return build_v15_campaign_manifest(
            config_path,
            budget=JointPairNearZeroBudget(seed=int(seed)),
        )
    if (
        getattr(definition, "contact_preserving_planned_lift_campaign", None)
        is not None
    ):
        from .tuning.contact_preserving_planned_lift_campaign import (
            build_contact_preserving_planned_lift_manifest,
        )

        return build_contact_preserving_planned_lift_manifest(
            config_path, seed=seed
        )
    if getattr(definition, "scaled_contact_downsize_campaign", None) is not None:
        from .tuning.scaled_contact_downsize_campaign import (
            build_scaled_contact_downsize_manifest,
        )

        return build_scaled_contact_downsize_manifest(config_path, seed=seed)
    from .actual_contact_grasp_pose_catalog import build_campaign_manifest

    return build_campaign_manifest(config_path, seed=seed)


def _publish_v9_returned_candidates(
    result: dict,
    output_dir: Path,
    *,
    target_success_count: int,
    experiment_id: str = LEGACY_ACTUAL_CONTACT_EXPERIMENT_ID,
    allow_contact_point_catalog: bool = False,
) -> dict[str, str]:
    """Publish optional runner candidates under two Viewer-compatible catalogs.

    A runner may publish the catalogs itself and return ``catalogs``.  The
    candidate-list form keeps the runner contract convenient for tests and for
    staged searches: this function is the one canonical publisher in that
    case.
    """

    catalogs = result.get("catalogs")
    if catalogs is not None:
        if not isinstance(catalogs, dict):
            raise RuntimeError("schema-v9 runner catalogs must be a mapping")
        validated: dict[str, str] = {}
        for key, value in catalogs.items():
            raw_path = Path(str(value))
            path = (
                raw_path.expanduser().resolve()
                if raw_path.is_absolute()
                else (output_dir / raw_path).resolve()
            )
            if not path.is_relative_to(output_dir) or not path.is_file():
                raise RuntimeError(
                    "schema-v9 runner catalog must be a file inside output_dir"
                )
            payload = json.loads(path.read_text(encoding="utf-8"))
            if key == "contact_point" and allow_contact_point_catalog:
                plans = payload.get("plans") if isinstance(payload, dict) else None
                selected_plan_id = (
                    payload.get("selected_point_plan_id")
                    if isinstance(payload, dict)
                    else None
                )
                plan_ids = {
                    value.get("point_plan_id")
                    for value in plans or ()
                    if isinstance(value, dict)
                }
                if (
                    not isinstance(payload, dict)
                    or payload.get("contact_point_catalog_schema_version") != 1
                    or payload.get("complete") is not True
                    or payload.get("experiment_id") != experiment_id
                    or not isinstance(plans, list)
                    or (
                        selected_plan_id is not None
                        and selected_plan_id not in plan_ids
                    )
                ):
                    raise RuntimeError(
                        "schema-v12 runner returned an invalid contact-point catalog"
                    )
                validated[str(key)] = str(path.relative_to(output_dir))
                continue
            if (
                not isinstance(payload, dict)
                or payload.get("experiment_id") != experiment_id
                or not isinstance(payload.get("trajectories"), list)
            ):
                raise RuntimeError("schema-v9 runner returned an invalid catalog")
            aliases = payload.get("aliases", {})
            if not isinstance(aliases, dict):
                raise RuntimeError("schema-v9 catalog aliases must be a mapping")
            if "best_first" in aliases:
                target = aliases["best_first"]
                matching = [
                    entry
                    for entry in payload["trajectories"]
                    if isinstance(entry, dict)
                    and entry.get("trajectory_id") == target
                ]
                if len(matching) != 1 or matching[0].get("classification") != "success":
                    raise RuntimeError(
                        "schema-v9 best_first must name one canonical success"
                    )
            validated[str(key)] = str(path.relative_to(output_dir))
        return validated

    from .actual_contact_grasp_pose_catalog import (
        export_actual_contact_grasp_pose_catalog,
        export_actual_contact_manipulation_catalog,
    )

    published: dict[str, str] = {}
    grasp_candidates = result.get("grasp_pose_candidates")
    if grasp_candidates is not None:
        destination = output_dir / "grasp_pose_catalog"
        export_actual_contact_grasp_pose_catalog(
            grasp_candidates,
            destination,
            selected_count=target_success_count,
        )
        published["grasp_pose"] = str(
            (destination / "catalog.json").relative_to(output_dir)
        )
    manipulation_candidates = result.get("manipulation_candidates")
    if manipulation_candidates is not None:
        destination = output_dir / "manipulation_catalog"
        export_actual_contact_manipulation_catalog(
            manipulation_candidates,
            destination,
            selected_count=target_success_count,
        )
        published["manipulation"] = str(
            (destination / "catalog.json").relative_to(output_dir)
        )
    return published


def _command_tune_actual_contact_grasp_pose(
    args: argparse.Namespace,
    *,
    config_path: Path,
    output_dir: Path,
    experiment_id: str,
) -> int:
    """Dispatch a resumable measured-grasp-pose campaign."""

    resume = bool(getattr(args, "resume", False))
    target_success_count = int(
        getattr(args, "target_success_count", None) or 1
    )
    if output_dir.exists() and not resume:
        raise FileExistsError(
            f"output directory already exists: {output_dir}; pass --resume"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    from .actual_contact_grasp_pose_catalog import initialize_or_resume_campaign

    template = load_config(config_path)
    definition = resolve_experiment(template)
    manifest = _build_actual_contact_campaign_manifest(
        definition, config_path, seed=int(args.seed)
    )
    # Authenticate before any worker is allowed to reuse a completed stage.
    # On a fresh run the dedicated runner creates this exact workspace itself;
    # doing so here would make its ordinary ``resume=False`` path ambiguous.
    if resume:
        initialize_or_resume_campaign(output_dir, manifest, resume=True)
    runner = _load_actual_contact_tune_runner(definition)
    runner_kwargs = {
        "resume": resume,
        "target_success_count": target_success_count,
        "workers": int(args.workers),
        "seed": int(args.seed),
    }
    evidence_anchor_paths = tuple(
        getattr(args, "evidence_grasp_anchor", ()) or ()
    )
    if evidence_anchor_paths:
        runner_kwargs["evidence_anchor_paths"] = evidence_anchor_paths
    result = runner(config_path, output_dir, **runner_kwargs)
    if not isinstance(result, dict):
        raise RuntimeError("schema-v9 tune runner must return a mapping")
    # This is also the runner interface contract: a reported result is not
    # accepted unless its filesystem workspace binds the exact config, model,
    # uv.lock and implementation source used by this invocation.
    initialize_or_resume_campaign(output_dir, manifest, resume=True)
    grasp_is_campaign_target = (
        getattr(definition, "contact_point_search", None) is not None
    )
    catalogs = _publish_v9_returned_candidates(
        result,
        output_dir,
        target_success_count=target_success_count,
        experiment_id=experiment_id,
        allow_contact_point_catalog=grasp_is_campaign_target,
    )
    if catalogs:
        result["catalogs"] = catalogs
    full_success_count = int(
        result.get(
            "full_success_count",
            result.get("success_count", result.get("passing_candidates", 0)),
        )
    )
    grasp_success_count = int(result.get("grasp_success_count", full_success_count))
    achieved_count = grasp_success_count if grasp_is_campaign_target else full_success_count
    console = {
        "experiment_id": experiment_id,
        "resume": resume,
        "target_success_count": target_success_count,
        "grasp_success_count": grasp_success_count,
        "full_success_count": full_success_count,
        "target_metric": (
            "grasp_success_count"
            if grasp_is_campaign_target
            else "full_success_count"
        ),
        "target_reached": achieved_count >= target_success_count,
        "catalogs": catalogs,
        "output_dir": str(output_dir),
    }
    print(json_text(console))
    return 0 if console["target_reached"] else 2


def _relative_wrist_post_validation_catalog(
    search_roots: tuple[Path, ...],
) -> Path | None:
    """Return the preferred manipulation catalog for v11 post-validation.

    The existing actual-contact robustness command accepts either a catalog or
    a campaign directory.  Density revalidation is intentionally stricter: it
    authenticates a published manipulation catalog.  Preserve the caller's
    root order and prefer target-five over target-one within a campaign root.
    Result-only inputs remain valid for the legacy robustness audit, but cannot
    silently stand in for a canonical post-validation catalog.
    """

    relative_candidates = (
        Path("catalogs/target_5/manipulation/catalog.json"),
        Path("catalogs/target_1/manipulation/catalog.json"),
        Path("target_5/manipulation/catalog.json"),
        Path("target_1/manipulation/catalog.json"),
        Path("manipulation/catalog.json"),
        Path("catalog.json"),
    )
    seen: set[Path] = set()
    for root in search_roots:
        candidates = (root,) if root.is_file() else tuple(
            root / relative for relative in relative_candidates
        )
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if resolved.is_file() and resolved.name == "catalog.json":
                return resolved
    return None


def command_robustness(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    preflight_config(config)

    definition = resolve_experiment(config)
    sidecar_output = getattr(
        args, "contact_preserving_sidecar_output_dir", None
    )
    if sidecar_output is not None:
        if (
            getattr(definition, "contact_preserving_planned_lift_campaign", None)
            is None
        ):
            raise ValueError(
                "--contact-preserving-sidecar-output-dir requires the registered "
                "schema-v14 contact-preserving planned-lift experiment"
            )
        search_roots = tuple(getattr(args, "search_root", ()) or ())
        if len(search_roots) != 1:
            raise ValueError(
                "v14 robustness sidecar requires exactly one --search-root catalog"
            )
        source_result = getattr(
            args, "contact_preserving_source_result", None
        )
        if source_result is None:
            raise ValueError(
                "v14 robustness sidecar requires "
                "--contact-preserving-source-result"
            )
        from .tuning.contact_preserving_robustness_sidecar import (
            command_contact_preserving_robustness_sidecar,
        )

        sidecar_args = argparse.Namespace(
            catalog=search_roots[0],
            source_result=source_result,
            output_dir=sidecar_output,
            resume=bool(
                getattr(args, "resume_contact_preserving_sidecar", False)
            ),
            workers=int(args.workers),
            seed=int(args.seed),
        )
        return int(command_contact_preserving_robustness_sidecar(sidecar_args))
    relative_post_validation_options = {
        "post_validation_catalog": getattr(args, "post_validation_catalog", None),
        "post_validation_output_dir": getattr(
            args, "post_validation_output_dir", None
        ),
        "resume_post_validation": bool(
            getattr(args, "resume_post_validation", False)
        ),
        "render_density_videos": bool(
            getattr(args, "render_density_videos", False)
        ),
    }
    if definition.relative_wrist_pose_search is None and any(
        value is not None if key.endswith(("catalog", "output_dir")) else bool(value)
        for key, value in relative_post_validation_options.items()
    ):
        raise ValueError(
            "relative-wrist post-validation options require an experiment with "
            "relative_wrist_pose_search capability"
        )
    if definition.contact_point_search is not None:
        from .tuning.contact_point_grasp_robustness import (
            discover_v12_grasp_robustness_sources,
            run_v12_grasp_perturbation_audit,
        )

        supplied_roots = tuple(getattr(args, "search_root", ()) or ())
        if supplied_roots:
            search_roots = tuple(Path(value).resolve() for value in supplied_roots)
        else:
            tune_root = REPO_ROOT / definition.artifact_root / "tune"
            campaign_roots = (tune_root / "campaign", tune_root)
            candidates = tuple(
                campaign_root
                / "catalogs"
                / f"target_{target}"
                / "grasp_pose"
                / "catalog.json"
                for campaign_root in campaign_roots
                for target in (5, 1)
            )
            observed: set[Path] = set()
            search_roots = tuple(
                resolved
                for path in candidates
                if path.is_file()
                and (resolved := path.resolve()) not in observed
                and not observed.add(resolved)
            )
            if not search_roots:
                raise FileNotFoundError(
                    "no schema-v12 grasp-pose catalog was found under "
                    f"{tune_root}; pass --search-root explicitly"
                )
        output = Path(
            args.output
            or (
                REPO_ROOT
                / definition.artifact_root
                / "robustness"
                / "grasp_perturbation_report.json"
            )
        ).resolve()
        if output.exists():
            raise FileExistsError(
                f"output file already exists: {output}; choose a new --output"
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        sources = discover_v12_grasp_robustness_sources(search_roots)
        report = run_v12_grasp_perturbation_audit(
            sources,
            output,
            workers=int(args.workers),
            seed=int(args.seed),
        )
        console = {
            "selected_grasp_count": report["selected_grasp_count"],
            "total_perturbation_count": report["total_perturbation_count"],
            "best_grasp_passes": report["best_robustness"]["grasp_passes"],
            "required_best_passes": report["best_robustness"][
                "required_grasp_passes"
            ],
            "robust_passed": bool(report["robust_passed"]),
            "success_scope": "grasp_only_contact_point_hard_checks",
            "output": str(output),
        }
        print(json_text(console))
        return 0 if console["robust_passed"] else 2
    if is_actual_contact_definition(definition):
        from .tuning.actual_contact_grasp_pose_robustness import (
            run_v9_robustness_campaign,
        )

        supplied_roots = tuple(getattr(args, "search_root", ()) or ())
        if supplied_roots:
            search_roots = tuple(Path(value).resolve() for value in supplied_roots)
        else:
            campaign_root = REPO_ROOT / definition.artifact_root / "tune" / "campaign"
            preferred = (
                campaign_root
                / "catalogs"
                / "target_5"
                / "manipulation"
                / "catalog.json"
            )
            fallback = (
                campaign_root
                / "catalogs"
                / "target_1"
                / "manipulation"
                / "catalog.json"
            )
            search_roots = tuple(
                path for path in (preferred, fallback) if path.is_file()
            )
            if not search_roots:
                raise FileNotFoundError(
                    "no actual-contact manipulation catalog was found under "
                    f"{campaign_root}; pass --search-root explicitly"
                )
        output = Path(
            args.output
            or (
                REPO_ROOT
                / definition.artifact_root
                / "robustness"
                / "perturbation_report.json"
            )
        ).resolve()
        if output.exists():
            raise FileExistsError(
                f"output file already exists: {output}; choose a new --output"
            )
        output.parent.mkdir(parents=True, exist_ok=True)

        relative_post_validation: dict | None = None
        if definition.relative_wrist_pose_search is not None:
            from .tuning.relative_wrist_pose_post_validation import (
                load_canonical_fixed_160g_sources,
                run_relative_wrist_pose_post_validation,
            )

            explicit_catalog = relative_post_validation_options[
                "post_validation_catalog"
            ]
            post_validation_catalog = (
                Path(explicit_catalog).resolve()
                if explicit_catalog is not None
                else _relative_wrist_post_validation_catalog(search_roots)
            )
            post_validation_output = Path(
                relative_post_validation_options["post_validation_output_dir"]
                or (
                    REPO_ROOT
                    / definition.artifact_root
                    / "post_validation"
                )
            ).resolve()
            if post_validation_catalog is None:
                relative_post_validation = {
                    "status": "not_run_without_manipulation_catalog",
                    "catalog": None,
                    "output_dir": str(post_validation_output),
                    "passed": False,
                }
            else:
                try:
                    canonical_sources = load_canonical_fixed_160g_sources(
                        post_validation_catalog
                    )
                except ValueError as exc:
                    if str(exc) != (
                        "catalog contains no canonical fixed-160-g full success"
                    ):
                        raise
                    canonical_sources = ()
                if not canonical_sources:
                    relative_post_validation = {
                        "status": "not_run_without_fixed_mass_full_success",
                        "catalog": str(post_validation_catalog),
                        "output_dir": str(post_validation_output),
                        "passed": False,
                    }
                else:
                    post_report = run_relative_wrist_pose_post_validation(
                        post_validation_catalog,
                        post_validation_output,
                        resume=relative_post_validation_options[
                            "resume_post_validation"
                        ],
                        seed=int(args.seed),
                        workers=int(args.workers),
                        render_density_videos=relative_post_validation_options[
                            "render_density_videos"
                        ],
                    )
                    fixed_successes = int(
                        post_report["fixed_160g"]["full_success_count"]
                    )
                    density_successes = int(
                        post_report["constant_density"]["full_success_count"]
                    )
                    pose_friction_passed = bool(
                        post_report["best_first_pose_friction"]["robust_passed"]
                    )
                    relative_post_validation = {
                        "status": "complete",
                        "catalog": str(post_validation_catalog),
                        "output_dir": str(post_validation_output),
                        "fixed_160g_full_success_count": fixed_successes,
                        "constant_density_full_success_count": density_successes,
                        "pose_friction_robust_passed": pose_friction_passed,
                        "passed": bool(
                            fixed_successes > 0
                            and density_successes > 0
                            and pose_friction_passed
                        ),
                    }
        report = run_v9_robustness_campaign(
            search_roots,
            output,
            workers=int(args.workers),
            seed=int(args.seed),
        )
        console = {
            "selected_nominal_count": report["selected_nominal_count"],
            "total_perturbation_count": report["total_perturbation_count"],
            "best_perturbation_passes": report["best_robustness"][
                "perturbation_passes"
            ],
            "required_best_passes": report["best_robustness"][
                "required_perturbation_passes"
            ],
            "robust_passed": bool(report["robust_passed"]),
            "output": str(output),
        }
        exit_passed = bool(report["robust_passed"])
        if relative_post_validation is not None:
            console["fixed_mass_robustness_passed"] = exit_passed
            console["relative_wrist_pose_post_validation"] = (
                relative_post_validation
            )
            exit_passed = bool(
                exit_passed and relative_post_validation["passed"]
            )
            console["robust_passed"] = exit_passed
        print(json_text(console))
        return 0 if exit_passed else 2

    if getattr(args, "search_root", None):
        raise ValueError(
            "--search-root is available only for schema-v9-or-later "
            "actual-contact robustness"
        )
    output_value = _resolved_default_output(
        config, args.output, LEGACY_ROBUSTNESS_OUTPUT, "robustness"
    )
    output = Path(output_value).resolve()
    if output.exists():
        raise FileExistsError(
            f"output file already exists: {output}; choose a new --output"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = run_metadata(config_path)
    result = robustness(config, workers=args.workers, seed=args.seed)
    if int(config.get("schema_version", 1)) >= 2:
        _add_v2_robustness_margin_fields(config, result)
    result["metadata"] = metadata
    input_status = copy.deepcopy(config.get("experiment_status"))
    result_config = resolved_run_config(config, result["nominal_summary"])
    if int(config.get("schema_version", 1)) >= 4:
        status = result_config["experiment_status"]
        reported_robustness_passed = bool(result["robust_passed"])
        canonical_robustness_passed = reported_robustness_passed
        if int(config.get("schema_version", 1)) >= 5:
            canonical_robustness_passed = bool(
                reported_robustness_passed
                and status.get("campaign_validated", False)
                and status.get("run_context") is None
            )
        status.update(
            {
                "robustness_passed": canonical_robustness_passed,
                "robustness_seed": int(result["seed"]),
                "robustness_passes": int(result["perturbation_passes"]),
                "robustness_trial_count": int(
                    result["perturbation_trial_count"]
                ),
                "robustness_required_passes": int(
                    result["required_perturbation_passes"]
                ),
            }
        )
        if canonical_robustness_passed:
            if int(config.get("schema_version", 1)) >= 5:
                status["classification"] = "validated_far_hand_fingertip_robust"
                status["note"] = (
                    "The canonical 60 mm, 160 g, friction-0.8 far-hand "
                    "fingertip run and this report's registered robustness "
                    "envelope both passed."
                )
            else:
                status["classification"] = "validated_aligned_contacts_robust"
                status["note"] = (
                    "The canonical nominal run and this report's registered "
                    "aligned-contact robustness envelope both passed."
                )
    result["input_experiment_status"] = input_status
    result["config"] = result_config
    result["hardest_passing_config"] = _hardest_passing_config(config, result)
    with tempfile.TemporaryDirectory(
        dir=output.parent, prefix=f".{output.stem}.staging."
    ) as staging_name:
        staged_output = Path(staging_name) / output.name
        write_json(staged_output, result)
        if output.exists():
            raise FileExistsError(
                f"output file appeared during execution: {output}; refusing to overwrite"
            )
        staged_output.rename(output)
    print(
        json_text(
            {
                "grid_passes": result["grid_passes"],
                "grid_case_count": result["grid_case_count"],
                "perturbation_passes": result["perturbation_passes"],
                "perturbation_trial_count": result["perturbation_trial_count"],
                "robust_passed": result["robust_passed"],
            }
        )
    )
    return 0 if result["robust_passed"] else 2


def command_catalog(args: argparse.Namespace) -> int:
    """Export selected, independently rerun passing robustness trajectories."""

    labels: dict[int, str] = {}
    for specification in args.label or ():
        index_text, separator, label = specification.partition("=")
        if not separator:
            raise ValueError("--label must use GRID_INDEX=LABEL syntax")
        try:
            grid_index = int(index_text)
        except ValueError as exc:
            raise ValueError("--label grid index must be an integer") from exc
        if grid_index in labels:
            raise ValueError(f"duplicate --label for grid index {grid_index}")
        labels[grid_index] = label
    result = export_trajectory_catalog(
        args.config,
        args.robustness,
        args.grid_index,
        args.output_dir,
        video=bool(args.video),
        labels=labels,
    )
    print(
        json_text(
            {
                "trajectory_count": result["trajectory_count"],
                "selected_grid_indices": result["selected_grid_indices"],
                "all_reruns_full_success": result["all_reruns_full_success"],
                "output_dir": str(Path(args.output_dir).resolve()),
            }
        )
    )
    return 0


def command_grasp_catalog(args: argparse.Namespace) -> int:
    """Materialize a versioned manifest and publish real grasp trajectories."""

    candidates, best_candidate_id, manifest_metadata = (
        load_grasp_campaign_manifest(args.manifest)
    )
    result = export_grasp_trajectory_catalog(
        candidates,
        args.output_dir,
        video=not bool(args.no_video),
        best_candidate_id=best_candidate_id,
    )
    manifest_path = Path(args.manifest).expanduser().resolve()
    base_config_path = Path(manifest_metadata["base_config_path"])
    result["source_manifest"] = {
        **manifest_metadata,
        "manifest_sha256": file_sha256(manifest_path),
        "base_config_sha256": file_sha256(base_config_path),
    }
    write_json(Path(args.output_dir).expanduser().resolve() / "catalog.json", result)
    print(
        json_text(
            {
                "validation_scope": result["validation_scope"],
                "trajectory_count": result["trajectory_count"],
                "validated_grasp_count": result["validated_grasp_count"],
                "failed_grasp_count": result["failed_grasp_count"],
                "best_grasp": result["aliases"].get("best_grasp"),
                "all_reruns_full_success": result["all_reruns_full_success"],
                "output_dir": str(Path(args.output_dir).expanduser().resolve()),
            }
        )
    )
    return 0 if "best_grasp" in result["aliases"] else 2


def command_view(args: argparse.Namespace) -> int:
    """Run live deterministic physics or explicitly replay recorded states."""

    measured_report = args.measured_report
    catalog_metadata_selectors = (
        args.catalog_edge_mm,
        args.catalog_mapping_mode,
        args.catalog_source_alias,
    )
    if any(value is not None for value in catalog_metadata_selectors):
        if args.catalog is None:
            raise ValueError("catalog metadata selectors require --catalog")
        if measured_report is not None or args.config is not None or args.trace is not None:
            raise ValueError(
                "catalog metadata selectors cannot be combined with measured/config/trace"
            )
        if args.trajectory != "nominal":
            raise ValueError(
                "catalog metadata selectors cannot be combined with --trajectory"
            )
    elif args.catalog_rank != 1:
        raise ValueError("--catalog-rank requires a catalog metadata selector")
    measured_selector_values = (
        args.data_index,
        args.candidate_id,
        args.select_edge_mm,
    )
    if args.list_data:
        if measured_report is None:
            raise ValueError("--list-data requires --measured-report")
        if any(value is not None for value in measured_selector_values):
            raise ValueError("--list-data cannot be combined with a data selector")
        if args.edge_rank is not None:
            raise ValueError("--edge-rank requires --select-edge-mm")
        if args.catalog is not None or args.config is not None or args.trace is not None:
            raise ValueError(
                "--list-data uses only --measured-report, not catalog/config/trace"
            )
        entries = discover_measured_viewer_data(measured_report)
        print(format_measured_viewer_data(entries))
        return 0

    if measured_report is None:
        if any(value is not None for value in measured_selector_values):
            raise ValueError("measured data selectors require --measured-report")
        if args.edge_rank is not None:
            raise ValueError("--edge-rank requires --measured-report and --select-edge-mm")
    else:
        if args.catalog is not None or args.config is not None or args.trace is not None:
            raise ValueError(
                "--measured-report cannot be combined with catalog/config/trace"
            )

    measured_source = None
    if measured_report is not None:
        measured_source = resolve_measured_viewer_source(
            measured_report,
            data_index=args.data_index,
            candidate_id=args.candidate_id,
            edge_mm=args.select_edge_mm,
            edge_rank=args.edge_rank,
        )

    if args.state_replay:
        if args.pause_at_event is not None:
            raise ValueError(
                "--pause-at-event is available only for live physics and "
                "cannot be combined with --state-replay"
            )
        override_names = (
            "edge_mm",
            "mass_g",
            "density_scale",
            "friction",
            "finger_down_deg",
            "hand_roll_deg",
            "hand_yaw_deg",
            "hand_rpy_deg",
            "press_mm",
            "root_cube_distance_mm",
            "cube_in_root_mm",
            "cube_rpy_deg",
            "clockwise_orbit_deg",
            "root_delta_cube_mm",
            "wrist_local_rotvec_deg",
            "grasp_target_rad",
            "manipulation_delta_rad",
            "output_dir",
            "contact_environment",
        )
        if any(getattr(args, name) is not None for name in override_names):
            raise ValueError(
                "--state-replay cannot be combined with simulation overrides "
                "or --output-dir"
            )
        if measured_source is not None:
            if measured_source.trace_path is None:  # pragma: no cover - invariant
                raise RuntimeError("selected measured data has no retained trace")
            source = ReplaySource(
                measured_source.config_path,
                measured_source.trace_path,
                measured_source.trajectory,
            )
        else:
            source = resolve_replay_source(
                catalog_path=args.catalog,
                trajectory=args.trajectory,
                config_path=args.config,
                trace_path=args.trace,
                catalog_edge_mm=args.catalog_edge_mm,
                catalog_mapping_mode=args.catalog_mapping_mode,
                catalog_source_alias=args.catalog_source_alias,
                catalog_rank=args.catalog_rank,
            )
        replay_kwargs = {
            "speed": args.speed,
            "loop": bool(args.loop),
            "start_paused": bool(args.start_paused),
        }
        if args.joint_monitor is not None:
            replay_kwargs["joint_monitor"] = args.joint_monitor
        if args.show_joint_pair is not None:
            replay_kwargs["joint_pair"] = tuple(args.show_joint_pair)
        if args.show_coordinate_frames:
            replay_kwargs["show_coordinate_frames"] = True
        replay_in_viewer(source, **replay_kwargs)
        return 0

    source = measured_source or resolve_viewer_source(
        catalog_path=args.catalog,
        trajectory=args.trajectory,
        config_path=args.config,
        trace_path=args.trace,
        catalog_edge_mm=args.catalog_edge_mm,
        catalog_mapping_mode=args.catalog_mapping_mode,
        catalog_source_alias=args.catalog_source_alias,
        catalog_rank=args.catalog_rank,
    )
    source_config = load_config(source.config_path)
    contact_environment = None
    environment_is_override = args.contact_environment is not None
    environment_source = (
        Path(args.contact_environment).expanduser().resolve()
        if args.contact_environment is not None
        else source.contact_environment_path
    )
    if environment_source is not None:
        environment_path = Path(environment_source).expanduser().resolve()
        if not environment_path.is_file():
            raise FileNotFoundError(
                f"contact environment does not exist: {environment_path}"
            )
        environment_document = json.loads(
            environment_path.read_text(encoding="utf-8")
        )
        if (
            isinstance(environment_document, dict)
            and isinstance(environment_document.get("environment"), dict)
        ):
            environment_document = environment_document["environment"]
        contact_environment = ContactEnvironmentSpec.from_config(
            environment_document
        )
    grasp_target_rad = parse_actuator_overrides(
        args.grasp_target_rad,
        option="--grasp-target-rad",
    )
    manipulation_delta_rad = parse_actuator_overrides(
        args.manipulation_delta_rad,
        option="--manipulation-delta-rad",
    )
    config, overridden = apply_viewer_overrides(
        source_config,
        edge_mm=args.edge_mm,
        mass_g=args.mass_g,
        density_scale=args.density_scale,
        friction=args.friction,
        finger_down_deg=args.finger_down_deg,
        hand_roll_deg=args.hand_roll_deg,
        hand_yaw_deg=args.hand_yaw_deg,
        hand_rpy_deg=args.hand_rpy_deg,
        press_mm=args.press_mm,
        root_cube_distance_mm=args.root_cube_distance_mm,
        cube_in_root_mm=args.cube_in_root_mm,
        cube_rpy_deg=args.cube_rpy_deg,
        clockwise_orbit_deg=args.clockwise_orbit_deg,
        root_delta_cube_mm=args.root_delta_cube_mm,
        wrist_local_rotvec_deg=args.wrist_local_rotvec_deg,
        grasp_target_rad=grasp_target_rad or None,
        manipulation_delta_rad=manipulation_delta_rad or None,
    )
    viewer_kwargs = {
        "speed": args.speed,
        "loop": bool(args.loop),
        "start_paused": bool(args.start_paused),
        "output_dir": args.output_dir,
        "parameter_overridden": overridden or environment_is_override,
    }
    if args.joint_monitor is not None:
        viewer_kwargs["joint_monitor"] = args.joint_monitor
    if contact_environment is not None:
        viewer_kwargs["contact_environment"] = contact_environment
    if args.show_joint_pair is not None:
        viewer_kwargs["joint_pair"] = tuple(args.show_joint_pair)
    if args.pause_at_event is not None:
        viewer_kwargs["pause_at_event"] = args.pause_at_event
    if args.show_coordinate_frames:
        viewer_kwargs["show_coordinate_frames"] = True
    result = simulate_in_viewer(source, config, **viewer_kwargs)
    return result.exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run one fixed experiment")
    run_source = run_parser.add_mutually_exclusive_group()
    run_source.add_argument("--config", default=str(DEFAULT_CONFIG))
    run_source.add_argument(
        "--hardest-from",
        metavar="ROBUSTNESS_JSON",
        help="load hardest_passing_config directly from a robustness result",
    )
    run_parser.add_argument("--output-dir")
    run_parser.add_argument("--video", action="store_true")
    run_parser.add_argument("--video-filename", default="nominal.mp4")
    run_parser.add_argument("--no-trace", action="store_true")
    run_parser.add_argument("--edge-mm", type=float)
    run_parser.add_argument("--mass-g", type=float)
    run_parser.add_argument("--friction", type=float)
    run_parser.set_defaults(func=command_run)

    tune_parser = subparsers.add_parser(
        "tune", help="deterministically tune pose and targets"
    )
    tune_parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    tune_parser.add_argument("--output-dir")
    tune_parser.add_argument("--samples", type=int, default=512)
    tune_parser.add_argument("--refine-top", type=int, default=16)
    tune_parser.add_argument("--refine-per", type=int, default=16)
    tune_parser.add_argument(
        "--workers", type=int, default=min(8, os.cpu_count() or 1)
    )
    tune_parser.add_argument("--seed", type=int, default=20260821)
    tune_parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "actual-contact campaigns: authenticate and resume the existing search "
            "workspace without rerunning committed stages"
        ),
    )
    tune_parser.add_argument(
        "--rescue-from",
        metavar="SOURCE_CAMPAIGN",
        help=(
            "schema v14 only: authenticate a completed contact-preserving campaign "
            "as immutable input and run the dedicated refinement/time-warp rescue "
            "in a separate --output-dir"
        ),
    )
    tune_parser.add_argument(
        "--event-rescue-from",
        metavar="COMPLETED_RESCUE",
        help=(
            "schema v14 only: authenticate a completed zero-success refinement/"
            "time-warp rescue and run the separate contact-witness event rescue "
            "in a new --output-dir"
        ),
    )
    tune_parser.add_argument(
        "--adaptive-event-rescue-from",
        metavar="COMPLETED_EVENT_RESCUE",
        help=(
            "schema v14 only: authenticate a completed event-rescue campaign and "
            "run a physical-plan-deduplicated, projected jerk-first rescue in a "
            "new --output-dir"
        ),
    )
    tune_parser.add_argument(
        "--force-debias-rescue-from",
        metavar="COMPLETED_ADAPTIVE_RESCUE",
        help=(
            "schema v14 only: authenticate a completed adaptive-event rescue and "
            "run the separate broad force-debias planning/feedback rescue in a "
            "new --output-dir"
        ),
    )
    tune_parser.add_argument(
        "--micro-jerk-rescue-from",
        metavar="COMPLETED_FORCE_DEBIAS_RESCUE",
        help=(
            "schema v14 only: authenticate a completed force-debias rescue and "
            "run the final fixed-budget, trace-local micro-jerk refinement in a "
            "new --output-dir"
        ),
    )
    tune_parser.add_argument(
        "--contact-mode-pose-rescue-from",
        metavar="RETAINED_V14_CANDIDATE",
        help=(
            "schema v14 only: authenticate one retained jerk-only 79 mm candidate "
            "and run the fixed-budget hand-root/grasp-qpos contact-mode rescue in "
            "a new --output-dir"
        ),
    )
    tune_parser.add_argument(
        "--adaptive-pose-followup-from",
        metavar="COMPLETED_CONTACT_MODE_POSE_CAMPAIGN",
        help=(
            "schema v14 only: authenticate a completed zero-success 256-candidate "
            "contact-mode pose campaign, run 64 one-axis probes, and only if those "
            "have no hard pass run 64 safe sparse pose combinations in a new "
            "--output-dir"
        ),
    )
    tune_parser.add_argument(
        "--reuse-refinement-from",
        metavar="COMPLETED_RESCUE",
        help=(
            "schema v14 rescue only: authenticate and reuse a fully committed "
            "1,024-candidate refinement stage in a new hash-bound workspace"
        ),
    )
    tune_parser.add_argument(
        "--target-success-count",
        type=int,
        choices=(1, 5),
        help=(
            "actual-contact campaigns: stop after the first campaign success or "
            "continue the same deterministic campaign to five (schema v12 targets "
            "grasp success; schemas v9-v11 and v13-v15 target full manipulation success)"
        ),
    )
    tune_parser.add_argument(
        "--evidence-grasp-anchor",
        action="append",
        default=[],
        metavar="DIRECTORY",
        help=(
            "schema v9-v11 actual-contact campaigns: authenticate and prioritize a "
            "reproduced grasp directory; schemas v13-v15 instead use their registered, "
            "hash-bound source evidence"
        ),
    )
    tune_parser.add_argument(
        "--kinematic-samples-per-pitch",
        type=int,
        help="v2 override; default is the versioned experiment budget",
    )
    tune_parser.add_argument(
        "--dynamic-candidates",
        type=int,
        help="v2 override; candidates retained for full dynamics",
    )
    tune_parser.add_argument(
        "--local-refine-seeds",
        type=int,
        help="v2 override; top dynamic seeds to refine",
    )
    tune_parser.add_argument(
        "--local-refine-per-seed",
        type=int,
        help="v2 override; local dynamic trials per seed",
    )
    tune_parser.add_argument(
        "--final-candidates",
        type=int,
        help="v2 override; finalists receiving perturbation probes",
    )
    tune_parser.add_argument(
        "--perturbations-per-final",
        type=int,
        help="v2 override; fixed-seed probes per finalist",
    )
    tune_parser.add_argument(
        "--fallback-physics-candidates",
        type=int,
        help="v2 override; 0 disables the post-nominal size/material fallback",
    )
    tune_parser.add_argument(
        "--fallback-kinematic-samples-per-pitch",
        type=int,
        help="v2 override; alternative-size static samples per palm pitch",
    )
    tune_parser.add_argument(
        "--no-catalog-video",
        action="store_true",
        help=(
            "schema-v4 lightweight validation only: publish trajectory traces "
            "without MP4 files (formal tuning defaults to video)"
        ),
    )
    tune_parser.set_defaults(func=command_tune)

    robustness_parser = subparsers.add_parser(
        "robustness", help="run the experiment's registered robustness suite"
    )
    robustness_parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    robustness_parser.add_argument("--output")
    robustness_parser.add_argument(
        "--workers", type=int, default=min(8, os.cpu_count() or 1)
    )
    robustness_parser.add_argument("--seed", type=int, default=20260821)
    robustness_parser.add_argument(
        "--search-root",
        action="append",
        default=[],
        metavar="CATALOG_OR_DIRECTORY",
        help=(
            "actual-contact campaigns: canonical catalog, result.json, or campaign "
            "directory; schema v12 expects grasp-pose evidence, older schemas expect "
            "manipulation evidence; repeat to merge sources"
        ),
    )
    robustness_parser.add_argument(
        "--post-validation-catalog",
        help=(
            "relative-wrist actual-contact experiments only: canonical "
            "manipulation catalog used for fixed-mass and equal-density "
            "post-validation; defaults to the preferred --search-root catalog"
        ),
    )
    robustness_parser.add_argument(
        "--post-validation-output-dir",
        help=(
            "relative-wrist actual-contact experiments only: resumable "
            "fixed/equal-density post-validation workspace"
        ),
    )
    robustness_parser.add_argument(
        "--resume-post-validation",
        action="store_true",
        help=(
            "relative-wrist actual-contact experiments only: authenticate and "
            "resume its existing post-validation workspace"
        ),
    )
    robustness_parser.add_argument(
        "--render-density-videos",
        action="store_true",
        help=(
            "relative-wrist actual-contact experiments only: render and fully "
            "decode passing equal-density reruns"
        ),
    )
    robustness_parser.add_argument(
        "--contact-preserving-sidecar-output-dir",
        help=(
            "schema v14 only: create a separate authenticated robustness workspace "
            "for one successful --search-root manipulation catalog"
        ),
    )
    robustness_parser.add_argument(
        "--contact-preserving-source-result",
        help=(
            "schema v14 sidecar only: completed source campaign result JSON bound "
            "to the supplied manipulation catalog"
        ),
    )
    robustness_parser.add_argument(
        "--resume-contact-preserving-sidecar",
        action="store_true",
        help="authenticate and resume an existing schema-v14 robustness sidecar",
    )
    robustness_parser.set_defaults(func=command_robustness)

    catalog_parser = subparsers.add_parser(
        "catalog",
        help="rerun selected passing robustness cases and export trajectories",
    )
    catalog_parser.add_argument("--config", required=True)
    catalog_parser.add_argument("--robustness", required=True)
    catalog_parser.add_argument(
        "--grid-index",
        action="append",
        required=True,
        type=int,
        help="zero-based passing robustness grid index; repeat for multiple paths",
    )
    catalog_parser.add_argument("--output-dir", required=True)
    catalog_parser.add_argument("--video", action="store_true")
    catalog_parser.add_argument(
        "--label",
        action="append",
        metavar="GRID_INDEX=LABEL",
        help="optional safe display label; directory names remain stable grid IDs",
    )
    catalog_parser.set_defaults(func=command_catalog)

    grasp_catalog_parser = subparsers.add_parser(
        "grasp-catalog",
        help=(
            "materialize a high-thumb grasp manifest and independently rerun "
            "its acquisition trajectories"
        ),
    )
    grasp_catalog_parser.add_argument("--manifest", required=True)
    grasp_catalog_parser.add_argument("--output-dir", required=True)
    grasp_catalog_parser.add_argument(
        "--no-video",
        action="store_true",
        help="omit MP4 generation while retaining full and acquisition NPZ traces",
    )
    grasp_catalog_parser.set_defaults(func=command_grasp_catalog)

    view_parser = subparsers.add_parser(
        "view",
        help="rerun physics in the interactive MuJoCo Viewer",
    )
    view_parser.add_argument(
        "--catalog",
        help="trajectory catalog JSON; select an entry with --trajectory",
    )
    view_parser.add_argument(
        "--measured-report",
        "--measured-root",
        dest="measured_report",
        help=(
            "measured expanded_report.json, its measured/dynamic directory, or "
            "campaign root; select evidence with --data-index, --candidate-id, "
            "or --select-edge-mm"
        ),
    )
    view_parser.add_argument(
        "--list-data",
        action="store_true",
        help="authenticate and list measured grasp data without opening Viewer",
    )
    measured_selector = view_parser.add_mutually_exclusive_group()
    measured_selector.add_argument(
        "--data-index",
        type=int,
        help="select the one-based index printed by --list-data",
    )
    measured_selector.add_argument(
        "--candidate-id",
        help="select an exact measured candidate ID without numeric rounding",
    )
    measured_selector.add_argument(
        "--select-edge-mm",
        type=float,
        help="select existing measured evidence by cube edge (does not override it)",
    )
    view_parser.add_argument(
        "--edge-rank",
        type=int,
        help="one-based deterministic rank when an edge has multiple data",
    )
    view_parser.add_argument(
        "--trajectory",
        default="nominal",
        help=(
            "catalog label, trajectory_id, grid index, or candidate ID "
            "(default: nominal)"
        ),
    )
    view_parser.add_argument(
        "--catalog-edge-mm",
        type=float,
        help="select a catalog trajectory by stored cube edge metadata",
    )
    view_parser.add_argument(
        "--catalog-mapping-mode",
        choices=("proportional_face_yz", "absolute_face_yz"),
        help="select a catalog trajectory by stored contact-point mapping mode",
    )
    view_parser.add_argument(
        "--catalog-source-alias",
        help="select a catalog trajectory by its authenticated source alias",
    )
    view_parser.add_argument(
        "--catalog-rank",
        type=int,
        default=1,
        help="one-based rank among matching catalog metadata (default: 1)",
    )
    view_parser.add_argument("--config", help="resolved experiment config JSON")
    view_parser.add_argument(
        "--trace",
        help="optional exact-reference NPZ; required for direct --state-replay",
    )
    view_parser.add_argument(
        "--state-replay",
        action="store_true",
        help="animate recorded NPZ states instead of integrating physics",
    )
    view_parser.add_argument(
        "--speed", type=float, default=1.0, help="real-time playback multiplier"
    )
    view_parser.add_argument("--loop", action="store_true", help="loop playback")
    view_parser.add_argument(
        "--start-paused", action="store_true", help="open on the first frame"
    )
    view_parser.add_argument(
        "--output-dir",
        help="save the live run's resolved config, JSON result and NPZ trace",
    )
    view_parser.add_argument("--edge-mm", type=float)
    view_mass = view_parser.add_mutually_exclusive_group()
    view_mass.add_argument("--mass-g", type=float)
    view_mass.add_argument(
        "--density-scale",
        type=float,
        help="scale source material density, accounting for --edge-mm",
    )
    view_parser.add_argument("--friction", type=float)
    view_parser.add_argument("--finger-down-deg", type=float)
    view_parser.add_argument("--hand-roll-deg", type=float)
    view_parser.add_argument("--hand-yaw-deg", type=float)
    view_parser.add_argument(
        "--hand-rpy-deg", type=float, nargs=3, metavar=("R", "P", "Y")
    )
    view_relative_pose = view_parser.add_mutually_exclusive_group()
    view_relative_pose.add_argument("--press-mm", type=float)
    view_relative_pose.add_argument(
        "--root-cube-distance-mm",
        type=float,
        help="set the hand-root to cube-centre distance along the resolved ray",
    )
    view_relative_pose.add_argument(
        "--cube-in-root-mm", type=float, nargs=3, metavar=("X", "Y", "Z")
    )
    view_parser.add_argument(
        "--cube-rpy-deg", type=float, nargs=3, metavar=("R", "P", "Y")
    )
    view_parser.add_argument(
        "--clockwise-orbit-deg",
        type=float,
        help=(
            "replace the stored clockwise orbit about cube-local +Z; omitted "
            "relative-pose coordinates keep their candidate values and the "
            "full transform is rebuilt once from anchor_hand_pose"
        ),
    )
    view_parser.add_argument(
        "--root-delta-cube-mm",
        type=float,
        nargs=3,
        metavar=("DX", "DY", "DZ"),
        help=(
            "replace the stored cube-frame root residual in mm; omitted "
            "orbit/rotation values inherit the candidate values"
        ),
    )
    view_parser.add_argument(
        "--wrist-local-rotvec-deg",
        type=float,
        nargs=3,
        metavar=("RX", "RY", "RZ"),
        help=(
            "replace the stored hand-local rotation vector in degrees; omitted "
            "orbit/translation values inherit the candidate values"
        ),
    )
    view_parser.add_argument(
        "--grasp-target-rad",
        action="append",
        metavar="ACTUATOR=VALUE",
        help="override one grasp target by name; repeat for multiple actuators",
    )
    view_parser.add_argument(
        "--manipulation-delta-rad",
        action="append",
        metavar="ACTUATOR=VALUE",
        help="override one manipulation delta by name; repeat as needed",
    )
    view_parser.add_argument(
        "--joint-monitor",
        metavar="ACTUATOR",
        help=(
            "draw an actuator's joint axis and print target/actual/contact telemetry; "
            "schema-v5 defaults to the thumb bend actuator"
        ),
    )
    view_parser.add_argument(
        "--show-joint-pair",
        nargs=2,
        metavar=("JOINT_A", "JOINT_B"),
        help=(
            "draw two named joint anchors/axes, their connecting line and a "
            "same-length cube +Y reference; press J in Viewer to toggle"
        ),
    )
    view_parser.add_argument(
        "--show-coordinate-frames",
        "--show-frames",
        dest="show_coordinate_frames",
        action="store_true",
        help=(
            "draw the hand-root and cube body frames (X red, Y green, Z blue); "
            "press F in Viewer to toggle them"
        ),
    )
    view_parser.add_argument(
        "--pause-at-event",
        choices=("grasp_lock",),
        help=(
            "live physics only: pause once on the exact post-step frame where "
            "the grasp gate latches; Space resumes, and restart/loop re-arm it"
        ),
    )
    view_parser.add_argument(
        "--contact-environment",
        metavar="ENVIRONMENT_JSON",
        help=(
            "apply a versioned MuJoCo contact/solver environment sidecar; "
            "the grasp/object/controller configuration remains unchanged and "
            "the live run is re-evaluated as a parameter override"
        ),
    )
    view_parser.set_defaults(func=command_view)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
