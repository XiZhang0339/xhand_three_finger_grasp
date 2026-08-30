"""Resumable orchestration for the schema-v15 near-zero joint-pair campaign.

The module owns deterministic identities, stage coverage, resume ledgers and
catalog aliases.  MuJoCo work is an injected boundary, which keeps recovery
logic testable without replacing the production physics implementation.
"""

from __future__ import annotations

import copy
import json
import math
import os
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ..actual_contact_grasp_pose_catalog import (
    commit_campaign_stage,
    initialize_or_resume_campaign,
    validate_stage_ledger,
)
from ..artifacts import (
    REPO_ROOT,
    aggregate_source_sha256,
    file_sha256,
    implementation_paths,
    write_json,
)
from ..config import ACTIVE_ACTUATORS, load_config, validate_config
from ..experiment import resolve_experiment
from ..grasp_pose import canonical_sha256
from ..v15_identity import install_v15_top_level_identities
from .joint_pair_near_zero_campaign import (
    EXPERIMENT_ID,
    AuthenticatedNearZeroSource,
    JointPairNearZeroBudget,
    StaticPerturbation,
    authenticate_near_zero_source,
    generate_static_perturbations,
    grasp_control_variants,
    joint_pair_feedback_variants,
    near_zero_candidate_rank,
    rank_near_zero_candidates,
)


RUNNER_SCHEMA_VERSION = 1
CATALOG_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class V15CampaignJob:
    stage: str
    index: int
    parent_candidate_id: int | None
    payload: Mapping[str, Any]
    candidate_id: int

    def descriptor(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "index": self.index,
            "parent_candidate_id": self.parent_candidate_id,
            "payload": copy.deepcopy(dict(self.payload)),
            "candidate_id": self.candidate_id,
        }


def _job(stage: str, index: int, parent: int | None, payload: Mapping[str, Any]) -> V15CampaignJob:
    identity = {
        "runner_schema_version": RUNNER_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "stage": stage,
        "index": int(index),
        "parent_candidate_id": parent,
        "payload": copy.deepcopy(dict(payload)),
    }
    candidate_id = 15_200_000_000_000_000 + int(canonical_sha256(identity)[:13], 16) % 100_000_000_000_000
    return V15CampaignJob(stage, int(index), parent, identity["payload"], candidate_id)


def build_static_jobs(budget: JointPairNearZeroBudget = JointPairNearZeroBudget()) -> tuple[V15CampaignJob, ...]:
    return tuple(
        V15CampaignJob(
            "static_filter", value.index, None, value.as_mapping(), value.candidate_id
        )
        for value in generate_static_perturbations(budget)
    )


def _record_id(value: Mapping[str, Any]) -> int:
    candidate = value.get("candidate_id")
    if isinstance(candidate, bool):
        raise ValueError("candidate_id cannot be bool")
    return int(candidate)


def _selected(records: Sequence[Mapping[str, Any]], count: int, *, field: str | None = None) -> tuple[Mapping[str, Any], ...]:
    eligible = [
        value for value in records
        if field is None or bool(value.get(field, False))
    ]
    return tuple(rank_near_zero_candidates(eligible)[: int(count)])


def build_grasp_jobs(static_records: Sequence[Mapping[str, Any]], budget: JointPairNearZeroBudget = JointPairNearZeroBudget()) -> tuple[V15CampaignJob, ...]:
    parents = _selected(static_records, budget.static_retain_count, field="static_pass")
    result: list[V15CampaignJob] = []
    for parent in parents:
        for variant in grasp_control_variants():
            result.append(_job("dynamic_grasp", len(result), _record_id(parent), {
                "static_candidate_id": _record_id(parent),
                "controller_variant": variant.as_mapping(),
            }))
    return tuple(result)


def build_plan_jobs(grasp_records: Sequence[Mapping[str, Any]], budget: JointPairNearZeroBudget = JointPairNearZeroBudget()) -> tuple[V15CampaignJob, ...]:
    parents = _selected(grasp_records, budget.measured_grasp_retain_count, field="grasp_success")
    result: list[V15CampaignJob] = []
    for parent in parents:
        for plan_index in range(budget.sequential_plans_per_grasp):
            result.append(_job("sequential_planning", len(result), _record_id(parent), {
                "grasp_candidate_id": _record_id(parent),
                "plan_index": plan_index,
                "segment_count": 20,
                "probe_count_per_segment": 17,
                "requires_full_reset_rerun": True,
            }))
    return tuple(result)


def build_feedback_jobs(plan_records: Sequence[Mapping[str, Any]], budget: JointPairNearZeroBudget = JointPairNearZeroBudget()) -> tuple[V15CampaignJob, ...]:
    result: list[V15CampaignJob] = []
    for parent in plan_records[: budget.sequential_plan_count]:
        for variant in joint_pair_feedback_variants():
            result.append(_job("feedback_grid", len(result), _record_id(parent), {
                "plan_candidate_id": _record_id(parent),
                "feedback_variant": variant.as_mapping(),
            }))
    return tuple(result)


def _summary_mapping(record: Mapping[str, Any]) -> Mapping[str, Any]:
    summary = record.get("summary", {})
    return summary if isinstance(summary, Mapping) else {}


def _nested_mapping(
    value: Mapping[str, Any], *names: str
) -> Mapping[str, Any]:
    current: Any = value
    for name in names:
        if not isinstance(current, Mapping):
            return {}
        current = current.get(name, {})
    return current if isinstance(current, Mapping) else {}


def _finite_number(value: Any, fallback: float) -> float:
    if isinstance(value, bool):
        return fallback
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    return result if math.isfinite(result) else fallback


def _feedback_refinement_evidence(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract the full-reset evidence needed before expensive refinement.

    Missing fields fail closed.  In particular, a planner checkpoint summary
    cannot masquerade as an executed manipulation merely because it contains
    optimistic predicted path metrics.
    """

    summary = _summary_mapping(record)
    stage = _nested_mapping(summary, "stage_status")
    checks = _nested_mapping(summary, "checks")
    metrics = _nested_mapping(summary, "metrics")
    pair = _nested_mapping(metrics, "joint_pair_alignment")
    contact = _nested_mapping(metrics, "contact_preserving_planned_lift")
    collision = _nested_mapping(metrics, "active_finger_self_collision")
    duties = _nested_mapping(contact, "target_face_effective_duty")

    grasp_success = bool(
        record.get("grasp_success", stage.get("grasp_success", False))
    )
    operation_samples = _finite_number(
        contact.get("operation_sample_count"), 0.0
    )
    manipulation_start_step = int(
        _finite_number(metrics.get("manipulation_start_step"), -1.0)
    )
    # ``checks.operation_executed`` historically means that the controller
    # completed its whole event sequence.  A safety abort can make that check
    # false even after thousands of genuine MANIPULATE samples.  Refinement is
    # intended precisely for those near misses, so actual execution is proven
    # from the raw-trace-derived start frame and a 250-sample minimum instead.
    operation_executed = bool(
        manipulation_start_step >= 0 and operation_samples >= 250.0
    )
    reported_operation_executed_check = checks.get("operation_executed") is True
    manipulation_completed = checks.get("manipulation_completed") is True
    pair_p95_deg = _finite_number(
        pair.get("operation_p95_deg", pair.get("operation_angle_p95_deg")),
        math.inf,
    )
    pair_max_deg = _finite_number(
        pair.get("operation_max_deg", pair.get("operation_angle_max_deg")),
        math.inf,
    )
    simultaneous_duty = _finite_number(
        contact.get("simultaneous_target_face_effective_duty"), -math.inf
    )
    finger_duties = {
        name: _finite_number(duties.get(name), -math.inf)
        for name in ("thumb", "index", "mid")
    }
    minimum_finger_duty = min(finger_duties.values())
    no_self_collision = bool(
        checks.get(
            "no_active_finger_self_collision",
            collision.get("collision_free", False),
        )
    )
    median_lift_m = _finite_number(
        metrics.get("operation_median_lift_m", metrics.get("median_lift_m")),
        -math.inf,
    )
    requirements = {
        "grasp_success": grasp_success,
        "operation_executed": operation_executed,
        "joint_pair_p95_at_most_0p5_deg": pair_p95_deg <= 0.5 + 1e-12,
        "joint_pair_max_at_most_1_deg": pair_max_deg <= 1.0 + 1e-12,
        "simultaneous_target_face_contact_duty_at_least_0p99": (
            simultaneous_duty + 1e-12 >= 0.99
        ),
        "no_active_finger_self_collision": no_self_collision,
    }
    return {
        "eligible": all(requirements.values()),
        "requirements": requirements,
        "failed_requirements": sorted(
            name for name, passed in requirements.items() if not passed
        ),
        "manipulation_start_step": manipulation_start_step,
        "operation_sample_count": int(operation_samples),
        "reported_operation_executed_check": (
            reported_operation_executed_check
        ),
        "manipulation_completed": manipulation_completed,
        "pair_p95_deg": pair_p95_deg,
        "pair_max_deg": pair_max_deg,
        "simultaneous_target_face_contact_duty": simultaneous_duty,
        "minimum_finger_target_face_contact_duty": minimum_finger_duty,
        "median_lift_m": median_lift_m,
    }


def _feedback_refinement_rank(record: Mapping[str, Any]) -> tuple[Any, ...]:
    evidence = _feedback_refinement_evidence(record)
    summary = _summary_mapping(record)
    full_success = bool(
        record.get("full_success", summary.get("passed", False))
    )
    candidate = _record_id(record)
    return (
        not full_success,
        -float(evidence["median_lift_m"]),
        -float(evidence["simultaneous_target_face_contact_duty"]),
        -float(evidence["minimum_finger_target_face_contact_duty"]),
        float(evidence["pair_p95_deg"]),
        float(evidence["pair_max_deg"]),
        near_zero_candidate_rank(record),
        candidate,
    )


def _feedback_fallback_rank(record: Mapping[str, Any]) -> tuple[Any, ...]:
    evidence = _feedback_refinement_evidence(record)
    return (
        -sum(bool(value) for value in evidence["requirements"].values()),
        not bool(evidence["requirements"]["operation_executed"]),
        _feedback_refinement_rank(record),
    )


def _persisted_refinement_evidence(
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    result = copy.deepcopy(dict(evidence))
    for name in (
        "pair_p95_deg",
        "pair_max_deg",
        "simultaneous_target_face_contact_duty",
        "minimum_finger_target_face_contact_duty",
        "median_lift_m",
    ):
        value = result[name]
        if not math.isfinite(float(value)):
            result[name] = None
    return result


def _feedback_plan_parent_id(record: Mapping[str, Any]) -> int:
    parent = record.get("parent_candidate_id")
    if parent is None:
        # Synthetic/injected backends predating the physical stage link each
        # feedback record to itself.  The production backend always persists
        # the true sequential-plan parent.
        return _record_id(record)
    if isinstance(parent, bool):
        raise ValueError("feedback parent_candidate_id cannot be bool")
    return int(parent)


def select_refinement_feedback_parents(
    feedback_records: Sequence[Mapping[str, Any]],
    count: int,
) -> tuple[dict[str, Any], ...]:
    """Choose at most one feedback result per distinct sequential plan.

    Hard-eligible winners are selected first.  If fewer than ``count`` plans
    produced an eligible result, one deterministic best near-miss from each
    remaining plan fills the unused slots.  Every returned record carries the
    selection mode and failed hard filters, so fallback cannot be mistaken for
    a validated manipulation candidate.
    """

    if isinstance(count, bool) or int(count) <= 0:
        raise ValueError("refinement parent count must be a positive integer")
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for record in feedback_records:
        grouped.setdefault(_feedback_plan_parent_id(record), []).append(record)

    eligible_winners: list[dict[str, Any]] = []
    fallback_winners: list[dict[str, Any]] = []
    for plan_id in sorted(grouped):
        ordered = sorted(grouped[plan_id], key=_feedback_refinement_rank)
        eligible = [
            record
            for record in ordered
            if bool(_feedback_refinement_evidence(record)["eligible"])
        ]
        if eligible:
            selected = eligible[0]
        else:
            # Prefer the fallback that cleared the most hard evidence, then a
            # real operation over a non-started prediction, before applying
            # the same lift/contact/alignment ordering as eligible records.
            selected = min(ordered, key=_feedback_fallback_rank)
        materialized = copy.deepcopy(dict(selected))
        evidence = _feedback_refinement_evidence(selected)
        persisted_evidence = _persisted_refinement_evidence(evidence)
        materialized["refinement_parent_selection"] = {
            "schema_version": 1,
            "plan_candidate_id": int(plan_id),
            "mode": (
                "eligible" if evidence["eligible"] else "deterministic_fallback"
            ),
            **persisted_evidence,
        }
        if evidence["eligible"]:
            eligible_winners.append(materialized)
        else:
            fallback_winners.append(materialized)

    eligible_winners.sort(key=_feedback_refinement_rank)
    fallback_winners.sort(key=_feedback_refinement_rank)
    selected = eligible_winners[: int(count)]
    if len(selected) < int(count):
        selected.extend(fallback_winners[: int(count) - len(selected)])
    return tuple(selected)


def build_refinement_jobs(feedback_records: Sequence[Mapping[str, Any]], budget: JointPairNearZeroBudget = JointPairNearZeroBudget()) -> tuple[V15CampaignJob, ...]:
    parents = select_refinement_feedback_parents(
        feedback_records, budget.feedback_refine_plan_count
    )
    result: list[V15CampaignJob] = []
    for parent in parents:
        selection = parent["refinement_parent_selection"]
        for local_index in range(budget.feedback_refine_per_plan):
            result.append(_job("feedback_refinement", len(result), _record_id(parent), {
                "feedback_candidate_id": _record_id(parent),
                "plan_candidate_id": int(selection["plan_candidate_id"]),
                "local_index": local_index,
                "reprobe_if_pose_changed": True,
                "refinement_parent_selection": copy.deepcopy(selection),
            }))
    return tuple(result)


def _exact_plan_lineage_id(
    record: Mapping[str, Any],
    records_by_id: Mapping[int, Mapping[str, Any]],
) -> int:
    explicit = record.get("plan_candidate_id")
    if explicit is not None:
        if isinstance(explicit, bool):
            raise ValueError("plan_candidate_id cannot be bool")
        return int(explicit)
    current = record
    visited: set[int] = set()
    while True:
        candidate_id = _record_id(current)
        if candidate_id in visited:
            raise RuntimeError("candidate parent lineage contains a cycle")
        visited.add(candidate_id)
        stage = str(current.get("stage", ""))
        parent = current.get("parent_candidate_id")
        if parent is None:
            return candidate_id
        if isinstance(parent, bool):
            raise ValueError("parent_candidate_id cannot be bool")
        parent_id = int(parent)
        if stage == "feedback_grid":
            return parent_id
        parent_record = records_by_id.get(parent_id)
        if parent_record is None:
            return parent_id
        current = parent_record


def select_exact_rerun_parents(
    candidate_records: Sequence[Mapping[str, Any]],
    count: int,
) -> tuple[dict[str, Any], ...]:
    """Select exact reruns without regressing to stationary low-angle plans."""

    if isinstance(count, bool) or int(count) <= 0:
        raise ValueError("exact rerun count must be a positive integer")
    by_id = {_record_id(record): record for record in candidate_records}
    tiers: dict[str, list[dict[str, Any]]] = {
        "full_success": [],
        "eligible_operation": [],
        "deterministic_fallback": [],
    }
    for raw in candidate_records:
        record = copy.deepcopy(dict(raw))
        evidence = _feedback_refinement_evidence(record)
        summary = _summary_mapping(record)
        full_success = bool(
            record.get("full_success", summary.get("passed", False))
        )
        mode = (
            "full_success"
            if full_success
            else "eligible_operation"
            if evidence["eligible"]
            else "deterministic_fallback"
        )
        record["exact_parent_selection"] = {
            "schema_version": 1,
            "mode": mode,
            "plan_lineage_candidate_id": _exact_plan_lineage_id(
                record, by_id
            ),
            **_persisted_refinement_evidence(evidence),
        }
        tiers[mode].append(record)

    selected: list[dict[str, Any]] = []
    selected_lineages: set[int] = set()
    for mode in ("full_success", "eligible_operation", "deterministic_fallback"):
        ordered = sorted(
            tiers[mode],
            key=(
                _feedback_fallback_rank
                if mode == "deterministic_fallback"
                else _feedback_refinement_rank
            ),
        )
        # Within each priority tier, maximize plan-lineage coverage before
        # spending exact reruns on a second controller from the same plan.
        unique: list[dict[str, Any]] = []
        duplicate: list[dict[str, Any]] = []
        tier_lineages = set(selected_lineages)
        for record in ordered:
            lineage = int(
                record["exact_parent_selection"]["plan_lineage_candidate_id"]
            )
            if lineage in tier_lineages:
                duplicate.append(record)
            else:
                unique.append(record)
                tier_lineages.add(lineage)
        for record in (*unique, *duplicate):
            if len(selected) >= int(count):
                break
            selected.append(record)
            selected_lineages.add(
                int(record["exact_parent_selection"]["plan_lineage_candidate_id"])
            )
        if len(selected) >= int(count):
            break
    return tuple(selected)


def build_exact_rerun_jobs(candidate_records: Sequence[Mapping[str, Any]], budget: JointPairNearZeroBudget = JointPairNearZeroBudget()) -> tuple[V15CampaignJob, ...]:
    parents = select_exact_rerun_parents(
        candidate_records, budget.exact_rerun_count
    )
    return tuple(
        _job("exact_rerun", index, _record_id(parent), {
            "candidate_id_to_rerun": _record_id(parent),
            "timestep_s": 0.001,
            "from_initial_no_contact_state": True,
            "retain_trace": True,
            "exact_parent_selection": copy.deepcopy(
                parent["exact_parent_selection"]
            ),
        })
        for index, parent in enumerate(parents)
    )


def build_perturbation_jobs(final_records: Sequence[Mapping[str, Any]], budget: JointPairNearZeroBudget = JointPairNearZeroBudget()) -> tuple[V15CampaignJob, ...]:
    parents = _selected(final_records, budget.final_candidate_count, field="full_success")
    result: list[V15CampaignJob] = []
    for parent in parents:
        for perturbation_index in range(budget.perturbations_per_final):
            result.append(_job("local_perturbation", len(result), _record_id(parent), {
                "nominal_candidate_id": _record_id(parent),
                "perturbation_index": perturbation_index,
                "seed": budget.seed,
            }))
    return tuple(result)


def build_robustness_jobs(final_records: Sequence[Mapping[str, Any]], budget: JointPairNearZeroBudget = JointPairNearZeroBudget()) -> tuple[V15CampaignJob, ...]:
    parents = _selected(final_records, 1, field="full_success")
    if not parents:
        return ()
    parent = parents[0]
    return tuple(
        _job("robustness", index, _record_id(parent), {
            "nominal_candidate_id": _record_id(parent), "trial_index": index,
            "seed": budget.seed,
        })
        for index in range(budget.robustness_trials)
    )


def materialize_v15_seed_config(
    template: Mapping[str, Any], source: AuthenticatedNearZeroSource
) -> dict[str, Any]:
    """Migrate immutable v14 measured geometry into a valid v15 seed."""

    resolved = copy.deepcopy(dict(template))
    original = source.config
    for key in (
        "cube", "hand_pose", "pose_constraints", "contact_topology",
        "contact_point_plan", "closure_alignment", "fingertip_contact_preferences",
        "grasp_pose", "pose_preservation", "control",
        "contact_force_targets_n", "contact_feedback",
    ):
        if key in original:
            resolved[key] = copy.deepcopy(original[key])
    with np.load(source.trace_path, allow_pickle=False) as trace:
        actual = np.asarray(trace["grasp_pose_actual_qpos_rad"], dtype=np.float64)
    if actual.shape != (len(ACTIVE_ACTUATORS),) or not np.isfinite(actual).all():
        raise RuntimeError("authenticated source lost its actual grasp qpos")
    resolved["grasp_pose"]["nominal_joint_qpos_rad"] = {
        name: float(actual[index]) for index, name in enumerate(ACTIVE_ACTUATORS)
    }
    resolved["schema_version"] = 15
    resolved["experiment_id"] = EXPERIMENT_ID
    resolved["control_protocol"] = copy.deepcopy(template["control_protocol"])
    for key in ("settle_s", "close_s", "verify_timeout_s", "stable_window_s"):
        if key in original.get("control_protocol", {}):
            resolved["control_protocol"][key] = original["control_protocol"][key]
    resolved["control_protocol"]["strategy"] = (
        "grasp_verify_then_joint_pair_aligned_contact_preserving_planned_lift"
    )
    resolved["candidate_metadata"] = {
        "schema_version": 15,
        "candidate_id": build_static_jobs()[0].candidate_id,
        "cube_pose_sampled": False,
        "hand_root_fixed_during_simulation": True,
        "source_candidate_id": int(source.result["candidate_id"]),
        "source_id": source.source_id,
        "source_config_sha256": file_sha256(source.config_path),
        "source_result_sha256": file_sha256(source.result_path),
        "source_trace_sha256": file_sha256(source.trace_path),
        "requires_physical_replanning": True,
    }
    install_v15_top_level_identities(resolved)
    validate_config(resolved)
    return resolved


def build_v15_campaign_manifest(
    config_path: str | Path,
    *,
    repository_root: str | Path = REPO_ROOT,
    budget: JointPairNearZeroBudget = JointPairNearZeroBudget(),
) -> dict[str, Any]:
    config_path = Path(config_path).expanduser().resolve()
    root = Path(repository_root).expanduser().resolve()
    config = load_config(config_path)
    definition = resolve_experiment(config)
    if definition.experiment_id != EXPERIMENT_ID or int(config.get("schema_version", 0)) != 15:
        raise ValueError("v15 campaign manifest requires the registered near-zero config")
    source = authenticate_near_zero_source(root)
    source_files = implementation_paths()
    bound = {
        "joint_pair_near_zero_campaign_manifest_schema_version": RUNNER_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "seed": budget.seed,
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "model_sha256": file_sha256(root / "xhand_left.xml"),
        "uv_lock_sha256": file_sha256(root / "uv.lock"),
        "actual_qpos_source_manifest_sha256": file_sha256(source.catalog_path),
        "source_sha256": aggregate_source_sha256(source_files),
        "source_id": source.source_id,
        "source_artifact_sha256": {
            "config": file_sha256(source.config_path),
            "result": file_sha256(source.result_path),
            "trace": file_sha256(source.trace_path),
            "catalog": file_sha256(source.catalog_path),
        },
        "budget": budget.as_mapping(),
    }
    return {**bound, "campaign_input_sha256": canonical_sha256(bound)}


class StageRunner(Protocol):
    def __call__(
        self, stage: str, jobs: Sequence[V15CampaignJob], workspace: Path,
        context: Mapping[str, Any],
    ) -> Sequence[Mapping[str, Any]]: ...


@dataclass(frozen=True, slots=True)
class V15CampaignBackend:
    stage_runner: StageRunner


def _run_stage(
    workspace: Path, stage: str, jobs: Sequence[V15CampaignJob],
    backend: V15CampaignBackend, context: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    ledger = validate_stage_ledger(workspace)
    report_path = workspace / "stages" / stage / "report.json"
    if stage in ledger["stages"]:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        return tuple(report["records"])
    raw = backend.stage_runner(stage, jobs, workspace, context)
    records = tuple(copy.deepcopy(dict(value)) for value in raw)
    expected_ids = [job.candidate_id for job in jobs]
    observed_ids = [_record_id(value) for value in records]
    if observed_ids != expected_ids:
        raise RuntimeError(f"stage {stage} did not return every job in canonical order")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "joint_pair_near_zero_stage_report_schema_version": 1,
        "complete": True,
        "stage": stage,
        "job_count": len(jobs),
        "job_sha256": canonical_sha256([job.descriptor() for job in jobs]),
        "records": list(records),
    }
    write_json(report_path, payload)
    artifacts = [report_path]
    for record in records:
        for field in ("config_path", "result_path", "trace_path"):
            raw_path = record.get(field)
            if raw_path is not None:
                path = Path(str(raw_path)).expanduser().resolve()
                if path.is_file() and path.is_relative_to(workspace):
                    artifacts.append(path)
    commit_campaign_stage(
        workspace, stage,
        stage_input={"job_sha256": payload["job_sha256"], "context_sha256": canonical_sha256(context)},
        artifacts=tuple(dict.fromkeys(artifacts)),
        summary={"job_count": len(jobs)},
    )
    return records


def _copy_or_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def publish_v15_viewer_catalog(
    records: Sequence[Mapping[str, Any]], destination: str | Path,
    *, robust_candidate_id: int | None = None,
) -> Path:
    """Publish up to five authenticated full-reset entries and truthful aliases."""

    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    ranked = list(rank_near_zero_candidates(records)[:5])
    entries: list[dict[str, Any]] = []
    for rank, record in enumerate(ranked, 1):
        candidate_id = _record_id(record)
        trajectory_id = f"candidate_{candidate_id}"
        member = destination / trajectory_id
        paths: dict[str, str | None] = {}
        hashes: dict[str, str] = {}
        for field, filename in (("config_path", "resolved_config.json"), ("result_path", "result.json"), ("trace_path", "trace.npz")):
            raw = record.get(field)
            if raw is None:
                paths[field.removesuffix("_path")] = None
                continue
            source = Path(str(raw)).expanduser().resolve()
            if not source.is_file():
                raise FileNotFoundError(source)
            target = member / filename
            _copy_or_link(source, target)
            key = field.removesuffix("_path")
            paths[key] = f"{trajectory_id}/{filename}"
            hashes[key] = file_sha256(target)
        entries.append({
            "trajectory_id": trajectory_id,
            "candidate_id": candidate_id,
            "classification": "success" if bool(record.get("full_success")) else "diagnostic",
            "grasp_success": bool(record.get("grasp_success")),
            "full_success": bool(record.get("full_success")),
            "rank": rank,
            "aliases": [],
            "artifacts": {**paths, "video": None, "sha256": hashes},
        })
    aliases = {f"pair_rank_{index:02d}": entry["trajectory_id"] for index, entry in enumerate(entries, 1)}
    successes = [entry for entry in entries if entry["full_success"]]
    if successes:
        aliases["best_first"] = successes[0]["trajectory_id"]
        aliases["best_nominal"] = successes[0]["trajectory_id"]
    elif entries:
        aliases["best_attempt"] = entries[0]["trajectory_id"]
    if robust_candidate_id is not None:
        match = [entry for entry in successes if entry["candidate_id"] == int(robust_candidate_id)]
        if match:
            aliases["best_robust"] = match[0]["trajectory_id"]
    for entry in entries:
        entry["aliases"] = sorted(alias for alias, target in aliases.items() if target == entry["trajectory_id"])
    payload = {
        "trajectory_catalog_schema_version": 1,
        "joint_pair_near_zero_viewer_catalog_schema_version": CATALOG_SCHEMA_VERSION,
        "complete": True,
        "experiment_id": EXPERIMENT_ID,
        "catalog_kind": "manipulation",
        "selection_policy": "full_success_then_robustness_then_alignment_then_contact",
        "success_count": len(successes),
        "aliases": aliases,
        "trajectories": entries,
    }
    path = destination / "catalog.json"
    write_json(path, payload)
    return path


def run_joint_pair_near_zero_campaign(
    config_path: str | Path, output_dir: str | Path, *, resume: bool,
    target_success_count: int, backend: V15CampaignBackend,
    repository_root: str | Path = REPO_ROOT,
    budget: JointPairNearZeroBudget = JointPairNearZeroBudget(),
    publish_catalog: bool = True,
) -> dict[str, Any]:
    """Run or resume every declared v15 stage through final publication."""

    if target_success_count not in (1, 5):
        raise ValueError("target_success_count must be one or five")
    manifest = build_v15_campaign_manifest(config_path, repository_root=repository_root, budget=budget)
    workspace = initialize_or_resume_campaign(output_dir, manifest, resume=resume)
    source = authenticate_near_zero_source(repository_root)
    seed = materialize_v15_seed_config(load_config(config_path), source)
    seed_path = workspace / "source" / "resolved_seed_config.json"
    if not seed_path.exists():
        seed_path.parent.mkdir(parents=True)
        write_json(seed_path, seed)
        commit_campaign_stage(workspace, "source_migration", stage_input={"source_id": source.source_id}, artifacts=(seed_path,))
    elif canonical_sha256(json.loads(seed_path.read_text("utf-8"))) != canonical_sha256(seed):
        raise RuntimeError("resumed v15 migrated seed changed")
    context: dict[str, Any] = {"seed_config_path": str(seed_path), "target_success_count": target_success_count}
    static = _run_stage(workspace, "static_filter", build_static_jobs(budget), backend, context)
    grasp_jobs = build_grasp_jobs(static, budget)
    grasp = _run_stage(workspace, "dynamic_grasp", grasp_jobs, backend, {**context, "parents": static})
    plan_jobs = build_plan_jobs(grasp, budget)
    plans = _run_stage(workspace, "sequential_planning", plan_jobs, backend, {**context, "parents": grasp})
    feedback_jobs = build_feedback_jobs(plans, budget)
    feedback = _run_stage(workspace, "feedback_grid", feedback_jobs, backend, {**context, "parents": plans})
    refine_jobs = build_refinement_jobs(feedback, budget)
    refined = _run_stage(workspace, "feedback_refinement", refine_jobs, backend, {**context, "parents": feedback})
    exact_jobs = build_exact_rerun_jobs((*feedback, *refined), budget)
    exact = _run_stage(workspace, "exact_rerun", exact_jobs, backend, {**context, "parents": (*feedback, *refined)})
    perturb_jobs = build_perturbation_jobs(exact, budget)
    perturb = _run_stage(workspace, "local_perturbation", perturb_jobs, backend, {**context, "parents": exact})
    pass_counts: dict[int, int] = {}
    for record in perturb:
        parent = int(record.get("parent_candidate_id", -1))
        pass_counts[parent] = pass_counts.get(parent, 0) + int(bool(record.get("full_success")))
    ranked_exact = []
    for record in exact:
        value = copy.deepcopy(dict(record))
        value["perturbation_pass_count"] = pass_counts.get(_record_id(record), 0)
        ranked_exact.append(value)
    robust_jobs = build_robustness_jobs(ranked_exact, budget)
    robustness = _run_stage(workspace, "robustness", robust_jobs, backend, {**context, "parents": ranked_exact})
    robust_parent = None
    if robustness and sum(bool(value.get("full_success")) for value in robustness) >= budget.robustness_required_passes:
        robust_parent = robust_jobs[0].parent_candidate_id
    catalog_path = None
    if publish_catalog:
        catalog_root = workspace / "catalogs" / f"target_{target_success_count}" / "manipulation"
        if not catalog_root.exists():
            catalog_path = publish_v15_viewer_catalog(ranked_exact, catalog_root, robust_candidate_id=robust_parent)
            commit_campaign_stage(workspace, f"catalog_target_{target_success_count}", stage_input={"exact_stage": file_sha256(workspace / "stages/exact_rerun/report.json")}, artifacts=tuple(path for path in catalog_root.rglob("*") if path.is_file()))
        else:
            catalog_path = catalog_root / "catalog.json"
    successes = [value for value in ranked_exact if bool(value.get("full_success"))]
    return {
        "joint_pair_near_zero_campaign_result_schema_version": 1,
        "complete": True,
        "experiment_id": EXPERIMENT_ID,
        "workspace": str(workspace),
        "target_success_count": target_success_count,
        "full_success_count": len(successes),
        "target_reached": len(successes) >= target_success_count,
        "robust_candidate_id": robust_parent,
        "catalog_path": str(catalog_path) if catalog_path else None,
        "stage_counts": {
            "static_filter": len(static), "dynamic_grasp": len(grasp),
            "sequential_planning": len(plans), "feedback_grid": len(feedback),
            "feedback_refinement": len(refined), "exact_rerun": len(exact),
            "local_perturbation": len(perturb), "robustness": len(robustness),
        },
    }


__all__ = [
    "V15CampaignBackend", "V15CampaignJob", "build_exact_rerun_jobs",
    "build_feedback_jobs", "build_grasp_jobs", "build_plan_jobs",
    "build_refinement_jobs", "build_robustness_jobs", "build_static_jobs",
    "build_v15_campaign_manifest", "materialize_v15_seed_config",
    "publish_v15_viewer_catalog", "run_joint_pair_near_zero_campaign",
    "select_exact_rerun_parents",
    "select_refinement_feedback_parents",
]
