"""Authenticated sidecar robustness for completed schema-v14 catalogs.

The event-aware rescue campaign deliberately seals its result before any
robustness work is attempted.  This module evaluates a completed manipulation
catalog without adding files to that immutable campaign.  A separate sidecar
workspace binds the source catalog, source result, source ledger, model,
``uv.lock`` and implementation tree in its own manifest.

The physical audit reuses the registered v14 runner: every published nominal
success receives 16 full-reset perturbations, the local-16 leader receives 50
additional perturbations, and at least 45 of those 50 must pass before a
``best_robust`` alias is published.  The alias catalog is created only inside
the sidecar workspace and always refers to the nominal full-reset trajectory,
never to a checkpoint or a perturbation trace.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..actual_contact_grasp_pose_catalog import (
    authenticated_catalog_artifact_paths,
    commit_campaign_stage,
    initialize_or_resume_campaign,
    validate_stage_ledger,
)
from ..artifacts import (
    REPO_ROOT,
    aggregate_source_sha256,
    file_sha256,
    implementation_paths,
    json_text,
    write_json,
)
from ..experiment import resolve_experiment
from ..grasp_pose import canonical_sha256
from .actual_contact_grasp_pose_robustness import (
    V9RobustnessSource,
    discover_v9_robustness_sources,
)
from .contact_preserving_planned_lift_campaign import (
    _default_robustness_runner,
    _publish_robust_alias_catalog,
)


EXPERIMENT_ID = "left_opposed_face_palm_down_contact_preserving_planned_lift"
SIDECAR_SCHEMA_VERSION = 1
SOURCE_SCHEMA_VERSION = 1
DEFAULT_SEED = 20260821
LOCAL_PERTURBATION_COUNT = 16
BEST_PERTURBATION_COUNT = 50
REQUIRED_BEST_PASSES = 45
MAX_NOMINAL_COUNT = 5

_SOURCE_AUDIT_STAGE = "source_audit"
_ROBUSTNESS_STAGE = "robustness_evidence"
_SOURCE_AUDIT_NAME = "source_audit.json"
_EVIDENCE_DIRECTORY = "evidence"
_STAGING_DIRECTORY = ".robustness_evidence.staging"
_REPORT_NAME = "perturbation_report.json"
_ROBUST_CATALOG_DIRECTORY = "robust_manipulation"


RobustnessRunner = Callable[..., Mapping[str, Any]]
AliasPublisher = Callable[..., Path]
CandidateDiscoverer = Callable[
    [Sequence[str | Path]], tuple[V9RobustnessSource, ...]
]
CatalogAuthenticator = Callable[[str | Path], tuple[Path, ...]]


@dataclass(frozen=True, slots=True)
class AuthenticatedV14RobustnessSource:
    """One immutable completed v14 catalog and its campaign result."""

    root: Path
    catalog_path: Path
    result_path: Path
    manifest_path: Path
    ledger_path: Path
    catalog_artifact_paths: tuple[Path, ...]
    successful_sources: tuple[V9RobustnessSource, ...]
    best_nominal_candidate_id: str
    source_authentication_id: str

    @property
    def nominal_candidate_ids(self) -> tuple[str, ...]:
        return tuple(value.candidate_id for value in self.successful_sources)

    @property
    def representative_config_path(self) -> Path:
        path = self.successful_sources[0].config_path
        if path is None:  # Catalog discovery always supplies this path.
            raise RuntimeError("authenticated v14 source lost its config path")
        return path

    def descriptor(self) -> dict[str, Any]:
        artifact_hashes = {
            str(path.relative_to(self.root)): file_sha256(path)
            for path in self.catalog_artifact_paths
        }
        payload = {
            "contact_preserving_robustness_source_schema_version": (
                SOURCE_SCHEMA_VERSION
            ),
            "source_root": str(self.root),
            "source_catalog": str(self.catalog_path),
            "source_catalog_sha256": file_sha256(self.catalog_path),
            "source_result": str(self.result_path),
            "source_result_sha256": file_sha256(self.result_path),
            "source_manifest": str(self.manifest_path),
            "source_manifest_sha256": file_sha256(self.manifest_path),
            "source_ledger": str(self.ledger_path),
            "source_ledger_sha256": file_sha256(self.ledger_path),
            "source_catalog_artifact_set_sha256": canonical_sha256(
                artifact_hashes
            ),
            "nominal_candidate_ids": list(self.nominal_candidate_ids),
            "best_nominal_candidate_id": self.best_nominal_candidate_id,
            "read_only": True,
        }
        observed = canonical_sha256(payload)
        if observed != self.source_authentication_id:
            raise RuntimeError("v14 robustness source changed after authentication")
        return {**payload, "source_authentication_id": observed}


@dataclass(frozen=True, slots=True)
class RobustnessSidecarBackend:
    """Injectable physics/publication boundary used by focused tests."""

    robustness_runner: RobustnessRunner = _default_robustness_runner
    alias_publisher: AliasPublisher = _publish_robust_alias_catalog
    candidate_discoverer: CandidateDiscoverer = discover_v9_robustness_sources
    catalog_authenticator: CatalogAuthenticator = (
        authenticated_catalog_artifact_paths
    )


DEFAULT_BACKEND = RobustnessSidecarBackend()


def _load_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"{label} must be a JSON object")
    return copy.deepcopy(dict(payload))


def _ledger_artifact_paths(root: Path, ledger: Mapping[str, Any]) -> set[Path]:
    observed: set[Path] = set()
    stages = ledger.get("stages", {})
    if not isinstance(stages, Mapping):
        raise RuntimeError("completed v14 source has no stage mapping")
    for record in stages.values():
        if not isinstance(record, Mapping):
            raise RuntimeError("completed v14 source stage is malformed")
        artifacts = record.get("artifacts", {})
        if not isinstance(artifacts, Mapping):
            raise RuntimeError("completed v14 source stage has no artifacts")
        for relative in artifacts:
            path = (root / str(relative)).resolve()
            if path.is_relative_to(root):
                observed.add(path)
    return observed


def authenticate_completed_v14_manipulation_catalog(
    catalog_path: str | Path,
    result_path: str | Path,
    *,
    backend: RobustnessSidecarBackend = DEFAULT_BACKEND,
) -> AuthenticatedV14RobustnessSource:
    """Authenticate a completed, ledger-bound v14 nominal catalog.

    The result must live at the source campaign root and must reference the
    exact supplied catalog.  Every catalog member and both top-level files must
    be covered by the source stage ledger.  A catalog with no full manipulation
    success is useful diagnostic evidence but is not a robustness source.
    """

    catalog = Path(catalog_path).expanduser().resolve()
    result = Path(result_path).expanduser().resolve()
    root = result.parent
    manifest = root / "campaign_manifest.json"
    ledger_path = root / "stage_ledger.json"
    if not catalog.is_relative_to(root):
        raise RuntimeError("v14 robustness catalog must be inside its source campaign")
    if not manifest.is_file() or not ledger_path.is_file():
        raise RuntimeError("v14 robustness source is not a completed campaign")

    ledger = validate_stage_ledger(root)
    catalog_payload = _load_object(catalog, "v14 manipulation catalog")
    result_payload = _load_object(result, "v14 campaign result")
    if catalog_payload.get("complete") is not True:
        raise RuntimeError("v14 manipulation catalog is incomplete")
    if catalog_payload.get("catalog_kind") != "manipulation":
        raise RuntimeError("v14 robustness requires a nominal manipulation catalog")
    if catalog_payload.get("experiment_id") != EXPERIMENT_ID:
        raise RuntimeError("v14 manipulation catalog has the wrong experiment")
    if result_payload.get("complete") is not True:
        raise RuntimeError("v14 source campaign result is incomplete")
    if result_payload.get("experiment_id") != EXPERIMENT_ID:
        raise RuntimeError("v14 source campaign result has the wrong experiment")
    if int(result_payload.get("full_success_count", 0)) <= 0:
        raise RuntimeError("v14 source campaign has no full manipulation success")

    catalogs = result_payload.get("catalogs")
    if not isinstance(catalogs, Mapping):
        raise RuntimeError("v14 source result has no catalog mapping")
    relative_catalog = catalogs.get("manipulation")
    if (
        not isinstance(relative_catalog, str)
        or Path(relative_catalog).is_absolute()
        or ".." in Path(relative_catalog).parts
    ):
        raise RuntimeError("v14 source result has an unsafe manipulation catalog")
    result_catalog = (root / relative_catalog).resolve()
    if not result_catalog.is_relative_to(root) or result_catalog != catalog:
        raise RuntimeError("v14 source result does not bind the supplied catalog")

    artifact_paths = tuple(backend.catalog_authenticator(catalog))
    committed = _ledger_artifact_paths(root, ledger)
    required = {result, *artifact_paths}
    missing = sorted(str(path.relative_to(root)) for path in required - committed)
    if missing:
        raise RuntimeError(
            "v14 source evidence is not committed in its stage ledger: "
            + ", ".join(missing)
        )

    discovered = tuple(backend.candidate_discoverer((catalog,)))
    if not discovered:
        raise RuntimeError("v14 manipulation catalog has no authenticated candidates")
    for source in discovered:
        if (
            source.experiment_id != EXPERIMENT_ID
            or int(source.config.get("schema_version", 0)) != 14
        ):
            raise RuntimeError("v14 robustness source contains a foreign candidate")
    successful = tuple(value for value in discovered if value.nominal_full_success)
    if not successful:
        raise RuntimeError("v14 manipulation catalog has no nominal full success")
    if len(successful) > MAX_NOMINAL_COUNT:
        raise RuntimeError("v14 manipulation catalog exceeds five nominal successes")
    declared_successes = int(catalog_payload.get("success_count", -1))
    if declared_successes != len(successful):
        raise RuntimeError("v14 catalog success count disagrees with its evidence")

    aliases = catalog_payload.get("aliases")
    if not isinstance(aliases, Mapping) or not isinstance(
        aliases.get("best_nominal"), str
    ):
        raise RuntimeError("v14 manipulation catalog has no best_nominal alias")
    by_trajectory = {
        str(value.get("trajectory_id")): str(value.get("candidate_id"))
        for value in catalog_payload.get("trajectories", ())
        if isinstance(value, Mapping)
    }
    best_nominal = by_trajectory.get(str(aliases["best_nominal"]))
    successful_ids = {value.candidate_id for value in successful}
    if best_nominal not in successful_ids:
        raise RuntimeError("v14 best_nominal alias is not a full success")

    artifact_hashes = {
        str(path.relative_to(root)): file_sha256(path) for path in artifact_paths
    }
    descriptor = {
        "contact_preserving_robustness_source_schema_version": (
            SOURCE_SCHEMA_VERSION
        ),
        "source_root": str(root),
        "source_catalog": str(catalog),
        "source_catalog_sha256": file_sha256(catalog),
        "source_result": str(result),
        "source_result_sha256": file_sha256(result),
        "source_manifest": str(manifest),
        "source_manifest_sha256": file_sha256(manifest),
        "source_ledger": str(ledger_path),
        "source_ledger_sha256": file_sha256(ledger_path),
        "source_catalog_artifact_set_sha256": canonical_sha256(artifact_hashes),
        "nominal_candidate_ids": [value.candidate_id for value in successful],
        "best_nominal_candidate_id": best_nominal,
        "read_only": True,
    }
    return AuthenticatedV14RobustnessSource(
        root=root,
        catalog_path=catalog,
        result_path=result,
        manifest_path=manifest,
        ledger_path=ledger_path,
        catalog_artifact_paths=artifact_paths,
        successful_sources=successful,
        best_nominal_candidate_id=str(best_nominal),
        source_authentication_id=canonical_sha256(descriptor),
    )


def build_contact_preserving_robustness_sidecar_manifest(
    source: AuthenticatedV14RobustnessSource,
    *,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Bind source evidence and executable inputs for safe sidecar resume."""

    if int(seed) != DEFAULT_SEED:
        raise ValueError("v14 robustness sidecar seed must be 20260821")
    config = source.representative_config_path
    model = (REPO_ROOT / "xhand_left.xml").resolve()
    lock = (REPO_ROOT / "uv.lock").resolve()
    sources = tuple(implementation_paths())
    bound = {
        "campaign_manifest_schema_version": 1,
        "contact_preserving_robustness_sidecar_schema_version": (
            SIDECAR_SCHEMA_VERSION
        ),
        "experiment_id": EXPERIMENT_ID,
        "seed": int(seed),
        "config_path": str(config),
        "config_sha256": file_sha256(config),
        "model_path": str(model),
        "model_sha256": file_sha256(model),
        "uv_lock_path": str(lock),
        "uv_lock_sha256": file_sha256(lock),
        "actual_qpos_source_manifest_path": str(source.catalog_path),
        "actual_qpos_source_manifest_sha256": file_sha256(source.catalog_path),
        "source_result_path": str(source.result_path),
        "source_result_sha256": file_sha256(source.result_path),
        "source_manifest_sha256": file_sha256(source.manifest_path),
        "source_ledger_sha256": file_sha256(source.ledger_path),
        "source_authentication_id": source.source_authentication_id,
        "source_sha256": aggregate_source_sha256(sources),
        "source_files": [str(path.relative_to(REPO_ROOT)) for path in sources],
        "local_perturbations_per_nominal": LOCAL_PERTURBATION_COUNT,
        "best_perturbations": BEST_PERTURBATION_COUNT,
        "required_best_passes": REQUIRED_BEST_PASSES,
    }
    return {**bound, "campaign_input_sha256": canonical_sha256(bound)}


def _trial_sequence_valid(
    trials: Any,
    *,
    count: int,
    family: str,
    source_candidate_id: str,
) -> bool:
    if not isinstance(trials, list) or len(trials) != count:
        return False
    indices: list[int] = []
    for value in trials:
        if (
            not isinstance(value, Mapping)
            or value.get("family") != family
            or str(value.get("source_candidate_id")) != source_candidate_id
        ):
            return False
        try:
            indices.append(int(value["trial"]))
        except (KeyError, TypeError, ValueError):
            return False
    return sorted(indices) == list(range(count))


def _validate_returned_report(
    report: Mapping[str, Any], source: AuthenticatedV14RobustnessSource
) -> dict[str, Any]:
    """Require the exact registered 16-per-nominal plus leader-50 budget."""

    payload = copy.deepcopy(dict(report))
    if (
        payload.get("complete") is not True
        or int(payload.get("v9_robustness_report_schema_version", 0)) != 1
        or int(
            payload.get("contact_preserving_robustness_schema_version", 0)
        )
        != 1
    ):
        raise RuntimeError("v14 robustness runner returned an incomplete report")
    if payload.get("experiment_id") != EXPERIMENT_ID:
        raise RuntimeError("v14 robustness report has the wrong experiment")
    if int(payload.get("seed", -1)) != DEFAULT_SEED:
        raise RuntimeError("v14 robustness report changed the registered seed")
    if payload.get("best_selection_policy") != "local_perturbation_passes":
        raise RuntimeError("v14 robustness report used the wrong leader policy")
    if payload.get("best_50_selection_alias") != "best_robust_local_16_leader":
        raise RuntimeError("v14 robustness report lost its leader alias")
    if int(payload.get("local_perturbations_per_nominal", -1)) != 16:
        raise RuntimeError("v14 robustness report did not run 16 local trials")
    if int(payload.get("best_perturbation_count", -1)) != 50:
        raise RuntimeError("v14 robustness report did not run 50 leader trials")
    if payload.get("registered_budget_complete") is not True:
        raise RuntimeError("v14 robustness report did not complete its budget")

    expected_ids = set(source.nominal_candidate_ids)
    selected_ids = {str(value) for value in payload.get("selected_nominal_candidate_ids", ())}
    if (
        int(payload.get("selected_nominal_count", -1)) != len(expected_ids)
        or selected_ids != expected_ids
    ):
        raise RuntimeError("v14 robustness report changed its nominal source set")
    per_nominal = payload.get("per_nominal")
    if not isinstance(per_nominal, list) or len(per_nominal) != len(expected_ids):
        raise RuntimeError("v14 robustness report has incomplete local trials")
    observed_local_ids: set[str] = set()
    for value in per_nominal:
        if not isinstance(value, Mapping):
            raise RuntimeError("v14 robustness local record is malformed")
        identifier = str(value.get("candidate_id"))
        observed_local_ids.add(identifier)
        if (
            int(value.get("perturbation_count", -1)) != 16
            or not _trial_sequence_valid(
                value.get("trials"),
                count=16,
                family="per_full_success_local_16",
                source_candidate_id=identifier,
            )
        ):
            raise RuntimeError("v14 robustness local trial set is incomplete")
    if observed_local_ids != expected_ids:
        raise RuntimeError("v14 robustness local records changed their candidates")

    best = payload.get("best_robustness")
    if not isinstance(best, Mapping):
        raise RuntimeError("v14 robustness report has no leader result")
    leader = str(payload.get("best_50_candidate_id", ""))
    if leader not in expected_ids or str(best.get("candidate_id")) != leader:
        raise RuntimeError("v14 robustness leader is not a nominal success")
    if str(payload.get("nominal_best_candidate_id")) != (
        source.best_nominal_candidate_id
    ):
        raise RuntimeError("v14 robustness report changed best_nominal")
    passes = int(best.get("perturbation_passes", -1))
    if (
        int(best.get("perturbation_count", -1)) != 50
        or int(best.get("required_perturbation_passes", -1)) != 45
        or int(payload.get("required_passes", -1)) != 45
        or not 0 <= passes <= 50
        or not _trial_sequence_valid(
            best.get("trials"),
            count=50,
            family="best_first_pose_material_50",
            source_candidate_id=leader,
        )
    ):
        raise RuntimeError("v14 robustness leader trial set is incomplete")
    expected_total = len(expected_ids) * 16 + 50
    if int(payload.get("total_perturbation_count", -1)) != expected_total:
        raise RuntimeError("v14 robustness report has the wrong total trial count")
    expected_robust = bool(payload.get("robust_passed", False) and passes >= 45)
    if bool(payload.get("robust_success", False)) != expected_robust:
        raise RuntimeError("v14 robustness report has an inconsistent 45/50 verdict")
    if bool(best.get("robust_passed", False)) != bool(
        payload.get("robust_passed", False)
    ):
        raise RuntimeError("v14 robustness leader verdict disagrees with the report")
    return payload


def _load_committed_sidecar_report(workspace: Path) -> dict[str, Any] | None:
    ledger = validate_stage_ledger(workspace)
    evidence = workspace / _EVIDENCE_DIRECTORY
    staging = workspace / _STAGING_DIRECTORY
    if _ROBUSTNESS_STAGE not in ledger["stages"]:
        if evidence.exists() or staging.exists():
            raise RuntimeError("uncommitted partial v14 robustness evidence exists")
        return None
    if staging.exists():
        raise RuntimeError("committed v14 robustness sidecar has stale staging data")
    report_path = evidence / _REPORT_NAME
    payload = _load_object(report_path, "committed v14 robustness report")
    if (
        payload.get("complete") is not True
        or int(payload.get("contact_preserving_robustness_sidecar_schema_version", 0))
        != SIDECAR_SCHEMA_VERSION
    ):
        raise RuntimeError("committed v14 robustness sidecar report is incomplete")
    return payload


def run_contact_preserving_robustness_sidecar(
    catalog_path: str | Path,
    result_path: str | Path,
    output_dir: str | Path,
    *,
    resume: bool,
    workers: int,
    seed: int = DEFAULT_SEED,
    backend: RobustnessSidecarBackend = DEFAULT_BACKEND,
) -> dict[str, Any]:
    """Run or authenticate one immutable v14 robustness sidecar workspace."""

    if int(workers) <= 0:
        raise ValueError("workers must be positive")
    if int(seed) != DEFAULT_SEED:
        raise ValueError("v14 robustness sidecar seed must be 20260821")
    source = authenticate_completed_v14_manipulation_catalog(
        catalog_path, result_path, backend=backend
    )
    workspace = Path(output_dir).expanduser().resolve()
    if (
        workspace == source.root
        or workspace.is_relative_to(source.root)
        or source.root.is_relative_to(workspace)
    ):
        raise ValueError("robustness sidecar must be outside its immutable source")
    manifest = build_contact_preserving_robustness_sidecar_manifest(
        source, seed=int(seed)
    )
    initialize_or_resume_campaign(workspace, manifest, resume=bool(resume))

    source_audit_path = workspace / _SOURCE_AUDIT_NAME
    ledger = validate_stage_ledger(workspace)
    descriptor = source.descriptor()
    if _SOURCE_AUDIT_STAGE in ledger["stages"]:
        audit = _load_object(source_audit_path, "v14 robustness source audit")
        if canonical_sha256(audit) != canonical_sha256(
            {"complete": True, "source": descriptor}
        ):
            raise RuntimeError("v14 robustness source audit changed on resume")
    else:
        if source_audit_path.exists():
            raise RuntimeError("uncommitted partial v14 source audit exists")
        audit = {"complete": True, "source": descriptor}
        write_json(source_audit_path, audit)
        commit_campaign_stage(
            workspace,
            _SOURCE_AUDIT_STAGE,
            stage_input={"source_authentication_id": source.source_authentication_id},
            artifacts=(source_audit_path,),
            summary={"complete": True, "nominal_count": len(source.successful_sources)},
        )

    existing = _load_committed_sidecar_report(workspace)
    if existing is not None:
        if existing.get("source", {}).get("source_authentication_id") != (
            source.source_authentication_id
        ):
            raise RuntimeError("committed robustness report changed its source")
        return existing

    staging = workspace / _STAGING_DIRECTORY
    evidence = workspace / _EVIDENCE_DIRECTORY
    if staging.exists() or evidence.exists():
        raise RuntimeError("partial v14 robustness output would be overwritten")
    staging.mkdir()
    staging_report = staging / _REPORT_NAME
    definition = resolve_experiment(source.successful_sources[0].config)
    campaign = definition.contact_preserving_planned_lift_campaign
    if campaign is None:
        raise RuntimeError("v14 robustness source lost its registered campaign")
    raw = backend.robustness_runner(
        source.catalog_path,
        staging_report,
        workers=int(workers),
        seed=int(seed),
        campaign=campaign,
    )
    report = _validate_returned_report(raw, source)
    robust_catalog_relative: str | None = None
    robust_catalog_sha256: str | None = None
    if report["robust_success"]:
        robust_catalog = backend.alias_publisher(
            source.catalog_path,
            staging / _ROBUST_CATALOG_DIRECTORY,
            robust_candidate_id=str(report["best_50_candidate_id"]),
        ).resolve()
        if not robust_catalog.is_relative_to(staging):
            raise RuntimeError("v14 robust alias publisher escaped its staging root")
        backend.catalog_authenticator(robust_catalog)
        robust_catalog_relative = str(
            (Path(_EVIDENCE_DIRECTORY) / robust_catalog.relative_to(staging))
        )
        robust_catalog_sha256 = file_sha256(robust_catalog)

    report = {
        **report,
        "contact_preserving_robustness_sidecar_schema_version": (
            SIDECAR_SCHEMA_VERSION
        ),
        "source": descriptor,
        "robust_catalog": robust_catalog_relative,
        "robust_catalog_sha256": robust_catalog_sha256,
    }
    write_json(staging_report, report)

    # A long 16-per-nominal plus leader-50 audit must not silently span a
    # source or implementation edit.  Re-authenticate both the immutable
    # campaign and the complete executable manifest before publication.
    final_source = authenticate_completed_v14_manipulation_catalog(
        source.catalog_path, source.result_path, backend=backend
    )
    if final_source.source_authentication_id != source.source_authentication_id:
        raise RuntimeError("v14 robustness source changed during execution")
    final_manifest = build_contact_preserving_robustness_sidecar_manifest(
        final_source, seed=int(seed)
    )
    if canonical_sha256(final_manifest) != canonical_sha256(manifest):
        raise RuntimeError("v14 robustness code or inputs changed during execution")
    staging.rename(evidence)
    artifact_paths = tuple(path for path in sorted(evidence.rglob("*")) if path.is_file())
    commit_campaign_stage(
        workspace,
        _ROBUSTNESS_STAGE,
        stage_input={
            "source_authentication_id": source.source_authentication_id,
            "source_catalog_sha256": file_sha256(source.catalog_path),
            "source_result_sha256": file_sha256(source.result_path),
            "local_perturbations_per_nominal": LOCAL_PERTURBATION_COUNT,
            "best_perturbations": BEST_PERTURBATION_COUNT,
            "required_best_passes": REQUIRED_BEST_PASSES,
        },
        artifacts=artifact_paths,
        summary={
            "complete": True,
            "robust_success": bool(report["robust_success"]),
            "best_perturbation_passes": int(
                report["best_robustness"]["perturbation_passes"]
            ),
        },
    )
    validate_stage_ledger(workspace)
    return copy.deepcopy(report)


def contact_preserving_robustness_sidecar_console(
    report: Mapping[str, Any], output_dir: str | Path
) -> dict[str, Any]:
    """Return the stable console payload used by the CLI integration."""

    root = Path(output_dir).expanduser().resolve()
    best = report.get("best_robustness", {})
    robust_relative = report.get("robust_catalog")
    return {
        "experiment_id": report.get("experiment_id"),
        "selected_nominal_count": int(report.get("selected_nominal_count", 0)),
        "total_perturbation_count": int(report.get("total_perturbation_count", 0)),
        "best_perturbation_passes": int(best.get("perturbation_passes", 0)),
        "required_best_passes": REQUIRED_BEST_PASSES,
        "robust_success": bool(report.get("robust_success", False)),
        "report": str(root / _EVIDENCE_DIRECTORY / _REPORT_NAME),
        "robust_catalog": (
            None
            if robust_relative is None
            else str((root / str(robust_relative)).resolve())
        ),
        "output_dir": str(root),
    }


def command_contact_preserving_robustness_sidecar(args: Any) -> int:
    """Argparse-compatible helper; parser registration remains in ``cli.py``."""

    report = run_contact_preserving_robustness_sidecar(
        args.catalog,
        args.source_result,
        args.output_dir,
        resume=bool(args.resume),
        workers=int(args.workers),
        seed=int(getattr(args, "seed", DEFAULT_SEED)),
    )
    print(
        json_text(
            contact_preserving_robustness_sidecar_console(
                report, args.output_dir
            )
        )
    )
    return 0 if report["robust_success"] else 2


__all__ = [
    "AuthenticatedV14RobustnessSource",
    "BEST_PERTURBATION_COUNT",
    "DEFAULT_BACKEND",
    "DEFAULT_SEED",
    "LOCAL_PERTURBATION_COUNT",
    "REQUIRED_BEST_PASSES",
    "RobustnessSidecarBackend",
    "authenticate_completed_v14_manipulation_catalog",
    "build_contact_preserving_robustness_sidecar_manifest",
    "command_contact_preserving_robustness_sidecar",
    "contact_preserving_robustness_sidecar_console",
    "run_contact_preserving_robustness_sidecar",
]
