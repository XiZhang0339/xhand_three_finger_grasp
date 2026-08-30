"""Canonical top-level identity checks for schema-v14 configurations.

Schema v14 deliberately separates the object, measured grasp, object/grasp
pair, offline planner and runtime controller identities.  The first three are
domain hashes over the resolved configuration.  Planner/controller hashes
have several historical, versioned payloads because the rescue campaigns add
new planning stages without rewriting sealed artifacts.

The tune template contains none of the five IDs.  A pre-planning source-pair
configuration may contain only the first three; this is an explicit search
intermediate and those three values are still recomputed here.  Once a
planner or controller ID is installed (or a non-search run context is added),
all five IDs are required and verified.
"""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping
from typing import Any

from .grasp_pose import canonical_sha256, grasp_pose_context


V14_TOP_LEVEL_ID_FIELDS = (
    "object_config_id",
    "grasp_pose_id",
    "grasp_object_pair_id",
    "planner_id",
    "controller_id",
)
V14_PAIR_ID_FIELDS = V14_TOP_LEVEL_ID_FIELDS[:3]
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

# Eight warm-start sweep configs were sealed before the selected warm source
# and scale were persisted in candidate_metadata.  Their whole canonical
# config bodies are the only honest authentication boundary available.  This
# registry is deliberately exact and finite: it must never become a wildcard
# for arbitrary configs lacking planner provenance.
_LEGACY_WARM_CONFIG_PLANNER_IDS = {
    "739dc13626e320ec6b97d55713e5d190777cd7ad43d8c3433e29f5dd6220bf89": "c011a7c1038c85dc0ce7231fe1cbc91aa8cf50703336ee4ca851681f072d4249",
    "dbf4a446ba09d7441eb71e32c65d9ed0e592d47318130047a9e2d5d8d38630cc": "ff024bc9637fc6240717d16d7681764c298b160dc11458f2e6a060d5b0bec760",
    "82524915f14345e78253c2bfc0b1e5df93748801660f13a669178b6c7e7174e5": "f636ca3491c697933fc387b51f95e94dd2b7a2fa48b043b10a4c9bf1dde2a064",
    "e32a2955719ebbd000dc3c19d9564c72a405d4b2f25de6892075805b70bcdc7e": "32c47f68f330f252e9b0e8b2e59a52453ecfde48b952653615b53521384b1077",
    "e3a094d5aac419b1ae753f3c73ca031edb9c128f14f07a5690c0a4cb045757b2": "13aede02f9ec1ee5576f0696e5c58a1ec1fdbffece2c7914ccc500723d9af162",
    "9e5b8daee328e5e44bdf4f7af775ceb242e702b6bc49734d1a25218842508f26": "c3ffea2d3db3478f5e9f954054495e034388cf4ee0f921fe104aacb9302caf72",
    "120a602b45ef8e3f5693279272a15c5139fa3e3eccafb5d07dcb7a7c587849ee": "095855fe535fc5f2499b7af93b1b584179bec8aa0d6e416100a530e9f1e73bf7",
    "118b29640984a4d134498947c743b70dacd73f270b58ed71ed0998d8a305f2e6": "39c91167b3f1bc6b8eb2954ba9c8b9a7d04c6f3055613712274adeeaf774d7fc",
}


def _domain_id(domain: str, payload: Any) -> str:
    return canonical_sha256(
        {
            "identity_schema_version": 1,
            "domain": str(domain),
            "payload": payload,
        }
    )


def object_configuration_context(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the physical object/initial-support identity payload."""

    cube = config.get("cube")
    scene = config.get("scene")
    if not isinstance(cube, Mapping) or not isinstance(scene, Mapping):
        raise ValueError("object identity requires cube and scene mappings")
    return {
        "cube": copy.deepcopy(dict(cube)),
        "support": {
            key: copy.deepcopy(scene[key])
            for key in sorted(scene)
            if key.startswith("support_") or key == "floor_z_m"
        },
    }


def v14_object_config_id(config: Mapping[str, Any]) -> str:
    return _domain_id(
        "v14_object_configuration", object_configuration_context(config)
    )


def v14_grasp_pose_id(config: Mapping[str, Any]) -> str:
    return _domain_id("v14_actual_grasp_pose", grasp_pose_context(config))


def v14_grasp_object_pair_id(config: Mapping[str, Any]) -> str:
    return _domain_id(
        "v14_grasp_object_pair",
        {
            "object_config_id": v14_object_config_id(config),
            "grasp_pose_id": v14_grasp_pose_id(config),
        },
    )


def v14_upright_rescue_resolved_payload_sha256(
    config: Mapping[str, Any],
) -> str:
    """Hash every physical/control field mutable by the upright rescue."""

    control = config.get("control")
    protocol = config.get("control_protocol")
    plan = config.get("manipulation_plan")
    feedback = config.get("contact_feedback")
    targets = config.get("contact_force_targets_n")
    if not all(
        isinstance(value, Mapping)
        for value in (control, protocol, plan, feedback, targets)
    ):
        raise ValueError("upright rescue identity requires resolved control blocks")
    assert isinstance(control, Mapping)
    assert isinstance(protocol, Mapping)
    assert isinstance(plan, Mapping)
    assert isinstance(feedback, Mapping)
    assert isinstance(targets, Mapping)
    return canonical_sha256(
        {
            "schema_version": 1,
            "kind": "v14_upright_grasp_resolved_candidate_payload",
            "grasp_object_pair_id": config["grasp_object_pair_id"],
            "hand_pose": copy.deepcopy(config["hand_pose"]),
            "grasp_pose": copy.deepcopy(config["grasp_pose"]),
            "precontact_targets_rad": copy.deepcopy(
                control["precontact_targets_rad"]
            ),
            "contact_preload_targets_rad": copy.deepcopy(
                control["contact_preload_targets_rad"]
            ),
            "manipulation_delta_rad": copy.deepcopy(
                control["manipulation_delta_rad"]
            ),
            "close_profile": copy.deepcopy(control.get("close_profile")),
            "close_s": float(protocol["close_s"]),
            "plan_id": plan["plan_id"],
            "feedback_id": feedback["feedback_id"],
            "target_id": targets["target_id"],
        }
    )


def v14_sequential_planner_id(
    report_id: str, attempt_report_id: str
) -> str:
    """Identity used by the sequential checkpoint relinearized planner."""

    return canonical_sha256(
        {
            "schema_version": 1,
            "kind": "sequential_checkpoint_relinearized_contact_plan",
            "report_id": str(report_id),
            "attempt_report_id": str(attempt_report_id),
            "fresh_full_reset_required": True,
        }
    )


def v14_base_controller_id(
    config: Mapping[str, Any], *, bind_planner: bool
) -> str:
    """Identity used by the original v14 plan/feedback campaign.

    Early plan records bound the planner explicitly.  The subsequent feedback
    grid intentionally used the same payload without ``planner_id``.  Both are
    sealed historical formats and are kept distinct here.
    """

    payload = {
        "schema_version": 1,
        "grasp_object_pair_id": config["grasp_object_pair_id"],
        "plan_id": config["manipulation_plan"]["plan_id"],
        "target_id": config["contact_force_targets_n"]["target_id"],
        "feedback_id": config["contact_feedback"]["feedback_id"],
    }
    if bind_planner:
        payload["planner_id"] = config["planner_id"]
    return canonical_sha256(payload)


def v14_time_warp_controller_id(config: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "schema_version": 1,
            "kind": "v14_contact_preserving_time_warp_controller",
            "grasp_object_pair_id": config["grasp_object_pair_id"],
            "plan_id": config["manipulation_plan"]["plan_id"],
            "target_id": config["contact_force_targets_n"]["target_id"],
            "feedback_id": config["contact_feedback"]["feedback_id"],
            "planner_id": config["planner_id"],
            "contact_preload_targets_rad": config["control"][
                "contact_preload_targets_rad"
            ],
        }
    )


def v14_joint_refined_controller_id(config: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "schema_version": 1,
            "kind": "v14_joint_refined_controller",
            "grasp_object_pair_id": config["grasp_object_pair_id"],
            "plan_id": config["manipulation_plan"]["plan_id"],
            "target_id": config["contact_force_targets_n"]["target_id"],
            "feedback_id": config["contact_feedback"]["feedback_id"],
            "contact_preload_targets_rad": config["control"][
                "contact_preload_targets_rad"
            ],
        }
    )


def v14_refinement_rescue_controller_id(config: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "schema_version": 1,
            "kind": "v14_contact_preserving_refinement_rescue_controller",
            "grasp_object_pair_id": config["grasp_object_pair_id"],
            "planner_id": config["planner_id"],
            "plan_id": config["manipulation_plan"]["plan_id"],
            "target_id": config["contact_force_targets_n"]["target_id"],
            "feedback_id": config["contact_feedback"]["feedback_id"],
            "contact_preload_targets_rad": config["control"][
                "contact_preload_targets_rad"
            ],
        }
    )


def _candidate_metadata(config: Mapping[str, Any]) -> Mapping[str, Any]:
    value = config.get("candidate_metadata", {})
    return value if isinstance(value, Mapping) else {}


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _direct_planner_expectations(
    config: Mapping[str, Any],
) -> tuple[tuple[str, str], ...]:
    """Return exact planner hashes derivable from persisted lineage.

    Some sealed time-warp/adaptive-event artifacts predate self-contained
    planner payload persistence.  Their planner is still bound by a canonical
    controller payload, so those families intentionally return no direct
    expectation here and use the compatibility path in the main validator.
    Newer rescue families persist every input and are recomputed exactly.
    """

    metadata = _candidate_metadata(config)

    upright_rescue = _mapping(
        metadata.get("v14_upright_grasp_self_collision_rescue")
    )
    if upright_rescue is not None:
        schema_version = int(upright_rescue["schema_version"])
        candidate_sha = str(upright_rescue["candidate_sha256"])
        payload: dict[str, Any] = {
            "schema_version": schema_version,
            "kind": "v14_upright_grasp_self_collision_rescue_plan",
            "source_authentication_id": upright_rescue[
                "source_authentication_id"
            ],
            "grasp_source_candidate_id": upright_rescue[
                "grasp_source_candidate_id"
            ],
            "plan_source_candidate_id": upright_rescue[
                "plan_source_candidate_id"
            ],
            "source_plan_id": upright_rescue["source_plan_id"],
            "resolved_plan_id": config["manipulation_plan"]["plan_id"],
            "grasp_object_pair_id": config["grasp_object_pair_id"],
            "candidate_sha256": candidate_sha,
        }
        if schema_version >= 2:
            recomputed_candidate_sha = canonical_sha256(
                {
                    "schema_version": schema_version,
                    "source_authentication_id": upright_rescue[
                        "source_authentication_id"
                    ],
                    "budget": upright_rescue["budget"],
                    "sequence": int(upright_rescue["sequence"]),
                    "normalized_sample": upright_rescue["normalized_sample"],
                }
            )
            if candidate_sha != recomputed_candidate_sha:
                raise ValueError(
                    "schema-v14 upright rescue candidate lineage changed"
                )
            expected_candidate_id = (
                15_800_000_000_000_000
                + int(candidate_sha[:13], 16) % 100_000_000_000_000
            )
            if int(upright_rescue["candidate_id"]) != expected_candidate_id:
                raise ValueError("schema-v14 upright rescue candidate ID changed")
            if schema_version >= 3:
                budget = _mapping(upright_rescue.get("budget"))
                normalized = upright_rescue.get("normalized_sample")
                if budget is None or not isinstance(normalized, (list, tuple)):
                    raise ValueError(
                        "schema-v14 upright rescue diagnostics are incomplete"
                    )
                scale_range = budget.get("index_bend_plan_scale_range")
                if not isinstance(scale_range, (list, tuple)) or len(scale_range) != 2:
                    raise ValueError(
                        "schema-v14 upright rescue bend-scale budget changed"
                    )
                expected_scale = float(scale_range[0]) + (
                    float(normalized[-1]) + 1.0
                ) * 0.5 * (float(scale_range[1]) - float(scale_range[0]))
                if not math.isclose(
                    float(upright_rescue["index_bend_plan_scale"]),
                    expected_scale,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        "schema-v14 upright rescue bend-scale diagnostic changed"
                    )
                diagnostic_budget_fields = {
                    "grasp_geometry_blend_fraction": (
                        "grasp_geometry_blend_fraction"
                    ),
                    "use_grasp_close_timing": "use_grasp_close_timing",
                    "index_bend_max_abs_delta_rad": (
                        "index_bend_max_abs_delta_rad"
                    ),
                }
                for diagnostic_key, budget_key in diagnostic_budget_fields.items():
                    if upright_rescue.get(diagnostic_key) != budget.get(budget_key):
                        raise ValueError(
                            "schema-v14 upright rescue budget diagnostic changed"
                        )
            resolved_payload_sha = v14_upright_rescue_resolved_payload_sha256(
                config
            )
            if upright_rescue.get("resolved_candidate_payload_sha256") != (
                resolved_payload_sha
            ):
                raise ValueError(
                    "schema-v14 upright rescue resolved payload changed"
                )
            payload["resolved_candidate_payload_sha256"] = resolved_payload_sha
        expected = canonical_sha256(payload)
        return (("upright_grasp_self_collision_rescue", expected),)

    adaptive_pose = _mapping(metadata.get("v14_adaptive_pose_followup"))
    if adaptive_pose is not None:
        expected = canonical_sha256(
            {
                "schema_version": int(adaptive_pose["schema_version"]),
                "kind": "v14_adaptive_pose_followup_reused_plan",
                "source_authentication_id": adaptive_pose[
                    "source_authentication_id"
                ],
                "center_candidate_id": adaptive_pose["center_candidate_id"],
                "source_plan_id": config["manipulation_plan"]["plan_id"],
                "grasp_object_pair_id": config["grasp_object_pair_id"],
                "candidate_sha256": adaptive_pose["candidate_sha256"],
            }
        )
        return (("adaptive_pose_followup", expected),)

    contact_mode = _mapping(metadata.get("v14_contact_mode_pose_rescue"))
    if contact_mode is not None:
        expected = canonical_sha256(
            {
                "schema_version": int(contact_mode["schema_version"]),
                "kind": "v14_contact_mode_pose_rescue_reused_plan",
                "source_authentication_id": contact_mode[
                    "source_authentication_id"
                ],
                "source_plan_id": config["manipulation_plan"]["plan_id"],
                "grasp_object_pair_id": config["grasp_object_pair_id"],
                "candidate_sha256": contact_mode["candidate_sha256"],
            }
        )
        return (("contact_mode_pose_rescue", expected),)

    bounded_micro = _mapping(metadata.get("v14_bounded_micro_jerk_rescue"))
    if bounded_micro is not None:
        expected = canonical_sha256(
            {
                "schema_version": int(bounded_micro["schema_version"]),
                "kind": "v14_bounded_micro_jerk_plan",
                "source_center_id": bounded_micro["source_center_id"],
                "normalized_parameters": bounded_micro[
                    "normalized_parameters"
                ],
                "physical_parameters": bounded_micro["physical_parameters"],
            }
        )
        return (("bounded_micro_jerk_rescue", expected),)

    # The two immutable design-evidence configs were produced before the
    # bounded-rescue generator existed.  They retain older force metadata,
    # but their final planner payload was not persisted in source; do not
    # incorrectly validate them against that older stage.  Recovery configs
    # that also carry bounded-rescue metadata were handled above.
    if "v14_micro_jerk_design_evidence" in metadata:
        return ()

    force_debias = _mapping(
        metadata.get("v14_contact_preserving_force_debias_rescue")
    )
    if force_debias is not None:
        expected = canonical_sha256(
            {
                "schema_version": int(force_debias["schema_version"]),
                "kind": "v14_contact_preserving_force_debias_plan",
                "source_center_id": force_debias["source_center_id"],
                "descriptor_id": force_debias["descriptor_id"],
                "polytope_id": force_debias["polytope_id"],
                "stage": force_debias["stage"],
                "projected_parameters": force_debias[
                    "projected_parameters"
                ],
            }
        )
        return (("force_debias_rescue", expected),)

    # These two historical stages did not persist the parent physical-plan
    # digest/source-plan ID required by their planner payload.  Do not guess.
    if (
        "v14_contact_preserving_adaptive_event_rescue" in metadata
        or "v14_contact_preserving_time_warp" in metadata
    ):
        return ()

    formal = _mapping(metadata.get("formal_warm_start"))
    if formal is not None:
        terminal = config["control"]["manipulation_delta_rad"]
        expected = canonical_sha256(
            {
                "schema_version": int(formal["schema_version"]),
                "kind": "authenticated_79mm_c008_contact_compensated_v1",
                "source_config_sha256": formal["source_config_sha256"],
                "source_result_semantic_sha256": formal[
                    "source_result_semantic_sha256"
                ],
                "terminal_delta_rad": terminal,
                "fresh_full_reset_required": True,
            }
        )
        return (("formal_warm_start", expected),)

    sequential = _mapping(metadata.get("sequential_checkpoint_planning"))
    if sequential is not None:
        return (
            (
                "sequential_checkpoint_planning",
                v14_sequential_planner_id(
                    str(sequential["report_id"]),
                    str(sequential["attempt_report_id"]),
                ),
            ),
        )

    nonlinear = _mapping(
        metadata.get("v14_authenticated_nonlinear_warm_start")
    )
    if nonlinear is not None:
        expected = canonical_sha256(
            {
                "schema_version": int(nonlinear["schema_version"]),
                "kind": "authenticated_v13_nonlinear_warm_start",
                "source_config_sha256": nonlinear[
                    "source_config_sha256"
                ],
                "source_result_semantic_sha256": nonlinear[
                    "source_result_semantic_sha256"
                ],
                "scale": float(nonlinear["scale"]),
            }
        )
        return (("authenticated_v13_nonlinear_warm_start", expected),)

    # Eight sealed nonlinear warm-start configs predate a dedicated planner
    # metadata block.  The top-level source hashes describe the grasp source,
    # not necessarily the selected nonlinear warm-start record, so they are
    # intentionally not used to guess the missing planner payload.
    return ()


def _controller_expectations(
    config: Mapping[str, Any],
) -> tuple[tuple[str, str, bool], ...]:
    """Return ``(format, expected_id, binds_planner)`` candidates."""

    return (
        ("base_feedback", v14_base_controller_id(config, bind_planner=False), False),
        ("base_planner_bound", v14_base_controller_id(config, bind_planner=True), True),
        ("time_warp", v14_time_warp_controller_id(config), True),
        ("joint_refinement", v14_joint_refined_controller_id(config), False),
        (
            "refinement_rescue",
            v14_refinement_rescue_controller_id(config),
            True,
        ),
    )


def _non_search_run_context(config: Mapping[str, Any]) -> bool:
    context = config.get("run_context")
    if not isinstance(context, Mapping):
        return False
    # Both registered contexts execute a resolved physical run.  Search
    # intermediates do not install run_context at all.
    return context.get("kind") in {"parameter_override_run", "robustness_trial"}


def _present_identity_fields(config: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        name
        for name in V14_TOP_LEVEL_ID_FIELDS
        if isinstance(config.get(name), str) and bool(config.get(name))
    )


def _validate_sha_field(config: Mapping[str, Any], name: str) -> str:
    value = config.get(name)
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"schema-v14 {name} must be a lowercase SHA-256")
    return value


def validate_v14_top_level_identities(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute and validate schema-v14 top-level identities.

    Returns a small audit record used by trace persistence/evaluation.  Schema
    <=13 is a no-op so legacy numerical paths remain untouched.
    """

    if int(config.get("schema_version", 1)) != 14:
        return {
            "state": "not_schema_v14",
            "values": {},
            "planner_verification": "not_applicable",
            "controller_format": "not_applicable",
        }

    present = _present_identity_fields(config)
    present_set = set(present)
    resolved = bool(
        present_set.intersection({"planner_id", "controller_id"})
        or _non_search_run_context(config)
    )

    if resolved:
        missing = [
            name for name in V14_TOP_LEVEL_ID_FIELDS if name not in present_set
        ]
        if missing:
            raise ValueError(
                "resolved schema-v14 config requires all five top-level IDs; "
                f"missing {', '.join(missing)}"
            )
    elif present_set and present_set != set(V14_PAIR_ID_FIELDS):
        raise ValueError(
            "schema-v14 search intermediate may contain either no top-level "
            "IDs or exactly object_config_id, grasp_pose_id and "
            "grasp_object_pair_id"
        )

    values = {
        name: str(config.get(name, "")) if name in present_set else ""
        for name in V14_TOP_LEVEL_ID_FIELDS
    }
    if not present_set:
        return {
            "state": "unresolved_template",
            "values": values,
            "planner_verification": "absent",
            "controller_format": "absent",
        }

    for name in present:
        _validate_sha_field(config, name)

    recomputed_pair = {
        "object_config_id": v14_object_config_id(config),
        "grasp_pose_id": v14_grasp_pose_id(config),
        "grasp_object_pair_id": v14_grasp_object_pair_id(config),
    }
    for name, expected in recomputed_pair.items():
        if values[name] != expected:
            raise ValueError(
                f"schema-v14 {name} does not match the resolved configuration"
            )

    if not resolved:
        return {
            "state": "preplanning_source_pair",
            "values": values,
            "planner_verification": "absent",
            "controller_format": "absent",
        }

    planner_expectations = _direct_planner_expectations(config)
    planner_matches = tuple(
        label
        for label, expected in planner_expectations
        if values["planner_id"] == expected
    )
    if planner_expectations and not planner_matches:
        raise ValueError(
            "schema-v14 planner_id does not match its persisted canonical "
            "planner lineage"
        )
    legacy_warm_body = canonical_sha256(config)
    legacy_warm = (
        not planner_expectations
        and _LEGACY_WARM_CONFIG_PLANNER_IDS.get(legacy_warm_body)
        == values["planner_id"]
    )
    controller_matches = tuple(
        (label, binds_planner)
        for label, expected, binds_planner in _controller_expectations(config)
        if values["controller_id"] == expected
    )
    if not controller_matches:
        raise ValueError(
            "schema-v14 controller_id does not match any registered canonical "
            "controller payload"
        )
    if (
        not planner_expectations
        and not any(binds for _, binds in controller_matches)
        and not legacy_warm
    ):
        raise ValueError(
            "schema-v14 planner_id lacks self-contained lineage and is not "
            "an explicitly registered legacy warm config"
        )

    return {
        "state": "resolved",
        "values": values,
        "planner_verification": (
            ",".join(planner_matches)
            if planner_matches
            else "legacy_lineage_unavailable"
            if legacy_warm
            else "legacy_controller_bound_lineage"
        ),
        "controller_format": ",".join(
            label for label, _ in controller_matches
        ),
    }


def v14_identity_trace_values(config: Mapping[str, Any]) -> dict[str, str]:
    """Return validated scalar values to persist in schema-v14 NPZ traces."""

    audit = validate_v14_top_level_identities(config)
    if audit["state"] == "not_schema_v14":
        return {}
    return dict(audit["values"])


__all__ = [
    "V14_PAIR_ID_FIELDS",
    "V14_TOP_LEVEL_ID_FIELDS",
    "object_configuration_context",
    "v14_base_controller_id",
    "v14_grasp_object_pair_id",
    "v14_grasp_pose_id",
    "v14_identity_trace_values",
    "v14_joint_refined_controller_id",
    "v14_object_config_id",
    "v14_refinement_rescue_controller_id",
    "v14_sequential_planner_id",
    "v14_time_warp_controller_id",
    "v14_upright_rescue_resolved_payload_sha256",
    "validate_v14_top_level_identities",
]
