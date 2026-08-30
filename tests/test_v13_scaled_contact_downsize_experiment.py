from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from xhand_grasp.config import load_config, validate_config
from xhand_grasp.experiment import (
    ContactPointPlanParameters,
    ScaledContactDownsizeCampaignParameters,
    ScaledContactMappingParameters,
    get_experiment,
)
from xhand_grasp.tuning.contact_point_downsize import audit_downsize_source
from xhand_grasp.experiments.opposed_face_palm_down_90mm_contact_point_targeted_actual_grasp_pose import (
    EXPERIMENT_ID as V12_EXPERIMENT_ID,
)
from xhand_grasp.experiments.opposed_face_palm_down_scaled_centered_spread_actual_grasp_then_lift import (
    EDGES_M,
    EXPERIMENT_ID,
    SCALED_CONTACT_DOWNSIZE_CAMPAIGN,
    SOURCE_ALIASES,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPO_ROOT / "grasp_configs" / (
    "left_opposed_face_palm_down_scaled_centered_spread_actual_grasp_then_lift.json"
)
MANIFEST = REPO_ROOT / (
    "artifacts/left_opposed_face_palm_down_scaled_centered_spread_actual_"
    "grasp_then_lift/source_manifests/centered_spread_authenticated_sources.json"
)
SOURCE_ROOT = REPO_ROOT / (
    "artifacts/left_opposed_face_palm_down_larger_relative_wrist_pose_actual_contact_"
    "smooth_vertical_lift/tune/centered_thumb_expanded_opposing_spread_from_"
    "positive_y_best_v1"
)


def _template() -> dict:
    return json.loads(TEMPLATE.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_v13_template_loads_and_registered_budget_is_exact() -> None:
    config = load_config(TEMPLATE)
    definition = get_experiment(EXPERIMENT_ID)
    campaign = definition.scaled_contact_downsize_campaign

    assert config["schema_version"] == 13
    assert config["experiment_id"] == EXPERIMENT_ID
    assert config["search"]["shared_faces"] == ["+X"]
    assert campaign is SCALED_CONTACT_DOWNSIZE_CAMPAIGN
    assert EDGES_M == tuple(value / 1000.0 for value in range(60, 89))
    assert campaign.source_aliases == SOURCE_ALIASES
    assert campaign.mapping_modes == (
        "proportional_face_yz",
        "absolute_face_yz",
    )
    assert campaign.stratum_count == 174
    assert campaign.maximum_dynamic_grasp_candidate_count == 2_088
    assert campaign.local_pose_count == 116
    assert campaign.selected_grasp_count == 87
    assert config["search"]["budget"] == campaign.budget_config()
    assert config["cube"]["mass_kg"] == pytest.approx(0.160)
    assert config["cube"]["friction"] == pytest.approx(0.8)
    assert config["actual_contact_grasp_pose_campaign"]["validation_labels"] == {
        "grasp": "validated_fixed_160g_scaled_grasp_ablation",
        "manipulation": "validated_fixed_160g_scaled_lift_ablation",
        "robust": "validated_fixed_160g_scaled_robust_full_success_ablation",
    }


def test_mapping_modes_and_full_target_circle_edge_guard() -> None:
    campaign = SCALED_CONTACT_DOWNSIZE_CAMPAIGN
    proportional = campaign.mapped_face_point_m(
        face="-X",
        reference_face_yz_m=(0.010, -0.020),
        edge_m=0.060,
        mapping_mode="proportional_face_yz",
    )
    assert proportional == pytest.approx(
        (-0.030, 0.010 * 0.060 / 0.089, -0.020 * 0.060 / 0.089)
    )
    assert campaign.mapped_face_point_m(
        face="+X",
        reference_face_yz_m=(0.0275, 0.0),
        edge_m=0.060,
        mapping_mode="absolute_face_yz",
    ) == pytest.approx((0.030, 0.0275, 0.0))
    with pytest.raises(ValueError, match="target circle"):
        campaign.mapped_face_point_m(
            face="+X",
            reference_face_yz_m=(0.027500001, 0.0),
            edge_m=0.060,
            mapping_mode="absolute_face_yz",
        )


def test_v13_requires_frozen_plan_and_authenticated_mapping_derivation() -> None:
    config = _template()
    plan = ContactPointPlanParameters.from_config(config["contact_point_plan"])
    mapping = ScaledContactMappingParameters.from_config(
        config["scaled_contact_mapping"]
    )
    assert mapping.validate_for_campaign(
        SCALED_CONTACT_DOWNSIZE_CAMPAIGN, plan
    ) is plan

    missing_plan = copy.deepcopy(config)
    missing_plan.pop("contact_point_plan")
    with pytest.raises(ValueError, match="requires contact_point_plan"):
        validate_config(missing_plan)

    tampered = copy.deepcopy(config)
    tampered["scaled_contact_mapping"]["derived_target_face_yz_m"]["index"][0] += 1e-5
    with pytest.raises(ValueError, match="mapping provenance"):
        validate_config(tampered)

    bad_evidence = copy.deepcopy(config)
    bad_evidence["scaled_contact_mapping"]["stable_window_end_step"] += 1
    with pytest.raises(ValueError, match="canonical payload"):
        validate_config(bad_evidence)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("edge_m", 0.059, "registered campaign grid"),
        ("mass_kg", 0.161, "mass must match"),
        ("friction", 0.81, "friction must match"),
    ),
)
def test_v13_rejects_cube_values_outside_campaign(
    field: str, value: float, message: str
) -> None:
    config = _template()
    config["cube"][field] = value
    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_authenticated_source_manifest_hashes_and_evidence_are_intact() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    catalog = manifest["source_catalog"]
    assert _sha256(REPO_ROOT / catalog["path"]) == catalog["sha256"]
    assert tuple(source["alias"] for source in manifest["sources"]) == SOURCE_ALIASES

    for source in manifest["sources"]:
        artifacts = source["artifacts"]
        for field in ("resolved_config", "result", "trace"):
            assert _sha256(SOURCE_ROOT / artifacts[field]) == artifacts["sha256"][field]
        evidence = source["reference_contact_evidence"]
        mapping_config = {
            "schema_version": 1,
            "source_alias": source["alias"],
            "mapping_mode": "absolute_face_yz",
            "config_sha256": artifacts["sha256"]["resolved_config"],
            "result_sha256": artifacts["sha256"]["result"],
            "trace_sha256": artifacts["sha256"]["trace"],
            "stable_window_start_step": evidence["stable_window_start_step"],
            "stable_window_end_step": evidence["stable_window_end_step"],
            "reference_edge_m": manifest["reference_edge_m"],
            "reference_target_face_yz_m": evidence["reference_target_face_yz_m"],
            "reference_contact_evidence_sha256": evidence[
                "reference_contact_evidence_sha256"
            ],
            "target_edge_m": 0.088,
            "derived_target_face_yz_m": evidence["reference_target_face_yz_m"],
            "contact_point_plan_id": "0" * 64,
        }
        parsed = ScaledContactMappingParameters.from_config(mapping_config)
        assert parsed.computed_evidence_sha256 == evidence[
            "reference_contact_evidence_sha256"
        ]
        audited = audit_downsize_source(
            REPO_ROOT / catalog["path"], source["alias"]
        )
        assert audited.reference_contact_evidence_sha256(source["alias"]) == (
            evidence["reference_contact_evidence_sha256"]
        )
        assert audited.reference_contact_payload(source["alias"])[
            "reference_target_face_yz_m"
        ] == evidence["reference_target_face_yz_m"]


def test_schema_v12_template_remains_registered_and_loadable() -> None:
    config = load_config(
        REPO_ROOT
        / "grasp_configs/left_opposed_face_palm_down_90mm_contact_point_targeted_actual_grasp_pose.json"
    )
    assert config["experiment_id"] == V12_EXPERIMENT_ID
    assert "scaled_contact_downsize_campaign" not in config


def test_campaign_rejects_wrong_mapping_order() -> None:
    values = dict(SCALED_CONTACT_DOWNSIZE_CAMPAIGN.__dict__)
    values["mapping_modes"] = tuple(reversed(values["mapping_modes"]))
    with pytest.raises(ValueError, match="mapping_modes"):
        ScaledContactDownsizeCampaignParameters(**values)
