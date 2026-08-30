from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from xhand_grasp.config import ACTIVE_ACTUATORS
from xhand_grasp.tuning.contact_point_targeted_search import (
    PointTargetTrialEvaluation,
    PointTargetVariables,
)
from xhand_grasp.tuning.joint_pair_near_zero_campaign import (
    StaticPerturbation,
    grasp_control_variants,
)
from xhand_grasp.tuning.joint_pair_near_zero_pose_search import (
    JointPairStaticObservation,
    PairNullspaceRefinementSettings,
    build_v15_local_contact_policy,
    materialize_grasp_control_variant,
    materialize_static_perturbation,
    refine_pair_alignment_in_contact_nullspace,
    static_perturbation_variables,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "grasp_configs" / (
    "left_opposed_face_palm_down_joint_pair_near_zero_"
    "contact_preserving_planned_lift.json"
)


def _config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_v15_materializes_all_fourteen_variables_without_changing_cube() -> None:
    config = _config()
    perturbation = StaticPerturbation(
        7,
        (0.001, -0.002, 0.003, -0.004, 0.005, -0.006, 0.007, -0.008),
        (0.0002, -0.0003, 0.0004),
        (0.1, -0.2, 0.3),
    )
    base = PointTargetVariables.from_config(config)
    variables = static_perturbation_variables(config, perturbation)

    assert np.allclose(
        np.asarray(variables.actual_joint_qpos_rad),
        np.asarray(base.actual_joint_qpos_rad)
        + np.asarray(perturbation.joint_qpos_offset_rad),
    )
    assert np.allclose(variables.root_delta_cube_m, perturbation.root_delta_cube_m)
    assert np.allclose(
        variables.wrist_local_rotvec_rad,
        np.radians(perturbation.wrist_local_rotvec_deg),
    )

    resolved = materialize_static_perturbation(config, perturbation)
    assert resolved["cube"] == config["cube"]
    assert resolved["hand_pose"] != config["hand_pose"]
    nominal = resolved["grasp_pose"]["nominal_joint_qpos_rad"]
    preload = resolved["control"]["contact_preload_targets_rad"]
    assert [nominal[name] for name in ACTIVE_ACTUATORS] == [
        preload[name] for name in ACTIVE_ACTUATORS
    ]
    metadata = resolved["candidate_metadata"]["contact_point_target_search"]
    assert np.allclose(metadata["root_delta_cube_m"], perturbation.root_delta_cube_m)
    assert np.allclose(
        metadata["wrist_local_rotvec_deg"], perturbation.wrist_local_rotvec_deg
    )
    assert metadata["cube_pose_sampled"] is False


def test_v15_six_control_variants_do_not_change_pose_or_preload() -> None:
    config = _config()
    pose = materialize_static_perturbation(
        config,
        StaticPerturbation(0, (0.0,) * 8, (0.0,) * 3, (0.0,) * 3),
    )
    cube = copy.deepcopy(pose["cube"])
    hand = copy.deepcopy(pose["hand_pose"])
    nominal = copy.deepcopy(pose["grasp_pose"]["nominal_joint_qpos_rad"])

    resolved = [
        materialize_grasp_control_variant(
            pose,
            value,
            reference_config=config,
        )
        for value in grasp_control_variants()
    ]

    assert len(resolved) == 6
    assert [value["control_protocol"]["close_s"] for value in resolved] == [
        1.25,
        1.25,
        1.5,
        1.5,
        1.75,
        1.75,
    ]
    assert all(value["cube"] == cube for value in resolved)
    assert all(value["hand_pose"] == hand for value in resolved)
    assert all(value["grasp_pose"]["nominal_joint_qpos_rad"] == nominal for value in resolved)
    candidate_nominal = pose["grasp_pose"]["nominal_joint_qpos_rad"]
    reference_nominal = config["grasp_pose"]["nominal_joint_qpos_rad"]
    reference_preload = config["control"]["contact_preload_targets_rad"]
    expected_original = {
        name: candidate_nominal[name]
        + reference_preload[name]
        - reference_nominal[name]
        for name in ACTIVE_ACTUATORS
    }
    for index, value in enumerate(resolved):
        observed = value["control"]["contact_preload_targets_rad"]
        if index % 2 == 0:
            assert observed == expected_original
        else:
            assert observed == nominal
    synchronized = resolved[1]["control"]["close_profile"]
    assert set(synchronized) == set(ACTIVE_ACTUATORS)
    assert all(value == {"start_fraction": 0.0, "end_fraction": 1.0} for value in synchronized.values())
    assert resolved[0]["control"]["close_profile"] == pose["control"]["close_profile"]


def test_pair_nullspace_refinement_reduces_angle_and_preserves_contact_gate() -> None:
    config = _config()
    start = PointTargetVariables.from_config(config)
    reference = np.asarray(start.actual_joint_qpos_rad, dtype=np.float64)
    policy = build_v15_local_contact_policy(config)
    joint_bounds = {name: (-2.0, 2.0) for name in ACTIVE_ACTUATORS}
    precontact = reference - 0.002
    static_result = SimpleNamespace(
        static_geometry_pass=True,
        precontact_joint_qpos_rad=tuple(float(value) for value in precontact),
    )
    target_measurement = (
        0.00015,
        0.00015,
        0.00015,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
    )

    def contact(_: object) -> PointTargetTrialEvaluation:
        return PointTargetTrialEvaluation(
            static_result,
            target_measurement,
            ((0.0, 0.0),) * 3,
            (0.0, 0.0, 0.0),
            (),
        )

    def pair(candidate: object) -> JointPairStaticObservation:
        variables = PointTargetVariables.from_config(candidate)
        qpos = np.asarray(variables.actual_joint_qpos_rad)
        # Two independent, differentiable directions give the solver a real
        # rank-two pair objective while contact remains in its null space.
        residual = np.asarray(
            (
                0.004 + qpos[4] - reference[4],
                -0.003 + qpos[6] - reference[6],
            )
        )
        vector = np.asarray((0.020 * residual[0], 0.020, 0.020 * residual[1]))
        angle = math.degrees(math.atan(float(np.linalg.norm(residual))))
        return JointPairStaticObservation(
            tuple(vector),
            tuple(residual),
            float(np.linalg.norm(vector)),
            angle,
            True,
            False,
        )

    before = pair(config).angle_deg
    result = refine_pair_alignment_in_contact_nullspace(
        config,
        start,
        contact_evaluator=contact,
        pair_evaluator=pair,
        policy=policy,
        joint_bounds=joint_bounds,
        settings=PairNullspaceRefinementSettings(
            maximum_iterations=3,
            target_angle_deg=0.01,
        ),
    )

    assert result.improved
    assert result.pair_observation.angle_deg < before * 0.01
    assert result.pair_observation.angle_deg <= 0.01
    assert result.contact_evaluation.safety_violations == ()
    assert result.config["cube"] == config["cube"]
    assert result.config["candidate_metadata"][
        "precontact_derived_from_contact_point_jacobian"
    ] is True


def test_pair_observation_rejects_direction_mismatch() -> None:
    try:
        JointPairStaticObservation(
            (0.0, -0.02, 0.0),
            (0.0, 0.0),
            0.02,
            0.0,
            True,
            False,
        )
    except ValueError as error:
        assert "positive_y" in str(error)
    else:  # pragma: no cover - explicit fail-closed assertion
        raise AssertionError("direction mismatch was accepted")
