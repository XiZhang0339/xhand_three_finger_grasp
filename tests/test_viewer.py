from __future__ import annotations

import hashlib
import json
import math
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

import xhand_grasp.cli as cli
import xhand_grasp.viewer as viewer_module
from xhand_grasp.config import ACTIVE_ACTUATORS, load_config
from xhand_grasp.contact_environment import ContactEnvironmentSpec
from xhand_grasp.relative_wrist_pose import transform_relative_wrist_pose
from xhand_grasp.scene import build_model
from xhand_grasp.tuning.joint_pair_near_zero_campaign_runner import (
    publish_v15_viewer_catalog,
)
from xhand_grasp.viewer import (
    LiveViewerResult,
    ReplaySource,
    ViewerSource,
    _require_interactive_gl,
    _append_joint_axis_marker,
    _append_joint_pair_markers,
    _append_coordinate_frame_markers,
    _write_live_output,
    actual_grasp_pose_telemetry,
    apply_replay_frame,
    apply_viewer_overrides,
    compare_reference_trace,
    copy_physics_to_display,
    format_actual_grasp_pose_telemetry,
    format_joint_monitor_telemetry,
    format_joint_pair_telemetry,
    joint_monitor_telemetry,
    joint_pair_telemetry,
    load_replay_trace,
    parse_actuator_overrides,
    resolve_joint_monitor,
    resolve_joint_pair,
    resolve_replay_source,
    resolve_viewer_source,
    simulate_in_viewer,
    validate_trace_model_binding,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_larger_cube_relative_pose_rescue_validated.json"
)
V4_CONFIG = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_tilted_down_aligned_contacts_grasp_then_lift.json"
)
V5_CONFIG = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_tilted_down_far_hand_fingertip_grasp_then_lift.json"
)
V9_CONFIG = ROOT / "grasp_configs" / (
    "left_opposed_face_palm_down_actual_contact_grasp_pose_"
    "smooth_vertical_lift.json"
)
V11_CONFIG = ROOT / "grasp_configs" / (
    "left_opposed_face_palm_down_larger_relative_wrist_pose_actual_contact_"
    "smooth_vertical_lift.json"
)


def _write_trace(path: Path, model: mujoco.MjModel, frames: int = 3) -> dict:
    actuator_order = np.asarray(
        [model.actuator(index).name for index in range(model.nu)]
    )
    payload = {
        "time": np.linspace(0.001, 0.003, frames),
        "cube_pos": np.tile([0.071, -0.027, 0.115], (frames, 1)),
        "cube_quat": np.tile([1.0, 0.0, 0.0, 0.0], (frames, 1)),
        "cube_velocity": np.zeros((frames, 6)),
        "ctrl": np.zeros((frames, model.nu)),
        "joint_qpos": np.zeros((frames, model.nu)),
        "joint_qvel": np.zeros((frames, model.nu)),
        "actuator_order": actuator_order,
    }
    payload["cube_pos"][-1] = [0.063, -0.035, 0.127]
    payload["joint_qpos"][-1] = np.linspace(0.01, 0.12, model.nu)
    np.savez(path, **payload)
    return payload


def test_catalog_label_resolves_config_and_trace_relative_to_catalog(tmp_path):
    directory = tmp_path / "grid_037"
    directory.mkdir()
    config = directory / "resolved_config.json"
    trace = directory / "trace.npz"
    config.write_text("{}\n", encoding="utf-8")
    trace.write_bytes(b"NPZ")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "trajectories": [
                    {
                        "label": "nominal",
                        "trajectory_id": "grid_037",
                        "grid_index": 37,
                        "candidate_id": 1616000032129682,
                        "artifacts": {
                            "resolved_config": "grid_037/resolved_config.json",
                            "trace": "grid_037/trace.npz",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    by_label = resolve_replay_source(catalog_path=catalog, trajectory="nominal")
    by_id = resolve_replay_source(catalog_path=catalog, trajectory="grid_037")
    by_index = resolve_replay_source(catalog_path=catalog, trajectory="37")
    by_candidate = resolve_replay_source(
        catalog_path=catalog, trajectory="1616000032129682"
    )
    by_prefixed_candidate = resolve_replay_source(
        catalog_path=catalog, trajectory="candidate_1616000032129682"
    )

    assert by_label == by_id == by_index == by_candidate == by_prefixed_candidate
    assert by_label.config_path == config.resolve()
    assert by_label.trace_path == trace.resolve()


def test_catalog_entry_alias_resolves_to_the_same_trajectory(tmp_path):
    directory = tmp_path / "candidate_000007"
    directory.mkdir()
    config = directory / "resolved_config.json"
    trace = directory / "trace.npz"
    config.write_text("{}\n", encoding="utf-8")
    trace.write_bytes(b"NPZ")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "trajectories": [
                    {
                        "label": "tilt_17p5",
                        "trajectory_id": "candidate_000007",
                        "aliases": ["best_nominal"],
                        "artifacts": {
                            "resolved_config": (
                                "candidate_000007/resolved_config.json"
                            ),
                            "trace": "candidate_000007/trace.npz",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    source = resolve_viewer_source(
        catalog_path=catalog,
        trajectory="best_nominal",
    )

    assert source.config_path == config.resolve()
    assert source.trace_path == trace.resolve()
    assert source.trajectory == "tilt_17p5"


def test_catalog_contact_environment_is_authenticated_and_forwarded(tmp_path):
    directory = tmp_path / "candidate"
    directory.mkdir()
    config = directory / "resolved_config.json"
    trace = directory / "trace.npz"
    environment = directory / "environment.json"
    config.write_text("{}\n", encoding="utf-8")
    trace.write_bytes(b"NPZ")
    environment.write_text(
        json.dumps(ContactEnvironmentSpec().as_config()), encoding="utf-8"
    )
    environment_sha = hashlib.sha256(environment.read_bytes()).hexdigest()
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "aliases": {"best_environment": "candidate"},
                "trajectories": [
                    {
                        "label": "candidate",
                        "trajectory_id": "candidate",
                        "aliases": ["best_environment"],
                        "artifacts": {
                            "resolved_config": "candidate/resolved_config.json",
                            "trace": "candidate/trace.npz",
                            "contact_environment": "candidate/environment.json",
                            "sha256": {
                                "contact_environment": environment_sha,
                            },
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    source = resolve_viewer_source(
        catalog_path=catalog, trajectory="best_environment"
    )

    assert source.contact_environment_path == environment.resolve()


def _publish_v15_catalog_for_viewer(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    source = tmp_path / "exact_candidate"
    source.mkdir()
    paths = {
        "config": source / "resolved_config.json",
        "result": source / "result.json",
        "trace": source / "trace.npz",
    }
    paths["config"].write_text(
        '{"schema_version":15,"kind":"resolved_config"}\n',
        encoding="utf-8",
    )
    paths["result"].write_text(
        '{"candidate_id":15200000000000001}\n', encoding="utf-8"
    )
    paths["trace"].write_bytes(b"schema-v15-locked-one-ms-trace")
    catalog = publish_v15_viewer_catalog(
        (
            {
                "candidate_id": 15200000000000001,
                "grasp_success": True,
                "full_success": False,
                "summary": {"passed": False, "metrics": {}},
                "config_path": str(paths["config"]),
                "result_path": str(paths["result"]),
                "trace_path": str(paths["trace"]),
            },
        ),
        tmp_path / "viewer_catalog",
    )
    return catalog, paths


def test_v15_published_config_artifact_key_resolves_and_is_hashed(
    tmp_path: Path,
) -> None:
    catalog, _ = _publish_v15_catalog_for_viewer(tmp_path)
    payload = json.loads(catalog.read_text(encoding="utf-8"))
    entry = payload["trajectories"][0]
    assert "config" in entry["artifacts"]
    assert "resolved_config" not in entry["artifacts"]

    source = resolve_viewer_source(
        catalog_path=catalog, trajectory="best_attempt"
    )
    assert source.config_path == (
        catalog.parent / entry["artifacts"]["config"]
    ).resolve()
    assert source.trace_path == (
        catalog.parent / entry["artifacts"]["trace"]
    ).resolve()


@pytest.mark.parametrize("field", ["config", "trace"])
def test_v15_published_catalog_requires_and_verifies_runtime_member_hashes(
    tmp_path: Path, field: str
) -> None:
    catalog, _ = _publish_v15_catalog_for_viewer(tmp_path)
    payload = json.loads(catalog.read_text(encoding="utf-8"))
    entry = payload["trajectories"][0]
    del entry["artifacts"]["sha256"][field]
    catalog.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=rf"schema-v15 catalog is missing required SHA-256 for {field}",
    ):
        resolve_viewer_source(catalog_path=catalog, trajectory="best_attempt")


@pytest.mark.parametrize("field", ["config", "trace"])
def test_v15_published_catalog_rejects_replaced_runtime_members(
    tmp_path: Path, field: str
) -> None:
    catalog, _ = _publish_v15_catalog_for_viewer(tmp_path)
    payload = json.loads(catalog.read_text(encoding="utf-8"))
    entry = payload["trajectories"][0]
    member = (catalog.parent / entry["artifacts"][field]).resolve()
    member.write_bytes(b"tampered-after-publication")

    with pytest.raises(ValueError, match=rf"SHA-256 mismatch for {field}"):
        resolve_viewer_source(catalog_path=catalog, trajectory="best_attempt")


@pytest.mark.parametrize("replaced_field", ["resolved_config", "trace"])
def test_v5_catalog_alias_binds_to_hashed_source_members(
    tmp_path,
    replaced_field,
):
    directory = tmp_path / "candidate_000007"
    directory.mkdir()
    config = directory / "resolved_config.json"
    trace = directory / "trace.npz"
    config.write_text('{"schema_version": 5}\n', encoding="utf-8")
    trace.write_bytes(b"NPZ")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "experiment_id": (
                    "left_opposed_face_palm_tilted_down_far_hand_fingertip_"
                    "grasp_then_lift"
                ),
                "declared_tilt_bands_deg": [10.0, 12.5, 15.0, 17.5, 20.0],
                "aliases": {"best_attempt": "candidate_000007"},
                "trajectories": [
                    {
                        "label": "near_miss_tilt_17p5",
                        "trajectory_id": "candidate_000007",
                        "aliases": ["best_attempt"],
                        "artifacts": {
                            "resolved_config": (
                                "candidate_000007/resolved_config.json"
                            ),
                            "trace": "candidate_000007/trace.npz",
                            "sha256": {
                                "resolved_config": hashlib.sha256(
                                    config.read_bytes()
                                ).hexdigest(),
                                "trace": hashlib.sha256(
                                    trace.read_bytes()
                                ).hexdigest(),
                            },
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    source = resolve_viewer_source(
        catalog_path=catalog,
        trajectory="best_attempt",
    )
    assert source.config_path == config.resolve()
    assert source.trace_path == trace.resolve()

    replaced = config if replaced_field == "resolved_config" else trace
    replaced.write_bytes(b"replaced")
    with pytest.raises(ValueError, match=f"SHA-256 mismatch for {replaced_field}"):
        resolve_viewer_source(
            catalog_path=catalog,
            trajectory="best_attempt",
        )


def test_catalog_rejects_disagreement_between_top_level_and_entry_aliases(tmp_path):
    entries = []
    for candidate_id in (7, 8):
        directory = tmp_path / f"candidate_{candidate_id:06d}"
        directory.mkdir()
        config = directory / "resolved_config.json"
        trace = directory / "trace.npz"
        config.write_text("{}\n", encoding="utf-8")
        trace.write_bytes(b"NPZ")
        entries.append(
            {
                "trajectory_id": f"candidate_{candidate_id:06d}",
                "aliases": ["best_nominal"] if candidate_id == 7 else [],
                "artifacts": {
                    "resolved_config": (
                        f"candidate_{candidate_id:06d}/resolved_config.json"
                    ),
                    "trace": f"candidate_{candidate_id:06d}/trace.npz",
                },
            }
        )
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "aliases": {"best_nominal": "candidate_000008"},
                "trajectories": entries,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="disagrees with trajectory aliases"):
        resolve_viewer_source(
            catalog_path=catalog,
            trajectory="best_nominal",
        )


def test_catalog_accepts_authoritative_top_level_alias_when_entries_omit_aliases(
    tmp_path,
):
    directory = tmp_path / "pair_rank_01_7"
    directory.mkdir()
    config = directory / "resolved_config.json"
    trace = directory / "trace.npz"
    config.write_text("{}\n", encoding="utf-8")
    trace.write_bytes(b"NPZ")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "contact_preserving_viewer_catalog_schema_version": 1,
                "experiment_id": (
                    "left_opposed_face_palm_down_contact_preserving_planned_lift"
                ),
                "aliases": {"best_attempt": "pair_rank_01_7"},
                "trajectories": [
                    {
                        "trajectory_id": "pair_rank_01_7",
                        "artifacts": {
                            "resolved_config": (
                                "pair_rank_01_7/resolved_config.json"
                            ),
                            "trace": "pair_rank_01_7/trace.npz",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    source = resolve_viewer_source(
        catalog_path=catalog,
        trajectory="best_attempt",
    )
    assert source.config_path == config.resolve()
    assert source.trace_path == trace.resolve()
    assert source.trajectory == "pair_rank_01_7"


def test_catalog_rejects_unversioned_top_level_only_alias(tmp_path):
    directory = tmp_path / "candidate_7"
    directory.mkdir()
    config = directory / "resolved_config.json"
    trace = directory / "trace.npz"
    config.write_text("{}\n", encoding="utf-8")
    trace.write_bytes(b"NPZ")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "aliases": {"best_attempt": "candidate_7"},
                "trajectories": [
                    {
                        "trajectory_id": "candidate_7",
                        "artifacts": {
                            "resolved_config": "candidate_7/resolved_config.json",
                            "trace": "candidate_7/trace.npz",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="disagrees with trajectory aliases"):
        resolve_viewer_source(
            catalog_path=catalog,
            trajectory="best_attempt",
        )


def test_catalog_digest_rejects_a_replaced_trace(tmp_path):
    directory = tmp_path / "grid_001"
    directory.mkdir()
    config = directory / "resolved_config.json"
    trace = directory / "trace.npz"
    config.write_text("{}\n", encoding="utf-8")
    trace.write_bytes(b"replaced")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "trajectories": [
                    {
                        "label": "nominal",
                        "artifacts": {
                            "resolved_config": "grid_001/resolved_config.json",
                            "trace": "grid_001/trace.npz",
                            "sha256": {
                                "resolved_config": hashlib.sha256(
                                    config.read_bytes()
                                ).hexdigest(),
                                "trace": "0" * 64,
                            },
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch for trace"):
        resolve_replay_source(catalog_path=catalog, trajectory="nominal")


def test_schema_v4_catalog_requires_config_and_trace_digests(tmp_path):
    directory = tmp_path / "tilt_15"
    directory.mkdir()
    (directory / "resolved_config.json").write_text("{}\n", encoding="utf-8")
    (directory / "trace.npz").write_bytes(b"NPZ")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "experiment_id": (
                    "left_opposed_face_palm_tilted_down_aligned_contacts_"
                    "grasp_then_lift"
                ),
                "trajectories": [
                    {
                        "label": "tilt_15",
                        "artifacts": {
                            "resolved_config": "tilt_15/resolved_config.json",
                            "trace": "tilt_15/trace.npz",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required SHA-256"):
        resolve_replay_source(catalog_path=catalog, trajectory="tilt_15")


def test_replay_source_requires_one_complete_source_pair(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        resolve_replay_source()
    with pytest.raises(ValueError, match="both --config and --trace"):
        resolve_replay_source(config_path=tmp_path / "config.json")
    with pytest.raises(ValueError, match="exactly one"):
        resolve_replay_source(
            catalog_path=tmp_path / "catalog.json",
            config_path=tmp_path / "config.json",
            trace_path=tmp_path / "trace.npz",
        )


def test_apply_frame_restores_recorded_cube_and_actuator_state(tmp_path):
    model, info = build_model(load_config(CONFIG))
    trace_path = tmp_path / "trace.npz"
    expected = _write_trace(trace_path, model)
    trace = load_replay_trace(trace_path)
    validate_trace_model_binding(model, info, trace)
    data = mujoco.MjData(model)

    apply_replay_frame(model, data, info, trace, 2)

    np.testing.assert_allclose(
        data.qpos[info.cube_qpos_adr : info.cube_qpos_adr + 3],
        expected["cube_pos"][2],
    )
    np.testing.assert_allclose(
        data.qpos[info.actuator_qpos_adrs], expected["joint_qpos"][2]
    )
    assert data.time == pytest.approx(expected["time"][2])


def test_trace_rejects_non_monotonic_time_and_actuator_order(tmp_path):
    model, info = build_model(load_config(CONFIG))
    trace_path = tmp_path / "trace.npz"
    payload = _write_trace(trace_path, model)
    payload["time"] = np.asarray([0.001, 0.003, 0.002])
    np.savez(trace_path, **payload)
    with pytest.raises(ValueError, match="strictly increasing"):
        load_replay_trace(trace_path)

    payload["time"] = np.asarray([0.001, 0.002, 0.003])
    payload["actuator_order"] = payload["actuator_order"][::-1]
    np.savez(trace_path, **payload)
    trace = load_replay_trace(trace_path)
    with pytest.raises(ValueError, match="actuator_order"):
        validate_trace_model_binding(model, info, trace)


def test_project_osmesa_default_is_rejected_with_copyable_hint(monkeypatch):
    monkeypatch.setenv("MUJOCO_GL", "osmesa")
    with pytest.raises(RuntimeError, match="MUJOCO_GL=glfw"):
        _require_interactive_gl()


def test_state_replay_cli_forwards_catalog_selection_without_opening_gui(monkeypatch):
    expected = ReplaySource(Path("config.json"), Path("trace.npz"), "nominal")
    seen = {}
    monkeypatch.setattr(cli, "resolve_replay_source", lambda **kwargs: expected)
    monkeypatch.setattr(
        cli,
        "replay_in_viewer",
        lambda source, **kwargs: seen.update(source=source, **kwargs),
    )
    args = cli.build_parser().parse_args(
        [
            "view",
            "--catalog",
            "catalog.json",
            "--trajectory",
            "nominal",
            "--state-replay",
            "--speed",
            "1.5",
            "--loop",
            "--start-paused",
            "--show-coordinate-frames",
            "--show-joint-pair",
            "left_hand_index_joint1",
            "left_hand_mid_joint1",
        ]
    )

    assert args.func(args) == 0
    assert seen == {
        "source": expected,
        "speed": 1.5,
        "loop": True,
        "start_paused": True,
        "joint_pair": (
            "left_hand_index_joint1",
            "left_hand_mid_joint1",
        ),
        "show_coordinate_frames": True,
    }


def test_state_replay_rejects_live_grasp_lock_pause(monkeypatch):
    monkeypatch.setattr(
        cli,
        "resolve_replay_source",
        lambda **kwargs: pytest.fail("source resolution must follow option validation"),
    )
    args = cli.build_parser().parse_args(
        [
            "view",
            "--config",
            "config.json",
            "--trace",
            "trace.npz",
            "--state-replay",
            "--pause-at-event",
            "grasp_lock",
        ]
    )

    with pytest.raises(ValueError, match="only for live physics"):
        args.func(args)


def test_live_source_accepts_direct_config_without_trace():
    source = resolve_viewer_source(config_path=CONFIG)

    assert source == ViewerSource(CONFIG.resolve(), None, CONFIG.stem, False)


def test_view_cli_defaults_to_live_physics_and_forwards_overrides(monkeypatch):
    source = ViewerSource(Path("config.json"), None, "candidate", False)
    base_config = load_config(CONFIG)
    resolved_config = deepcopy(base_config)
    seen = {}
    monkeypatch.setattr(cli, "resolve_viewer_source", lambda **kwargs: source)
    monkeypatch.setattr(cli, "load_config", lambda path: base_config)
    monkeypatch.setattr(
        cli,
        "apply_viewer_overrides",
        lambda config, **kwargs: (resolved_config, True),
    )
    monkeypatch.setattr(
        cli,
        "simulate_in_viewer",
        lambda selected, config, **kwargs: (
            seen.update(source=selected, config=config, **kwargs)
            or LiveViewerResult(2, True, {"passed": False}, None)
        ),
    )
    args = cli.build_parser().parse_args(
        [
            "view",
            "--config",
            "config.json",
            "--edge-mm",
            "61",
            "--density-scale",
            "1.1",
            "--finger-down-deg",
            "15",
            "--press-mm",
            "5",
            "--output-dir",
            "viewer-output",
        ]
    )

    assert args.func(args) == 2
    assert seen == {
        "source": source,
        "config": resolved_config,
        "speed": 1.0,
        "loop": False,
        "start_paused": False,
        "output_dir": "viewer-output",
        "parameter_overridden": True,
    }


def test_view_cli_loads_contact_environment_as_a_parameter_override(
    tmp_path, monkeypatch
):
    source = ViewerSource(Path("config.json"), None, "candidate", False)
    base_config = load_config(CONFIG)
    environment = ContactEnvironmentSpec(iterations=200)
    environment_path = tmp_path / "environment.json"
    environment_path.write_text(
        json.dumps(environment.as_config()), encoding="utf-8"
    )
    seen = {}
    monkeypatch.setattr(cli, "resolve_viewer_source", lambda **kwargs: source)
    monkeypatch.setattr(cli, "load_config", lambda path: base_config)
    monkeypatch.setattr(
        cli,
        "apply_viewer_overrides",
        lambda config, **kwargs: (config, False),
    )
    monkeypatch.setattr(
        cli,
        "simulate_in_viewer",
        lambda selected, config, **kwargs: (
            seen.update(kwargs)
            or LiveViewerResult(2, True, {"passed": False}, None)
        ),
    )

    args = cli.build_parser().parse_args(
        [
            "view",
            "--config",
            "config.json",
            "--contact-environment",
            str(environment_path),
        ]
    )

    assert args.func(args) == 2
    assert seen["parameter_overridden"] is True
    assert seen["contact_environment"] == environment


def test_view_cli_forwards_distance_named_targets_and_joint_monitor(monkeypatch):
    source = ViewerSource(Path("config.json"), None, "candidate", False)
    base_config = load_config(CONFIG)
    apply_seen = {}
    simulate_seen = {}
    monkeypatch.setattr(cli, "resolve_viewer_source", lambda **kwargs: source)
    monkeypatch.setattr(cli, "load_config", lambda path: base_config)

    def fake_apply(config, **kwargs):
        apply_seen.update(kwargs)
        return config, True

    monkeypatch.setattr(cli, "apply_viewer_overrides", fake_apply)
    monkeypatch.setattr(
        cli,
        "simulate_in_viewer",
        lambda selected, config, **kwargs: (
            simulate_seen.update(kwargs)
            or LiveViewerResult(2, True, {"passed": False}, None)
        ),
    )
    args = cli.build_parser().parse_args(
        [
            "view",
            "--config",
            "config.json",
            "--root-cube-distance-mm",
            "150",
            "--grasp-target-rad",
            "left_hand_thumb_bend_joint_actuator=1.2",
            "--manipulation-delta-rad",
            "left_hand_thumb_bend_joint_actuator=-0.1",
            "--joint-monitor",
            "left_hand_thumb_bend_joint_actuator",
            "--pause-at-event",
            "grasp_lock",
            "--show-coordinate-frames",
            "--show-joint-pair",
            "left_hand_index_joint1",
            "left_hand_mid_joint1",
        ]
    )

    assert args.func(args) == 2
    assert apply_seen["root_cube_distance_mm"] == 150.0
    assert apply_seen["grasp_target_rad"] == {
        "left_hand_thumb_bend_joint_actuator": 1.2
    }
    assert apply_seen["manipulation_delta_rad"] == {
        "left_hand_thumb_bend_joint_actuator": -0.1
    }
    assert simulate_seen["joint_monitor"] == (
        "left_hand_thumb_bend_joint_actuator"
    )
    assert simulate_seen["pause_at_event"] == "grasp_lock"
    assert simulate_seen["show_coordinate_frames"] is True
    assert simulate_seen["joint_pair"] == (
        "left_hand_index_joint1",
        "left_hand_mid_joint1",
    )


def test_viewer_overrides_preserve_density_and_solve_requested_press():
    config = load_config(CONFIG)
    config["candidate_metadata"] = {
        "palm_press_depth_m": 0.005,
        "cube_in_root_m": [0.1, -0.03, 0.11],
    }
    source_root_z = float(config["hand_pose"]["translation_m"][2])
    source_density = float(config["cube"]["mass_kg"]) / float(
        config["cube"]["edge_m"]
    ) ** 3

    overridden, changed = apply_viewer_overrides(
        config,
        edge_mm=60.0,
        density_scale=1.1,
        friction=0.9,
        finger_down_deg=15.0,
        press_mm=5.0,
    )

    assert changed
    assert overridden["cube"]["edge_m"] == pytest.approx(0.060)
    assert overridden["cube"]["mass_kg"] == pytest.approx(
        source_density * 0.060**3 * 1.1
    )
    assert overridden["cube"]["friction"] == pytest.approx(0.9)
    assert overridden["hand_pose"]["translation_m"][2] == pytest.approx(
        source_root_z - 0.005, abs=1e-12
    )
    assert "experiment_status" not in overridden
    assert "candidate_metadata" not in overridden
    assert overridden["run_context"] == {"kind": "parameter_override_run"}


def test_viewer_distance_and_named_control_overrides_are_exact():
    config = load_config(CONFIG)
    cube_world = viewer_module._configured_cube_center(config)
    source_rotation = viewer_module._rpy_matrix(config["hand_pose"]["rpy_deg"])
    source_root = np.asarray(config["hand_pose"]["translation_m"])
    source_relative = source_rotation.T @ (cube_world - source_root)

    overridden, changed = apply_viewer_overrides(
        config,
        root_cube_distance_mm=150.0,
        grasp_target_rad={"left_hand_thumb_bend_joint_actuator": 1.2},
        manipulation_delta_rad={"left_hand_thumb_bend_joint_actuator": -0.1},
    )

    assert changed
    result_rotation = viewer_module._rpy_matrix(
        overridden["hand_pose"]["rpy_deg"]
    )
    result_root = np.asarray(overridden["hand_pose"]["translation_m"])
    result_relative = result_rotation.T @ (cube_world - result_root)
    assert np.linalg.norm(result_relative) == pytest.approx(0.150, abs=1e-12)
    np.testing.assert_allclose(
        result_relative / np.linalg.norm(result_relative),
        source_relative / np.linalg.norm(source_relative),
        atol=1e-12,
    )
    assert overridden["control"]["grasp_targets_rad"][
        "left_hand_thumb_bend_joint_actuator"
    ] == pytest.approx(1.2)
    assert overridden["control"]["manipulation_delta_rad"][
        "left_hand_thumb_bend_joint_actuator"
    ] == pytest.approx(-0.1)
    assert overridden["run_context"] == {"kind": "parameter_override_run"}


def test_viewer_coupled_relative_wrist_override_matches_rigid_transform():
    config = load_config(V9_CONFIG)
    source_cube_position = viewer_module._configured_cube_center(config)
    source_cube_rotation = viewer_module._rpy_matrix(config["cube"]["rpy_deg"])
    source_root_position = np.asarray(config["hand_pose"]["translation_m"])
    source_root_rotation = viewer_module._rpy_matrix(
        config["hand_pose"]["rpy_deg"]
    )
    expected = transform_relative_wrist_pose(
        source_cube_world_position_m=source_cube_position,
        source_cube_world_rotation=source_cube_rotation,
        source_root_world_position_m=source_root_position,
        source_root_world_rotation=source_root_rotation,
        clockwise_orbit_deg=7.5,
        root_delta_cube_m=[0.001, -0.002, 0.003],
        wrist_local_rotvec_deg=[1.0, -2.0, 0.5],
    )

    overridden, changed = apply_viewer_overrides(
        config,
        clockwise_orbit_deg=7.5,
        root_delta_cube_mm=[1.0, -2.0, 3.0],
        wrist_local_rotvec_deg=[1.0, -2.0, 0.5],
    )

    assert changed
    np.testing.assert_allclose(
        overridden["hand_pose"]["translation_m"],
        expected.root_world_position_m,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        viewer_module._rpy_matrix(overridden["hand_pose"]["rpy_deg"]),
        expected.root_world_rotation,
        atol=2e-9,
    )
    diagnostics = overridden["relative_wrist_pose_diagnostics"]
    assert diagnostics["clockwise_orbit_deg"] == pytest.approx(7.5)
    assert diagnostics["root_delta_cube_m"] == pytest.approx(
        [0.001, -0.002, 0.003]
    )
    assert diagnostics["wrist_local_rotvec_deg"] == pytest.approx(
        [1.0, -2.0, 0.5]
    )
    assert overridden["run_context"] == {"kind": "parameter_override_run"}


def test_v11_relative_wrist_overrides_replace_stored_values_from_anchor_once(
    monkeypatch,
):
    config = load_config(V11_CONFIG)
    cube_position = viewer_module._configured_cube_center(config)
    cube_rotation = viewer_module._rpy_matrix(config["cube"]["rpy_deg"])
    anchor = deepcopy(config["hand_pose"])
    stored_orbit = 7.5
    stored_delta = np.asarray([0.001, -0.002, 0.003])
    stored_rotvec = np.asarray([1.0, -2.0, 0.5])
    stored = transform_relative_wrist_pose(
        source_cube_world_position_m=cube_position,
        source_cube_world_rotation=cube_rotation,
        source_root_world_position_m=anchor["translation_m"],
        source_root_world_rotation=viewer_module._rpy_matrix(anchor["rpy_deg"]),
        clockwise_orbit_deg=stored_orbit,
        root_delta_cube_m=stored_delta,
        wrist_local_rotvec_deg=stored_rotvec,
    )
    config["hand_pose"] = {
        "translation_m": list(stored.root_world_position_m),
        "rpy_deg": viewer_module.rotation_matrix_to_rpy_degrees(
            stored.root_world_rotation,
            reference_rpy_deg=anchor["rpy_deg"],
        ).tolist(),
    }
    config["experiment_status"] = {"classification": "full_success"}
    config["candidate_metadata"] = {
        "candidate_id": 4864000014401350,
        "measured_grasp_pose_success": True,
        "relative_wrist_pose_search": {
            "anchor_hand_pose": anchor,
            "clockwise_orbit_deg": stored_orbit,
            "root_delta_cube_m": stored_delta.tolist(),
            "wrist_local_rotvec_deg": stored_rotvec.tolist(),
        },
    }
    # This unit isolates the absolute coordinate algebra from registered pose
    # limits and model compilation; those remain covered by live Viewer tests.
    monkeypatch.setattr(viewer_module, "validate_config", lambda value: None)
    monkeypatch.setattr(viewer_module, "preflight_config", lambda value: None)

    orbit_replaced, changed = apply_viewer_overrides(
        config, clockwise_orbit_deg=12.5
    )
    assert changed
    expected_orbit = transform_relative_wrist_pose(
        source_cube_world_position_m=cube_position,
        source_cube_world_rotation=cube_rotation,
        source_root_world_position_m=anchor["translation_m"],
        source_root_world_rotation=viewer_module._rpy_matrix(anchor["rpy_deg"]),
        clockwise_orbit_deg=12.5,
        root_delta_cube_m=stored_delta,
        wrist_local_rotvec_deg=stored_rotvec,
    )
    np.testing.assert_allclose(
        orbit_replaced["hand_pose"]["translation_m"],
        expected_orbit.root_world_position_m,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        viewer_module._rpy_matrix(orbit_replaced["hand_pose"]["rpy_deg"]),
        expected_orbit.root_world_rotation,
        atol=2e-9,
    )

    # Applying another Viewer override to the already-overridden config must
    # still restart from the same anchor instead of accumulating a second
    # orbit.  Omitted orbit/rotvec values inherit the stored effective values.
    delta_replaced, _ = apply_viewer_overrides(
        orbit_replaced, root_delta_cube_mm=[4.0, 5.0, 6.0]
    )
    expected_delta = transform_relative_wrist_pose(
        source_cube_world_position_m=cube_position,
        source_cube_world_rotation=cube_rotation,
        source_root_world_position_m=anchor["translation_m"],
        source_root_world_rotation=viewer_module._rpy_matrix(anchor["rpy_deg"]),
        clockwise_orbit_deg=12.5,
        root_delta_cube_m=[0.004, 0.005, 0.006],
        wrist_local_rotvec_deg=stored_rotvec,
    )
    np.testing.assert_allclose(
        delta_replaced["hand_pose"]["translation_m"],
        expected_delta.root_world_position_m,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        viewer_module._rpy_matrix(delta_replaced["hand_pose"]["rpy_deg"]),
        expected_delta.root_world_rotation,
        atol=2e-9,
    )

    rotvec_replaced, _ = apply_viewer_overrides(
        delta_replaced, wrist_local_rotvec_deg=[-3.0, 2.0, 1.0]
    )
    expected_rotvec = transform_relative_wrist_pose(
        source_cube_world_position_m=cube_position,
        source_cube_world_rotation=cube_rotation,
        source_root_world_position_m=anchor["translation_m"],
        source_root_world_rotation=viewer_module._rpy_matrix(anchor["rpy_deg"]),
        clockwise_orbit_deg=12.5,
        root_delta_cube_m=[0.004, 0.005, 0.006],
        wrist_local_rotvec_deg=[-3.0, 2.0, 1.0],
    )
    np.testing.assert_allclose(
        viewer_module._rpy_matrix(rotvec_replaced["hand_pose"]["rpy_deg"]),
        expected_rotvec.root_world_rotation,
        atol=2e-9,
    )

    metadata = rotvec_replaced["candidate_metadata"]
    assert metadata["campaign_kind"] == "viewer_parameter_override_diagnostic"
    assert metadata["source_success_evidence_inherited"] is False
    assert "candidate_id" not in metadata
    assert "measured_grasp_pose_success" not in metadata
    relative = metadata["relative_wrist_pose_search"]
    assert relative["diagnostic_only"] is True
    assert relative["source_success_evidence_inherited"] is False
    np.testing.assert_allclose(
        relative["anchor_hand_pose"]["translation_m"],
        anchor["translation_m"],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        viewer_module._rpy_matrix(relative["anchor_hand_pose"]["rpy_deg"]),
        viewer_module._rpy_matrix(anchor["rpy_deg"]),
        atol=2e-9,
    )
    assert relative["clockwise_orbit_deg"] == pytest.approx(12.5)
    assert relative["root_delta_cube_m"] == pytest.approx([0.004, 0.005, 0.006])
    assert relative["wrist_local_rotvec_deg"] == pytest.approx([-3.0, 2.0, 1.0])
    assert "experiment_status" not in rotvec_replaced
    assert rotvec_replaced["run_context"] == {"kind": "parameter_override_run"}


def test_v11_relative_wrist_anchor_moves_with_cube_override_for_later_runs(
    monkeypatch,
):
    config = load_config(V11_CONFIG)
    source_cube_position = viewer_module._configured_cube_center(config)
    source_cube_rotation = viewer_module._rpy_matrix(config["cube"]["rpy_deg"])
    anchor = deepcopy(config["hand_pose"])
    stored_orbit = 5.0
    stored_delta = [0.001, -0.002, 0.003]
    stored_rotvec = [1.0, -1.5, 0.5]
    resolved = transform_relative_wrist_pose(
        source_cube_world_position_m=source_cube_position,
        source_cube_world_rotation=source_cube_rotation,
        source_root_world_position_m=anchor["translation_m"],
        source_root_world_rotation=viewer_module._rpy_matrix(anchor["rpy_deg"]),
        clockwise_orbit_deg=stored_orbit,
        root_delta_cube_m=stored_delta,
        wrist_local_rotvec_deg=stored_rotvec,
    )
    config["hand_pose"] = {
        "translation_m": list(resolved.root_world_position_m),
        "rpy_deg": viewer_module.rotation_matrix_to_rpy_degrees(
            resolved.root_world_rotation,
            reference_rpy_deg=anchor["rpy_deg"],
        ).tolist(),
    }
    config["candidate_metadata"] = {
        "candidate_id": 123,
        "relative_wrist_pose_search": {
            "anchor_hand_pose": anchor,
            "clockwise_orbit_deg": stored_orbit,
            "root_delta_cube_m": stored_delta,
            "wrist_local_rotvec_deg": stored_rotvec,
        },
    }
    monkeypatch.setattr(viewer_module, "validate_config", lambda value: None)
    monkeypatch.setattr(viewer_module, "preflight_config", lambda value: None)

    moved_cube, _ = apply_viewer_overrides(
        config,
        edge_mm=90.0,
        cube_rpy_deg=[2.0, -3.0, 35.0],
        clockwise_orbit_deg=10.0,
    )
    target_cube_position = viewer_module._configured_cube_center(moved_cube)
    target_cube_rotation = viewer_module._rpy_matrix(moved_cube["cube"]["rpy_deg"])
    expected_anchor = transform_relative_wrist_pose(
        source_cube_world_position_m=source_cube_position,
        source_cube_world_rotation=source_cube_rotation,
        source_root_world_position_m=anchor["translation_m"],
        source_root_world_rotation=viewer_module._rpy_matrix(anchor["rpy_deg"]),
        target_cube_world_position_m=target_cube_position,
        target_cube_world_rotation=target_cube_rotation,
    )
    stored_target_anchor = moved_cube["candidate_metadata"][
        "relative_wrist_pose_search"
    ]["anchor_hand_pose"]
    np.testing.assert_allclose(
        stored_target_anchor["translation_m"],
        expected_anchor.root_world_position_m,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        viewer_module._rpy_matrix(stored_target_anchor["rpy_deg"]),
        expected_anchor.root_world_rotation,
        atol=2e-9,
    )

    # A subsequent absolute replacement must use that transferred anchor and
    # therefore equal a single original-source -> target transform.
    second, _ = apply_viewer_overrides(
        moved_cube, root_delta_cube_mm=[4.0, 5.0, 6.0]
    )
    expected_second = transform_relative_wrist_pose(
        source_cube_world_position_m=source_cube_position,
        source_cube_world_rotation=source_cube_rotation,
        source_root_world_position_m=anchor["translation_m"],
        source_root_world_rotation=viewer_module._rpy_matrix(anchor["rpy_deg"]),
        target_cube_world_position_m=target_cube_position,
        target_cube_world_rotation=target_cube_rotation,
        clockwise_orbit_deg=10.0,
        root_delta_cube_m=[0.004, 0.005, 0.006],
        wrist_local_rotvec_deg=stored_rotvec,
    )
    np.testing.assert_allclose(
        second["hand_pose"]["translation_m"],
        expected_second.root_world_position_m,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        viewer_module._rpy_matrix(second["hand_pose"]["rpy_deg"]),
        expected_second.root_world_rotation,
        atol=2e-9,
    )


def test_relative_wrist_cli_help_describes_absolute_replacement_semantics(capsys):
    with pytest.raises(SystemExit) as caught:
        cli.build_parser().parse_args(["view", "--help"])
    assert caught.value.code == 0
    help_text = capsys.readouterr().out
    assert "replace the stored clockwise orbit" in help_text
    assert "full transform is rebuilt" in help_text
    assert "once from" in help_text
    assert "anchor_hand_pose" in help_text
    assert "replace the stored cube-frame root residual" in help_text
    assert "replace the stored hand-local rotation vector" in help_text


def test_coupled_relative_wrist_override_rejects_legacy_pose_controls():
    config = load_config(V9_CONFIG)

    with pytest.raises(ValueError, match="cannot be combined with legacy"):
        apply_viewer_overrides(
            config,
            clockwise_orbit_deg=5.0,
            root_cube_distance_mm=150.0,
        )
    with pytest.raises(ValueError, match="three finite values"):
        apply_viewer_overrides(config, root_delta_cube_mm=[1.0, 2.0])
    with pytest.raises(ValueError, match="finite"):
        apply_viewer_overrides(
            config, wrist_local_rotvec_deg=[0.0, float("nan"), 0.0]
        )


def test_view_cli_forwards_coupled_relative_wrist_overrides(monkeypatch):
    source = ViewerSource(Path("config.json"), None, "candidate", False)
    base_config = load_config(CONFIG)
    apply_seen = {}
    monkeypatch.setattr(cli, "resolve_viewer_source", lambda **kwargs: source)
    monkeypatch.setattr(cli, "load_config", lambda path: base_config)

    def fake_apply(config, **kwargs):
        apply_seen.update(kwargs)
        return config, True

    monkeypatch.setattr(cli, "apply_viewer_overrides", fake_apply)
    monkeypatch.setattr(
        cli,
        "simulate_in_viewer",
        lambda selected, config, **kwargs: LiveViewerResult(
            2, True, {"passed": False}, None
        ),
    )
    args = cli.build_parser().parse_args(
        [
            "view",
            "--config",
            "config.json",
            "--clockwise-orbit-deg",
            "12.5",
            "--root-delta-cube-mm",
            "1",
            "-2",
            "3",
            "--wrist-local-rotvec-deg",
            "4",
            "-5",
            "6",
        ]
    )

    assert args.func(args) == 2
    assert apply_seen["clockwise_orbit_deg"] == pytest.approx(12.5)
    assert apply_seen["root_delta_cube_mm"] == pytest.approx([1.0, -2.0, 3.0])
    assert apply_seen["wrist_local_rotvec_deg"] == pytest.approx(
        [4.0, -5.0, 6.0]
    )


def test_v5_rejects_press_and_relative_pose_overrides_are_mutually_exclusive():
    config = load_config(CONFIG)
    schema_v5 = deepcopy(config)
    schema_v5["schema_version"] = 5

    with pytest.raises(ValueError, match="not supported by schema-v5"):
        apply_viewer_overrides(schema_v5, press_mm=5.0)
    with pytest.raises(ValueError, match="mutually exclusive"):
        apply_viewer_overrides(
            config,
            root_cube_distance_mm=145.0,
            cube_in_root_mm=[90.0, -28.0, 105.0],
        )


def test_real_v5_override_revalidates_distance_targets_and_default_monitor():
    config = load_config(V5_CONFIG)
    source_rotation = viewer_module._rpy_matrix(config["hand_pose"]["rpy_deg"])
    source_relative = source_rotation.T @ (
        viewer_module._configured_cube_center(config)
        - np.asarray(config["hand_pose"]["translation_m"])
    )
    overridden, changed = apply_viewer_overrides(
        config,
        root_cube_distance_mm=145.0,
        finger_down_deg=15.0,
        hand_roll_deg=1.5,
        hand_yaw_deg=-2.5,
        grasp_target_rad={"left_hand_thumb_bend_joint_actuator": 1.2},
        manipulation_delta_rad={"left_hand_thumb_bend_joint_actuator": -0.1},
    )
    rotation = viewer_module._rpy_matrix(overridden["hand_pose"]["rpy_deg"])
    relative = rotation.T @ (
        viewer_module._configured_cube_center(overridden)
        - np.asarray(overridden["hand_pose"]["translation_m"])
    )
    model, _ = build_model(overridden)
    binding = resolve_joint_monitor(model, overridden, None)

    assert changed
    assert np.linalg.norm(relative) == pytest.approx(0.145, abs=1e-12)
    np.testing.assert_allclose(
        relative / np.linalg.norm(relative),
        source_relative / np.linalg.norm(source_relative),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(overridden["hand_pose"]["translation_m"]),
        viewer_module._configured_cube_center(overridden) - rotation @ relative,
        atol=1e-12,
    )
    assert overridden["control"]["grasp_targets_rad"][
        "left_hand_thumb_bend_joint_actuator"
    ] == pytest.approx(1.2)
    assert binding is not None
    assert binding.actuator_name == "left_hand_thumb_bend_joint_actuator"
    with pytest.raises(ValueError, match="not supported by schema-v5"):
        apply_viewer_overrides(config, press_mm=5.0)


def test_repeatable_actuator_override_parser_is_strict():
    parsed = parse_actuator_overrides(
        [
            "left_hand_thumb_bend_joint_actuator=1.2",
            "left_hand_index_joint2_actuator=1.7",
        ],
        option="--grasp-target-rad",
    )
    assert parsed == {
        "left_hand_thumb_bend_joint_actuator": 1.2,
        "left_hand_index_joint2_actuator": 1.7,
    }
    with pytest.raises(ValueError, match="unknown or inactive"):
        parse_actuator_overrides(
            ["left_hand_ring_joint1_actuator=0.1"],
            option="--grasp-target-rad",
        )


def test_v9_grasp_override_changes_preload_but_not_nominal_actual_pose():
    config = load_config(V9_CONFIG)
    original_nominal = deepcopy(
        config["grasp_pose"]["nominal_joint_qpos_rad"]
    )

    overridden, changed = apply_viewer_overrides(
        config,
        grasp_target_rad={"left_hand_thumb_bend_joint_actuator": 1.65},
    )

    assert changed
    assert overridden["control"]["contact_preload_targets_rad"][
        "left_hand_thumb_bend_joint_actuator"
    ] == pytest.approx(1.65)
    assert overridden["grasp_pose"]["nominal_joint_qpos_rad"] == original_nominal
    assert overridden["run_context"] == {"kind": "parameter_override_run"}


def test_v9_actual_grasp_pose_telemetry_separates_three_values():
    config = load_config(V9_CONFIG)
    model, _ = build_model(config)
    data = mujoco.MjData(model)
    thumb_name = "left_hand_thumb_bend_joint_actuator"
    thumb_id = model.actuator(thumb_name).id
    thumb_joint = int(model.actuator_trnid[thumb_id, 0])
    data.qpos[int(model.jnt_qposadr[thumb_joint])] = 1.48
    mujoco.mj_forward(model, data)

    stable = np.asarray(
        [
            config["grasp_pose"]["nominal_joint_qpos_rad"][name]
            for name in ACTIVE_ACTUATORS
        ],
        dtype=np.float64,
    )
    stable[0] = 1.49
    rows = actual_grasp_pose_telemetry(model, data, config, stable)

    assert rows is not None
    assert rows[thumb_name]["measured_qpos_rad"] == pytest.approx(1.48)
    assert rows[thumb_name]["lock_sample_qpos_rad"] == pytest.approx(1.48)
    assert rows[thumb_name]["stable_window_median_qpos_rad"] == pytest.approx(1.49)
    assert rows[thumb_name]["nominal_actual_qpos_rad"] == pytest.approx(1.5)
    assert rows[thumb_name]["contact_preload_command_rad"] == pytest.approx(1.6)
    text = format_actual_grasp_pose_telemetry(rows)
    assert "preload is diagnostic, not pose evidence" in text
    assert "stable_median=1.490000 rad" in text
    assert "lock_sample=1.480000 rad" in text
    assert "actual=1.480000 rad" in text
    with pytest.raises(ValueError, match="repeats actuator"):
        parse_actuator_overrides(
            [
                "left_hand_thumb_bend_joint_actuator=1.0",
                "left_hand_thumb_bend_joint_actuator=1.1",
            ],
            option="--grasp-target-rad",
        )
    with pytest.raises(ValueError, match="finite number"):
        parse_actuator_overrides(
            ["left_hand_thumb_bend_joint_actuator=nan"],
            option="--grasp-target-rad",
        )


def test_v4_press_override_is_absolute_from_versioned_reference_pose():
    config = load_config(V4_CONFIG)
    reference_z = float(
        config["pose_constraints"]["reference_hand_translation_m"][2]
    )

    overridden, changed = apply_viewer_overrides(config, press_mm=5.0)

    assert changed
    assert overridden["hand_pose"]["translation_m"][2] == pytest.approx(
        reference_z - 0.005, abs=1e-12
    )
    # The template already has a 5 mm press; treating --press-mm as an
    # incremental displacement would incorrectly lower this by another 5 mm.
    assert overridden["hand_pose"]["translation_m"][2] == pytest.approx(
        config["hand_pose"]["translation_m"][2], abs=1e-12
    )


def test_physics_to_display_copy_is_one_way():
    config = load_config(CONFIG)
    physics_model, _ = build_model(config)
    display_model, _ = build_model(config)
    physics_data = mujoco.MjData(physics_model)
    display_data = mujoco.MjData(display_model)
    physics_data.qpos[:] = physics_model.qpos0
    physics_data.qpos[-7:-4] += [0.001, -0.002, 0.003]
    mujoco.mj_forward(physics_model, physics_data)

    copy_physics_to_display(
        physics_model, physics_data, display_model, display_data
    )
    physics_qpos = physics_data.qpos.copy()
    np.testing.assert_array_equal(display_data.qpos, physics_qpos)
    display_data.qpos[:] += 0.5
    mujoco.mj_forward(display_model, display_data)

    np.testing.assert_array_equal(physics_data.qpos, physics_qpos)


def test_joint_monitor_defaults_for_v5_and_reports_pad_telemetry():
    config = load_config(CONFIG)
    model, _ = build_model(config)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    v5_identity = {"schema_version": 5}

    binding = resolve_joint_monitor(model, v5_identity, None)

    assert binding is not None
    assert binding.actuator_name == "left_hand_thumb_bend_joint_actuator"
    data.ctrl[binding.actuator_id] = 1.2
    data.qpos[binding.qpos_adr] = 1.1
    traces = {
        "finger_contact_force": np.asarray([[2.5, 0.0, 0.0]]),
        "distal_pad_force_n": np.asarray([[2.0, 0.0, 0.0]]),
        "distal_pad_force_fraction": np.asarray([[0.8, 0.0, 0.0]]),
        "distal_active_taxel_count": np.asarray([[4, 0, 0]]),
    }
    telemetry = joint_monitor_telemetry(model, data, traces, 0, binding)
    line = format_joint_monitor_telemetry(telemetry)

    assert telemetry["error_rad"] == pytest.approx(0.1)
    assert telemetry["contact_force_n"] == pytest.approx(2.5)
    assert telemetry["pad_force_n"] == pytest.approx(2.0)
    assert telemetry["pad_force_fraction"] == pytest.approx(0.8)
    assert telemetry["active_taxel_count"] == 4
    assert "target_rad=1.2000rad" in line
    assert "active_taxel_count=4" in line

    with pytest.raises(ValueError, match="unknown actuator"):
        resolve_joint_monitor(model, config, "not_an_actuator")


def test_joint_monitor_axis_is_appended_to_user_scene():
    config = load_config(CONFIG)
    model, _ = build_model(config)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    binding = resolve_joint_monitor(
        model,
        config,
        "left_hand_thumb_bend_joint_actuator",
    )
    assert binding is not None
    scene = mujoco.MjvScene(model, maxgeom=4)
    handle = SimpleNamespace(user_scn=scene)

    _append_joint_axis_marker(handle, data, binding)

    assert scene.ngeom == 1
    assert int(scene.geoms[0].type) == int(mujoco.mjtGeom.mjGEOM_ARROW)


def test_joint_pair_resolves_and_measures_in_cube_coordinates():
    model, _ = build_model(load_config(CONFIG))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    names = ("left_hand_index_joint1", "left_hand_mid_joint1")
    binding = resolve_joint_pair(model, names)
    assert binding is not None

    telemetry = joint_pair_telemetry(data, binding)
    first = np.asarray(data.xanchor[binding.first_joint_id])
    second = np.asarray(data.xanchor[binding.second_joint_id])
    cube_rotation = np.asarray(data.xmat[binding.cube_body_id]).reshape(3, 3)
    expected = cube_rotation.T @ (second - first)
    expected_angle = math.degrees(
        math.acos(abs(float(expected[1])) / float(np.linalg.norm(expected)))
    )
    np.testing.assert_allclose(telemetry["vector_cube_m"], expected)
    assert telemetry["angle_to_cube_y_deg"] == pytest.approx(expected_angle)
    line = format_joint_pair_telemetry(telemetry)
    assert "left_hand_index_joint1 -> left_hand_mid_joint1" in line
    assert "angle_to_cube_y=" in line

    with pytest.raises(ValueError, match="distinct"):
        resolve_joint_pair(model, (names[0], names[0]))
    with pytest.raises(ValueError, match="unknown joint"):
        resolve_joint_pair(model, (names[0], "not_a_joint"))


def test_joint_pair_overlay_draws_anchors_axes_line_and_cube_y_reference():
    model, _ = build_model(load_config(CONFIG))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    binding = resolve_joint_pair(
        model,
        ("left_hand_index_joint1", "left_hand_mid_joint1"),
    )
    assert binding is not None
    scene = mujoco.MjvScene(model, maxgeom=6)
    handle = SimpleNamespace(user_scn=scene)

    _append_joint_pair_markers(handle, data, binding, visible=True)

    assert scene.ngeom == 6
    assert [int(scene.geoms[index].type) for index in range(6)] == [
        int(mujoco.mjtGeom.mjGEOM_SPHERE),
        int(mujoco.mjtGeom.mjGEOM_ARROW),
        int(mujoco.mjtGeom.mjGEOM_SPHERE),
        int(mujoco.mjtGeom.mjGEOM_ARROW),
        int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        int(mujoco.mjtGeom.mjGEOM_ARROW),
    ]
    np.testing.assert_allclose(
        scene.geoms[0].pos,
        data.xanchor[binding.first_joint_id],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        scene.geoms[2].pos,
        data.xanchor[binding.second_joint_id],
        atol=1e-12,
    )
    assert scene.geoms[0].label == "left_hand_index_joint1"
    assert scene.geoms[2].label == "left_hand_mid_joint1"
    assert scene.geoms[4].label.startswith("joint_pair_")
    assert scene.geoms[5].label == "cube_+Y_at_joint_pair"

    limited = mujoco.MjvScene(model, maxgeom=3)
    _append_joint_pair_markers(
        SimpleNamespace(user_scn=limited), data, binding, visible=True
    )
    assert limited.ngeom == 3
    _append_joint_pair_markers(handle, data, binding, visible=False)
    assert scene.ngeom == 6


def test_hand_root_and_cube_coordinate_frames_use_compiled_body_transforms():
    config = load_config(CONFIG)
    model, _ = build_model(config)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    scene = mujoco.MjvScene(model, maxgeom=8)
    handle = SimpleNamespace(user_scn=scene)

    _append_coordinate_frame_markers(
        handle,
        model,
        data,
        visible=True,
    )

    assert scene.ngeom == 6
    colors = (
        np.asarray([0.95, 0.10, 0.10, 1.0]),
        np.asarray([0.10, 0.90, 0.20, 1.0]),
        np.asarray([0.10, 0.35, 1.00, 1.0]),
    )
    specs = (
        ("hand_root", "left_hand_link", 0.035),
        ("cube", "three_finger_cube", 0.030),
    )
    for frame_index, (frame_label, body_name, axis_length) in enumerate(specs):
        body_id = int(model.body(body_name).id)
        origin = np.asarray(data.xpos[body_id])
        rotation = np.asarray(data.xmat[body_id]).reshape(3, 3)
        for axis_index, axis_name in enumerate(("X", "Y", "Z")):
            geom = scene.geoms[3 * frame_index + axis_index]
            assert int(geom.type) == int(mujoco.mjtGeom.mjGEOM_ARROW)
            assert geom.label == f"{frame_label}_{axis_name}"
            np.testing.assert_allclose(geom.pos, origin, atol=1e-12)
            np.testing.assert_allclose(
                np.asarray(geom.mat).reshape(3, 3)[:, 2],
                rotation[:, axis_index],
                atol=1e-7,
            )
            assert geom.size[2] == pytest.approx(axis_length)
            np.testing.assert_allclose(geom.rgba, colors[axis_index], atol=1e-7)


def test_coordinate_frames_respect_visibility_and_scene_capacity():
    model, _ = build_model(load_config(CONFIG))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    scene = mujoco.MjvScene(model, maxgeom=4)
    handle = SimpleNamespace(user_scn=scene)

    _append_coordinate_frame_markers(handle, model, data, visible=False)
    assert scene.ngeom == 0

    _append_coordinate_frame_markers(handle, model, data, visible=True)
    assert scene.ngeom == 4


def test_v8_markers_append_closure_arrows_vertical_corridor_and_path():
    config = load_config(CONFIG)
    model, _ = build_model(config)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    binding = resolve_joint_monitor(
        model,
        config,
        "left_hand_thumb_bend_joint_actuator",
    )
    assert binding is not None
    scene = mujoco.MjvScene(model, maxgeom=64)
    handle = SimpleNamespace(user_scn=scene)
    cube_pos = np.asarray(
        [
            [0.0710, -0.0270, 0.1150],
            [0.0710, -0.0270, 0.1150],
            [0.0714, -0.0270, 0.1200],
            [0.0712, -0.0270, 0.1260],
        ]
    )
    witnesses = np.asarray(
        [
            [0.040, -0.027, 0.115],
            [0.102, -0.032, 0.115],
            [0.102, -0.022, 0.115],
        ]
    )
    outward = np.asarray(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    velocity = -0.01 * outward
    traces = {
        "target_face_contact_centroid_world_m": np.tile(
            witnesses[None, :, :], (4, 1, 1)
        ),
        "target_face_contact_centroid_valid": np.ones((4, 3), dtype=bool),
        "closure_witness_world_m": np.tile(witnesses[None, :, :], (4, 1, 1)),
        "closure_command_velocity_world_m_s": np.tile(
            velocity[None, :, :], (4, 1, 1)
        ),
        "closure_cube_outward_normal_world": np.tile(
            outward[None, :, :], (4, 1, 1)
        ),
        "closure_alignment_valid": np.ones((4, 3), dtype=bool),
        "cube_pos": cube_pos,
        "control_state": np.asarray(
            ["VERIFY", "MANIPULATE", "MANIPULATE", "HOLD"]
        ),
        "manipulation_start_step": np.asarray(1, dtype=np.int64),
    }

    viewer_module._update_viewer_markers(
        handle,
        model,
        data,
        traces,
        3,
        show_alignment=True,
        joint_monitor=binding,
    )

    types = [int(scene.geoms[index].type) for index in range(scene.ngeom)]
    assert scene.ngeom == 16
    assert types.count(int(mujoco.mjtGeom.mjGEOM_SPHERE)) == 3
    assert types.count(int(mujoco.mjtGeom.mjGEOM_BOX)) == 3
    # Six v8 arrows plus the existing monitored-joint arrow coexist.
    assert types.count(int(mujoco.mjtGeom.mjGEOM_ARROW)) == 7
    assert types.count(int(mujoco.mjtGeom.mjGEOM_CAPSULE)) == 1
    assert types.count(int(mujoco.mjtGeom.mjGEOM_LINE)) == 2
    corridor = scene.geoms[types.index(int(mujoco.mjtGeom.mjGEOM_CAPSULE))]
    assert corridor.size[0] == pytest.approx(0.002)
    np.testing.assert_allclose(
        scene.geoms[scene.ngeom - 1].rgba,
        [1.0, 0.72, 0.05, 0.95],
        atol=1e-7,
    )


def test_top_level_marker_reset_keeps_legacy_joint_axis_compatible():
    config = load_config(CONFIG)
    model, _ = build_model(config)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    binding = resolve_joint_monitor(
        model,
        config,
        "left_hand_thumb_bend_joint_actuator",
    )
    assert binding is not None
    scene = mujoco.MjvScene(model, maxgeom=16)
    scene.ngeom = 9
    handle = SimpleNamespace(user_scn=scene)
    traces = {
        "target_face_contact_centroid_world_m": np.asarray(
            [[[0.04, -0.027, 0.115], [0.10, -0.032, 0.115], [0.10, -0.022, 0.115]]]
        ),
        "target_face_contact_centroid_valid": np.ones((1, 3), dtype=bool),
    }

    viewer_module._update_viewer_markers(
        handle,
        model,
        data,
        traces,
        0,
        show_alignment=True,
        joint_monitor=binding,
    )
    assert scene.ngeom == 7
    assert int(scene.geoms[scene.ngeom - 1].type) == int(
        mujoco.mjtGeom.mjGEOM_ARROW
    )

    viewer_module._update_viewer_markers(
        handle,
        model,
        data,
        traces,
        0,
        show_alignment=True,
        joint_monitor=binding,
        show_coordinate_frames=True,
    )
    assert scene.ngeom == 13
    assert [scene.geoms[index].label for index in range(6)] == [
        "hand_root_X",
        "hand_root_Y",
        "hand_root_Z",
        "cube_X",
        "cube_Y",
        "cube_Z",
    ]

    viewer_module._update_viewer_markers(
        handle,
        model,
        data,
        traces,
        0,
        show_alignment=False,
        joint_monitor=binding,
    )
    assert scene.ngeom == 1
    assert int(scene.geoms[0].type) == int(mujoco.mjtGeom.mjGEOM_ARROW)


def test_reference_comparison_is_exact_and_ignores_video_sampling_metadata():
    actual = {
        "time": np.asarray([0.001, 0.002]),
        "cube_pos": np.zeros((2, 3)),
        "video_frame_steps": np.asarray([], dtype=np.int64),
    }
    reference = deepcopy(actual)
    reference["video_frame_steps"] = np.asarray([0], dtype=np.int64)
    assert compare_reference_trace(actual, reference) == (True, [])

    reference["cube_pos"][1, 2] = np.nextafter(0.0, 1.0)
    matches, fields = compare_reference_trace(actual, reference)
    assert not matches
    assert fields == ["cube_pos"]


def test_parameter_override_output_replaces_catalog_validation_status(tmp_path):
    config = load_config(CONFIG)
    config.pop("experiment_status", None)
    config["run_context"] = {"kind": "parameter_override_run"}
    output = tmp_path / "live-output"
    source = ViewerSource(CONFIG, None, "nominal", False)
    summary = {
        "passed": False,
        "failed_checks": ["minimum_lift_reached"],
    }

    _write_live_output(
        output,
        config,
        summary,
        {"time": np.asarray([0.001])},
        source=source,
        overridden=True,
        reference_match=None,
    )

    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    status = result["config"]["experiment_status"]
    assert result["run_kind"] == "parameter_override_run"
    assert status["classification"] == "parameter_override_run"
    assert status["passed"] is False
    assert status["failed_checks"] == ["minimum_lift_reached"]


def test_live_viewer_integrates_real_session_with_isolated_display_data(monkeypatch):
    from mujoco import viewer as mujoco_viewer

    config = load_config(CONFIG)
    source = ViewerSource(CONFIG, None, "nominal", False)
    created = {}
    real_session = viewer_module.SimulationSession

    def session_factory(value):
        session = real_session(value)
        created["physics_data"] = session.data
        return session

    class FakeHandle:
        def __init__(self, model, data):
            del model
            created["display_data"] = data
            self.cam = mujoco.MjvCamera()
            self.user_scn = SimpleNamespace(ngeom=0)
            self._running = True
            self._sync_count = 0
            created["display_camera"] = self.cam

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def lock(self):
            return nullcontext()

        def is_running(self):
            if self._running:
                self._running = False
                return True
            return False

        def sync(self):
            self._sync_count += 1
            if self._sync_count == 1:
                # Emulate an orbit/pan/zoom gesture after the initial scene
                # sync.  A later frame must not reinitialize this camera.
                self.cam.azimuth = 17.25
                self.cam.elevation = -28.5
                self.cam.distance = 0.321
                self.cam.lookat[:] = [0.11, -0.07, 0.19]
            elif self._sync_count == 2:
                created["camera_after_frame"] = {
                    "azimuth": float(self.cam.azimuth),
                    "elevation": float(self.cam.elevation),
                    "distance": float(self.cam.distance),
                    "lookat": np.array(self.cam.lookat, copy=True),
                }

    monkeypatch.setattr(viewer_module, "SimulationSession", session_factory)
    monkeypatch.setattr(viewer_module, "_require_interactive_gl", lambda: None)
    monkeypatch.setattr(
        mujoco_viewer,
        "launch_passive",
        lambda model, data, key_callback: FakeHandle(model, data),
    )
    clock = iter((0.0, 10.0))
    monkeypatch.setattr(viewer_module.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(viewer_module.time, "sleep", lambda seconds: None)

    result = simulate_in_viewer(source, config)

    assert result.completed
    assert result.exit_code == 0
    assert result.summary is not None and result.summary["passed"]
    assert created["physics_data"] is not created["display_data"]
    assert int(created["display_camera"].type) == int(
        mujoco.mjtCamera.mjCAMERA_FREE
    )
    assert created["display_camera"].fixedcamid == -1
    assert created["camera_after_frame"] == {
        "azimuth": pytest.approx(17.25),
        "elevation": pytest.approx(-28.5),
        "distance": pytest.approx(0.321),
        "lookat": pytest.approx([0.11, -0.07, 0.19]),
    }


def test_live_viewer_pauses_exactly_at_grasp_lock_and_space_resumes(monkeypatch):
    from mujoco import viewer as mujoco_viewer
    from mujoco.glfw import glfw

    config = load_config(CONFIG)
    source = ViewerSource(CONFIG, None, "nominal", False)
    created = {}
    real_session = viewer_module.SimulationSession

    def session_factory(value):
        session = real_session(value)
        created["session"] = session
        return session

    class FakeHandle:
        def __init__(self, model, data, key_callback):
            del model, data
            self.cam = mujoco.MjvCamera()
            self.user_scn = SimpleNamespace(ngeom=0)
            self._key_callback = key_callback
            self._iterations = 0
            self._resumed = False

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def lock(self):
            return nullcontext()

        def is_running(self):
            self._iterations += 1
            session = created["session"]
            return self._iterations <= 8 and not session.complete

        def sync(self):
            session = created["session"]
            if (
                session.controller is not None
                and session.controller.acquired
                and not self._resumed
            ):
                created["resume_step_index"] = session.step_index
                created["acquisition_step"] = (
                    session.controller.grasp_acquisition_step
                )
                self._resumed = True
                # The event pause has just set paused=True.  Space must resume
                # from that exact sample.  Without the event pause this would
                # instead pause the still-running session and the test would
                # leave it incomplete.
                self._key_callback(glfw.KEY_SPACE)

    monkeypatch.setattr(viewer_module, "SimulationSession", session_factory)
    monkeypatch.setattr(viewer_module, "_require_interactive_gl", lambda: None)
    monkeypatch.setattr(
        mujoco_viewer,
        "launch_passive",
        lambda model, data, key_callback: FakeHandle(
            model, data, key_callback
        ),
    )
    clock = iter(float(index * 10) for index in range(20))
    monkeypatch.setattr(viewer_module.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(viewer_module.time, "sleep", lambda seconds: None)

    result = simulate_in_viewer(
        source,
        config,
        pause_at_event="grasp_lock",
    )

    assert result.completed
    assert created["resume_step_index"] == created["acquisition_step"] + 1


def test_live_viewer_loop_rearms_grasp_lock_pause(monkeypatch):
    from mujoco import viewer as mujoco_viewer
    from mujoco.glfw import glfw

    config = load_config(CONFIG)
    source = ViewerSource(CONFIG, None, "nominal", False)
    counts = {"reset": 0, "finalize": 0, "clock": -10.0}
    created = {"resumed_generations": []}
    real_session = viewer_module.SimulationSession

    class TrackingSession(real_session):
        def reset(self):
            counts["reset"] += 1
            return super().reset()

        def finalize(self, **kwargs):
            counts["finalize"] += 1
            return super().finalize(**kwargs)

    def session_factory(value):
        session = TrackingSession(value)
        created["session"] = session
        return session

    class FakeHandle:
        def __init__(self, model, data, key_callback):
            del model, data
            self.cam = mujoco.MjvCamera()
            self.user_scn = SimpleNamespace(ngeom=0)
            self._key_callback = key_callback
            self._iterations = 0

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def lock(self):
            return nullcontext()

        def is_running(self):
            self._iterations += 1
            return counts["finalize"] < 2 and self._iterations <= 12

        def sync(self):
            session = created["session"]
            generation = counts["reset"]
            if (
                session.controller is not None
                and session.controller.acquired
                and generation not in created["resumed_generations"]
            ):
                created["resumed_generations"].append(generation)
                self._key_callback(glfw.KEY_SPACE)

    def monotonic():
        counts["clock"] += 10.0
        return counts["clock"]

    monkeypatch.setattr(viewer_module, "SimulationSession", session_factory)
    monkeypatch.setattr(viewer_module, "_require_interactive_gl", lambda: None)
    monkeypatch.setattr(
        mujoco_viewer,
        "launch_passive",
        lambda model, data, key_callback: FakeHandle(
            model, data, key_callback
        ),
    )
    monkeypatch.setattr(viewer_module.time, "monotonic", monotonic)
    monkeypatch.setattr(viewer_module.time, "sleep", lambda seconds: None)

    result = simulate_in_viewer(
        source,
        config,
        loop=True,
        pause_at_event="grasp_lock",
    )

    assert result.completed
    assert counts["finalize"] == 2
    assert created["resumed_generations"] == [1, 2]


def test_live_viewer_restart_rearms_grasp_lock_pause(monkeypatch):
    from mujoco import viewer as mujoco_viewer
    from mujoco.glfw import glfw

    config = load_config(CONFIG)
    source = ViewerSource(CONFIG, None, "nominal", False)
    counts = {"reset": 0, "clock": -10.0}
    created = {"handled_generations": []}
    real_session = viewer_module.SimulationSession

    class TrackingSession(real_session):
        def reset(self):
            counts["reset"] += 1
            return super().reset()

    def session_factory(value):
        session = TrackingSession(value)
        created["session"] = session
        return session

    class FakeHandle:
        def __init__(self, model, data, key_callback):
            del model, data
            self.cam = mujoco.MjvCamera()
            self.user_scn = SimpleNamespace(ngeom=0)
            self._key_callback = key_callback
            self._iterations = 0
            self._resumed_after_reset = False

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def lock(self):
            return nullcontext()

        def is_running(self):
            self._iterations += 1
            session = created["session"]
            return self._iterations <= 12 and not session.complete

        def sync(self):
            session = created["session"]
            generation = counts["reset"]
            if generation == 2 and not self._resumed_after_reset:
                # Restart preserves the ordinary pause state, so resume the
                # fresh run explicitly before waiting for its event pause.
                self._resumed_after_reset = True
                self._key_callback(glfw.KEY_SPACE)
                return
            if (
                session.controller is not None
                and session.controller.acquired
                and generation not in created["handled_generations"]
            ):
                created["handled_generations"].append(generation)
                if generation == 1:
                    self._key_callback(glfw.KEY_R)
                else:
                    self._key_callback(glfw.KEY_SPACE)

    def monotonic():
        counts["clock"] += 10.0
        return counts["clock"]

    monkeypatch.setattr(viewer_module, "SimulationSession", session_factory)
    monkeypatch.setattr(viewer_module, "_require_interactive_gl", lambda: None)
    monkeypatch.setattr(
        mujoco_viewer,
        "launch_passive",
        lambda model, data, key_callback: FakeHandle(
            model, data, key_callback
        ),
    )
    monkeypatch.setattr(viewer_module.time, "monotonic", monotonic)
    monkeypatch.setattr(viewer_module.time, "sleep", lambda seconds: None)

    result = simulate_in_viewer(
        source,
        config,
        pause_at_event="grasp_lock",
    )

    assert result.completed
    assert counts["reset"] == 2
    assert created["handled_generations"] == [1, 2]


def test_draggable_camera_is_free_and_initially_targets_cube():
    config = load_config(CONFIG)
    model, _ = build_model(config)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
    camera.fixedcamid = int(model.camera("three_finger_camera").id)
    handle = SimpleNamespace(cam=camera)

    viewer_module._initialize_draggable_camera(handle, model, data)

    cube_body_id = int(model.body("three_finger_cube").id)
    assert int(camera.type) == int(mujoco.mjtCamera.mjCAMERA_FREE)
    assert camera.fixedcamid == -1
    assert camera.trackbodyid == -1
    assert camera.lookat == pytest.approx(data.xpos[cube_body_id])
    assert math.isfinite(camera.azimuth)
    assert math.isfinite(camera.elevation)
    assert math.isfinite(camera.distance) and camera.distance > 0.0


def test_live_viewer_returns_three_when_window_closes_before_completion(monkeypatch):
    from mujoco import viewer as mujoco_viewer

    class ClosedHandle:
        cam = mujoco.MjvCamera()
        user_scn = SimpleNamespace(ngeom=0)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def lock(self):
            return nullcontext()

        def is_running(self):
            return False

        def sync(self):
            return None

    monkeypatch.setattr(viewer_module, "_require_interactive_gl", lambda: None)
    monkeypatch.setattr(
        mujoco_viewer,
        "launch_passive",
        lambda model, data, key_callback: ClosedHandle(),
    )

    result = simulate_in_viewer(
        ViewerSource(CONFIG, None, "nominal", False),
        load_config(CONFIG),
    )

    assert result == LiveViewerResult(3, False, None, None)
