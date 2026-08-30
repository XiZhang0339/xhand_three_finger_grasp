#!/usr/bin/env python3
"""Deterministic left-hand, three-finger cube-lift experiment for XHAND1.

This repository-level module is a compatibility façade.  The implementation
lives in :mod:`xhand_grasp`, where experiments can share one simulation loop
without importing this CLI entry point.
"""

from __future__ import annotations

from xhand_grasp.artifacts import (
    file_sha256,
    json_compatible as _json_compatible,
    json_text,
    run_metadata,
    write_json,
)
from xhand_grasp.cli import (
    build_parser,
    command_catalog,
    command_robustness,
    command_run,
    command_tune,
    command_view,
    main,
)
from xhand_grasp.config import (
    ACTIVE_ACTUATORS,
    ACTIVE_FINGERS,
    ALL_FINGERS,
    DEFAULT_CONFIG,
    DISTAL_BODY_NAMES,
    INACTIVE_ACTUATORS,
    SCRIPT_DIR,
    SEARCH_TARGET_BOUNDS,
    _finite_sequence,
    load_config,
    validate_config,
)
from xhand_grasp.evaluation import (
    evaluate_trace,
    orientation_angles as _orientation_angles,
)
from xhand_grasp.rendering import (
    open_video_encoder as _open_video_encoder,
    probe_video as _probe_video,
)
from xhand_grasp.scene import (
    ModelInfo,
    _body_part,
    _is_descendant,
    build_model,
    cube_inertia,
    rpy_degrees_to_quaternion,
)
from xhand_grasp.search import (
    _perturbed_cube_cases,
    _run_candidate,
    _run_candidates,
    _sample_candidate,
    candidate_rank,
    latin_hypercube,
    normalized_acceptance_margins,
    robustness,
    robustness_cases,
    tune,
)
from xhand_grasp.simulation import (
    SimulationSession,
    SimulationStep,
    _contact_snapshot,
    run_simulation,
)
from xhand_grasp.trajectory import (
    _phase_steps,
    actuator_target_vector,
    minimum_jerk,
    preflight_config,
    smoothstep,
)


__all__ = [
    "ACTIVE_ACTUATORS",
    "ACTIVE_FINGERS",
    "ALL_FINGERS",
    "DEFAULT_CONFIG",
    "DISTAL_BODY_NAMES",
    "INACTIVE_ACTUATORS",
    "SCRIPT_DIR",
    "SEARCH_TARGET_BOUNDS",
    "ModelInfo",
    "SimulationSession",
    "SimulationStep",
    "actuator_target_vector",
    "build_model",
    "build_parser",
    "candidate_rank",
    "command_catalog",
    "command_robustness",
    "command_run",
    "command_tune",
    "command_view",
    "cube_inertia",
    "evaluate_trace",
    "file_sha256",
    "json_text",
    "latin_hypercube",
    "load_config",
    "main",
    "minimum_jerk",
    "normalized_acceptance_margins",
    "preflight_config",
    "robustness",
    "robustness_cases",
    "rpy_degrees_to_quaternion",
    "run_metadata",
    "run_simulation",
    "smoothstep",
    "tune",
    "validate_config",
    "write_json",
]


if __name__ == "__main__":
    main()
