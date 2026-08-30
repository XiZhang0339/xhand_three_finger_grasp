# XHAND left 79 mm three-finger grasp pose

This directory is a portable export of the high-quality three-finger grasp
selected as `best_attempt` in the schema-v16 rolling-aware campaign.

The grasp itself passed its strict acquisition checks.  The source trajectory
is still classified as a manipulation diagnostic because its lift exceeded
the jerk and lateral-displacement limits.  Consumers must not interpret the
bundle as a fully validated manipulation policy.

## Stable interface

Use `portable_grasp_pose.json` when importing the grasp into another project.
It records:

- the fixed hand-root world pose;
- the free cube pose and physical parameters;
- actuator names and inactive-finger zero targets;
- precontact and contact-preload commands;
- the measured eight-joint contact pose;
- target and measured cube-local contact points; and
- concise grasp-validation evidence.

`best/resolved_config.json` is the complete xhand1-specific configuration.
`best/trace.npz` and `best/result.json` preserve the original simulation
evidence.  `manifest.json` authenticates the copied files and records the
required model and lock-file hashes.

Verify the standalone files with the standard-library-only helper:

```bash
python3 verify_bundle.py
```

From an xhand1 checkout, also verify the runtime bindings:

```bash
python3 verify_bundle.py --repo-root ../..
```

The actual grasp pose is the measured joint-position median over the verified
250 ms contact window.  It is not the actuator preload command.

## View with xhand1

From the xhand1 repository root:

```bash
MUJOCO_GL=glfw ./scripts/uv.sh run --frozen python grasp_cube.py view \
  --catalog grasp_pose_exports/xhand_left_79mm_three_finger_grasp_v1/catalog.json \
  --trajectory best_attempt \
  --pause-at-event grasp_lock \
  --show-joint-pair left_hand_index_joint1 left_hand_mid_joint1 \
  --show-coordinate-frames \
  --joint-monitor left_hand_thumb_bend_joint_actuator \
  --loop
```

To rerun the physics without using the reference trace:

```bash
MUJOCO_GL=glfw ./scripts/uv.sh run --frozen python grasp_cube.py view \
  --config grasp_pose_exports/xhand_left_79mm_three_finger_grasp_v1/best/resolved_config.json \
  --pause-at-event grasp_lock \
  --show-coordinate-frames \
  --loop
```

## Use from another project

The JSON and NPZ files can be parsed independently.  Re-running the exact
MuJoCo dynamics additionally requires the matching `xhand_grasp` source,
`xhand_left.xml`, `assets/left/`, and frozen uv environment.  A different
project may keep this bundle at any absolute path and pass its catalog path to
the xhand1 Viewer.

Keep the `catalog.json` and `best/` relative layout unchanged.  The old
`artifacts/...` paths inside the full config are provenance only; the Viewer
does not dereference them during a normal run.  This bundle is not a resumable
copy of the original search campaign.

This export adds no license grant.  Use and redistribute it only where the
permissions of the parent model and repository allow.
