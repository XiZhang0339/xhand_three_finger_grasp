#!/usr/bin/env python3
"""CLI for schema-v9 full-reset pose/material robustness."""

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xhand_grasp.tuning.actual_contact_grasp_pose_robustness import main


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
