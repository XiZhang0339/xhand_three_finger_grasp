#!/usr/bin/env python3
"""CLI entry point for the resumable six-seed acquisition campaign."""

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xhand_grasp.tuning.pose_preserving_seed_dynamic import main


if __name__ == "__main__":
    raise SystemExit(main())
