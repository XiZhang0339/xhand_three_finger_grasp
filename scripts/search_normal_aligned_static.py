#!/usr/bin/env python3
"""Direct CLI for the resumable schema-v8 static pose campaign."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xhand_grasp.tuning.normal_aligned_static_runner import main


if __name__ == "__main__":
    raise SystemExit(main())
