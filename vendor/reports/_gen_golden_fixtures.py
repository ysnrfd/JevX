# Copyright (c) 2026 ysnrfd
"""Capture the §94 golden reference fixtures by RUNNING the pipeline.

Usage (repo root, network required once):

    python vendor/reports/_gen_golden_fixtures.py

Writes tests/golden/fixtures/golden_*.json — never hand-edited; regenerate
deliberately when a behavior change is intended and update the report.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "golden"))

import test_golden_references as golden  # noqa: E402

if __name__ == "__main__":
    golden.generate_fixtures()
