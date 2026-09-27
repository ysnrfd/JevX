# Copyright (c) 2026 ysnrfd
"""Create the checked-in modality fixtures (deterministic bytes):

* tests/golden/fixtures/input_image_224.png — 224x224 PIL pattern
* tests/golden/fixtures/input_audio_1s.wav — 1 s / 16 kHz via soundfile

Re-running never overwrites existing fixtures (tests may re-create them
identically when missing).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "integration"))

import matrix_common  # noqa: E402

if __name__ == "__main__":
    out = matrix_common.ensure_fixture_assets()
    print(f"fixtures ready under {out}")
    print(f"image: {matrix_common.IMAGE_FIXTURE} ({matrix_common.IMAGE_FIXTURE.stat().st_size} bytes)")
    print(f"wav:   {matrix_common.WAV_FIXTURE} ({matrix_common.WAV_FIXTURE.stat().st_size} bytes)")
