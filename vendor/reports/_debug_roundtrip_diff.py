# Copyright (c) 2026 ysnrfd
"""Param-level save→reload diff: which weights change for clip/wav2vec2?"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from jevx import JevX, JevXForDecision

CASES = [
    "hf-internal-testing/tiny-random-CLIPModel",
    "hf-internal-testing/tiny-random-Wav2Vec2Model",
    "hf-internal-testing/tiny-random-bert",  # control: known-zero
]


def main() -> int:
    torch.set_grad_enabled(False)
    for model_id in CASES:
        print(f"\n===== {model_id} =====")
        wrapper = JevX.from_pretrained(model_id)
        wrapper.eval()
        tmp = Path(tempfile.mkdtemp(prefix="jevx-rt-"))
        try:
            wrapper.save_pretrained(tmp)
            reloaded = JevXForDecision.from_pretrained(tmp)
            reloaded.eval()
            a, b = wrapper.state_dict(), reloaded.state_dict()
            only_a = sorted(set(a) - set(b))
            only_b = sorted(set(b) - set(a))
            changed = {}
            for key in sorted(set(a) & set(b)):
                ta, tb = a[key], b[key]
                if ta.shape != tb.shape:
                    changed[key] = f"SHAPE {tuple(ta.shape)} vs {tuple(tb.shape)}"
                    continue
                delta = float((ta.float() - tb.float()).abs().max())
                if delta > 0:
                    changed[key] = f"{delta:.3e}"
            print("missing after reload:", only_a[:10], f"({len(only_a)})")
            print("extra after reload:", only_b[:10], f"({len(only_b)})")
            print(f"changed tensors: {len(changed)} / {len(set(a) & set(b))}")
            for key, val in list(changed.items())[:15]:
                print(f"  {key}: {val}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
