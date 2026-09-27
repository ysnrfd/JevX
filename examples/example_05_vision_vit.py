# Copyright (c) 2026 ysnrfd
"""Example 05 — vision (ViT): decisions from an image state.

What this shows
---------------
Wrapping an image encoder (``hf-internal-testing/tiny-random-ViTModel``,
vision family, native ``ViT pooler``). The state is a PIL image passed as a
raw modality payload (``{"images": ...}``) — JevX routes it through the
model's own ``ViTImageProcessor`` (no tokenizer exists for vision-only
families; audit Q8). Question/schema text still flows through the hashed
question encoder, which needs no tokenizer of its own.

Deterministic input: the same seeded 224x224 pattern the Phase-92 matrix used
(tests/golden/fixtures/input_image_224.png), regenerated inline so this file
is self-contained.

Honesty note: ``tiny-random-ViTModel`` carries RANDOM weights (pipeline
fixture). Printed numbers are real outputs with asserted invariants, not
predictive quality.

Run from the repo root:  python examples/example_05_vision_vit.py
Expected runtime: ~5-15 s (first run downloads the checkpoint + processor;
measured load 10.15 s — reports/REAL_MODEL_VALIDATION.md).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from jevx import JevX

MODEL_ID = "hf-internal-testing/tiny-random-ViTModel"


def fixture_image():
    """Deterministic 224x224 RGB pattern (same recipe as the matrix fixture)."""
    from PIL import Image

    seed = 0
    yy, xx = np.mgrid[0:224, 0:224]
    r = ((xx * 5 + seed) % 256).astype(np.uint8)
    g = ((yy * 11 + seed * 3) % 256).astype(np.uint8)
    b = (((xx + yy) * 7 + seed * 7) % 256).astype(np.uint8)
    cx, cy, rad = 112, 96, 48
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= rad**2
    r[mask], g[mask], b[mask] = 240, 40, 40
    return Image.fromarray(np.stack([r, g, b], axis=-1), mode="RGB")


def main() -> None:
    model = JevX.from_pretrained(MODEL_ID)
    model.eval()
    print(f"model: {MODEL_ID}")
    print(f"adapter: {model.adapter.name} | conversion: {model.capability_report.jevx_conversion}"
          f" | representation_field: {model.capability_report.representation_field}"
          f" | processor_required: {model.capability_report.processor_required}")

    state = {"images": fixture_image()}
    questions = ["What does this image show?", "Is this image acceptable?"]
    options = [["a red circle on a checkered background", "something else"],
               ["accept", "reject"]]

    with torch.no_grad():
        out = model.choice(state, questions, options=options)

    print("\n[choice] decisions over one image")
    for q, opts, dec in zip(questions, options, out.decisions):
        probs = [round(float(p), 4) for p in dec.probabilities[0, : len(opts)]]
        print(f"  q={q!r}")
        print(f"    probs={dict(zip(opts, probs))}")
        print(f"    predicted={dec.predicted_value!r} confidence={float(dec.confidence[0]):.4f}")

    # --- invariants: running the example IS testing -------------------------
    assert model.capability_report.vision and not model.capability_report.text
    assert model.processor is not None, "vision wrap must resolve a processor"
    assert len(out.decisions) == len(questions)
    for opts, dec in zip(options, out.decisions):
        probs = dec.probabilities[0]
        k = len(opts)
        assert torch.all(probs >= 0) and torch.all(probs <= 1)
        assert abs(float(probs[:k].sum()) - 1.0) < 1e-6
        assert 0 <= int(dec.predicted_index[0]) < k
        assert dec.predicted_value in opts
        assert 0.0 <= float(dec.confidence[0]) <= 1.0
    assert bool(torch.isfinite(out.state_representation).all())
    print("\nAll invariants held (vision capability detected, processor resolved, "
          "probs sum to 1, index in schema, state repr finite).")


if __name__ == "__main__":
    main()
