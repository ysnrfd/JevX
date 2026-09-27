# Copyright (c) 2026 ysnrfd
"""Example 07 — multimodal (CLIP): decisions from image + text together.

What this shows
---------------
Wrapping a joint image-text encoder (``hf-internal-testing/tiny-random-CLIPModel``,
multimodal family). Both towers run in ONE forward; the extracted state
representation is the POST-projection ``text_embeds`` (the documented
"post-projection for cross-modal comparability" convention of the
compatibility matrix clip row), selected by the semantic-field priority
sentence_embedding -> text_embeds -> image_embeds (adapters/base.py).

State shape: ``{"images": <PIL image>, "texts": "<caption>"}`` — routed
through the model's own ``CLIPProcessor`` (audit Q8 unified call).

Honesty note: ``tiny-random-CLIPModel`` carries RANDOM weights (pipeline
fixture). Printed numbers are real outputs with asserted invariants, not
predictive quality.

Run from the repo root:  python examples/example_07_multimodal_clip.py
Expected runtime: ~5-15 s (first run downloads the checkpoint; measured load
10.94 s — reports/REAL_MODEL_VALIDATION.md).
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

MODEL_ID = "hf-internal-testing/tiny-random-CLIPModel"


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
          f" | representation_field: {model.capability_report.representation_field}")

    state = {"images": fixture_image(),
             "texts": "a red circle drawn on a checkered background"}
    questions = ["Does the caption match the image?", "Which caption fits best?"]
    options = [["yes", "no"],
               ["a red circle on a checkered background",
                "a blue square on a plain background"]]

    with torch.no_grad():
        out = model.choice(state, questions, options=options)

    print("\n[choice] decisions from image + text (one joint forward)")
    print(f"  state_representation: {tuple(out.state_representation.shape)} "
          "(projected text-side embedding)")
    for q, opts, dec in zip(questions, options, out.decisions):
        probs = [round(float(p), 4) for p in dec.probabilities[0, : len(opts)]]
        print(f"  q={q!r}")
        print(f"    probs={dict(zip(opts, probs))}")
        print(f"    predicted={dec.predicted_value!r} confidence={float(dec.confidence[0]):.4f}")

    # capability metadata documents what actually ran for this batch.
    meta = out.capability_metadata
    print(f"\n  capability_metadata: adapter={meta['adapter']} "
          f"representation_field={meta['representation_field']} "
          f"backbone_forward_calls={meta['backbone_forward_calls']} "
          f"pair_ordering={meta['pair_ordering']}")

    # --- invariants: running the example IS testing -------------------------
    assert model.capability_report.vision and model.capability_report.text
    assert model.capability_report.representation_field == "text_embeds", (
        f"CLIP convention pins post-projection text_embeds; detector reported "
        f"{model.capability_report.representation_field!r}"
    )
    assert model.adapter.name == "multimodal"
    assert len(out.decisions) == len(questions)
    assert meta["backbone_forward_calls"] == 1, "both towers in one forward"
    for opts, dec in zip(options, out.decisions):
        probs = dec.probabilities[0]
        k = len(opts)
        assert torch.all(probs >= 0) and torch.all(probs <= 1)
        assert abs(float(probs[:k].sum()) - 1.0) < 1e-6
        assert 0 <= int(dec.predicted_index[0]) < k
        assert dec.predicted_value in opts
        assert 0.0 <= float(dec.confidence[0]) <= 1.0
    assert bool(torch.isfinite(out.state_representation).all())
    # noul head over the same multimodal state: probability strictly in (0,1).
    with torch.no_grad():
        noul_out = model.noul(state, ["Is the image bright?"])
    p_yes = float(noul_out.decisions[0].probabilities[0, 0])
    assert 0.0 < p_yes < 1.0, f"noul p_yes={p_yes} outside (0, 1)"
    print(f"  [noul] Is the image bright? -> P(yes)={p_yes:.4f} "
          f"confidence={float(noul_out.decisions[0].confidence[0]):.4f}")
    print("\nAll invariants held (multimodal text_embeds convention, one forward, "
          "probs sum to 1, schema membership, noul in (0,1)).")


if __name__ == "__main__":
    main()
