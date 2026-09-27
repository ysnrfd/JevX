# Copyright (c) 2026 ysnrfd
"""Example 03 — encoder-decoder (T5): decisions from encoder states.

What this shows
---------------
Wrapping an encoder-decoder checkpoint (``hf-internal-testing/tiny-random-t5``,
encoder_decoder family). JevX extracts the ENCODER side
(``encoder_last_hidden_state`` via a direct ``model.get_encoder()`` call —
probe-verified equal to the full model's field in eval mode, audit Q6) and
applies the documented ``encoder_pool`` = masked-mean convention
(compatibility-matrix "encoder" rows, sentence-T5 convention).

Demonstrates: batched states (B=2) x shared questions (m=2) with the
state-major decisions layout (entry ``b*m + j``), plus score/noul typed heads.

Honesty note: ``tiny-random-t5`` carries RANDOM weights (pipeline fixture).
Printed numbers are real outputs with asserted invariants, not quality.

Run from the repo root:  python examples/example_03_encoder_decoder_t5.py
Expected runtime: ~5-10 s (first run downloads the checkpoint).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from jevx import JevX

MODEL_ID = "hf-internal-testing/tiny-random-t5"

STATES = [
    "The maintenance crew inspected the bridge deck and confirmed that the "
    "support damage had been repaired before the storm arrived.",
    "Analysts expect the quarterly report to show steady revenue growth across all regions.",
]
QUESTIONS = ["Is this content safe to deploy?", "Which disposition fits this item?"]
OPTIONS = [["safe", "unsafe"], ["approve", "review", "reject"]]


def main() -> None:
    model = JevX.from_pretrained(MODEL_ID)
    model.eval()
    print(f"model: {MODEL_ID}")
    print(f"adapter: {model.adapter.name} | conversion: {model.capability_report.jevx_conversion}"
          f" | representation_field: {model.capability_report.representation_field}")

    # Batch of B=2 states, m=2 shared questions -> B*m = 4 decisions.
    with torch.no_grad():
        out = model.choice(STATES, QUESTIONS, options=OPTIONS)

    print(f"\n[choice] B={len(STATES)} states x m={len(QUESTIONS)} questions "
          f"-> {len(out.decisions)} decisions (state-major b*m + j)")
    for b, state in enumerate(STATES):
        print(f"  state[{b}]={state[:56]!r}...")
        for j, (q, opts) in enumerate(zip(QUESTIONS, OPTIONS)):
            dec = out.decisions[b * len(QUESTIONS) + j]
            probs = [round(float(p), 4) for p in dec.probabilities[0, : len(opts)]]
            print(f"    q={q!r} probs={dict(zip(opts, probs))} "
                  f"predicted={dec.predicted_value!r} "
                  f"confidence={float(dec.confidence[0]):.4f}")

    # --- invariants: running the example IS testing -------------------------
    B, m = len(STATES), len(QUESTIONS)
    assert len(out.decisions) == B * m, "flat list must hold B*m decisions"
    assert out.state_representation.shape[0] == B, (
        f"state_representation batch must be {B}, got {tuple(out.state_representation.shape)}"
    )
    for b in range(B):
        for j, (q, opts) in enumerate(zip(QUESTIONS, OPTIONS)):
            dec = out.decisions[b * m + j]
            assert dec.schema_metadata["option_keys"] == OPTIONS[j], (
                f"state-major layout violated at flat index {b * m + j} "
                f"(state {b}, question {j})"
            )
            probs = dec.probabilities[0]
            k = len(opts)
            assert torch.all(probs >= 0) and torch.all(probs <= 1)
            assert abs(float(probs[:k].sum()) - 1.0) < 1e-6
            assert 0 <= int(dec.predicted_index[0]) < k
            assert dec.predicted_value in opts
            assert 0.0 <= float(dec.confidence[0]) <= 1.0
    print("\nAll invariants held (B*m layout verified against schema_metadata, "
          "probs sum to 1, index in schema, confidence in [0,1]).")


if __name__ == "__main__":
    main()
