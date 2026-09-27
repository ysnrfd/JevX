# Copyright (c) 2026 ysnrfd
"""Example 04 — semantic embedding backbone (sentence-transformers MiniLM).

What this shows
---------------
``sentence-transformers/all-MiniLM-L6-v2`` is the ONE real trained checkpoint
in the validated matrix (reports/REAL_MODEL_VALIDATION.md): a bert-architecture
sentence-embedding model whose pooling convention is pinned via
``pooling_method="masked_mean"`` (the sentence-transformers convention) —
the same configuration the compatibility matrix and Phase-92 validation used.

Demonstrates: taking typed decisions where the state representation is a
trained sentence space, and inspecting ``JevXOutput.state_representation``
(the fused H_S projection, [B, D]) directly.

Validation status (scoped exactly as reports/REAL_MODEL_VALIDATION.md scopes
it): FUNCTIONALLY validated on this checkpoint (load, invariants, save/load
equality, extraction correctness — all error 0.0); NOT quality-validated (no
similarity-task metrics were run; benchmark quality numbers belong to
reports/DATASET_BENCHMARKS.md and cover a different backbone).

Run from the repo root:  python examples/example_04_semantic_embedding.py
Expected runtime: ~5-15 s (27.6M params; first run downloads the checkpoint,
measured load 3.86 s — reports/REAL_MODEL_VALIDATION.md).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from jevx import JevX

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

STATES = [
    "The bridge inspection crew repaired the support damage before the storm.",
    "Quarterly revenue grew steadily across every reported region.",
]
QUESTIONS = ["Which topic does this sentence belong to?",
             "Is the sentiment of this sentence positive?"]
OPTIONS = [["infrastructure", "finance"], ["yes", "no"]]


def main() -> None:
    # pooling_method pins the sentence-transformers masked_mean convention.
    model = JevX.from_pretrained(MODEL_ID, pooling_method="masked_mean")
    model.eval()
    print(f"model: {MODEL_ID} (real trained embedding checkpoint)")
    print(f"pooling_method={model.config.pooling_method} | adapter: {model.adapter.name}"
          f" | conversion: {model.capability_report.jevx_conversion}")

    with torch.no_grad():
        out = model.choice(STATES, QUESTIONS, options=OPTIONS)

    print(f"\nstate_representation: {tuple(out.state_representation.shape)} "
          f"(B={len(STATES)} projected H_S vectors)")
    for b, state in enumerate(STATES):
        print(f"  state[{b}]={state[:60]!r}...")
        for j, (q, opts) in enumerate(zip(QUESTIONS, OPTIONS)):
            dec = out.decisions[b * len(QUESTIONS) + j]
            probs = [round(float(p), 4) for p in dec.probabilities[0, : len(opts)]]
            print(f"    q={q!r} probs={dict(zip(opts, probs))} "
                  f"predicted={dec.predicted_value!r} "
                  f"confidence={float(dec.confidence[0]):.4f}")

    # --- invariants: running the example IS testing -------------------------
    B, m = len(STATES), len(QUESTIONS)
    assert model.config.pooling_method == "masked_mean", "convention must be pinned in config"
    # The detector's STATIC preferred field for this checkpoint is
    # pooler_output (the BertModel base class ships a pooler — observed on
    # this id, matches vendor/reports/_jevx_matrix_results.json). Pinning
    # pooling_method="masked_mean" makes extraction skip the native pooler
    # (adapters/base.py: native_allowed = strategy in (None, "native")) and
    # pool the sequence field instead — so the ACTUAL representation field
    # for this call is last_hidden_state, per the sentence-transformers
    # convention the matrix row pins.
    assert model.capability_report.representation_field == "pooler_output", (
        f"detector's preferred native field for MiniLM is pooler_output; "
        f"reported {model.capability_report.representation_field!r}"
    )
    assert out.capability_metadata["representation_field"] == "last_hidden_state", (
        f"pinned masked_mean extraction must pool last_hidden_state; "
        f"got {out.capability_metadata['representation_field']!r}"
    )
    assert len(out.decisions) == B * m
    assert out.state_representation.shape == (B, model.config.decision_hidden_size), (
        f"state_representation must be [B, D]={(B, model.config.decision_hidden_size)}, "
        f"got {tuple(out.state_representation.shape)}"
    )
    assert bool(torch.isfinite(out.state_representation).all()), "H_S must be finite"
    for dec, opts in zip(out.decisions, [o for _ in range(B) for o in OPTIONS]):
        probs = dec.probabilities[0]
        k = len(opts)
        assert torch.all(probs >= 0) and torch.all(probs <= 1)
        assert abs(float(probs[:k].sum()) - 1.0) < 1e-6
        assert dec.predicted_value in opts
        assert 0.0 <= float(dec.confidence[0]) <= 1.0
    # Determinism: identical weights + eval mode => identical probabilities.
    with torch.no_grad():
        again = model.choice(STATES, QUESTIONS, options=OPTIONS)
    for a, b_ in zip(out.decisions, again.decisions):
        assert torch.equal(a.probabilities, b_.probabilities), (
            "within-process eval determinism violated (contract: bitwise equal)"
        )
    print("\nAll invariants held (config convention pinned, [B, D] state repr finite, "
          "probs sum to 1, schema membership, bitwise eval determinism).")


if __name__ == "__main__":
    main()
