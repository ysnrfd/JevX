# Copyright (c) 2026 ysnrfd
"""Example 02 — decoder-only LLM: decisions over a causal backbone.

What this shows
---------------
Wrapping a causal LM (``sshleifer/tiny-gpt2``, decoder family) and taking
typed decisions about a state. Decoder-only stacks ship no trained sentence
pooler (compatibility-matrix level 2, "lossy pooling" semantics), so JevX
pins the family convention: mask-aware ``last_valid`` pooling — identical for
left-, right- and unpadded batches (tests/unit/test_padding_determinism.py).

Also demonstrated: the shared-state property — m questions in ONE call run
the backbone exactly once (reports/LATENCY_BENCHMARKS.md measures the payoff:
up to 11.06x faster than m separate calls at m=32 on this hardware).

Honesty note: ``sshleifer/tiny-gpt2`` has RANDOM weights (pipeline fixture;
see reports/REAL_MODEL_VALIDATION.md). Printed numbers are real outputs, not
predictive quality.

Run from the repo root:  python examples/example_02_decoder_llm.py
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

MODEL_ID = "sshleifer/tiny-gpt2"


def main() -> None:
    model = JevX.from_pretrained(MODEL_ID)
    model.eval()
    print(f"model: {MODEL_ID}")
    print(f"adapter: {model.adapter.name} | conversion: {model.capability_report.jevx_conversion}"
          f" | default pooling: {model.adapter.default_pooling}")

    state = (
        "Analysts expect the quarterly report to show steady revenue growth "
        "across all regions, although margins may compress slightly."
    )
    questions = [
        "Is this content safe to deploy?",
        "Which disposition fits this item?",
        "Is the tone optimistic?",
        "Does this text contain a guarantee?",
    ]
    options = [
        ["safe", "unsafe"],
        ["approve", "review", "reject"],
        ["yes", "no"],
        ["yes", "no"],
    ]

    # Count backbone forward calls to demonstrate the shared-state property.
    calls = {"n": 0}
    handle = model.model.register_forward_hook(lambda *a, **k: calls.__setitem__("n", calls["n"] + 1))

    # Mixed types in ONE call: decision_type is a list with m entries; types
    # are grouped so probability spaces are never concatenated.
    with torch.no_grad():
        out = model.decide(
            state,
            questions,
            decision_type=["choice", "choice", "choice", "noul"],
            options=[options[0], options[1], options[2], None],
        )
    handle.remove()

    print("\n[decide] 4 questions, 1 backbone forward")
    for q, dec in zip(questions, out.decisions):
        dtype = dec.schema_metadata["decision_type"]
        keys = dec.schema_metadata["option_keys"]
        if keys is None:  # noul
            print(f"  [{dtype}] {q!r}")
            print(f"    P(yes)={float(dec.probabilities[0, 0]):.4f} "
                  f"predicted={int(dec.predicted_index[0])} "
                  f"confidence={float(dec.confidence[0]):.4f}")
        else:
            probs = [round(float(p), 4) for p in dec.probabilities[0, : len(keys)]]
            print(f"  [{dtype}] {q!r}")
            print(f"    probs={dict(zip(keys, probs))}")
            print(f"    predicted={dec.predicted_value!r} confidence={float(dec.confidence[0]):.4f}")

    # --- invariants: running the example IS testing -------------------------
    assert calls["n"] == 1, (
        f"shared-state property: expected exactly 1 backbone forward for 4 "
        f"questions, observed {calls['n']}"
    )
    assert len(out.decisions) == 4  # state-major flat layout b*m + j
    # Choice rows: probability simplex over K options + schema membership.
    # (The 4th decision is noul — a single sigmoid probability, NOT a
    # simplex row; it gets its own (0, 1) check below.)
    for dec, keys in zip(out.decisions[:3], options[:3]):
        k = len(keys)
        probs = dec.probabilities[0]
        assert torch.all(probs >= 0) and torch.all(probs <= 1)
        assert abs(float(probs[:k].sum()) - 1.0) < 1e-6
        assert 0 <= int(dec.predicted_index[0]) < k
        assert dec.predicted_value in keys
        assert 0.0 <= float(dec.confidence[0]) <= 1.0
    p_yes = float(out.decisions[3].probabilities[0, 0])
    assert 0.0 < p_yes < 1.0, "noul probability strictly inside (0, 1)"
    assert int(out.decisions[3].predicted_index[0]) in (0, 1)
    assert 0.0 <= float(out.decisions[3].confidence[0]) <= 1.0
    print(f"\nAll invariants held (backbone_forward_calls={calls['n']} for m=4, "
          "probs sum to 1, schema membership, confidence in [0,1]).")


if __name__ == "__main__":
    main()
