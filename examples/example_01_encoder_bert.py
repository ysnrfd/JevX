# Copyright (c) 2026 ysnrfd
"""Example 01 — encoder-only BERT: typed choice/score/noul decisions.

What this shows
---------------
Wrapping a bidirectional encoder checkpoint (``hf-internal-testing/tiny-random-bert``,
encoder family, native ``pooler_output``) and answering three kinds of typed
questions about ONE state in a single shared-state call:

* ``choice``  — categorical pick over a schema (variable K in one call);
* ``score``   — ordinal level distribution + expected score;
* ``noul``    — binary yes/no probability.

Honesty note: ``tiny-random-bert`` carries RANDOM weights (it is a pipeline
fixture, not a trained model — see reports/REAL_MODEL_VALIDATION.md). The
printed numbers are real pipeline outputs with their invariants asserted, but
they are NOT predictive quality.

Run from the repo root:  python examples/example_01_encoder_bert.py
Expected runtime on this hardware class: ~5-10 s (first run downloads the
checkpoint, ~3.2 s measured load afterwards).
"""

from __future__ import annotations

import sys
from pathlib import Path

# No packaging file exists yet (documented gap in README.md): the repo root is
# put on sys.path so `import jevx` works when running this file directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from jevx import JevX

MODEL_ID = "hf-internal-testing/tiny-random-bert"


def main() -> None:
    model = JevX.from_pretrained(MODEL_ID)
    model.eval()
    print(f"model: {MODEL_ID}")
    print(f"adapter: {model.adapter.name} | conversion: {model.capability_report.jevx_conversion}"
          f" | representation_field: {model.capability_report.representation_field}")

    state = (
        "The maintenance crew inspected the bridge deck and confirmed that "
        "the support damage had been repaired before the storm arrived."
    )
    questions = [
        "Is this content safe to deploy?",
        "Which disposition fits this item?",
        "How severe is the situation?",
    ]
    # Schema per question: K=2, K=3, K=4 (variable K in ONE call).
    options = [
        ["safe", "unsafe"],
        ["approve", "review", "reject"],
        ["1", "2", "3", "4"],
    ]

    # --- typed decisions in one shared-state call ---------------------------
    with torch.no_grad():
        choice_out = model.choice(state, questions[:2], options=options[:2])
        score_out = model.score(state, questions[2:], options=options[2:])
        noul_out = model.noul(state, ["Does this text mention a guarantee?"])

    # --- print real outputs -------------------------------------------------
    print("\n[choice] typed categorical decisions")
    for q, opts, dec in zip(questions[:2], options[:2], choice_out.decisions):
        probs = [round(float(p), 4) for p in dec.probabilities[0, : len(opts)]]
        print(f"  q={q!r}")
        print(f"    probs={dict(zip(opts, probs))}")
        print(f"    predicted={dec.predicted_value!r} confidence={float(dec.confidence[0]):.4f}")

    dec = score_out.decisions[0]
    probs = [round(float(p), 4) for p in dec.probabilities[0]]
    print("\n[score] ordinal distribution over levels 1..4")
    print(f"  probs={dict(zip(options[2], probs))}")
    print(f"  expected_score={float(dec.expected_score[0]):.4f} "
          f"predicted_level={dec.predicted_value!r} "
          f"confidence={float(dec.confidence[0]):.4f}")

    dec = noul_out.decisions[0]
    print("\n[noul] binary decision")
    print(f"  P(yes)={float(dec.probabilities[0, 0]):.4f} "
          f"predicted={int(dec.predicted_index[0])} "
          f"confidence={float(dec.confidence[0]):.4f}")

    # --- invariants: running the example IS testing -------------------------
    assert model.capability_report.jevx_conversion == "supported"
    # Shared-state layout: m questions about ONE state -> m decisions.
    assert len(choice_out.decisions) == 2
    assert len(score_out.decisions) == 1
    assert len(noul_out.decisions) == 1
    for opts, dec in zip(options[:2], choice_out.decisions):
        probs = dec.probabilities[0]
        k = len(opts)
        assert torch.all(probs >= 0) and torch.all(probs <= 1)
        assert abs(float(probs[:k].sum()) - 1.0) < 1e-6, "choice rows must sum to 1"
        idx = int(dec.predicted_index[0])
        assert 0 <= idx < k
        assert dec.predicted_value in opts, "projected value must be in schema C"
        assert 0.0 <= float(dec.confidence[0]) <= 1.0
    score_probs = score_out.decisions[0].probabilities[0]
    assert abs(float(score_probs.sum()) - 1.0) < 1e-6
    es = float(score_out.decisions[0].expected_score[0])
    assert 0.0 <= es <= 3.0 + 1e-6, "expected score bounded by [0, K-1]"
    p_yes = float(noul_out.decisions[0].probabilities[0, 0])
    assert 0.0 < p_yes < 1.0, "noul probability strictly inside (0, 1)"
    print("\nAll invariants held (probs sum to 1, index in schema, "
          "expected_score bounded, noul in (0,1), confidence in [0,1]).")


if __name__ == "__main__":
    main()
