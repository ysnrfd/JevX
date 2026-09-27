# Copyright (c) 2026 ysnrfd
"""Q-002 noul confidence fix (spec Def 9(e) / Def 10 abstention).

The pre-fix engine hard-wired ``confidence = 1.0`` for any single-column
probability row, so ``confidence_threshold=`` was a silent no-op for every
noul decision (audit Q-002). The fix reads width-1 rows as the binary pair
``(p, 1 - p)``: ``confidence = max(p, 1 - p) ∈ [0.5, 1]``.

``tests/unit/test_confidence.py::test_width_one_binary_rule`` pins the
engine-level values (0.7 → 0.7, 0.4954 → 0.5046, p == 1 → 1); this file
pins the numbers from the audit probe AND the end-to-end consequence the
audit called broken: abstention now actually fires for noul decisions.
"""

from __future__ import annotations

import copy

import pytest
import torch

from jevx import JevXConfig, JevXForDecision
from jevx.confidence import ConfidenceEngine

HIDDEN = 32
STATE = "hello world"
QUESTION = "is it so?"


def _build_wrapper(backbone, tokenizer=None, **config_overrides) -> JevXForDecision:
    """Wrap ``backbone`` (deep-copied) with a tiny decision config, eval mode."""
    overrides = dict(
        decision_hidden_size=HIDDEN,
        question_hidden_size=HIDDEN,
        schema_hidden_size=HIDDEN,
    )
    overrides.update(config_overrides)
    wrapper = JevXForDecision(
        JevXConfig(**overrides),
        backbone=copy.deepcopy(backbone),
        tokenizer=tokenizer,
    )
    wrapper.eval()
    return wrapper


def test_engine_probe_p_04954_gives_05046_and_fires_at_threshold() -> None:
    """The audit probe: p = 0.4954 -> conf = max(p, 1-p) = 0.5046 (NOT 1.0),
    so a 0.6 threshold abstains where the old hard-wired 1.0 never could."""
    engine = ConfidenceEngine("max_probability")
    probs = torch.full((1, 1), 0.4954)
    conf = engine(probs)
    assert float(conf[0]) == pytest.approx(0.5046, abs=1e-6)
    # the abstention predicate (confidence < threshold) fires at 0.6
    assert float(conf[0]) < 0.6
    # genuine K=1 CHOICE rows (p == 1) still read exactly 1.0
    assert float(engine(torch.ones(1, 1))[0]) == 1.0


def test_noul_forward_confidence_is_binary_max_not_one(
    tiny_bert, tiny_tokenizer
) -> None:
    """End to end: the noul decision's stored confidence equals max(p, 1-p)
    for its own probability — the Q-002 fix on the real forward path."""
    wrapper = _build_wrapper(tiny_bert, tokenizer=tiny_tokenizer)
    with torch.no_grad():
        out = wrapper.noul(STATE, [QUESTION])
    decision = out.decisions[0]
    p = float(decision.probabilities[0, 0])
    conf = float(decision.confidence[0])
    assert 0.0 < p < 1.0  # T4 clamp keeps the sigmoid strictly interior
    assert conf == pytest.approx(max(p, 1.0 - p), abs=1e-6)
    # pre-fix this was hard-wired to exactly 1.0 for every noul decision
    assert conf < 1.0


def test_noul_abstention_threshold_actually_fires(
    tiny_bert, tiny_tokenizer
) -> None:
    """Def 10 (1.1): a threshold above the noul confidence abstains and
    clears the predicted fields; a threshold below it commits. Before the
    fix, conf == 1.0 meant the first case could never happen."""
    wrapper = _build_wrapper(tiny_bert, tokenizer=tiny_tokenizer)
    with torch.no_grad():
        committed = wrapper.noul(STATE, [QUESTION])
    decision = committed.decisions[0]
    conf = float(decision.confidence[0])
    assert decision.abstained is None  # no threshold given -> no flag

    # threshold strictly ABOVE the confidence -> must abstain
    high_threshold = min(1.0, (conf + 1.0) / 2.0)
    assert conf < high_threshold
    with torch.no_grad():
        abstained = wrapper.noul(
            STATE, [QUESTION], confidence_threshold=high_threshold
        )
    got = abstained.decisions[0]
    assert bool(got.abstained[0]) is True
    assert got.predicted_index is None
    assert got.predicted_value is None
    # probabilities are untouched by abstention (documented contract)
    assert torch.equal(got.probabilities, decision.probabilities)

    # threshold strictly BELOW the confidence -> commits
    low_threshold = conf / 2.0
    assert low_threshold < conf
    with torch.no_grad():
        kept = wrapper.noul(STATE, [QUESTION], confidence_threshold=low_threshold)
    assert bool(kept.decisions[0].abstained[0]) is False
    assert kept.decisions[0].predicted_index is not None


def test_choice_k_one_rows_still_read_confidence_one(
    tiny_bert, tiny_tokenizer
) -> None:
    """A genuine single-OPTION choice row carries p == 1, so the binary rule
    still yields confidence 1.0 there (spec 9(e) note, regression guard)."""
    wrapper = _build_wrapper(tiny_bert, tokenizer=tiny_tokenizer)
    with torch.no_grad():
        out = wrapper.choice(STATE, [QUESTION], options=[["only"]])
    decision = out.decisions[0]
    assert float(decision.probabilities[0, 0]) == pytest.approx(1.0, abs=1e-6)
    assert float(decision.confidence[0]) == pytest.approx(1.0, abs=1e-6)
    # threshold of 1.0 cannot abstain a row that reads exactly 1.0
    with torch.no_grad():
        kept = wrapper.choice(
            STATE, [QUESTION], options=[["only"]], confidence_threshold=1.0
        )
    assert bool(kept.decisions[0].abstained[0]) is False
