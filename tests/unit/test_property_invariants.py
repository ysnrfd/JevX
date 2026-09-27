# Copyright (c) 2026 ysnrfd
"""Property-based invariants over seeded random inputs: probability simplex,
score bounds, noul range, per-question isolation and state_dict round trips.

All randomness is seeded (``torch.manual_seed`` per trial) so failures are
reproducible.
"""

from __future__ import annotations

import io

import pytest
import torch

from jevx.decisions import (
    ChoiceHead,
    NoulHead,
    ScoreHead,
    build_decision_head,
    mask_from_num_options,
)

TRIALS = 60


def _rand_shape(rng_bound_low: int, high: int) -> int:
    return int(torch.randint(rng_bound_low, high, (1,)))


@pytest.mark.parametrize("trial", range(TRIALS))
def test_choice_probability_invariants(trial):
    torch.manual_seed(1000 + trial)
    b, k, d = _rand_shape(1, 9), _rand_shape(1, 13), _rand_shape(4, 17)
    interaction = ("dot", "bilinear", "mlp")[trial % 3]
    head = ChoiceHead(d, interaction=interaction)
    fused = torch.randn(b, d)
    options = torch.randn(b, k, d)

    # half the trials exercise a variable-K mask with contiguous valid prefixes
    if trial % 2 == 0:
        counts = torch.randint(1, k + 1, (b,))
        mask = mask_from_num_options(counts, k)
        out = head(fused, options, option_mask=mask)
    else:
        counts = torch.full((b,), k)
        mask = None
        out = head(fused, options)

    probs = out.probabilities
    assert torch.isfinite(probs).all()
    assert (probs >= 0.0).all(), "probabilities must be non-negative"
    assert torch.allclose(
        probs.sum(dim=-1), torch.ones(b), atol=1e-5
    ), "each row must sum to 1"
    if mask is not None:
        assert (probs[~mask] == 0.0).all()
        rows = torch.arange(b)
        assert mask[rows, out.predicted_index].all(), "argmax must be valid"
    else:
        assert (out.predicted_index >= 0).all()
        assert (out.predicted_index < k).all()


@pytest.mark.parametrize("trial", range(TRIALS))
def test_score_expected_value_in_bounds(trial):
    torch.manual_seed(2000 + trial)
    b, d = _rand_shape(1, 9), _rand_shape(4, 17)
    if trial % 2 == 0:
        k = _rand_shape(1, 13)
        head = ScoreHead(d, num_levels=k)
        fused = torch.randn(b, d) * _rand_shape(1, 20)
        out = head(fused)
    else:
        k = _rand_shape(1, 13)
        head = ScoreHead(d, interaction="bilinear" if trial % 4 == 1 else "dot")
        fused = torch.randn(b, d)
        options = torch.randn(b, k, d)
        out = head(fused, options)

    s = out.expected_score
    assert torch.isfinite(s).all()
    assert (s >= 0.0 - 1e-5).all(), f"score below 0: {s.min()}"
    assert (s <= (k - 1) + 1e-5).all(), f"score above K-1: {s.max()}"
    assert (out.predicted_index >= 0).all() and (out.predicted_index < k).all()
    assert torch.allclose(
        out.probabilities.sum(dim=-1), torch.ones(b), atol=1e-5
    )


@pytest.mark.parametrize("trial", range(TRIALS))
def test_noul_probability_range(trial):
    torch.manual_seed(3000 + trial)
    b, d = _rand_shape(1, 9), _rand_shape(4, 17)
    head = NoulHead(d)
    out = head(torch.randn(b, d))
    p = out.probabilities
    assert torch.isfinite(p).all()
    # moderate-scale inputs cannot saturate fp32 sigmoid
    assert (p > 0.0).all(), "p_yes must be strictly positive"
    assert (p < 1.0).all(), "p_yes must be strictly below 1"
    assert set(out.predicted_index.unique().tolist()) <= {0, 1}
    assert out.logits.shape == (b, 1)


@pytest.mark.parametrize("trial", range(40))
def test_batch_rows_are_isolated(trial):
    """Perturbing other rows / masked-out entries must not change a row."""
    torch.manual_seed(4000 + trial)
    b, k, d = 4, 6, 8
    interaction = ("dot", "bilinear", "mlp")[trial % 3]
    head = ChoiceHead(d, interaction=interaction)
    head.eval()  # no dropout in these heads, but be explicit

    fused = torch.randn(b, d)
    options = torch.randn(b, k, d)
    counts = torch.tensor([3, 5, 1, 6])
    mask = mask_from_num_options(counts, k)
    baseline = head(fused, options, option_mask=mask)

    # (a) Masked-out entries carry NO information: perturb every invalid
    # option embedding to a huge value — all rows' probabilities, valid
    # logits and predictions must be bit-identical.
    options_masked = options.clone()
    options_masked[~mask] = 1000.0
    after_mask_perturb = head(fused, options_masked, option_mask=mask)
    assert torch.equal(
        baseline.probabilities, after_mask_perturb.probabilities
    ), "masked-out entries must have zero influence on probabilities"
    assert torch.equal(
        baseline.logits.masked_fill(~mask, 0.0),
        after_mask_perturb.logits.masked_fill(~mask, 0.0),
    ), "valid logits must ignore masked-out option features"
    assert torch.equal(
        baseline.predicted_index, after_mask_perturb.predicted_index
    )

    # (b) Rows are independent: perturb ONLY rows 1..3 of the fused state —
    # row 0's full output must be bit-identical (no cross-question math).
    fused_rows = fused.clone()
    fused_rows[1:] += 7.5
    after_row_perturb = head(fused_rows, options, option_mask=mask)
    assert torch.equal(baseline.logits[0], after_row_perturb.logits[0])
    assert torch.equal(
        baseline.probabilities[0], after_row_perturb.probabilities[0]
    )
    assert torch.equal(
        baseline.predicted_index[0], after_row_perturb.predicted_index[0]
    )


_SERIALIZATION_CASES = [
    ("choice", {"hidden_size": 6, "interaction": "dot"}),
    ("choice", {"hidden_size": 6, "interaction": "bilinear"}),
    ("choice", {"hidden_size": 6, "interaction": "mlp"}),
    ("score", {"hidden_size": 6, "num_levels": 4}),
    ("score", {"hidden_size": 6, "interaction": "dot"}),
    ("noul", {"hidden_size": 6}),
]


@pytest.mark.parametrize("name,kwargs", _SERIALIZATION_CASES)
def test_head_state_dict_round_trip_reproduces_outputs(name, kwargs):
    d = kwargs["hidden_size"]
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        head = build_decision_head(name, **kwargs)
        torch.manual_seed(seed + 500)
        fused = torch.randn(5, d)
        options = (
            torch.randn(5, 4, d) if name in ("choice", "score") else None
        )
        out = head(fused, options)
        reference_logits = out.logits.detach().clone()
        reference_probs = out.probabilities.detach().clone()
        reference_index = out.predicted_index.detach().clone()
        reference_extra = (
            out.expected_score.detach().clone() if out.expected_score is not None else None
        )

        # real serialization through torch.save / torch.load
        buffer = io.BytesIO()
        torch.save(head.state_dict(), buffer)
        buffer.seek(0)
        state = torch.load(buffer, weights_only=True)

        torch.manual_seed(seed + 999)  # fresh head with DIFFERENT random init
        restored = build_decision_head(name, **kwargs)
        restored.load_state_dict(state)

        out2 = restored(fused, options)
        assert torch.allclose(out2.logits, reference_logits, atol=1e-6)
        assert torch.allclose(out2.probabilities, reference_probs, atol=1e-6)
        assert torch.equal(out2.predicted_index, reference_index)
        if reference_extra is not None:
            assert torch.allclose(out2.expected_score, reference_extra, atol=1e-6)


def test_head_forward_is_deterministic_for_fixed_inputs():
    torch.manual_seed(0)
    head = ChoiceHead(8, interaction="bilinear")
    fused = torch.randn(4, 8)
    options = torch.randn(4, 5, 8)
    a = head(fused, options)
    b = head(fused, options)
    assert torch.equal(a.logits, b.logits)
    assert torch.equal(a.probabilities, b.probabilities)
