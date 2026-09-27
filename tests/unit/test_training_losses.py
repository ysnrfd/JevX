# Copyright (c) 2026 ysnrfd
"""Unit tests for jevx.training.losses: masked CE vs hand computation,
target validation, extreme-logit finiteness, hybrid weight scaling,
gradient flow, and a seeded random property loop."""

from __future__ import annotations

import math
import random

import pytest
import torch

from jevx.training import (
    choice_loss,
    multilabel_loss,
    noul_loss,
    score_loss,
)


# ---------------------------------------------------------------------------
# Masked cross-entropy vs a manual computation
# ---------------------------------------------------------------------------


def test_choice_loss_masked_matches_manual():
    """Masked CE must equal -log softmax over VALID options only."""
    torch.manual_seed(0)
    logits = torch.tensor(
        [[0.5, -1.2, 3.0], [2.0, 0.25, -0.75]], dtype=torch.float64
    )
    target = torch.tensor([2, 0])
    mask = torch.tensor([[True, False, True], [True, True, False]])

    # Manual: per row, -log p[target] with p a softmax over VALID options only:
    #   -log( exp(z_t) / sum_{valid} exp(z) ) = logsumexp(valid) - z_t
    manual_terms = []
    for i in range(logits.shape[0]):
        valid = logits[i][mask[i]]
        position_in_valid = int((torch.arange(3)[mask[i]] == target[i]).nonzero())
        manual_terms.append(
            torch.logsumexp(valid, dim=0) - valid[position_in_valid]
        )
    manual = torch.stack(manual_terms).mean()

    got = choice_loss(logits, target, mask)
    assert math.isclose(float(got), float(manual), rel_tol=0, abs_tol=1e-12)


def test_choice_loss_ignores_invalid_columns():
    """A masked-out column's logit value must not change the loss."""
    target = torch.tensor([0, 1])
    mask = torch.tensor([[True, True, False], [True, True, True]])
    base = torch.tensor([[1.0, 2.0, 0.0], [0.5, -0.5, 2.0]], dtype=torch.float64)
    changed = base.clone()
    changed[0, 2] = 999.0  # invalid column: arbitrary junk
    assert math.isclose(
        float(choice_loss(base, target, mask)),
        float(choice_loss(changed, target, mask)),
        rel_tol=0,
        abs_tol=1e-12,
    )


def test_choice_loss_unmasked_equals_cross_entropy():
    torch.manual_seed(1)
    logits = torch.randn(8, 5, dtype=torch.float64)
    target = torch.randint(0, 5, (8,))
    expected = torch.nn.functional.cross_entropy(logits, target)
    assert math.isclose(
        float(choice_loss(logits, target)), float(expected), rel_tol=0, abs_tol=1e-12
    )


# ---------------------------------------------------------------------------
# Target validation
# ---------------------------------------------------------------------------


def test_choice_loss_target_out_of_range_raises():
    logits = torch.randn(4, 3)
    with pytest.raises(ValueError, match=r"target.*\[0, 3\)"):
        choice_loss(logits, torch.tensor([0, 1, 3, 2]))  # 3 >= K
    with pytest.raises(ValueError, match=r"target"):
        choice_loss(logits, torch.tensor([0, 1, -1, 2]))  # negative


def test_choice_loss_target_masked_position_raises():
    logits = torch.randn(2, 3)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    with pytest.raises(ValueError, match=r"invalid"):
        choice_loss(logits, torch.tensor([2, 0]), mask)  # row 0 option 2 masked


def test_choice_loss_non_integral_float_target_raises():
    logits = torch.randn(2, 3)
    with pytest.raises(ValueError, match=r"non-integral"):
        choice_loss(logits, torch.tensor([1.0, 1.5]))


def test_choice_loss_integral_float_target_accepted():
    """JSON labels arrive as floats (1.0) — exactly integral ones are fine."""
    logits = torch.randn(4, 3, dtype=torch.float64)
    target_int = torch.tensor([0, 1, 2, 1])
    target_float = torch.tensor([0.0, 1.0, 2.0, 1.0])
    assert math.isclose(
        float(choice_loss(logits, target_int)),
        float(choice_loss(logits, target_float)),
        rel_tol=0,
        abs_tol=0.0,
    )


def test_choice_loss_all_masked_row_raises():
    logits = torch.randn(2, 3)
    mask = torch.tensor([[False, False, False], [True, True, True]])
    with pytest.raises(ValueError, match=r"no valid option"):
        choice_loss(logits, torch.tensor([0, 0]), mask)


def test_choice_loss_unknown_reduction_raises():
    logits = torch.randn(2, 3)
    with pytest.raises(ValueError, match=r"reduction"):
        choice_loss(logits, torch.tensor([0, 1]), reduction="batchmean")


def test_noul_loss_non_binary_target_raises():
    with pytest.raises(ValueError, match=r"\{0, 1\}"):
        noul_loss(torch.randn(4), torch.tensor([0, 1, 2, 1]))


def test_score_loss_unknown_mode_raises():
    logits = torch.randn(4, 3)
    with pytest.raises(ValueError, match=r"mode"):
        score_loss(logits, torch.tensor([0, 1, 2, 1]), mode="ranking")


def test_score_loss_expectation_target_out_of_range_raises():
    logits = torch.randn(4, 3)
    with pytest.raises(ValueError, match=r"\[0, 2\]"):
        score_loss(logits, torch.tensor([0.0, 1.0, 2.0, 3.5]), mode="expectation")


def test_multilabel_loss_non_binary_target_raises():
    with pytest.raises(ValueError, match=r"\{0, 1\}"):
        multilabel_loss(torch.randn(2, 3), torch.tensor([[0.0, 1.0, 2.0]] * 2))


# ---------------------------------------------------------------------------
# Extreme logits stay finite
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_noul_loss_finite_at_extreme_logits(sign):
    logits = torch.full((6,), sign * 1e4, requires_grad=True)
    target = torch.tensor([0, 1, 0, 1, 1, 0])
    loss = noul_loss(logits, target)
    assert torch.isfinite(loss)
    grad = torch.autograd.grad(loss, logits)[0]
    assert torch.isfinite(grad).all()


@pytest.mark.parametrize("mode", ["classification", "expectation", "hybrid"])
def test_score_loss_finite_at_extreme_logits(mode):
    torch.manual_seed(3)
    logits = torch.empty(5, 4)
    logits.bernoulli_(0.5).mul_(2e4).sub_(1e4)  # entries +-1e4
    logits.requires_grad_(True)
    target = torch.randint(0, 4, (5,))
    loss = score_loss(logits, target, mode=mode)
    assert torch.isfinite(loss)
    grad = torch.autograd.grad(loss, logits)[0]
    assert torch.isfinite(grad).all()


def test_choice_loss_finite_at_extreme_logits():
    logits = torch.tensor([[1e4, -1e4, 5.0], [-1e4, 1e4, -5.0]], requires_grad=True)
    loss = choice_loss(logits, torch.tensor([0, 1]))
    assert torch.isfinite(loss)


def test_multilabel_loss_finite_at_extreme_logits():
    logits = torch.full((4, 3), 1e4, requires_grad=True)
    target = torch.ones(4, 3, dtype=torch.float32)
    loss = multilabel_loss(logits, target)
    assert torch.isfinite(loss)
    grad = torch.autograd.grad(loss, logits)[0]
    assert torch.isfinite(grad).all()


# ---------------------------------------------------------------------------
# Hybrid weight scaling
# ---------------------------------------------------------------------------


def test_hybrid_equals_classification_plus_weight_times_expectation():
    torch.manual_seed(4)
    logits = torch.randn(16, 5, dtype=torch.float64)
    target = torch.randint(0, 5, (16,))
    for weight in (0.0, 0.5, 1.0, 3.7):
        ce = score_loss(logits, target, mode="classification")
        exp = score_loss(logits, target, mode="expectation")
        hybrid = score_loss(
            logits, target, mode="hybrid", expectation_weight=weight
        )
        assert math.isclose(
            float(hybrid),
            float(ce) + weight * float(exp),
            rel_tol=1e-9,
            abs_tol=1e-9,
        ), f"weight={weight}"


def test_expectation_weight_validated():
    logits = torch.randn(4, 3)
    target = torch.tensor([0.0, 1.0, 2.0, 1.0])
    with pytest.raises(ValueError, match=r"expectation_weight"):
        score_loss(logits, target, mode="hybrid", expectation_weight=-1.0)


def test_expectation_mode_matches_manual_mse():
    torch.manual_seed(5)
    logits = torch.randn(10, 4, dtype=torch.float64)
    target = torch.rand(10) * 3  # [0, 3]
    probs = torch.softmax(logits, dim=-1)
    levels = torch.arange(4, dtype=torch.float64)
    expected = ((probs * levels).sum(-1) - target).pow(2).mean()
    got = score_loss(logits, target, mode="expectation")
    assert math.isclose(float(got), float(expected), rel_tol=1e-12, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_choice_loss_has_nonzero_grad():
    torch.manual_seed(6)
    logits = torch.randn(8, 4, requires_grad=True)
    loss = choice_loss(logits, torch.randint(0, 4, (8,)))
    grad = torch.autograd.grad(loss, logits)[0]
    assert grad.abs().sum() > 0


@pytest.mark.parametrize("mode", ["classification", "expectation", "hybrid"])
def test_score_loss_has_nonzero_grad(mode):
    torch.manual_seed(7)
    logits = torch.randn(8, 4, requires_grad=True)
    loss = score_loss(logits, torch.randint(0, 4, (8,)), mode=mode)
    grad = torch.autograd.grad(loss, logits)[0]
    assert grad.abs().sum() > 0, mode


def test_noul_loss_has_nonzero_grad():
    torch.manual_seed(8)
    logits = torch.randn(16, requires_grad=True)
    loss = noul_loss(logits, torch.randint(0, 2, (16,)))
    grad = torch.autograd.grad(loss, logits)[0]
    assert grad.abs().sum() > 0


def test_multilabel_loss_has_nonzero_grad():
    torch.manual_seed(9)
    logits = torch.randn(8, 5, requires_grad=True)
    loss = multilabel_loss(logits, torch.randint(0, 2, (8, 5)).float())
    grad = torch.autograd.grad(loss, logits)[0]
    assert grad.abs().sum() > 0


def test_masked_positions_get_zero_grad():
    """Invalid columns must be excluded from the gradient entirely."""
    torch.manual_seed(10)
    logits = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[True, True, False]] * 4)
    loss = choice_loss(logits, torch.tensor([0, 1, 0, 1]), mask)
    grad = torch.autograd.grad(loss, logits)[0]
    assert grad[:, 2].abs().sum() == 0
    assert grad[:, :2].abs().sum() > 0


# ---------------------------------------------------------------------------
# Seeded random property loop
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(10))
def test_random_property_loop(seed):
    """Random batches: losses match independent manual formulas."""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    b, k = rng.randint(1, 12), rng.randint(2, 9)

    logits = torch.randn(b, k, dtype=torch.float64)
    # Random mask with at least one valid option per row.
    mask = torch.rand(b, k) < 0.7
    mask[:, 0] = True

    # Valid targets only: pick a random valid column per row.
    target = torch.tensor(
        [
            int(torch.arange(k)[mask[i]][torch.randint(0, int(mask[i].sum()), (1,))])
            for i in range(b)
        ],
        dtype=torch.long,
    )

    # 1) choice_loss == logsumexp(valid) - z_target (float64).
    manual = torch.stack(
        [
            torch.logsumexp(logits[i][mask[i]], dim=0)
            - logits[i][mask[i]][
                torch.nonzero(mask[i]).squeeze(-1).tolist().index(int(target[i]))
            ]
            for i in range(b)
        ]
    ).mean()
    got = choice_loss(logits, target, mask)
    assert math.isclose(float(got), float(manual), rel_tol=0, abs_tol=1e-9)

    # 2) score classification == choice_loss.
    assert math.isclose(
        float(score_loss(logits, target, mask, mode="classification")),
        float(got),
        rel_tol=0,
        abs_tol=0.0,
    )

    # 3) expectation MSE against a manual masked-softmax expectation.
    filled = logits.masked_fill(~mask, torch.finfo(torch.float64).min)
    probs = torch.softmax(filled, dim=-1)
    levels = torch.arange(k, dtype=torch.float64)
    scores = torch.rand(b, dtype=torch.float64) * (k - 1)
    manual_mse = ((probs * levels).sum(-1) - scores).pow(2).mean()
    got_mse = score_loss(logits, scores, mask, mode="expectation")
    assert math.isclose(float(got_mse), float(manual_mse), rel_tol=1e-9,
                        abs_tol=1e-9)

    # 4) noul == BCE-with-logits computed via softplus identity.
    noul_logits = torch.randn(b, dtype=torch.float64, requires_grad=True)
    noul_target = torch.randint(0, 2, (b,))
    manual_bce = (
        F_softplus(noul_logits) - noul_logits * noul_target.double()
    ).mean()
    got_bce = noul_loss(noul_logits, noul_target).detach()
    assert math.isclose(float(got_bce), float(manual_bce.detach()), rel_tol=1e-12,
                        abs_tol=1e-12)

    # 5) reduction="none" has length B and mean equals reduction="mean".
    per_sample = choice_loss(logits, target, mask, reduction="none")
    assert per_sample.shape == (b,)
    assert math.isclose(
        float(per_sample.mean()), float(got), rel_tol=0, abs_tol=1e-12
    )


def F_softplus(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable softplus: log(1 + exp(x)) via the max trick."""
    return torch.maximum(x, torch.zeros_like(x)) + torch.log1p(
        torch.exp(-torch.abs(x))
    )
