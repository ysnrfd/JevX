# Copyright (c) 2026 ysnrfd
"""Pooling tests: every strategy, padding invariance, grads, fp16, errors.

Covers Definition 4 of ``docs/MATHEMATICAL_SPECIFICATION.md`` against
``jevx.representations.pooling`` with real tensors (no mocks).
"""

from __future__ import annotations

import pytest
import torch

from jevx.representations.pooling import (
    POOLING_STRATEGIES,
    AttentionPooler,
    LearnerPooler,
    available_pooling_strategies,
    pool,
)

torch.manual_seed(0)

B, T, D = 4, 7, 8

#: Strategies that consume a mask when one is given (and validate non-empty
#: rows). ``mean`` is unmasked by definition; ``native`` needs pooler_output.
MASK_AWARE = (
    "cls",
    "masked_mean",
    "encoder_pool",
    "last_valid",
    "max",
    "attention_pool",
    "learned_pool",
)


@pytest.fixture()
def seq() -> torch.Tensor:
    return torch.randn(B, T, D)


@pytest.fixture()
def mask() -> torch.Tensor:
    """Random bool mask, at least one valid token per row."""
    m = torch.rand(B, T) > 0.4
    m[:, 0] = True  # guarantee non-empty rows (axiom D1.1)
    return m


def test_every_registered_strategy_returns_b_d(seq: torch.Tensor, mask: torch.Tensor) -> None:
    pooler_output = torch.randn(B, D)
    for strategy in available_pooling_strategies():
        out = pool(seq, mask, strategy, pooler_output=pooler_output)
        assert out.shape == (B, D), strategy
        assert torch.isfinite(out).all(), strategy
    # and without a mask
    for strategy in available_pooling_strategies():
        out = pool(seq, None, strategy, pooler_output=pooler_output)
        assert out.shape == (B, D), strategy


def test_registry_keys_match_contract(seq: torch.Tensor) -> None:
    expected = {
        "native",
        "cls",
        "mean",
        "masked_mean",
        "last_valid",
        "max",
        "attention_pool",
        "learned_pool",
        "encoder_pool",
    }
    assert set(POOLING_STRATEGIES) == expected


def test_native_requires_pooler_output(seq: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="pooler"):
        pool(seq, None, "native", pooler_output=None)
    pooled = torch.randn(B, D)
    assert torch.equal(pool(seq, None, "native", pooler_output=pooled), pooled)


def test_unknown_strategy_lists_registry(seq: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="available strategies"):
        pool(seq, None, "not_a_strategy")


def test_masked_mean_matches_manual(seq: torch.Tensor, mask: torch.Tensor) -> None:
    out = pool(seq, mask, "masked_mean")
    m = mask.unsqueeze(-1).to(seq.dtype)
    manual = (seq * m).sum(dim=1) / m.sum(dim=1)
    assert torch.allclose(out, manual, atol=1e-6)


def test_mean_ignores_mask(seq: torch.Tensor, mask: torch.Tensor) -> None:
    assert torch.allclose(pool(seq, mask, "mean"), seq.mean(dim=1), atol=1e-6)
    # ...and therefore differs from masked_mean when rows are padded
    assert not torch.allclose(pool(seq, mask, "mean"), pool(seq, mask, "masked_mean"))


def test_encoder_pool_is_masked_mean_alias(seq: torch.Tensor, mask: torch.Tensor) -> None:
    assert torch.equal(pool(seq, mask, "encoder_pool"), pool(seq, mask, "masked_mean"))


def test_cls_first_valid(seq: torch.Tensor, mask: torch.Tensor) -> None:
    out = pool(seq, mask, "cls")
    first = mask.to(torch.int64)
    first = torch.where(mask, first, torch.full_like(first, T))
    idx = first.argmin(dim=-1)
    manual = seq[torch.arange(B), idx]
    assert torch.equal(out, manual)
    # no mask -> literal index 0
    assert torch.equal(pool(seq, None, "cls"), seq[:, 0, :])


def test_last_valid_padding_invariance_left_right_unpadded() -> None:
    """P4.3: left-pad == right-pad == unpadded for last_valid (and peers)."""
    length = 4
    base = torch.randn(B, length, D)  # unpadded reference rows
    garbage = torch.randn(B, 2, D) * 100.0  # masked-out rows carry garbage

    right_seq = torch.cat([base, garbage], dim=1)
    right_mask = torch.cat(
        [torch.ones(B, length, dtype=torch.bool), torch.zeros(B, 2, dtype=torch.bool)], dim=1
    )
    left_seq = torch.cat([garbage, base], dim=1)
    left_mask = torch.cat(
        [torch.zeros(B, 2, dtype=torch.bool), torch.ones(B, length, dtype=torch.bool)], dim=1
    )

    unpadded = pool(base, None, "last_valid")
    right = pool(right_seq, right_mask, "last_valid")
    left = pool(left_seq, left_mask, "last_valid")
    assert torch.allclose(left, unpadded, atol=1e-6)
    assert torch.allclose(right, unpadded, atol=1e-6)
    assert torch.allclose(left, right, atol=1e-6)

    # every deterministic mask-aware strategy shares the padding invariance
    for strategy in ("cls", "masked_mean", "encoder_pool", "last_valid", "max"):
        left_out = pool(left_seq, left_mask, strategy)
        right_out = pool(right_seq, right_mask, strategy)
        assert torch.allclose(left_out, right_out, atol=1e-5), strategy

    # learned poolers: same property with FIXED weights (the stateless pool()
    # path re-initializes its module per call, so instances are compared)
    attention_pooler = AttentionPooler(D)
    assert torch.allclose(
        attention_pooler(left_seq, left_mask),
        attention_pooler(right_seq, right_mask),
        atol=1e-5,
    )
    learner_pooler = LearnerPooler(D)
    assert torch.allclose(
        learner_pooler(left_seq, left_mask),
        learner_pooler(right_seq, right_mask),
        atol=1e-5,
    )


def test_attention_pool_gradients_flow() -> None:
    x = torch.randn(B, T, D, requires_grad=True)
    pooler = AttentionPooler(D)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[:, -1] = False
    out = pooler(x, mask)
    assert out.shape == (B, D)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert pooler.query.grad is not None
    # masked position received no gradient mass (mask respect, P4.2)
    assert torch.all(x.grad[:, -1, :] == 0)


def test_learned_pool_gradients_flow() -> None:
    x = torch.randn(B, T, D, requires_grad=True)
    pooler = LearnerPooler(D)
    out = pooler(x, None)
    assert out.shape == (B, D)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None for p in pooler.parameters())


def test_stateless_attention_pool_grads_to_input(seq: torch.Tensor, mask: torch.Tensor) -> None:
    x = seq.clone().requires_grad_(True)
    out = pool(x, mask, "attention_pool")
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.parametrize("strategy", MASK_AWARE)
def test_all_masked_row_raises(strategy: str) -> None:
    x = torch.randn(2, 5, D)
    mask = torch.ones(2, 5, dtype=torch.bool)
    mask[1] = False  # second row fully masked
    with pytest.raises(ValueError, match="valid"):
        pool(x, mask, strategy)


def test_fp16_inputs_stay_finite_and_dtype(seq: torch.Tensor, mask: torch.Tensor) -> None:
    half_seq = seq.half()
    pooler_output = torch.randn(B, D).half()
    for strategy in available_pooling_strategies():
        out = pool(half_seq, mask, strategy, pooler_output=pooler_output)
        assert out.dtype == torch.float16, strategy
        assert torch.isfinite(out).all(), strategy


@pytest.mark.parametrize("seed", range(10))
def test_random_mask_property_masked_mean(seed: int) -> None:
    """Seeded random-mask loop: manual equality + padding invariance."""
    generator = torch.Generator().manual_seed(seed)
    t = int(torch.randint(3, 12, (1,), generator=generator))
    x = torch.randn(3, t, D, generator=generator)
    m = torch.rand(3, t, generator=generator) > 0.5
    m[:, 0] = True
    out = pool(x, m, "masked_mean")
    counts = m.sum(dim=-1).unsqueeze(-1).to(x.dtype)
    manual = (x * m.unsqueeze(-1)).sum(dim=1) / counts
    assert torch.allclose(out, manual, atol=1e-6)
    # append garbage-masked rows: pooled result unchanged (P4.3)
    padded_x = torch.cat([x, torch.randn(3, 4, D, generator=generator) * 1e3], dim=1)
    padded_m = torch.cat([m, torch.zeros(3, 4, dtype=torch.bool)], dim=1)
    assert torch.allclose(pool(padded_x, padded_m, "masked_mean"), out, atol=1e-6)
    # last_valid identical too
    assert torch.equal(pool(padded_x, padded_m, "last_valid"), pool(x, m, "last_valid"))


def test_integer_mask_accepted_like_bool(seq: torch.Tensor, mask: torch.Tensor) -> None:
    int_mask = mask.to(torch.int64)
    assert torch.equal(pool(seq, int_mask, "masked_mean"), pool(seq, mask, "masked_mean"))


def test_shape_and_type_errors(seq: torch.Tensor, mask: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="rank 3"):
        pool(torch.randn(B, D), mask, "mean")
    with pytest.raises(ValueError, match="attention_mask"):
        pool(seq, torch.ones(B, T + 1, dtype=torch.bool), "mean")
    with pytest.raises(TypeError, match="floating-point"):
        pool(torch.randint(0, 9, (B, T, D)), None, "mean")
    with pytest.raises(TypeError, match="strategy"):
        pool(seq, None, 123)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="torch.Tensor"):
        pool(seq, "mask", "mean")  # type: ignore[arg-type]
