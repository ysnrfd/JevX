# Copyright (c) 2026 ysnrfd
"""Mask-aware sequence pooling strategies (Definition 4 of the math spec).

transformers v5.16.1 ships **no generic pooling utility** (audit Q7: poolers
exist only on bert/roberta/albert/vit/clip/siglip/ast), so JevX implements
the pooling operators formally once, here, and every adapter routes through
them.

Mathematical contract (``docs/MATHEMATICAL_SPECIFICATION.md`` Definition 4):
for ``X ∈ R^{B×T×D}`` with validity mask ``M ∈ {0,1}^{B×T}`` and valid set
``I_b = {t : M[b,t] = 1}``:

* ``mean``           — unmasked ``mean_t X[b,t]`` (mask ignored).
* ``masked_mean``    — ``|I_b|^{-1} Σ_{t∈I_b} X[b,t]``          (P4.a)
* ``max``            — ``max_{t∈I_b} X[b,t]`` componentwise      (P4.b)
* ``cls``            — ``X[b, t1]``, ``t1 = min I_b`` (index 0 without
                       a mask)                                   (P4.c)
* ``last_valid``     — ``X[b, i_b]``, ``i_b = max I_b`` (last index
                       without a mask)                           (P4.d)
* ``attention_pool`` — ``Σ_t a_t x_t`` with ``a = softmax(q·x_t)``
                       masked (learned query, P4.e / AttentionPooler)
* ``learned_pool``   — ``Σ_t a_t x_t`` with ``a = softmax(wᵀ tanh(W x_t))``
                       masked (LearnerPooler, the general g_φ form)
* ``native``         — the model's own ``pooler_output``; raises when the
                       model exposes none (never silently pools).
* ``encoder_pool``   — documented alias of ``masked_mean``: encoder(-decoder)
                       states are conventionally mean-pooled (sentence-T5
                       convention; compatibility matrix "encoder" rows).

Invariants enforced:

1. Padding invariance (P4.3): masked rows never contribute — left *and*
   right padding give identical results for every mask-aware strategy.
2. fp16/bf16 inputs are pooled in fp32 and cast back (audit §low-precision
   convention shared with ``jevx.utils.numerical``).
3. A fully-masked row raises a descriptive error — masked pooling over an
   empty set is undefined (D1.1), never NaN.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import torch
import torch.nn as nn

from jevx.utils.numerical import LOW_PRECISION_DTYPES, masked_softmax, stable_softmax
from jevx.utils.shapes import assert_shape

__all__ = [
    "POOLING_STRATEGIES",
    "AttentionPooler",
    "LearnerPooler",
    "available_pooling_strategies",
    "pool",
]


class AttentionPooler(nn.Module):
    """Learned-query dot-product attention pooling (P4.e).

    ``scores[b,t] = <q, x[b,t]> / sqrt(D)`` with a single learned query
    ``q ∈ R^D``; masked softmax over ``t``; output ``Σ_t a[b,t] x[b,t]``.

    Shapes: forward ``x [B, T, D]``, optional bool ``attention_mask [B, T]``
    (``True`` = valid) → ``[B, D]``. Masked softmax never yields NaN for
    partially-masked rows; fully-masked rows raise (via ``masked_softmax``).
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        if not isinstance(hidden_size, int) or hidden_size < 1:
            raise ValueError(f"hidden_size must be a positive int, got {hidden_size!r}.")
        self.hidden_size = hidden_size
        self.query = nn.Parameter(torch.randn(hidden_size) / (hidden_size**0.5))

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        assert_shape(x, (None, None, self.hidden_size), "x")
        compute = x.float() if x.dtype in LOW_PRECISION_DTYPES else x
        scores = torch.einsum("btd,d->bt", compute, self.query.to(compute.dtype))
        scores = scores / (self.hidden_size**0.5)
        weights = masked_softmax(scores, attention_mask)
        pooled = torch.einsum("bt,btd->bd", weights.to(compute.dtype), compute)
        return pooled.to(x.dtype) if pooled.dtype != x.dtype else pooled


class LearnerPooler(nn.Module):
    """General learned pooling ``P_g`` via an additive scorer (P4.e).

    ``g_φ(x) = wᵀ tanh(W x + b₁) + b₂``; ``a = softmax(g_φ(X) + B(M))``;
    output ``Σ_t a_t x_t``. This is the ``wᵀ tanh(W x_t)`` form written in
    the mathematical specification, with a configurable inner width.

    Shapes: forward ``x [B, T, D]``, optional bool ``attention_mask [B, T]``
    → ``[B, D]``.
    """

    def __init__(self, hidden_size: int, attn_hidden: Optional[int] = None) -> None:
        super().__init__()
        if not isinstance(hidden_size, int) or hidden_size < 1:
            raise ValueError(f"hidden_size must be a positive int, got {hidden_size!r}.")
        inner = attn_hidden if attn_hidden is not None else hidden_size
        if not isinstance(inner, int) or inner < 1:
            raise ValueError(f"attn_hidden must be a positive int, got {inner!r}.")
        self.hidden_size = hidden_size
        self.score = nn.Sequential(
            nn.Linear(hidden_size, inner),
            nn.Tanh(),
            nn.Linear(inner, 1),
        )

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        assert_shape(x, (None, None, self.hidden_size), "x")
        compute = x.float() if x.dtype in LOW_PRECISION_DTYPES else x
        scores = self.score(compute).squeeze(-1)  # [B, T]
        weights = masked_softmax(scores, attention_mask)
        pooled = torch.einsum("bt,btd->bd", weights.to(compute.dtype), compute)
        return pooled.to(x.dtype) if pooled.dtype != x.dtype else pooled


# ---------------------------------------------------------------------------
# Stateless strategies
# ---------------------------------------------------------------------------


def _validate(
    sequence_output: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.dtype]:
    """Common validation: rank-3 sequence, optional bool mask, work dtype.

    Returns ``(compute_x, mask_bool, original_dtype)`` where ``compute_x``
    is fp32 for fp16/bf16 inputs (pooled in fp32, cast back by ``pool``).
    """
    assert_shape(sequence_output, (None, None, None), "sequence_output")
    if not sequence_output.is_floating_point():
        raise TypeError(
            f"sequence_output: expected a floating-point tensor, got {sequence_output.dtype}."
        )
    mask: Optional[torch.Tensor] = None
    if attention_mask is not None:
        if not isinstance(attention_mask, torch.Tensor):
            raise TypeError(
                f"attention_mask: expected a torch.Tensor or None, got "
                f"{type(attention_mask).__name__}."
            )
        if attention_mask.dtype != torch.bool:
            if not (
                attention_mask.dtype
                in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
                or attention_mask.is_floating_point()
            ):
                raise TypeError(
                    f"attention_mask: expected a bool or integer/float mask, got {attention_mask.dtype}."
                )
            attention_mask = attention_mask.to(torch.bool)
        expected = (sequence_output.shape[0], sequence_output.shape[1])
        if tuple(attention_mask.shape) != expected:
            raise ValueError(
                f"attention_mask: expected shape {expected} to match sequence_output "
                f"{tuple(sequence_output.shape)}, got {tuple(attention_mask.shape)}."
            )
        mask = attention_mask
    compute = (
        sequence_output.float()
        if sequence_output.dtype in LOW_PRECISION_DTYPES
        else sequence_output
    )
    return compute, mask, sequence_output.dtype


def _require_nonempty_rows(mask: torch.Tensor, strategy: str) -> None:
    """Raise when any row of ``mask`` has no valid token (D1.1 violation)."""
    empty = ~mask.any(dim=-1)
    if bool(empty.any()):
        rows = torch.nonzero(empty, as_tuple=False).flatten().tolist()
        raise ValueError(
            f"pool(strategy={strategy!r}): row(s) {rows} have no valid tokens "
            f"(attention_mask is all-False) — masked pooling over an empty set is "
            f"undefined; check padding or filter empty sequences (axiom D1.1)."
        )


def _native(
    sequence_output: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    pooler_output: Optional[torch.Tensor],
    hidden_states: Optional[tuple],
) -> torch.Tensor:
    if pooler_output is None:
        raise ValueError(
            "pool(strategy='native'): the model exposes no pooler_output — this "
            "backbone has no trained pooler (audit Q7: llama/gpt2/t5/distilbert/"
            "wav2vec2 ship none). Use an explicit strategy ('masked_mean', "
            "'cls', 'last_valid', ...) instead."
        )
    return assert_shape(pooler_output, (None, None), "pooler_output")


def _cls(
    sequence_output: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    pooler_output: Optional[torch.Tensor],
    hidden_states: Optional[tuple],
) -> torch.Tensor:
    if attention_mask is None:
        return sequence_output[:, 0, :]
    _require_nonempty_rows(attention_mask, "cls")
    # First valid index per row: 1 where mask else large → argmin.
    first = attention_mask.to(torch.int64)
    first = torch.where(attention_mask, first, torch.full_like(first, sequence_output.shape[1]))
    idx = first.argmin(dim=-1)
    return sequence_output[torch.arange(sequence_output.shape[0], device=sequence_output.device), idx]


def _mean(
    sequence_output: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    pooler_output: Optional[torch.Tensor],
    hidden_states: Optional[tuple],
) -> torch.Tensor:
    # Unmasked mean by definition (mask deliberately ignored).
    return sequence_output.mean(dim=1)


def _masked_mean(
    sequence_output: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    pooler_output: Optional[torch.Tensor],
    hidden_states: Optional[tuple],
) -> torch.Tensor:
    if attention_mask is None:
        return sequence_output.mean(dim=1)
    _require_nonempty_rows(attention_mask, "masked_mean")
    m = attention_mask.unsqueeze(-1).to(sequence_output.dtype)
    counts = m.sum(dim=1).clamp(min=1.0)
    return (sequence_output * m).sum(dim=1) / counts


def _last_valid(
    sequence_output: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    pooler_output: Optional[torch.Tensor],
    hidden_states: Optional[tuple],
) -> torch.Tensor:
    if attention_mask is None:
        return sequence_output[:, -1, :]
    _require_nonempty_rows(attention_mask, "last_valid")
    # i_b = max{t : mask[b,t]=1}: reverse argmax of the mask.
    idx = attention_mask.to(torch.int64).flip(-1).argmax(dim=-1)
    idx = attention_mask.shape[1] - 1 - idx
    return sequence_output[torch.arange(sequence_output.shape[0], device=sequence_output.device), idx]


def _max(
    sequence_output: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    pooler_output: Optional[torch.Tensor],
    hidden_states: Optional[tuple],
) -> torch.Tensor:
    if attention_mask is None:
        return sequence_output.max(dim=1).values
    _require_nonempty_rows(attention_mask, "max")
    very_neg = torch.finfo(sequence_output.dtype).min
    filled = sequence_output.masked_fill(~attention_mask.unsqueeze(-1), very_neg)
    return filled.max(dim=1).values


def _attention_pool(
    sequence_output: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    pooler_output: Optional[torch.Tensor],
    hidden_states: Optional[tuple],
) -> torch.Tensor:
    # Stateless call path: parameters are re-initialized per call; for a
    # *trained* pooler own an AttentionPooler module explicitly. The re-init
    # draw is FORKED + FIXED-SEEDED (phase 74): rebuilding a random query on
    # every call made two identical eval calls disagree bit-for-bit and
    # advanced the caller RNG stream as a side effect.
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(0)  # deterministic stateless re-init
        module = AttentionPooler(sequence_output.shape[-1])
    module = module.to(
        device=sequence_output.device,
        dtype=sequence_output.dtype if sequence_output.dtype not in LOW_PRECISION_DTYPES else torch.float32,
    )
    return module(sequence_output, attention_mask)


def _learned_pool(
    sequence_output: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    pooler_output: Optional[torch.Tensor],
    hidden_states: Optional[tuple],
) -> torch.Tensor:
    # Same determinism contract as _attention_pool (phase 74): fork the RNG
    # and fix the re-init seed so repeated eval calls are bitwise identical
    # and the caller RNG stream is left untouched.
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(0)
        module = LearnerPooler(sequence_output.shape[-1])
    module = module.to(
        device=sequence_output.device,
        dtype=sequence_output.dtype if sequence_output.dtype not in LOW_PRECISION_DTYPES else torch.float32,
    )
    return module(sequence_output, attention_mask)


#: Registry mapping strategy name → ``fn(x, mask, pooler_output, hidden_states)``.
#: All entries return ``[B, D]``.
POOLING_STRATEGIES: Dict[str, Callable[..., torch.Tensor]] = {
    "native": _native,
    "cls": _cls,
    "mean": _mean,
    "masked_mean": _masked_mean,
    "encoder_pool": _masked_mean,  # documented alias (audit: encoder states)
    "last_valid": _last_valid,
    "max": _max,
    "attention_pool": _attention_pool,
    "learned_pool": _learned_pool,
}


def available_pooling_strategies() -> list[str]:
    """Sorted names of every registered pooling strategy."""
    return sorted(POOLING_STRATEGIES)


def pool(
    sequence_output: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    strategy: str,
    *,
    pooler_output: Optional[torch.Tensor] = None,
    hidden_states: Optional[tuple] = None,
) -> torch.Tensor:
    """Pool a sequence to a per-row vector.

    Args:
        sequence_output: Hidden states ``[B, T, D]``.
        attention_mask: Optional validity mask ``[B, T]`` — bool, or an
            integer/float mask converted to bool (``True``/nonzero = valid).
            Handles left *and* right padding identically (mask-driven).
        strategy: One of :func:`available_pooling_strategies`.
        pooler_output: Native pooled vector ``[B, D]`` — required by
            ``strategy="native"``, ignored otherwise.
        hidden_states: Optional per-layer tuple; unused by current
            strategies but accepted so adapters can forward output field
            collections uniformly (audit: ``hidden_states[-1]`` is the
            last layer, pre-norm for vision/audio models).

    Returns:
        Pooled tensor ``[B, D]`` with the input's dtype and device.

    Raises:
        TypeError: Non-tensor inputs, non-float sequence, or a bad mask type.
        ValueError: Unknown strategy (message lists the registry), a
            shape-mismatched mask, or a fully-masked row (per-strategy,
            message names the offending rows).
    """
    if not isinstance(strategy, str):
        raise TypeError(f"strategy: expected a str, got {type(strategy).__name__}.")
    if strategy not in POOLING_STRATEGIES:
        raise ValueError(
            f"unknown pooling strategy {strategy!r}; available strategies are "
            f"{available_pooling_strategies()}."
        )
    if strategy == "native":
        # native bypasses sequence validation: pooler_output alone decides.
        return _native(sequence_output, attention_mask, pooler_output, hidden_states)
    compute, mask, original_dtype = _validate(sequence_output, attention_mask)
    pooled = POOLING_STRATEGIES[strategy](compute, mask, pooler_output, hidden_states)
    assert_shape(pooled, (sequence_output.shape[0], None), f"pool({strategy!r})")
    return pooled.to(original_dtype) if pooled.dtype != original_dtype else pooled
