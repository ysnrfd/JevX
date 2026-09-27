# Copyright (c) 2026 ysnrfd
"""Decision fusion ``F(H_S, H_Q, H_C)`` (Definition 5 of the math spec).

Contract::

    fused          = F(state, question)           # [B, D]  — the H consumed
                                                   #           by decision heads
    option_features = Psi(options ; state+question) # [B, K, D] — per-option rows

Shared option formula (documented, applied by every fusion kind)::

    c     = F_kind(s, q)                      context, [B, D]
    o-hat = o_k / max(||o_k||, eps)           option DIRECTION, [B, K, D]
    g     = sigmoid(W_g c + b_g) in (0,1)^D   per-option gate from the context
    e_k   = g ⊙ W_o o-hat_k + (1 - g) ⊙ c     k = 1..K

(the ``"cross_attention"`` kind keeps the gate shape but substitutes its
option-conditioned attention row ``u_k`` for ``W_o o_k`` — see
:class:`CrossAttentionFusion`; it also reads the keyword-only
``state_sequence=`` input that every other kind forbids).

i.e. each option row is reduced to a **unit direction** and then
*projected* (``W_o``) and fused per-option with the state+question context
through a learned gate; masked positions are filled with exact ``0`` *after*
the fusion via ``masked_fill`` (never multiply-by-zero, never NaN). When
``options is None``, ``option_features is None``.

**Why the option row is a direction and not a vector (EXP-H follow-up,
2026-09-25; normative — ``option_normalize=True``).** The option row carries
option IDENTITY, never a magnitude. ``H_C`` reaches the fusion as
``Projection(schema_rows)`` = ``LayerNorm(Linear(hashed embedding))`, so every
option row arrives with the same LayerNorm scale ``||o_k|| = sqrt(D)`` whatever
the option says — and those rows are *learnable*, so a short training budget
can inflate them arbitrarily. The option branch therefore entered the head's
option-interaction residual at a scale set by the schema encoder rather than
by the scoring function. Measured on frozen wav2vec2/RAVDESS features at
initialization (seed 0, D = 768, K = 8): ``||o_k|| = 26.6`, ``||c|| = 19.7`,
``||e_k|| = 12.6`, and the choice head's logits already had
``max|z| = 21.6``, ``std(z) = 4.28`` against a uniform NLL of ``ln 8 = 2.079`
— the cross-entropy is saturated *before the first step*, the epoch-1 train
loss starts at 178–239 (vs 8.4 for the same stack with the option pathway
bypassed), and the gradient that reaches the head — the only module that can
actually classify — is the smallest in the stack. Dividing by ``||o_k||`` makes
the option branch's scale a property of ``W_o``'s own initialization instead of
of the learned schema rows; the result is a strict *scale* superset (every
direction is still reachable, and ``W_o`` can still grow the branch), with the
measured effect reported in ``reports/ABLATIONS.md`` §8 and
``reports/MODALITY_BENCHMARKS.md`` §10. ``option_normalize=False`` restores the
pre-EXP-H formula exactly, for the ablation.

Context kinds (``build_fusion(kind, hidden_size)``):

* ``"concat"``    — ``c = W_cat [s ; q] + b``                (spec 5.a shape)
* ``"additive"``  — ``c = s + W_q q``                        (spec 5.c family;
  alias ``"add"`` as spelled in ``JevXConfig.fusion_type`` docs)
* ``"gated"``     — ``g = σ(W_g [s ; q] + b); c = g ⊙ s + (1-g) ⊙ W_q q``
                    (spec 5.b with identity ``φ_S``)
* ``"bilinear"``  — ``c = s ⊙ (W_q q)`` (diagonal bilinear interaction,
                    the vector-level ``β_diag`` of spec 5.d)
* ``"dot"``       — ``c = s ⊙ q`` (Hadamard product: the elementwise
                    counterpart of the ``dot`` scoring interaction)
* ``"mlp"``       — ``c = MLP([s ; q])`` with two Linear layers + GELU
* ``"identity"``  — ``c = W_s s + b_s``, ``W_s = I, b_s = 0`` at
                    construction: the CONTEXT-CONSTANT kind for a fusion input
                    that does not vary per row (the conditioning path run in
                    ``JevXConfig.conditioning_mode="constant"`). One
                    ``Linear(D, D)`` on the state instead of concat's
                    ``W_cat ∈ R^{D x 2D}`` — half the context parameters, and
                    exactly ``c = s`` before training. It IGNORES ``q``, so
                    the question reaches the decision heads only through the
                    per-option formula and the head's own pathways.
* ``"cross_attention"`` — spec Def 5.e: the question (context) and every
  option query the UNPOOLED state sequence ``H_S^seq`` when it is passed as
  the keyword-only ``state_sequence=`` argument (see :class:`CrossAttentionFusion`
  for the exact equations). All other kinds ignore/forbid it, so every
  existing kind's outputs stay byte-identical.

The ``s_C`` schema-summary summands of spec 5.a–5.d are carried by the
per-option formula (``e_k`` above) instead — there is no separate
``φ_C(s_C)`` term (spec 5: "implementation form is normative"; audit Q-022).

Shape validation uses ``jevx.utils.shapes`` so every mismatch names the
offending tensor.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Optional

import torch
import torch.nn as nn

from jevx.utils.shapes import assert_batch_alignment, assert_shape, check_finite

__all__ = [
    "FUSION_REGISTRY",
    "Fusion",
    "FusionOutput",
    "AdditiveFusion",
    "BilinearFusion",
    "ConcatFusion",
    "CrossAttentionFusion",
    "DotFusion",
    "GatedFusion",
    "MLPFusion",
    "IdentityFusion",
    "available_fusions",
    "build_fusion",
]


@dataclass
class FusionOutput:
    """Result of one fusion call.

    Args:
        fused: Fused state ``H`` of shape ``[B, D]`` — input ``fused`` of the
            decision heads.
        option_features: Per-option embeddings ``[B, K, D]`` (schema order,
            masked positions exactly 0), or ``None`` when no options were
            given.
    """

    fused: torch.Tensor
    option_features: Optional[torch.Tensor]


class Fusion(nn.Module, ABC):
    """Base fusion module: context computation + the shared option formula.

    Args:
        hidden_size: Shared width ``D`` of state/question/options.
        option_normalize: Reduce every option row to a unit DIRECTION before
            ``W_o`` (default ``True``, the normative EXP-H-follow-up
            contract; see the module docstring for the measurement that
            motivates it). ``False`` restores the pre-EXP-H formula
            ``e_k = g ⊙ (W_o o_k) + (1 - g) ⊙ c`` verbatim, for the ablation.

    Raises:
        ValueError: ``hidden_size`` is not a positive int, or a validated
            input violates the documented shapes (messages name the tensor).
    """

    kind: ClassVar[str] = ""
    #: Floor on the option-row norm inside the unit-direction map, so an
    #: all-zero (padded or degenerate) option row normalizes to an exact 0
    #: vector instead of 0/0 = NaN.
    OPTION_NORM_EPS: ClassVar[float] = 1e-12

    def __init__(self, hidden_size: int, option_normalize: bool = True) -> None:
        super().__init__()
        if not isinstance(hidden_size, int) or hidden_size < 1:
            raise ValueError(f"hidden_size: expected a positive int, got {hidden_size!r}.")
        if not isinstance(option_normalize, bool):
            raise TypeError(
                f"option_normalize: expected a bool, got {type(option_normalize).__name__}."
            )
        self.hidden_size = hidden_size
        self.option_normalize = option_normalize
        # Shared option-side parameters (formula in the module docstring).
        self.option_proj = nn.Linear(hidden_size, hidden_size)
        self.option_gate = nn.Linear(hidden_size, hidden_size)

    @abstractmethod
    def context(self, state: torch.Tensor, question: torch.Tensor) -> torch.Tensor:
        """Compute the fused context ``c = F_kind(s, q)`` of shape ``[B, D]``."""

    def _option_features(
        self,
        context: torch.Tensor,
        options: torch.Tensor,
        option_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply the shared per-option formula and zero masked rows.

        With ``option_normalize=True`` the option rows enter as unit
        DIRECTIONS (``o_k / max(||o_k||, eps)``), so the option branch's scale is
        fixed by ``W_o``'s initialization and not by the learned schema rows.
        The normalization is per row, elementwise-safe on an all-zero row
        (the ``eps`` floor gives an exact 0, never NaN), and the
        ``masked_fill`` below still runs LAST, so an invalid option row is
        exactly 0 whatever the option rows held.
        """
        gate = torch.sigmoid(self.option_gate(context))  # [B, D]
        if self.option_normalize:
            options = torch.nn.functional.normalize(
                options, p=2, dim=-1, eps=self.OPTION_NORM_EPS
            )
        projected = self.option_proj(options)  # [B, K, D]
        gate = gate.unsqueeze(1)  # [B, 1, D]
        context_row = context.unsqueeze(1)  # [B, 1, D]
        features = gate * projected + (1.0 - gate) * context_row
        if option_mask is not None:
            # masked_fill (not multiplication): exact 0 even if a padded
            # option row contained non-finite garbage.
            features = features.masked_fill(~option_mask.unsqueeze(-1), 0.0)
        return features

    def _fuse_with_sequence(
        self,
        state: torch.Tensor,
        question: torch.Tensor,
        options: Optional[torch.Tensor],
        option_mask: Optional[torch.Tensor],
        state_sequence: torch.Tensor,
        state_mask: Optional[torch.Tensor],
    ) -> FusionOutput:
        """Sequence-conditioned fusion path (audit Q-006 / spec Def 5.e).

        Base kinds compute `c = F(s, q)` from the POOLED state only and
        cannot consume `H_S^seq`, so passing `state_sequence=` to them is
        a caller error (never silently ignored). `CrossAttentionFusion`
        overrides this.
        """
        raise ValueError(
            f"{type(self).__name__} (fusion kind {self.kind!r}) does not consume a state "
            "sequence — pass state_sequence= only to the 'cross_attention' kind "
            "(spec Def 5.e), or drop the argument."
        )

    def forward(
        self,
        state: torch.Tensor,
        question: torch.Tensor,
        options: Optional[torch.Tensor] = None,
        option_mask: Optional[torch.Tensor] = None,
        *,
        state_sequence: Optional[torch.Tensor] = None,
        state_mask: Optional[torch.Tensor] = None,
    ) -> FusionOutput:
        """Fuse one batch.

        Args:
            state: State representation ``H_S`` of shape ``[B, D]``.
            question: Question representation ``H_Q`` of shape ``[B, D]``
                (same batch as ``state``).
            options: Schema rows ``H_C`` of shape ``[B, K, D]``, or ``None``.
            option_mask: Optional bool ``[B, K]`` (``True`` = valid option);
                requires ``options``.
            state_sequence: Optional unpooled state ``H_S^seq`` ``[B, T, D]``
                (audit Q-006 / spec Def 5.e). ``None`` (default) keeps every
                existing kind on its exact pre-Q-006 code path — only
                ``"cross_attention"`` consumes it; any other kind raises a
                ValueError naming the kind.
            state_mask: Optional bool ``[B, T]`` token-validity mask aligned
                with ``state_sequence`` (``True`` = valid position): padded
                positions get zero attention weight and can never contribute.
                Requires ``state_sequence``.

        Returns:
            :class:`FusionOutput` with ``fused [B, D]`` and
            ``option_features [B, K, D]`` (all-masked positions exactly 0).

        Raises:
            TypeError: A non-tensor input, or a non-bool ``option_mask```
                / ``state_mask``.
            ValueError: Wrong rank/dimension, batch mismatch, mask shape
                mismatch, mask without its tensor, an all-masked sequence
                row, non-finite inputs, or ``state_sequence=`` passed to a
                kind that does not consume it.
        """
        assert_shape(state, (None, self.hidden_size), "state")
        assert_shape(question, (None, self.hidden_size), "question")
        check_finite(state, "state")
        check_finite(question, "question")
        if options is not None:
            assert_shape(options, (None, None, self.hidden_size), "options")
            check_finite(options, "options")
        if option_mask is not None:
            if options is None:
                raise ValueError(
                    "option_mask was provided without options — the mask indexes the option "
                    "axis K, which only exists when `options` is given."
                )
            if option_mask.dtype != torch.bool:
                raise TypeError(
                    f"option_mask: expected a torch.bool tensor (True = valid option), got "
                    f"{option_mask.dtype}; convert it with `.bool()`."
                )
            expected = (options.shape[0], options.shape[1])
            if tuple(option_mask.shape) != expected:
                raise ValueError(
                    f"option_mask: expected shape {expected} to match options "
                    f"{tuple(options.shape)}, got {tuple(option_mask.shape)}."
                )
        # -- sequence inputs (audit Q-006 / spec Def 5.e) ------------------
        if state_sequence is None:
            if state_mask is not None:
                raise ValueError(
                    "state_mask was provided without state_sequence — the mask indexes "
                    "the sequence axis T, which only exists when state_sequence is given."
                )
        else:
            assert_shape(state_sequence, (None, None, self.hidden_size), "state_sequence")
            check_finite(state_sequence, "state_sequence")
            if int(state_sequence.shape[1]) < 1:
                raise ValueError(
                    "state_sequence: expected at least one token position (T >= 1), got "
                    f"T={int(state_sequence.shape[1])}."
                )
            if state_mask is not None:
                if state_mask.dtype != torch.bool:
                    raise TypeError(
                        f"state_mask: expected a torch.bool tensor (True = valid "
                        f"position), got {state_mask.dtype}; convert it with "
                        "`.bool()`."
                    )
                assert_shape(
                    state_mask,
                    (state_sequence.shape[0], state_sequence.shape[1]),
                    "state_mask",
                )
                empty = ~state_mask.any(dim=-1)
                if bool(empty.any()):
                    row = int(empty.nonzero(as_tuple=False)[0])
                    raise ValueError(
                        f"state_mask: row {row} has no valid position — attention "
                        "cannot run over an all-padded sequence (axiom D1.1: "
                        "all-masked rows raise; there is nothing valid to attend to)."
                    )
        assert_batch_alignment(
            state=state,
            question=question,
            options=options,
            option_mask=option_mask,
            state_sequence=state_sequence,
            state_mask=state_mask,
        )

        if state_sequence is not None:
            return self._fuse_with_sequence(
                state, question, options, option_mask, state_sequence, state_mask
            )

        context = self.context(state, question)
        assert_shape(context, (state.shape[0], self.hidden_size), f"{type(self).__name__}.context")
        check_finite(context, f"{type(self).__name__}.context")
        option_features = (
            self._option_features(context, options, option_mask)
            if options is not None
            else None
        )
        return FusionOutput(fused=context, option_features=option_features)


class ConcatFusion(Fusion):
    """``c = W_cat [s ; q] + b`` with ``W_cat in R^{D x 2D}`` (spec 5.a)."""

    kind = "concat"

    def __init__(self, hidden_size: int, option_normalize: bool = True) -> None:
        super().__init__(hidden_size, option_normalize=option_normalize)
        self.mix = nn.Linear(2 * hidden_size, hidden_size)

    def context(self, state: torch.Tensor, question: torch.Tensor) -> torch.Tensor:
        return self.mix(torch.cat([state, question], dim=-1))


class AdditiveFusion(Fusion):
    """``c = s + W_q q`` (bias-free; spec 5.c family without the LN/residual)."""

    kind = "additive"

    def __init__(self, hidden_size: int, option_normalize: bool = True) -> None:
        super().__init__(hidden_size, option_normalize=option_normalize)
        self.w_question = nn.Linear(hidden_size, hidden_size, bias=False)

    def context(self, state: torch.Tensor, question: torch.Tensor) -> torch.Tensor:
        return state + self.w_question(question)


class GatedFusion(Fusion):
    """``g = σ(W_g [s;q] + b); c = g ⊙ s + (1-g) ⊙ W_q q`` (spec 5.b)."""

    kind = "gated"

    def __init__(self, hidden_size: int, option_normalize: bool = True) -> None:
        super().__init__(hidden_size, option_normalize=option_normalize)
        self.gate = nn.Linear(2 * hidden_size, hidden_size)
        self.w_question = nn.Linear(hidden_size, hidden_size, bias=False)

    def context(self, state: torch.Tensor, question: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(torch.cat([state, question], dim=-1)))
        return g * state + (1.0 - g) * self.w_question(question)


class BilinearFusion(Fusion):
    """Diagonal bilinear context ``c = s ⊙ (W_q q)`` (vector ``β_diag``, 5.d)."""

    kind = "bilinear"

    def __init__(self, hidden_size: int, option_normalize: bool = True) -> None:
        super().__init__(hidden_size, option_normalize=option_normalize)
        self.w_question = nn.Linear(hidden_size, hidden_size, bias=False)

    def context(self, state: torch.Tensor, question: torch.Tensor) -> torch.Tensor:
        return state * self.w_question(question)


class DotFusion(Fusion):
    """Hadamard context ``c = s ⊙ q`` (elementwise counterpart of ``dot``)."""

    kind = "dot"

    def context(self, state: torch.Tensor, question: torch.Tensor) -> torch.Tensor:
        return state * question


class MLPFusion(Fusion):
    """``c = W_2 · GELU(W_1 [s ; q] + b_1) + b_2``."""

    kind = "mlp"

    def __init__(self, hidden_size: int, option_normalize: bool = True) -> None:
        super().__init__(hidden_size, option_normalize=option_normalize)
        self.mix = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def context(self, state: torch.Tensor, question: torch.Tensor) -> torch.Tensor:
        return self.mix(torch.cat([state, question], dim=-1))




class IdentityFusion(Fusion):
    """``c = W_s s + b_s``, initialized to the IDENTITY map.

    The context-constant fusion kind: the conditioning input is the same for
    every call (that is what ``JevXConfig.conditioning_mode="constant"``
    caches), so the context does not need to read the question at all. It is
    a single ``nn.Linear(D, D)`` on the state — ``D^2 + D`` parameters
    against concat's ``2D^2 + D`` — and it is initialized to ``W_s = I`,
    ``b_s = 0`, so ``c == s`` exactly at construction and the untrained
    model is exactly the linear state probe's input.

    ``q`` is accepted and ignored (the base ``Fusion.forward`` contract still
    validates it), and the per-option formula, its masking and the
    ``option_features [B, K, D]`` shape are the inherited, unchanged ones.
    """

    kind = "identity"

    def __init__(self, hidden_size: int, option_normalize: bool = True) -> None:
        super().__init__(hidden_size, option_normalize=option_normalize)
        self.state_map = nn.Linear(hidden_size, hidden_size)
        with torch.no_grad():
            self.state_map.weight.copy_(torch.eye(hidden_size))
            self.state_map.bias.zero_()

    def context(self, state: torch.Tensor, question: torch.Tensor) -> torch.Tensor:
        """``c = W_s s + b_s`` — the question does not enter the context."""
        return self.state_map(state)


class CrossAttentionFusion(Fusion):
    """Spec Def 5.e — the question and every option cross-attend the UNPOOLED
    state sequence `H_S^seq` (plumbed in as the keyword-only
    `state_sequence=` argument, audit Q-006).

    Equations (`X = state_sequence`` in ``[B, T, D]``, pooled state `s`,
    question `q`, option row `o_k`, token mask `M`; multi-head with
    `H = num_heads` heads of width `d_h = D / H` — heads are computed in
    parallel and concatenated along `d_h` then re-projected by `W^o`, per
    the spec's "concatenate `h` heads along `d_k` and apply `W^O`"):

        context (fused `H`):
            Q_h = (W^q q)[B,H,d_h],  K_h = (W^k X)[B,H,T,d_h],
            V_h = (W^v X)[B,H,T,d_h]
            a_h = softmax_t( Q_h K_h^T / sqrt(d_h) + B(M) )     [B, H, T]
            c   = W^o concat_h(a_h V_h)                          [B, D]

        per option (option queries over the sequence — spec 5.e's `Q_o`;
        keeps the shared option-gate shape `e_k = g ⊙ (·) + (1-g) ⊙ c`,
        with `W_o o_k` replaced by the option-conditioned row `u_k`):
            u_k = CrossAttn(query = W^cq [o-hat_k ; c], keys = K, values = V)
            g   = sigmoid(W_g c + b_g)                (shared option gate)
            e_k = g ⊙ u_k + (1-g) ⊙ c,  then masked_fill(invalid) = 0

    Masking: `B(M)` pads with `torch.finfo(dtype).min` (repo convention —
    never `-inf`, so gradients stay NaN-free), which underflows to an EXACT
    zero attention weight at every padded position as long as each row keeps
    >= 1 valid position (an all-padded row raises — axiom D1.1). Padded
    hidden values therefore contribute exactly 0: perturbing them cannot move
    `c` or any `e_k` (padding invariance, tested).

    Pooled `state` is not read when a sequence is supplied (it is a pool of
    the same sequence). Direct :meth:`forward` calls WITHOUT a sequence use a
    degenerate T=2 sequence `[s ; q]` so the equations above still hold;
    :class:`jevx.JevXForDecision` always supplies the real sequence for this
    kind (plumbing in `modeling_jevx.forward`) and raises an actionable
    ValueError when the adapter produced no sequence at all.

    Args:
        hidden_size: Shared width `D` (must be divisible by `num_heads`).
        num_heads: Number of attention heads (default 1 = single-head scaled
            dot-product; requires `D % num_heads == 0`).
        option_normalize: Same contract as the base class - the option rows
            enter `W^{cq}[H_{C,k}; c]` as unit directions (default True).

    Raises:
        ValueError: `num_heads` is not a positive int or does not divide
            `hidden_size`.
    """

    kind = "cross_attention"

    def __init__(
        self, hidden_size: int, num_heads: int = 1, option_normalize: bool = True
    ) -> None:
        super().__init__(hidden_size, option_normalize=option_normalize)
        if not isinstance(num_heads, int) or num_heads < 1:
            raise ValueError(
                f"num_heads: expected a positive int, got {num_heads!r}."
            )
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"num_heads: {num_heads} must divide hidden_size {hidden_size} "
                "(hidden_size % num_heads == 0) so every head gets the same "
                "width."
            )
        self.num_heads = num_heads
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.option_query = nn.Linear(2 * hidden_size, hidden_size)

    # -- shared scaled dot-product primitive (Def 5.e) -----------------------

    def _attend(
        self,
        query: torch.Tensor,          # [B, D] or [B, N, D]
        keys: torch.Tensor,           # [B, T, D]
        values: torch.Tensor,         # [B, T, D]
        mask: Optional[torch.Tensor],  # [B, T] bool (True = valid) or None
    ) -> torch.Tensor:                # [B, D] or [B, N, D]
        """Single-/multi-head masked scaled dot-product attention."""
        squeeze = query.dim() == 2
        if squeeze:
            query = query.unsqueeze(1)
        b, n, d = query.shape
        t = keys.shape[1]
        heads, dh = self.num_heads, d // self.num_heads
        q = self.query_proj(query).view(b, n, heads, dh).transpose(1, 2)
        k = self.key_proj(keys).view(b, t, heads, dh).transpose(1, 2)
        v = self.value_proj(values).view(b, t, heads, dh).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) * (dh ** -0.5)   # [B, heads, N, T]
        if mask is not None:
            # finfo.min (not -inf): softmax underflows padded scores to an
            # exact 0.0 weight with NaN-free gradients (repo convention).
            scores = scores.masked_fill(
                ~mask[:, None, None, :], torch.finfo(scores.dtype).min
            )
        attn = torch.softmax(scores, dim=-1)                 # [B, heads, N, T]
        out = (attn @ v).transpose(1, 2).reshape(b, n, d)    # [B, N, D]
        out = self.out_proj(out)
        return out.squeeze(1) if squeeze else out

    # -- Fusion hooks ---------------------------------------------------------

    def context(self, state: torch.Tensor, question: torch.Tensor) -> torch.Tensor:
        """No sequence supplied: attend over the degenerate T=2 sequence
        `[s ; q]` (attention then mixes the pooled state and the question)."""
        seq = torch.stack([state, question], dim=1)          # [B, 2, D]
        return self._attend(question, seq, seq, None)        # [B, D]

    def _fuse_with_sequence(
        self,
        state: torch.Tensor,
        question: torch.Tensor,
        options: Optional[torch.Tensor],
        option_mask: Optional[torch.Tensor],
        state_sequence: torch.Tensor,
        state_mask: Optional[torch.Tensor],
    ) -> FusionOutput:
        """Def 5.e path: question-as-query context + per-option queries."""
        fused = self._attend(
            question, state_sequence, state_sequence, state_mask
        )                                                # [B, D]
        assert_shape(fused, (state.shape[0], self.hidden_size),
                     "CrossAttentionFusion.context")
        check_finite(fused, "CrossAttentionFusion.context")
        option_features: Optional[torch.Tensor] = None
        if options is not None:
            k = options.shape[1]
            if self.option_normalize:
                # Same option-DIRECTION contract as the base formula: the
                # option row carries identity, not magnitude.
                options = torch.nn.functional.normalize(
                    options, p=2, dim=-1, eps=self.OPTION_NORM_EPS
                )
            # concat(option, context) -> query (spec 5.e option queries).
            query = self.option_query(
                torch.cat(
                    [options, fused.unsqueeze(1).expand(-1, k, -1)], dim=-1
                )
            )                                             # [B, K, D]
            rows = self._attend(
                query, state_sequence, state_sequence, state_mask
            )                                             # [B, K, D]
            check_finite(rows, "CrossAttentionFusion.option_rows")
            gate = torch.sigmoid(self.option_gate(fused)).unsqueeze(1)  # [B,1,D]
            context_row = fused.unsqueeze(1)              # [B, 1, D]
            features = gate * rows + (1.0 - gate) * context_row
            if option_mask is not None:
                # masked_fill: exact 0 even for garbage padded queries.
                features = features.masked_fill(~option_mask.unsqueeze(-1), 0.0)
            option_features = features
        return FusionOutput(fused=fused, option_features=option_features)

#: Registry of fusion kinds. ``"add"`` is a documented alias of ``"additive"``
#: (the spelling used in ``JevXConfig.fusion_type`` documentation).
FUSION_REGISTRY: dict[str, type[Fusion]] = {
    "concat": ConcatFusion,
    "additive": AdditiveFusion,
    "add": AdditiveFusion,
    "gated": GatedFusion,
    "bilinear": BilinearFusion,
    "dot": DotFusion,
    "mlp": MLPFusion,
    "identity": IdentityFusion,
    "cross_attention": CrossAttentionFusion,
}


def available_fusions() -> list[str]:
    """Sorted names of every registered fusion kind (including aliases)."""
    return sorted(FUSION_REGISTRY)


def build_fusion(kind: str, hidden_size: int, **kwargs: Any) -> Fusion:
    """Construct the fusion module registered under ``kind``.

    Args:
        kind: Registry key (see :data:`FUSION_REGISTRY`).
        hidden_size: Shared width ``D``.
        **kwargs: Extra constructor arguments for the specific kind.

    Raises:
        TypeError: ``kind`` is not a string.
        ValueError: ``kind`` is unknown; the message lists the registry.
    """
    if not isinstance(kind, str):
        raise TypeError(f"kind: expected a str, got {type(kind).__name__}.")
    if kind not in FUSION_REGISTRY:
        raise ValueError(
            f"unknown fusion kind {kind!r}; registered kinds are {available_fusions()}."
        )
    return FUSION_REGISTRY[kind](hidden_size, **kwargs)
