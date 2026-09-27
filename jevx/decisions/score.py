# Copyright (c) 2026 ysnrfd
"""Score decision head: distribution over ordinal levels with an expected score."""

from __future__ import annotations

from typing import Any

import torch

from jevx.modeling_outputs import DecisionOutput

from .generic import (
    MLPStatePathway,
    DecisionHead,
    _build_interaction,
    _interact,
    register_decision_head,
)

__all__ = ["ScoreHead"]


@register_decision_head("score")
class ScoreHead(DecisionHead):
    """Ordinal score over levels ``0 .. K-1``.

    Mathematical contract::

        z_k = state_k + lambda * r_k(s or h_s, level_k)   # the same scoring
             state_k = s^T W[k, :] + b[k]                 #   identity as the
                    or h_s^T W2[k, :] + b[k]             #   choice head, so
             h_s = act(W1 s + b1)                        #   lambda = 0 makes
        p   = softmax(z) over valid levels                #   the head a pure
        s   = sum_{k=0}^{K-1} k * p_k                     #   state probe
        y   = argmax_k p_k                # predicted level (tie -> lowest index)

    With the fixed ``num_levels=K`` schema there is no option row, hence no
    residual: ``level_proj`` (an ``nn.Linear(D, K)``, the linear state pathway)
    IS the state score there, and ``state_pathway="mlp"`` replaces it with the
    nonlinear one. Either way ``lambda`` is inert for that schema.

    ``s`` is a convex combination of ``{0, ..., K-1}`` because ``p >= 0`` and
    ``sum p = 1``; the implementation *asserts* the bound (tolerance 1e-4)
    instead of silently clamping — an escape means a numerics bug upstream.

    Two schemas are supported:

    * variable ``K`` via ``option_features [B, K, D]`` (level embeddings
      scored with the chosen interaction), or
    * a fixed ``num_levels=K`` at construction, scoring ``fused`` with a
      learned linear map (then ``option_features`` is optional).

    Args:
        hidden_size: Input dim ``D``.
        interaction: Scoring mode for the option-features schema:
            ``"dot"``, ``"bilinear"`` or ``"mlp"`` (default ``"dot"``).
        num_levels: Fixed ``K`` for the linear schema; ``None`` means
            ``option_features`` must be supplied at every forward.
        state_pathway: Kind of the primary state score, ``"linear"`` (default)
            or ``"mlp"``; see :class:``jevx.decisions.DecisionHead`.
        state_hidden_size: Hidden width ``H`` of the ``"mlp"`` pathway.
        state_activation: Elementwise activation of the ``"mlp"`` pathway.
        conditioning_scale_init: Initial ``lambda`` of the residual.
        conditioning_enabled: When False the residual is exactly zero.
    """

    def __init__(
        self,
        hidden_size: int,
        interaction: str = "dot",
        num_levels: int | None = None,
        *,
        state_pathway: str = "linear",
        state_hidden_size: int | None = None,
        state_activation: str = "gelu",
        conditioning_scale_init: float = 1.0,
        conditioning_enabled: bool = True,
    ) -> None:
        super().__init__(
            hidden_size,
            state_pathway=state_pathway,
            state_hidden_size=state_hidden_size,
            state_activation=state_activation,
            conditioning_scale_init=conditioning_scale_init,
            conditioning_enabled=conditioning_enabled,
            # The fixed-K schema has no option row, hence no residual to scale.
            has_interaction=num_levels is None,
        )
        if num_levels is not None:
            if not isinstance(num_levels, int) or num_levels < 1:
                raise ValueError(f"num_levels must be a positive int, got {num_levels!r}.")
            if state_pathway == "mlp":
                # Nonlinear fixed-K scorer; `level_proj` is the LINEAR one and
                # stays the transplant target of convert() in the default mode.
                self.level_proj = None
                self.state_pathway = MLPStatePathway(
                    hidden_size,
                    num_levels,
                    self.state_hidden_size,
                    state_activation,
                )
            else:
                self.level_proj = torch.nn.Linear(hidden_size, num_levels)
                self.state_pathway = None
        else:
            self.level_proj = None
            self.state_pathway = self._make_state_pathway(state_pathway, None)
        self.interaction = interaction
        self.num_levels = num_levels
        self.bilinear_weight, self.mlp = _build_interaction(interaction, hidden_size)

    def _expected_width(self, option_features: torch.Tensor | None) -> int | None:
        if option_features is not None:
            return option_features.shape[1]
        return self.num_levels

    def compute_logits(
        self,
        fused: torch.Tensor,
        option_features: torch.Tensor | None = None,
        *,
        conditioning_enabled: bool | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if option_features is None and self.level_proj is not None:
            # Fixed-K schema: no option row, so no residual — the linear state
            # pathway is the whole score (unchanged from the shipped head).
            return self.level_proj(fused)
        width = self._expected_width(option_features)
        if width is None:  # pragma: no cover - both scorers absent
            raise ValueError(
                "ScoreHead: provide option_features [B, K, D] or construct the head with "
                "num_levels=K — there is nothing to produce K level logits from `fused` alone."
            )
        state_logits, vector = self._state_scores(fused, width)
        if option_features is None:
            return state_logits  # type: ignore[return-value]  # nonlinear fixed-K
        residual = _interact(
            vector,
            option_features,
            self.interaction,
            bilinear_weight=self.bilinear_weight,
            mlp=self.mlp,
        )
        return self._combine(state_logits, residual, conditioning_enabled)

    def build_output(
        self, logits: torch.Tensor, mask: torch.Tensor | None = None
    ) -> DecisionOutput:
        probs, predicted_index = self._probs_and_index(logits, mask)
        k = probs.shape[-1]
        levels = torch.arange(k, dtype=probs.dtype, device=probs.device)
        expected_score = (probs * levels).sum(dim=-1)

        # Bound check, NOT a clamp: s is a convex combination of {0..K-1}.
        tol = 1e-4
        lo, hi = -tol, (k - 1) + tol
        s_min = float(expected_score.detach().min())
        s_max = float(expected_score.detach().max())
        if not (expected_score >= lo).all() or not (expected_score <= hi).all():
            raise RuntimeError(
                f"ScoreHead: expected score escaped the convex hull [0, {k - 1}] "
                f"(observed min={s_min!r}, max={s_max!r}, tol={tol}); "
                "probabilities must be non-negative and sum to 1."
            )
        return DecisionOutput(
            logits=logits,
            probabilities=probs,
            predicted_index=predicted_index,
            expected_score=expected_score,
        )
