# Copyright (c) 2026 ysnrfd
"""Noul/Binary decision head: a single sigmoid probability."""

from __future__ import annotations

from typing import Any

import torch

from jevx.modeling_outputs import DecisionOutput

from .generic import MLPStatePathway, DecisionHead, register_decision_head

__all__ = ["NoulHead"]


@register_decision_head("noul")
class NoulHead(DecisionHead):
    """Binary ("noul") decision: ``p_yes = sigmoid(fused^T w + b) in (0, 1)``.

    Mathematical contract::

        logit  = w . fused + b            # [B, 1]
        p_yes  = clamp(sigmoid(logit), eps, 1 - eps)  # P(yes) in (0, 1), T4
        y      = 1[p_yes >= 0.5]          # thresholded binary decision

    Numerics note: raw fp32 ``sigmoid`` saturates to *exactly* ``1.0`` for
    logits ``>= ~17`` and to exact ``0.0`` for very large negative logits
    (PROOFS.md Remark 4.1). To keep STORED probabilities strictly inside
    ``(0, 1)`` for EVERY finite logit — T4 / Definition 6b —
    ``build_output`` clamps to ``[eps, 1 - eps]`` with
    ``eps = finfo(dtype).eps`` (the sanctioned implementation invariant; the
    induced bias is bounded by ``eps``). Options/masks do not apply to a
    binary decision and are rejected.

    The state pathway IS the scorer here: a binary decision has no option rows,
    so no interaction residual ``r_k`` exists and ``lambda`` /
    ``conditioning_enabled`` are inert (they are still accepted and validated so
    a head built from one uniform recipe stays constructible). The default
    ``state_pathway="linear"`` keeps ``proj`` an ``nn.Linear(D, 1)`` — the shape
    ``convert()`` transplants a binary classifier into (``heads.noul.proj``).

    Args:
        hidden_size: Input dim ``D``.
        state_pathway: ``"linear"`` (default, ``proj`` is an ``nn.Linear(D, 1)``)
            or ``"mlp"`` (a hidden layer of width ``state_hidden_size``).
        state_hidden_size: Hidden width ``H`` of the ``"mlp"`` pathway.
        state_activation: Elementwise activation of the ``"mlp"`` pathway.
        conditioning_scale_init: Accepted for a uniform recipe; inert (no
            residual).
        conditioning_enabled: Accepted for a uniform recipe; inert (no
            residual).
    """

    def __init__(
        self,
        hidden_size: int,
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
            has_interaction=False,  # a binary decision has no option rows
        )
        self.proj: torch.nn.Module = (
            MLPStatePathway(hidden_size, 1, self.state_hidden_size, state_activation)
            if state_pathway == "mlp"
            else torch.nn.Linear(hidden_size, 1)
        )

    def _state_module(self) -> torch.nn.Module | None:
        """``proj`` is the state pathway of a binary head (see the class docstring)."""
        return self.proj

    def _expected_width(self, option_features: torch.Tensor | None) -> int | None:
        return 1

    def validate_schema(
        self,
        fused: torch.Tensor,
        option_features: torch.Tensor | None = None,
        option_mask: torch.Tensor | None = None,
        num_options: torch.Tensor | int | None = None,
    ) -> None:
        if option_features is not None or option_mask is not None or num_options is not None:
            raise ValueError(
                "NoulHead: emits one binary probability per row — option_features / "
                "option_mask / num_options are not applicable (got "
                f"option_features={'set' if option_features is not None else 'None'}, "
                f"option_mask={'set' if option_mask is not None else 'None'}, "
                f"num_options={'set' if num_options is not None else 'None'})."
            )
        super().validate_schema(fused, option_features, option_mask, num_options)

    def compute_logits(
        self,
        fused: torch.Tensor,
        option_features: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.proj(fused)  # [B, 1]

    def build_output(
        self, logits: torch.Tensor, mask: torch.Tensor | None = None
    ) -> DecisionOutput:
        probs = torch.sigmoid(logits)  # [B, 1]
        # T4 / PROOFS.md Remark 4.1: fp sigmoid saturates to exact 0/1 far
        # from zero (e^{-17} rounds 1 + e^{-z} to 1.0 in binary32); clamp to
        # the dtype interior so stored probabilities satisfy 0 < p < 1 for
        # every finite fp-representable logit. Implementation invariant: the
        # induced bias on p is bounded by finfo(dtype).eps.
        eps = torch.finfo(probs.dtype).eps
        probs = probs.clamp(min=eps, max=1.0 - eps)
        predicted_index = (probs >= 0.5).to(torch.long).reshape(-1)
        return DecisionOutput(
            logits=logits,
            probabilities=probs,
            predicted_index=predicted_index,
        )
