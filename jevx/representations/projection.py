# Copyright (c) 2026 ysnrfd
"""Dimension projection between representation widths (``D_model -> D_decision``).

The backbone emits hidden states of width ``d`` while the JevX decision layer
operates at its own width ``D`` (``JevXConfig.decision_hidden_size`` /
``question_hidden_size`` / ``schema_hidden_size``). :class:`Projection`
realizes the ``W · x (+ b)`` maps denoted by ``P_*`` / ``phi_*`` in the
mathematical specification (Definition 4/5: "all projections map into a
common width ``d``"), either as a single affine map or as a 2-layer MLP.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["Projection"]


class Projection(nn.Module):
    """Map the last axis from ``in_dim`` to ``out_dim``.

    Math (rank-agnostic, applies to ``[B, D_in]`` and ``[B, K, D_in]`` alike):

    * ``mlp=False`` — ``y = LN(W x + b)``, ``W in R^{D_out x D_in}``
    * ``mlp=True``  — ``y = LN(W_2 · GELU(W_1 x + b_1) + b_2)`` (2-layer MLP)

    with ``LN`` the identity when ``layer_norm=False``.

    Args:
        in_dim: Expected size of the last axis of the input.
        out_dim: Size of the last axis of the output.
        layer_norm: Append a (learnable) ``LayerNorm(out_dim)`` after the map
            (default True — stabilizes widths coming from different
            backbones).
        mlp: Use the 2-layer MLP form instead of a single linear map.

    Raises:
        ValueError: A dimension is not a positive int.
    """

    def __init__(self, in_dim: int, out_dim: int, layer_norm: bool = True, mlp: bool = False) -> None:
        super().__init__()
        for label, value in (("in_dim", in_dim), ("out_dim", out_dim)):
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{label}: expected a positive int, got {value!r}.")
        if not isinstance(layer_norm, bool) or not isinstance(mlp, bool):
            raise TypeError("layer_norm and mlp must be bools.")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mlp = mlp
        if mlp:
            self.proj: nn.Module = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.GELU(),
                nn.Linear(out_dim, out_dim),
            )
        else:
            self.proj = nn.Linear(in_dim, out_dim)
        self.norm: nn.Module = nn.LayerNorm(out_dim) if layer_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project ``x [..., in_dim]`` to ``[..., out_dim]``."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Projection: expected a torch.Tensor, got {type(x).__name__}.")
        if x.shape[-1] != self.in_dim:
            raise ValueError(
                f"Projection: expected last dimension {self.in_dim}, got {x.shape[-1]} "
                f"(input shape {tuple(x.shape)})."
            )
        return self.norm(self.proj(x))

    def extra_repr(self) -> str:
        return f"in_dim={self.in_dim}, out_dim={self.out_dim}, mlp={self.mlp}"
