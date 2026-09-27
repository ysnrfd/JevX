# Copyright (c) 2026 ysnrfd
"""Adapter for vision encoders (vit/...).

# justification: audit Q7/Q4 — ViT ships a trained pooler (level-4 route,
# preferred via priority 1), but its ``hidden_states[-1]`` is the **pre-final-
# norm** state: vision models disable the last-element tie
# (``@capture_outputs(tie_last_hidden_states=False)``,
# ``models/vit/modeling_vit.py:354``; probe in this
# environment: ``torch.equal(hidden_states[-1], last_hidden_state) is False``).
# The shared extraction therefore reads ``pooler_output`` /
# ``last_hidden_state`` and only reaches ``hidden_states[-1]`` as a last
# resort (documented pre-norm caveat). Inputs are ``pixel_values``
# (``main_input_name``), prepared through an image processor or the
# ``processor(images=...)`` route (audit Q8);
# * the CNN / dense-prediction vision backbones do **not** return a token
#   sequence: ``ConvNextModel``/``ResNetModel``/``MobileNetV2Model`` return
#   ``last_hidden_state`` as a rank-4 ``[B, C, H, W]`` map (stacked 2-D
#   convolutions, no token axis) and ``SegformerModel`` a rank-4
#   ``[B, decoder_channels, H/4, W/4]`` map. ``_normalize_sequence`` flattens
#   such a map to the ``[B, H*W, C]`` token layout torch uses for
#   NCHW-to-token conversions, which is what the pooling contract
#   (``[B, T, D]``) requires.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from .generic import GenericAdapter

__all__ = ["VisionAdapter"]


class VisionAdapter(GenericAdapter):
    """Vision-only stacks: native ``pooler_output`` preferred, patch-level
    fallback pooling ``masked_mean`` (math spec: patch/frame sequences ->
    ``P_mean``/``P_max``/``P_att``)."""

    name = "vision"
    default_pooling = "masked_mean"

    def can_handle(self, model: Any) -> bool:
        """Vision inputs without a text side (multimodal composites go to
        ``MultimodalAdapter``)."""
        report = self.describe_capabilities(model)
        return bool(report.vision and not report.text)

    def _normalize_sequence(
        self, sequence: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Flatten a rank-4 ``[B, C, H, W]`` spatial map to ``[B, H*W, C]``.

        The transformer vision backbones already return a rank-3 token
        sequence, so this is the identity for them. The convolutional and
        dense-prediction ones (convnext, resnet, mobilenet_v2, segformer and
        every sibling exposing a spatial ``last_hidden_state``) return a
        feature map, which the ``[B, T, D]`` pooling contract cannot consume.
        Every spatial position is a valid token there, so a ``[B, T]`` mask is
        dropped rather than passed at the wrong length (``pool`` raises on a
        shape mismatch).

        Args:
            sequence: The tensor selected from the forward output.
            mask: The mask resolved for it, or ``None``.

        Returns:
            ``(sequence, mask)`` with ``sequence`` always rank 3.
        """
        if sequence.dim() == 4:
            batch, channels, height, width = sequence.shape
            sequence = sequence.reshape(batch, channels, height * width).transpose(1, 2)
            return sequence, None
        return sequence, mask
