# Copyright (c) 2026 ysnrfd
"""Adapter for bidirectional text encoders (bert/roberta/albert/distilbert/...).

# justification: generic extraction already works for these models; this
# subclass exists because the *pooling default* must be family-specific —
# audit Q7 shows trained poolers ship only on bert/roberta/albert (preferred
# automatically via priority 1), while distilbert/deberta-v2 expose only
# ``last_hidden_state`` where the established convention for bidirectional
# stacks is first-token (CLS) pooling (math spec Definition 4: bidirectional
# encoders -> ``P_first``/``P_mean``; compatibility matrix "deberta-v2" /
# "distilbert" rows: "last_hidden_state -> CLS/mean").
"""

from __future__ import annotations

from typing import Any

from .generic import GenericAdapter

__all__ = ["EncoderAdapter"]


class EncoderAdapter(GenericAdapter):
    """Bidirectional, non-multimodal encoder stacks.

    Default pooling: ``"cls"`` (mask-aware first-valid token), used only when
    no native ``pooler_output`` exists. Extraction, field priority and input
    preparation are inherited unchanged from :class:`GenericAdapter`.
    """

    name = "encoder"
    default_pooling = "cls"

    def can_handle(self, model: Any) -> bool:
        """Bidirectional text encoder: encoder direction, no vision/audio/enc-dec."""
        report = self.describe_capabilities(model)
        return bool(
            report.encoder
            and not report.decoder
            and not report.encoder_decoder
            and not report.vision
            and not report.audio
        )
