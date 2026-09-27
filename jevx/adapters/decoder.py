# Copyright (c) 2026 ysnrfd
"""Adapter for causal / decoder-only language models (llama/gpt2/qwen/...).

# justification: audit Q7 — no pooler ships with any decoder-only family
# (llama/qwen2/mistral/gemma/phi3/gpt2 all return sequence states only), and
# transformers' own ``GenericForSequenceClassification`` convention scores
# every position then selects the last non-pad one: the pinned JevX
# convention is therefore ``last_valid`` pooling (math spec P4.d: ``P_last``
# is "the natural pooling for causal/decoder-only backbones"). Two further
# audit facts drive the behavior below:

* §5/§8.7 — ``use_cache`` defaults to True and only allocates a KV cache
  extraction never needs: pass ``use_cache=False`` (done in the shared
  extraction path).
* §5 — positions are computed from ``position_ids``; with *left* padding the
  default ``arange`` would shift every real token's position by the pad
  count. Deriving ``position_ids`` cumsum-style from the attention mask makes
  left-pad, right-pad and unpadded runs numerically equivalent (tested in
  ``tests/unit/test_adapters.py``).
"""

from __future__ import annotations

import inspect
from typing import Any, Dict

import torch

from .generic import GenericAdapter

__all__ = ["DecoderAdapter"]


class DecoderAdapter(GenericAdapter):
    """Decoder-only stacks: mask-aware ``last_valid`` pooling, left-pad aware.

    Default pooling: ``"last_valid"`` — ``i_b = max{t : mask[b,t] = 1}``,
    identical for left padding, right padding and no padding (P4.3 padding
    invariance).
    """

    name = "decoder"
    default_pooling = "last_valid"

    def can_handle(self, model: Any) -> bool:
        """Cache-capable causal stack, not encoder-decoder and not vision/audio."""
        report = self.describe_capabilities(model)
        return bool(
            report.decoder
            and not report.encoder_decoder
            and not report.vision
            and not report.audio
        )

    def _postprocess_forward_kwargs(
        self, model: Any, forward_inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Derive ``position_ids`` from the attention mask when absent.

        Only applies when the forward declares ``position_ids`` and an
        ``attention_mask`` is present; ``position_ids = cumsum(mask) - 1``
        (clamped at 0) gives every *real* token position ``0..L-1``
        independent of which side the padding sits on.
        """
        try:
            params = set(inspect.signature(model.forward).parameters)
        except (TypeError, ValueError):  # pragma: no cover - uninspectable
            return forward_inputs
        mask = forward_inputs.get("attention_mask")
        if "position_ids" in params and "position_ids" not in forward_inputs and isinstance(
            mask, torch.Tensor
        ) and mask.dim() == 2:
            forward_inputs = dict(forward_inputs)
            forward_inputs["position_ids"] = (mask.long().cumsum(dim=-1) - 1).clamp(min=0)
        return forward_inputs
