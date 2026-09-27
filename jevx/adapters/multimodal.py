# Copyright (c) 2026 ysnrfd
"""Adapter for composite multimodal models (clip/llava/qwen3_5/...).

# justification: the generic route works mechanically but its *semantics*
# are a pinned convention, not a trained canonical vector (compatibility
# matrix level-2 rows: "interleaved multimodal tokens", "pre- vs
# post-projection contrastive spaces"). This subclass pins them:

* input prep uses the probe-verified unified call
  ``processor(images=..., text=..., audio=..., return_tensors="pt")``
  (``ProcessorMixin.__call__`` signature, audit §4/Q8);
* per-modality features are **kept** in ``Representation.raw_output`` (CLIP's
  ``text_embeds``/``image_embeds``/``logits_*``, LLaVA's
  ``image_hidden_states`` — never dropped by extraction);
* for joint contrastive outputs the semantic-field priority selects
  ``sentence_embedding`` → ``text_embeds`` → ``image_embeds`` (text wins when
  several exist — documented choice, "post-projection for cross-modal
  comparability" note in the matrix);
* pooling over an interleaved image/text sequence covers **all** mask-valid
  positions (image placeholder spans included): the token-selection
  convention the matrix demands be pinned. ``JevXCapabilityDetector``
  reports ``jevx_conversion="experimental"`` for this case;
* **TEXT-ONLY states** (``input_ids``/``attention_mask``, no pixel/audio
  arrays) derive ``position_ids = cumsum(mask) - 1`` when the backbone
  declares them (see :meth:`MultimodalAdapter._postprocess_forward_kwargs`) —
  the padding-side-consistent rule the decoder path already pins (§5).
"""

from __future__ import annotations

import inspect
from typing import Any, Dict

import torch

from .generic import GenericAdapter

__all__ = ["MultimodalAdapter"]

#: Modality array keys whose presence means "this is NOT a text-only call":
#: the multimodal forward computes its own 3-D/multimodal ``position_ids``
#: (e.g. ``Qwen3_5Model.compute_3d_position_ids``) and a hand-derived 2-D
#: mask-cumsum would be wrong there.
_NON_TEXT_INPUT_KEYS = (
    "pixel_values",
    "pixel_values_videos",
    "input_features",
    "input_values",
)


class MultimodalAdapter(GenericAdapter):
    """Composite text+vision models; conventions pinned in the docstring."""

    name = "multimodal"
    default_pooling = "masked_mean"

    def can_handle(self, model: Any) -> bool:
        """Both a text and a vision input side (clip/llava families)."""
        report = self.describe_capabilities(model)
        return bool(report.vision and report.text)

    def _postprocess_forward_kwargs(
        self, model: Any, forward_inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Text-only: derive padding-consistent ``position_ids`` from the mask.

        Why (Qwen3.5 selection quality, Known finding #2): on a text-only
        call the multimodal wrapper does NOT derive positions from the
        ``attention_mask`` — ``Qwen3_5Model.compute_3d_position_ids`` returns
        ``None`` without multimodal inputs and
        ``Qwen3_5TextModel.forward`` defaults to ``arange`` over COLUMNS
        (``modeling_qwen3_5.py:1174-1177``), so a *left*-padded batch shifts
        every real token's absolute RoPE position by the pad count. Deriving
        ``position_ids = cumsum(mask) - 1`` (clamped at 0) gives each real
        token positions ``0..L-1`` regardless of which side the padding sits
        on — the exact rule ``DecoderAdapter`` already applies (§5 /
        ``docs/HYBRID_ATTENTION_ANALYSIS.md`` §7.5), and the backbone
        ``Qwen3_5Model.forward`` consumes ``position_ids`` when given.

        Measured on Qwen3.5-0.8B (probe, battery-6b/6c protocol): left-vs-right
        pooled 0.125 → 0.0625 (sequence-level 0.0465 → 0.0409, below the
        backbone-alone 0.0465); right-pad numbers unchanged
        (0.125 / 0.03125 — right-padded content positions already matched);
        all values ≤ the 2·ulp bf16 noise floor 0.25.

        Guards (never touches other call shapes): skipped when any modality
        array is present (multimodal forwards compute their own 3-D
        positions), when ``input_ids`` is absent, when there is no 2-D
        ``attention_mask``, when the caller already supplied ``position_ids``,
        or when ``model.forward`` does not declare ``position_ids``.
        Unpadded inputs are unaffected: ``cumsum`` on an all-ones mask *is*
        ``arange``, so extraction stays bitwise identical.
        """
        if any(key in forward_inputs for key in _NON_TEXT_INPUT_KEYS):
            return forward_inputs
        if "input_ids" not in forward_inputs or "position_ids" in forward_inputs:
            return forward_inputs
        mask = forward_inputs.get("attention_mask")
        if not isinstance(mask, torch.Tensor) or mask.dim() != 2:
            return forward_inputs
        try:
            params = inspect.signature(model.forward).parameters
        except (TypeError, ValueError):  # pragma: no cover - uninspectable
            return forward_inputs
        if "position_ids" not in params:
            return forward_inputs
        forward_inputs = dict(forward_inputs)
        forward_inputs["position_ids"] = (mask.long().cumsum(dim=-1) - 1).clamp(min=0)
        return forward_inputs
