# Copyright (c) 2026 ysnrfd
"""Adapter for audio encoders (wav2vec2/ast/whisper encoders/...).

# justification: audit Q7 — no audio family except AST ships a pooler, and
# wav2vec2's own ``ForSequenceClassification`` head masks then **mean-pools
# over time** (``models/wav2vec2/modeling_wav2vec2.py:1730-1735``): the pinned
# JevX convention for frame sequences is therefore ``masked_mean`` over the
# time axis (math spec: frame sequences -> ``P_mean``). Input contract (audit
# Q8 / compatibility matrix "audio" rows):

* ``input_values`` (raw waveform features; ``main_input_name``) or
  ``input_features`` (mel spectrograms, e.g. whisper's fixed 3000 frames) —
  a raw tensor state maps onto ``main_input_name``;
* frame masks: an ``attention_mask`` at the *input* length must be re-aligned
  to the downsampled frame axis before pooling — wav2vec2 convolves the time
  axis ~320x and exposes ``_get_feature_vector_attention_mask`` for exactly
  this (``modeling_wav2vec2.py:964``); the shared mask resolver calls it
  duck-typed and *drops* the mask when no re-alignment hook exists (never a
  length-mismatched mask into ``pool``);
* preprocessing goes through ``AutoProcessor``/``AutoFeatureExtractor`` —
  AST has **no** ``PROCESSOR_MAPPING`` entry (``auto_mappings.py:1022``,
  audit §4/Q8), so feature-extractor-style callables are supported
  positionally.
"""

from __future__ import annotations

from typing import Any, Dict

from .generic import GenericAdapter

__all__ = ["AudioAdapter"]


class _PaddedAudioProcessor:
    """Proxy that injects ``padding=True`` into a processor/FE call.

    justification (modality-benchmark EXP-A): wav2vec2-style feature
    extractors default to ``padding=False``, so the generic
    ``processor(payload, return_tensors="pt")`` call fails on a BATCH of
    variable-length waveforms with
    ``ValueError: Unable to convert output 'input_values' ... inhomogeneous
    shape ... You can try: 1. Use padding=True`` (probe, transformers
    5.16.1). ``padding=True`` pads to the longest item of the batch with
    zeros - exactly what the frame-mask resolver and the masked-mean pooling
    convention expect. Single-item payloads are unaffected (nothing to pad).
    If a callable rejects the keyword (non-standard FE), the call is retried
    once without it so exotic families keep their previous behaviour.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if "padding" in kwargs:
            return self._inner(*args, **kwargs)
        try:
            return self._inner(*args, padding=True, **kwargs)
        except TypeError:
            return self._inner(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        # sub-object detection (tokenizer/image_processor/feature_extractor/
        # video_processor) and every other attribute pass through untouched.
        return getattr(self._inner, name)


class AudioAdapter(GenericAdapter):
    """Audio stacks: ``masked_mean`` over frames as the family convention."""

    name = "audio"
    default_pooling = "masked_mean"

    @staticmethod
    def _call_processor(processor: Any, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generic processor routing, with batched audio padded (see
        ``_PaddedAudioProcessor`` for the probe and justification)."""
        if "audio" not in state and "audios" not in state:
            return GenericAdapter._call_processor(processor, state)
        return GenericAdapter._call_processor(
            _PaddedAudioProcessor(processor), state
        )

    def can_handle(self, model: Any) -> bool:
        """Audio inputs without a text side (whisper-style enc-dec is claimed
        by ``EncoderDecoderAdapter`` first in the resolution chain)."""
        report = self.describe_capabilities(model)
        return bool(report.audio and not report.text)
