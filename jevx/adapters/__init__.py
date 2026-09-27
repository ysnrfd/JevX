# Copyright (c) 2026 ysnrfd
"""Input/representation adapters: raw state ``S`` → model kwargs + ``H_S``.

Public surface:

* :class:`Representation` / :class:`BaseAdapter` — the pinned contract;
* :class:`GenericAdapter` — introspection-driven implementation for any
  model;
* family subclasses — ``EncoderAdapter`` (pooler-first bidirectional),
  ``DecoderAdapter`` (causal ``last_valid``, left-pad aware,
  ``use_cache=False``), ``EncoderDecoderAdapter`` (``encoder_state`` /
  ``decoder_state`` / ``encoder_decoder_fusion``), ``VisionAdapter``,
  ``AudioAdapter``, ``MultimodalAdapter`` (composite);
* :func:`register_adapter` / :func:`resolve_adapter` /
  :func:`available_adapters` — the registry;
* text helpers (:mod:`jevx.adapters.text`) for tokenizer-based input prep.
"""

from .audio import AudioAdapter
from .base import (
    FORWARD_CONTROL_FIELDS,
    SEMANTIC_FIELDS,
    BaseAdapter,
    Representation,
    forward_kwargs,
    observed_output_fields,
)
from .decoder import DecoderAdapter
from .encoder import EncoderAdapter
from .encoder_decoder import ENC_DECODER_MODES, EncoderDecoderAdapter
from .generic import GenericAdapter
from .multimodal import MultimodalAdapter
from .registry import (
    BUILTIN_ADAPTER_CHAIN,
    USER_ADAPTERS,
    available_adapters,
    register_adapter,
    resolve_adapter,
)
from .text import TEXT_PROCESSOR_KEYS, is_text_state, tokenize_texts
from .vision import VisionAdapter

__all__ = [
    "BUILTIN_ADAPTER_CHAIN",
    "ENC_DECODER_MODES",
    "FORWARD_CONTROL_FIELDS",
    "SEMANTIC_FIELDS",
    "TEXT_PROCESSOR_KEYS",
    "USER_ADAPTERS",
    "AudioAdapter",
    "BaseAdapter",
    "DecoderAdapter",
    "EncoderAdapter",
    "EncoderDecoderAdapter",
    "GenericAdapter",
    "MultimodalAdapter",
    "Representation",
    "VisionAdapter",
    "available_adapters",
    "forward_kwargs",
    "is_text_state",
    "observed_output_fields",
    "register_adapter",
    "resolve_adapter",
    "tokenize_texts",
]
