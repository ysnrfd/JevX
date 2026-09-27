# Copyright (c) 2026 ysnrfd
"""Capability detection: structured reports and the representation error type.

``JevXCapabilityDetector`` performs *introspection only* (forward signature,
config flags, module tree, output-class annotation) — no forward pass — and
:meth:`JevXCapabilityDetector.probe` optionally confirms the contract with a
single cautious ``eval()``/``no_grad`` forward. See
``reports/TRANSFORMERS_SOURCE_AUDIT.md`` for the v5.16.1 facts each rule
encodes.

Direction detection is read from four independent structural signals, never
from a model name: ``config.is_decoder``; a module tagged
``is_causal is True`` (:func:`causal_attention_present`); a recurrent /
state-space decoding-state output field (:data:`RECURRENT_STATE_FIELDS`); and
the legacy cache heuristic (a ``past_key_values``-bearing output with neither
a pooler nor an ``encoder`` submodule).
``reports/ARCHITECTURE_BREADTH.md`` records the measurement that
motivated each rule.
"""

from .detector import (
    NATIVE_REPRESENTATION_FIELDS,
    RECURRENT_STATE_FIELDS,
    CapabilityReport,
    JevXCapabilityDetector,
    causal_attention_present,
    pooler_present,
)
from .errors import UnsupportedRepresentationError

__all__ = [
    "NATIVE_REPRESENTATION_FIELDS",
    "RECURRENT_STATE_FIELDS",
    "CapabilityReport",
    "JevXCapabilityDetector",
    "UnsupportedRepresentationError",
    "causal_attention_present",
    "pooler_present",
]
