# Copyright (c) 2026 ysnrfd
"""Confidence engine package.

``ConfidenceEngine(method)`` resolves a registered method name
(``max_probability``, ``margin``, ``entropy``, ``energy``, ``learned``) and
returns per-question confidence in ``[0, 1]`` (energy is raw until fitted —
see ``jevx.confidence.base`` module docstring).
"""

from .base import (
    CONFIDENCE_REGISTRY,
    ConfidenceEngine,
    EnergyConfidence,
    EntropyConfidence,
    LearnedConfidence,
    MarginConfidence,
    MaxProbabilityConfidence,
    available_confidence_methods,
    get_confidence_method,
    register_confidence_method,
)

__all__ = [
    "CONFIDENCE_REGISTRY",
    "ConfidenceEngine",
    "EnergyConfidence",
    "EntropyConfidence",
    "LearnedConfidence",
    "MarginConfidence",
    "MaxProbabilityConfidence",
    "available_confidence_methods",
    "get_confidence_method",
    "register_confidence_method",
]
