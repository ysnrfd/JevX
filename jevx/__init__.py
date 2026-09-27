# Copyright (c) 2026 ysnrfd
"""JevX: typed probabilistic decision layer over Hugging Face backbones.

Public API (imported eagerly; ``import jevx`` must succeed cleanly)::

    from jevx import JevX, JevXForDecision, JevXConfig, ...

``JevX`` is an alias of :class:`JevXForDecision` (§45/§63 factory entry:
``JevX.from_pretrained(model_id, ...)``). The Auto layer lives in
:mod:`jevx.auto` (importing it also performs the session-local
``AutoConfig``/``AutoModel`` registration).
"""

from __future__ import annotations

__version__ = "0.1.0"

from .calibration import CalibrationEngine
from .capabilities import JevXCapabilityDetector, UnsupportedRepresentationError
from .configuration_jevx import JevXConfig
from .confidence import ConfidenceEngine
from .conversion import ConversionMode, ConversionReport, EquivalenceKind, convert
from .decisions import build_decision_head, register_decision_head
from .modeling_outputs import DecisionOutput, JevXOutput
from .modeling_jevx import JevX, JevXForDecision, coverage_at_risk, risk_coverage_curve
from .questions import Option
from .representations import build_fusion, pool
from .validation import ValidationEngine

# Re-exported from the adapters registry (after the core names so a registry
# issue cannot mask the version/model exports above).
from .adapters import register_adapter  # noqa: E402

__all__ = [
    "CalibrationEngine",
    "ConfidenceEngine",
    "ConversionMode",
    "ConversionReport",
    "DecisionOutput",
    "EquivalenceKind",
    "JevX",
    "JevXCapabilityDetector",
    "JevXConfig",
    "JevXForDecision",
    "JevXOutput",
    "Option",
    "UnsupportedRepresentationError",
    "ValidationEngine",
    "__version__",
    "build_decision_head",
    "build_fusion",
    "convert",
    "coverage_at_risk",
    "pool",
    "register_adapter",
    "register_decision_head",
    "risk_coverage_curve",
]
