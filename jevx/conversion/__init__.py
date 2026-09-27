# Copyright (c) 2026 ysnrfd
"""JevX conversion subsystem: the weight-conversion principle, made executable.

JevX never claims a lossless weight conversion of a checkpoint into a
decision system. Instead every conversion is one of four explicitly
distinguishable operations (:class:`ConversionMode`), and every equivalence
claim is one that was actually measured (:class:`EquivalenceKind`, honesty
invariants enforced by :class:`ConversionReport`):

* **representation reuse** — backbone weights reused verbatim, JevX heads
  freshly initialized (`convert()` on an id or in-memory module);
* **weight transplant / head attachment** — narrow, shape-checked grafts of
  a source task head into an existing JevX head parameterization
  (`transplant_classifier_head`, `transplant_state_encoder_proj`);
* **architectural conversion** — reserved constant for future structural
  rewrites, never auto-assigned today;
* **functional / behavioral equivalence** — measured by executable checkers
  (`check_base_equivalence`, `check_representation_equivalence`,
  `check_decision_roundtrip`), never asserted.

See ``docs/ARCHITECTURE.md`` section 5 (reuse vs transplant) and the module
docstrings for the full semantic contracts.
"""

from .converter import convert
from .equivalence import (
    PROBE_TEXT,
    check_base_equivalence,
    check_decision_roundtrip,
    check_functional_equivalence,
    check_representation_equivalence,
    derive_probe_inputs,
)
from .report import ConversionMode, ConversionReport, EquivalenceKind
from .transplant import (
    CompatReport,
    probe_head_compat,
    transplant_classifier_head,
    transplant_state_encoder_proj,
)

__all__ = [
    "PROBE_TEXT",
    "CompatReport",
    "ConversionMode",
    "ConversionReport",
    "EquivalenceKind",
    "check_base_equivalence",
    "check_decision_roundtrip",
    "check_functional_equivalence",
    "check_representation_equivalence",
    "convert",
    "derive_probe_inputs",
    "probe_head_compat",
    "transplant_classifier_head",
    "transplant_state_encoder_proj",
]
