# Copyright (c) 2026 ysnrfd
"""Executable numerical verification suite (JevX Phase 24 + section 27-executable + section 68).

Every test in this package numerically instantiates a structural claim proven
on paper in ``docs/PROOFS.md`` (Theorems T1-T8) or specified in
``docs/MATHEMATICAL_SPECIFICATION.md``, against the REAL implementation. The
suite checks that the implementation SATISFIES each theorem's statement; it
does not re-prove the mathematics.

* ``test_theorem_verification.py`` - T1-T8 plus the padding/masking and
  risk-coverage structural claims.
* ``test_numerical_stability.py`` - section 38 numerical stability of the
  head stack under extreme logits and low-precision inputs.

Results table: ``reports/MATHEMATICAL_VERIFICATION.md``.
"""
