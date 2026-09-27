# Copyright (c) 2026 ysnrfd
"""Question (``E_Q``) and schema (``E_C``) encoders.

* :class:`QuestionEncoder` — encode question strings to ``H_Q [B, D_q]``
  (backbone path or hashed bag-of-tokens fallback) plus a trainable
  residual transform.
* :class:`SchemaEncoder` / :class:`Option` — encode variable-``K`` option
  sets to ``H_C [B, K_max, D_c]`` with a bool mask aligned with
  ``DecisionHead.option_mask``.
"""

from .encoder import QUESTION_METHODS, QuestionEncoder
from .schema import Option, SchemaEncoder

__all__ = ["QUESTION_METHODS", "Option", "QuestionEncoder", "SchemaEncoder"]
