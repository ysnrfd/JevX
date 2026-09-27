# Copyright (c) 2026 ysnrfd
"""Thin ``AutoJevX`` factory — no forking of transformers Auto internals.

Session-local registration (audit Q2/Q3): ``AutoConfig.register("jevx",
JevXConfig)`` makes ``AutoConfig.from_pretrained`` on a saved JevX
directory resolve OUR config class without ``trust_remote_code`` /
``auto_map``. Registration is **process-local** — it must run again in
every new process (which importing :mod:`jevx.auto` does), and it is
idempotent via ``try/except ValueError`` (a second registration raises).
``AutoModel.register(JevXConfig, JevXForDecision)`` is registered as a
documented extra so plain ``AutoModel.from_pretrained(jevx_dir)`` also
works within the session.

``trust_remote_code`` is never implied: ``AutoJevX.from_pretrained``
forwards whatever the caller passes (default ``False``) — on Windows the
prompt path does not exist (no SIGALRM), so an explicit ``True`` is the
only way to opt in (audit §2).
"""

from __future__ import annotations

from typing import Any

from transformers import AutoConfig, AutoModel

from ..configuration_jevx import JevXConfig
from ..modeling_jevx import JevXForDecision

__all__ = ["AutoJevX"]

# Session-local registration (documented above). Both are idempotent:
# re-registration raises, which we swallow deliberately.
try:
    AutoConfig.register("jevx", JevXConfig)
except ValueError:  # already registered in this process
    pass
try:
    AutoModel.register(JevXConfig, JevXForDecision)
except Exception:  # noqa: BLE001 - already registered / mapping constraints
    pass


class AutoJevX:
    """Factory resolving JevX checkpoints vs. base checkpoints.

    ``model_type == "jevx"`` → direct reload of a JevX checkpoint;
    anything else → wrap the base model (the §45/§63 factory path of
    :meth:`JevXForDecision.from_pretrained`).
    """

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path: Any, **kwargs: Any) -> JevXForDecision:
        """Load a JevX wrapper from ``model_id``.

        Args:
            pretrained_model_name_or_path: Hub id or local directory — a
                saved JevX checkpoint (``model_type == "jevx"``) or a base
                checkpoint to wrap.
            **kwargs: Forwarded to
                :meth:`JevXForDecision.from_pretrained` — the documented
                factory keywords (``decision_type``, ``trust_remote_code``,
                ``tokenizer``, ``processor``, ``adapter``) plus loader /
                config fields.

        Returns:
            A :class:`JevXForDecision` instance (``JevX`` alias).
        """
        return JevXForDecision.from_pretrained(pretrained_model_name_or_path, **kwargs)
