# Copyright (c) 2026 ysnrfd
"""Structured error raised when no representation can be extracted from a model.

The exception carries machine-readable fields so callers (and the Validation
Engine) can report *what was seen*, *what was tried*, and *what to do next*
without parsing the message string. The message itself is always
human-readable and actionable.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional


class UnsupportedRepresentationError(Exception):
    """No JevX representation could be produced for ``model_name``.

    Args:
        model_name: Checkpoint id or model class name (best effort).
        architecture: ``config.architectures[0]`` when available, else the
            model class name; ``None`` only when neither exists.
        detected_outputs: Output field names observed on the model's output
            (or the best-effort output class when no forward ran).
        attempted_strategies: Representation strategies tried, in order.
        failed_strategies: Strategies that failed, with a short reason each
            (``"native: no pooler_output in output"`` style).
        suggestion: Actionable next step for the user.
        message: Optional pre-built message; a structured message is derived
            from the other fields when omitted.

    Attributes mirror the arguments one-to-one, so
    ``except UnsupportedRepresentationError as e: e.attempted_strategies``
    is always populated.
    """

    def __init__(
        self,
        model_name: Any,
        architecture: Any,
        detected_outputs: Iterable[Any],
        attempted_strategies: Iterable[Any],
        failed_strategies: Iterable[Any],
        suggestion: Any,
        message: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.architecture = architecture
        self.detected_outputs = list(detected_outputs)
        self.attempted_strategies = list(attempted_strategies)
        self.failed_strategies = list(failed_strategies)
        self.suggestion = suggestion
        if message is None:
            message = (
                f"Unsupported representation for model {model_name!r} "
                f"(architecture={architecture!r}): "
                f"detected outputs {self.detected_outputs}; "
                f"attempted strategies {self.attempted_strategies}; "
                f"failed: {self.failed_strategies}. "
                f"Suggestion: {suggestion}"
            )
        self.message = message
        super().__init__(message)
