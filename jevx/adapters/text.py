# Copyright (c) 2026 ysnrfd
"""Thin text-input helpers shared by the adapters.

Audit Q8: text-only families (bert/t5/bart/llama/...) have **no** processor
mapping — inputs come from ``AutoTokenizer`` via one
``tokenizer(texts, return_tensors="pt", padding=True)`` call. These helpers
centralize that call and the raw-modality keys that *do* require a
processor/feature extractor.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

__all__ = ["TEXT_PROCESSOR_KEYS", "is_text_state", "tokenize_texts"]

#: State-dict keys that carry *raw* modality payloads requiring a processor
#: (``ProcessorMixin.__call__(images=..., text=..., videos=..., audio=...)``,
#: audit §4/Q8). When any is present, ``prepare_inputs`` routes through the
#: processor instead of the plain dict passthrough.
TEXT_PROCESSOR_KEYS: Tuple[str, ...] = ("images", "image", "videos", "video", "audio")


def is_text_state(state: Any) -> bool:
    """True iff ``state`` is a str or a non-empty list/tuple of strs."""
    if isinstance(state, str):
        return True
    if isinstance(state, (list, tuple)):
        return len(state) > 0 and all(isinstance(item, str) for item in state)
    return False


def tokenize_texts(
    state: Any,
    tokenizer: Any,
    *,
    max_length: int | None = None,
) -> Dict[str, Any]:
    """Tokenize raw text state into model kwargs.

    Args:
        state: ``str`` or ``list[str]``/``tuple[str, ...]``.
        tokenizer: A HF tokenizer (``AutoTokenizer`` output). ``None`` raises.
        max_length: Optional truncation length (enables ``truncation=True``).

    Returns:
        ``dict`` batch (``input_ids`` ``[B, T]``, ``attention_mask``
        ``[B, T]``, plus family-specific fields such as ``token_type_ids``).

    Raises:
        ValueError: ``tokenizer`` is None, the batch is empty, or an entry is
            not a non-empty string.
    """
    if tokenizer is None:
        raise ValueError(
            "tokenize_texts: tokenizer=None — raw text state requires a tokenizer "
            "(text-only families have no processor mapping; resolve one with "
            "AutoTokenizer.from_pretrained, audit Q8)."
        )
    if isinstance(state, str):
        texts = [state]
    elif isinstance(state, (list, tuple)):
        texts = list(state)
    else:
        raise TypeError(f"tokenize_texts: expected str or list of str, got {type(state).__name__}.")
    if len(texts) == 0:
        raise ValueError("tokenize_texts: got an empty text batch.")
    for i, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"tokenize_texts: texts[{i}] is not a non-empty string — every state in the "
                "batch must carry content."
            )
    call_kwargs: Dict[str, Any] = {"return_tensors": "pt", "padding": True}
    if max_length is not None:
        call_kwargs.update(truncation=True, max_length=max_length)
    batch = tokenizer(state, **call_kwargs)
    return dict(batch)
