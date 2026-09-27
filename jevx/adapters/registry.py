# Copyright (c) 2026 ysnrfd
"""Adapter registration and resolution.

Resolution order (:func:`resolve_adapter`):

1. **user-registered** adapters, in registration order (``@register_adapter``)
   — the first whose ``can_handle(model)`` is True wins;
2. the built-in **introspection chain**, most specific first
   (multimodal → encoder_decoder → vision → audio → decoder → encoder);
3. :class:`~jevx.adapters.generic.GenericAdapter` — ``can_handle`` is
   always True, so resolution never fails.

No model-name matching happens anywhere: every ``can_handle`` is
capability-introspection only (no forward pass).
"""

from __future__ import annotations

from typing import Any, List, Type

from .audio import AudioAdapter
from .base import BaseAdapter
from .decoder import DecoderAdapter
from .encoder import EncoderAdapter
from .encoder_decoder import EncoderDecoderAdapter
from .generic import GenericAdapter
from .multimodal import MultimodalAdapter
from .vision import VisionAdapter

__all__ = [
    "BUILTIN_ADAPTER_CHAIN",
    "USER_ADAPTERS",
    "available_adapters",
    "register_adapter",
    "resolve_adapter",
]

#: Built-in chain, most specific first; ends with the universal fallback.
BUILTIN_ADAPTER_CHAIN: List[Type[BaseAdapter]] = [
    MultimodalAdapter,
    EncoderDecoderAdapter,
    VisionAdapter,
    AudioAdapter,
    DecoderAdapter,
    EncoderAdapter,
    GenericAdapter,
]

#: User-registered adapter classes (checked before the built-in chain).
USER_ADAPTERS: List[Type[BaseAdapter]] = []


def register_adapter(cls: Type[BaseAdapter]) -> Type[BaseAdapter]:
    """Class decorator: register a custom adapter with highest priority.

    Example:
        >>> @register_adapter
        ... class MyAdapter(GenericAdapter):
        ...     name = "my.adapter"
        ...     def can_handle(self, model): ...
    """
    if not isinstance(cls, type) or not issubclass(cls, BaseAdapter):
        raise TypeError(
            f"register_adapter: expected a BaseAdapter subclass, got {cls!r}."
        )
    if any(existing is cls for existing in USER_ADAPTERS):
        return cls  # idempotent re-registration
    for existing in USER_ADAPTERS:
        if getattr(existing, "name", None) == cls.name:
            raise ValueError(
                f"register_adapter: an adapter named {cls.name!r} is already registered "
                f"({existing.__name__}); choose a different name."
            )
    USER_ADAPTERS.append(cls)
    return cls


def resolve_adapter(model: Any) -> BaseAdapter:
    """Return the adapter instance whose ``can_handle(model)`` matches first.

    User-registered adapters win over the built-in chain; the generic
    fallback guarantees a result.

    Raises:
        TypeError: No adapter class in the chain is a ``BaseAdapter``
            (programming error in a user registration).
        RuntimeError: Unreachable defensive path — no adapter matched.
    """
    for cls in [*USER_ADAPTERS, *BUILTIN_ADAPTER_CHAIN]:
        if not isinstance(cls, type) or not issubclass(cls, BaseAdapter):
            raise TypeError(
                f"resolve_adapter: registered adapter {cls!r} is not a BaseAdapter subclass."
            )
        adapter = cls()
        if adapter.can_handle(model):
            return adapter
    raise RuntimeError(  # pragma: no cover - GenericAdapter.can_handle is always True
        "resolve_adapter: no adapter matched and the generic fallback was skipped — "
        "this indicates a broken user-registered chain."
    )


def available_adapters() -> List[str]:
    """Adapter names in resolution order (user-registered first)."""
    seen: List[str] = []
    for cls in [*USER_ADAPTERS, *BUILTIN_ADAPTER_CHAIN]:
        name = getattr(cls, "name", None)
        if isinstance(name, str) and name not in seen:
            seen.append(name)
    return seen
