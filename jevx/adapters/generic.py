# Copyright (c) 2026 ysnrfd
"""Model-agnostic :class:`~jevx.adapters.base.BaseAdapter` implementation.

``GenericAdapter`` contains **no per-family branches** (no ``if bert`` /
``elif llama`` chains): it reads capabilities from
:class:`~jevx.capabilities.JevXCapabilityDetector`, filters inputs against
``model.forward``'s signature, and extracts representations through the
shared field-priority logic in :class:`BaseAdapter`. Family-specific
subclasses exist only where this generic route genuinely fails (see their
``# justification`` comments).
"""

from __future__ import annotations

import inspect
from typing import Any, Dict

import torch

from .base import FORWARD_CONTROL_FIELDS, BaseAdapter
from .text import TEXT_PROCESSOR_KEYS, is_text_state, tokenize_texts

__all__ = ["GenericAdapter"]


class GenericAdapter(BaseAdapter):
    """Introspection-driven adapter usable with any HF-style model.

    ``prepare_inputs`` accepts:

    * ``str`` / ``list[str]`` — tokenized via ``tokenizer`` (required);
    * ``dict`` / ``BatchFeature`` — validated against ``model.forward``'s
      declared parameters (+ output-control flags) when ``model=`` is given;
      raw modality payloads (``images``/``audio``/``videos`` keys) are routed
      through ``processor`` following the probe-verified
      ``processor(images=..., text=..., audio=..., return_tensors="pt")``
      pattern (audit Q8), with a positional fallback for feature-extractor-
      style callables (AST: ``AutoFeatureExtractor``, audit §4);
    * ``torch.Tensor`` — mapped onto ``model.main_input_name`` (requires
      ``model=``).

    ``extract_representation`` runs the shared priority list (see the module
    docstring of ``base``), passes ``use_cache=False`` whenever the forward
    can receive it — a declared ``use_cache`` parameter OR ``**kwargs``
    (audit §5/§8.7; kwargs-only wrappers such as ``Qwen3_5Model`` would
    otherwise build a ``DynamicCache`` from the config default), and only
    ever requests ``output_hidden_states`` when asked (audit Q4).
    """

    name = "generic"
    default_pooling = "masked_mean"

    def can_handle(self, model: Any) -> bool:
        """Fallback adapter: always True (resolution order makes it last)."""
        return True

    # -- input preparation ---------------------------------------------------

    @staticmethod
    def _call_processor(processor: Any, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run the processor over raw modality payloads + leftover fields."""
        images = state.pop("images", state.pop("image", None))
        videos = state.pop("videos", state.pop("video", None))
        audio = state.pop("audio", None)
        text = state.pop("text", state.pop("texts", None))
        sub_objects = ("tokenizer", "image_processor", "feature_extractor", "video_processor")
        if any(hasattr(processor, attr) for attr in sub_objects):
            # ProcessorMixin.__call__(images=None, text=None, videos=None,
            # audio=None, **kw) -> BatchFeature (audit §4/Q8 probe).
            batch = processor(
                images=images, text=text, videos=videos, audio=audio, return_tensors="pt"
            )
        else:
            # Feature-extractor / image-processor style: single payload,
            # positional (e.g. ASTFeatureExtractor(raw_speech, ...)).
            payload = images if images is not None else (audio if audio is not None else text)
            if payload is None:
                raise ValueError(
                    "prepare_inputs: processor state contains no raw payload "
                    "(images/audio/text)."
                )
            batch = processor(payload, return_tensors="pt")
        result = dict(batch)
        result.update(state)  # already-tensor fields (e.g. pixel_values) win
        return result

    @staticmethod
    def _validate_keys(model: Any, keys: Any, source: str) -> None:
        """Reject state keys the model's forward does not declare."""
        try:
            sig = inspect.signature(model.forward)
        except (TypeError, ValueError):  # pragma: no cover - uninspectable
            return
        explicit = {
            name
            for name, p in sig.parameters.items()
            if name != "self"
            and p.kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        allowed = explicit | FORWARD_CONTROL_FIELDS
        unknown = sorted(set(keys) - allowed)
        if unknown:
            raise ValueError(
                f"prepare_inputs: {source} key(s) {unknown} are not accepted by "
                f"{type(model).__name__}.forward; declared inputs are "
                f"{sorted(explicit)} (output-control flags {sorted(FORWARD_CONTROL_FIELDS)} "
                "are also allowed)."
            )

    def prepare_inputs(
        self,
        state: Any,
        *,
        tokenizer: Any = None,
        processor: Any = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Turn raw state into validated model kwargs.

        Keyword extras are merged into the result (e.g. ``model=`` for
        signature validation; any other value becomes an extra model input).

        Raises:
            TypeError: ``state`` is not a supported type.
            ValueError: Missing tokenizer/processor/model for the required
                path, a processor payload is absent, or dict keys are not
                accepted by ``model.forward``.
        """
        model = kwargs.pop("model", None)
        if is_text_state(state):
            result = tokenize_texts(state, tokenizer)
        elif isinstance(state, torch.Tensor):
            if model is None:
                raise ValueError(
                    "prepare_inputs: a raw tensor state requires model= to resolve "
                    "model.main_input_name (e.g. input_values for wav2vec2, pixel_values "
                    "for ViT)."
                )
            main_input = getattr(model, "main_input_name", None)
            if not isinstance(main_input, str) or not main_input:
                raise ValueError(
                    f"prepare_inputs: {type(model).__name__} exposes no main_input_name for "
                    "tensor state."
                )
            result = {main_input: state}
        elif isinstance(state, dict):
            result = dict(state)
            if any(key in result for key in TEXT_PROCESSOR_KEYS):
                if processor is None:
                    raise ValueError(
                        "prepare_inputs: raw modality payloads "
                        f"{[k for k in TEXT_PROCESSOR_KEYS if k in result]} require "
                        "processor= (AutoProcessor / AutoFeatureExtractor, audit Q8); "
                        "already-processed tensors can be passed as plain dict fields "
                        "instead."
                    )
                result = self._call_processor(processor, result)
        else:
            raise TypeError(
                "prepare_inputs: state must be str, list[str], dict/BatchFeature or "
                f"torch.Tensor, got {type(state).__name__}."
            )
        for key, value in kwargs.items():
            result.setdefault(key, value)
        if model is not None:
            self._validate_keys(model, result.keys(), "state")
        return result
