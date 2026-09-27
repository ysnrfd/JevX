# Copyright (c) 2026 ysnrfd
"""Adapter call-boundary contracts (audit Q-013 + Q-014).

Q-013 — wrong-modality kwargs used to surface as a cryptic UPSTREAM
``AttributeError``/``TypeError`` from the backbone call with no JevX
context. ``call_backbone`` now re-wraps exactly the input-contract
failures (unexpected-kwarg naming a forwarded key, or the documented
``'NoneType' object has no attribute ...`` partial-processor-output
signature) into an actionable ``ValueError`` that names the model class,
the forwarded keys and the processor/modality fix, with the original
exception preserved as ``__cause__`` — genuine model errors propagate
themselves, untouched.

Q-014 — the ``use_cache=False`` injection lives in ONE shared helper
(``inject_no_cache`` in ``adapters/base.py``) used by the base/generic/
decoder/encoder-decoder paths: kwargs-only forwards receive
``use_cache=False``, declared parameters are honored, caller-supplied
``True`` passes through, and models that accept neither form get no key.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from jevx.adapters import GenericAdapter
from jevx.adapters import base as adapters_base
from jevx.adapters import encoder_decoder as encoder_decoder_module
from jevx.adapters.base import call_backbone, inject_no_cache

IDS = torch.ones(1, 4, dtype=torch.long)


# ---------------------------------------------------------------------------
# Q-013 — the wrap boundary
# ---------------------------------------------------------------------------


class StrictBackbone(nn.Module):
    """Declares no ``pixel_values``: Python rejects the kwarg before dispatch."""

    def forward(self, input_ids=None, attention_mask=None):
        raise AssertionError("unreachable — the call is rejected before dispatch")


class FakeVLBackbone(nn.Module):
    """VL-style forward that dies on missing companion metadata (the
    documented ``image_grid_thw``-missing failure from the Qwen battery)."""

    def forward(
        self, input_ids=None, attention_mask=None, pixel_values=None,
        image_grid_thw=None,
    ):
        if pixel_values is not None and image_grid_thw is None:
            raise AttributeError("'NoneType' object has no attribute 'device'")
        raise AssertionError("unreachable when pixel_values is given")


class GenuineErrorBackbone(nn.Module):
    """Raises the caller's own exception object — must propagate as-is."""

    def __init__(self, error: BaseException) -> None:
        super().__init__()
        self.error = error

    def forward(self, input_ids=None):
        raise self.error


def test_unexpected_kwarg_is_wrapped_with_actionable_message() -> None:
    """input_ids + pixel_values on a text-only backbone: the raw
    ``unexpected keyword argument`` TypeError becomes a JevX ValueError
    naming pixel_values, the processor fix and the model class."""
    inputs = {"input_ids": IDS, "pixel_values": torch.randn(1, 3, 4, 4)}
    with pytest.raises(ValueError) as excinfo:
        call_backbone(StrictBackbone(), inputs)
    message = str(excinfo.value)
    assert "pixel_values" in message
    assert "processor" in message
    assert "StrictBackbone" in message
    assert isinstance(excinfo.value.__cause__, TypeError)
    assert "unexpected keyword argument" in str(excinfo.value.__cause__)


def test_none_type_modality_failure_is_wrapped() -> None:
    """The cryptic `'NoneType' object has no attribute 'device'` signature
    (partial processor output) gets JevX context with the original kept as
    ``__cause__``."""
    inputs = {"input_ids": IDS, "pixel_values": torch.randn(1, 3, 4, 4)}
    with pytest.raises(ValueError) as excinfo:
        call_backbone(FakeVLBackbone(), inputs)
    message = str(excinfo.value)
    assert "pixel_values" in message
    assert "processor" in message
    assert "FakeVLBackbone" in message
    assert isinstance(excinfo.value.__cause__, AttributeError)
    assert "NoneType" in str(excinfo.value.__cause__)


def test_genuine_errors_propagate_themselves_not_wrapped() -> None:
    """Narrow matching rule: anything that is not an input-contract
    failure keeps its own exception object and message."""
    # ValueError carrying an upstream shape/computation message
    original = ValueError("mat1 and mat2 shapes cannot be multiplied (2, 3) and (4, 5)")
    with pytest.raises(ValueError) as excinfo:
        call_backbone(GenuineErrorBackbone(original), {"input_ids": IDS})
    assert excinfo.value is original
    assert "JevX could not call" not in str(excinfo.value)

    # RuntimeError (bad tensor shape) is not even in the caught types
    shape_error = RuntimeError("Expected input to be 2D, got 3D in linear")
    with pytest.raises(RuntimeError) as runtime_info:
        call_backbone(GenuineErrorBackbone(shape_error), {"input_ids": IDS})
    assert runtime_info.value is shape_error

    # an unexpected-kwarg message naming a key JevX never forwarded is NOT
    # an input-contract failure — the wrapper must not swallow it
    mismatched = TypeError("got an unexpected keyword argument 'frobnicate'")
    with pytest.raises(TypeError) as kwarg_info:
        call_backbone(GenuineErrorBackbone(mismatched), {"input_ids": IDS})
    assert kwarg_info.value is mismatched


def test_extract_representation_wraps_modality_failure_end_to_end() -> None:
    """On the real adapter boundary: the failure surfaces as a JevX
    ValueError, not as a raw upstream traceback."""
    adapter = GenericAdapter()
    inputs = {"input_ids": IDS, "pixel_values": torch.randn(1, 3, 4, 4)}
    with pytest.raises(ValueError) as excinfo:
        adapter.extract_representation(FakeVLBackbone(), inputs)
    message = str(excinfo.value)
    assert "pixel_values" in message
    assert "processor" in message
    assert "FakeVLBackbone" in message
    assert isinstance(excinfo.value.__cause__, AttributeError)


# ---------------------------------------------------------------------------
# Q-014 — one shared use_cache helper
# ---------------------------------------------------------------------------


def test_inject_no_cache_kwargs_only_forward_receives_false() -> None:
    """A forward declaring only ``**kwargs`` (the v5 TransformersKwargs
    pattern) still receives ``use_cache=False``."""

    class KwargsOnly(nn.Module):
        def forward(self, **kwargs):
            return SimpleNamespace(pooler_output=None)

    fwd = inject_no_cache({}, KwargsOnly())
    assert "use_cache" in fwd
    assert fwd["use_cache"] is False


def test_inject_no_cache_declared_parameter_and_caller_value() -> None:
    """A declared ``use_cache`` parameter is injected with False (the
    extraction contract), and a caller-supplied True passes through."""

    class Declared(nn.Module):
        def forward(self, input_ids=None, use_cache=True):
            return SimpleNamespace(pooler_output=None)

    assert inject_no_cache({}, Declared())["use_cache"] is False
    assert inject_no_cache({"use_cache": True}, Declared())["use_cache"] is True


def test_inject_no_cache_neither_form_gets_no_key() -> None:
    """A forward that accepts neither form gets NO use_cache key (an
    undeclared flag can never surface as a deep TypeError); undeclared
    extras are filtered by the shared forward_kwargs keep-set."""

    class Plain(nn.Module):
        def forward(self, input_ids=None):
            return SimpleNamespace(pooler_output=None)

    fwd = inject_no_cache({"input_ids": IDS, "frobnicate": 1}, Plain())
    assert "use_cache" not in fwd
    assert "frobnicate" not in fwd


def test_encoder_decoder_path_imports_the_same_single_helper() -> None:
    """No duplication: the encoder-decoder module routes through the one
    helper defined in adapters/base.py (same object, not a re-implementation)."""
    assert encoder_decoder_module.inject_no_cache is adapters_base.inject_no_cache
    assert encoder_decoder_module.call_backbone is adapters_base.call_backbone


def test_extract_representation_injects_use_cache_on_path() -> None:
    """Through the real extraction path: a kwargs-only backbone sees
    ``use_cache=False`` by default and a caller-supplied True passes."""
    seen: dict = {}

    class Recorder(nn.Module):
        def forward(self, input_ids=None, attention_mask=None, **kwargs):
            seen.clear()
            seen.update(kwargs)
            return SimpleNamespace(
                pooler_output=torch.zeros(2, 8),
                last_hidden_state=torch.zeros(2, 3, 8),
            )

    adapter = GenericAdapter()
    inputs = {
        "input_ids": torch.ones(2, 3, dtype=torch.long),
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
    }
    representation = adapter.extract_representation(Recorder(), inputs)
    assert seen.get("use_cache") is False
    assert representation.pooled is not None
    assert tuple(representation.pooled.shape) == (2, 8)

    adapter.extract_representation(Recorder(), {**inputs, "use_cache": True})
    assert seen.get("use_cache") is True
