# Copyright (c) 2026 ysnrfd
"""Pinning tests for the WAVE-1c hygiene/robustness batch.

Each test below pins ONE reviewed defect and is designed so that restoring the
pre-fix source makes it FAIL (see the module docstring of the wave report for
the mutation-check results):

* G11 - ``BaseAdapter._normalize_sequence`` existed three times in
  ``jevx/adapters/base.py``; only the last definition was ever bound, the other
  two were dead duplicates. Exactly one canonical identity hook must remain and
  the vision override must still flatten rank-4 maps.
* G12 - ``LinearProbePathway``'s class docstring advertised a kaiming-uniform
  init while ``reset_parameters`` zero-initializes.
* G14 - ``has_inf`` skipped complex tensors although ``has_nan`` handles them.
* G15 - ``_resolve_dtype`` carried string special cases unreachable through
  ``getattr(torch, name)``.
* G16 - ``batch_equivalence_tolerance`` leaked a bare ``KeyError`` for a dtype
  with no measured safety factor.
"""

from __future__ import annotations

import ast
import inspect
import re
import textwrap

import pytest
import torch

from jevx.adapters.base import BaseAdapter
from jevx.adapters.generic import GenericAdapter
from jevx.adapters.vision import VisionAdapter
from jevx.decisions.generic import LinearProbePathway
from jevx.utils.numerical import BATCH_EQUIVALENCE_SAFETY, batch_equivalence_tolerance
from jevx.utils.shapes import has_inf, has_nan

# ---------------------------------------------------------------------------
# G11 - one canonical `_normalize_sequence` on BaseAdapter
# ---------------------------------------------------------------------------


def test_g11_base_adapter_defines_normalize_sequence_exactly_once() -> None:
    """G11: the two dead duplicate definitions must be gone (source pin).

    Python binds only the last of the three copies, so the duplicates were
    unreachable code rather than a behaviour difference; the count is asserted
    against the class source so a re-introduced copy fails here.
    """
    source = inspect.getsource(BaseAdapter)
    definitions = re.findall(r"^\s{4}def _normalize_sequence\(", source, flags=re.M)
    assert len(definitions) == 1, (
        "BaseAdapter must define _normalize_sequence exactly once; found "
        f"{len(definitions)} (dead duplicate definitions are unreachable code)."
    )


def test_g11_canonical_hook_is_the_identity_on_both_returned_objects() -> None:
    """G11: the surviving definition is the documented identity hook."""
    adapter = GenericAdapter()
    sequence = torch.randn(2, 5, 8)
    mask = torch.ones(2, 5, dtype=torch.bool)
    out_sequence, out_mask = adapter._normalize_sequence(sequence, mask)
    assert out_sequence is sequence
    assert out_mask is mask
    assert (out_sequence, out_mask) == (sequence, mask)  # mask=None is legal


def test_g11_vision_override_still_flattens_rank4_maps() -> None:
    """G11: the breadth-sweep CNN path ([B, C, H, W] -> [B, H*W, C]) is intact.

    This is the behaviour the CNN / dense-prediction families of the 70-arch
    sweep depend on; deleting the base duplicates must not disturb it.
    """
    adapter = VisionAdapter()
    feature_map = torch.arange(2 * 3 * 2 * 4, dtype=torch.float32).reshape(2, 3, 2, 4)
    mask = torch.ones(2, 8, dtype=torch.bool)
    sequence, out_mask = adapter._normalize_sequence(feature_map, mask)
    assert tuple(sequence.shape) == (2, 8, 3)
    # channel-major flatten then transpose: token k == pixel (h, w) channel c
    assert torch.equal(sequence[0, 0], feature_map[0, :, 0, 0])
    assert torch.equal(sequence[1, 7], feature_map[1, :, 1, 3])
    # every spatial position is valid, so the [B, T] mask is dropped
    assert out_mask is None
    # a rank-3 token sequence is still the identity (transformer vision)
    tokens = torch.randn(2, 8, 3)
    same, _ = adapter._normalize_sequence(tokens, mask)
    assert same is tokens


# ---------------------------------------------------------------------------
# G12 - LinearProbePathway docstring must match reset_parameters
# ---------------------------------------------------------------------------


def test_g12_class_docstring_documents_the_zero_init_it_performs() -> None:
    """G12: the docstring may not claim the kaiming-uniform init it never runs."""
    doc = (LinearProbePathway.__doc__ or "").lower()
    # the pre-fix sentence that made the claim the code never honours
    assert "initialization is the standard" not in doc
    assert "kaiming-uniform) one" not in doc
    assert "initialization is zero" in doc
    assert "reset_parameters" in doc
    # the method docstring is the authority the class now points at
    assert "zero-initialized" in (LinearProbePathway.reset_parameters.__doc__ or "").lower()


def test_g12_fresh_pathway_is_bitwise_zero_matching_the_documented_init() -> None:
    """G12: a constructed pathway contributes exactly nothing (zero-init)."""
    torch.manual_seed(0)
    pathway = LinearProbePathway(hidden_size=8, num_options=4)
    assert torch.equal(pathway.weight, torch.zeros(4, 8))
    assert torch.equal(pathway.bias, torch.zeros(4))
    fused = torch.randn(3, 8)
    assert torch.equal(pathway(fused), torch.zeros(3, 4))
    # reset_parameters is idempotent and re-zeroes after perturbation
    with torch.no_grad():
        pathway.weight.add_(1.0)
        pathway.bias.add_(1.0)
    pathway.reset_parameters()
    assert torch.equal(pathway(fused), torch.zeros(3, 4))


# ---------------------------------------------------------------------------
# G14 - has_inf must handle complex tensors like has_nan
# ---------------------------------------------------------------------------


def test_g14_has_inf_detects_infinite_components_of_complex_tensors() -> None:
    """G14: a complex tensor with an infinite real or imag part is infinite."""
    finite = torch.tensor([1 + 2j, complex(0.0, 3.0)])
    assert not has_inf(finite)
    assert not has_nan(finite)

    pos_real = torch.tensor([complex(float("inf"), 0.0), 1 + 0j])
    neg_imag = torch.tensor([1 + 0j, complex(0.0, float("-inf"))])
    for tensor in (pos_real, neg_imag):
        assert has_inf(tensor)
        assert not has_nan(tensor)

    nan_complex = torch.tensor([complex(float("nan"), 0.0), 1 + 0j])
    assert has_nan(nan_complex)
    assert not has_inf(nan_complex)


def test_g14_has_inf_still_short_circuits_on_integer_and_bool_tensors() -> None:
    """G14: the widening must not change the integer/bool answer."""
    assert not has_inf(torch.arange(4))
    assert not has_inf(torch.zeros(3, dtype=torch.bool))
    assert not has_nan(torch.arange(4))
    with pytest.raises(TypeError, match="has_inf"):
        has_inf([1.0, float("inf")])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# G15 - _resolve_dtype has no unreachable string special cases
# ---------------------------------------------------------------------------


def test_g15_resolve_dtype_source_has_no_dead_string_branch() -> None:
    """G15: the `dtype == "bfloat16"` / `"half"` fallbacks are unreachable.

    ``getattr(torch, "bfloat16")`` and ``getattr(torch, "half")`` are already
    ``torch.dtype`` values, so both ``if`` arms could never run; the source is
    pinned so re-adding either one fails here.
    """
    from jevx.utils import numerical

    source = inspect.getsource(numerical._resolve_dtype)
    assert '== "bfloat16"' not in source
    assert '== "half"' not in source
    # exactly one dtype lookup in the BODY: no per-name fallback arms
    body_calls = [
        node
        for node in ast.walk(ast.parse(textwrap.dedent(source)))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
    ]
    assert len(body_calls) == 1, f"expected one getattr lookup, found {len(body_calls)}"


def test_g15_short_dtype_names_still_resolve() -> None:
    """G15: deleting the dead arms must not change any accepted name."""
    from jevx.utils.numerical import _resolve_dtype

    assert _resolve_dtype("bfloat16") is torch.bfloat16
    assert _resolve_dtype("half") is torch.float16
    assert _resolve_dtype("float16") is torch.float16
    assert _resolve_dtype("float32") is torch.float32
    assert _resolve_dtype("float64") is torch.float64
    # non-dtype torch attributes and unknown names are still TypeErrors
    for bad in ("nope", "tensor", "nn", 3, None):
        with pytest.raises(TypeError):
            _resolve_dtype(bad)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# G16 - batch_equivalence_tolerance raises ValueError, not KeyError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.int32, torch.float8_e4m3fn, torch.complex64])
def test_g16_uncalibrated_dtype_raises_value_error_naming_the_supported_ones(
    dtype: torch.dtype,
) -> None:
    """G16: a real torch dtype with no safety factor is a ValueError, not KeyError."""
    with pytest.raises(ValueError) as excinfo:
        batch_equivalence_tolerance(dtype, 8)
    message = str(excinfo.value)
    assert "KeyError" not in message
    for supported in BATCH_EQUIVALENCE_SAFETY:
        assert supported in message
    assert str(dtype).removeprefix("torch.") in message


def test_g16_unknown_dtype_name_is_still_a_type_error() -> None:
    """G16: a name that is not a torch dtype at all keeps the TypeError contract."""
    with pytest.raises(TypeError):
        batch_equivalence_tolerance("nope", 8)
    with pytest.raises(TypeError):
        batch_equivalence_tolerance(3, 8)  # type: ignore[arg-type]


def test_g16_calibrated_dtypes_still_return_the_measured_rule() -> None:
    """G16: fp32/bf16/fp16/fp64 values are bitwise unchanged by the new lookup."""
    for name, safety in BATCH_EQUIVALENCE_SAFETY.items():
        dtype = getattr(torch, name)
        expected = (
            safety
            * float(torch.finfo(dtype).eps)
            * ((128 * 11) ** 0.5)
            * 1.0
        )
        assert batch_equivalence_tolerance(dtype, 128, stages=11) == expected
        assert batch_equivalence_tolerance(name, 128, stages=11) == expected
    # an explicit safety= still overrides the table for an uncalibrated dtype
    float8 = torch.float8_e4m3fn
    assert batch_equivalence_tolerance(float8, 8, safety=1.0) == pytest.approx(
        1.0 * float(torch.finfo(float8).eps) * (8 ** 0.5)
    )
