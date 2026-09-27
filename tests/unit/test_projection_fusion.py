# Copyright (c) 2026 ysnrfd
"""Projection and fusion tests: shapes per kind, masked fill, shape errors.

Covers Definition 5 of ``docs/MATHEMATICAL_SPECIFICATION.md`` and the pinned
``Fusion`` contract (``fused [B, D]``, ``option_features [B, K, D]``, masked
positions exactly 0, no NaN).
"""

from __future__ import annotations

import pytest
import torch

from jevx.representations import Projection, build_fusion
from jevx.representations.fusion import (
    FUSION_REGISTRY,
    AdditiveFusion,
    ConcatFusion,
    Fusion,
    available_fusions,
)

torch.manual_seed(0)

D = 8
B, K = 3, 4

#: Canonical kinds from the pinned contract (the "add" alias is tested too).
CANONICAL_KINDS = ("concat", "additive", "gated", "bilinear", "dot", "mlp")


# --- Projection -------------------------------------------------------------


def test_projection_linear_shapes_and_grads() -> None:
    proj = Projection(16, D, layer_norm=True, mlp=False)
    x2 = torch.randn(B, 16, requires_grad=True)
    y2 = proj(x2)
    assert y2.shape == (B, D)
    x3 = torch.randn(B, K, 16, requires_grad=True)
    y3 = proj(x3)
    assert y3.shape == (B, K, D)
    y3.sum().backward()
    assert x3.grad is not None and torch.isfinite(x3.grad).all()
    assert all(p.grad is not None for p in proj.parameters())


def test_projection_mlp_variant_has_more_parameters() -> None:
    linear = Projection(16, D, mlp=False)
    mlp = Projection(16, D, mlp=True)
    n_linear = sum(p.numel() for p in linear.parameters())
    n_mlp = sum(p.numel() for p in mlp.parameters())
    assert n_mlp > n_linear
    assert mlp(torch.randn(B, 16)).shape == (B, D)


def test_projection_layer_norm_normalizes_rows() -> None:
    proj = Projection(32, 64, layer_norm=True, mlp=False)
    out = proj(torch.randn(B, 32) * 10)
    assert out.mean(dim=-1).abs().max() < 1e-4
    assert out.std(dim=-1).mean().item() == pytest.approx(1.0, abs=1e-2)


def test_projection_layer_norm_off_has_no_norm_params() -> None:
    proj = Projection(16, D, layer_norm=False)
    assert not any("norm" in name for name, _ in proj.named_parameters())
    assert proj(torch.randn(B, 16)).shape == (B, D)


def test_projection_errors() -> None:
    with pytest.raises(ValueError, match="in_dim"):
        Projection(0, 4)
    with pytest.raises(ValueError, match="out_dim"):
        Projection(4, -1)
    proj = Projection(16, D)
    with pytest.raises(ValueError, match="last dimension"):
        proj(torch.randn(B, 8))
    with pytest.raises(TypeError, match="torch.Tensor"):
        proj("not a tensor")  # type: ignore[arg-type]


# --- Fusion -----------------------------------------------------------------


def test_registry_contains_contract_kinds() -> None:
    assert set(CANONICAL_KINDS) <= set(FUSION_REGISTRY)
    assert "add" in FUSION_REGISTRY  # documented JevXConfig spelling alias
    assert FUSION_REGISTRY["add"] is AdditiveFusion
    assert FUSION_REGISTRY["concat"] is ConcatFusion


@pytest.mark.parametrize("kind", CANONICAL_KINDS)
def test_fusion_kinds_shapes_masked_fill_and_grads(kind: str) -> None:
    fusion = build_fusion(kind, D)
    assert isinstance(fusion, Fusion)
    state = torch.randn(B, D, requires_grad=True)
    question = torch.randn(B, D, requires_grad=True)
    options = torch.randn(B, K, D, requires_grad=True)
    option_mask = torch.tensor(
        [[True, True, True, False], [True, True, False, False], [True, False, False, False]]
    )
    out = fusion(state, question, options, option_mask)
    assert out.fused.shape == (B, D)
    assert out.option_features is not None
    assert out.option_features.shape == (B, K, D)
    assert torch.isfinite(out.fused).all()
    assert torch.isfinite(out.option_features).all()
    # masked positions exactly zero — no NaN anywhere
    assert (out.option_features[~option_mask] == 0).all()
    out.fused.sum().backward()
    assert state.grad is not None and question.grad is not None


def test_fusion_without_options_returns_none_option_features() -> None:
    fusion = build_fusion("concat", D)
    out = fusion(torch.randn(B, D), torch.randn(B, D))
    assert out.fused.shape == (B, D)
    assert out.option_features is None


def test_fusion_masked_fill_survives_extreme_option_values() -> None:
    """Garbage magnitudes at masked positions are replaced, not propagated."""
    fusion = build_fusion("gated", D)
    options = torch.full((1, 3, D), 1e30)
    mask = torch.tensor([[True, True, False]])
    out = fusion(torch.randn(1, D), torch.randn(1, D), options, mask)
    assert (out.option_features[0, 2] == 0).all()
    assert torch.isfinite(out.option_features).all()


def test_fusion_detects_batch_mismatch() -> None:
    fusion = build_fusion("concat", D)
    with pytest.raises(ValueError, match="batch size"):
        fusion(torch.randn(3, D), torch.randn(2, D))


def test_fusion_detects_hidden_mismatch() -> None:
    fusion = build_fusion("concat", D)
    with pytest.raises(ValueError, match="state"):
        fusion(torch.randn(B, D + 1), torch.randn(B, D + 1))


def test_fusion_rejects_non_bool_mask() -> None:
    fusion = build_fusion("concat", D)
    with pytest.raises(TypeError, match="torch.bool"):
        fusion(
            torch.randn(B, D),
            torch.randn(B, D),
            torch.randn(B, K, D),
            torch.ones(B, K, dtype=torch.int64),
        )


def test_fusion_rejects_mask_shape_mismatch() -> None:
    fusion = build_fusion("concat", D)
    with pytest.raises(ValueError, match="option_mask"):
        fusion(
            torch.randn(B, D),
            torch.randn(B, D),
            torch.randn(B, K, D),
            torch.ones(2, K, dtype=torch.bool),
        )


def test_fusion_rejects_mask_without_options() -> None:
    fusion = build_fusion("concat", D)
    with pytest.raises(ValueError, match="without options"):
        fusion(
            torch.randn(B, D),
            torch.randn(B, D),
            None,
            torch.ones(B, K, dtype=torch.bool),
        )


def test_fusion_rejects_non_finite_inputs() -> None:
    fusion = build_fusion("concat", D)
    state = torch.randn(B, D)
    state[0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        fusion(state, torch.randn(B, D))


def test_build_fusion_unknown_kind_lists_registry() -> None:
    with pytest.raises(ValueError, match="registered kinds"):
        build_fusion("definitely_not_a_kind", D)
    with pytest.raises(TypeError, match="kind"):
        build_fusion(42, D)  # type: ignore[arg-type]


def test_available_fusions_sorted() -> None:
    names = available_fusions()
    assert names == sorted(names)
    assert set(CANONICAL_KINDS) <= set(names)
