# Copyright (c) 2026 ysnrfd
"""Regression tests for the architecture-breadth VISION adapter fix.

Found by the breadth sweep (``benchmarks/run_breadth_sweep.py``): the
convolutional / dense-prediction vision backbones do not return a token
sequence. ``ConvNextModel``, ``ResNetModel``, ``MobileNetV2Model`` and
``SegformerModel`` all emit ``last_hidden_state`` as a rank-4 ``[B, C, H, W]``
feature map, and ``DPTModel`` does the same on its neck path.

Before the fix, ``BaseAdapter.extract_representation`` handed that rank-4 map
straight to ``Representation`` / ``pool``, which require ``[B, T, D]``::

    ValueError: sequence_output: expected shape (None, None, None) (rank 3)
                but got shape (2, 64, 1, 1) (rank 4)

``VisionAdapter._normalize_sequence`` now flattens a rank-4 map to the
``[B, H*W, C]`` token layout torch itself uses for NCHW-to-token conversions.
"""

from __future__ import annotations

import torch
from transformers import (
    ConvNextConfig,
    ConvNextModel,
    SegformerConfig,
    SegformerModel,
)

from jevx.adapters import VisionAdapter, resolve_adapter
from jevx.capabilities import JevXCapabilityDetector
from jevx.representations.pooling import pool

WIDTH = 32
IMAGE = 32


def _convnext() -> ConvNextModel:
    """Tiny ConvNextModel: 4 conv stages, rank-4 ``last_hidden_state``."""
    torch.manual_seed(0)
    config = ConvNextConfig(
        num_channels=3, hidden_sizes=[8, 16, 32, 64], depths=[1, 1, 1, 1],
        layer_norm_eps=1e-6, patch_size=4, image_size=IMAGE, num_attention_heads=4,
    )
    return ConvNextModel(config).eval()


def _segformer() -> SegformerModel:
    """Tiny SegformerModel: no pooler, rank-4 decoder feature map."""
    torch.manual_seed(1)
    config = SegformerConfig(
        num_channels=3, hidden_sizes=[8, 16, 32, 64], depths=[1, 1, 1, 1],
        layer_norm_eps=1e-6, image_size=IMAGE, num_attention_heads=[1, 2, 4, 8],
        mlp_ratios=[2, 2, 2, 2], mlp_hidden_sizes=[8, 16, 32, 64],
        num_encoder_blocks=2, num_decoder_blocks=1, decoder_hidden_size=8,
    )
    return SegformerModel(config).eval()


def test_normalize_sequence_flattens_rank4_maps() -> None:
    """``[B, C, H, W]`` -> ``[B, H*W, C]``, and rank-3 is the identity."""
    adapter = VisionAdapter()
    spatial = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
    flat, mask = adapter._normalize_sequence(spatial, torch.ones(2, 3, dtype=torch.long))
    assert flat.shape == (2, 20, 3), (
        f"a [B, C, H, W] map must flatten to [B, H*W, C]; got {tuple(flat.shape)}"
    )
    assert mask is None, "a spatial map has no token mask; a stale mask must be dropped"
    # Token index n maps to spatial position (n // W, n % W); the channel
    # axis becomes the last axis of the token vector.
    assert torch.equal(flat[0, 7, :], spatial[0, :, 1, 2])
    tokens = torch.randn(2, 5, 3)
    same, same_mask = adapter._normalize_sequence(tokens, torch.ones(2, 5, dtype=torch.long))
    assert torch.equal(same, tokens) and same_mask is not None, (
        "a rank-3 token sequence must pass through untouched"
    )


def test_convnext_extraction_yields_a_rank3_sequence() -> None:
    """convnext: native pooler + a flattened spatial sequence."""
    model = _convnext()
    adapter = resolve_adapter(model)
    assert isinstance(adapter, VisionAdapter)
    report = JevXCapabilityDetector.detect(model)
    assert report.vision and not report.text
    with torch.no_grad():
        raw = model(pixel_values=torch.randn(2, 3, IMAGE, IMAGE))
    assert raw.last_hidden_state.dim() == 4, (
        "the fixture must be a spatial backbone; if transformers changed the "
        "output rank, revisit VisionAdapter._normalize_sequence"
    )
    with torch.no_grad():
        representation = adapter.extract_representation(
            model, {"pixel_values": torch.randn(2, 3, IMAGE, IMAGE)}
        )
    assert representation.layout == "btd"
    assert representation.sequence_output is not None
    assert representation.sequence_output.dim() == 3
    assert representation.pooled is not None and representation.pooled.dim() == 2
    assert torch.isfinite(representation.pooled).all()


def test_segformer_extraction_pools_the_spatial_map() -> None:
    """segformer: no pooler, so masked-mean over the flattened map."""
    model = _segformer()
    adapter = resolve_adapter(model)
    with torch.no_grad():
        representation = adapter.extract_representation(
            model, {"pixel_values": torch.randn(2, 3, IMAGE, IMAGE)}
        )
    assert representation.field == "last_hidden_state"
    sequence = representation.sequence_output
    assert sequence is not None and sequence.dim() == 3
    pooled = pool(sequence, representation.attention_mask, "masked_mean")
    assert pooled.shape == (2, sequence.shape[-1])
    assert torch.isfinite(pooled).all()
    assert representation.pooled is not None
    assert representation.pooled.shape == pooled.shape


def test_rank3_vision_backbones_are_unchanged() -> None:
    """A transformer vision tower keeps its token sequence as-is (vit)."""
    from transformers import ViTConfig, ViTModel

    torch.manual_seed(2)
    model = ViTModel(
        ViTConfig(hidden_size=WIDTH, num_hidden_layers=2, num_attention_heads=4,
                  intermediate_size=64, image_size=16, patch_size=8, num_channels=3)
    ).eval()
    with torch.no_grad():
        representation = resolve_adapter(model).extract_representation(
            model, {"pixel_values": torch.randn(2, 3, 16, 16)}
        )
    assert representation.sequence_output is not None
    assert representation.sequence_output.dim() == 3
    assert representation.sequence_output.shape[1] == 4 + 1, (
        "16/8 -> a 2x2 patch grid plus the CLS token; the flatten path must not "
        "have been taken for a rank-3 sequence"
    )
