# Copyright (c) 2026 ysnrfd
"""Unit tests for jevx.training.freeze: requires_grad toggling, Mode A/B
backbone control (duck-typed), layer-wise unfreezing on tiny locally
constructed transformers models (no downloads), and parameter counting."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, GPT2Config, GPT2Model

from jevx.training import (
    count_parameters,
    freeze_backbone,
    freeze_module,
    unfreeze_backbone,
    unfreeze_last_n_layers,
    unfreeze_module,
)


def _tiny_gpt2(n_layer: int = 4) -> GPT2Model:
    torch.manual_seed(0)
    return GPT2Model(
        GPT2Config(
            n_layer=n_layer,
            n_embd=16,
            n_head=2,
            n_positions=32,
            vocab_size=64,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
    )


def _tiny_bert(n_layer: int = 3) -> BertModel:
    torch.manual_seed(0)
    return BertModel(
        BertConfig(
            vocab_size=64,
            hidden_size=16,
            num_hidden_layers=n_layer,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=32,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )
    )


class _Wrapped(nn.Module):
    """Wrapper with a plain ``.backbone`` child (duck-typing rule 2)."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 8))
        self.head = nn.Linear(8, 2)


class _TrainableApi(nn.Module):
    """Wrapper exposing ``trainable_parameters()`` (duck-typing rule 1)."""

    def __init__(self):
        super().__init__()
        self.base = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)

    def trainable_parameters(self):
        return self.head.parameters()


class _Opaque(nn.Module):
    """Neither a trainable_parameters() API nor a model/backbone child."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 4)


# ---------------------------------------------------------------------------
# freeze_module / unfreeze_module
# ---------------------------------------------------------------------------


def test_freeze_unfreeze_toggles_requires_grad():
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    assert all(p.requires_grad for p in model.parameters())
    freeze_module(model)
    assert not any(p.requires_grad for p in model.parameters())
    unfreeze_module(model)
    assert all(p.requires_grad for p in model.parameters())


def test_freeze_module_returns_module_for_chaining():
    model = nn.Linear(3, 3)
    assert freeze_module(model) is model
    assert unfreeze_module(model) is model


def test_freeze_module_rejects_non_module():
    with pytest.raises(TypeError, match=r"nn.Module"):
        freeze_module("not a module")  # type: ignore[arg-type]


def test_freeze_only_touches_target_submodule():
    model = _Wrapped()
    freeze_module(model.backbone)
    assert not any(p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.head.parameters())


# ---------------------------------------------------------------------------
# freeze_backbone / unfreeze_backbone (duck-typed)
# ---------------------------------------------------------------------------


def test_freeze_backbone_via_trainable_parameters_api():
    model = _TrainableApi()
    freeze_backbone(model)
    assert not any(p.requires_grad for p in model.base.parameters())
    assert all(p.requires_grad for p in model.head.parameters())
    unfreeze_backbone(model)
    assert all(p.requires_grad for p in model.parameters())


def test_freeze_backbone_via_named_child():
    model = _Wrapped()
    freeze_backbone(model)
    assert not any(p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.head.parameters())
    unfreeze_backbone(model)
    assert all(p.requires_grad for p in model.parameters())


def test_freeze_backbone_opaque_module_raises_informatively():
    with pytest.raises(ValueError, match=r"trainable_parameters|backbone"):
        freeze_backbone(_Opaque())


def test_freeze_backbone_rejects_non_module():
    with pytest.raises(TypeError, match=r"nn.Module"):
        freeze_backbone(None)  # type: ignore[arg-type]


def test_freeze_module_source_never_imports_modeling_jevx():
    """freeze.py must not import the (possibly absent) model module at all —
    this is what keeps it cycle-free and importable before modeling exists."""
    import jevx.training.freeze as freeze_mod

    assert "modeling_jevx" not in vars(freeze_mod)
    with open(freeze_mod.__file__, encoding="utf-8") as handle:
        source = handle.read()
    assert "import modeling_jevx" not in source
    assert "from jevx.modeling_jevx" not in source


# ---------------------------------------------------------------------------
# unfreeze_last_n_layers
# ---------------------------------------------------------------------------


def test_unfreeze_last_n_layers_gpt2_exact_identity():
    """Unfreezing the last n of a fully frozen GPT-2 touches EXACTLY those
    layers' parameters (checked by parameter identity, not just counts)."""
    model = _tiny_gpt2(n_layer=4)
    freeze_module(model)
    before = {id(p): p.requires_grad for p in model.parameters()}
    assert not any(before.values())

    unfreeze_last_n_layers(model, 2)

    trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
    expected_ids = {id(p) for layer in model.h[2:] for p in layer.parameters()}
    assert trainable_ids == expected_ids
    # Untouched params kept their exact flag object state.
    for p in model.parameters():
        if id(p) not in expected_ids:
            assert p.requires_grad is False


@pytest.mark.parametrize("n", [0, 1, 3])
def test_unfreeze_last_n_layers_gpt2_various_n(n):
    model = _tiny_gpt2(n_layer=3)
    freeze_module(model)
    unfreeze_last_n_layers(model, n)
    per_layer = [any(p.requires_grad for p in layer.parameters()) for layer in model.h]
    assert per_layer == [False] * (3 - n) + [True] * n


def test_unfreeze_last_n_layers_bert_encoder_path():
    """BERT nests layers at ``encoder.layer`` — the third conventional path."""
    model = _tiny_bert(n_layer=3)
    freeze_module(model)
    unfreeze_last_n_layers(model, 1)
    per_layer = [
        any(p.requires_grad for p in layer.parameters())
        for layer in model.encoder.layer
    ]
    assert per_layer == [False, False, True]
    # Everything outside the last encoder layer stays frozen.
    assert not any(p.requires_grad for p in model.embeddings.parameters())
    assert not any(p.requires_grad for p in model.pooler.parameters())


def test_unfreeze_last_n_layers_wrapper_with_model_child():
    """``ForCausalLM``-style wrappers nest the base model under ``.model``."""
    base = _tiny_gpt2(n_layer=3)

    class Wrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.model = m
            self.head = nn.Linear(16, 64)

    wrapper = Wrapper(base)
    freeze_module(wrapper)
    unfreeze_last_n_layers(wrapper, 1)
    per_layer = [any(p.requires_grad for p in l.parameters()) for l in base.h]
    assert per_layer == [False, False, True]
    assert not any(p.requires_grad for p in wrapper.head.parameters())


def test_unfreeze_last_n_layers_n_zero_is_noop():
    model = _tiny_gpt2(n_layer=2)
    freeze_module(model)
    unfreeze_last_n_layers(model, 0)
    assert not any(p.requires_grad for p in model.parameters())


def test_unfreeze_last_n_layers_invalid_n():
    model = _tiny_gpt2(n_layer=2)
    with pytest.raises(ValueError, match=r"only has 2"):
        unfreeze_last_n_layers(model, 5)
    with pytest.raises(ValueError, match=r">= 0"):
        unfreeze_last_n_layers(model, -1)
    with pytest.raises(TypeError, match=r"int"):
        unfreeze_last_n_layers(model, 1.5)  # type: ignore[arg-type]


def test_unfreeze_last_n_layers_no_layer_list_raises():
    with pytest.raises(ValueError, match=r"layer list"):
        unfreeze_last_n_layers(_Opaque(), 1)


# ---------------------------------------------------------------------------
# count_parameters
# ---------------------------------------------------------------------------


def test_count_parameters_matches_manual_count():
    model = nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 2))
    total_manual = sum(p.numel() for p in model.parameters())
    counts = count_parameters(model)
    assert counts["total"] == total_manual
    assert counts["trainable"] + counts["frozen"] == counts["total"]
    assert counts["trainable"] == total_manual  # nothing frozen yet

    freeze_module(model[0])
    counts = count_parameters(model)
    frozen_manual = sum(p.numel() for p in model[0].parameters())
    assert counts["frozen"] == frozen_manual
    assert counts["trainable"] == total_manual - frozen_manual
    assert counts["trainable"] + counts["frozen"] == counts["total"]


def test_count_parameters_on_frozen_backbone_wrapper():
    model = _Wrapped()
    freeze_backbone(model)
    counts = count_parameters(model)
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    head_params = sum(p.numel() for p in model.head.parameters())
    assert counts == {
        "trainable": head_params,
        "frozen": backbone_params,
        "total": head_params + backbone_params,
    }
    # After unfreezing everything, frozen == 0.
    unfreeze_backbone(model)
    assert count_parameters(model)["frozen"] == 0


def test_count_parameters_rejects_non_module():
    with pytest.raises(TypeError, match=r"nn.Module"):
        count_parameters([1, 2, 3])  # type: ignore[arg-type]
