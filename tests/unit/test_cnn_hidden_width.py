# Copyright (c) 2026 ysnrfd
"""The hidden-width rule for CNN backbones (the 4 PARTIAL breadth-sweep rows).

``JevXForDecision`` sizes ``state_proj`` from the backbone config. Convolutional
families declare no scalar width, so :func:`jevx.modeling_jevx._hidden_width`
gained two rules, both derived from the vendored source rather than from a
per-family table:

* rule 4, ``_stage_width`` -- a per-stage width LIST (``hidden_sizes``) is
  resolved at the last stage the model ACTUALLY builds:

  - convnext: ``self.classifier = nn.Linear(config.hidden_sizes[-1],
    config.num_labels)`` (``modeling_convnext.py:304``), one stage per
    ``depths`` entry (``configuration_convnext.py:50,61``), so ``[-1]`` is it;
  - resnet: ``nn.Sequential(nn.Flatten(), nn.Linear(config.hidden_sizes[-1],
    config.num_labels))`` (``modeling_resnet.py:346-348``);
  - segformer: ``SegformerModel`` builds exactly ``config.num_encoder_blocks``
    stages (``modeling_segformer.py:384-386``) and ``SegformerStage`` emits
    ``hidden_size=config.hidden_sizes[stage_idx]``
    (``modeling_segformer.py:321``), so a list longer than the block count is
    indexed AT THE BLOCK COUNT, not at ``-1``;
  - poolformer declares the same field (``modeling_poolformer.py:326``).

* rule 5, ``_last_conv_width`` -- a config with NO width field at all:
  ``MobileNetV2Config`` declares none (``configuration_mobilenet_v2.py:60-76``)
  and its head reads the width off the final convolution --
  ``last_hidden_size = self.mobilenet_v2.conv_1x1.convolution.out_channels``
  (``modeling_mobilenet_v2.py:381``), the last module the feature extractor
  builds (``modeling_mobilenet_v2.py:313-318``).

Both rules run AFTER the untouched scalar / ``projection_dim`` / nested-config
rules, so they can only fire where the function used to return ``None``. A
fourth, tiny guard makes that true in practice: a stage list the config CLASS
never declared is ignored, because ``PreTrainedConfig`` copies unknown
constructor kwargs onto the instance (that is exactly what a stray
``hidden_sizes=[8, 16]`` on ``MobileNetV2Config`` is). The guard family
(bert / llama / vit / t5) pins the ordering: their widths must stay the scalar
config key. The end-to-end tests pin the contract that actually matters -- the
resolved width equals the width extraction returns, and ``JevXForDecision``
runs a full ``.choice`` forward on all four CNNs.

``resnet`` needed a second fix in ``forward``: ``ResNetModel`` returns its
``pooler_output`` unflattened (``[B, C, 1, 1]``, ``modeling_resnet.py:297,
322-329``) where the sibling ``MobileNetV2Model`` flattens the identical op
(``modeling_mobilenet_v2.py:354``), so the wrapper merges the trailing spatial
axes into the documented ``[B, D]`` (``adapters/base.py:76``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import pytest
import torch
from transformers import (
    BertConfig,
    BertModel,
    ConvNextConfig,
    ConvNextModel,
    LlamaConfig,
    LlamaModel,
    MobileNetV2Config,
    MobileNetV2Model,
    ResNetConfig,
    ResNetModel,
    SegformerConfig,
    SegformerModel,
    T5Config,
    T5Model,
    ViTConfig,
    ViTModel,
)

from jevx import JevXForDecision
from jevx.adapters import resolve_adapter
from jevx.configuration_jevx import JevXConfig
from jevx.modeling_jevx import _declares, _hidden_width, _last_conv_width, _stage_width
from jevx.representations.pooling import pool

#: Tiny in-memory CNN configs -- every size minimized, no download and no
#: checkpoint. Sizes mirror the breadth sweep so the two stay comparable.
CNN_SPECS: Dict[str, Tuple[Any, Any, Dict[str, Any]]] = {
    "convnext": (
        ConvNextConfig,
        ConvNextModel,
        {"num_channels": 3, "hidden_sizes": [8, 16, 32, 64], "depths": [1, 1, 1, 1],
         "layer_norm_eps": 1e-6, "patch_size": 4, "image_size": 32,
         "num_attention_heads": 4},
    ),
    "resnet": (
        ResNetConfig,
        ResNetModel,
        {"num_channels": 3, "hidden_sizes": [8, 16, 32, 64], "depths": [1, 1, 1, 1],
         "layer_norm_eps": 1e-6, "image_size": 32, "num_attention_heads": 4},
    ),
    # num_encoder_blocks=2 over a 4-long list: the last BUILT stage is 16, not
    # hidden_sizes[-1]=64 -- exactly what a naive "last stage" rule gets wrong.
    "segformer": (
        SegformerConfig,
        SegformerModel,
        {"num_channels": 3, "hidden_sizes": [8, 16, 32, 64], "depths": [1, 1, 1, 1],
         "layer_norm_eps": 1e-6, "image_size": 32,
         "num_attention_heads": [1, 2, 4, 8], "mlp_ratios": [2, 2, 2, 2],
         "num_encoder_blocks": 2, "decoder_hidden_size": 8},
    ),
    # MobileNetV2Config declares NO width field. `hidden_sizes`/`depths` here are
    # STRAY kwargs PreTrainedConfig copies onto the instance, so rule 4 must
    # ignore them and rule 5 must return the real output width.
    "mobilenet_v2": (
        MobileNetV2Config,
        MobileNetV2Model,
        {"num_channels": 3, "hidden_sizes": [8, 16], "depths": [1, 1],
         "layer_norm_eps": 1e-6, "image_size": 32, "width_mult": 1.0},
    ),
}

#: The width each family must resolve to, with the rule that produces it.
EXPECTED_WIDTH: Dict[str, int] = {
    "convnext": 64,         # hidden_sizes[-1]
    "resnet": 64,           # hidden_sizes[-1]
    "segformer": 16,        # hidden_sizes[num_encoder_blocks - 1]
    "mobilenet_v2": 1280,   # conv_1x1.convolution.out_channels
}

#: Families whose width must stay EXACTLY the scalar config key (the rows that
#: already passed must not move).
GUARD_SPECS: Dict[str, Tuple[Any, Any, Dict[str, Any], str]] = {
    "bert": (
        BertConfig, BertModel,
        {"hidden_size": 24, "num_hidden_layers": 1, "num_attention_heads": 2,
         "intermediate_size": 32, "vocab_size": 64},
        "hidden_size",
    ),
    "llama": (
        LlamaConfig, LlamaModel,
        {"hidden_size": 24, "num_hidden_layers": 1, "num_attention_heads": 2,
         "intermediate_size": 32, "vocab_size": 64},
        "hidden_size",
    ),
    "vit": (
        ViTConfig, ViTModel,
        {"hidden_size": 24, "num_hidden_layers": 1, "num_attention_heads": 2,
         "intermediate_size": 32, "image_size": 16, "patch_size": 8,
         "num_channels": 3},
        "hidden_size",
    ),
    "t5": (
        T5Config, T5Model,
        {"d_model": 24, "num_layers": 1, "num_heads": 2, "d_kv": 8, "d_ff": 32,
         "vocab_size": 64},
        "d_model",
    ),
}

#: Root of the vendored transformers source clone (the citations below).
VENDOR_MODELS = Path(__file__).resolve().parents[2] / "vendor" / "transformers" / "src" / "transformers" / "models"

def _extracted_width(model: Any, image_size: int) -> int:
    """Width of the pooled state vector the adapter actually returns.

    Args:
        model: A constructed backbone.
        image_size: Pixel edge length for the synthetic input.

    Returns:
        The last dimension of the pooled ``[B, D]`` representation.
    """
    adapter = resolve_adapter(model)
    inputs = adapter.prepare_inputs(
        {"pixel_values": torch.randn(2, 3, image_size, image_size)}, model=model
    )
    with torch.no_grad():
        representation = adapter.extract_representation(model, inputs)
    pooled = representation.pooled
    if representation.sequence_output is not None:
        pooled = pool(
            representation.sequence_output,
            representation.attention_mask,
            adapter.default_pooling,
        )
    assert pooled is not None and pooled.dim() == 2, tuple(pooled.shape)
    return int(pooled.shape[-1])


@pytest.mark.parametrize("family", sorted(CNN_SPECS))
def test_cnn_width_equals_the_extracted_width(family: str) -> None:
    """The resolved width IS the width extraction returns, for all four CNNs."""
    config_cls, model_cls, kwargs = CNN_SPECS[family]
    torch.manual_seed(0)
    config = config_cls(**kwargs)
    model = model_cls(config).eval()
    assert _hidden_width(config, model) == EXPECTED_WIDTH[family], family
    assert _hidden_width(config, model) == _extracted_width(model, 32), family


@pytest.mark.parametrize("family", sorted(CNN_SPECS))
def test_cnn_width_rule_reaches_the_right_step(family: str) -> None:
    """Each family is resolved by the rule its own source requires."""
    config_cls, model_cls, kwargs = CNN_SPECS[family]
    config = config_cls(**kwargs)
    model = model_cls(config).eval()
    if family == "mobilenet_v2":
        # No width field is declared at all: the stray hidden_sizes kwarg must
        # NOT be taken, and the final convolution must be.
        assert getattr(config, "hidden_sizes", None) == [8, 16], (
            "the stray kwarg must survive for this guard to be meaningful"
        )
        assert not _declares(config, "hidden_sizes"), (
            "MobileNetV2Config must not declare hidden_sizes"
        )
        assert _stage_width(config) is None, "a stray kwarg is not a width schema"
        assert _hidden_width(config) is None, "rule 5 needs the model"
        assert _last_conv_width(model) == EXPECTED_WIDTH[family]
        assert model.conv_1x1.convolution.out_channels == EXPECTED_WIDTH[family], (
            "modeling_mobilenet_v2.py:381 reads exactly this attribute"
        )
        return
    assert _declares(config, "hidden_sizes"), f"{family} must declare hidden_sizes"
    assert _stage_width(config) == EXPECTED_WIDTH[family], family
    assert _hidden_width(config) == EXPECTED_WIDTH[family], "config-only path"
    assert _hidden_width(config, model) == EXPECTED_WIDTH[family], "with-model path"
    if family == "segformer":
        # The list is longer than the block count, so the last entry is WRONG.
        assert config.hidden_sizes[-1] == 64
        assert EXPECTED_WIDTH[family] == config.hidden_sizes[config.num_encoder_blocks - 1] == 16
        assert _last_conv_width(model) != EXPECTED_WIDTH[family], (
            "the Mix-FFN depth-wise conv is the MLP internal width, not the stage"
        )
    else:
        assert EXPECTED_WIDTH[family] == config.hidden_sizes[-1]


@pytest.mark.parametrize("family", sorted(GUARD_SPECS))
def test_scalar_width_families_are_unchanged(family: str) -> None:
    """The already-passing families still resolve to their SCALAR config key."""
    config_cls, model_cls, kwargs, key = GUARD_SPECS[family]
    torch.manual_seed(0)
    config = config_cls(**kwargs)
    model = model_cls(config).eval()
    expected = getattr(config, key)
    assert isinstance(expected, int) and expected > 0
    assert _hidden_width(config) == expected, family
    assert _hidden_width(config, model) == expected, family
    # ... and the new rules must not have hijacked them.
    assert not _declares(config, "hidden_sizes"), family
    assert _stage_width(config) is None, family


def test_scalar_key_wins_over_a_stage_list() -> None:
    """A config with BOTH a scalar width and a stage list keeps the scalar.

    ``DPTConfig`` is the audited case: ``hidden_size: int = 768``
    (``configuration_dpt.py:90``) is the width of the representation it
    returns, while ``neck_hidden_sizes`` (``configuration_dpt.py:107``) only
    sizes the fusion neck. The rules are ordered, not merged.
    """
    from transformers import DPTConfig

    config = DPTConfig(hidden_size=32, neck_hidden_sizes=[16, 16, 16, 16])
    assert _hidden_width(config) == 32
    # ... and with a width-less config of the same shape the stage list is used.
    assert _hidden_width({"hidden_sizes": [16, 16, 16, 16]}) == 16


@pytest.mark.parametrize("family", sorted(CNN_SPECS))
def test_cnn_for_decision_runs_a_full_choice_forward(family: str) -> None:
    """Construct + ``.choice`` + every decision invariant, for each CNN row."""
    config_cls, model_cls, kwargs = CNN_SPECS[family]
    torch.manual_seed(0)
    config = config_cls(**kwargs)
    model = model_cls(config).eval()
    extracted = _extracted_width(model, 32)
    jevx_config = JevXConfig(
        base_model_type=config.model_type,
        decision_hidden_size=16,
        question_hidden_size=16,
        schema_hidden_size=16,
        decision_primitives=["choice"],
        freeze_base_model=True,
        runtime_validation=True,
    )
    wrapper = JevXForDecision(jevx_config, backbone=model).eval()
    assert wrapper.state_proj.in_dim == extracted, family
    assert wrapper.config.decision_hidden_size == 16
    state = {"pixel_values": torch.randn(2, 3, 32, 32)}
    questions = ["which option wins?", "is it safe?"]
    options = [["alpha", "beta", "gamma"], ["yes", "no", "maybe"]]
    with torch.no_grad():
        output = wrapper.choice(state=state, questions=questions, options=options)
    assert len(output.decisions) == 2 * len(questions), family
    assert tuple(output.state_representation.shape) == (2, 16), family
    for decision, keys in zip(output.decisions, options * 2):
        probabilities = decision.probabilities[0]
        assert int(probabilities.shape[-1]) == len(keys), family
        assert torch.allclose(probabilities.sum(), torch.tensor(1.0), atol=1e-5), family
        assert 0 <= int(decision.predicted_index[0]) < len(keys), family
        assert decision.predicted_value in keys, family
        assert 0.0 <= float(decision.confidence[0]) <= 1.0, family


def test_resnet_pooler_map_is_repaired_to_a_row_vector() -> None:
    """``ResNetModel.pooler_output`` is a MAP; the wrapper merges it to ``[B, D]``.

    ``ResNetModel`` pools with ``nn.AdaptiveAvgPool2d((1, 1))`` and returns the
    result unflattened (``modeling_resnet.py:297, 322-329``), so the native
    pooler route hands back ``[B, C, 1, 1]`` while ``Representation.pooled`` is
    documented as ``[B, D]`` (``adapters/base.py:76``). Without the repair the
    runtime validator raises and no CNN row can pass.
    """
    torch.manual_seed(0)
    model = ResNetModel(ResNetConfig(**CNN_SPECS["resnet"][2])).eval()
    adapter = resolve_adapter(model)
    inputs = adapter.prepare_inputs(
        {"pixel_values": torch.randn(2, 3, 32, 32)}, model=model
    )
    with torch.no_grad():
        raw = model(**inputs)
    assert raw.pooler_output.dim() == 4, tuple(raw.pooler_output.shape)
    with torch.no_grad():
        representation = adapter.extract_representation(model, inputs)
    assert representation.pooled.dim() > 2, "the adapter output is a map"
    repaired = representation.pooled.flatten(start_dim=1)
    assert tuple(repaired.shape) == (2, EXPECTED_WIDTH["resnet"])
    # The width the rule picks is the flattened one, so state_proj matches.
    assert _hidden_width(model.config, model) == int(repaired.shape[-1])


#: ``(file, line, needle)`` for every source line the two rules cite. Pinned
#: so a transformers bump that moves one fails here instead of silently
#: invalidating the docstrings. Skipped when the vendor clone is absent.
CITED_LINES: Tuple[Tuple[str, int, str], ...] = (
    ("convnext/modeling_convnext.py", 304, "config.hidden_sizes[-1]"),
    ("resnet/modeling_resnet.py", 347, "nn.Flatten()"),
    ("resnet/modeling_resnet.py", 348, "config.hidden_sizes[-1]"),
    ("resnet/modeling_resnet.py", 297, "AdaptiveAvgPool2d((1, 1))"),
    ("segformer/modeling_segformer.py", 321, "config.hidden_sizes[stage_idx]"),
    ("segformer/modeling_segformer.py", 385, "config.num_encoder_blocks"),
    ("segformer/modeling_segformer.py", 285, "int(hidden_size * mlp_ratio)"),
    ("segformer/modeling_segformer.py", 211, "nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)"),
    ("mobilenet_v2/modeling_mobilenet_v2.py", 313, "self.conv_1x1 = MobileNetV2ConvLayer("),
    ("mobilenet_v2/modeling_mobilenet_v2.py", 354, "torch.flatten"),
    ("mobilenet_v2/modeling_mobilenet_v2.py", 381, "conv_1x1.convolution.out_channels"),
    ("poolformer/modeling_poolformer.py", 326, "config.hidden_sizes[-1]"),
    ("convnext/configuration_convnext.py", 49, "hidden_sizes"),
    ("resnet/configuration_resnet.py", 57, "hidden_sizes"),
    ("segformer/configuration_segformer.py", 63, "hidden_sizes"),
)


@pytest.mark.parametrize("relative, line_number, needle", CITED_LINES)
def test_cited_source_lines_are_stable(relative: str, line_number: int, needle: str) -> None:
    """Each rule cites a real line of the vendored transformers clone."""
    path = VENDOR_MODELS / relative
    if not path.is_file():
        pytest.skip(f"vendored transformers source not present: {path}")
    line = path.read_text(encoding="utf-8").splitlines()[line_number - 1]
    assert needle in line, f"{relative}:{line_number} is {line.strip()!r}, expected {needle!r}"


def test_mobilenet_config_declares_no_width_field() -> None:
    """The premise of rule 5: MobileNetV2Config has no width field to find."""
    config = MobileNetV2Config()
    for key in ("hidden_size", "d_model", "n_embd", "hidden_dim", "d_embed",
                "embed_dim", "model_dim", "dim", "projection_dim"):
        assert not _declares(config, key), key
    assert _hidden_width(config) is None
    assert _stage_width(config) is None

