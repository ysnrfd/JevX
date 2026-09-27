# Copyright (c) 2026 ysnrfd
"""``CapabilityReport.attention_layout`` — hybrid layer_types parsing (Known #1).

Pins the PRESENCE and the parsing of ``config.layer_types`` (previously the
detector was layer_types-blind and ``tests/integration/test_qwen35_hybrid.py``
recorded ``capability_mentions_layer_types=False``):

* the real Qwen3.5 hybrid layout, built locally from the verbatim config in
  ``vendor/reports/QWEN35_RECON.md`` (24 entries: 18 ``linear_attention`` +
  6 ``full_attention``), whose list lives in the NESTED ``text_config`` —
  no download, no checkpoint needed;
* uniform layouts (``"full"`` / ``"linear"``);
* configs that declare no ``layer_types`` (plain ``BertConfig``, ``LlamaConfig``)
  → ``None`` — parse-only design: nothing is ever inferred;
* config-order aggregation of counts and the hybrid ``reasons`` line.

The detector's conversion verdicts must be untouched by this informational
field (no upgrades/downgrades without execution evidence).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn
from transformers import BertConfig, LlamaConfig
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5TextConfig,
)

from jevx.capabilities import JevXCapabilityDetector

#: ``text_config`` verbatim from vendor/reports/QWEN35_RECON.md ("Config
#: verbatim" block) — Qwen3.5-0.8B, 24 layers, layer_types [L,L,L,F] x 6.
QWEN35_TEXT_CONFIG = {
    "attention_bias": False,
    "attention_dropout": 0.0,
    "attn_output_gate": True,
    "dtype": "bfloat16",
    "eos_token_id": 248044,
    "full_attention_interval": 4,
    "head_dim": 256,
    "hidden_act": "silu",
    "hidden_size": 1024,
    "initializer_range": 0.02,
    "intermediate_size": 3584,
    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 6,
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 16,
    "linear_value_head_dim": 128,
    "max_position_embeddings": 262144,
    "mlp_only_layers": [],
    "model_type": "qwen3_5_text",
    "mtp_num_hidden_layers": 1,
    "mtp_use_dedicated_embeddings": False,
    "num_attention_heads": 8,
    "num_hidden_layers": 24,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-06,
    "tie_word_embeddings": True,
    "use_cache": True,
    "vocab_size": 248320,
    "mamba_ssm_dtype": "float32",
    "rope_parameters": {
        "mrope_interleaved": True,
        "mrope_section": [11, 11, 10],
        "rope_type": "default",
        "rope_theta": 10000000,
        "partial_rotary_factor": 0.25,
    },
}

#: ``vision_config`` verbatim from the same RECON block.
QWEN35_VISION_CONFIG = {
    "deepstack_visual_indexes": [],
    "depth": 12,
    "hidden_act": "gelu_pytorch_tanh",
    "hidden_size": 768,
    "in_channels": 3,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "model_type": "qwen3_5",
    "num_heads": 12,
    "num_position_embeddings": 2304,
    "out_hidden_size": 1024,
    "patch_size": 16,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
}

#: The exact hybrid summary string the Qwen3.5 layout must parse to.
QWEN35_HYBRID_LAYOUT = "hybrid:18xlinear_attention+6xfull_attention"
#: The exact hybrid reasons line (spec wording).
QWEN35_HYBRID_REASON = (
    "hybrid attention layout: 18 linear_attention + 6 full_attention layers "
    "(config.layer_types)"
)


class _StubModel(nn.Module):
    """Model-like carrier for a config (the detector never runs forward).

    Forward mirrors ``Qwen3_5Model``'s declared text/vision inputs so the
    modality side of detection matches the real wrapper.
    """

    main_input_name = "input_ids"

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        raise RuntimeError("detect() must never execute the model")  # pragma: no cover


# torch is only needed for the type hints above; import for runtime safety.
import torch  # noqa: E402


def _qwen35_config() -> Qwen3_5Config:
    return Qwen3_5Config(
        architectures=["Qwen3_5ForConditionalGeneration"],
        image_token_id=248056,
        model_type="qwen3_5",
        text_config=dict(QWEN35_TEXT_CONFIG),
        vision_config=dict(QWEN35_VISION_CONFIG),
        tie_word_embeddings=True,
        video_token_id=248057,
        vision_end_token_id=248054,
        vision_start_token_id=248053,
    )


def test_qwen35_nested_text_config_parses_hybrid_layout() -> None:
    """Real Qwen3.5 config (layer_types nested in text_config) → hybrid."""
    config = _qwen35_config()
    # The nested lookup is genuinely needed: the top config has no attribute.
    assert not hasattr(config, "layer_types")
    assert len(config.text_config.layer_types) == 24

    report = JevXCapabilityDetector.detect(_StubModel(config))
    assert report.attention_layout == QWEN35_HYBRID_LAYOUT
    # Human-readable hybrid line surfaces in reasons (battery greps for
    # "layer_types" in reasons — previously always False).
    assert QWEN35_HYBRID_REASON in report.reasons
    assert any("layer_types" in reason for reason in report.reasons)
    # INFORMATIONAL ONLY: verdicts unchanged by this field (no upgrades).
    assert report.jevx_conversion == "experimental"
    assert report.representation_field == "last_hidden_state"
    assert report.decoder is True and report.vision is True
    # to_dict is dataclass-driven: the new key rides along, JSON-serializable.
    as_dict = report.to_dict()
    assert as_dict["attention_layout"] == QWEN35_HYBRID_LAYOUT
    json.dumps(as_dict)  # must not raise


def test_top_level_layer_types_parsed_without_nesting() -> None:
    """A config declaring layer_types on itself takes the direct path."""
    text_config = Qwen3_5TextConfig(**QWEN35_TEXT_CONFIG)
    report = JevXCapabilityDetector.detect(_StubModel(text_config))
    assert report.attention_layout == QWEN35_HYBRID_LAYOUT
    assert QWEN35_HYBRID_REASON in report.reasons


def test_uniform_full_attention_layout_is_full() -> None:
    config = Qwen3_5TextConfig(
        num_hidden_layers=4,
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        vocab_size=100,
        full_attention_interval=1,  # (i+1) % 1 == 0 -> every layer full_attention
    )
    assert set(config.layer_types) == {"full_attention"}
    report = JevXCapabilityDetector.detect(_StubModel(config))
    assert report.attention_layout == "full"
    # Uniform layouts are not hybrid: no hybrid reasons line.
    assert not any(r.startswith("hybrid attention layout") for r in report.reasons)


def test_uniform_linear_attention_layout_is_linear() -> None:
    config = Qwen3_5TextConfig(
        num_hidden_layers=6,
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        vocab_size=100,
        layer_types=["linear_attention"] * 6,
    )
    report = JevXCapabilityDetector.detect(_StubModel(config))
    assert report.attention_layout == "linear"
    assert not any(r.startswith("hybrid attention layout") for r in report.reasons)


def test_nonstandard_uniform_layout_uses_count_form() -> None:
    """Uniform-but-not-full/linear types still report their declared type."""
    config = Qwen3_5TextConfig(
        num_hidden_layers=3,
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        vocab_size=100,
        layer_types=["sliding_attention"] * 3,
    )
    report = JevXCapabilityDetector.detect(_StubModel(config))
    assert report.attention_layout == "hybrid:3xsliding_attention"


def test_counts_aggregate_in_config_order_not_sorted() -> None:
    """First-seen order is preserved (full first stays full first)."""
    config = Qwen3_5TextConfig(
        num_hidden_layers=3,
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        vocab_size=100,
        layer_types=["full_attention", "linear_attention", "full_attention"],
    )
    report = JevXCapabilityDetector.detect(_StubModel(config))
    assert report.attention_layout == "hybrid:2xfull_attention+1xlinear_attention"
    assert (
        "hybrid attention layout: 2 full_attention + 1 linear_attention layers "
        "(config.layer_types)"
    ) in report.reasons


def test_plain_bert_config_has_no_layer_types() -> None:
    """Parse-only: BertConfig declares no layer_types -> None, never 'full'."""
    report = JevXCapabilityDetector.detect(_StubModel(BertConfig()))
    assert report.attention_layout is None
    assert not any("layer_types" in reason for reason in report.reasons)


def test_llama_config_has_no_layer_types() -> None:
    report = JevXCapabilityDetector.detect(_StubModel(LlamaConfig(num_hidden_layers=1)))
    assert report.attention_layout is None


def test_empty_or_non_string_layer_types_is_none() -> None:
    # Empty / None through the real strict config class...
    for bad in ([], None):
        config = Qwen3_5TextConfig(
            num_hidden_layers=2,
            hidden_size=32,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=64,
            vocab_size=100,
        )
        config.layer_types = bad
        report = JevXCapabilityDetector.detect(_StubModel(config))
        assert report.attention_layout is None, bad
    # ...malformed shapes a plain config object can hold (strict configs
    # reject them at assignment; the detector's guard is defensive).
    for bad in ([1, 2, 3], "full_attention", 42):
        config = SimpleNamespace(
            model_type="odd",
            architectures=["OddModel"],
            layer_types=bad,
        )
        report = JevXCapabilityDetector.detect(_StubModel(config))
        assert report.attention_layout is None, bad
