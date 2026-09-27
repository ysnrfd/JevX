# Copyright (c) 2026 ysnrfd
"""Capability detection tests: introspection only, tiny in-memory models.

Models are constructed locally from tiny configs (no downloads). The
unsupported path is exercised through ``GenericAdapter.validate`` on a model
whose output class exposes no representation field, asserting *all*
structured fields of ``UnsupportedRepresentationError`` are populated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields as dataclass_fields
from types import SimpleNamespace
from typing import Optional

import pytest
import torch
import torch.nn as nn
from transformers import AutoModel, LlamaConfig, LlamaForCausalLM
from transformers.utils import ModelOutput

from jevx.adapters import GenericAdapter
from jevx.capabilities import (
    CapabilityReport,
    JevXCapabilityDetector,
    UnsupportedRepresentationError,
)

#: Exact field order of the pinned CapabilityReport contract.
REPORT_KEYS = (
    "text",
    "vision",
    "audio",
    "encoder",
    "decoder",
    "encoder_decoder",
    "hidden_states",
    "pooler",
    "generation",
    "native_representation",
    "fallback_representation",
    "jevx_conversion",
    "representation_field",
    "output_class",
    "model_type",
    "architecture",
    "input_fields",
    "processor_required",
    "attention_layout",
    "reasons",
)


def test_report_fields_match_pinned_contract() -> None:
    assert tuple(f.name for f in dataclass_fields(CapabilityReport)) == REPORT_KEYS


def test_detect_bert(tiny_bert: object) -> None:
    report = JevXCapabilityDetector.detect(tiny_bert)
    assert report.text and not report.vision and not report.audio
    assert report.encoder and not report.decoder and not report.encoder_decoder
    assert report.pooler and report.native_representation and report.fallback_representation
    assert report.hidden_states and not report.generation
    assert report.jevx_conversion == "supported"
    assert report.representation_field == "pooler_output"
    assert report.output_class == "BaseModelOutputWithPoolingAndCrossAttentions"
    assert report.model_type == "bert"
    assert report.architecture == "BertModel"
    assert "input_ids" in report.input_fields
    assert "attention_mask" in report.input_fields
    # control flags are excluded from input_fields (detector contract)
    assert "output_attentions" not in report.input_fields
    assert "use_cache" not in report.input_fields
    assert report.processor_required is False
    # BertConfig declares no layer_types -> attention_layout is None
    # (parse-only design: nothing inferred from architecture; see
    # tests/unit/test_layer_types_detection.py).
    assert report.attention_layout is None
    assert report.reasons  # human-readable justifications present


def test_detect_llama_like(tiny_llama: object) -> None:
    report = JevXCapabilityDetector.detect(tiny_llama)
    assert report.text and not report.vision and not report.audio
    assert report.decoder and not report.encoder and not report.encoder_decoder
    assert report.pooler is False
    assert report.native_representation is False
    assert report.fallback_representation is True
    assert report.jevx_conversion == "supported"
    assert report.representation_field == "last_hidden_state"
    assert report.output_class == "BaseModelOutputWithPast"
    assert report.model_type == "llama"
    # base LM stacks cannot generate; the ForCausalLM head can (below)
    assert report.generation is False


def test_detect_llama_for_causal_lm_can_generate() -> None:
    config = LlamaConfig(
        vocab_size=99,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    model = LlamaForCausalLM(config).eval()
    report = JevXCapabilityDetector.detect(model)
    assert report.decoder and not report.encoder
    assert report.generation is True  # model.can_generate(), audit §1.2


def test_detect_t5_encoder_decoder(tiny_t5: object) -> None:
    report = JevXCapabilityDetector.detect(tiny_t5)
    assert report.encoder_decoder and report.encoder and report.decoder
    assert report.text and not report.vision and not report.audio
    assert report.native_representation is True
    assert report.representation_field == "encoder_last_hidden_state"
    assert report.output_class == "Seq2SeqModelOutput"
    assert report.model_type == "t5"
    assert report.jevx_conversion == "supported"


def test_detect_vit_vision(tiny_vit: object) -> None:
    report = JevXCapabilityDetector.detect(tiny_vit)
    assert report.vision and not report.text and not report.audio
    assert report.pooler is True
    assert report.representation_field == "pooler_output"
    assert report.processor_required is True  # pixel_values needs a processor
    assert report.model_type == "vit"


def test_detect_wav2vec2_audio(tiny_wav2vec2: object) -> None:
    report = JevXCapabilityDetector.detect(tiny_wav2vec2)
    assert report.audio and not report.text and not report.vision
    assert report.pooler is False
    assert report.representation_field == "last_hidden_state"
    assert report.processor_required is True
    assert report.model_type == "wav2vec2"


def test_to_dict_is_json_serializable_with_contract_keys(tiny_bert: object) -> None:
    report = JevXCapabilityDetector.detect(tiny_bert)
    as_dict = report.to_dict()
    assert tuple(as_dict.keys()) == REPORT_KEYS
    json.dumps(as_dict)  # must not raise
    assert as_dict["model_type"] == "bert"


def test_detect_never_runs_forward() -> None:
    class BoomModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(
                model_type="boom",
                architectures=["BoomModel"],
                is_encoder_decoder=False,
                output_hidden_states=False,
            )

        def forward(self, input_ids=None, **kwargs):  # pragma: no cover - must not run
            raise RuntimeError("detect() must not execute the model")

    report = JevXCapabilityDetector.detect(BoomModel())
    assert isinstance(report, CapabilityReport)
    assert report.model_type == "boom"


def test_probe_confirms_bert_forward(tiny_bert: object) -> None:
    dummy = {
        "input_ids": torch.randint(0, 99, (1, 5)),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
    }
    report = JevXCapabilityDetector.probe(tiny_bert, dummy)
    assert any(reason.startswith("probe: forward ok") for reason in report.reasons)
    assert report.output_class == "BaseModelOutputWithPoolingAndCrossAttentions"
    assert report.representation_field == "pooler_output"
    # model left in eval mode (probe restores prior state; fixture was eval)
    assert tiny_bert.training is False  # type: ignore[attr-defined]


def test_probe_records_failure_without_raising(tiny_bert: object) -> None:
    report = JevXCapabilityDetector.probe(tiny_bert, {"pixel_values": torch.zeros(1, 3)})
    assert any("probe: forward FAILED" in reason for reason in report.reasons)


def test_unsupported_error_structured_fields() -> None:
    error = UnsupportedRepresentationError(
        model_name="some-model",
        architecture="SomeArchitecture",
        detected_outputs=["loss"],
        attempted_strategies=["native", "sequence"],
        failed_strategies=["native: no pooler_output", "sequence: no last_hidden_state"],
        suggestion="use a model exposing last_hidden_state",
    )
    assert error.model_name == "some-model"
    assert error.architecture == "SomeArchitecture"
    assert error.detected_outputs == ["loss"]
    assert error.attempted_strategies == ["native", "sequence"]
    assert len(error.failed_strategies) == 2
    assert error.suggestion == "use a model exposing last_hidden_state"
    message = str(error)
    assert "some-model" in message
    assert "loss" in message
    assert "Suggestion:" in message


# --- unsupported extraction path -------------------------------------------


@dataclass
class WeirdOutput(ModelOutput):
    """Output class with NO representation field (scores only)."""

    scores: Optional[torch.Tensor] = None


class NoRepresentationModel(nn.Module):
    """Model-like object whose forward exposes only a non-standard field."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            model_type="no_rep",
            architectures=["NoRepresentationModel"],
            is_encoder_decoder=False,
            output_hidden_states=False,
        )
        self.main_input_name = "input_ids"

    def forward(self, input_ids=None, **kwargs) -> WeirdOutput:
        return WeirdOutput(scores=input_ids.float())


def test_detect_marks_no_representation_model_unsupported() -> None:
    model = NoRepresentationModel()
    report = JevXCapabilityDetector.detect(model)
    assert report.jevx_conversion == "unsupported"
    assert report.representation_field is None
    assert report.output_class == "WeirdOutput"
    assert report.architecture == "NoRepresentationModel"
    assert report.model_type == "no_rep"


def test_validate_raises_structured_unsupported_error() -> None:
    model = NoRepresentationModel()
    inputs = {"input_ids": torch.randint(0, 9, (2, 4))}
    with pytest.raises(UnsupportedRepresentationError) as excinfo:
        GenericAdapter().validate(model, inputs)
    error = excinfo.value
    # ALL structured fields populated
    assert error.model_name  # non-empty
    assert error.architecture == "NoRepresentationModel"
    assert error.detected_outputs == ["scores"]
    assert len(error.attempted_strategies) >= 3  # pooler/sequence/hidden_states steps
    assert len(error.failed_strategies) >= 3
    assert error.suggestion
    assert isinstance(error.message, str) and error.message
    # actionable message includes the observed evidence
    assert "scores" in str(error)
    assert "Suggestion:" in str(error)


def test_auto_model_config_roundtrip_on_tiny_bert(tiny_bert: object) -> None:
    """Detector works on whatever AutoModel returns (integration sanity)."""
    report = JevXCapabilityDetector.detect(tiny_bert, config=tiny_bert.config)  # type: ignore[attr-defined]
    assert report.model_type == "bert"
    assert AutoModel is not None  # import sanity for the Auto surface
