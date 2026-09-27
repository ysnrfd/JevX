# Copyright (c) 2026 ysnrfd
"""Adapter tests on tiny in-memory transformers models (no downloads).

Each family is exercised end-to-end — ``resolve_adapter`` →
``prepare_inputs`` → ``extract_representation`` → ``validate`` — with the
chosen representation ``field`` asserted against the contract, plus the
decoder left-pad/right-pad equivalence property and the input-validation
error paths. See ``test_adapter_real_models.py`` for hub checkpoints.
"""

from __future__ import annotations

import pytest
import torch
from transformers import BertModel, LlamaModel, T5Model, ViTModel, Wav2Vec2Model

from jevx.adapters import (
    AudioAdapter,
    DecoderAdapter,
    EncoderAdapter,
    EncoderDecoderAdapter,
    GenericAdapter,
    Representation,
    VisionAdapter,
    available_adapters,
    register_adapter,
    resolve_adapter,
)
from jevx.adapters.registry import USER_ADAPTERS

TEXT = "hello world foo bar"


# --- family resolution -------------------------------------------------------


def test_resolve_picks_family_adapters(
    tiny_bert: BertModel,
    tiny_llama: LlamaModel,
    tiny_t5: T5Model,
    tiny_vit: ViTModel,
    tiny_wav2vec2: Wav2Vec2Model,
) -> None:
    assert isinstance(resolve_adapter(tiny_bert), EncoderAdapter)
    assert isinstance(resolve_adapter(tiny_llama), DecoderAdapter)
    assert isinstance(resolve_adapter(tiny_t5), EncoderDecoderAdapter)
    assert isinstance(resolve_adapter(tiny_vit), VisionAdapter)
    assert isinstance(resolve_adapter(tiny_wav2vec2), AudioAdapter)
    assert resolve_adapter(tiny_bert).name == "encoder"
    assert resolve_adapter(tiny_llama).name == "decoder"
    assert resolve_adapter(tiny_t5).name == "encoder_decoder"


def test_available_adapters_lists_chain() -> None:
    names = available_adapters()
    for expected in ("multimodal", "encoder_decoder", "vision", "audio", "decoder", "encoder", "generic"):
        assert expected in names
    assert names[-1] == "generic"  # universal fallback is last


# --- bert: pooler/native -----------------------------------------------------


def test_bert_native_pooler_extraction(tiny_bert: BertModel, tiny_tokenizer) -> None:
    adapter = resolve_adapter(tiny_bert)
    inputs = adapter.prepare_inputs(TEXT, tokenizer=tiny_tokenizer, model=tiny_bert)
    rep = adapter.extract_representation(tiny_bert, inputs)
    assert isinstance(rep, Representation)
    assert rep.field == "pooler_output"  # chosen field == contract expectation
    assert rep.layout == "btd"  # sequence kept alongside the pooled vector
    assert rep.pooled is not None and rep.pooled.shape == (1, 32)
    assert rep.sequence_output is not None and rep.sequence_output.dim() == 3
    assert rep.attention_mask is not None
    with torch.no_grad():
        expected = tiny_bert(**inputs).pooler_output
    assert torch.equal(rep.pooled, expected)
    adapter.validate(tiny_bert, inputs)  # must not raise


def test_bert_explicit_strategy_overrides_native(tiny_bert: BertModel, tiny_tokenizer) -> None:
    adapter = resolve_adapter(tiny_bert)
    inputs = adapter.prepare_inputs(TEXT, tokenizer=tiny_tokenizer, model=tiny_bert)
    rep = adapter.extract_representation(tiny_bert, inputs, strategy="cls")
    assert rep.field == "last_hidden_state"  # sequence path chosen deliberately
    assert rep.pooled is not None and rep.pooled.shape == (1, 32)
    assert torch.equal(rep.pooled, rep.sequence_output[:, 0, :])  # mask all-ones


# --- llama: decoder left-pad / right-pad equivalence --------------------------


def _decoder_representations(llama: LlamaModel, tokenizer, adapter, text: str, side: str, width: int | None):
    tokenizer.padding_side = side
    if width is None:
        batch = tokenizer(text, return_tensors="pt")
    else:
        batch = tokenizer(text, return_tensors="pt", padding="max_length", max_length=width)
    return adapter.extract_representation(llama, dict(batch))


def test_decoder_field_and_padding_equivalence(tiny_llama: LlamaModel, tiny_tokenizer) -> None:
    adapter = resolve_adapter(tiny_llama)
    assert isinstance(adapter, DecoderAdapter)

    unpadded = _decoder_representations(tiny_llama, tiny_tokenizer, adapter, TEXT, "right", None)
    left = _decoder_representations(tiny_llama, tiny_tokenizer, adapter, TEXT, "left", 12)
    right = _decoder_representations(tiny_llama, tiny_tokenizer, adapter, TEXT, "right", 12)

    assert unpadded.field == "last_hidden_state"  # contract expectation
    assert unpadded.pooled is not None and unpadded.pooled.shape == (1, 32)
    # left-pad == right-pad == unpadded row (last_valid pooling, P4.3)
    assert torch.allclose(left.pooled, unpadded.pooled, atol=1e-5)
    assert torch.allclose(right.pooled, unpadded.pooled, atol=1e-5)
    assert torch.allclose(left.pooled, right.pooled, atol=1e-6)
    # extraction never allocates a KV cache (audit §5/§8.7)
    assert getattr(unpadded.raw_output, "past_key_values", "missing") is None
    adapter.validate(tiny_llama, dict(tiny_tokenizer(TEXT, return_tensors="pt")))


def test_decoder_position_ids_derivation_hook(tiny_llama: LlamaModel) -> None:
    adapter = DecoderAdapter()
    fwd = {
        "input_ids": torch.ones(1, 4, dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 0, 0]]),
    }
    out = adapter._postprocess_forward_kwargs(tiny_llama, dict(fwd))
    # cumsum(mask) - 1: real tokens get 0..L-1 regardless of pad side
    assert torch.equal(out["position_ids"], torch.tensor([[0, 1, 1, 1]]))
    left = adapter._postprocess_forward_kwargs(
        tiny_llama,
        {
            "input_ids": torch.ones(1, 4, dtype=torch.long),
            "attention_mask": torch.tensor([[0, 0, 1, 1]]),
        },
    )
    assert torch.equal(left["position_ids"], torch.tensor([[0, 0, 0, 1]]))
    # an explicit position_ids is respected untouched
    explicit = {**fwd, "position_ids": torch.tensor([[5, 6, 7, 8]])}
    out2 = adapter._postprocess_forward_kwargs(tiny_llama, dict(explicit))
    assert torch.equal(out2["position_ids"], torch.tensor([[5, 6, 7, 8]]))


# --- t5: encoder-decoder -----------------------------------------------------


def test_t5_encoder_state_equals_direct_encoder_call(tiny_t5: T5Model, tiny_tokenizer) -> None:
    adapter = resolve_adapter(tiny_t5)
    inputs = adapter.prepare_inputs(TEXT, tokenizer=tiny_tokenizer, model=tiny_t5)
    assert "decoder_input_ids" in inputs  # mirrored for full-forward modes (probe)
    tiny_t5.eval()
    with torch.no_grad():
        rep = adapter.extract_representation(tiny_t5, inputs)  # default: encoder_state
        expected = tiny_t5.encoder(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        ).last_hidden_state
    assert rep.field == "encoder_last_hidden_state"  # contract expectation
    assert rep.sequence_output is not None
    assert torch.allclose(rep.sequence_output, expected, atol=1e-6)
    assert rep.pooled is not None and rep.pooled.shape == (1, 32)
    assert rep.attention_mask is not None
    adapter.validate(tiny_t5, inputs)


def test_t5_decoder_state_mode(tiny_t5: T5Model, tiny_tokenizer) -> None:
    adapter = resolve_adapter(tiny_t5)
    inputs = adapter.prepare_inputs(TEXT, tokenizer=tiny_tokenizer, model=tiny_t5)
    with torch.no_grad():
        rep = adapter.extract_representation(tiny_t5, inputs, mode="decoder_state")
    assert rep.field == "last_hidden_state"  # decoder side of Seq2SeqModelOutput
    assert rep.sequence_output is not None and rep.sequence_output.dim() == 3
    assert rep.pooled is not None and rep.pooled.shape == (1, 32)


def test_t5_encoder_decoder_fusion_mode(tiny_t5: T5Model, tiny_tokenizer) -> None:
    adapter = resolve_adapter(tiny_t5)
    inputs = adapter.prepare_inputs(TEXT, tokenizer=tiny_tokenizer, model=tiny_t5)
    with torch.no_grad():
        rep = adapter.extract_representation(tiny_t5, inputs, mode="encoder_decoder_fusion")
    assert rep.field == "encoder_decoder_fusion"
    assert rep.pooled is not None and rep.pooled.shape == (1, 32)
    assert torch.isfinite(rep.pooled).all()


def test_t5_unknown_mode_raises(tiny_t5: T5Model, tiny_tokenizer) -> None:
    adapter = resolve_adapter(tiny_t5)
    inputs = adapter.prepare_inputs(TEXT, tokenizer=tiny_tokenizer, model=tiny_t5)
    with pytest.raises(ValueError, match="encoder_state"):
        adapter.extract_representation(tiny_t5, inputs, mode="sideways")


# --- vit: vision -------------------------------------------------------------


def test_vit_native_pooler(tiny_vit: ViTModel) -> None:
    adapter = resolve_adapter(tiny_vit)
    inputs = {"pixel_values": torch.randn(2, 3, 32, 32)}
    rep = adapter.extract_representation(tiny_vit, inputs)
    assert rep.field == "pooler_output"  # contract expectation
    assert rep.pooled is not None and rep.pooled.shape == (2, 32)
    adapter.validate(tiny_vit, inputs)
    # explicit sequence strategy switches to last_hidden_state (post-norm),
    # NOT the pre-norm hidden_states[-1] (audit Q4 vision caveat)
    rep_cls = adapter.extract_representation(tiny_vit, inputs, strategy="cls")
    assert rep_cls.field == "last_hidden_state"
    assert torch.equal(rep_cls.pooled, rep_cls.sequence_output[:, 0, :])


# --- wav2vec2: audio ---------------------------------------------------------


def test_wav2vec2_last_hidden_state(tiny_wav2vec2: Wav2Vec2Model) -> None:
    adapter = resolve_adapter(tiny_wav2vec2)
    inputs = {"input_values": torch.randn(2, 3200)}
    rep = adapter.extract_representation(tiny_wav2vec2, inputs)
    assert rep.field == "last_hidden_state"  # contract expectation
    assert rep.pooled is not None and rep.pooled.shape == (2, 32)
    assert rep.layout == "btd"
    adapter.validate(tiny_wav2vec2, inputs)


def test_wav2vec2_tensor_state_maps_to_main_input(tiny_wav2vec2: Wav2Vec2Model) -> None:
    adapter = resolve_adapter(tiny_wav2vec2)
    inputs = adapter.prepare_inputs(torch.randn(1, 3200), model=tiny_wav2vec2)
    assert list(inputs) == ["input_values"]  # main_input_name


def test_wav2vec2_frame_mask_realigned_to_sequence(tiny_wav2vec2: Wav2Vec2Model) -> None:
    """Input-length mask is downsampled to the frame axis via the family hook."""
    adapter = resolve_adapter(tiny_wav2vec2)
    inputs = {
        "input_values": torch.randn(2, 3200),
        "attention_mask": torch.ones(2, 3200, dtype=torch.long),
    }
    rep = adapter.extract_representation(tiny_wav2vec2, inputs)
    assert rep.sequence_output is not None
    assert rep.attention_mask is not None
    assert tuple(rep.attention_mask.shape) == tuple(rep.sequence_output.shape[:2])


# --- input preparation & validation errors -----------------------------------


def test_prepare_rejects_undeclared_keys(tiny_bert: BertModel) -> None:
    adapter = resolve_adapter(tiny_bert)
    with pytest.raises(ValueError, match="not accepted"):
        adapter.prepare_inputs(
            {"pixel_values": torch.randn(1, 3, 8, 8)}, model=tiny_bert
        )


def test_prepare_text_requires_tokenizer(tiny_bert: BertModel) -> None:
    adapter = resolve_adapter(tiny_bert)
    with pytest.raises(ValueError, match="tokenizer"):
        adapter.prepare_inputs(TEXT)


def test_prepare_tensor_requires_model(tiny_bert: BertModel) -> None:
    with pytest.raises(ValueError, match="main_input_name"):
        GenericAdapter().prepare_inputs(torch.randn(1, 5))


def test_prepare_rejects_unknown_state_type() -> None:
    with pytest.raises(TypeError, match="state must be"):
        GenericAdapter().prepare_inputs(42)


def test_prepare_dict_passthrough_validated(tiny_bert: BertModel, tiny_tokenizer) -> None:
    adapter = resolve_adapter(tiny_bert)
    batch = dict(tiny_tokenizer(TEXT, return_tensors="pt"))
    out = adapter.prepare_inputs(dict(batch), model=tiny_bert)
    assert set(out) == set(batch)


def test_describe_capabilities_is_introspection_only(tiny_bert: BertModel) -> None:
    adapter = resolve_adapter(tiny_bert)
    report = adapter.describe_capabilities(tiny_bert)
    assert report.model_type == "bert"
    assert report.representation_field == "pooler_output"


# --- registry ----------------------------------------------------------------


def test_user_registered_adapter_wins() -> None:
    class FirstAdapter(GenericAdapter):
        name = "test.first"

        def can_handle(self, model: object) -> bool:
            return isinstance(model, BertModel)

    register_adapter(FirstAdapter)
    try:
        assert available_adapters()[0] == "test.first"
        assert isinstance(resolve_adapter(tiny_bert_stub()), FirstAdapter)
    finally:
        USER_ADAPTERS.remove(FirstAdapter)


def tiny_bert_stub() -> BertModel:
    """Minimal BertModel for the registry test (cheap, local)."""
    from transformers import BertConfig

    config = BertConfig(
        vocab_size=50,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=16,
    )
    return BertModel(config).eval()


def test_register_adapter_rejects_non_subclass() -> None:
    with pytest.raises(TypeError, match="BaseAdapter"):
        register_adapter(int)  # type: ignore[arg-type]
