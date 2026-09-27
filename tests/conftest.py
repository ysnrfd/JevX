# Copyright (c) 2026 ysnrfd
"""Shared pytest fixtures for JevX unit tests.

Created because ``tests/conftest.py`` did not exist; fixtures are namespaced
(``tiny_*``) to avoid collisions with other suites. All models are tiny
in-memory ``transformers`` constructions (no downloads, CPU-friendly) in
``eval()`` mode — tests treat them as read-only. The tokenizer fixture is
function-scoped because tests mutate ``padding_side``.
"""

from __future__ import annotations

import pytest
import torch
from tokenizers import Tokenizer, models as tokenizers_models, pre_tokenizers
from transformers import (
    BertConfig,
    BertModel,
    LlamaConfig,
    LlamaModel,
    PreTrainedTokenizerFast,
    T5Config,
    T5Model,
    ViTConfig,
    ViTModel,
    Wav2Vec2Config,
    Wav2Vec2Model,
)

#: Word-level vocabulary covering the English words used in tests; anything
#: else maps to ``[UNK]``. Built locally so no tokenizer download is needed.
TINY_VOCAB = {
    "[UNK]": 0,
    "[PAD]": 1,
    "hello": 2,
    "world": 3,
    "foo": 4,
    "bar": 5,
    "baz": 6,
    "the": 7,
    "cat": 8,
    "sat": 9,
    "a": 10,
    "b": 11,
    "c": 12,
    "question": 13,
    "option": 14,
}


@pytest.fixture()
def tiny_tokenizer() -> PreTrainedTokenizerFast:
    """In-memory WordLevel tokenizer with ``[PAD]`` (padding-safe)."""
    backend = Tokenizer(tokenizers_models.WordLevel(dict(TINY_VOCAB), unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]")


@pytest.fixture(scope="session")
def tiny_bert() -> BertModel:
    """2-layer BERT encoder with pooler (hidden 32), eval mode."""
    config = BertConfig(
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    return BertModel(config).eval()


@pytest.fixture(scope="session")
def tiny_llama() -> LlamaModel:
    """2-layer decoder-only Llama stack (hidden 32), eval mode, no pooler."""
    config = LlamaConfig(
        vocab_size=99,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    torch.manual_seed(1)
    return LlamaModel(config).eval()


@pytest.fixture(scope="session")
def tiny_t5() -> T5Model:
    """2-layer T5 encoder-decoder (d_model 32), eval mode."""
    config = T5Config(
        vocab_size=99,
        d_model=32,
        d_kv=8,
        d_ff=64,
        num_layers=2,
        num_heads=4,
    )
    torch.manual_seed(2)
    return T5Model(config).eval()


@pytest.fixture(scope="session")
def tiny_vit() -> ViTModel:
    """2-layer ViT (hidden 32, 32x32 images, patch 8), eval mode, with pooler."""
    config = ViTConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        image_size=32,
        patch_size=8,
        num_channels=3,
    )
    torch.manual_seed(3)
    return ViTModel(config).eval()


@pytest.fixture(scope="session")
def tiny_wav2vec2() -> Wav2Vec2Model:
    """2-layer Wav2Vec2 base model (hidden 32), eval mode, no pooler."""
    config = Wav2Vec2Config(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
    )
    torch.manual_seed(4)
    return Wav2Vec2Model(config).eval()
