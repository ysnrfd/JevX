# Copyright (c) 2026 ysnrfd
"""Question encoder (E_Q) and schema encoder (E_C) tests.

Covers both text paths (hashed bag-of-tokens fallback; frozen-backbone
path), the trainable residual transform, variable-``K`` schema batching with
bool masks aligned to ``DecisionHead.option_mask``, and the documented error
cases (empty question, ``K = 0``).
"""

from __future__ import annotations

import pytest
import torch
from transformers import BertConfig, BertModel

from jevx.decisions import ChoiceHead
from jevx.questions import Option, QuestionEncoder, SchemaEncoder

torch.manual_seed(0)

HIDDEN = 16


# --- QuestionEncoder: hashed path -------------------------------------------


def test_hashed_projection_shapes_and_determinism() -> None:
    encoder = QuestionEncoder(hidden_size=HIDDEN)  # method="projection" default
    texts = ["hello world", "the cat sat"]
    first = encoder.encode_texts(texts, tokenizer=None)
    second = encoder.encode_texts(texts, tokenizer=None)
    assert first.shape == (2, HIDDEN)
    assert torch.equal(first, second)  # deterministic per string
    other = encoder.encode_texts(["foo bar baz", "hello"], tokenizer=None)
    assert other.shape == (2, HIDDEN)
    assert not torch.allclose(first[0], other[1])  # different text, different hash


def test_hashed_path_accepts_single_string() -> None:
    encoder = QuestionEncoder(hidden_size=HIDDEN)
    out = encoder.encode_texts("hello world", tokenizer=None)  # type: ignore[arg-type]
    assert out.shape == (1, HIDDEN)


def test_question_forward_residual_transform() -> None:
    encoder = QuestionEncoder(hidden_size=HIDDEN)
    h = torch.randn(3, HIDDEN, requires_grad=True)
    out = encoder(h)
    assert out.shape == (3, HIDDEN)
    out.sum().backward()
    assert h.grad is not None  # residual keeps the input gradient path
    assert all(p.grad is not None for p in encoder.transform.parameters())


def test_empty_or_whitespace_question_raises() -> None:
    encoder = QuestionEncoder(hidden_size=HIDDEN)
    with pytest.raises(ValueError, match="empty or whitespace"):
        encoder.encode_texts(["   "], tokenizer=None)
    with pytest.raises(ValueError, match="empty or whitespace"):
        encoder.encode_texts(["hello", ""], tokenizer=None)
    with pytest.raises(ValueError, match="at least one question"):
        encoder.encode_texts([], tokenizer=None)
    with pytest.raises(TypeError, match="str"):
        encoder.encode_texts([123], tokenizer=None)  # type: ignore[list-item]


def test_invalid_method_and_widths_raise() -> None:
    with pytest.raises(ValueError, match="hidden_size"):
        QuestionEncoder(hidden_size=0)
    with pytest.raises(ValueError, match="method"):
        QuestionEncoder(hidden_size=HIDDEN, method="telepathy")
    with pytest.raises(ValueError, match="backbone_hidden"):
        QuestionEncoder(hidden_size=HIDDEN, method="backbone")  # missing width


# --- QuestionEncoder: backbone path -----------------------------------------


def _fresh_bert() -> BertModel:
    config = BertConfig(
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    torch.manual_seed(7)
    return BertModel(config).eval()


def test_backbone_path_shapes(tiny_bert: BertModel, tiny_tokenizer) -> None:
    encoder = QuestionEncoder(hidden_size=HIDDEN, backbone_hidden=32, method="backbone")
    out = encoder.encode_texts(
        ["hello world", "foo bar baz"], tokenizer=tiny_tokenizer, backbone=tiny_bert
    )
    assert out.shape == (2, HIDDEN)


def test_frozen_backbone_produces_no_gradients(tiny_bert: BertModel, tiny_tokenizer) -> None:
    encoder = QuestionEncoder(hidden_size=HIDDEN, backbone_hidden=32, method="backbone")
    out = encoder.encode_texts(
        ["hello world"], tokenizer=tiny_tokenizer, backbone=tiny_bert, frozen=True
    )
    # backbone ran under no_grad: no parameter ever received a .grad
    assert all(p.grad is None for p in tiny_bert.parameters())
    # ...while the projection/transform side stays trainable
    out.sum().backward()
    assert encoder.project is not None
    assert any(p.grad is not None for p in encoder.project.parameters())
    assert all(p.grad is None for p in tiny_bert.parameters())


def test_non_frozen_backbone_receives_gradients(tiny_tokenizer) -> None:
    bert = _fresh_bert()
    encoder = QuestionEncoder(hidden_size=HIDDEN, backbone_hidden=32, method="backbone")
    out = encoder.encode_texts(
        ["hello world"], tokenizer=tiny_tokenizer, backbone=bert, frozen=False
    )
    out.sum().backward()
    assert any(p.grad is not None for p in bert.parameters())
    bert.zero_grad(set_to_none=True)


def test_backbone_path_requires_backbone_and_tokenizer(tiny_tokenizer) -> None:
    encoder = QuestionEncoder(hidden_size=HIDDEN, backbone_hidden=32, method="backbone")
    with pytest.raises(ValueError, match="requires backbone"):
        encoder.encode_texts(["hello"], tokenizer=tiny_tokenizer, backbone=None)
    with pytest.raises(ValueError, match="requires tokenizer"):
        encoder.encode_texts(["hello"], tokenizer=None, backbone=_fresh_bert())


def test_backbone_hidden_mismatch_raises(tiny_tokenizer) -> None:
    encoder = QuestionEncoder(hidden_size=HIDDEN, backbone_hidden=64, method="backbone")
    with pytest.raises(ValueError, match="backbone_hidden"):
        encoder.encode_texts(["hello"], tokenizer=tiny_tokenizer, backbone=_fresh_bert())


def test_forward_rejects_wrong_width() -> None:
    encoder = QuestionEncoder(hidden_size=HIDDEN)
    with pytest.raises(ValueError, match="question_repr"):
        encoder(torch.randn(2, HIDDEN + 1))


# --- SchemaEncoder -----------------------------------------------------------


def test_option_text_variants() -> None:
    assert Option("k", "label", "desc").text() == "label: desc"
    assert Option("k", "label").text() == "label"
    assert Option("k", description="desc").text() == "desc"
    assert Option("k").text() == "k"  # falls back to the key, never empty
    assert Option("k").metadata == {}


def test_variable_k_batch_shapes_and_masks() -> None:
    encoder = SchemaEncoder(hidden_size=HIDDEN)
    batch = [
        [Option("a", "a"), Option("b", "b"), Option("c", "c")],
        [Option("d", "d")],
    ]
    embeddings, mask = encoder.encode(batch, tokenizer=None)
    assert embeddings.shape == (2, 3, HIDDEN)
    assert mask.dtype == torch.bool
    assert mask.tolist() == [
        [True, True, True],
        [True, False, False],
    ]
    # padded rows are exactly zero
    assert (embeddings[1, 1:] == 0).all()


def test_k_zero_rejected() -> None:
    encoder = SchemaEncoder(hidden_size=HIDDEN)
    with pytest.raises(ValueError, match="K=0"):
        encoder.encode([[Option("a")], []], tokenizer=None)
    with pytest.raises(ValueError, match="non-empty list"):
        encoder.encode([], tokenizer=None)


def test_k_one_allowed() -> None:
    encoder = SchemaEncoder(hidden_size=HIDDEN)
    embeddings, mask = encoder.encode([[Option("only", "one")]], tokenizer=None)
    assert embeddings.shape == (1, 1, HIDDEN)
    assert mask.tolist() == [[True]]


def test_duplicate_options_encode_identically() -> None:
    encoder = SchemaEncoder(hidden_size=HIDDEN)
    embeddings, mask = encoder.encode(
        [[Option("x", "same text"), Option("y", "same text")]], tokenizer=None
    )
    assert mask.all()
    assert torch.equal(embeddings[0, 0], embeddings[0, 1])


def test_non_option_entries_raise() -> None:
    encoder = SchemaEncoder(hidden_size=HIDDEN)
    with pytest.raises(TypeError, match="Option"):
        encoder.encode([["just a string"]], tokenizer=None)  # type: ignore[list-item]
    with pytest.raises(TypeError, match="list of Option"):
        encoder.encode([Option("a")], tokenizer=None)  # type: ignore[list-item]


def test_schema_mask_aligns_with_decision_head() -> None:
    encoder = SchemaEncoder(hidden_size=HIDDEN)
    batch = [[Option("a", "a"), Option("b", "b"), Option("c", "c")], [Option("d", "d")]]
    option_features, option_mask = encoder.encode(batch, tokenizer=None)
    head = ChoiceHead(hidden_size=HIDDEN, interaction="dot")
    fused = torch.randn(2, HIDDEN)
    out = head(fused, option_features, option_mask=option_mask)
    assert out.logits.shape == (2, 3)
    assert out.probabilities is not None
    assert torch.allclose(out.probabilities.sum(dim=-1), torch.ones(2), atol=1e-6)
    assert (out.probabilities[1, 1:] == 0).all()  # padded options carry no mass
    # row 1 has exactly one valid option, so its prediction is forced to 0
    assert int(out.predicted_index[1]) == 0
    assert 0 <= int(out.predicted_index[0]) <= 2


def test_schema_backbone_path(tiny_bert, tiny_tokenizer) -> None:
    encoder = SchemaEncoder(hidden_size=HIDDEN, backbone_hidden=32, method="backbone")
    batch = [[Option("a", "hello world"), Option("b", "foo bar")], [Option("c", "the cat")]]
    embeddings, mask = encoder.encode(
        batch, tokenizer=tiny_tokenizer, backbone=tiny_bert
    )
    assert embeddings.shape == (2, 2, HIDDEN)
    assert mask.tolist() == [[True, True], [True, False]]
    assert all(p.grad is None for p in tiny_bert.parameters())  # frozen by default


def test_schema_projection_is_independent_of_question_encoder() -> None:
    """E_C owns its parameters: not shared with any QuestionEncoder instance."""
    question_encoder = QuestionEncoder(hidden_size=HIDDEN)
    schema_encoder = SchemaEncoder(hidden_size=HIDDEN)
    question_params = {id(p) for p in question_encoder.parameters()}
    schema_params = {id(p) for p in schema_encoder.parameters()}
    assert question_params.isdisjoint(schema_params)
