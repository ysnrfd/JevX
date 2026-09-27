# Copyright (c) 2026 ysnrfd
"""Long-sequence numerical edges (JevX phase 72).

Contract under test:

* token-sequence states of length 8 / 64 / 512 that lie WITHIN the tiny
  model's learned position limit run end to end: probabilities finite, rows
  sum to 1, no NaN in any representation;
* a state where 512 EXCEEDS the backbone's learned position limit follows
  the documented behavior: a descriptive ValueError naming both lengths
  (never a silent truncation, never a cryptic kernel RuntimeError — see
  reports/FAILURE_ANALYSIS.md row 6).

Fixtures are built in-memory (no network): a BERT encoder with
max_position_embeddings=512 for the within-limit sweep, the shared tiny_bert
(max_position_embeddings=64) for the exceeding case, and a GPT-2 decoder with
n_positions=64 as the decoder-family cross-check.
"""

from __future__ import annotations

import copy

import pytest
import torch
from transformers import BertConfig, BertModel, GPT2Config, GPT2Model

from jevx import JevXConfig, JevXForDecision

HIDDEN = 32
LENGTHS_WITHIN_LIMIT = (8, 64, 512)


def build_wrapper(backbone, tokenizer=None, **config_overrides) -> JevXForDecision:
    overrides = dict(
        decision_hidden_size=HIDDEN,
        question_hidden_size=HIDDEN,
        schema_hidden_size=HIDDEN,
    )
    overrides.update(config_overrides)
    wrapper = JevXForDecision(
        JevXConfig(**overrides),
        backbone=copy.deepcopy(backbone),
        tokenizer=tokenizer,
    )
    wrapper.eval()
    return wrapper


@pytest.fixture(scope="module")
def bert_512() -> BertModel:
    """2-layer BERT whose learned position table accepts 512 tokens."""
    config = BertConfig(
        vocab_size=512,
        hidden_size=HIDDEN,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=512,
    )
    torch.manual_seed(11)
    return BertModel(config).eval()


@pytest.fixture(scope="module")
def gpt2_64() -> GPT2Model:
    """2-layer GPT-2 (decoder family) with n_positions=64."""
    config = GPT2Config(
        vocab_size=99,
        n_embd=HIDDEN,
        n_layer=2,
        n_head=4,
        n_positions=64,
    )
    torch.manual_seed(12)
    return GPT2Model(config).eval()


def _state_batch(length: int, vocab: int, seed: int) -> dict:
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(2, vocab, (1, length), generator=generator)
    return {"input_ids": ids, "attention_mask": torch.ones(1, length, dtype=torch.long)}


@pytest.mark.parametrize("length", LENGTHS_WITHIN_LIMIT)
def test_long_sequence_within_position_limit_runs_without_nan(bert_512, length) -> None:
    wrapper = build_wrapper(bert_512)
    vocab = int(bert_512.config.vocab_size)
    with torch.no_grad():
        out = wrapper.choice(
            _state_batch(length, vocab, seed=length),
            ["q1", "q2"],
            options=[["a", "b", "c"], ["a", "b", "c"]],
        )
    assert len(out.decisions) == 2
    for decision in out.decisions:
        probs = decision.probabilities
        assert torch.isfinite(probs).all(), f"non-finite probabilities at length {length}"
        assert not torch.isnan(probs).any()
        assert float(probs.sum()) == pytest.approx(1.0, abs=1e-6)
        assert torch.isfinite(decision.confidence).all()
    assert torch.isfinite(out.state_representation).all()
    assert torch.isfinite(out.question_representations).all()
    assert torch.isfinite(out.schema_representations).all()


def test_long_sequence_exceeding_position_limit_raises_documented_error(
    tiny_bert,
) -> None:
    """tiny_bert limit is 64: a 512-token state must hit the documented error."""
    wrapper = build_wrapper(tiny_bert)
    vocab = int(tiny_bert.config.vocab_size)
    with pytest.raises(ValueError) as excinfo:
        with torch.no_grad():
            wrapper.choice(_state_batch(512, vocab, seed=512), ["q"],
                           options=[["a", "b"]])
    message = str(excinfo.value)
    assert "state token length 512 exceeds" in message
    assert "learned position limit 64" in message
    # actionable remedy present
    assert "truncate" in message


def test_decoder_long_sequence_within_and_exceeding(gpt2_64) -> None:
    """Decoder family: at-limit runs, past-limit raises the same contract."""
    wrapper = build_wrapper(gpt2_64)
    vocab = int(gpt2_64.config.vocab_size)
    with torch.no_grad():
        out = wrapper.choice(_state_batch(64, vocab, seed=64), ["q"],
                             options=[["a", "b"]])
    probs = out.decisions[0].probabilities
    assert torch.isfinite(probs).all()
    assert float(probs.sum()) == pytest.approx(1.0, abs=1e-6)

    with pytest.raises(ValueError, match="state token length 200 exceeds"):
        with torch.no_grad():
            wrapper.choice(_state_batch(200, vocab, seed=200), ["q"],
                           options=[["a", "b"]])


def test_position_check_absent_for_backbones_without_learned_table(
    tiny_llama,
) -> None:
    """Relative/RoPE backbones expose no learned position table -> no pre-check.

    tiny_llama uses RoPE: _learned_position_limit is None, so JevX does not
    impose an artificial token-length cap on such families.
    """
    wrapper = build_wrapper(tiny_llama)
    assert wrapper._learned_position_limit is None
