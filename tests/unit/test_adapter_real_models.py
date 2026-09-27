# Copyright (c) 2026 ysnrfd
"""End-to-end adapter tests against REAL hub checkpoints (downloads).

Checkpoint IDs (revisions: each repository's default branch ``main``; the
resolved commit hash is recorded by transformers in ``config._commit_hash``
after load):

* ``hf-internal-testing/tiny-random-bert`` — encoder family, native pooler.
* ``hf-internal-testing/tiny-random-gpt2`` — decoder-only family; on load
  failure the documented fallback ``sshleifer/tiny-gpt2`` is used instead.

Each test runs the full contract: ``detect`` → ``prepare_inputs`` (with the
real tokenizer) → ``extract_representation`` → ``validate``.
"""

from __future__ import annotations

import pytest
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from jevx.adapters import BaseAdapter, resolve_adapter
from jevx.capabilities import JevXCapabilityDetector

BERT_ID = "hf-internal-testing/tiny-random-bert"
GPT2_ID = "hf-internal-testing/tiny-random-gpt2"
GPT2_FALLBACK_ID = "sshleifer/tiny-gpt2"


@pytest.fixture(scope="module")
def real_bert() -> PreTrainedModel:
    model = AutoModel.from_pretrained(BERT_ID)
    model.eval()
    return model


@pytest.fixture(scope="module")
def real_bert_tokenizer() -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(BERT_ID)
    if tokenizer.pad_token is None and tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token
    return tokenizer


def _load_gpt2() -> tuple[PreTrainedModel, PreTrainedTokenizerBase, str]:
    try:
        model = AutoModel.from_pretrained(GPT2_ID)
        tokenizer = AutoTokenizer.from_pretrained(GPT2_ID)
        used_id = GPT2_ID
    except Exception:  # documented fallback for hub hiccups/renames
        model = AutoModel.from_pretrained(GPT2_FALLBACK_ID)
        tokenizer = AutoTokenizer.from_pretrained(GPT2_FALLBACK_ID)
        used_id = GPT2_FALLBACK_ID
    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token or tokenizer.unk_token or tokenizer.sep_token
        )
    model.eval()
    return model, tokenizer, used_id


@pytest.fixture(scope="module")
def real_gpt2() -> tuple[PreTrainedModel, PreTrainedTokenizerBase, str]:
    return _load_gpt2()


def test_real_bert_detect_prepare_extract_validate(
    real_bert: PreTrainedModel, real_bert_tokenizer: PreTrainedTokenizerBase
) -> None:
    report = JevXCapabilityDetector.detect(real_bert)
    assert report.model_type == "bert"
    assert report.jevx_conversion == "supported"
    assert report.representation_field == "pooler_output"
    assert report.pooler is True
    assert report.encoder and not report.decoder

    adapter = resolve_adapter(real_bert)
    assert isinstance(adapter, BaseAdapter)
    inputs = adapter.prepare_inputs(
        "hello world", tokenizer=real_bert_tokenizer, model=real_bert
    )
    assert "input_ids" in inputs and "attention_mask" in inputs

    rep = adapter.extract_representation(real_bert, inputs)
    hidden = real_bert.config.hidden_size
    assert rep.pooled is not None and tuple(rep.pooled.shape) == (1, hidden)
    assert rep.field in {"pooler_output", "last_hidden_state"}
    assert rep.sequence_output is not None

    adapter.validate(real_bert, inputs)  # full no_grad dry run, no raise


def test_real_gpt2_detect_prepare_extract_validate(
    real_gpt2: tuple[PreTrainedModel, PreTrainedTokenizerBase, str],
) -> None:
    model, tokenizer, used_id = real_gpt2
    assert used_id in {GPT2_ID, GPT2_FALLBACK_ID}

    report = JevXCapabilityDetector.detect(model)
    assert report.decoder and not report.encoder and not report.encoder_decoder
    assert report.pooler is False
    assert report.representation_field == "last_hidden_state"

    adapter = resolve_adapter(model)
    assert adapter.name in {"decoder", "generic"}

    inputs = adapter.prepare_inputs(
        "hello world", tokenizer=tokenizer, model=model
    )
    rep = adapter.extract_representation(model, inputs)
    n_embd = model.config.n_embd
    assert rep.pooled is not None and tuple(rep.pooled.shape) == (1, n_embd)
    assert rep.field == "last_hidden_state"

    adapter.validate(model, inputs)


def test_real_gpt2_batched_padding_equivalence(
    real_gpt2: tuple[PreTrainedModel, PreTrainedTokenizerBase, str],
) -> None:
    """Left-pad batch equals unpadded row for last_valid (real tokenizer)."""
    model, tokenizer, _ = real_gpt2
    adapter = resolve_adapter(model)
    text = "hello world foo bar"

    tokenizer.padding_side = "right"
    unpadded = adapter.extract_representation(
        model, dict(tokenizer(text, return_tensors="pt"))
    )
    tokenizer.padding_side = "left"
    left_padded = adapter.extract_representation(
        model,
        dict(tokenizer(text, return_tensors="pt", padding="max_length", max_length=12)),
    )
    tokenizer.padding_side = "right"
    right_padded = adapter.extract_representation(
        model,
        dict(tokenizer(text, return_tensors="pt", padding="max_length", max_length=12)),
    )
    assert unpadded.pooled is not None
    assert left_padded.pooled is not None
    assert right_padded.pooled is not None
    assert (left_padded.pooled - unpadded.pooled).abs().max().item() < 1e-4
    assert (right_padded.pooled - unpadded.pooled).abs().max().item() < 1e-4
    assert (left_padded.pooled - right_padded.pooled).abs().max().item() < 1e-5
