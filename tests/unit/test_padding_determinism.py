# Copyright (c) 2026 ysnrfd
"""Padding invariance (phase 73) and eval determinism (phase 74).

Phase 73 — the same logical sequence encoded left-padded, right-padded and
unpadded must produce matching pooled representations:

* decoder adapter with tiny-gpt2 (real checkpoint, sshleifer/tiny-gpt2 with
  hf-internal-testing/tiny-random-gpt2 fallback): batch padded-left vs the
  single unpadded row — max_abs_error is REPORTED (print) and asserted
  < 1e-5; left vs right is asserted the same way;
* encoder adapter with bert (real checkpoint hf-internal-testing/tiny-random-bert):
  right-pad vs unpadded, same row — max_abs_error reported and asserted.

Phase 74 — determinism: the same input evaluated twice in eval() mode must be
BITWISE identical (torch.equal on fp32 probabilities on CPU). Cross-process
bitwise identity is deliberately NOT required or asserted: separate OS
processes may schedule OpenMP/oneDNN reductions on different thread counts or
choose different kernels based on CPU feature probing and environment, so
only within-process eval determinism is part of the contract.
"""

from __future__ import annotations

import copy

import pytest
import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from jevx import JevXConfig, JevXForDecision
from jevx.adapters import resolve_adapter

BERT_ID = "hf-internal-testing/tiny-random-bert"
GPT2_ID = "hf-internal-testing/tiny-random-gpt2"
GPT2_FALLBACK_ID = "sshleifer/tiny-gpt2"

TEXT = "hello world foo bar"
PAD_TO = 16


@pytest.fixture(scope="module")
def real_bert() -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    model = AutoModel.from_pretrained(BERT_ID)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(BERT_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token or "[PAD]"
    return model, tokenizer


@pytest.fixture(scope="module")
def real_gpt2() -> tuple[PreTrainedModel, PreTrainedTokenizerBase, str]:
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


def _max_abs_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach() - b.detach()).abs().max())


# ---------------------------------------------------------------------------
# Phase 73 — decoder (tiny-gpt2): left-pad vs unpadded vs right-pad
# ---------------------------------------------------------------------------


def test_decoder_left_pad_vs_unpadded_max_abs_error(real_gpt2) -> None:
    model, tokenizer, used_id = real_gpt2
    assert used_id in {GPT2_ID, GPT2_FALLBACK_ID}
    adapter = resolve_adapter(model)

    tokenizer.padding_side = "right"
    unpadded = adapter.extract_representation(
        model, dict(tokenizer(TEXT, return_tensors="pt"))
    )
    tokenizer.padding_side = "left"
    left_padded = adapter.extract_representation(
        model,
        dict(tokenizer(TEXT, return_tensors="pt", padding="max_length",
                       max_length=PAD_TO)),
    )
    tokenizer.padding_side = "right"
    right_padded = adapter.extract_representation(
        model,
        dict(tokenizer(TEXT, return_tensors="pt", padding="max_length",
                       max_length=PAD_TO)),
    )
    assert unpadded.pooled is not None
    assert left_padded.pooled is not None
    assert right_padded.pooled is not None

    err_left = _max_abs_error(left_padded.pooled, unpadded.pooled)
    err_right = _max_abs_error(right_padded.pooled, unpadded.pooled)
    err_lr = _max_abs_error(left_padded.pooled, right_padded.pooled)
    print(
        f"[decoder {used_id}] left-vs-unpadded max_abs_error={err_left:.3e} "
        f"right-vs-unpadded={err_right:.3e} left-vs-right={err_lr:.3e}"
    )
    assert err_left < 1e-5, f"left-pad vs unpadded drifted: {err_left:.3e}"
    assert err_right < 1e-5, f"right-pad vs unpadded drifted: {err_right:.3e}"
    assert err_lr < 1e-5, f"left vs right drifted: {err_lr:.3e}"


def test_decoder_padding_equivalence_through_full_choice(real_gpt2) -> None:
    """End-to-end wrapper: same probs for padded-left batch vs single state."""
    model, tokenizer, _ = real_gpt2
    width = int(model.config.n_embd)
    wrapper = JevXForDecision(
        JevXConfig(decision_hidden_size=width, question_hidden_size=width,
                   schema_hidden_size=width),
        backbone=copy.deepcopy(model),
        tokenizer=tokenizer,
    )
    wrapper.eval()

    longer = TEXT + " foo bar baz hello"
    tokenizer.padding_side = "right"
    with torch.no_grad():
        single_short = wrapper.choice(TEXT, ["q"], options=[["a", "b"]])
        single_long = wrapper.choice(longer, ["q"], options=[["a", "b"]])
    tokenizer.padding_side = "left"  # shorter row gets LEFT-padded in the batch
    try:
        with torch.no_grad():
            padded = wrapper.choice([TEXT, longer], ["q"], options=[["a", "b"]])
    finally:
        tokenizer.padding_side = "right"

    # row 0 is the short, left-padded member; row 1 is the longest (unpadded)
    for index, reference in ((0, single_short), (1, single_long)):
        row_probs = padded.decisions[index].probabilities
        error = _max_abs_error(row_probs, reference.decisions[0].probabilities)
        print(f"[decoder full .choice] batch row {index} vs single max_abs_error={error:.3e}")
        assert error < 1e-5, f"row {index} drifted from single: {error:.3e}"


# ---------------------------------------------------------------------------
# Phase 73 — encoder (bert): right-pad vs unpadded, same row
# ---------------------------------------------------------------------------


def test_encoder_right_pad_vs_unpadded_max_abs_error(real_bert) -> None:
    model, tokenizer = real_bert
    adapter = resolve_adapter(model)

    tokenizer.padding_side = "right"
    unpadded = adapter.extract_representation(
        model, dict(tokenizer(TEXT, return_tensors="pt"))
    )
    right_padded = adapter.extract_representation(
        model,
        dict(tokenizer(TEXT, return_tensors="pt", padding="max_length",
                       max_length=PAD_TO)),
    )
    assert unpadded.pooled is not None and right_padded.pooled is not None
    error = _max_abs_error(right_padded.pooled, unpadded.pooled)
    print(f"[encoder {BERT_ID}] right-vs-unpadded max_abs_error={error:.3e}")
    assert error < 1e-5, f"right-pad vs unpadded drifted: {error:.3e}"


def test_encoder_padding_invariance_with_local_pooling_strategies(tiny_bert,
                                                                  tiny_tokenizer) -> None:
    """Mask-aware strategies are pad-side invariant on the encoder too.

    Uses the in-memory fixture so the claim covers every pooling route
    (native pooler is exercised by the real-checkpoint test above).
    """
    from jevx.representations.pooling import pool

    model = copy.deepcopy(tiny_bert)
    adapter = resolve_adapter(model)
    ids = tiny_tokenizer(TEXT, return_tensors="pt")["input_ids"]
    t = ids.shape[1]
    hidden = int(model.config.hidden_size)

    unpadded = adapter.extract_representation(model, {"input_ids": ids})
    seq = unpadded.sequence_output
    assert seq is not None and seq.shape == (1, t, hidden)

    right_mask = torch.ones(1, t, dtype=torch.bool)
    right_mask[:, t - 2 :] = False  # two right pads
    left_mask = torch.ones(1, t, dtype=torch.bool)
    left_mask[:, :2] = False  # two left pads
    valid = right_mask[:, : t - 2]  # the logical sequence alone

    for strategy in ("masked_mean", "cls", "last_valid", "max"):
        reference = pool(seq[:, : t - 2, :], valid, strategy)
        right = pool(seq, right_mask, strategy)
        error_right = _max_abs_error(right, reference)
        # left-pad rows: shift the SAME logical token block to the right side
        shifted = torch.zeros(1, t, hidden)
        shifted[:, 2:, :] = seq[:, : t - 2, :]
        left = pool(shifted, left_mask, strategy)
        error_left = _max_abs_error(left, reference)
        print(
            f"[pool {strategy}] right-vs-unpadded={error_right:.3e} "
            f"left-vs-unpadded={error_left:.3e}"
        )
        assert error_right < 1e-5, f"{strategy} right-pad drifted: {error_right:.3e}"
        assert error_left < 1e-5, f"{strategy} left-pad drifted: {error_left:.3e}"


# ---------------------------------------------------------------------------
# Phase 74 — determinism: same input twice in eval -> bitwise identical
# ---------------------------------------------------------------------------


def _build_wrapper(tiny_bert, tiny_tokenizer) -> JevXForDecision:
    wrapper = JevXForDecision(
        JevXConfig(decision_hidden_size=32, question_hidden_size=32,
                   schema_hidden_size=32),
        backbone=copy.deepcopy(tiny_bert),
        tokenizer=tiny_tokenizer,
    )
    wrapper.eval()
    return wrapper


def test_same_input_twice_in_eval_is_bitwise_identical(tiny_bert,
                                                        tiny_tokenizer) -> None:
    wrapper = _build_wrapper(tiny_bert, tiny_tokenizer)
    questions = ["q1", "q2", "q3"]
    options = [["a", "b"], ["c", "d", "e"], ["a", "b", "c"]]
    with torch.no_grad():
        first = wrapper.choice(TEXT, questions, options=options)
        second = wrapper.choice(TEXT, questions, options=options)

    for i, (a, b) in enumerate(zip(first.decisions, second.decisions)):
        assert torch.equal(a.probabilities, b.probabilities), (
            f"decision {i}: probabilities differ bit-for-bit between identical "
            "eval runs"
        )
        assert torch.equal(a.logits, b.logits), f"decision {i}: logits drifted"
        assert torch.equal(a.confidence, b.confidence), f"decision {i}: confidence drifted"
        assert a.predicted_value == b.predicted_value
    assert torch.equal(first.state_representation, second.state_representation)

    # Cross-process bitwise identity is intentionally NOT asserted: separate
    # processes may pick different oneDNN/OMP thread partitions or kernels
    # (documented non-goal — within-process eval determinism is the contract).

    # eval-mode stability across intervening unrelated calls too
    with torch.no_grad():
        wrapper.choice("some other state entirely", ["other"], options=[["x", "y"]])
        third = wrapper.choice(TEXT, questions, options=options)
    for i, (a, b) in enumerate(zip(first.decisions, third.decisions)):
        assert torch.equal(a.probabilities, b.probabilities), (
            f"decision {i}: an intervening call changed identical-run results"
        )


def test_adaptive_pooling_strategies_also_deterministic(tiny_bert,
                                                         tiny_tokenizer) -> None:
    """attention_pool / learned_pool rebuild stateless modules per call —
    they must still be bitwise reproducible inside one process (fresh modules
    are seeded by torch's global RNG, untouched in eval runs)."""
    wrapper = JevXForDecision(
        JevXConfig(decision_hidden_size=32, question_hidden_size=32,
                   schema_hidden_size=32, pooling_method="attention_pool"),
        backbone=copy.deepcopy(tiny_bert),
        tokenizer=tiny_tokenizer,
    )
    wrapper.eval()
    torch.manual_seed(1234)  # fixes module re-init draws for stateless pools
    with torch.no_grad():
        first = wrapper.choice(TEXT, ["q"], options=[["a", "b"]])
        second = wrapper.choice(TEXT, ["q"], options=[["a", "b"]])
    assert torch.equal(first.decisions[0].probabilities,
                       second.decisions[0].probabilities)
