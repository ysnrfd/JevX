# Copyright (c) 2026 ysnrfd
"""Q-006 cross-attention fusion (spec Def 5.e) tests.

The fix wave registered the ``"cross_attention"`` fusion kind
(``CrossAttentionFusion``) and plumbed the unpooled state sequence
``H_S^seq`` through ``Fusion.forward(state, ..., *, state_sequence=,
state_mask=)`` and ``JevXForDecision.forward`` (only when
``fusion_type == "cross_attention"``). This file pins the contract the
guarded smoke in ``test_public_surface.py`` does not cover:

1. output shapes + masked option rows exactly 0;
2. padding invariance — perturbing padded sequence positions cannot move
   the fused output (``< 1e-7``);
3. gradient flow — valid positions receive finite gradients, padded
   positions receive exactly zero;
4. every OTHER kind stays byte-identical whether the sequence kwargs are
   omitted or passed as ``None``, and rejects a real sequence with the
   documented error;
5. model-level plumbing for both a sequence-consuming and a plain kind.
"""

from __future__ import annotations

import copy

import pytest
import torch

from jevx import JevXConfig, JevXForDecision
from jevx.representations import FUSION_REGISTRY, build_fusion

HIDDEN = 32
STATE = "hello world"
QUESTIONS = ["question option?"]
OPTIONS = [["a", "b", "c"]]

#: Every registered kind except the sequence consumer (includes the "add"
#: alias so the alias is pinned too).
_PLAIN_KINDS = sorted(k for k in FUSION_REGISTRY if k != "cross_attention")


def _build_wrapper(backbone, tokenizer=None, **config_overrides) -> JevXForDecision:
    """Wrap ``backbone`` (deep-copied) with a tiny decision config, eval mode."""
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


def _cross_attention_inputs(batch: int = 2):
    """Fixed inputs: pooled/question/options + a masked 5-token sequence."""
    torch.manual_seed(7)
    state = torch.randn(batch, HIDDEN)
    question = torch.randn(batch, HIDDEN)
    options = torch.randn(batch, 3, HIDDEN)
    option_mask = torch.tensor([[True, True, False], [True, True, True]][:batch])
    state_sequence = torch.randn(batch, 5, HIDDEN)
    state_mask = torch.tensor(
        [[True, True, True, False, False]] * batch, dtype=torch.bool
    )
    return state, question, options, option_mask, state_sequence, state_mask


def test_cross_attention_shapes_and_masked_options_are_zero() -> None:
    """Def 5.e output contract: fused [B, D], option_features [B, K, D],
    invalid option rows exactly 0, everything finite."""
    fusion = build_fusion("cross_attention", HIDDEN)
    state, question, options, option_mask, seq, mask = _cross_attention_inputs()
    with torch.no_grad():
        out = fusion(state, question, options=options, option_mask=option_mask,
                     state_sequence=seq, state_mask=mask)
    assert tuple(out.fused.shape) == (2, HIDDEN)
    assert tuple(out.option_features.shape) == (2, 3, HIDDEN)
    assert torch.isfinite(out.fused).all()
    assert torch.isfinite(out.option_features).all()
    assert torch.all(out.option_features[0, 2] == 0.0), (
        "masked option rows must be exactly 0 (masked_fill contract)"
    )


def test_padding_invariance_under_padded_perturbation() -> None:
    """Perturbing padded sequence positions must move the fused output by
    less than 1e-7 (exact-zero attention weights, D1.2)."""
    fusion = build_fusion("cross_attention", HIDDEN)
    state, question, options, option_mask, seq, mask = _cross_attention_inputs()
    with torch.no_grad():
        base = fusion(state, question, options=options, option_mask=option_mask,
                      state_sequence=seq, state_mask=mask)
        perturbed = seq.clone()
        perturbed[~mask] = 1e3
        again = fusion(state, question, options=options, option_mask=option_mask,
                       state_sequence=perturbed, state_mask=mask)
    assert torch.allclose(base.fused, again.fused, atol=1e-7, rtol=0.0), (
        "padded sequence positions must not contribute to the fused output"
    )
    assert torch.allclose(
        base.option_features, again.option_features, atol=1e-7, rtol=0.0
    ), "padded sequence positions must not contribute to the option features"


def test_gradient_flow_reaches_valid_positions_only() -> None:
    """Backprop from the fused output reaches question/options/sequence at
    VALID positions with finite grads; padded positions get exact zeros."""
    fusion = build_fusion("cross_attention", HIDDEN)
    state, question, options, option_mask, seq, mask = _cross_attention_inputs()
    question = question.clone().requires_grad_(True)
    options = options.clone().requires_grad_(True)
    seq = seq.clone().requires_grad_(True)
    out = fusion(state, question, options=options, option_mask=option_mask,
                 state_sequence=seq, state_mask=mask)
    # fused alone never reads `options`; sum both outputs so every input
    # participates in the graph.
    loss = out.fused.sum() + out.option_features.sum()
    loss.backward()

    for name, tensor in (("question", question), ("options", options),
                         ("state_sequence", seq)):
        assert tensor.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(tensor.grad).all(), f"{name} gradient not finite"
    assert question.grad.abs().sum() > 0, "question gradient must be non-trivial"
    # padded sequence positions carry exactly zero gradient
    assert bool((seq.grad[~mask] == 0).all()), (
        "padded positions must receive exactly zero gradient"
    )
    # parameters of the attention stack were trained by the backward pass
    param_grads = [p.grad for p in fusion.parameters() if p.grad is not None]
    assert param_grads, "fusion parameters received no gradient"
    assert all(torch.isfinite(g).all() for g in param_grads)


@pytest.mark.parametrize("kind", _PLAIN_KINDS)
def test_other_kinds_byte_identical_and_reject_sequence(kind: str) -> None:
    """State-sequence plumbing must be inert for every non-cross kind:
    explicit ``None`` kwargs are bit-identical to omitting them, and a real
    sequence raises the documented ValueError naming the kind."""
    fusion = build_fusion(kind, HIDDEN)
    state, question, options, option_mask, seq, _mask = _cross_attention_inputs()
    with torch.no_grad():
        plain = fusion(state, question, options, option_mask)
        explicit_none = fusion(state, question, options, option_mask,
                               state_sequence=None, state_mask=None)
    assert torch.equal(plain.fused, explicit_none.fused), (
        f"kind {kind!r}: passing state_sequence=None changed the output"
    )
    assert torch.equal(plain.option_features, explicit_none.option_features)
    with pytest.raises(ValueError, match=r"cross_attention|state_sequence"):
        fusion(state, question, options, option_mask,
               state_sequence=seq, state_mask=torch.ones(2, 5, dtype=torch.bool))


def test_model_plumbing_sequence_kind_and_plain_kind(
    tiny_bert, tiny_tokenizer
) -> None:
    """End to end: fusion_type='cross_attention' consumes the backbone's
    [B, T, D] sequence through forward; the default 'concat' kind keeps its
    plumbing inert (sequence never required, decisions finite + simplex)."""
    for fusion_type, needs_sequence in (("cross_attention", True), ("concat", False)):
        wrapper = _build_wrapper(
            tiny_bert, tokenizer=tiny_tokenizer, fusion_type=fusion_type
        )
        assert wrapper._fusion_needs_sequence is needs_sequence
        with torch.no_grad():
            out = wrapper(state=STATE, questions=QUESTIONS, options=OPTIONS)
        decision = out.decisions[0]
        assert torch.isfinite(decision.logits).all()
        assert float(decision.probabilities.sum()) == pytest.approx(1.0, abs=1e-5)
        assert decision.predicted_index is not None


def test_model_padding_invariance_end_to_end(tiny_bert, tiny_tokenizer) -> None:
    """Cross-attention + real batch padding: perturbing the padded token ids
    of a variable-length batch leaves every decision unchanged (< 1e-6) —
    the state_mask plumbing keeps padded positions inert all the way out."""
    wrapper = _build_wrapper(
        tiny_bert, tokenizer=tiny_tokenizer, fusion_type="cross_attention"
    )
    states = [STATE, "hello world extra words padding sample"]  # B=2, T padded
    questions = QUESTIONS
    options = OPTIONS
    with torch.no_grad():
        base = wrapper(state=states, questions=questions, options=options)
        # doctored batch: same lengths/mask, different padded token ids
        batch = wrapper.tokenizer(states, padding=True, return_tensors="pt")
        input_ids = batch["input_ids"].clone()
        attention_mask = batch["attention_mask"]
        input_ids[attention_mask == 0] = wrapper.tokenizer.pad_token_id or 0
        # flip every padded id to a different in-vocab token
        alt = (input_ids[attention_mask == 0] + 1) % int(
            wrapper.tokenizer.vocab_size
        )
        input_ids[attention_mask == 0] = alt
        doctored = wrapper(
            state={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
            questions=questions,
            options=options,
        )
    for a, b in zip(base.decisions, doctored.decisions):
        assert torch.allclose(a.probabilities, b.probabilities, atol=1e-6), (
            "perturbed padded token ids changed the decisions — the state "
            "sequence mask is not keeping padded positions inert"
        )
