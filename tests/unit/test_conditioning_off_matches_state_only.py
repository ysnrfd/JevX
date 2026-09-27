# Copyright (c) 2026 ysnrfd
"""``conditioning_mode="off"`` must BE the state-only reference (WAVE-2 P0-2).

Defect this module pins, measured in ``reports/MULTI_QUESTION.md`` §5.2/§5.3:
the shipped ``"off"`` mode zeroed the OPTION-INTERACTION RESIDUAL but still
concatenated the QUESTION embedding into ``ConcatFusion``'s ``W_cat [s ; q]``,
so half the conditioning stack stayed live in a mode whose name says it is
off. The Wave-2 grid therefore measured a separate ``state_only`` REFERENCE arm
— ``conditioning_mode="off"`` plus a forward hook on ``question_encoder``
returning exact zeros — and it beat the shipped ``off`` by ~1 pp pooled over
11 questions (0.7294 vs 0.7199).

The reference implementation is reproduced verbatim below (:class:`StateOnlyReference`)
so the equality is against the measured arm, not against a paraphrase of it.
With the fix, ``off`` IS that arm, bitwise.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List

import torch
import torch.nn as nn

from jevx import JevXConfig, JevXForDecision
from jevx.representations import build_fusion

HIDDEN = 32  # matches the tiny_bert fixture
STATES = ["hello world", "foo bar", "baz cat", "hello cat"]
QUESTIONS = ["which one"]
#: ONE schema per call (m = 1), K = 3 — JevXForDecision takes 1, m or B*m
#: schemas, never an arbitrary count.
OPTIONS = [["hello", "world", "foo"]]
OTHER_SCHEMA = [["bar", "baz", "cat"]]
#: A second question with a different K, for the variable-K shape contract.
OPTIONS_TWO_QUESTIONS = [["hello", "world", "foo"], ["a", "b", "c", "the"]]


def _wrapper(backbone: Any, tokenizer: Any, **overrides: Any) -> JevXForDecision:
    settings: Dict[str, Any] = {
        "decision_hidden_size": HIDDEN,
        "question_hidden_size": HIDDEN,
        "schema_hidden_size": HIDDEN,
    }
    settings.update(overrides)
    return JevXForDecision(
        JevXConfig(**settings),
        backbone=copy.deepcopy(backbone),
        tokenizer=tokenizer,
    ).eval()


class StateOnlyReference:
    """The WAVE-2 ``state_only`` arm, reproduced bit-for-bit.

    ``benchmarks/run_multi_question.py::_JevXArm.zero_question`` installed
    exactly this: ``conditioning_mode="off"`` (``mode = "off" if arm ==
    "state_only" else arm``) plus a forward hook on ``model.question_encoder``
    replacing its output with ``torch.zeros_like(o)``, and passed
    ``condition_options=False``. The hook is installed per call and removed in
    a ``finally``, so nothing survives it.
    """

    def __init__(self, model: JevXForDecision) -> None:
        self.model = model

    def __call__(self, **call: Any):
        hook = self.model.question_encoder.register_forward_hook(
            lambda _m, _i, output: torch.zeros_like(output)
        )
        try:
            with torch.no_grad():
                return self.model(condition_options=False, **call)
        finally:
            hook.remove()


def _logits(model: JevXForDecision, **call: Any) -> List[torch.Tensor]:
    with torch.no_grad():
        out = model(**call)
    return [d.logits.reshape(-1).clone() for d in out.decisions]


def _same(a: List[torch.Tensor], b: List[torch.Tensor]) -> bool:
    return len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))


def _fill_state_pathway(model: JevXForDecision, seed: int = 1234) -> None:
    """Give the ZERO-INIT state pathway non-zero parameters.

    The shipped ``LinearProbePathway`` starts at exactly zero, so a fresh head
    emits the SAME logit for every class whatever the fusion computes — which
    would make "the two arms agree" and "the state path trains" both vacuous.
    Same remedy as ``tests/unit/test_state_only_path.py``.
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, head in model.heads.items():
            pathway = head.state_pathway
            if not isinstance(pathway, nn.Module):  # selector string / None
                continue
            pathway.weight.copy_(
                torch.randn(
                    pathway.width, head.hidden_size, generator=generator
                ) * 0.1
            )
            pathway.bias.copy_(
                torch.randn(pathway.width, generator=generator) * 0.1
            )


def _max_abs_diff(a: List[torch.Tensor], b: List[torch.Tensor]) -> float:
    return max(float((x - y).abs().max()) for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# 1. The equality itself
# ---------------------------------------------------------------------------


def test_off_is_bitwise_the_state_only_reference(
    tiny_bert: Any, tiny_tokenizer: Any
) -> None:
    """``off`` and the measured ``state_only`` arm agree to the last bit."""
    torch.manual_seed(0)
    off = _wrapper(tiny_bert, tiny_tokenizer, conditioning_mode="off")
    _fill_state_pathway(off)
    # Same weights, same seed -> the two models are bitwise identical builds.
    torch.manual_seed(0)
    reference_model = _wrapper(tiny_bert, tiny_tokenizer, conditioning_mode="off")
    _fill_state_pathway(reference_model)

    got = _logits(off, state=STATES, questions=QUESTIONS, options=OPTIONS)
    want = StateOnlyReference(reference_model)(
        state=STATES, questions=QUESTIONS, options=OPTIONS
    )
    want = [d.logits.reshape(-1).clone() for d in want.decisions]

    assert _same(got, want), (
        'conditioning_mode="off" is not the state-only reference: max abs logit '
        f"diff {_max_abs_diff(got, want):.6e}"
    )
    # Guard against a degenerate "everything is zero" agreement.
    assert any(float(row.max() - row.min()) > 0.0 for row in got), (
        "the two arms agree only because the logits are all equal"
    )


def test_off_is_invariant_to_the_question_text(
    tiny_bert: Any, tiny_tokenizer: Any
) -> None:
    """The property the Wave-2 §6.2 note actually claims for ``state_only``.

    "the fused context stops depending on ``q`` at all": two different
    question texts, one state-only call, identical decisions.
    """
    torch.manual_seed(1)
    model = _wrapper(tiny_bert, tiny_tokenizer, conditioning_mode="off")
    a = _logits(model, state=STATES, questions=["which one"], options=OPTIONS)
    b = _logits(
        model, state=STATES, questions=["pick the best thing"], options=OPTIONS
    )
    assert _same(a, b), (
        "the question text still moves the off-mode decision, so the question "
        "path is not neutralized"
    )


def test_off_is_still_invariant_to_the_schema(
    tiny_bert: Any, tiny_tokenizer: Any
) -> None:
    """The pre-existing schema-independence property is preserved."""
    torch.manual_seed(2)
    model = _wrapper(tiny_bert, tiny_tokenizer, conditioning_mode="off")
    a = _logits(model, state=STATES, questions=QUESTIONS, options=OPTIONS)
    b = _logits(model, state=STATES, questions=QUESTIONS, options=OTHER_SCHEMA)
    assert _same(a, b)


# ---------------------------------------------------------------------------
# 2. ``full`` / ``constant`` are unaffected
# ---------------------------------------------------------------------------


def test_full_is_unaffected_by_the_off_change(
    tiny_bert: Any, tiny_tokenizer: Any
) -> None:
    """``full`` must still READ the question and the schema.

    The control for the test above: a mode that neutralizes nothing is the only
    thing that makes the previous test meaningful, and it also pins the
    documented user path (``conditioning_mode="full"``) for a user who wants
    question conditioning.
    """
    torch.manual_seed(3)
    model = _wrapper(tiny_bert, tiny_tokenizer, conditioning_mode="full")
    a = _logits(model, state=STATES, questions=["which one"], options=OPTIONS)
    b = _logits(
        model, state=STATES, questions=["pick the best thing"], options=OPTIONS
    )
    c = _logits(model, state=STATES, questions=QUESTIONS, options=OTHER_SCHEMA)
    assert not _same(a, b), (
        'conditioning_mode="full" is now question-blind: the question text '
        "does not change the decision"
    )
    assert not _same(a, c), (
        'conditioning_mode="full" is now schema-blind: the option texts do not '
        "change the decision"
    )
    # And it differs from the off mode on the same weights.
    torch.manual_seed(3)
    off = _wrapper(tiny_bert, tiny_tokenizer, conditioning_mode="off")
    off_logits = _logits(off, state=STATES, questions=QUESTIONS, options=OPTIONS)
    assert not _same(a, off_logits)


def test_constant_is_unaffected_by_the_off_change(
    tiny_bert: Any, tiny_tokenizer: Any
) -> None:
    """``constant`` is the memoized conditioning path, not the off path."""
    torch.manual_seed(4)
    model = _wrapper(tiny_bert, tiny_tokenizer, conditioning_mode="constant")
    a = _logits(model, state=STATES, questions=["which one"], options=OPTIONS)
    b = _logits(
        model, state=STATES, questions=["pick the best thing"], options=OPTIONS
    )
    assert not _same(a, b), 'mode="constant" no longer conditions on the question'
    torch.manual_seed(4)
    off = _wrapper(tiny_bert, tiny_tokenizer, conditioning_mode="off")
    assert not _same(a, _logits(off, state=STATES, questions=QUESTIONS, options=OPTIONS))


# ---------------------------------------------------------------------------
# 3. The shape / gradient / contract consequences
# ---------------------------------------------------------------------------


def test_the_fused_state_shape_is_unchanged_by_off(
    tiny_bert: Any, tiny_tokenizer: Any
) -> None:
    """No dimension change: the fused state is still [n, D] under ``off``."""
    torch.manual_seed(5)
    model = _wrapper(tiny_bert, tiny_tokenizer, conditioning_mode="off")
    with torch.no_grad():
        out = model(state=STATES, questions=QUESTIONS, options=OPTIONS)
    assert out.state_representation.shape == (len(STATES), HIDDEN)
    assert out.question_representations.shape == (len(STATES), 1, HIDDEN)
    for decision in out.decisions:
        assert decision.logits.shape == (1, 3)  # K = 3
        assert torch.allclose(
            decision.probabilities.sum(dim=-1), torch.ones(1), atol=1e-6
        )
        assert 0.0 <= float(decision.confidence[0]) <= 1.0
    assert out.schema_representations.shape == (len(STATES), 3, HIDDEN)


def test_the_variable_k_shape_contract_survives_off(
    tiny_bert: Any, tiny_tokenizer: Any
) -> None:
    """K columns, padded columns and the mask contract are untouched.

    The question zeroing happens upstream of fusion, so it must not disturb
    the [B*m, K_max, D] schema block or the padded-column masking.
    """
    torch.manual_seed(5)
    model = _wrapper(tiny_bert, tiny_tokenizer, conditioning_mode="off")
    with torch.no_grad():
        out = model(
            state=STATES,
            questions=["which one", "which one"],
            options=OPTIONS_TWO_QUESTIONS,
        )
    assert len(out.decisions) == 2 * len(STATES)
    for pair in range(len(STATES)):
        short, long = out.decisions[pair * 2], out.decisions[pair * 2 + 1]
        assert short.logits.shape == (1, 4)  # K_max = 4
        assert torch.all(short.probabilities[0, 3:] == 0.0), "padded stays exactly 0"
        assert torch.allclose(
            short.probabilities[0, :3].sum(), torch.tensor(1.0), atol=1e-6
        )
        assert 0 <= int(short.predicted_index[0]) < 3
        assert long.logits.shape == (1, 4)
        assert 0 <= int(long.predicted_index[0]) < 4


def test_off_reaches_no_gradient_in_the_question_encoder(
    tiny_bert: Any, tiny_tokenizer: Any
) -> None:
    """80-89 % of the trainable parameters get NOTHING under ``off``.

    Measured in ``reports/MULTI_QUESTION.md`` §5.4 (exactly 0 at step 1 for
    ``question_encoder``, ``schema_encoder`` and the fusion in the ``off``
    arm). The fix extends that to the question path.
    """
    torch.manual_seed(6)
    model = _wrapper(tiny_bert, tiny_tokenizer, conditioning_mode="off")
    _fill_state_pathway(model)
    model.train()
    out = model(state=STATES, questions=QUESTIONS, options=OPTIONS)
    logits = torch.stack([d.logits.reshape(-1) for d in out.decisions])
    torch.nn.functional.cross_entropy(
        logits, torch.zeros(len(STATES), dtype=torch.long)
    ).backward()
    for name, param in model.question_encoder.named_parameters():
        assert param.grad is None or float(param.grad.abs().max()) == 0.0, (
            f"question_encoder.{name} received gradient under mode='off'"
        )
    for name, param in model.schema_encoder.named_parameters():
        assert param.grad is None or float(param.grad.abs().max()) == 0.0, (
            f"schema_encoder.{name} received gradient under mode='off'"
        )
    # The state path still trains.
    assert model.state_proj.proj.weight.grad is not None
    assert float(model.state_proj.proj.weight.grad.abs().max()) > 0.0


def test_full_still_trains_the_question_encoder(
    tiny_bert: Any, tiny_tokenizer: Any
) -> None:
    """The revert path works end to end: a user CAN condition on the question."""
    torch.manual_seed(7)
    model = _wrapper(tiny_bert, tiny_tokenizer, conditioning_mode="full")
    model.train()
    out = model(state=STATES, questions=QUESTIONS, options=OPTIONS)
    logits = torch.stack([d.logits.reshape(-1) for d in out.decisions])
    torch.nn.functional.cross_entropy(
        logits, torch.zeros(len(STATES), dtype=torch.long)
    ).backward()
    assert model.question_encoder.hash_embedding.weight.grad is not None
    assert float(model.question_encoder.hash_embedding.weight.grad.abs().max()) > 0.0
    assert model.schema_encoder.text_encoder.hash_embedding.weight.grad is not None


def test_off_equals_the_per_call_false_override(
    tiny_bert: Any, tiny_tokenizer: Any
) -> None:
    """The documented equivalence ``off == forward(condition_options=False)``."""
    torch.manual_seed(8)
    off = _wrapper(tiny_bert, tiny_tokenizer, conditioning_mode="off")
    torch.manual_seed(8)
    full = _wrapper(tiny_bert, tiny_tokenizer, conditioning_mode="full")
    a = _logits(off, state=STATES, questions=QUESTIONS, options=OPTIONS)
    b = _logits(
        full,
        state=STATES,
        questions=QUESTIONS,
        options=OPTIONS,
        condition_options=False,
    )
    assert _same(a, b), (
        'mode="off" is no longer bitwise forward(condition_options=False) — the '
        "two now take different question paths"
    )


def test_off_neutralizes_a_non_finite_question_rather_than_raising(
    tiny_bert: Any, tiny_tokenizer: Any
) -> None:
    """The scope of the NaN-question failure mode, pinned explicitly.

    ``check_finite`` on ``question_representations H_Q`` only fires on a path
    that READS the question. ``off`` substitutes exact zeros before the
    projection, so a NaN question cannot reach a logit at all — the decision is
    the same finite one it would be for any other question text. This is a
    consequence of "off does not read the question", and it is recorded here
    rather than left implicit next to
    ``tests/unit/test_failure_modes.py::test_nan_preencoded_questions_raise_check_finite_with_tensor_name``
    (which pins the ``full``-mode guard).
    """
    torch.manual_seed(9)
    off = _wrapper(tiny_bert, tiny_tokenizer, conditioning_mode="off")
    clean = _logits(off, state=STATES, questions=QUESTIONS, options=OPTIONS)
    with torch.no_grad():
        out = off(
            state=STATES,
            questions=torch.full((1, HIDDEN), float("nan")),
            options=OPTIONS,
        )
    assert torch.isfinite(out.decisions[0].logits).all()
    for got, want in zip([d.logits.reshape(-1) for d in out.decisions], clean):
        assert torch.equal(got, want), (
            "a NaN question changed the off-mode decision; the question path "
            "is not fully neutralized"
        )


def test_full_still_rejects_a_non_finite_question(
    tiny_bert: Any, tiny_tokenizer: Any
) -> None:
    """The control: the reading path keeps the named ValueError."""
    torch.manual_seed(9)
    model = _wrapper(tiny_bert, tiny_tokenizer, conditioning_mode="full")
    try:
        with torch.no_grad():
            model(
                state=STATES,
                questions=torch.full((1, HIDDEN), float("nan")),
                options=OPTIONS,
            )
    except ValueError as exc:
        assert "question_representations H_Q" in str(exc)
    else:
        raise AssertionError(
            'conditioning_mode="full" accepted a NaN question tensor'
        )


# ---------------------------------------------------------------------------
# 4. Every fusion kind that reads the question must see the zero
# ---------------------------------------------------------------------------


def test_the_question_is_zero_before_every_fusion_kind() -> None:
    """The zeroing happens in ``forward``, upstream of fusion, so it covers
    every registered kind — not just ``ConcatFusion``, which is only the
    default. ``IdentityFusion`` reads nothing, which is the same contract.
    """
    for kind in ("concat", "additive", "gated", "bilinear", "dot", "mlp", "identity"):
        state = torch.randn(3, 8)
        question = torch.randn(3, 8)
        fusion = build_fusion(kind, 8).eval()
        with torch.no_grad():
            on = fusion.context(state, question)
            off = fusion.context(state, torch.zeros_like(question))
        assert torch.isfinite(off).all(), kind
        if kind != "identity":
            assert not torch.allclose(on, off), (
                f"fusion kind {kind!r} ignores the question rows, so the off "
                "path is not neutralized for it"
            )
        else:
            assert torch.allclose(on, off), "identity fusion must ignore q by design"
