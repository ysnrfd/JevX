# Copyright (c) 2026 ysnrfd
"""The ``conditioning_mode="constant"`` memo must be WEIGHT-keyed, not only
content-keyed (WAVE-2 follow-up, P0-1).

Defect this module pins, reported in ``reports/MULTI_QUESTION.md`` §2.2 and
§7.7 and reproduced here: the memo's key is the question/option TEXT, and the
only gate on consulting it was ``eval()`` + ``torch.is_grad_enabled() == False``.
Neither of those notices that the ENCODER PARAMETERS moved since the cached
``H_Q``/``H_C`` were computed, so the reachable ordering

    eval  ->  one AdamW step  ->  eval        (no clear_condition_cache())

served a stale ``H_Q``/``H_C``. Measured in the Wave-2 probe: max abs logit
diff **4.1556e+04** against a fresh-cache evaluation of the same model.

The contract, as implemented in ``modeling_jevx.py``:

* the memo is invalidated whenever ANY parameter/buffer TENSOR of
  ``question_encoder`` or ``schema_encoder`` has been mutated in place since
  the memo was filled. The check is a tuple of ``torch.Tensor._version``
  autograd version counters — one integer read per TENSOR, never a walk over
  scalars;
* a memo filled from a model whose weights have NOT changed is still served
  bitwise-identically (the Wave-2 timing/identity evidence depends on this);
* the check is cheap and is measured below, in the same units the report uses.
"""

from __future__ import annotations

import copy
import time
from typing import Any, Dict

import torch
import torch.nn as nn

from jevx import JevXConfig, JevXForDecision

HIDDEN = 32  # matches the tiny_bert fixture
STATE = "hello world"
CALL: Dict[str, Any] = dict(
    state=STATE, questions=["which one"], options=[["a", "b", "c"]]
)


def _wrapper(backbone: Any, tokenizer: Any, **overrides: Any) -> JevXForDecision:
    settings: Dict[str, Any] = {
        "decision_hidden_size": HIDDEN,
        "question_hidden_size": HIDDEN,
        "schema_hidden_size": HIDDEN,
        "conditioning_mode": "constant",
    }
    settings.update(overrides)
    return JevXForDecision(
        JevXConfig(**settings),
        backbone=copy.deepcopy(backbone),
        tokenizer=tokenizer,
    ).eval()


def _logits(model: JevXForDecision, **call: Any) -> torch.Tensor:
    with torch.no_grad():
        return model(**{**CALL, **call}).decisions[0].logits.reshape(-1).clone()


def _count_encoder_calls(model: JevXForDecision) -> Dict[str, int]:
    """Count real invocations of the two memoized encoder entry points."""
    counters = {"question": 0, "schema": 0}
    question_original = model.question_encoder.encode_texts
    schema_original = model.schema_encoder.encode

    def counting_question(texts: Any, **kwargs: Any) -> Any:
        counters["question"] += 1
        return question_original(texts, **kwargs)

    def counting_schema(rows: Any, **kwargs: Any) -> Any:
        counters["schema"] += 1
        return schema_original(rows, **kwargs)

    model.question_encoder.encode_texts = counting_question  # type: ignore[method-assign]
    model.schema_encoder.encode = counting_schema  # type: ignore[method-assign]
    return counters


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max())


# ---------------------------------------------------------------------------
# 1. The regression: eval -> optimizer step -> eval must not serve stale H_Q/H_C
# ---------------------------------------------------------------------------


def test_optimizer_step_between_evals_invalidates_the_memo(
    tiny_bert: Any, tiny_tokenizer: Any
) -> None:
    """The exact Wave-2 protocol, on a real ``JevXForDecision``.

    Reference is a FRESH model with the SAME post-step weights and an empty
    memo — i.e. what a correct recomputation must produce. Before the fix the
    second eval returned the pre-step ``H_Q``/``H_C`` and the two differed by
    the reported 4.1556e+04.
    """
    torch.manual_seed(0)
    model = _wrapper(tiny_bert, tiny_tokenizer)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-2)

    first = _logits(model)
    assert model._condition_cache, "the first eval must fill the memo"

    # One real AdamW step on a real loss (the memo is bypassed under grad).
    model.train()
    optimizer.zero_grad(set_to_none=True)
    out = model(**CALL)
    loss = nn.functional.cross_entropy(
        torch.stack([d.logits.reshape(-1) for d in out.decisions]),
        torch.zeros(1, dtype=torch.long),
    )
    loss.backward()
    optimizer.step()
    model.eval()

    counters = _count_encoder_calls(model)
    stale = _logits(model)
    # A correct recomputation ran both encoders again.
    assert counters == {"question": 1, "schema": 1}, (
        f"the memo survived a parameter update (encoder calls: {counters})"
    )

    # The reference: the same weights, no memo.
    reference_model = _wrapper(tiny_bert, tiny_tokenizer)
    reference_model.load_state_dict(model.state_dict(), strict=False)
    reference_model.eval()
    reference_model.config.conditioning_mode = "full"
    fresh = _logits(reference_model)

    difference = _max_abs_diff(stale, fresh)
    assert difference <= 1e-6, (
        "the memo served encodings computed from the PRE-step weights: max abs "
        f"logit diff vs a fresh-cache evaluation is {difference:.4e} "
        "(Wave-2 probe measured 4.1556e+04)"
    )
    # The control: the step really moved the weights, so this is not a vacuous
    # comparison of two identical models.
    assert _max_abs_diff(stale, first) > 1e-3, (
        "the optimizer step did not move the decision at all, so the test is "
        "not exercising the invalidation"
    )


def test_in_place_parameter_edit_invalidates_the_memo(
    tiny_bert: Any, tiny_tokenizer: Any
) -> None:
    """The same guarantee for a hand-made in-place edit, no optimizer.

    Covers the "the one weight change the model cannot observe" case named in
    the old ``clear_condition_cache`` docstring.
    """
    torch.manual_seed(1)
    model = _wrapper(tiny_bert, tiny_tokenizer)
    before = _logits(model)

    with torch.no_grad():
        weight = model.question_encoder.hash_embedding.weight
        weight.add_(torch.randn_like(weight) * 0.5)

    counters = _count_encoder_calls(model)
    after = _logits(model)
    assert counters == {"question": 1, "schema": 1}, (
        "an in-place edit of a memoized encoder tensor was not observed"
    )

    reference_model = _wrapper(tiny_bert, tiny_tokenizer)
    reference_model.load_state_dict(model.state_dict(), strict=False)
    reference_model.config.conditioning_mode = "full"
    fresh = _logits(reference_model)
    assert _max_abs_diff(after, fresh) <= 1e-6
    assert _max_abs_diff(after, before) > 1e-3


def test_schema_encoder_only_change_also_invalidates(
    tiny_bert: Any, tiny_tokenizer: Any
) -> None:
    """H_C is memoized separately from H_Q; editing E_C must invalidate too."""
    torch.manual_seed(2)
    model = _wrapper(tiny_bert, tiny_tokenizer)
    _logits(model)
    with torch.no_grad():
        model.schema_encoder.text_encoder.hash_embedding.weight.add_(0.25)
    counters = _count_encoder_calls(model)
    _logits(model)
    assert counters == {"question": 1, "schema": 1}


# ---------------------------------------------------------------------------
# 2. The STATIC-model guarantee: the fast path is preserved bitwise
# ---------------------------------------------------------------------------


def test_static_model_still_hits_the_memo_bitwise(
    tiny_bert: Any, tiny_tokenizer: Any
) -> None:
    """(a) Repeated identical content on an UNCHANGED model is a bitwise hit.

    Requirement (d): the cache-hit fast path must still avoid re-encoding, and
    requirement (a): the Wave-2 identity evidence depends on it.
    """
    torch.manual_seed(3)
    model = _wrapper(tiny_bert, tiny_tokenizer)
    model.config.conditioning_mode = "full"
    reference = _logits(model)
    model.config.conditioning_mode = "constant"

    counters = _count_encoder_calls(model)
    cold = _logits(model)
    warm = _logits(model)
    warm_again = _logits(model)
    assert counters == {"question": 1, "schema": 1}, (
        f"a STATIC model re-encoded: encoder calls {counters}; the memo's whole "
        "purpose is that the 2nd and 3rd identical calls do no encoder work"
    )
    assert torch.equal(cold, reference)
    assert torch.equal(warm, reference)
    assert torch.equal(warm_again, reference)


def test_the_version_check_does_not_invalidate_a_never_mutated_model(
    tiny_bert: Any, tiny_tokenizer: Any
) -> None:
    """Reading the signature must be side-effect free."""
    torch.manual_seed(4)
    model = _wrapper(tiny_bert, tiny_tokenizer)
    _logits(model)
    filled = dict(model._condition_cache)
    for _ in range(5):
        model._condition_cache_signature()
    assert model._condition_cache == filled


def test_signature_covers_only_the_two_memoized_encoders(
    tiny_bert: Any, tiny_tokenizer: Any
) -> None:
    """The projections run AFTER the lookup, so editing them must NOT evict.

    Pinned so the invalidation set is not silently widened: an over-wide set
    would be correct but would throw away the memo on every head update.
    """
    torch.manual_seed(5)
    model = _wrapper(tiny_bert, tiny_tokenizer)
    _logits(model)
    with torch.no_grad():
        model.question_proj.proj.weight.add_(0.1)
        model.schema_proj.proj.weight.add_(0.1)
    counters = _count_encoder_calls(model)
    _logits(model)
    assert counters == {"question": 0, "schema": 0}, (
        "question_proj/schema_proj are applied AFTER the lookup, so editing "
        f"them must not drop the memo (they evicted it: {counters})"
    )


# ---------------------------------------------------------------------------
# 3. The cost of the check — tensor-level, measured
# ---------------------------------------------------------------------------


def test_the_invalidation_check_is_tensor_level_and_cheap(
    tiny_bert: Any, tiny_tokenizer: Any
) -> None:
    """(c) No per-scalar walk, and the per-call cost is documented.

    The check reads one ``int`` per parameter/buffer TENSOR of the two
    encoders. The scalars behind those tensors are ~1.1 M (the conditioning
    half of a JevX stack); a per-scalar loop would be ~1.1 M Python steps, so
    the tensor count is asserted to be orders of magnitude smaller.
    """
    model = _wrapper(tiny_bert, tiny_tokenizer)
    tracked = [
        tensor
        for module in (model.question_encoder, model.schema_encoder)
        for tensor in JevXForDecision._module_tensors(module)
    ]
    # The hand-rolled walk must agree exactly with torch's own view.
    assert set(tracked) == set(
        [*model.question_encoder.parameters(), *model.question_encoder.buffers(),
         *model.schema_encoder.parameters(), *model.schema_encoder.buffers()]
    )
    scalars = sum(tensor.numel() for tensor in tracked)
    assert 0 < len(tracked) <= 64, (
        f"the tracked tensor set is {len(tracked)} tensors; the check must be "
        "tensor-level (one version counter per tensor), not per-parameter"
    )
    assert scalars > 100_000, (
        f"the fixture only tracks {scalars} scalars, so this measurement does "
        "not demonstrate anything about a 100M+ scalar stack"
    )
    assert len(model._condition_cache_signature()) == len(tracked)

    # Measured: 20k signature computations on a contended 4-core box.
    iterations = 20_000
    started = time.perf_counter()
    for _ in range(iterations):
        model._condition_cache_signature()
    per_call_us = (time.perf_counter() - started) / iterations * 1e6
    assert per_call_us < 500.0, (
        f"the invalidation check costs {per_call_us:.1f} us per call; it must "
        "stay far below the ~7000-12000 us/example conditioning forward it "
        "guards (reports/MULTI_QUESTION.md §2.3)"
    )
    print(
        f"\n  [cost] signature over {len(tracked)} tensors / {scalars} scalars: "
        f"{per_call_us:.2f} us per call ({iterations} iterations)"
    )


# ---------------------------------------------------------------------------
# 4. Interaction with the documented invalidation entry points
# ---------------------------------------------------------------------------


def test_clear_condition_cache_still_forces_a_miss_and_is_idempotent(
    tiny_bert: Any, tiny_tokenizer: Any
) -> None:
    """The hand-written hint keeps working (and now also resets the key)."""
    torch.manual_seed(6)
    model = _wrapper(tiny_bert, tiny_tokenizer)
    _logits(model)
    assert model._condition_cache
    model.clear_condition_cache()
    assert model._condition_cache == {}
    assert model._condition_cache_versions is None
    model.clear_condition_cache()  # idempotent
    counters = _count_encoder_calls(model)
    _logits(model)
    assert counters == {"question": 1, "schema": 1}


def test_load_state_dict_still_clears_the_memo(
    tiny_bert: Any, tiny_tokenizer: Any
) -> None:
    """Regression guard for the pre-existing (already-correct) hook."""
    torch.manual_seed(7)
    model = _wrapper(tiny_bert, tiny_tokenizer)
    _logits(model)
    other = _wrapper(tiny_bert, tiny_tokenizer)
    model.load_state_dict(other.state_dict(), strict=False)
    assert model._condition_cache == {}


def test_a_train_step_that_never_touches_the_still_works(
    tiny_bert: Any, tiny_tokenizer: Any
) -> None:
    """After a step that only moves NON-encoder weights, the memo is kept.

    Not a correctness requirement (dropping it would also be correct) — it
    pins that the invalidation set is the two memoized encoders and not the
    whole model, so the memo does not become useless in a mixed freeze.
    """
    torch.manual_seed(8)
    model = _wrapper(tiny_bert, tiny_tokenizer)
    _logits(model)
    with torch.no_grad():
        model.fusion.option_proj.weight.add_(0.05)
    counters = _count_encoder_calls(model)
    _logits(model)
    assert counters == {"question": 0, "schema": 0}
