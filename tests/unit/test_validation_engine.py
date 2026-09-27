# Copyright (c) 2026 ysnrfd
"""Q-003 ValidationEngine tests (fix-wave contract, docs/API.md §3).

Covers:

1. Contract surface — ``jevx.validation.ValidationEngine`` exported from
   the package root, both methods keyword-only ``context=""``, and
   ``JevXConfig.runtime_validation`` defaulting to ``True``.
2. On-path violations — a doctored representation (NaN pooled) and a
   doctored head output (rows not summing to 1) make ``forward`` raise a
   ``ValueError`` whose message names the check and the call site.
3. The flag gate — ``runtime_validation=False`` skips the checks: a
   doctored call returns instead of raising, and a healthy call with the
   flag on/off produces bit-for-bit identical outputs.
4. Direct validator checks — choice/score simplex, noul interior,
   confidence range (with the documented energy pre-fit carve-out),
   predicted-index range, shape/finiteness errors, and the
   JevXOutput/list dispatch.
"""

from __future__ import annotations

import copy
import inspect

import pytest
import torch

import jevx
from jevx import JevXConfig, JevXForDecision, ValidationEngine
from jevx.adapters import Representation
from jevx.modeling_outputs import DecisionOutput, JevXOutput

HIDDEN = 32  # matches the tiny_* fixtures' hidden_size
STATE = "hello world"
QUESTIONS = ["question option?"]
OPTIONS = [["a", "b", "c"]]


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


def _healthy_choice_decision() -> DecisionOutput:
    """A valid single-question choice decision (simplex rows, tag set)."""
    probs = torch.tensor([[0.2, 0.3, 0.5]])
    return DecisionOutput(
        logits=torch.tensor([[1.0, 2.0, 3.0]]),
        probabilities=probs,
        confidence=torch.tensor([0.5]),
        predicted_index=torch.tensor([2]),
        schema_metadata={"decision_type": "choice", "option_keys": ["a", "b", "c"]},
    )


# ---------------------------------------------------------------------------
# 1. Contract surface
# ---------------------------------------------------------------------------


def test_runtime_validation_defaults_true_in_fresh_config() -> None:
    assert JevXConfig().runtime_validation is True
    off = JevXConfig(runtime_validation=False)
    assert off.runtime_validation is False
    # survives the config serialization round-trip
    assert JevXConfig.from_dict(off.to_dict()).runtime_validation is False
    with pytest.raises(TypeError, match="runtime_validation"):
        JevXConfig(runtime_validation="yes")  # type: ignore[arg-type]


def test_contract_signatures_and_root_export() -> None:
    from jevx import validation

    assert jevx.ValidationEngine is validation.ValidationEngine
    for method_name in ("validate_representation", "validate_decision"):
        method = getattr(ValidationEngine, method_name)
        sig = inspect.signature(method)
        context = sig.parameters["context"]
        assert context.kind is inspect.Parameter.KEYWORD_ONLY
        assert context.default == ""
        assert sig.return_annotation in ("None", type(None).__class__, None) or (
            "None" in str(sig.return_annotation)
        )
    # healthy input returns None (no raise)
    assert ValidationEngine().validate_decision(_healthy_choice_decision()) is None


# ---------------------------------------------------------------------------
# 2. On-path violations (doctored outputs through forward)
# ---------------------------------------------------------------------------


def test_on_path_representation_violation_raises(tiny_bert, tiny_tokenizer) -> None:
    """A NaN pooled state extracted by the adapter fails forward at the
    post-extraction boundary with the validator's contextual message."""
    wrapper = _build_wrapper(tiny_bert, tokenizer=tiny_tokenizer)
    doctored = Representation(
        sequence_output=None,
        pooled=torch.tensor([[float("nan"), 0.0]]),
        attention_mask=None,
        raw_output=None,
        field="pooler_output",
        layout="bd",
    )
    wrapper.adapter.extract_representation = lambda *args, **kwargs: doctored
    with pytest.raises(ValueError, match=r"state extraction"):
        wrapper(state=STATE, questions=QUESTIONS, options=OPTIONS)


def test_on_path_decision_violation_raises_and_flag_gates_it(
    tiny_bert, tiny_tokenizer
) -> None:
    """A doctored head output (rows summing to 0.6) fails forward when
    runtime_validation is on, and flows through untouched when it is off."""
    wrapper = _build_wrapper(tiny_bert, tokenizer=tiny_tokenizer)
    head = wrapper.heads["choice"]
    original_forward = head.forward

    def doctored(fused, option_features=None, *, option_mask=None, **kwargs):
        out = original_forward(
            fused, option_features, option_mask=option_mask, **kwargs
        )
        # Break the simplex invariant: 3 columns x 0.2 = 0.6 != 1.
        out.probabilities = torch.full_like(out.probabilities, 0.2)
        return out

    head.forward = doctored
    try:
        assert wrapper.config.runtime_validation is True
        with pytest.raises(ValueError, match=r"decision pair 0 \(choice\)") as excinfo:
            wrapper(state=STATE, questions=QUESTIONS, options=OPTIONS)
        message = str(excinfo.value)
        assert "simplex" in message and "row 0 sums to" in message

        wrapper.config.runtime_validation = False
        out = wrapper(state=STATE, questions=QUESTIONS, options=OPTIONS)
        # skipped check -> the doctored distribution comes back untouched
        assert float(out.decisions[0].probabilities.sum()) == pytest.approx(0.6)
    finally:
        del head.forward


def test_flag_off_outputs_bitwise_identical_to_flag_on(
    tiny_bert, tiny_tokenizer
) -> None:
    """runtime_validation=False must not change a single bit of a healthy
    call: validation is read-only (bitwise equality, not just closeness)."""
    wrapper = _build_wrapper(tiny_bert, tokenizer=tiny_tokenizer)
    assert wrapper.config.runtime_validation is True
    with torch.no_grad():
        flagged = wrapper(state=STATE, questions=QUESTIONS, options=OPTIONS)
    wrapper.config.runtime_validation = False
    with torch.no_grad():
        unflagged = wrapper(state=STATE, questions=QUESTIONS, options=OPTIONS)

    assert torch.equal(
        flagged.state_representation, unflagged.state_representation
    )
    assert len(flagged.decisions) == len(unflagged.decisions)
    for on, off in zip(flagged.decisions, unflagged.decisions):
        assert torch.equal(on.logits, off.logits)
        assert torch.equal(on.probabilities, off.probabilities)
        assert torch.equal(on.confidence, off.confidence)
        assert torch.equal(on.predicted_index, off.predicted_index)
        assert on.schema_metadata == off.schema_metadata


def test_healthy_forward_output_passes_validator(tiny_bert, tiny_tokenizer) -> None:
    """The real on-path happy path: validating a JevXOutput raises nothing."""
    wrapper = _build_wrapper(tiny_bert, tokenizer=tiny_tokenizer)
    with torch.no_grad():
        out = wrapper(state=STATE, questions=QUESTIONS, options=OPTIONS)
    engine = ValidationEngine()
    assert engine.validate_decision(out, context="probe") is None
    # mixed-type call exercises the noul branch too
    with torch.no_grad():
        mixed = wrapper(
            state=STATE,
            questions=QUESTIONS + ["binary?"],
            decision_type=["choice", "noul"],
            options=[OPTIONS[0], None],
        )
    assert engine.validate_decision(mixed, context="probe-mixed") is None


# ---------------------------------------------------------------------------
# 3. Direct validator checks (doctored DecisionOutput / Representation)
# ---------------------------------------------------------------------------


def test_simplex_violation_names_row_and_invariant() -> None:
    bad = _healthy_choice_decision()
    bad.probabilities = torch.full((1, 3), 0.2)  # sums to 0.6
    with pytest.raises(ValueError, match=r"simplex.*row 0 sums to 0\.6"):
        ValidationEngine().validate_decision(bad, context="unit")


def test_noul_interior_violation() -> None:
    bad = DecisionOutput(
        logits=torch.zeros(1, 1),
        probabilities=torch.ones(1, 1),  # p == 1 violates (0, 1)
        confidence=torch.tensor([1.0]),
        predicted_index=torch.tensor([1]),
        schema_metadata={"decision_type": "noul"},
    )
    with pytest.raises(ValueError, match=r"strictly inside \(0, 1\)"):
        ValidationEngine().validate_decision(bad)
    # K > 1 for noul is also a type-tag/shape violation
    wide = _healthy_choice_decision()
    wide.schema_metadata = {"decision_type": "noul"}
    with pytest.raises(ValueError, match=r"exactly ONE sigmoid column"):
        ValidationEngine().validate_decision(wide)


def test_confidence_range_check_and_energy_carve_out() -> None:
    bad = _healthy_choice_decision()
    bad.confidence = torch.tensor([1.5])
    engine = ValidationEngine()
    with pytest.raises(ValueError, match=r"confidence must lie in \[0, 1\]"):
        engine.validate_decision(bad)
    # documented pre-fit energy behavior stays usable when opted in
    assert (
        ValidationEngine(allow_unbounded_confidence=True).validate_decision(bad)
        is None
    )


def test_predicted_index_out_of_range() -> None:
    bad = _healthy_choice_decision()
    bad.predicted_index = torch.tensor([3])  # K = 3 -> valid is [0, 2]
    with pytest.raises(ValueError, match=r"predicted_index out of range.*\[0, 2\]"):
        ValidationEngine().validate_decision(bad)
    # noul thresholds into {0, 1} even though K = 1
    noul = DecisionOutput(
        logits=torch.zeros(1, 1),
        probabilities=torch.full((1, 1), 0.7),
        confidence=torch.tensor([0.7]),
        predicted_index=torch.tensor([2]),
        schema_metadata={"decision_type": "noul"},
    )
    with pytest.raises(ValueError, match=r"predicted_index out of range.*\[0, 1\]"):
        ValidationEngine().validate_decision(noul)


def test_shape_and_finiteness_violations() -> None:
    bad = _healthy_choice_decision()
    bad.probabilities = bad.probabilities.reshape(1, 1, 3)  # wrong rank
    with pytest.raises(ValueError, match=r"expected the logits shape"):
        ValidationEngine().validate_decision(bad)

    nonfinite = _healthy_choice_decision()
    nonfinite.logits = torch.tensor([[float("nan"), 0.0, 1.0]])
    with pytest.raises(ValueError, match=r"decision logits.*non-finite"):
        ValidationEngine().validate_decision(nonfinite)

    not_a_decision = object()
    with pytest.raises(TypeError, match=r"expected a DecisionOutput"):
        ValidationEngine().validate_decision(not_a_decision)
    with pytest.raises(ValueError, match=r"empty decision list"):
        ValidationEngine().validate_decision([])


def test_representation_checks_direct() -> None:
    engine = ValidationEngine()
    healthy = Representation(
        sequence_output=torch.randn(2, 4, HIDDEN),
        pooled=torch.randn(2, HIDDEN),
        attention_mask=torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]]),
        raw_output=None,
        field="last_hidden_state",
        layout="btd",
    )
    assert engine.validate_representation(healthy, context="unit") is None
    # bare tensor path
    assert engine.validate_representation(torch.randn(2, HIDDEN)) is None

    # pooled=None with a sequence present: extraction failed to pool, but
    # the object still constructs (Representation only aligns present tensors)
    missing = Representation(
        sequence_output=torch.randn(2, 4, HIDDEN),
        pooled=None,
        attention_mask=torch.ones(2, 4, dtype=torch.long),
        raw_output=None,
        field="last_hidden_state",
        layout="btd",
    )
    with pytest.raises(ValueError, match=r"no pooled state vector"):
        engine.validate_representation(missing, context="unit")

    wrong_rank = Representation(
        sequence_output=None,
        pooled=torch.randn(2, 3, HIDDEN),  # rank 3 as pooled
        attention_mask=None,
        raw_output=None,
        field="last_hidden_state",
        layout="bd",
    )
    with pytest.raises(ValueError, match=r"pooled state H_S.*\[B, D\]"):
        engine.validate_representation(wrong_rank, context="unit")

    with pytest.raises(TypeError, match=r"validate_representation"):
        engine.validate_representation("not-a-representation")
