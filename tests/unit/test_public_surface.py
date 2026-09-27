# Copyright (c) 2026 ysnrfd
"""Public-API surface tests (Q-010 / Q-011 of the quality audit).

Three groups:

1. **Q-011 — dynamic export sweep**: iterate ``jevx.__all__``, import every
   attribute from the package root, assert each is not ``None`` and carries an
   inspection docstring (public symbols must be documented), plus the
   version-triple equality ``jevx.__version__ == modeling_jevx.__version__ ==
   JevXConfig().version`` (the pyproject cross-check lives in the packaging
   agent's ``tests/test_version_consistency.py`` — deliberately not duplicated
   here).
2. **Q-011 — explicit smokes** for documented symbols the suite previously
   never named: ``DecisionDataset``, ``confusion_matrix``, ``extract_logits``,
   ``l2_regularization``, the calibration registry surface
   (``register_calibrator`` / ``build_calibrator`` / ``validate_fit_inputs`` /
   ``IdentityCalibrator``), the text-adapter helpers (``is_text_state`` /
   ``tokenize_texts``), the adapter observation constants
   (``observed_output_fields`` / ``NATIVE_REPRESENTATION_FIELDS`` /
   ``BUILTIN_ADAPTER_CHAIN``), and — guarded — ``ValidationEngine`` and the
   ``cross_attention`` fusion kind, both owned by the concurrent fix agent.
3. **Q-010 — §31 surface**: ``return_base_output=True`` populates
   ``JevXOutput.base_output``; the ``argument → config.decision_type →
   "choice"`` resolution rule (docs/API.md §1.3) including the config pin and
   explicit-argument override.

All models are the in-memory fixtures from ``tests/conftest.py`` — no network.
"""

from __future__ import annotations

import copy
import importlib
import inspect

import pytest
import torch

import jevx
from jevx import JevXConfig, JevXForDecision
from jevx import modeling_jevx
from jevx.adapters import (
    BUILTIN_ADAPTER_CHAIN,
    is_text_state,
    observed_output_fields,
    tokenize_texts,
)
from jevx.calibration import (
    IdentityCalibrator,
    build_calibrator,
    register_calibrator,
    validate_fit_inputs,
)
from jevx.capabilities import NATIVE_REPRESENTATION_FIELDS
from jevx.training import DecisionDataset, confusion_matrix, l2_regularization
from jevx.training.data import extract_logits

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


# ---------------------------------------------------------------------------
# Q-011 — dynamic sweep over jevx.__all__
# ---------------------------------------------------------------------------


def test_jevx_all_exports_are_importable_and_documented() -> None:
    """Every name in ``jevx.__all__`` imports from the root, is not None, and
    carries an inspection docstring (public symbols must be documented)."""
    assert isinstance(jevx.__all__, list) and len(jevx.__all__) > 0
    module = importlib.import_module("jevx")
    for name in jevx.__all__:
        attr = getattr(module, name, None)
        assert attr is not None, f"jevx.{name} resolves to None"
        doc = inspect.getdoc(attr)
        assert doc and doc.strip(), (
            f"jevx.{name} has no docstring/inspection doc — public symbols "
            "must be documented"
        )


def test_version_triple_is_consistent() -> None:
    """``jevx.__version__ == modeling_jevx.__version__ == JevXConfig().version``.

    (Cross-checks against pyproject belong to the packaging agent's
    test_version_consistency.py — not duplicated here.)
    """
    assert jevx.__version__ == modeling_jevx.__version__
    assert jevx.__version__ == JevXConfig().version


# ---------------------------------------------------------------------------
# Q-011 — explicit targeted smokes
# ---------------------------------------------------------------------------


def test_decision_dataset_structural_validation() -> None:
    """DecisionDataset accepts well-formed records, rejects missing keys and
    mismatched question/schema counts with the documented errors."""
    record = {
        "state": STATE,
        "questions": QUESTIONS,
        "decision_type": "choice",
        "options": OPTIONS,
        "labels": [0],
    }
    dataset = DecisionDataset([record])
    assert len(dataset) == 1
    assert dataset[0] is record

    with pytest.raises(ValueError, match="missing required key"):
        DecisionDataset([{k: v for k, v in record.items() if k != "labels"}])
    with pytest.raises(ValueError, match="must pair 1:1"):
        DecisionDataset([dict(record, questions=QUESTIONS * 2)])
    with pytest.raises(TypeError, match="expected a list"):
        DecisionDataset("not-a-list")
    with pytest.raises(TypeError, match="expected a dict"):
        DecisionDataset([record, "not-a-dict"])


def test_confusion_matrix_over_toy_logits() -> None:
    """Rows = true class, cols = predicted; counts land in the right cells."""
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 2.0]])
    labels = torch.tensor([0, 1, 0, 1])  # all four correct
    matrix = confusion_matrix(logits, labels)
    assert matrix.dtype == torch.long
    assert matrix.shape == (2, 2)
    assert matrix.tolist() == [[2, 0], [0, 2]]
    # one deliberate mistake moves a count off-diagonal
    bad = confusion_matrix(logits, torch.tensor([1, 1, 0, 1]))
    assert bad.sum().item() == 4
    assert bad[1, 0].item() == 1  # true class 1 predicted 0 once


def test_extract_logits_accepts_documented_shapes_and_rejects_others() -> None:
    """Bare tensor / ``.logits`` attribute / mapping key — else TypeError."""
    tensor = torch.randn(2, 3)
    assert extract_logits(tensor) is tensor

    class WithLogits:
        logits = tensor

    assert extract_logits(WithLogits()) is tensor
    assert extract_logits({"logits": tensor}) is tensor

    with pytest.raises(TypeError, match="extract_logits: could not find logits"):
        extract_logits(object())


def test_l2_regularization_value_and_guards() -> None:
    """``weight · Σ ‖p‖²`` over module parameters; negative weight rejected."""
    module = torch.nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        module.weight.fill_(2.0)  # 8 params × 4 = 32 → Σ‖p‖² = 32
    penalty = l2_regularization(module, 0.5)
    assert torch.is_tensor(penalty) and penalty.ndim == 0
    assert abs(float(penalty.detach()) - 0.5 * 32.0) < 1e-5

    with pytest.raises(ValueError, match=r"weight: expected a finite value >= 0"):
        l2_regularization(module, -1.0)
    with pytest.raises(TypeError, match="expected an nn.Module"):
        l2_regularization("not-a-module", 1.0)


def test_calibration_registry_surface() -> None:
    """register_calibrator / build_calibrator / validate_fit_inputs /
    IdentityCalibrator behave as documented (API.md §11)."""
    # validate_fit_inputs: shape/type/range guards name the offender
    logits = torch.randn(4, 3)
    labels = torch.tensor([0, 1, 2, 0])
    got_logits, got_labels = validate_fit_inputs(logits, labels)
    assert got_logits is logits and got_labels is labels
    with pytest.raises(ValueError, match=r"\[N, K\]"):
        validate_fit_inputs(torch.randn(4), labels[:1])
    with pytest.raises(TypeError, match="floating-point"):
        validate_fit_inputs(torch.ones(4, 3, dtype=torch.long), labels)

    # build_calibrator resolves registered names; identity is a no-op map
    identity = build_calibrator("none")
    assert isinstance(identity, IdentityCalibrator)
    out = identity.calibrate_logits(logits)
    assert torch.equal(out, logits)
    assert torch.equal(identity.calibrate_probs(torch.softmax(logits, -1)),
                       torch.softmax(logits, -1))

    # register_calibrator is idempotent for the same class, rejects collisions
    # (probe entries are removed afterwards so the exact-roster assertion in
    # test_calibration.py::... keeps passing regardless of run order)
    from jevx.calibration.base import CALIBRATOR_REGISTRY

    try:
        @register_calibrator("_surface_test_probe")
        class _Probe(IdentityCalibrator):
            """Probe calibrator registered only for this test."""

        assert _Probe.calibrator_name == "_surface_test_probe"
        with pytest.raises(ValueError, match="already registered"):
            @register_calibrator("_surface_test_probe")
            class _Other(IdentityCalibrator):
                """Second class colliding on the same registry name."""
    finally:
        CALIBRATOR_REGISTRY.pop("_surface_test_probe", None)


def test_text_adapter_helpers() -> None:
    """is_text_state / tokenize_texts over the in-memory tokenizer fixture."""
    assert is_text_state("hello world") is True
    assert is_text_state(["hello", "world"]) is True
    assert is_text_state([]) is False
    assert is_text_state(42) is False
    assert is_text_state(torch.zeros(2, 3)) is False

    with pytest.raises(ValueError, match="tokenizer"):
        tokenize_texts("hello world", None)


def test_tokenize_texts_returns_model_kwargs(tiny_tokenizer) -> None:
    batch = tokenize_texts("hello world", tiny_tokenizer)
    assert set(batch) >= {"input_ids", "attention_mask"}
    assert batch["input_ids"].shape[0] == 1
    assert batch["input_ids"].dtype == torch.long
    with pytest.raises(ValueError, match="empty"):
        tokenize_texts("   ", tiny_tokenizer)


def test_adapter_observation_constants() -> None:
    """observed_output_fields / NATIVE_REPRESENTATION_FIELDS /
    BUILTIN_ADAPTER_CHAIN are the documented shapes (API.md §7)."""
    fields = observed_output_fields({"last_hidden_state": torch.zeros(1, 2, 3),
                                     "pooler_output": None})
    assert fields == ["last_hidden_state"]
    assert observed_output_fields(torch.zeros(2)) == ["<tensor>"]

    assert isinstance(NATIVE_REPRESENTATION_FIELDS, frozenset)
    assert "pooler_output" in NATIVE_REPRESENTATION_FIELDS
    # (last_hidden_state is the generic fallback, NOT a "native" pooled field —
    # the frozenset holds named trained/pooled outputs only)
    assert "last_hidden_state" not in NATIVE_REPRESENTATION_FIELDS
    assert "encoder_last_hidden_state" in NATIVE_REPRESENTATION_FIELDS

    assert isinstance(BUILTIN_ADAPTER_CHAIN, list) and BUILTIN_ADAPTER_CHAIN
    names = [cls.name for cls in BUILTIN_ADAPTER_CHAIN]
    assert "generic" in names and "encoder" in names
    # documented resolution order ends with the always-true fallback
    assert names[-1] == "generic"


# ---------------------------------------------------------------------------
# Q-010 — return_base_output + decision_type config pin
# ---------------------------------------------------------------------------


def test_return_base_output_populates_base_output(tiny_bert, tiny_tokenizer) -> None:
    """``return_base_output=True`` → ``base_output`` not None and of the same
    class as the raw backbone's output (§31 surface)."""
    wrapper = _build_wrapper(tiny_bert, tokenizer=tiny_tokenizer)
    with torch.no_grad():
        out = wrapper(state=STATE, questions=QUESTIONS, options=OPTIONS,
                      return_base_output=True)
    assert out.base_output is not None, "return_base_output=True must populate base_output"
    # type matches what the backbone itself returns
    ids = torch.tensor([[2, 3, 1]])
    mask = torch.tensor([[1, 1, 0]])
    with torch.no_grad():
        raw = tiny_bert(input_ids=ids, attention_mask=mask)
    assert type(out.base_output) is type(raw)
    assert out.base_output.last_hidden_state is not None
    assert torch.isfinite(out.base_output.last_hidden_state).all()

    # default-off half stays None (also asserted in test_outputs_jevx)
    with torch.no_grad():
        default_out = wrapper(state=STATE, questions=QUESTIONS, options=OPTIONS)
    assert default_out.base_output is None


def test_decision_type_config_pin_resolves_forward(tiny_bert, tiny_tokenizer) -> None:
    """``config.decision_type="score"`` set at construction resolves when
    ``forward`` is called without a ``decision_type`` argument (the middle
    branch of argument → config.decision_type → "choice", API.md §1.3)."""
    wrapper = _build_wrapper(tiny_bert, tokenizer=tiny_tokenizer,
                             decision_type="score")
    assert wrapper.config.decision_type == "score"
    with torch.no_grad():
        out = wrapper(state=STATE, questions=QUESTIONS, options=OPTIONS)
    decision = out.decisions[0]
    # score decisions carry an expected score; choice decisions never do
    assert decision.expected_score is not None, (
        "config.decision_type='score' did not resolve — forward fell back to 'choice'"
    )
    assert float(decision.expected_score[0]) >= 0.0
    # config pin survives a to_dict/from_dict round-trip
    restored = JevXConfig.from_dict(wrapper.config.to_dict())
    assert restored.decision_type == "score"


def test_decision_type_explicit_argument_overrides_config_pin(
    tiny_bert, tiny_tokenizer
) -> None:
    """An explicit ``decision_type=`` argument beats the config pin."""
    wrapper = _build_wrapper(tiny_bert, tokenizer=tiny_tokenizer,
                             decision_type="score")
    with torch.no_grad():
        out = wrapper(state=STATE, questions=QUESTIONS, decision_type="choice",
                      options=OPTIONS)
    decision = out.decisions[0]
    assert decision.expected_score is None, (
        "explicit decision_type='choice' must override config.decision_type='score'"
    )
    assert decision.probabilities is not None
    assert tuple(decision.probabilities.shape) == (1, len(OPTIONS[0]))


# ---------------------------------------------------------------------------
# Q-011 — guarded: symbols owned by the concurrent fix agent
# ---------------------------------------------------------------------------
#
# The concurrent fix-pass code agent is landing `jevx/validation.py`
# (ValidationEngine) and the `cross_attention` fusion kind. If either import
# fails, only ITS test skips — with an explicit reason — so the rest of this
# file stays hard-imported and green. Never skip silently for pre-existing
# symbols (everything above this banner).


def test_validation_engine_contract() -> None:
    """ValidationEngine (jevx/validation.py): exact methods
    ``validate_representation(rep, *, context="")`` and
    ``validate_decision(decision_or_output, *, context="")``, exported from
    ``jevx``, wired behind ``JevXConfig.runtime_validation`` (default True)."""
    validation = pytest.importorskip(
        "jevx.validation",
        reason="jevx.validation not present yet — owned by the concurrent "
               "fix agent (ValidationEngine); this test skips only until it lands",
    )
    engine_cls = getattr(validation, "ValidationEngine", None)
    assert engine_cls is not None, (
        "jevx.validation exists but does not export ValidationEngine"
    )
    for method_name in ("validate_representation", "validate_decision"):
        method = getattr(engine_cls, method_name, None)
        assert method is not None, f"ValidationEngine missing {method_name}()"
        sig = inspect.signature(method)
        assert "context" in sig.parameters, (
            f"{method_name} must accept keyword-only context=''"
        )
        assert sig.parameters["context"].default == ""
        assert sig.parameters["context"].kind is inspect.Parameter.KEYWORD_ONLY

    assert getattr(jevx, "ValidationEngine", None) is engine_cls, (
        "ValidationEngine must be exported from the jevx package root"
    )
    config = JevXConfig()
    assert getattr(config, "runtime_validation", None) is True, (
        "JevXConfig.runtime_validation must default to True"
    )


def test_cross_attention_fusion_kind() -> None:
    """``fusion_type="cross_attention"`` (spec 5.e): state sequence as K/V,
    question embedding as query, masked stable softmax, optional keyword-only
    sequence input on ``Fusion.forward``."""
    from jevx.representations import FUSION_REGISTRY, build_fusion

    if "cross_attention" not in FUSION_REGISTRY:
        pytest.skip(
            "fusion kind 'cross_attention' not registered yet — owned by the "
            "concurrent fix agent; this test skips only until it lands"
        )
    fusion = build_fusion("cross_attention", HIDDEN)
    state = torch.randn(2, HIDDEN)
    question = torch.randn(2, HIDDEN)
    options = torch.randn(2, 3, HIDDEN)
    state_sequence = torch.randn(2, 5, HIDDEN)
    state_mask = torch.tensor([[True, True, True, False, False]] * 2)

    sig = inspect.signature(fusion.forward)
    assert "state_sequence" in sig.parameters, (
        "Fusion.forward must accept an optional keyword-only `state_sequence` "
        "input (5.e)"
    )
    assert sig.parameters["state_sequence"].kind is inspect.Parameter.KEYWORD_ONLY

    option_mask = torch.tensor([[True, True, False], [True, True, True]])
    with torch.no_grad():
        out = fusion(state, question, options=options,
                     option_mask=option_mask,
                     state_sequence=state_sequence, state_mask=state_mask)
    assert tuple(out.fused.shape) == (2, HIDDEN)
    assert out.option_features is not None
    assert tuple(out.option_features.shape) == (2, 3, HIDDEN)
    assert torch.isfinite(out.fused).all()
    # masked option rows are exactly zero (masked_fill contract, all kinds)
    assert torch.all(out.option_features[0, 2] == 0.0)
    # padded sequence positions are inert: perturbing them changes nothing
    perturbed = state_sequence.clone()
    perturbed[~state_mask] = 1e3
    with torch.no_grad():
        out2 = fusion(state, question, options=options,
                      option_mask=option_mask,
                      state_sequence=perturbed, state_mask=state_mask)
    assert torch.allclose(out.fused, out2.fused, atol=1e-5), (
        "padded sequence positions must not contribute (D1.2 padding invariance)"
    )
    # a non-cross-attention kind rejects the sequence input (documented raise)
    plain = build_fusion("concat", HIDDEN)
    with pytest.raises(ValueError, match="cross_attention|state_sequence"):
        plain(state, question, state_sequence=state_sequence)
