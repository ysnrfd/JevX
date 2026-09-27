# Copyright (c) 2026 ysnrfd
"""Failure-mode contract tests (JevX phase 90) — one test per documented row.

Every test asserts a USEFUL error: pytest.raises plus a match on a message
substring that names the offending argument or carries the structured fields
of UnsupportedRepresentationError. Rows:

1. empty questions list -> ValueError
2. empty options list for choice -> ValueError
3. single option K=1 choice -> works, probabilities == [1.0]
4. duplicate options -> allowed, invariants hold
5. very long option description (5000 chars) -> works, invariants hold
6. long state (~2000 tokens vs tiny max_positions) -> descriptive ValueError
   (documented contract: raise, never silently truncate — fixed in
   jevx/modeling_jevx.py forward; see reports/FAILURE_ANALYSIS.md)
7. mismatched attention_mask batch -> ValueError naming both batches
8. invalid mask (negative / float) -> descriptive error where JevX validates
   (fixed: forward validates mask values — see FAILURE_ANALYSIS.md)
9. NaN input tensor -> check_finite raises with the tensor name
10. wrong dtype (long where float expected, pre-encoded questions) -> TypeError
11. wrong device -> device-generic; skipped ONLY when no second device exists
12. incompatible model (logits-only forward) -> UnsupportedRepresentationError
    with structured fields
13. vision state without processor -> clear ValueError
14. remote-code: trust_remote_code=False on auto_map fixture -> explicit
    failure, no silent code execution
15. quantized model -> ENVIRONMENT-BLOCKED (CPU-only, bitsandbytes absent);
    documented in reports/FAILURE_ANALYSIS.md, deliberately no test here
16. save/load after isotonic fit -> sidecar round-trip works post-fit

All models are in-memory (tests/conftest.py fixtures); no network.
"""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass

import pytest
import torch
from transformers import BertConfig, BertModel, PreTrainedConfig, ViTModel
from transformers.utils import ModelOutput

from jevx import (
    JevX,
    JevXConfig,
    JevXForDecision,
    UnsupportedRepresentationError,
)

HIDDEN = 32
STATE = "hello world"


def build_wrapper(backbone, tokenizer=None, **config_overrides) -> JevXForDecision:
    """Wrap a deep-copied backbone with a tiny decision config (eval mode)."""
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


@pytest.fixture()
def wrapper(tiny_bert, tiny_tokenizer) -> JevXForDecision:
    return build_wrapper(tiny_bert, tokenizer=tiny_tokenizer)


# ---------------------------------------------------------------------------
# Row 1 — empty questions list
# ---------------------------------------------------------------------------


def test_empty_questions_list_raises_value_error(wrapper) -> None:
    with pytest.raises(ValueError, match="at least one question"):
        wrapper.choice(STATE, [], options=[["a", "b"]])


# ---------------------------------------------------------------------------
# Row 2 — empty options list for choice
# ---------------------------------------------------------------------------


def test_empty_options_list_for_choice_raises_value_error(wrapper) -> None:
    # (a) options=[] -> the pair has no schema at all
    with pytest.raises(ValueError, match="requires a schema"):
        wrapper.choice(STATE, ["q"], options=[])
    # (b) options=[[]] -> a schema with K=0 (axiom D3.1)
    with pytest.raises(ValueError, match="K=0"):
        wrapper.choice(STATE, ["q"], options=[[]])


# ---------------------------------------------------------------------------
# Row 3 — single option K=1 choice works, probs == [1.0]
# ---------------------------------------------------------------------------


def test_single_option_choice_probs_are_exactly_one(wrapper) -> None:
    with torch.no_grad():
        out = wrapper.choice(STATE, ["q"], options=[["only"]])
    decision = out.decisions[0]
    probs = decision.probabilities
    assert probs.shape == (1, 1)
    assert probs.tolist() == [[1.0]]
    assert int(decision.predicted_index) == 0
    assert decision.predicted_value == "only"
    assert float(decision.confidence) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Row 4 — duplicate options allowed, invariants hold
# ---------------------------------------------------------------------------


def test_duplicate_options_allowed_invariants_hold(wrapper) -> None:
    options = [["same", "same", "other"]]
    with torch.no_grad():
        out = wrapper.choice(STATE, ["q"], options=options)
    decision = out.decisions[0]
    probs = decision.probabilities
    # invariants: finite, in [0, 1], rows sum to 1, index in schema
    assert torch.isfinite(probs).all()
    assert bool((probs >= 0).all()) and bool((probs <= 1).all())
    assert float(probs.sum()) == pytest.approx(1.0, abs=1e-6)
    assert 0 <= int(decision.predicted_index) < 3
    # identical option texts encode identically -> identical probabilities
    assert float(probs[0, 0]) == float(probs[0, 1])
    # schema projection returns one of the (duplicate) keys
    assert decision.predicted_value in {"same", "other"}


# ---------------------------------------------------------------------------
# Row 5 — very long option description (5000 chars)
# ---------------------------------------------------------------------------


def test_very_long_option_description_works(wrapper) -> None:
    long_text = "x" * 5000
    options = [[{"key": "k1", "label": "first", "description": long_text}, {"key": "k2"}]]
    with torch.no_grad():
        out = wrapper.choice(STATE, ["q"], options=options)
    probs = out.decisions[0].probabilities
    assert probs.shape == (1, 2)
    assert torch.isfinite(probs).all()
    assert float(probs.sum()) == pytest.approx(1.0, abs=1e-6)
    assert 0 <= int(out.decisions[0].predicted_index) < 2


# ---------------------------------------------------------------------------
# Row 6 — long state vs learned position limit (documented: raise, no truncate)
# ---------------------------------------------------------------------------


def test_long_state_exceeding_position_limit_raises_descriptive_error(
    wrapper, tiny_tokenizer
) -> None:
    # fixture tiny_bert: max_position_embeddings=64 -> 2000 tokens must raise
    # a descriptive ValueError naming both lengths (never a kernel RuntimeError).
    ids = torch.randint(2, 99, (1, 2000))
    mask = torch.ones(1, 2000, dtype=torch.long)
    with pytest.raises(ValueError, match="state token length 2000 exceeds"):
        with torch.no_grad():
            wrapper.choice({"input_ids": ids, "attention_mask": mask}, ["q"],
                           options=[["a", "b"]])
    # raw-text path tokenizes to the same length -> same documented error
    long_text = " ".join(["hello"] * 2000)
    assert long_text.split() and len(wrapper.tokenizer(long_text)["input_ids"]) > 64
    with pytest.raises(ValueError, match="exceeds the backbone's learned position limit"):
        with torch.no_grad():
            wrapper.choice(long_text, ["q"], options=[["a", "b"]])
    # boundary: exactly at the limit still works
    at_limit = torch.randint(2, 99, (1, 64))
    with torch.no_grad():
        out = wrapper.choice(
            {"input_ids": at_limit, "attention_mask": torch.ones(1, 64, dtype=torch.long)},
            ["q"], options=[["a", "b"]],
        )
    assert float(out.decisions[0].probabilities.sum()) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Row 7 — mismatched batch (attention_mask batch != input_ids batch)
# ---------------------------------------------------------------------------


def test_mismatched_attention_mask_batch_raises_naming_both(wrapper) -> None:
    ids = torch.randint(2, 99, (2, 5))
    mask = torch.ones(3, 5, dtype=torch.long)
    with pytest.raises(ValueError, match="attention_mask"):
        with torch.no_grad():
            wrapper.choice({"input_ids": ids, "attention_mask": mask}, ["q"],
                           options=[["a", "b"]])
    with pytest.raises(ValueError, match="state batch 2"):
        with torch.no_grad():
            wrapper.choice({"input_ids": ids, "attention_mask": mask}, ["q"],
                           options=[["a", "b"]])


# ---------------------------------------------------------------------------
# Row 8 — invalid mask values (negative / float) -> descriptive error
# ---------------------------------------------------------------------------


def test_negative_attention_mask_values_raise_descriptive_error(wrapper) -> None:
    ids = torch.randint(2, 99, (1, 5))
    mask = torch.tensor([[1, -1, 1, -1, 1]])
    with pytest.raises(ValueError) as excinfo:
        with torch.no_grad():
            wrapper.choice({"input_ids": ids, "attention_mask": mask}, ["q"],
                           options=[["a", "b"]])
    message = str(excinfo.value)
    assert "attention_mask" in message
    assert "expected values in {0, 1}" in message
    assert "-1" in message  # names the offending value


def test_float_attention_mask_raises_descriptive_error(wrapper) -> None:
    ids = torch.randint(2, 99, (1, 5))
    mask = torch.tensor([[1.0, 0.5, 1.0, 0.0, 1.0]])
    with pytest.raises(ValueError) as excinfo:
        with torch.no_grad():
            wrapper.choice({"input_ids": ids, "attention_mask": mask}, ["q"],
                           options=[["a", "b"]])
    message = str(excinfo.value)
    assert "attention_mask" in message
    assert "floating dtype torch.float32" in message
    assert "binarize" in message


def test_bool_and_zero_one_int_masks_still_accepted(wrapper) -> None:
    ids = torch.randint(2, 99, (1, 5))
    for mask in (
        torch.tensor([[True, True, False, True, True]]),
        torch.tensor([[1, 1, 0, 1, 1]]),
    ):
        with torch.no_grad():
            out = wrapper.choice({"input_ids": ids, "attention_mask": mask}, ["q"],
                                 options=[["a", "b"]])
        probs = out.decisions[0].probabilities
        assert float(probs.sum()) == pytest.approx(1.0, abs=1e-6)


def test_float_option_mask_in_fusion_raises_type_error() -> None:
    from jevx.representations.fusion import build_fusion

    fusion = build_fusion("concat", HIDDEN)
    state = torch.randn(2, HIDDEN)
    question = torch.randn(2, HIDDEN)
    options = torch.randn(2, 3, HIDDEN)
    with pytest.raises(TypeError, match="option_mask"):
        fusion(state, question, options, torch.ones(2, 3))  # float, must be bool


# ---------------------------------------------------------------------------
# Row 9 — NaN input tensor -> check_finite names the tensor
# ---------------------------------------------------------------------------


def test_nan_preencoded_questions_raise_check_finite_with_tensor_name(tiny_bert, tiny_tokenizer) -> None:
    """A non-finite question tensor is rejected and NAMED.

    Explicit ``conditioning_mode="full"``: since DECISION-3 the shipped default
    is ``"off"``, and ``off`` neutralizes the question path by substituting exact
    zeros BEFORE the projection, so a NaN question is discarded rather than read
    (that is what "off" means - the call does not read the question at all).
    This contract is therefore about the path that DOES read the question, which
    is where a NaN can reach a logit. The companion property - that the default
    off path cannot be poisoned by a NaN question - is pinned in
    ``tests/unit/test_conditioning_off_matches_state_only.py``.
    """
    wrapper = build_wrapper(tiny_bert, tokenizer=tiny_tokenizer, conditioning_mode="full")
    questions = torch.full((1, HIDDEN), float("nan"))
    with pytest.raises(ValueError) as excinfo:
        with torch.no_grad():
            wrapper.choice(STATE, questions, options=[["a", "b"]])
    message = str(excinfo.value)
    assert "question_representations H_Q" in message  # tensor name
    assert "NaN" in message  # offending values reported


# ---------------------------------------------------------------------------
# Row 10 — wrong dtype (long where float expected in pre-encoded questions)
# ---------------------------------------------------------------------------


def test_long_dtype_preencoded_questions_raise_type_error(wrapper) -> None:
    questions = torch.ones(1, HIDDEN, dtype=torch.long)
    with pytest.raises(TypeError, match="floating point"):
        with torch.no_grad():
            wrapper.choice(STATE, questions, options=[["a", "b"]])


# ---------------------------------------------------------------------------
# Row 11 — wrong device (device-generic; skip ONLY with a single device)
# ---------------------------------------------------------------------------


def _available_devices() -> list[str]:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices += [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        devices.append("mps")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        devices.append("xpu")
    return devices


def test_cross_device_input_raises_device_error(tiny_bert, tiny_tokenizer) -> None:
    """Model on devices[0], questions tensor on devices[1] -> device error.

    SKIP CONDITION (documented): skipped ONLY when no second torch device
    exists — this environment is CPU-only (torch.cuda.is_available() False,
    no MPS/XPU), so a genuine cross-device mismatch cannot be constructed.
    """
    devices = _available_devices()
    if len(devices) < 2:
        pytest.skip(
            "only one torch device is available (CPU-only environment: "
            "torch.cuda.is_available() is False, no MPS/XPU) — a cross-device "
            "input mismatch cannot be constructed on this host"
        )
    wrapper = build_wrapper(tiny_bert, tokenizer=tiny_tokenizer)
    other = devices[1]
    questions = torch.ones(1, HIDDEN, device=other)  # deliberately wrong device
    with pytest.raises((RuntimeError, ValueError), match="device"):
        with torch.no_grad():
            wrapper.choice(STATE, questions, options=[["a", "b"]])


# ---------------------------------------------------------------------------
# Row 12 — incompatible model (forward returns only logits)
# ---------------------------------------------------------------------------


@dataclass
class LogitsOnlyOutput(ModelOutput):
    logits: torch.Tensor


class LogitsOnlyBert(BertModel):
    """BertModel subclass whose forward exposes ONLY a logits field.

    Every representation priority in the adapter (pooler_output, semantic
    fields, last_hidden_state, hidden_states[-1]) fails on this output.
    """

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        output_hidden_states=None,
        output_attentions=None,
        return_dict=None,
        **kwargs,
    ):
        base = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return LogitsOnlyOutput(logits=base.pooler_output)


def _logits_only_wrapper() -> JevXForDecision:
    config = BertConfig(
        vocab_size=99,
        hidden_size=HIDDEN,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    model = LogitsOnlyBert(config).eval()
    from tokenizers import Tokenizer, models as tokenizers_models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(tokenizers_models.WordLevel({"[UNK]": 0, "[PAD]": 1, "hello": 2,
                                                     "world": 3, "a": 4, "b": 5, "q": 6},
                                                    unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                                        pad_token="[PAD]")
    wrapper = JevXForDecision(
        JevXConfig(decision_hidden_size=HIDDEN, question_hidden_size=HIDDEN,
                   schema_hidden_size=HIDDEN),
        backbone=model,
        tokenizer=tokenizer,
    )
    wrapper.eval()
    return wrapper


def test_logits_only_model_raises_structured_unsupported_representation() -> None:
    wrapper = _logits_only_wrapper()
    with torch.no_grad():
        with pytest.raises(UnsupportedRepresentationError) as excinfo:
            wrapper.choice(STATE, ["q"], options=[["a", "b"]])
    error = excinfo.value
    # structured fields all populated and actionable
    assert error.model_name == "LogitsOnlyBert"
    assert error.architecture
    assert "logits" in error.detected_outputs
    assert error.attempted_strategies, "attempted_strategies must record what was tried"
    assert error.failed_strategies, "failed_strategies must record why each failed"
    assert any("pooler_output" in entry for entry in error.failed_strategies)
    assert any("last_hidden_state" in entry for entry in error.failed_strategies)
    assert error.suggestion
    # the human-readable message carries the same facts
    message = str(error)
    assert "Unsupported representation" in message
    assert "Suggestion:" in message


# ---------------------------------------------------------------------------
# Row 13 — vision state without a processor
# ---------------------------------------------------------------------------


def test_vision_state_without_processor_raises_clear_error(tiny_vit) -> None:
    wrapper = build_wrapper(tiny_vit, tokenizer=None)  # no processor attached
    images = torch.rand(1, 3, 32, 32)
    with pytest.raises(ValueError, match="require processor"):
        with torch.no_grad():
            wrapper.choice({"images": images}, ["q"], options=[["a", "b"]])


# ---------------------------------------------------------------------------
# Row 14 — remote code refused with trust_remote_code=False (local fixture)
# ---------------------------------------------------------------------------


def test_remote_code_fixture_refused_without_trust_remote_code(tmp_path) -> None:
    """auto_map config pointing at a MISSING module must fail explicitly.

    Asserts the trust_remote_code error AND that no code ran: the target
    module name never lands in sys.modules and no .py/__pycache__ artifacts
    are created inside the fixture directory.
    """
    import json

    fixture = tmp_path / "remote-code-fixture"
    fixture.mkdir()
    (fixture / "config.json").write_text(
        json.dumps(
            {
                "model_type": "jevx_remote_test",
                "architectures": ["RemoteTestModel"],
                "auto_map": {
                    "AutoConfig": "jevx_remote_missing.RemoteConfig",
                    "AutoModel": "jevx_remote_missing.RemoteModel",
                },
                "hidden_size": 32,
            }
        ),
        encoding="utf-8",
    )
    sys.modules.pop("jevx_remote_missing", None)
    with pytest.raises(Exception, match="trust_remote_code"):
        JevX.from_pretrained(str(fixture), trust_remote_code=False)
    # no silent code execution
    assert "jevx_remote_missing" not in sys.modules
    assert not list(fixture.glob("**/*.py"))
    assert not list(fixture.glob("**/__pycache__"))


# ---------------------------------------------------------------------------
# Row 15 — quantized model: ENVIRONMENT-BLOCKED, no fake test.
# See reports/FAILURE_ANALYSIS.md row 15 (bitsandbytes absent, CPU-only env).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Row 16 — save/load after calibration: fitted isotonic sidecar round-trip
# ---------------------------------------------------------------------------


def test_fitted_isotonic_sidecar_roundtrip_after_save_load(tmp_path, tiny_bert,
                                                           tiny_tokenizer) -> None:
    wrapper = build_wrapper(tiny_bert, tiny_tokenizer, calibration_method="isotonic")
    torch.manual_seed(7)
    wrapper.calibration_engine.fit(torch.randn(64, 4), torch.randint(0, 4, (64,)))
    assert wrapper.calibration_engine.is_fitted
    x_before = wrapper.calibrator.x_points.clone()
    y_before = wrapper.calibrator.y_points.clone()
    assert x_before.numel() > 0

    questions = ["q1", "q2"]
    options = [["a", "b", "c", "d"]] * 2
    with torch.no_grad():
        before = wrapper.choice(STATE, questions, options=options)
    assert before.decisions[0].calibration_metadata == {
        "method": "isotonic",
        "applied": True,
    }

    save_dir = tmp_path / "jevx-isotonic"
    wrapper.save_pretrained(save_dir)
    assert (save_dir / "jevx_calibration.pt").is_file()

    reloaded = JevXForDecision.from_pretrained(str(save_dir))
    reloaded.eval()
    assert reloaded.calibration_engine.method == "isotonic"
    assert reloaded.calibration_engine.is_fitted
    assert torch.equal(reloaded.calibrator.x_points, x_before)
    assert torch.equal(reloaded.calibrator.y_points, y_before)
    with torch.no_grad():
        after = reloaded.choice(STATE, questions, options=options)
    assert after.decisions[0].calibration_metadata == {
        "method": "isotonic",
        "applied": True,
    }
    max_abs_error = max(
        float((a.probabilities - b.probabilities).abs().max())
        for a, b in zip(before.decisions, after.decisions)
    )
    assert max_abs_error <= 1e-6, f"post-fit round-trip drifted: {max_abs_error}"
