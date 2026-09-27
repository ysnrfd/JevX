# Copyright (c) 2026 ysnrfd
"""The config <-> runtime contract of ``JevXConfig``.

A field that is declared, validated and documented but read by nothing is
worse than a missing field, so this module pins the contract mechanically:

1. ``test_every_config_field_is_read_or_pinned_as_reserved`` walks the AST
   of every ``jevx/**/*.py`` file and requires each config field to be read
   outside ``jevx/configuration_jevx.py``, or to appear in
   ``RESERVED_CONFIG_FIELDS`` with a one-line reason. Adding a field that
   does nothing therefore fails here instead of shipping.
2. the three previously-inert fields are pinned by their RUNTIME EFFECT:
   ``conditioning_mode`` switches the conditioning path (and the
   ``"constant"`` memo is proven to skip the encoders and to recompute on a
   content change), ``schema_encoder_width`` narrows the option encoder, and
   ``shared_question_schema_encoder`` REJECTS its non-default value with a
   named error instead of accepting a switch that does nothing.
3. ``decision_type`` is a real declared field: validated, round-tripped and
   equal to the factory kwarg it replaces.
"""
from __future__ import annotations

import ast
import copy
import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, List, Set

import pytest
import torch
from transformers.configuration_utils import PreTrainedConfig

import jevx
from jevx.configuration_jevx import JevXConfig
from jevx.modeling_jevx import JevXForDecision

#: Fields JevXConfig declares that NO computation under ``jevx/`` reads.
#: Each entry carries the one-line reason it is allowed to be inert; a field
#: that is neither read nor listed here fails the mechanical scan.
RESERVED_CONFIG_FIELDS: Dict[str, str] = {
    "representation_strategy": (
        "free-form metadata recorded in config.json; no computation reads it - "
        "the pooling strategy that actually runs is config.pooling_method"
    ),
    "version": (
        "checkpoint format metadata recorded in config.json; no computation reads it"
    ),
}

#: Fields that are neither wired into a computation nor inert: they are
#: SPECIFIED BUT NOT IMPLEMENTED, so JevXConfig must REFUSE any non-default
#: value with a named error. Pinned by
#: ``test_reserved_fields_reject_their_non_default_value``.
REJECTED_NON_DEFAULT_FIELDS: Dict[str, str] = {
    "shared_question_schema_encoder": (
        "the shared module would appear twice in state_dict() "
        "(question_encoder.* and schema_encoder.text_encoder.*) and transformers "
        "refuses to serialize those shared tensors"
    ),
}

#: Tiny widths: the fixtures are 32-wide BERTs, and these tests are about the
#: wiring, not the numerics.
HIDDEN = 8
STATE = "hello world"


def _wrapper(backbone: Any, tokenizer: Any, **overrides: Any) -> JevXForDecision:
    """A tiny wrapper over a deep copy of the read-only backbone fixture."""
    settings: Dict[str, Any] = {
        "decision_hidden_size": HIDDEN,
        "question_hidden_size": HIDDEN,
        "schema_hidden_size": HIDDEN,
    }
    settings.update(overrides)
    return JevXForDecision(
        JevXConfig(**settings), backbone=copy.deepcopy(backbone), tokenizer=tokenizer
    ).eval()


# ---------------------------------------------------------------------------
# 1. the mechanical scan
# ---------------------------------------------------------------------------


def _declared_config_fields() -> List[str]:
    """Config fields JevXConfig itself declares (not the inherited ones)."""
    inherited = {f.name for f in dataclasses.fields(PreTrainedConfig)}
    return [
        f.name
        for f in dataclasses.fields(JevXConfig)
        if f.name not in inherited
    ]


def _read_sites(field_names: Set[str]) -> Dict[str, List[str]]:
    """``file:line`` of every real READ of each field under ``jevx/``.

    A read is an attribute access (``cfg.<field>``), a bare name, a keyword
    argument, or a string key handed to ``getattr``/``hasattr``. Prose
    inside docstrings and unrelated string constants are deliberately NOT
    counted, so a field that is only *documented* cannot pass this scan.
    """
    root = Path(jevx.__file__).parent
    hits: Dict[str, List[str]] = {name: [] for name in field_names}
    for module in sorted(root.rglob("*.py")):
        if "__pycache__" in module.parts or module.name == "configuration_jevx.py":
            continue
        tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
        label_base = str(module.relative_to(root.parent))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in hits:
                hits[node.attr].append(f"{label_base}:{node.lineno}")
            elif isinstance(node, ast.Name) and node.id in hits:
                hits[node.id].append(f"{label_base}:{node.lineno}")
            elif isinstance(node, ast.keyword) and node.arg in hits:
                hits[node.arg].append(f"{label_base}:{node.value.lineno}")
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {"getattr", "hasattr"}
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value in hits
            ):
                hits[node.args[1].value].append(f"{label_base}:{node.lineno} (getattr)")
    return hits


def test_every_config_field_is_read_or_pinned_as_reserved() -> None:
    """No JevXConfig field may be declared, validated and then ignored."""
    declared = _declared_config_fields()
    assert declared, "JevXConfig declares no field of its own - the scan is broken"
    hits = _read_sites(set(declared))
    inert = sorted(name for name, sites in hits.items() if not sites)
    stale = sorted(set(RESERVED_CONFIG_FIELDS) - set(declared))
    assert not stale, (
        f"RESERVED_CONFIG_FIELDS names fields JevXConfig no longer declares: {stale}"
    )
    assert inert == sorted(RESERVED_CONFIG_FIELDS), (
        "the inert/expected-reserved sets drifted: "
        f"inert={inert} reserved={sorted(RESERVED_CONFIG_FIELDS)}. Either wire a "
        "field into a computation, add it to RESERVED_CONFIG_FIELDS with a "
        "reason, or drop it from there once it is read."
    )
    assert inert, "the scan reports nothing inert - it is not looking for anything"


def test_reserved_config_fields_are_still_serialized() -> None:
    """A reserved field is inert, not invisible: it must round-trip."""
    payload = JevXConfig().to_dict()
    for name in RESERVED_CONFIG_FIELDS:
        assert name in payload, f"{name} is reserved but not written to config.json"
        assert JevXConfig.from_dict(payload).to_dict()[name] == payload[name]
    assert all(reason.strip() for reason in RESERVED_CONFIG_FIELDS.values())
    assert all(
        reason.strip() for reason in REJECTED_NON_DEFAULT_FIELDS.values()
    ), "every rejected field needs a one-line reason"


def test_reserved_fields_reject_their_non_default_value() -> None:
    """A reserved switch must be refused, never accepted and ignored."""
    for name, reason in REJECTED_NON_DEFAULT_FIELDS.items():
        with pytest.raises(ValueError) as excinfo:
            JevXConfig(**{name: True})
        assert name in str(excinfo.value), reason
        assert getattr(JevXConfig(), name) is False, reason


# ---------------------------------------------------------------------------
# 2. conditioning_mode
# ---------------------------------------------------------------------------


def test_conditioning_mode_off_is_bitwise_the_per_call_false_call(
    tiny_bert: Any, tiny_tokenizer: Any
) -> None:
    """``mode="off"`` must be the shipped ``condition_options=False`` call."""
    model = _wrapper(tiny_bert, tiny_tokenizer, conditioning_mode="off")
    assert model.config.conditioning_enabled is False
    with torch.no_grad():
        defaulted = model(state=STATE, questions=["which one"], options=[["a", "b", "c"]])
        explicit = model(
            state=STATE,
            questions=["which one"],
            options=[["a", "b", "c"]],
            condition_options=False,
        )
    assert torch.equal(
        defaulted.decisions[0].logits, explicit.decisions[0].logits
    ), 'mode="off" is not bitwise the per-call False path'


def test_conditioning_mode_off_is_not_the_conditioned_call(
    tiny_bert: Any, tiny_tokenizer: Any
) -> None:
    """Guard the guard: ``"off"`` must not degenerate into the ON path."""
    model = _wrapper(tiny_bert, tiny_tokenizer, conditioning_mode="off")
    with torch.no_grad():
        off = model(state=STATE, questions=["which one"], options=[["a", "b", "c"]])
        on = model(
            state=STATE,
            questions=["which one"],
            options=[["a", "b", "c"]],
            condition_options=True,
        )
    assert not torch.equal(
        off.decisions[0].logits, on.decisions[0].logits
    ), "the per-call True override stopped working under mode='off' (or options do nothing)"


def test_conditioning_mode_is_read_not_just_the_derived_flag(
    tiny_bert: Any, tiny_tokenizer: Any
) -> None:
    """A runtime flip of the DERIVED flag must not override the mode.

    This is the assertion that fails if forward goes back to reading
    ``config.conditioning_enabled`` directly: the config now pins the pair, so
    only a runtime flip can tell the mode and the flag apart.
    """
    off = _wrapper(tiny_bert, tiny_tokenizer, conditioning_mode="off")
    off.config.conditioning_enabled = True  # contradicts mode="off"
    constant = _wrapper(tiny_bert, tiny_tokenizer, conditioning_mode="constant")
    constant.config.conditioning_enabled = False  # contradicts mode="constant"
    with torch.no_grad():
        off_default = off(state=STATE, questions=["which one"], options=[["a", "b", "c"]])
        off_false = off(
            state=STATE,
            questions=["which one"],
            options=[["a", "b", "c"]],
            condition_options=False,
        )
        const_default = constant(
            state=STATE, questions=["which one"], options=[["a", "b", "c"]]
        )
        const_true = constant(
            state=STATE,
            questions=["which one"],
            options=[["a", "b", "c"]],
            condition_options=True,
        )
    assert torch.equal(
        off_default.decisions[0].logits, off_false.decisions[0].logits
    ), 'mode="off" lost to a contradictory runtime conditioning_enabled=True'
    assert torch.equal(
        const_default.decisions[0].logits, const_true.decisions[0].logits
    ), 'mode="constant" lost to a contradictory runtime conditioning_enabled=False'


def _count_encoder_calls(model: JevXForDecision) -> Dict[str, List[Any]]:
    """Wrap both encoders so their invocations are observable."""
    counters: Dict[str, List[Any]] = {"question": [], "schema": []}
    question_original = model.question_encoder.encode_texts
    schema_original = model.schema_encoder.encode

    def counting_question(texts: Any, **kwargs: Any) -> Any:
        counters["question"].append(tuple(texts))
        return question_original(texts, **kwargs)

    def counting_schema(rows: Any, **kwargs: Any) -> Any:
        counters["schema"].append(len(rows))
        return schema_original(rows, **kwargs)

    model.question_encoder.encode_texts = counting_question  # type: ignore[method-assign]
    model.schema_encoder.encode = counting_schema  # type: ignore[method-assign]
    return counters


def test_conditioning_mode_constant_memoizes_and_recomputes_on_content_change(
    tiny_bert: Any, tiny_tokenizer: Any
) -> None:
    """The ``"constant"`` memo: same content reuses, changed content recomputes."""
    model = _wrapper(tiny_bert, tiny_tokenizer, conditioning_mode="constant")
    counters = _count_encoder_calls(model)
    first_call = dict(state=STATE, questions=["which one"], options=[["a", "b", "c"]])
    changed_call = dict(state=STATE, questions=["which one"], options=[["a", "b", "cat"]])

    with torch.no_grad():
        # Reference: the identical model computing per call.
        model.config.conditioning_mode = "full"
        full = model(**first_call)
        full_changed = model(**changed_call)

        # Count only the "constant" calls: the two "full" calls above are the
        # unmemoized baseline and would otherwise show up here.
        counters["question"].clear()
        counters["schema"].clear()
        model.config.conditioning_mode = "constant"
        cold = model(**first_call)  # MISS -> encodes, fills the memo
        warm = model(**first_call)  # HIT  -> no encoder call at all
        after_change = model(**changed_call)  # MISS -> encodes again

    # One question encode across three calls whose question text is identical;
    # two schema encodes - cold and after_change - and none for warm.
    assert counters["question"] == [("which one",)], (
        "the question encoder ran again on unchanged question text"
    )
    assert counters["schema"] == [1, 1], (
        "the schema encoder did not run exactly on the cold call and the "
        "content-changed call"
    )
    # Bitwise: a memoized H_Q/H_C projects to the same logits as a fresh encode.
    assert torch.equal(cold.decisions[0].logits, full.decisions[0].logits)
    assert torch.equal(warm.decisions[0].logits, full.decisions[0].logits)
    # A content change is a miss: that call recomputes the full path.
    assert torch.equal(
        after_change.decisions[0].logits, full_changed.decisions[0].logits
    )
    assert not torch.equal(
        after_change.decisions[0].logits, cold.decisions[0].logits
    ), "the memo served a stale H_C after the option texts changed"


def test_conditioning_mode_constant_memo_is_not_used_when_grad_is_needed(
    tiny_bert: Any, tiny_tokenizer: Any
) -> None:
    """A detached memo must never silently swallow the encoder gradients."""
    model = _wrapper(tiny_bert, tiny_tokenizer, conditioning_mode="constant")
    with torch.enable_grad():
        out = model(state=STATE, questions=["which one"], options=[["a", "b", "c"]])
        out.decisions[0].logits.sum().backward()
    assert model._condition_cache == {}, (
        "the memo was filled from a grad-enabled call; a second call would "
        "return detached embeddings and drop the encoder gradients"
    )
    assert model.question_encoder.hash_embedding.weight.grad is not None
    assert model.schema_encoder.text_encoder.hash_embedding.weight.grad is not None


def test_load_state_dict_clears_the_conditioning_memo(
    tiny_bert: Any, tiny_tokenizer: Any
) -> None:
    """The memo is keyed on content, so new weights must drop it."""
    model = _wrapper(tiny_bert, tiny_tokenizer, conditioning_mode="constant")
    with torch.no_grad():
        model(state=STATE, questions=["which one"], options=[["a", "b", "c"]])
    assert model._condition_cache
    other = _wrapper(tiny_bert, tiny_tokenizer, conditioning_mode="constant")
    # strict=False: JevXForDecision.state_dict() deliberately drops the
    # calibrator sidecar keys, so a round trip through it is never key-complete.
    model.load_state_dict(other.state_dict(), strict=False)
    assert model._condition_cache == {}
    model.clear_condition_cache()  # idempotent
    assert model._condition_cache == {}


# ---------------------------------------------------------------------------
# 3. schema_encoder_width / shared_question_schema_encoder
# ---------------------------------------------------------------------------


def test_schema_encoder_width_narrows_the_option_encoder_only(
    tiny_bert: Any, tiny_tokenizer: Any
) -> None:
    model = _wrapper(
        tiny_bert,
        tiny_tokenizer,
        schema_hidden_size=16,
        schema_encoder_width=4,
    )
    assert model.schema_encoder.hidden_size == 4
    assert model.schema_proj.in_dim == 4
    assert model.schema_proj.out_dim == HIDDEN
    assert model.question_encoder.hidden_size == HIDDEN, "E_Q must not be narrowed"
    with torch.no_grad():
        out = model(state=STATE, questions=["which one"], options=[["a", "b", "c"]])
    assert torch.isfinite(out.decisions[0].logits).all()


def test_schema_encoder_width_defaults_to_the_schema_hidden_size(
    tiny_bert: Any, tiny_tokenizer: Any
) -> None:
    model = _wrapper(tiny_bert, tiny_tokenizer, schema_hidden_size=16)
    assert model.schema_encoder.hidden_size == 16
    assert model.schema_proj.in_dim == 16


def test_the_default_model_keeps_two_independent_encoders(
    tiny_bert: Any, tiny_tokenizer: Any
) -> None:
    model = _wrapper(tiny_bert, tiny_tokenizer)
    assert model.schema_encoder.text_encoder is not model.question_encoder
    assert model.question_encoder.type_embedding is None
    assert model.schema_encoder.type_tag is None


def test_shared_question_schema_encoder_rejects_its_non_default_value() -> None:
    """RESERVED: a config must never accept a switch that does nothing."""
    with pytest.raises(ValueError) as excinfo:
        JevXConfig(shared_question_schema_encoder=True)
    message = str(excinfo.value)
    assert "shared_question_schema_encoder" in message
    assert "RESERVED" in message and "NOT implemented" in message
    assert "False" in message, "the message does not say what to do instead"
    with pytest.raises(TypeError, match="shared_question_schema_encoder"):
        JevXConfig(shared_question_schema_encoder="yes")


# ---------------------------------------------------------------------------
# 4. decision_type (G17)
# ---------------------------------------------------------------------------


def test_decision_type_is_a_declared_field() -> None:
    declared = set(_declared_config_fields())
    assert "decision_type" in declared, "decision_type is not a JevXConfig field"
    assert JevXConfig().decision_type is None


@pytest.mark.parametrize("head", ["choice", "score", "noul"])
def test_decision_type_accepts_registered_heads(head: str) -> None:
    assert JevXConfig(decision_type=head).decision_type == head


def test_decision_type_rejects_an_unregistered_head() -> None:
    with pytest.raises(ValueError, match="decision_type"):
        JevXConfig(decision_type="not_a_head")
    with pytest.raises(TypeError, match="decision_type"):
        JevXConfig(decision_type=3)


def test_config_round_trip_carries_every_wired_field() -> None:
    config = JevXConfig(
        decision_type="score",
        conditioning_mode="constant",
        schema_encoder_width=4,
        schema_hidden_size=16,
        question_hidden_size=16,
    )
    restored = JevXConfig.from_dict(config.to_dict())
    assert restored.decision_type == "score"
    assert restored.conditioning_mode == "constant"
    assert restored.conditioning_enabled is True
    assert restored.schema_encoder_width == 4
    assert restored.shared_question_schema_encoder is False
    assert restored.to_dict() == config.to_dict()
    assert copy.deepcopy(restored).to_dict() == config.to_dict()
    reparsed = JevXConfig.from_dict(json.loads(config.to_json_string()))
    assert reparsed.decision_type == "score"
    assert reparsed.schema_encoder_width == 4
