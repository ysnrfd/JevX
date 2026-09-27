# Copyright (c) 2026 ysnrfd
"""Total precedence between ``JevXConfig.conditioning_mode`` and ``conditioning_enabled`` (G0.2).

Pre-fix, ``__post_init__`` resolved the conflict in ONE direction only: the
pair ``("full", conditioning_enabled=False)`` silently became
``mode="off"`, while ``("constant", conditioning_enabled=False)`` kept two
contradictory values and dropped the caller's ``False` on the floor.

The rule now: ``conditioning_mode`` is the source of truth and
``conditioning_enabled`` is its derived alias, so all six (mode, enabled)
pairs end in AGREEMENT or in a named construction error. There is no third
outcome, and the table below is the whole contract.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, FrozenSet, List, Tuple

import pytest
import torch

from jevx.configuration_jevx import CONDITIONING_MODES, JevXConfig
from jevx.modeling_jevx import JevXForDecision

HIDDEN = 8
STATE = "hello world"

#: The full 3x2 table, pinned. ``outcome`` is "agree" (construction
#: succeeds and the two fields name the same behaviour) or "error" (a
#: ValueError naming BOTH fields); ``enabled`` is the value
#: ``conditioning_enabled`` must carry afterwards.
PAIR_TABLE: List[Tuple[str, bool, str, bool]] = [
    ("full", True, "agree", True),
    ("full", False, "error", False),
    ("constant", True, "agree", True),
    ("constant", False, "error", False),
    ("off", True, "agree", False),
    ("off", False, "agree", False),
]

#: The only (mode, enabled) states a constructed config may hold.
CONSISTENT_STATES: FrozenSet[Tuple[str, bool]] = frozenset(
    {("full", True), ("constant", True), ("off", False)}
)


def _pairs() -> List[Any]:
    return [
        pytest.param(mode, enabled, outcome, resolved, id=f"{mode}-{enabled}")
        for mode, enabled, outcome, resolved in PAIR_TABLE
    ]


def test_the_table_covers_every_pair_exactly_once() -> None:
    assert {(m, e) for m, e, _, _ in PAIR_TABLE} == {
        (mode, enabled) for mode in CONDITIONING_MODES for enabled in (True, False)
    }


@pytest.mark.parametrize("mode,enabled,outcome,resolved", _pairs())
def test_every_pair_ends_in_agreement_or_a_named_error(
    mode: str, enabled: bool, outcome: str, resolved: bool
) -> None:
    if outcome == "agree":
        config = JevXConfig(conditioning_mode=mode, conditioning_enabled=enabled)
        assert config.conditioning_mode == mode
        assert config.conditioning_enabled is resolved
        assert (config.conditioning_mode, config.conditioning_enabled) in CONSISTENT_STATES
        return
    with pytest.raises(ValueError) as excinfo:
        JevXConfig(conditioning_mode=mode, conditioning_enabled=enabled)
    message = str(excinfo.value)
    assert "conditioning_mode" in message and "conditioning_enabled" in message
    assert repr(mode) in message and "conditioning_enabled=False" in message


def test_no_pair_ever_leaves_two_contradictory_values() -> None:
    """The invariant, independent of the pinned table."""
    for mode in CONDITIONING_MODES:
        for enabled in (True, False):
            try:
                config = JevXConfig(conditioning_mode=mode, conditioning_enabled=enabled)
            except ValueError:
                continue
            assert (config.conditioning_mode, config.conditioning_enabled) in CONSISTENT_STATES, (
                f"({mode!r}, {enabled}) survived construction as "
                f"({config.conditioning_mode!r}, {config.conditioning_enabled!r})"
            )


def test_off_wins_over_an_explicitly_true_flag() -> None:
    """The mode is the source of truth, so "off" overrides ``enabled=True``."""
    config = JevXConfig(conditioning_mode="off", conditioning_enabled=True)
    assert config.conditioning_enabled is False
    assert copy.deepcopy(config).conditioning_enabled is False


def test_the_contradicting_pair_error_says_how_to_fix_it() -> None:
    with pytest.raises(ValueError) as excinfo:
        JevXConfig(conditioning_mode="full", conditioning_enabled=False)
    message = str(excinfo.value)
    assert "conditioning_enabled=True" in message, "the fix is not stated"
    assert "conditioning_mode='off'" in message, "the fix is not stated"


def test_an_unknown_mode_is_rejected_before_the_precedence_rule() -> None:
    with pytest.raises(ValueError, match="conditioning_mode"):
        JevXConfig(conditioning_mode="sometimes", conditioning_enabled=False)


def test_a_non_bool_flag_is_a_type_error() -> None:
    with pytest.raises(TypeError, match="conditioning_enabled"):
        JevXConfig(conditioning_enabled="yes")


# ---------------------------------------------------------------------------
# the runtime resolver
# ---------------------------------------------------------------------------


def _model(tiny_bert: Any, tiny_tokenizer: Any, **overrides: Any) -> JevXForDecision:
    settings: Dict[str, Any] = {
        "decision_hidden_size": HIDDEN,
        "question_hidden_size": HIDDEN,
        "schema_hidden_size": HIDDEN,
    }
    settings.update(overrides)
    return JevXForDecision(
        JevXConfig(**settings),
        backbone=copy.deepcopy(tiny_bert),
        tokenizer=tiny_tokenizer,
    ).eval()


@pytest.mark.parametrize(
    "mode,enabled,expected",
    [
        ("full", True, True),
        ("off", True, False),
        ("constant", True, True),
    ],
)
def test_resolve_condition_follows_the_mode(
    tiny_bert: Any, tiny_tokenizer: Any, mode: str, enabled: bool, expected: bool
) -> None:
    model = _model(tiny_bert, tiny_tokenizer, conditioning_mode=mode)
    assert model._resolve_condition(None) is expected
    # an explicit per-call bool always wins, for every mode
    assert model._resolve_condition(True) is True
    assert model._resolve_condition(False) is False


def test_resolve_condition_full_still_honours_a_runtime_flag_flip(
    tiny_bert: Any, tiny_tokenizer: Any
) -> None:
    """``conditioning_gate.apply_gate`` flips the flag on a live config.

    The wrapper is built with an EXPLICIT ``conditioning_mode="full"``: mode
    ``"full"`` is the one that reads ``conditioning_enabled`` on every call,
    and since DECISION-3 it is no longer the shipped default (``JevXConfig()``
    is ``"off"``), so a default-built model could not exercise this rule at all.
    Pinning the mode keeps the assertion about the gate, not about the default.
    """
    model = _model(tiny_bert, tiny_tokenizer, conditioning_mode="full")
    assert model._resolve_condition(None) is True
    model.config.conditioning_enabled = False
    assert model._resolve_condition(None) is False
    with torch.no_grad():
        gated = model(state=STATE, questions=["which one"], options=[["a", "b", "c"]])
        state_only = model(
            state=STATE,
            questions=["which one"],
            options=[["a", "b", "c"]],
            condition_options=False,
        )
    assert torch.equal(gated.decisions[0].logits, state_only.decisions[0].logits)


def test_forward_rejects_a_non_bool_condition_options(
    tiny_bert: Any, tiny_tokenizer: Any
) -> None:
    model = _model(tiny_bert, tiny_tokenizer)
    with pytest.raises(TypeError, match="condition_options"):
        model(
            state=STATE,
            questions=["which one"],
            options=[["a", "b", "c"]],
            condition_options="no",
        )
