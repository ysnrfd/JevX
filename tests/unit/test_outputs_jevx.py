# Copyright (c) 2026 ysnrfd
"""Unit tests for DecisionOutput / JevXOutput against transformers v5
ModelOutput semantics (verified: dataclass, one required field, None fields
omitted, to_tuple() powers return_dict=False via @can_return_tuple, no
to_dict on v5 ModelOutput)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from transformers.utils import ModelOutput, can_return_tuple

from jevx.modeling_outputs import DecisionOutput, JevXOutput


def test_decision_output_requires_only_first_field():
    probs = torch.full((2, 3), 1.0 / 3)
    with pytest.raises(TypeError, match="logits"):
        DecisionOutput(probabilities=probs)
    out = DecisionOutput(logits=torch.zeros(2, 3))
    assert isinstance(out, ModelOutput)
    assert isinstance(out, DecisionOutput)


def test_decision_output_fields_and_none_exclusion():
    logits = torch.randn(4, 5)
    probs = torch.softmax(logits, dim=-1)
    idx = probs.argmax(dim=-1)
    out = DecisionOutput(
        logits=logits,
        probabilities=probs,
        predicted_index=idx,
        expected_score=torch.rand(4) * 4,
        confidence=torch.rand(4),
        abstained=torch.zeros(4, dtype=torch.bool),
    )
    assert out.logits is logits
    assert out["logits"] is logits
    assert out[0] is logits  # integer indexing goes through to_tuple()
    # None fields are excluded from dict keys and tuple mode
    assert list(out.keys()) == [
        "logits",
        "probabilities",
        "confidence",
        "predicted_index",
        "expected_score",
        "abstained",
    ]
    assert out.predicted_value is None and "predicted_value" not in out
    assert out.calibration_metadata is None and "calibration_metadata" not in out
    assert out.schema_metadata is None and "schema_metadata" not in out
    t = out.to_tuple()
    assert len(t) == 6
    assert t[0] is logits and t[1] is probs
    for value in t:
        assert torch.is_tensor(value)


def test_decision_output_dict_only_fields_hold_nontensors():
    meta = {"method": "temperature", "temperature": 1.7}
    out = DecisionOutput(
        logits=torch.zeros(2, 2),
        calibration_metadata=meta,
        schema_metadata="schema/v1: [a, b]",
    )
    # Metadata is stored as-is; tensors never live in dict-only fields.
    assert out.calibration_metadata is meta
    assert isinstance(out["calibration_metadata"], dict)
    assert not torch.is_tensor(out["calibration_metadata"])
    assert out.schema_metadata == "schema/v1: [a, b]"
    assert torch.is_tensor(out["logits"])


def test_jevx_output_optional_fields_default_none():
    decisions = [DecisionOutput(logits=torch.zeros(2, 3))]
    out = JevXOutput(decisions=decisions)
    assert out.decisions is decisions
    assert out.state_representation is None
    assert out.question_representations is None
    assert out.schema_representations is None
    assert out.base_output is None
    assert out.hidden_states is None
    assert out.attentions is None
    assert out.capability_metadata is None
    # heavy fields stay out of the dict/tuple when unset
    assert list(out.keys()) == ["decisions"]
    assert out.to_tuple() == (decisions,)


def test_jevx_output_heavy_fields_present_when_requested():
    decisions = [DecisionOutput(logits=torch.zeros(2, 3))]
    hs = (torch.randn(2, 4, 8), torch.randn(2, 4, 8))
    att = (torch.randn(2, 2, 4, 4),)
    out = JevXOutput(
        decisions=decisions,
        state_representation=torch.randn(2, 8),
        question_representations=torch.randn(2, 6, 8),
        schema_representations=torch.randn(2, 3, 8),
        hidden_states=hs,
        attentions=att,
        capability_metadata={"primitives": ["choice"]},
    )
    assert out.state_representation.shape == (2, 8)
    assert out.question_representations.shape == (2, 6, 8)
    assert out.schema_representations.shape == (2, 3, 8)
    assert out.hidden_states is hs
    assert out.attentions is att
    assert out.capability_metadata == {"primitives": ["choice"]}
    assert len(out.to_tuple()) == 7  # all non-None fields


def test_jevx_output_with_stacked_tensor_decisions():
    stacked = torch.randn(2, 5)
    out = JevXOutput(decisions=stacked)
    assert out.decisions is stacked
    assert list(out.keys()) == ["decisions"]
    assert out.to_tuple() == (stacked,)


def test_v5_tuple_mode_via_can_return_tuple():
    """return_dict=False still exists in v5 through @can_return_tuple."""

    class Dummy:
        config = SimpleNamespace(return_dict=True)

        @can_return_tuple
        def forward(self):
            return DecisionOutput(
                logits=torch.zeros(2, 3), probabilities=torch.full((2, 3), 1 / 3)
            )

    dummy = Dummy()
    as_dict = dummy.forward()
    assert isinstance(as_dict, DecisionOutput)
    as_tuple = dummy.forward(return_dict=False)
    assert isinstance(as_tuple, tuple)
    assert len(as_tuple) == 2
    assert torch.is_tensor(as_tuple[0]) and torch.is_tensor(as_tuple[1])
    # v5 ModelOutput has no to_dict — use dict(output.items()) instead
    assert not hasattr(as_dict, "to_dict")
    assert dict(as_dict.items())["logits"] is as_dict.logits


def test_decision_output_updates_on_setattr():
    out = DecisionOutput(logits=torch.zeros(1, 2))
    new = torch.ones(1, 2)
    out.probabilities = new
    assert out["probabilities"] is new
    # v5 ModelOutput.__setattr__ stores non-None values in the dict; setting
    # None only clears the attribute (construction-time None fields are the
    # ones excluded from keys).
    out.probabilities = None
    assert out.probabilities is None
