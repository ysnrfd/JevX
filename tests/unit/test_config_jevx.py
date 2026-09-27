# Copyright (c) 2026 ysnrfd
"""Unit tests for JevXConfig: defaults, validation and JSON round trips."""

from __future__ import annotations

import json

import pytest

from jevx.configuration_jevx import CALIBRATION_METHODS, JevXConfig


def test_defaults():
    cfg = JevXConfig()
    assert cfg.model_type == "jevx"
    assert cfg.base_model_name_or_path is None
    assert cfg.base_model_type is None
    assert cfg.representation_strategy == "native"
    assert cfg.pooling_method is None
    assert cfg.decision_hidden_size == 768
    assert cfg.question_hidden_size == 768
    assert cfg.schema_hidden_size == 768
    assert cfg.fusion_type == "concat"
    assert cfg.decision_primitives == ["choice", "score", "noul"]
    assert cfg.calibration_method == "none"
    assert cfg.confidence_method == "max_probability"
    assert cfg.freeze_base_model is True
    assert cfg.version == "0.1.0"
    # inherited from PreTrainedConfig, not redeclared (property semantics)
    assert cfg.architectures is None
    assert cfg.output_hidden_states is False
    assert cfg.output_attentions is False


def test_custom_fields_and_dict_round_trip():
    cfg = JevXConfig(
        base_model_name_or_path="google-bert/bert-base-uncased",
        base_model_type="bert",
        representation_strategy="projected",
        pooling_method="cls",
        decision_hidden_size=256,
        question_hidden_size=128,
        schema_hidden_size=64,
        fusion_type="gated",
        decision_primitives=["choice", "noul"],
        calibration_method="vector",
        confidence_method="entropy",
        freeze_base_model=False,
        architectures=["JevXForDecision"],
        output_hidden_states=True,
        output_attentions=True,
    )
    d = cfg.to_dict()
    assert d["model_type"] == "jevx"
    restored = JevXConfig.from_dict(d)
    assert isinstance(restored, JevXConfig)
    assert restored.to_dict() == d
    assert restored.base_model_type == "bert"
    assert restored.base_model_name_or_path == "google-bert/bert-base-uncased"
    assert restored.representation_strategy == "projected"
    assert restored.pooling_method == "cls"
    assert restored.decision_hidden_size == 256
    assert restored.question_hidden_size == 128
    assert restored.schema_hidden_size == 64
    assert restored.fusion_type == "gated"
    assert restored.decision_primitives == ["choice", "noul"]
    assert restored.calibration_method == "vector"
    assert restored.confidence_method == "entropy"
    assert restored.freeze_base_model is False
    assert restored.architectures == ["JevXForDecision"]
    assert restored.output_hidden_states is True
    assert restored.output_attentions is True


def test_json_round_trip_full_and_diff(tmp_path):
    cfg = JevXConfig(
        base_model_type="bert",
        decision_primitives=["score"],
        calibration_method="temperature",
        confidence_method="margin",
    )
    for use_diff in (False, True):
        js = cfg.to_json_string(use_diff=use_diff)
        parsed = json.loads(js)  # must be valid JSON
        assert parsed["model_type"] == "jevx"
        restored = JevXConfig.from_dict(parsed)
        # Compare Python dicts (both sides have int keys in id2label; the raw
        # parsed JSON stringifies them, which is standard JSON behavior).
        assert restored.to_dict() == cfg.to_dict()

    path = tmp_path / "config.json"
    cfg.to_json_file(path, use_diff=True)
    loaded = JevXConfig.from_json_file(path)
    assert loaded.calibration_method == "temperature"
    assert loaded.confidence_method == "margin"
    assert loaded.decision_primitives == ["score"]


def test_composition_does_not_duplicate_base_config():
    cfg = JevXConfig(base_model_type="bert", base_model_name_or_path="some/bert")
    d = cfg.to_dict()
    # The wrapped model's full config is NOT embedded — only ids are stored.
    for base_key in (
        "vocab_size",
        "num_hidden_layers",
        "num_attention_heads",
        "intermediate_size",
        "hidden_act",
    ):
        assert base_key not in d
    assert d["base_model_type"] == "bert"


def test_calibration_method_enum_validated():
    assert set(CALIBRATION_METHODS) == {"none", "temperature", "vector", "isotonic"}
    for method in CALIBRATION_METHODS:
        assert JevXConfig(calibration_method=method).calibration_method == method
    with pytest.raises(ValueError, match="calibration_method"):
        JevXConfig(calibration_method="platt")


def test_hidden_sizes_must_be_positive():
    with pytest.raises(ValueError, match="decision_hidden_size"):
        JevXConfig(decision_hidden_size=0)
    with pytest.raises(ValueError, match="question_hidden_size"):
        JevXConfig(question_hidden_size=-3)


def test_decision_primitives_type_validated():
    with pytest.raises(TypeError, match="expected a list"):
        JevXConfig(decision_primitives="choice")
