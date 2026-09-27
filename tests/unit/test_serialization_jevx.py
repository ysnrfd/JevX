# Copyright (c) 2026 ysnrfd
"""Save/load round trips for ``JevXForDecision`` (§43 serialization).

Covers: the documented self-contained checkpoint (config.json incl. nested
``backbone_config`` + model.safetensors + jevx_metadata.json + tokenizer),
the safetensors shared-tensor problem for tied causal-LM children (audit
risk #1), and the calibration sidecar (``jevx_calibration.pt``) with its
documented method-mismatch ValueError. All models are in-memory; no network.
"""

from __future__ import annotations

import copy
import json

import pytest
import torch
from safetensors import safe_open
from transformers import GPT2Config, GPT2LMHeadModel

from jevx import JevXConfig, JevXForDecision

HIDDEN = 32
STATE = "hello world"
CHOICE_KW = dict(
    questions=["q1", "q2"],
    options=[["a", "b"], ["c", "d", "e"]],
)


def build_wrapper(backbone, tokenizer=None, **config_overrides) -> JevXForDecision:
    overrides = dict(
        decision_hidden_size=HIDDEN,
        question_hidden_size=HIDDEN,
        schema_hidden_size=HIDDEN,
    )
    overrides.update(config_overrides)
    wrapper = JevXForDecision(
        JevXConfig(**overrides), backbone=copy.deepcopy(backbone), tokenizer=tokenizer
    )
    wrapper.eval()
    return wrapper


def test_roundtrip_decisions_config_metadata_tokenizer(
    tmp_path, tiny_bert, tiny_tokenizer
) -> None:
    wrapper = build_wrapper(
        tiny_bert,
        tiny_tokenizer,
        fusion_type="gated",
        confidence_method="margin",
        pooling_method="cls",
    )
    with torch.no_grad():
        before = wrapper.choice(STATE, **CHOICE_KW)
    probs_before = [d.probabilities.clone() for d in before.decisions]

    save_dir = tmp_path / "jevx-bert"
    wrapper.save_pretrained(save_dir)
    for name in ("config.json", "model.safetensors", "jevx_metadata.json"):
        assert (save_dir / name).is_file(), f"{name} missing from checkpoint"

    # config.json: schema fields + nested backbone_config preserved
    cfg = json.loads((save_dir / "config.json").read_text(encoding="utf-8"))
    assert cfg["model_type"] == "jevx"
    assert cfg["decision_hidden_size"] == HIDDEN
    assert cfg["fusion_type"] == "gated"
    assert cfg["confidence_method"] == "margin"
    assert cfg["pooling_method"] == "cls"
    assert cfg["backbone_config"]["model_type"] == "bert"

    # jevx_metadata.json: parses and carries the capability report + identity
    meta = json.loads((save_dir / "jevx_metadata.json").read_text(encoding="utf-8"))
    assert meta["format"] == "jevx-metadata-v1"
    assert meta["jevx_version"]
    assert meta["pooling_method"] == "cls"
    assert meta["fusion_type"] == "gated"
    assert meta["capability"]["model_type"] == "bert"
    assert meta["adapter"]["class"]

    reloaded = JevXForDecision.from_pretrained(save_dir)
    reloaded.eval()
    with torch.no_grad():
        after = reloaded.choice(STATE, **CHOICE_KW)
    assert len(after.decisions) == len(probs_before)
    max_abs_error = 0.0
    for expected, got in zip(probs_before, [d.probabilities for d in after.decisions]):
        max_abs_error = max(
            max_abs_error, float((expected - got).abs().max())
        )
        assert torch.allclose(expected, got, atol=1e-6, rtol=0.0)
    assert max_abs_error <= 1e-6

    # config fields survive on the reloaded object
    assert isinstance(reloaded.config, JevXConfig)
    assert reloaded.config.decision_hidden_size == HIDDEN
    assert reloaded.config.fusion_type == "gated"
    assert reloaded.config.confidence_method == "margin"
    assert reloaded.config.pooling_method == "cls"
    assert reloaded.jevx_metadata is not None
    assert reloaded.jevx_metadata["format"] == "jevx-metadata-v1"

    # tokenizer re-attached from the save dir
    assert reloaded.tokenizer is not None
    ids = reloaded.tokenizer.encode("hello world")
    assert "hello" in reloaded.tokenizer.decode(ids)


def test_tied_causal_lm_roundtrip(tmp_path) -> None:
    """Audit risk #1: tied lm_head must not trip safetensors shared-tensor logic."""
    torch.manual_seed(42)
    gcfg = GPT2Config(
        n_embd=HIDDEN,
        n_layer=2,
        n_head=4,
        vocab_size=99,
        n_positions=64,
        bos_token_id=2,
        eos_token_id=2,
    )
    lm = GPT2LMHeadModel(gcfg).eval()
    assert lm.lm_head.weight.data_ptr() == lm.transformer.wte.weight.data_ptr()

    wrapper = build_wrapper(lm)  # no tokenizer: exercised via input_ids
    ids = torch.tensor([[2, 3, 4, 5]])
    mask = torch.ones_like(ids)
    with torch.no_grad():
        before = wrapper.choice(
            None, questions=["q1"], input_ids=ids, attention_mask=mask, options=[["a", "b"]]
        )

    save_dir = tmp_path / "jevx-gpt2lm"
    wrapper.save_pretrained(save_dir)  # must NOT raise RuntimeError

    with safe_open(str(save_dir / "model.safetensors"), "pt") as f:
        keys = list(f.keys())
    assert any(k.endswith("transformer.wte.weight") for k in keys)
    assert not any(k.endswith("lm_head.weight") for k in keys)  # alias dropped

    reloaded = JevXForDecision.from_pretrained(save_dir)
    reloaded.eval()
    # full backbone state equal, including the re-tied lm_head
    sd_before, sd_after = wrapper.model.state_dict(), reloaded.model.state_dict()
    assert set(sd_before) == set(sd_after)
    max_abs_error = max(
        float((sd_before[k].float() - sd_after[k].float()).abs().max())
        for k in sd_before
    )
    assert max_abs_error == 0.0
    # tie restored on load: same storage pointer
    assert (
        reloaded.model.lm_head.weight.data_ptr()
        == reloaded.model.transformer.wte.weight.data_ptr()
    )
    with torch.no_grad():
        after = reloaded.choice(
            None, questions=["q1"], input_ids=ids, attention_mask=mask, options=[["a", "b"]]
        )
    assert torch.allclose(
        before.decisions[0].probabilities,
        after.decisions[0].probabilities,
        atol=1e-6,
        rtol=0.0,
    )


def test_calibration_sidecar_roundtrip(tmp_path, tiny_bert, tiny_tokenizer) -> None:
    torch.manual_seed(5)
    wrapper = build_wrapper(tiny_bert, tiny_tokenizer, calibration_method="temperature")
    wrapper.calibration_engine.fit(torch.randn(32, 5), torch.randint(0, 5, (32,)))
    assert wrapper.calibration_engine.is_fitted
    temperature_before = float(wrapper.calibration_engine.calibrator.temperature)

    options = [["a", "b", "c", "d", "e"]]
    with torch.no_grad():
        before = wrapper.choice(STATE, ["q1"], options=options)
    assert before.decisions[0].calibration_metadata == {
        "method": "temperature",
        "applied": True,
    }
    probs_before = before.decisions[0].probabilities.clone()

    save_dir = tmp_path / "jevx-calibrated"
    wrapper.save_pretrained(save_dir)
    assert (save_dir / "jevx_calibration.pt").is_file()

    reloaded = JevXForDecision.from_pretrained(save_dir)
    reloaded.eval()
    assert reloaded.calibration_engine.method == "temperature"
    assert reloaded.calibration_engine.is_fitted
    assert float(reloaded.calibration_engine.calibrator.temperature) == pytest.approx(
        temperature_before, rel=1e-12
    )
    with torch.no_grad():
        after = reloaded.choice(STATE, ["q1"], options=options)
    assert torch.allclose(
        probs_before, after.decisions[0].probabilities, atol=1e-6, rtol=0.0
    )


def test_calibration_method_mismatch_raises(tmp_path, tiny_bert, tiny_tokenizer) -> None:
    torch.manual_seed(6)
    wrapper = build_wrapper(tiny_bert, tiny_tokenizer, calibration_method="temperature")
    wrapper.calibration_engine.fit(torch.randn(16, 3), torch.randint(0, 3, (16,)))
    save_dir = tmp_path / "jevx-cal-mismatch"
    wrapper.save_pretrained(save_dir)

    # documented behavior: reloading with a DIFFERENT calibration_method than
    # the sidecar was written with raises ValueError naming both methods.
    cfg = JevXConfig.from_pretrained(save_dir)
    cfg.calibration_method = "vector"
    with pytest.raises(ValueError, match="disagrees with") as exc:
        JevXForDecision.from_pretrained(save_dir, config=cfg)
    msg = str(exc.value)
    assert "temperature" in msg and "vector" in msg


def test_reload_with_explicit_config(tmp_path, tiny_bert, tiny_tokenizer) -> None:
    wrapper = build_wrapper(tiny_bert, tiny_tokenizer)
    with torch.no_grad():
        before = wrapper.choice(STATE, **CHOICE_KW)
    save_dir = tmp_path / "jevx-explicit-config"
    wrapper.save_pretrained(save_dir)

    cfg = JevXConfig.from_pretrained(save_dir)
    assert isinstance(cfg, JevXConfig)
    reloaded = JevXForDecision.from_pretrained(save_dir, config=cfg)
    assert isinstance(reloaded, JevXForDecision)
    reloaded.eval()
    with torch.no_grad():
        after = reloaded.choice(STATE, **CHOICE_KW)
    for expected, got in zip(before.decisions, after.decisions):
        assert torch.allclose(
            expected.probabilities, got.probabilities, atol=1e-6, rtol=0.0
        )
