# Copyright (c) 2026 ysnrfd
"""Auto layer: session registration, ``AutoJevX``, adapter priority, trust_remote_code.

Importing ``jevx.auto`` performs the session-local ``AutoConfig``/
``AutoModel`` registration (audit Q2/Q3); these tests verify it in-process.
One test downloads ``hf-internal-testing/tiny-random-bert`` (network — same
id as ``tests/unit/test_adapter_real_models.py``).
"""

from __future__ import annotations

import copy
import inspect
import json

import pytest
import torch
from transformers import AutoConfig, AutoModel

import jevx.auto  # noqa: F401 - performs the registration under test
from jevx import JevX, JevXConfig, JevXForDecision
from jevx.adapters import USER_ADAPTERS, GenericAdapter, resolve_adapter
from jevx.auto import AutoJevX

HIDDEN = 32
REAL_BERT_ID = "hf-internal-testing/tiny-random-bert"


@pytest.fixture()
def saved_jevx_dir(tmp_path, tiny_bert, tiny_tokenizer):
    """A saved JevX checkpoint (config model_type == 'jevx')."""
    wrapper = JevXForDecision(
        JevXConfig(
            decision_hidden_size=HIDDEN,
            question_hidden_size=HIDDEN,
            schema_hidden_size=HIDDEN,
        ),
        backbone=copy.deepcopy(tiny_bert),
        tokenizer=tiny_tokenizer,
    )
    wrapper.eval()
    save_dir = tmp_path / "jevx-saved"
    wrapper.save_pretrained(save_dir)
    return save_dir


def test_auto_config_registration_resolves_jevx_config(saved_jevx_dir) -> None:
    cfg = AutoConfig.from_pretrained(saved_jevx_dir)
    assert isinstance(cfg, JevXConfig)
    assert cfg.model_type == "jevx"
    assert cfg.decision_hidden_size == HIDDEN


def test_auto_model_registration_resolves_wrapper(saved_jevx_dir) -> None:
    # documented extra registration in jevx.auto
    model = AutoModel.from_pretrained(saved_jevx_dir)
    assert isinstance(model, JevXForDecision)


def test_auto_jevx_from_saved_dir(saved_jevx_dir) -> None:
    model = AutoJevX.from_pretrained(saved_jevx_dir)
    assert isinstance(model, JevXForDecision)
    model.eval()
    with torch.no_grad():
        out = model.choice(
            "hello world", ["q1", "q2"], options=[["a", "b"], ["c", "d", "e"]]
        )
    assert len(out.decisions) == 2
    for dec in out.decisions:
        assert float(dec.probabilities.sum()) == pytest.approx(1.0, abs=1e-6)


def test_auto_jevx_wraps_real_base_checkpoint() -> None:
    model = AutoJevX.from_pretrained(REAL_BERT_ID)
    assert isinstance(model, JevXForDecision)
    report = model.capability_report
    assert report.jevx_conversion == "supported"
    assert report.model_type == "bert"
    assert report.representation_field == "pooler_output"
    assert sum(p.numel() for p in model.parameters()) > 0

    with torch.no_grad():
        out = model.choice(
            "hello world", ["q1", "q2"], options=[["a", "b"], ["c", "d", "e"]]
        )
    assert len(out.decisions) == 2
    for dec, k in zip(out.decisions, (2, 3)):
        assert (dec.probabilities >= 0).all()
        assert float(dec.probabilities[0, :k].sum()) == pytest.approx(1.0, abs=1e-6)
        assert 0 <= int(dec.predicted_index[0]) < k
        assert 0.0 <= float(dec.confidence[0]) <= 1.0


def test_register_adapter_priority_and_restore(tiny_bert) -> None:
    class PriorityAdapter(GenericAdapter):
        """Test-only adapter claiming every model (unique registry name)."""

        name = "test.auto_factory_priority"

        def can_handle(self, model) -> bool:
            return True

    assert PriorityAdapter not in USER_ADAPTERS
    # register_adapter has no unregister counterpart — teardown pops the class
    # from the internal USER_ADAPTERS list (same pattern as test_adapters.py).
    JevX.register_adapter(PriorityAdapter)
    try:
        assert PriorityAdapter in USER_ADAPTERS
        resolved = resolve_adapter(copy.deepcopy(tiny_bert))
        assert type(resolved) is PriorityAdapter
    finally:
        USER_ADAPTERS[:] = [c for c in USER_ADAPTERS if c is not PriorityAdapter]
    # registry restored: builtin chain wins again for the same model
    assert type(resolve_adapter(copy.deepcopy(tiny_bert))) is not PriorityAdapter


def test_trust_remote_code_defaults_false(saved_jevx_dir) -> None:
    sig = inspect.signature(JevXForDecision.from_pretrained)
    assert sig.parameters["trust_remote_code"].default is False
    # the kwarg is accepted end-to-end; a plain local id never needs remote code
    model = AutoJevX.from_pretrained(saved_jevx_dir, trust_remote_code=False)
    assert isinstance(model, JevXForDecision)
    # no auto_map was written -> there is no remote-code path to attempt
    cfg = json.loads((saved_jevx_dir / "config.json").read_text(encoding="utf-8"))
    assert "auto_map" not in cfg


def test_peek_config_rejects_config_without_model_type(tmp_path) -> None:
    """A config.json with NO model_type key must not be misclassified as JEVX.

    Regression: JevXConfig.from_pretrained parses any JSON, and the class-level
    model_type == "jevx" attribute used to make the AutoConfig-failure fallback
    answer True for e.g. prajjwal1/bert-tiny (whose config.json predates the
    model_type convention), surfacing a misleading "missing backbone_config"
    error instead of the actionable AutoConfig error.
    """
    (tmp_path / "config.json").write_text(
        json.dumps({"hidden_size": 32, "num_hidden_layers": 2}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="model_type.*key"):
        JevXForDecision._peek_config(str(tmp_path), False, {})


def test_peek_config_fallback_requires_raw_jevx_model_type(
    tmp_path, saved_jevx_dir
) -> None:
    """With AutoConfig refusing the path, only RAW model_type=jevx classifies."""
    import jevx.modeling_jevx as modeling

    def _refuse(*_args, **_kwargs):
        raise ValueError("Unrecognized model type")

    real = modeling.AutoConfig.from_pretrained
    modeling.AutoConfig.from_pretrained = _refuse
    try:
        cfg, is_jevx = JevXForDecision._peek_config(str(saved_jevx_dir), False, {})
        assert is_jevx is True and isinstance(cfg, JevXConfig)
        foreign = tmp_path / "foreign"
        foreign.mkdir()
        (foreign / "config.json").write_text(
            json.dumps({"model_type": "gpt2", "n_embd": 16}), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="Unrecognized model"):
            JevXForDecision._peek_config(str(foreign), False, {})
    finally:
        modeling.AutoConfig.from_pretrained = real
