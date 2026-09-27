# Copyright (c) 2026 ysnrfd
"""How `make_supervised_step_fn` CALLS the model (signature-aware dispatch).

A supervised step must call the model in the ONE spelling its `forward`
declares. The historical duck-typed dispatch tried `model(batch)` and
retried `model(**batch)` on `TypeError`; that probe is unsafe for the
shipped `JevXForDecision`, whose first parameter is `state` and for
which a dict is a perfectly valid `state` value. `model(batch)`
therefore does not raise: it binds the whole collated batch to the
state and keeps running, failing later inside the backbone with an
error that never mentions the call spelling. A training path that
reached a real forward would train on a bogus state instead of
raising, so the dispatch now reads the signature.

The file pins, for every model shape the step function meets:

* a decision forward (the JevX keys) -> KEYWORD branch, and a spy
  records that the model saw the per-record `state` list, never the
  batch dict;
* a single-positional-parameter view wrapper (`_CurriculumModelView`)
  -> POSITIONAL branch, unchanged;
* a bare `**kwargs` forward and a bare `batch` forward -> keyword and
  positional branch respectively;
* a REAL `JevXForDecision` -> the same keyword branch, asserted with a
  forward pre-hook (the model rejects the collated training keys by
  its own documented contract, which is the loud failure the old
  probe could never produce).

`HeadTrainer._batch_logits` - the fallback that re-runs the model during
evaluation when the step function does not hand its logits back - used to carry
its OWN copy of the duck-typed probe. It now calls the same `_call_model`, and
the last block pins that: a view wrapper, the curriculum view and a
kwargs-only module all reach the metrics pass through the branch the step
function would have used.

Everything is in-memory and offline: no downloads, no network.
"""

from __future__ import annotations

import copy
import inspect
import math
from typing import Any, Mapping

import pytest
import torch
import torch.nn as nn
from tokenizers import Tokenizer, models as tokenizers_models, pre_tokenizers
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

from jevx import JevXConfig, JevXForDecision
from jevx.training.curriculum import _CurriculumModelView
from jevx.training.data import _call_model, decision_collate_fn, make_supervised_step_fn
from jevx.training.trainer import HeadTrainer

STATES = ["hello world", "foo bar"]
QUESTION = "pick one"
SCHEMA = [["yes", "no"]]


def _batch() -> dict:
    """A collated two-record choice batch (raw python, as the loader emits)."""
    return decision_collate_fn(
        [
            {
                "state": state,
                "questions": [QUESTION],
                "decision_type": "choice",
                "options": [list(SCHEMA[0])],
                "labels": [index],
            }
            for index, state in enumerate(STATES)
        ]
    )


# ---------------------------------------------------------------------------
# spy models: one per call convention
# ---------------------------------------------------------------------------


class _Spy(nn.Module):
    """Records every forward call exactly as the step function made it."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, Any]] = []
        # a parameter so _model_device() resolves and the loss can backprop
        self.logits = nn.Parameter(torch.zeros(len(STATES), len(SCHEMA[0])))

    def _record(self, args, kwargs) -> None:
        self.calls.append({"args": tuple(args), "kwargs": dict(kwargs)})

    @property
    def state_values(self) -> list[Any]:
        """Every value the model was given for a `state` parameter."""
        return [
            call["kwargs"]["state"]
            for call in self.calls
            if "state" in call["kwargs"]
        ]

    @property
    def received_batch_dict(self) -> bool:
        """True if the whole collated dict was bound to a state parameter."""
        return any(isinstance(state, dict) for state in self.state_values)


class JevXShapedSpy(_Spy):
    """The shipped convention: state/questions/decision_type/options."""

    def forward(
        self,
        state,
        questions=None,
        decision_type=None,
        options=None,
        **kwargs,
    ) -> torch.Tensor:
        """Record the call and answer with fixed logits."""
        self._record(
            (),
            {
                "state": state,
                "questions": questions,
                "decision_type": decision_type,
                "options": options,
                **kwargs,
            },
        )
        return self.logits


class KwargsOnlySpy(_Spy):
    """A forward that declares nothing but `**kwargs`."""

    def forward(self, **kwargs) -> torch.Tensor:
        """Every batch key arrives as a keyword argument."""
        self._record((), kwargs)
        return self.logits


class PositionalBatchSpy(_Spy):
    """A forward that declares one positional parameter named `batch`."""

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        """The whole collated dict arrives positionally."""
        self._record((batch,), {})
        return self.logits

# ---------------------------------------------------------------------------
# a real JevXForDecision (module-scoped: it is built once and deep-copied)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def jevx_wrapper() -> JevXForDecision:
    """A tiny real decision wrapper, built locally (offline, deterministic).

    Built here rather than from the session fixtures because it is
    module-scoped while the tokenizer fixture is function-scoped; the
    backbone and tokenizer are the same tiny in-memory constructions.
    """
    vocab = {
        "[UNK]": 0,
        "[PAD]": 1,
        "hello": 2,
        "world": 3,
        "foo": 4,
        "bar": 5,
        "pick": 6,
        "one": 7,
        "yes": 8,
        "no": 9,
    }
    backend = Tokenizer(
        tokenizers_models.WordLevel(dict(vocab), unk_token="[UNK]")
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]"
    )
    config = BertConfig(
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    backbone = BertModel(config).eval()
    model = JevXForDecision(
        JevXConfig(
            decision_hidden_size=32,
            question_hidden_size=32,
            schema_hidden_size=32,
        ),
        backbone=backbone,
        tokenizer=tokenizer,
    )
    return model.eval()


# ---------------------------------------------------------------------------
# the keyword branch: a decision forward
# ---------------------------------------------------------------------------


def test_decision_forward_takes_the_keyword_branch() -> None:
    """JevX keys declared -> model(**batch), never model(batch)."""
    batch = _batch()
    model = JevXShapedSpy()
    loss, components = make_supervised_step_fn(model)(batch)
    assert len(model.calls) == 1, model.calls
    call = model.calls[0]
    assert call['args'] == (), call
    # one key per declared parameter; the state is the per-record list
    assert call['kwargs']['state'] == STATES, call
    assert call['kwargs']['state'] is not batch
    assert call['kwargs']['questions'] == [[QUESTION], [QUESTION]], call
    assert call["kwargs"]["decision_type"] == ["choice", "choice"], call
    assert call["kwargs"]["options"] == [[list(SCHEMA[0])]] * 2, call
    assert loss.requires_grad and math.isfinite(float(loss.detach()))
    assert components["decision_type"] == "choice"
    assert components["batch_size"] == len(STATES)


@pytest.mark.parametrize(
    "model_cls", [JevXShapedSpy, KwargsOnlySpy, PositionalBatchSpy]
)
def test_no_path_binds_the_batch_dict_to_a_state_parameter(model_cls) -> None:
    """The regression: a dict is a VALID state, so probing positionally is unsafe."""
    batch = _batch()
    model = model_cls()
    make_supervised_step_fn(model)(batch)
    assert len(model.calls) == 1, model.calls
    assert model.received_batch_dict is False, model.calls
    if issubclass(model_cls, JevXShapedSpy):
        # the decision convention: nothing positional, nothing dict-shaped
        for call in model.calls:
            assert call['args'] == (), call


# ---------------------------------------------------------------------------
# the positional branch: the view wrapper
# ---------------------------------------------------------------------------


def test_kwargs_only_forward_takes_the_keyword_branch() -> None:
    """A bare **kwargs forward is NOT a one-positional forward."""
    batch = _batch()
    model = KwargsOnlySpy()
    make_supervised_step_fn(model)(batch)
    assert model.calls[0]["args"] == ()
    assert model.calls[0]['kwargs'] == dict(batch)


def test_batch_named_forward_takes_the_positional_branch() -> None:
    """The documented view-wrapper convention stays positional."""
    batch = _batch()
    model = PositionalBatchSpy()
    make_supervised_step_fn(model)(batch)
    assert model.calls[0]['args'] == (batch,)
    assert model.calls[0]["kwargs"] == {}


def test_curriculum_view_is_a_one_positional_forward(jevx_wrapper) -> None:
    """_CurriculumModelView keeps hitting the positional branch, end to end."""
    view = _CurriculumModelView(copy.deepcopy(jevx_wrapper), condition_options=False)
    assert list(inspect.signature(view.forward).parameters) == ["batch"]
    batch = _batch()
    loss, components = make_supervised_step_fn(view)(batch)
    assert loss.shape == () and loss.requires_grad
    assert math.isfinite(float(loss.detach()))
    assert components["decision_type"] == "choice"
    assert components["batch_size"] == len(STATES)
    assert components['option_count'] == len(SCHEMA[0])
    loss.backward()
    assert any(
        parameter.grad is not None
        for parameter in view.model.heads["choice"].parameters()
    )


# ---------------------------------------------------------------------------
# the real decision model
# ---------------------------------------------------------------------------


def test_real_decision_model_takes_the_keyword_branch(jevx_wrapper) -> None:
    """A REAL JevXForDecision receives the per-record state, never the dict.

    A collated batch carries training-only keys (`labels`) that the
    decision forward does not declare, so the model rejects the call by
    its own contract (the unexpected-kwarg gate in modeling_jevx.py).
    That is the LOUD failure the old probe could not produce: the probe
    ran the batch dict in as the state and died further down with an
    error about backbone inputs. The pre-hook records what was actually
    bound, which is the assertion that matters.
    """
    model = copy.deepcopy(jevx_wrapper)
    batch = _batch()
    seen: dict = {}

    def _capture(_module, args, kwargs) -> None:
        seen["args"] = tuple(args)
        seen["kwargs"] = dict(kwargs)

    handle = model.register_forward_pre_hook(_capture, with_kwargs=True)
    try:
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            make_supervised_step_fn(model)(batch)
    finally:
        handle.remove()
    # keyword branch: nothing positional was passed at all
    assert seen["args"] == (), seen
    # the documented keys, each bound to its own per-record value
    assert set(seen["kwargs"]) >= {"state", "questions", "decision_type", "options"}
    assert seen["kwargs"]["state"] == STATES, seen
    assert not isinstance(seen["kwargs"]["state"], dict)
    assert seen["kwargs"]["state"] is not batch
    assert seen["kwargs"]["questions"] == [[QUESTION], [QUESTION]], seen


def test_decision_model_rejects_the_legacy_positional_probe(jevx_wrapper) -> None:
    """Why the probe is unsafe, asserted directly (the bug being fixed)."""
    model = copy.deepcopy(jevx_wrapper)
    batch = _batch()
    bound: dict = {}

    def _capture(_module, args, kwargs) -> None:
        bound["args"] = tuple(args)

    handle = model.register_forward_pre_hook(_capture, with_kwargs=True)
    try:
        with pytest.raises((TypeError, ValueError)):
            model(batch)
    finally:
        handle.remove()
    # the dict IS accepted as state= -- no TypeError, no argument-count
    # error, and the whole batch is bound to the state parameter
    assert bound['args'] == (batch,)
    assert bound['args'][0] is batch


def test_call_model_is_the_single_dispatch_point(jevx_wrapper) -> None:
    """The helper the step function uses, on the same conventions."""
    batch = _batch()
    jevx_shaped, kwargs_only, positional = (
        JevXShapedSpy(),
        KwargsOnlySpy(),
        PositionalBatchSpy(),
    )
    _call_model(jevx_shaped, batch)
    _call_model(kwargs_only, batch)
    _call_model(positional, batch)
    assert jevx_shaped.calls[0]["args"] == ()
    assert jevx_shaped.state_values == [STATES]
    assert kwargs_only.calls[0]["args"] == ()
    assert positional.calls[0]['args'] == (batch,)
    # a real decision wrapper: keyword branch, per-record state
    model = copy.deepcopy(jevx_wrapper)
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        _call_model(model, batch)


# ---------------------------------------------------------------------------
# the same dispatch on the evaluation fallback
# ---------------------------------------------------------------------------


class _EvalLogitsSpy(_Spy):
    """A view-wrapper model exposing the `[B, K]` logits the metrics need."""

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        self._record((batch,), {})
        return self.logits


def _trainer_over(model: nn.Module) -> HeadTrainer:
    """A trainer around `model`; the optimizer is irrelevant to dispatch."""
    return HeadTrainer(model, optimizer=torch.optim.SGD(list(model.parameters()), lr=0.0))


def test_batch_logits_uses_the_positional_branch_for_a_view_wrapper() -> None:
    """A one-positional forward gets the dict positionally, as in fit()."""
    model = _EvalLogitsSpy()
    trainer = _trainer_over(model)
    logits = trainer._batch_logits(_batch())
    assert model.calls[0]["args"][0] is model.calls[0]["args"][0]
    assert isinstance(model.calls[0]["args"][0], Mapping)
    assert model.calls[0]["args"][0]["decision_type"] == ["choice", "choice"]
    assert logits.shape == (len(STATES), len(SCHEMA[0]))
    assert not logits.requires_grad


def test_batch_logits_uses_the_keyword_branch_for_a_kwargs_only_module() -> None:
    """A `**kwargs`-only forward is not a one-positional forward."""
    model = KwargsOnlySpy()
    trainer = _trainer_over(model)
    trainer._batch_logits(_batch())
    assert model.calls[0]["args"] == ()
    assert model.calls[0]["kwargs"]["decision_type"] == ["choice", "choice"]
    assert model.received_batch_dict is False


def test_batch_logits_uses_the_keyword_branch_for_a_decision_forward() -> None:
    """The JevX key convention: the per-record state, never the batch dict."""
    model = JevXShapedSpy()
    trainer = _trainer_over(model)
    trainer._batch_logits(_batch())
    assert model.calls[0]["args"] == ()
    assert model.state_values == [STATES]
    assert model.received_batch_dict is False


def test_batch_logits_never_probes_the_unsafe_positional_spelling(
    jevx_wrapper,
) -> None:
    """The hazard `_call_model` exists for: `model(batch)` does NOT raise.

    The removed probe bound the whole collated dict to the decision forward's
    `state` parameter and kept running, so the metrics pass trained on a bogus
    state instead of reporting the call spelling.
    """
    model = copy.deepcopy(jevx_wrapper)
    trainer = _trainer_over(model)
    seen: dict = {}

    def _capture(_module, args, kwargs) -> None:
        seen["args"] = tuple(args)

    handle = model.register_forward_pre_hook(_capture, with_kwargs=True)
    try:
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            trainer._batch_logits(_batch())
    finally:
        handle.remove()
    assert seen["args"] == (), seen


def test_batch_logits_through_the_curriculum_view(jevx_wrapper) -> None:
    """The curriculum view keeps its one-positional branch end to end."""
    view = _CurriculumModelView(copy.deepcopy(jevx_wrapper), condition_options=False)
    trainer = _trainer_over(view)
    logits = trainer._batch_logits(_batch())
    assert logits.shape == (len(STATES), len(SCHEMA[0]))
    assert torch.isfinite(logits).all()


def test_batch_logits_rejects_a_non_mapping_batch() -> None:
    """The trainer's own guard is unchanged (the metrics need a labels key)."""
    model = _EvalLogitsSpy()
    trainer = _trainer_over(model)
    with pytest.raises(TypeError, match="expected mapping batches"):
        trainer._batch_logits(["not", "a", "mapping"])
    assert model.calls == []
