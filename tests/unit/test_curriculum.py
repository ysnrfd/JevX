# Copyright (c) 2026 ysnrfd
"""Unit tests for staged (curriculum) training: `jevx.training.curriculum`.

Everything here is in-memory and fast: no network, no downloads, no
GPU. The decision heads are the SHIPPED ones, built through
`build_decision_head`, and the ramp, the stage semantics, the determinism and
the guardrails are asserted against what the curriculum actually did (the
values handed to `set_interaction_scale`, the gradients left on the option
pathway, the history file on disk) rather than against a re-implementation of
the schedule.

The pinned API this file depends on (landed by the head-side work):

* `JevXConfig.state_pathway / state_hidden_size / state_activation /
  conditioning_scale_init / conditioning_enabled`;
* `DecisionHead.set_interaction_scale(value)` and the
  `get_interaction_scale` property, backed by an `interaction_scale` buffer
  that rides the checkpoint;
* `JevXForDecision.forward(..., condition_options=...)` where False means
  state-only for that call;
* `build_decision_head(name, **kwargs)` passing those through.
"""

from __future__ import annotations

import copy
import math

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from jevx.decisions import build_decision_head
from jevx.training import (
    CurriculumConfig,
    CurriculumHistory,
    DecisionDataset,
    EpochRecord,
    decision_collate_fn,
    interaction_scale,
    read_interaction_scale,
    run_curriculum,
)

HIDDEN = 8
NUM_OPTIONS = 3
QUESTION = ["which class?"]

#: Head parameters that belong to the option-interaction residual, for the
#: default (bilinear) interaction the fixtures build.
_INTERACTION_PARAMETER_NAMES = frozenset({"bilinear_weight"})


# ---------------------------------------------------------------------------
# A tiny local model standing in for JevXForDecision
# ---------------------------------------------------------------------------


class TinyCurriculumChoice(nn.Module):
    """Float-state decision model around a real `ChoiceHead`.

    It speaks the same call convention the curriculum uses
    (`state`/`questions`/`decision_type`/`options`/`condition_options`) and
    keeps two real pathways, so the stage semantics are testable without a
    backbone:

    * the STATE pathway is the fused state (a fixed random feature map of the
      raw state vector) scored by the head's own additive state pathway;
    * the OPTION pathway is a learned embedding of the schema strings, which
      the head's interaction residual scores against the state.

    `condition_options=False` calls the head with the residual switched off
    (the pinned `conditioning_enabled` per-call flag), so the option
    embedding is never read and its parameters keep `grad is None`.
    `condition_options=None` defers to `self.conditioning_enabled`, exactly
    as the real wrapper defers to `config.conditioning_enabled`.

    Args:
        state_pathway: Passed straight to `build_decision_head` ("linear" or
            "mlp"), so the curriculum is exercised against the shipped
            pathway selector.
        conditioning_scale_init: The head's initial interaction scale.
        option_vocab: Every option string this model will ever see, mapped to
            an embedding row. Pre-registering keeps the embedding parameter
            out of the lazy-growth trap (it must exist before the optimizer
            is built).
    """

    def __init__(
        self,
        option_vocab: dict,
        *,
        state_pathway: str = "linear",
        conditioning_scale_init: float = 0.0,
        conditioning_enabled: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        features = torch.randn(4, HIDDEN, generator=generator)
        features = features / features.norm(dim=0, keepdim=True).clamp(min=1e-6)
        self.register_buffer("state_features", features)
        self.option_vocab = dict(option_vocab)
        self.option_embedding = nn.Embedding(len(self.option_vocab), HIDDEN)
        # Seed the embedding from an explicit generator: nn.Embedding would
        # otherwise draw from the GLOBAL RNG, which would make every fixture's
        # accuracy depend on the test session's history.
        with torch.no_grad():
            weights = torch.empty(
                len(self.option_vocab), HIDDEN
            ).normal_(generator=torch.Generator().manual_seed(seed + 1))
            self.option_embedding.weight.copy_(weights)
        self.conditioning_enabled = bool(conditioning_enabled)
        self.head = build_decision_head(
            "choice",
            hidden_size=HIDDEN,
            state_pathway=state_pathway,
            conditioning_scale_init=conditioning_scale_init,
        )

    def option_ids(self, options) -> torch.Tensor:
        """Map the `[B][K]` list of schemas to `[B, Kmax]` embedding rows."""
        try:
            index = [
                [self.option_vocab[text] for text in schema] for schema in options
            ]
        except KeyError as exc:
            raise KeyError(
                f"option {exc.args[0]!r} is not in this model's option_vocab; "
                f"register it at construction (known: {sorted(self.option_vocab)})."
            ) from exc
        width = max(len(schema) for schema in index)
        padded = torch.zeros(len(index), width, dtype=torch.long)
        for b, schema in enumerate(index):
            padded[b, : len(schema)] = torch.tensor(schema, dtype=torch.long)
        return self.option_embedding(padded)

    def forward(
        self,
        state,
        questions=None,
        decision_type=None,
        options=None,
        *,
        condition_options=None,
        **kwargs,
    ) -> torch.Tensor:
        """Score every option against the state; returns raw `[B, K]` logits."""
        if options is None:
            raise ValueError("TinyCurriculumChoice: a choice needs options.")
        enabled = (
            self.conditioning_enabled
            if condition_options is None
            else bool(condition_options)
        )
        # [B, D]: a single (state, question) call arrives as a rank-1 state.
        fused = torch.as_tensor(state, dtype=torch.float32).reshape(len(options), -1)
        fused = fused @ self.state_features
        option_features = self.option_ids(options)
        return self.head(
            fused,
            option_features,
            conditioning_enabled=enabled,
        ).logits


# ---------------------------------------------------------------------------
# Synthetic decision problems
# ---------------------------------------------------------------------------


def _records(features, labels, options) -> list:
    """Build DecisionDataset records from float features and class labels."""
    return [
        {
            "state": [float(value) for value in feature],
            "questions": list(QUESTION),
            "decision_type": "choice",
            "options": [list(schema)],
            "labels": [int(label)],
        }
        for feature, label, schema in zip(features, labels, options)
    ]


def _schema(index: int = 0) -> list:
    """The canonical `K`-option schema for row `index`."""
    return [f"opt{index}-{j}" for j in range(NUM_OPTIONS)]


def _separable_problem(n: int, seed: int, dim: int = 4) -> list:
    """Class-conditional Gaussian blobs, linearly separable by construction.

    The three class means are pushed `3.5` apart along distinct axes, so a
    linear probe on the fused state separates them and the curriculum's
    Stage A has something real to fit.
    """
    generator = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, NUM_OPTIONS, (n,), generator=generator)
    features = torch.randn(n, dim, generator=generator) * 0.6
    features[torch.arange(n), labels % dim] += 3.5
    options = [_schema(0) for _ in range(n)]
    return _records(features.tolist(), labels.tolist(), options)


def _xor_problem(n: int, seed: int) -> list:
    """Four-bit PARITY (iterated XOR) over +-1 features, i.e. an XOR-like
    problem that is provably NOT linearly separable.

    Both parity classes average to the zero vector, so no affine function
    (w.x + b) can be positive on all eight even-parity points and negative on
    all eight odd-parity ones: averaging the positive class forces b > 0 and
    averaging the negative class forces b < 0. A linear state pathway
    therefore cannot solve it, which is exactly what makes this the test of
    the mlp state pathway.
    """
    generator = torch.Generator().manual_seed(seed)
    features = torch.randint(0, 2, (n, 4), generator=generator).float() * 2.0 - 1.0
    labels = (features.prod(dim=1) < 0).long()
    options = [_schema(0) for _ in range(n)]
    return _records(features.tolist(), labels.tolist(), options)


def _loader(records, batch_size: int, *, shuffle: bool = True) -> DataLoader:
    """A curriculum-shaped loader (DecisionDataset + decision_collate_fn)."""
    return DataLoader(
        DecisionDataset(records),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=decision_collate_fn,
    )


def _spy_on_scale(head: nn.Module, calls: list) -> None:
    """Record every value the curriculum pushes into `set_interaction_scale`."""
    original = head.set_interaction_scale

    def spy(value):
        calls.append(float(value))
        original(value)

    head.set_interaction_scale = spy


# ---------------------------------------------------------------------------
# 1. End to end on a linearly separable problem
# ---------------------------------------------------------------------------


def test_end_to_end_stages_and_lambda_record():
    """Stage A fits the state pathway; Stage B inherits it and ramps 0 -> 1."""
    train_records = _separable_problem(120, seed=1)
    eval_records = _separable_problem(60, seed=2)
    model = TinyCurriculumChoice(option_vocab={"opt0-%d" % j: j for j in range(NUM_OPTIONS)})
    train_loader = _loader(train_records, batch_size=20)
    eval_loaders = {"cal": _loader(eval_records, batch_size=30, shuffle=False)}
    calls: list = []
    _spy_on_scale(model.head, calls)

    history = run_curriculum(
        model,
        train_loader,
        eval_loaders,
        curriculum=CurriculumConfig(
            stage_a_epochs=3, stage_b_epochs=3, lr=5e-2, seed=7
        ),
    )

    assert len(history) == 6
    assert [r.stage for r in history] == ["a", "a", "a", "b", "b", "b"]
    assert [r.epoch for r in history] == [1, 2, 3, 1, 2, 3]
    assert [r.global_epoch for r in history] == [1, 2, 3, 4, 5, 6]

    stage_a, stage_b = history.stage_a, history.stage_b

    # Stage A: the state-only evaluation is already high, and improves.
    first_acc = stage_a[0].metrics["cal_accuracy"]
    last_acc = stage_a[-1].metrics["cal_accuracy"]
    assert first_acc > 0.5, stage_a[0]
    assert last_acc >= 0.95, stage_a[-1].metrics
    assert last_acc >= first_acc

    # Stage B starts FROM that solution: no collapse on its first epoch.
    assert stage_b[0].metrics["cal_accuracy"] >= last_acc - 0.05, (
        f"stage B epoch 1 collapsed: {stage_b[0].metrics} after {last_acc}"
    )
    assert stage_b[-1].metrics["cal_accuracy"] >= last_acc - 0.05

    # The lambda record: exactly 0.0 at the Stage-A boundary, 1.0 at the end.
    assert [r.interaction_scale for r in stage_a] == [0.0, 0.0, 0.0]
    assert stage_b[0].interaction_scale_first == 0.0
    assert stage_b[-1].interaction_scale == 1.0
    assert history[-1].interaction_scale == 1.0

    # The head agrees at each stage boundary.
    # (re-read after the run: the last applied scale is still in force)
    assert read_interaction_scale(model) == 1.0

    # Losses are finite and every record carries the per-split metrics.
    for record in history:
        assert math.isfinite(record.train_loss)
        assert record.steps == 6
        assert set(record.metrics) == {"cal_loss", "cal_accuracy", "cal_macro_f1"}
        assert math.isfinite(record.metrics["cal_loss"])

    # Stage A never opens the interaction: every Stage-A call was 0.0.
    assert calls[: len(calls) // 2] == [0.0] * (len(calls) // 2)


def test_lambda_record_matches_the_applied_calls():
    """Each epoch's recorded scale is the LAST scale applied in that epoch."""
    train_records = _separable_problem(80, seed=3)
    model = TinyCurriculumChoice(option_vocab={"opt0-%d" % j: j for j in range(NUM_OPTIONS)})
    train_loader = _loader(train_records, batch_size=20)
    steps = len(train_loader)
    calls: list = []
    _spy_on_scale(model.head, calls)

    history = run_curriculum(
        model,
        train_loader,
        {},
        curriculum=CurriculumConfig(stage_a_epochs=2, stage_b_epochs=4, lr=1e-2, seed=5),
    )
    assert len(calls) == (2 + 4) * steps
    for index, record in enumerate(history):
        epoch_calls = calls[index * steps : (index + 1) * steps]
        assert record.interaction_scale == pytest.approx(epoch_calls[-1], abs=0.0)
        assert record.interaction_scale_first == pytest.approx(epoch_calls[0], abs=0.0)
        assert record.steps == steps


# ---------------------------------------------------------------------------
# 2. The lambda ramp: exact values, determinism, schedule shapes
# ---------------------------------------------------------------------------


def _expected_ramp(total: int, start: float, end: float) -> list:
    """The documented linear formula, written out by hand in the test."""
    if total == 1:
        return [start]
    return [start + (end - start) * (i / (total - 1)) for i in range(total)]


def test_linear_ramp_is_applied_step_by_step_exactly():
    """Every `set_interaction_scale` call matches the closed form, 0.0 -> 1.0."""
    train_records = _separable_problem(80, seed=4)
    eval_records = _separable_problem(40, seed=5)
    model = TinyCurriculumChoice(option_vocab={"opt0-%d" % j: j for j in range(NUM_OPTIONS)})
    train_loader = _loader(train_records, batch_size=20)
    steps = len(train_loader)
    stage_b_epochs = 3
    calls: list = []
    _spy_on_scale(model.head, calls)

    run_curriculum(
        model,
        train_loader,
        {"cal": _loader(eval_records, batch_size=20, shuffle=False)},
        curriculum=CurriculumConfig(
            stage_a_epochs=1, stage_b_epochs=stage_b_epochs, lr=1e-2, seed=11
        ),
    )

    total = stage_b_epochs * steps
    stage_b_calls = calls[steps:]  # the first epoch is Stage A (all 0.0)
    assert len(stage_b_calls) == total
    expected = _expected_ramp(total, 0.0, 1.0)
    assert stage_b_calls == pytest.approx(expected, abs=0.0)
    assert stage_b_calls[0] == 0.0
    assert stage_b_calls[-1] == 1.0
    assert stage_b_calls == sorted(stage_b_calls)


def test_cosine_ramp_endpoints_and_curvature():
    """The cosine schedule hits the same endpoints and bends upward."""
    train_records = _separable_problem(60, seed=6)
    model = TinyCurriculumChoice(option_vocab={"opt0-%d" % j: j for j in range(NUM_OPTIONS)})
    train_loader = _loader(train_records, batch_size=20)
    steps = len(train_loader)
    calls: list = []
    _spy_on_scale(model.head, calls)

    run_curriculum(
        model,
        train_loader,
        {},
        curriculum=CurriculumConfig(
            stage_a_epochs=1,
            stage_b_epochs=3,
            lambda_schedule="cosine",
            lr=1e-2,
            seed=13,
        ),
    )
    total = 3 * steps
    stage_b_calls = calls[steps:]
    expected = [
        0.0 + 1.0 * (0.5 - 0.5 * math.cos(math.pi * i / (total - 1)))
        for i in range(total)
    ]
    assert stage_b_calls == pytest.approx(expected, abs=0.0)
    assert stage_b_calls[0] == 0.0
    assert stage_b_calls[-1] == 1.0
    # Cosine is an S-curve: it lags the linear ramp to the midpoint, then
    # overtakes it (1 - cos is convex on the second half).
    linear = _expected_ramp(total, 0.0, 1.0)
    half = len(stage_b_calls) // 2
    assert all(c <= l + 1e-12 for c, l in zip(stage_b_calls[:half], linear[:half]))
    assert all(c >= l - 1e-12 for c, l in zip(stage_b_calls[half:], linear[half:]))


def test_custom_ramp_endpoints_are_honoured():
    train_records = _separable_problem(60, seed=7)
    model = TinyCurriculumChoice(option_vocab={"opt0-%d" % j: j for j in range(NUM_OPTIONS)})
    train_loader = _loader(train_records, batch_size=20)
    steps = len(train_loader)
    calls: list = []
    _spy_on_scale(model.head, calls)

    run_curriculum(
        model,
        train_loader,
        {},
        curriculum=CurriculumConfig(
            stage_a_epochs=1,
            stage_b_epochs=2,
            lambda_start=0.25,
            lambda_end=0.75,
            lr=1e-2,
            seed=17,
        ),
    )
    stage_b_calls = calls[steps:]
    assert stage_b_calls[0] == 0.25
    assert stage_b_calls[-1] == 0.75
    assert stage_b_calls == pytest.approx(
        _expected_ramp(2 * steps, 0.25, 0.75), abs=0.0
    )


def test_interaction_scale_function_is_pure_and_validated():
    """The exported ramp is a pure function with checked arguments."""
    assert interaction_scale(0, 5) == 0.0
    assert interaction_scale(4, 5) == 1.0
    assert interaction_scale(2, 5) == pytest.approx(0.5)
    assert interaction_scale(0, 1) == 0.0  # degenerate single step -> start
    assert interaction_scale(0, 1, lambda_start=0.3) == 0.3
    assert interaction_scale(0, 1, lambda_end=0.3) == 0.0
    # Same arguments, same answer - no hidden state.
    assert interaction_scale(3, 7, lambda_start=0.1, lambda_end=0.9) == (
        interaction_scale(3, 7, lambda_start=0.1, lambda_end=0.9)
    )
    with pytest.raises(ValueError, match="schedule"):
        interaction_scale(0, 3, schedule="quadratic")
    with pytest.raises(ValueError, match="total_steps"):
        interaction_scale(0, 0)
    with pytest.raises(ValueError, match="step_index"):
        interaction_scale(3, 3)


# ---------------------------------------------------------------------------
# 3. Stage A is genuinely state-only
# ---------------------------------------------------------------------------


def test_stage_a_ignores_the_schema_bitwise():
    """Under `condition_options=False` the schema cannot move a single bit.

    Two schemas are compared on the same state: the trained one, and a
    constant one that carries no information but keeps the columns distinct
    (a constant column would be a duplicate option, which the head is allowed
    to tie). The Stage-A logits must be bitwise equal; the same comparison at
    a non-zero interaction scale must DIFFER, so the test can tell a real
    bypass from a vacuous one.
    """
    vocab = {"opt0-%d" % j: j for j in range(NUM_OPTIONS)}
    vocab.update({"neutral-%d" % j: NUM_OPTIONS + j for j in range(NUM_OPTIONS)})
    model = TinyCurriculumChoice(option_vocab=vocab)
    # The state pathway is zero-initialised, which would make "the two
    # state-only forwards agree" vacuous (all logits would be 0). Give it a
    # deterministic, non-degenerate weight first (no RNG involved).
    with torch.no_grad():
        for parameter in model.head.state_pathway.parameters():
            parameter.copy_(
                torch.linspace(-0.5, 0.5, parameter.numel()).reshape(
                    parameter.shape
                )
            )
    state = [0.7, -0.2, 1.1, 0.4]
    trained_schema = [_schema(0)]
    neutral_schema = [["neutral-%d" % j for j in range(NUM_OPTIONS)]]

    with torch.no_grad():
        model.head.set_interaction_scale(0.0)
        state_only_a = model(
            state, QUESTION, ["choice"], trained_schema, condition_options=False
        )
        state_only_b = model(
            state, QUESTION, ["choice"], neutral_schema, condition_options=False
        )
        # Same two schemas with the interaction open at full scale.
        model.head.set_interaction_scale(1.0)
        mixed_a = model(
            state, QUESTION, ["choice"], trained_schema, condition_options=True
        )
        mixed_b = model(
            state, QUESTION, ["choice"], neutral_schema, condition_options=True
        )

    assert torch.equal(state_only_a, state_only_b), (
        "condition_options=False still let the schema change the logits: "
        f"{state_only_a.tolist()} vs {state_only_b.tolist()}"
    )
    assert not torch.equal(mixed_a, mixed_b), (
        "the interaction is not reading the schema at lambda=1, so the "
        "state-only comparison above proves nothing"
    )
    # The Stage-A rows are genuinely per-option scores, not a constant row.
    assert float(state_only_a[0, 0]) != float(state_only_a[0, 1])
    assert float(state_only_a.max()) > float(state_only_a.min())


def test_stage_a_is_schema_invariant_through_run_curriculum():
    """The same invariant, checked on the model as the curriculum drives it."""
    vocab = {"opt0-%d" % j: j for j in range(NUM_OPTIONS)}
    vocab.update({"alt-%d" % j: NUM_OPTIONS + j for j in range(NUM_OPTIONS)})
    model = TinyCurriculumChoice(option_vocab=vocab)
    train_records = _separable_problem(60, seed=8)
    doctored = [
        {**record, "options": [["alt-%d" % j for j in range(NUM_OPTIONS)]]}
        for record in train_records
    ]

    run_curriculum(
        model,
        _loader(train_records, batch_size=20),
        {"cal": _loader(doctored, batch_size=20, shuffle=False)},
        curriculum=CurriculumConfig(stage_a_epochs=2, stage_b_epochs=0, lr=1e-2, seed=19),
    )

    state = [0.3, 0.9, -1.2, 0.05]
    with torch.no_grad():
        # Stage A finished with the scale pinned at 0.0.
        assert read_interaction_scale(model) == 0.0
        a = model(
            state, QUESTION, ["choice"], [_schema(0)], condition_options=False
        )
        b = model(
            state,
            QUESTION,
            ["choice"],
            [["alt-%d" % j for j in range(NUM_OPTIONS)]],
            condition_options=False,
        )
    assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# 4. Determinism
# ---------------------------------------------------------------------------


def _run_once(seed: int, curriculum_seed: int) -> CurriculumHistory:
    train_records = _separable_problem(96, seed=21)
    eval_records = _separable_problem(48, seed=22)
    # Seed BEFORE the model. The state pathway is zero-initialised, but the
    # head's interaction residual (`bilinear_weight`) is drawn from the GLOBAL
    # RNG, and Stage B places that residual in the graph - so a model built
    # from an unseeded global stream is NOT "the same model" twice. This test
    # asserts same model + same seed -> identical history, so the fixture has
    # to build the same model. (It used not to need to, because the buggy
    # curriculum never put the residual in the graph at all.)
    torch.manual_seed(seed)
    model = TinyCurriculumChoice(option_vocab={"opt0-%d" % j: j for j in range(NUM_OPTIONS)})
    return run_curriculum(
        model,
        _loader(train_records, batch_size=24),
        {"cal": _loader(eval_records, batch_size=24, shuffle=False)},
        curriculum=CurriculumConfig(
            stage_a_epochs=2, stage_b_epochs=2, lr=1e-2, seed=curriculum_seed
        ),
    )


def test_same_seed_gives_identical_histories():
    """Same model + same seed -> bitwise identical history dicts.

    The model is built from a seeded RNG on purpose (see :func:`_run_once`):
    Stage B trains the interaction residual, so an unseeded model would make
    this a test of the global RNG rather than of the curriculum's determinism.
    """
    first = _run_once(seed=0, curriculum_seed=31).to_dict()
    second = _run_once(seed=0, curriculum_seed=31).to_dict()
    assert first == second
    assert first["records"][0]["train_loss"] == second["records"][0]["train_loss"]


def test_different_seed_changes_only_the_stochastic_parts():
    """A different seed moves the loss, never the schedule."""
    baseline = _run_once(seed=0, curriculum_seed=31)
    other = _run_once(seed=0, curriculum_seed=99)
    assert [r.interaction_scale for r in baseline] == [
        r.interaction_scale for r in other
    ]
    assert [r.interaction_scale_first for r in baseline] == [
        r.interaction_scale_first for r in other
    ]
    assert [r.steps for r in baseline] == [r.steps for r in other]
    assert [r.train_loss for r in baseline] != [r.train_loss for r in other]


# ---------------------------------------------------------------------------
# 5. The no-test-leakage guard
# ---------------------------------------------------------------------------


def test_test_named_eval_split_is_refused():
    train_records = _separable_problem(40, seed=23)
    eval_records = _separable_problem(20, seed=24)
    model = TinyCurriculumChoice(option_vocab={"opt0-%d" % j: j for j in range(NUM_OPTIONS)})
    with pytest.raises(ValueError, match="no-test-leakage"):
        run_curriculum(
            model,
            _loader(train_records, batch_size=20),
            {
                "cal": _loader(eval_records, batch_size=20, shuffle=False),
                "test": _loader(eval_records, batch_size=20, shuffle=False),
            },
            curriculum=CurriculumConfig(stage_a_epochs=1, stage_b_epochs=1),
        )


def test_split_attribute_is_refused_even_under_a_cal_key():
    """A loader whose dataset declares `split='test'` is refused too."""
    train_records = _separable_problem(40, seed=25)
    eval_records = _separable_problem(20, seed=26)
    model = TinyCurriculumChoice(option_vocab={"opt0-%d" % j: j for j in range(NUM_OPTIONS)})
    dataset = DecisionDataset(eval_records)
    dataset.split = "hidden_test"
    loader = DataLoader(dataset, batch_size=20, collate_fn=decision_collate_fn)
    with pytest.raises(ValueError, match="no-test-leakage"):
        run_curriculum(
            model,
            _loader(train_records, batch_size=20),
            {"cal": loader},
            curriculum=CurriculumConfig(stage_a_epochs=1, stage_b_epochs=1),
        )


def test_validation_splits_are_accepted():
    train_records = _separable_problem(40, seed=27)
    eval_records = _separable_problem(20, seed=28)
    model = TinyCurriculumChoice(option_vocab={"opt0-%d" % j: j for j in range(NUM_OPTIONS)})
    history = run_curriculum(
        model,
        _loader(train_records, batch_size=20),
        {
            "cal": _loader(eval_records, batch_size=20, shuffle=False),
            "val": _loader(eval_records, batch_size=20, shuffle=False),
        },
        curriculum=CurriculumConfig(stage_a_epochs=1, stage_b_epochs=0),
    )
    assert set(history[0].metrics) == {
        "cal_loss",
        "cal_accuracy",
        "cal_macro_f1",
        "val_loss",
        "val_accuracy",
        "val_macro_f1",
    }


# ---------------------------------------------------------------------------
# 6. Gradient isolation
# ---------------------------------------------------------------------------


def _option_pathway_parameters(model: TinyCurriculumChoice) -> dict:
    """Parameters ONLY the option/schema pathway can reach.

    The schema embedding plus the head's interaction residual. Roles are read
    off the parameter names, so the test does not hard-code a module layout;
    the fixture uses the default bilinear residual, whose only parameter is
    ~bilinear_weight~.
    """
    parameters = {
        "option_embedding.weight": model.option_embedding.weight,
    }
    for name, parameter in model.head.named_parameters():
        if name in _INTERACTION_PARAMETER_NAMES:
            parameters[f"head.{name}"] = parameter
    return parameters


def _state_pathway_parameters(model: TinyCurriculumChoice) -> dict:
    """Parameters of the head's state pathway (everything else it owns)."""
    return {
        f"head.{name}": parameter
        for name, parameter in model.head.named_parameters()
        if name not in _INTERACTION_PARAMETER_NAMES
    }


def test_stage_a_leaves_the_option_pathway_without_gradient():
    """Stage A: option-pathway grads are None, state-pathway grads are not."""
    train_records = _separable_problem(60, seed=29)
    model = TinyCurriculumChoice(option_vocab={"opt0-%d" % j: j for j in range(NUM_OPTIONS)})
    run_curriculum(
        model,
        _loader(train_records, batch_size=20),
        {},
        curriculum=CurriculumConfig(stage_a_epochs=2, stage_b_epochs=0, lr=1e-2, seed=37),
    )
    for name, parameter in _option_pathway_parameters(model).items():
        assert parameter.grad is None, f"{name} received a gradient during Stage A"
    state_parameters = _state_pathway_parameters(model)
    assert state_parameters, "the head owns no state-pathway parameters"
    for name, parameter in state_parameters.items():
        assert parameter.grad is not None, f"{name} did not train in Stage A"
        assert float(parameter.grad.abs().sum()) > 0.0


def test_stage_b_gives_the_option_pathway_a_gradient():
    """Stage B at lambda > 0: the option pathway is in the graph."""
    train_records = _separable_problem(60, seed=30)
    model = TinyCurriculumChoice(option_vocab={"opt0-%d" % j: j for j in range(NUM_OPTIONS)})
    run_curriculum(
        model,
        _loader(train_records, batch_size=20),
        {},
        curriculum=CurriculumConfig(
            stage_a_epochs=0,
            stage_b_epochs=2,
            lambda_start=1.0,
            lambda_end=1.0,
            lr=1e-2,
            seed=41,
        ),
    )
    for name, parameter in _option_pathway_parameters(model).items():
        assert parameter.grad is not None, f"{name} got no gradient in Stage B"
        assert float(parameter.grad.abs().sum()) > 0.0, f"{name} gradient is all zero"
    # The state pathway still trains in Stage B (it is the sum, not a swap).
    for name, parameter in _state_pathway_parameters(model).items():
        assert parameter.grad is not None, f"{name} lost its gradient in Stage B"


def test_zero_lambda_keeps_the_option_pathway_out_of_the_graph():
    """lambda == 0.0 is state-only even with the option pathway switched on."""
    train_records = _separable_problem(60, seed=31)
    model = TinyCurriculumChoice(option_vocab={"opt0-%d" % j: j for j in range(NUM_OPTIONS)})
    run_curriculum(
        model,
        _loader(train_records, batch_size=20),
        {},
        curriculum=CurriculumConfig(
            stage_a_epochs=0, stage_b_epochs=2, lr=1e-2, seed=43
        ),
    )
    # lambda_end is 1.0, so the run did open the pathway at the end: the
    # guard above is what makes the Stage-A assertion meaningful.
    for name, parameter in _option_pathway_parameters(model).items():
        assert parameter.grad is not None
    model2 = TinyCurriculumChoice(option_vocab={"opt0-%d" % j: j for j in range(NUM_OPTIONS)})
    run_curriculum(
        model2,
        _loader(train_records, batch_size=20),
        {},
        curriculum=CurriculumConfig(
            stage_a_epochs=0,
            stage_b_epochs=2,
            lambda_start=0.0,
            lambda_end=0.0,
            lr=1e-2,
            seed=43,
        ),
    )
    for name, parameter in _option_pathway_parameters(model2).items():
        assert parameter.grad is None, f"{name} received a gradient at lambda=0"


# ---------------------------------------------------------------------------
# 7. The curriculum must not hide a broken head: learnability per pathway
# ---------------------------------------------------------------------------


def _accuracy_after(history: CurriculumHistory) -> float:
    return history[-1].metrics["cal_accuracy"]


def test_linear_state_pathway_fits_a_linear_problem():
    train_records = _separable_problem(120, seed=32)
    eval_records = _separable_problem(60, seed=33)
    model = TinyCurriculumChoice(
        option_vocab={"opt0-%d" % j: j for j in range(NUM_OPTIONS)},
        state_pathway="linear",
    )
    history = run_curriculum(
        model,
        _loader(train_records, batch_size=20),
        {"cal": _loader(eval_records, batch_size=20, shuffle=False)},
        curriculum=CurriculumConfig(
            stage_a_epochs=4, stage_b_epochs=2, lr=5e-2, seed=47
        ),
    )
    assert _accuracy_after(history) >= 0.95, history[-1].metrics
    assert history.stage_a[-1].metrics["cal_accuracy"] >= 0.95


def test_mlp_state_pathway_fits_a_nonlinear_problem():
    """XOR over two bits: not linearly separable, so the mlp pathway is needed."""
    train_records = _xor_problem(240, seed=34)
    eval_records = _xor_problem(120, seed=35)
    model = TinyCurriculumChoice(
        option_vocab={"opt0-%d" % j: j for j in range(NUM_OPTIONS)},
        state_pathway="mlp",
    )
    history = run_curriculum(
        model,
        _loader(train_records, batch_size=20),
        {"cal": _loader(eval_records, batch_size=20, shuffle=False)},
        curriculum=CurriculumConfig(
            stage_a_epochs=8, stage_b_epochs=2, lr=5e-2, seed=53
        ),
    )
    assert history.stage_a[-1].metrics["cal_accuracy"] >= 0.9, history.stage_a[-1]
    assert _accuracy_after(history) >= 0.9, history[-1].metrics


# ---------------------------------------------------------------------------
# 8. History persistence
# ---------------------------------------------------------------------------


def test_history_save_load_roundtrip(tmp_path):
    train_records = _separable_problem(40, seed=36)
    model = TinyCurriculumChoice(option_vocab={"opt0-%d" % j: j for j in range(NUM_OPTIONS)})
    history = run_curriculum(
        model,
        _loader(train_records, batch_size=20),
        {},
        curriculum=CurriculumConfig(
            stage_a_epochs=1, stage_b_epochs=2, lr=1e-2, seed=59
        ),
    )
    path = history.save(tmp_path / "curriculum.pt")
    assert path.exists()
    restored = CurriculumHistory.load(path)
    assert restored.to_dict() == history.to_dict()
    assert restored.curriculum is not None
    assert restored.curriculum.lambda_end == history.curriculum.lambda_end
    assert len(restored.stage_b) == 2
    with pytest.raises(FileNotFoundError, match="no curriculum history"):
        CurriculumHistory.load(tmp_path / "absent.pt")
    with pytest.raises(ValueError, match="history format"):
        CurriculumHistory.from_dict({"format": "something-else", "records": []})


# ---------------------------------------------------------------------------
# 9. The real JevXForDecision: the pinned forward flag and the checkpoint
# ---------------------------------------------------------------------------


def _wrapper(tiny_bert, tiny_tokenizer, **config_overrides):
    """A tiny real JevXForDecision (deep-copied backbone, per the suite rule)."""
    from jevx import JevXConfig, JevXForDecision

    overrides = dict(
        decision_hidden_size=32,
        question_hidden_size=32,
        schema_hidden_size=32,
    )
    overrides.update(config_overrides)
    model = JevXForDecision(
        JevXConfig(**overrides),
        backbone=copy.deepcopy(tiny_bert),
        tokenizer=tiny_tokenizer,
    )
    return model.eval()


STATES = ["hello world", "foo bar", "baz cat", "the a"]
SCHEMA = [["hello", "world", "foo"]]
OTHER_SCHEMA = [["bar", "baz", "cat"]]


def _decision_logits(model, options, *, condition_options):
    """Raw logits of the single (state, question) decision, as a tensor."""
    with torch.no_grad():
        out = model.choice(
            STATES[0],
            ["pick one"],
            options=options,
            condition_options=condition_options,
        )
    return out.decisions[0].logits.clone()


def test_real_wrapper_condition_options_false_is_schema_invariant(
    tiny_bert, tiny_tokenizer
):
    """The shipped wrapper honours `condition_options=False` per call.

    Two invariants, both on the REAL `JevXForDecision` (not a stand-in head):

    * schema independence - with conditioning off the schema cannot move a
      single bit, whatever options the question was asked with;
    * NON-degeneracy - those state-only logits still DISCRIMINATE the options,
      i.e. the K state scores are NOT averaged into one value.

    Why the second invariant needs the state pathway seeded first. The state
    pathway is ZERO-initialized (the standard zero-init residual convention), so
    on a freshly built model every state-only logit is exactly 0 for a reason
    that has nothing to do with the duplicate-option tie. What the tie would do
    is collapse a row whose K per-class state scores DIFFER into a single
    shared value, so the fixture gives the pathway a strongly per-class bias
    first: logits_k = fused_k + 100 * k, whose K columns cannot coincide for any
    bounded activation. The head then has a real per-class state score to
    destroy, which is exactly what the tie did.

    That collapse was real. With conditioning off the wrapper hands the head
    EXACT-ZERO option features (the `condition_options=False` branch of the
    forward) purely to keep the `[B, K, D]` / mask / schema-metadata contract,
    and exact zeros are "identical" for every schema, so
    `_tie_duplicate_options` fired on the placeholder: the K state-only
    logits came out IDENTICAL (measured spread 0.0), the softmax was uniform,
    and because the tie gives every logit in the group the same value, the
    gradient into every state-pathway parameter was analytically ZERO - a
    state-only call could never be trained out of the uniform distribution. The
    fix landed, so the K logits discriminate again and a regression that re-arms
    the tie on this path fails the spread assertion below.
    """
    model = _wrapper(tiny_bert, tiny_tokenizer)
    head = model.heads["choice"]
    pathway = head.state_pathway
    seeded = min(int(pathway.width), int(head.hidden_size))
    with torch.no_grad():
        for index in range(seeded):
            pathway.weight[index].zero_()
            pathway.weight[index, index] = 1.0
            pathway.bias[index] = 100.0 * index
    with torch.no_grad():
        model.heads["choice"].set_interaction_scale(0.0)
        state_only_a = _decision_logits(model, SCHEMA, condition_options=False)
        state_only_b = _decision_logits(model, OTHER_SCHEMA, condition_options=False)
        model.heads["choice"].set_interaction_scale(1.0)
        mixed_a = _decision_logits(model, SCHEMA, condition_options=True)
        mixed_b = _decision_logits(model, OTHER_SCHEMA, condition_options=True)

    assert torch.equal(state_only_a, state_only_b), (
        "condition_options=False still let the schema change the logits on the "
        f"real wrapper: {state_only_a.tolist()} vs {state_only_b.tolist()}"
    )
    assert not torch.equal(mixed_a, mixed_b)

    # Stage A's state-only decisions must discriminate the options. state_k is
    # per-class BY CONSTRUCTION (state_k = s^T W[k, :] + b[k]), so a tied row is
    # the degenerate-tie regression, never a legitimate reading.
    for name, row in (("SCHEMA", state_only_a), ("OTHER_SCHEMA", state_only_b)):
        spread = float(row.reshape(-1).max() - row.reshape(-1).min())
        assert spread > 0.0, (
            "state-only logits on the real wrapper are TIED across the K options "
            f"({name}: {row.tolist()}, spread {spread:.6g}); the state-only path is "
            "degenerate again - the duplicate-option tie must stay suppressed when "
            "the head is handed the exact-zero option placeholder (the forward "
            "passes duplicate_tie=condition)."
        )


def test_real_wrapper_config_defaults_match_the_pinned_fields(
    tiny_bert, tiny_tokenizer
):
    """The pinned config fields land with their documented defaults.

    The curriculum never reads the head's INITIAL scale - it sets it at every
    step - so this pins the config surface it reads (`state_pathway` picks the
    head's state pathway, `conditioning_enabled` is what
    `condition_options=None` defers to) and the writable scale buffer it
    drives, not the constructor's wiring of the initial value.

    DECISION-3 (WAVE-2 follow-up) changed the shipped default mode; the pinned
    `conditioning_enabled` value below is that DERIVED alias, not the field
    default (which is still `True` and is forwarded by conditioning_mode="full").
    """
    model = _wrapper(tiny_bert, tiny_tokenizer)
    assert model.config.state_pathway == "linear"
    assert model.config.state_activation == "gelu"
    assert model.config.state_hidden_size is None
    # DECISION-3 (WAVE-2 follow-up): the shipped default MODE is "off", so a
    # fresh model derives conditioning_enabled=False. The class-level field
    # default is still True and is forwarded verbatim by an explicit
    # conditioning_mode="full". Reverting the default is one line in
    # configuration_jevx.py.
    assert model.config.conditioning_mode == "off"
    assert model.config.conditioning_enabled is False
    from jevx import JevXConfig

    assert JevXConfig(conditioning_mode="full").conditioning_enabled is True
    assert model.config.conditioning_scale_init == 0.0
    # The scale is a buffer, readable and writable on the real head.
    for name in ("choice", "score", "noul"):
        head = model.heads[name]
        assert "interaction_scale" in dict(head.named_buffers())
        before = read_interaction_scale(model)
        head.set_interaction_scale(0.25)
        assert head.get_interaction_scale == pytest.approx(0.25)
        head.set_interaction_scale(before)


def test_real_wrapper_runs_the_curriculum(tiny_bert, tiny_tokenizer):
    """A real wrapper trains through both stages with the ramp recorded."""
    model = _wrapper(tiny_bert, tiny_tokenizer)
    records = [
        {
            "state": state,
            "questions": ["pick one"],
            "decision_type": "choice",
            "options": [["hello", "world", "foo"]],
            "labels": [index % 3],
        }
        for index, state in enumerate(STATES * 4)
    ]
    train_loader = _loader(records, batch_size=4)
    eval_loaders = {"cal": _loader(records, batch_size=4, shuffle=False)}
    calls: list = []
    _spy_on_scale(model.heads["choice"], calls)

    history = run_curriculum(
        model,
        train_loader,
        eval_loaders,
        curriculum=CurriculumConfig(
            stage_a_epochs=1, stage_b_epochs=2, lr=1e-3, seed=61
        ),
    )
    assert len(history) == 3
    assert history.stage_a[0].interaction_scale == 0.0
    assert history.stage_b[0].interaction_scale_first == 0.0
    assert history.stage_b[-1].interaction_scale == 1.0
    assert read_interaction_scale(model) == 1.0
    steps = len(train_loader)
    stage_b_calls = calls[steps:]
    assert stage_b_calls == pytest.approx(
        _expected_ramp(2 * steps, 0.0, 1.0), abs=0.0
    )
    for record in history:
        assert math.isfinite(record.train_loss)
        assert 0.0 <= record.metrics["cal_accuracy"] <= 1.0


def test_save_pretrained_roundtrip_preserves_interaction_scale(
    tmp_path, tiny_bert, tiny_tokenizer
):
    """The scale is a buffer, so it survives the checkpoint unchanged."""
    model = _wrapper(tiny_bert, tiny_tokenizer)
    model.heads["choice"].set_interaction_scale(0.375)
    before = _decision_logits(model, SCHEMA, condition_options=True)
    model.save_pretrained(tmp_path / "curriculum-model")
    # The scale must be a BUFFER, or save_pretrained would drop it.
    assert "interaction_scale" in dict(model.heads["choice"].named_buffers())

    from jevx import JevXForDecision

    reloaded = JevXForDecision.from_pretrained(tmp_path / "curriculum-model").eval()
    # The buffer is float32, so compare at float32 precision, not bit-exactly.
    assert read_interaction_scale(reloaded) == pytest.approx(0.375)
    after = _decision_logits(reloaded, SCHEMA, condition_options=True)
    assert torch.equal(before, after), (
        "the reloaded wrapper produces different decisions: "
        f"{before.tolist()} vs {after.tolist()}"
    )


def test_save_pretrained_keeps_the_scale_after_a_curriculum_run(
    tmp_path, tiny_bert, tiny_tokenizer
):
    """End of run -> the checkpoint carries the lambda the run finished at."""
    model = _wrapper(tiny_bert, tiny_tokenizer)
    records = [
        {
            "state": state,
            "questions": ["pick one"],
            "decision_type": "choice",
            "options": [["hello", "world", "foo"]],
            "labels": [index % 3],
        }
        for index, state in enumerate(STATES * 2)
    ]
    run_curriculum(
        model,
        _loader(records, batch_size=4),
        {},
        curriculum=CurriculumConfig(
            stage_a_epochs=1, stage_b_epochs=2, lambda_end=0.6, lr=1e-3, seed=67
        ),
    )
    assert read_interaction_scale(model) == pytest.approx(0.6)
    model.save_pretrained(tmp_path / "ramped")

    from jevx import JevXForDecision

    reloaded = JevXForDecision.from_pretrained(tmp_path / "ramped").eval()
    assert read_interaction_scale(reloaded) == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# 10. CurriculumConfig
# ---------------------------------------------------------------------------


def test_config_from_config_reads_only_its_own_fields():
    class Unrelated:
        model_type = "jevx"
        stage_a_epochs = 6
        stage_b_epochs = 2
        lr = 3e-3
        not_a_field = "ignored"

    config = CurriculumConfig.from_config(Unrelated())
    assert config.stage_a_epochs == 6
    assert config.stage_b_epochs == 2
    assert config.lr == 3e-3
    assert config.lambda_start == 0.0
    assert not hasattr(config, "not_a_field")

    assert CurriculumConfig.from_config({"seed": 3}).seed == 3
    assert CurriculumConfig.from_config({"seed": 3}, seed=4).seed == 4
    assert CurriculumConfig.from_config().lr == CurriculumConfig().lr
    assert (
        CurriculumConfig.from_config(CurriculumConfig(seed=9), stage_b_epochs=1).seed == 9
    )
    assert config.to_dict()["stage_a_epochs"] == 6
    assert CurriculumConfig.from_config(config.to_dict()) == config


def test_config_validates_its_fields():
    with pytest.raises(ValueError, match="lambda_schedule"):
        CurriculumConfig(lambda_schedule="quadratic")
    with pytest.raises(ValueError, match="stage_a_epochs"):
        CurriculumConfig(stage_a_epochs=-1)
    with pytest.raises(ValueError, match="trains nothing"):
        CurriculumConfig(stage_a_epochs=0, stage_b_epochs=0)
    with pytest.raises(ValueError, match="lr"):
        CurriculumConfig(lr=0.0)
    with pytest.raises(ValueError, match="weight_decay"):
        CurriculumConfig(weight_decay=-1.0)
    with pytest.raises(ValueError, match="grad_clip"):
        CurriculumConfig(grad_clip=0.0)
    with pytest.raises(ValueError, match="decision_type"):
        CurriculumConfig(decision_type="")
    with pytest.raises(TypeError, match="stage_b_epochs"):
        CurriculumConfig(stage_b_epochs=1.5)


def test_model_without_the_pinned_flag_is_refused_before_any_training():
    """A model that cannot take the flag fails BEFORE anything is touched.

    ~condition_options~ is part of BOTH stages: Stage A passes False on every
    step (that is what makes it state-only), so a model without the flag
    cannot run a curriculum at all.
    """
    train_records = _separable_problem(40, seed=38)

    class NoFlagModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, NUM_OPTIONS)

        def forward(self, state, questions=None, decision_type=None, options=None):
            return self.linear(torch.as_tensor(state, dtype=torch.float32))

    for stage_b in (0, 1):
        model = NoFlagModel()
        before = model.linear.weight.detach().clone()
        with pytest.raises(TypeError, match="condition_options"):
            run_curriculum(
                model,
                _loader(train_records, batch_size=20),
                {},
                curriculum=CurriculumConfig(
                    stage_a_epochs=1, stage_b_epochs=stage_b
                ),
            )
        # Refused up front: no optimizer step and no gradient happened.
        assert model.linear.weight.grad is None
        assert torch.equal(model.linear.weight, before)


def test_stage_a_alone_needs_no_interaction_scale_surface():
    """stage_b_epochs=0 runs on a model with no set_interaction_scale at all."""
    train_records = _separable_problem(40, seed=39)
    eval_records = _separable_problem(20, seed=40)
    flags: list = []

    class StateOnlyModel(nn.Module):
        """Accepts the flag, exposes no conditioning head."""

        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, NUM_OPTIONS)

        def forward(
            self,
            state,
            questions=None,
            decision_type=None,
            options=None,
            *,
            condition_options=None,
            **kwargs,
        ):
            flags.append(condition_options)
            return self.linear(torch.as_tensor(state, dtype=torch.float32))

    model = StateOnlyModel()
    history = run_curriculum(
        model,
        _loader(train_records, batch_size=20),
        {"cal": _loader(eval_records, batch_size=20, shuffle=False)},
        curriculum=CurriculumConfig(
            stage_a_epochs=8, stage_b_epochs=0, lr=1e-1, seed=71
        ),
    )
    assert len(history) == 8
    assert all(record.interaction_scale == 0.0 for record in history)
    assert history[-1].metrics["cal_accuracy"] >= 0.9
    assert set(flags) == {False}, f"Stage A must pass condition_options=False: {flags}"


def test_custom_loss_fn_replaces_the_decision_type_dispatch():
    train_records = _separable_problem(40, seed=41)
    seen: list = []

    def custom_loss(logits, labels):
        seen.append(tuple(logits.shape))
        return torch.nn.functional.cross_entropy(logits, labels)

    model = TinyCurriculumChoice(option_vocab={"opt0-%d" % j: j for j in range(NUM_OPTIONS)})
    history = run_curriculum(
        model,
        _loader(train_records, batch_size=20),
        {},
        curriculum=CurriculumConfig(stage_a_epochs=1, stage_b_epochs=1, lr=1e-2, seed=73),
        loss_fn=custom_loss,
    )
    assert seen, "the custom loss was never called"
    assert all(shape[1] == NUM_OPTIONS for shape in seen)
    assert math.isfinite(history[-1].train_loss)


def test_batch_contract_violations_are_reported():
    """A record with two questions is refused, not silently coerced."""
    model = TinyCurriculumChoice(option_vocab={"opt0-%d" % j: j for j in range(NUM_OPTIONS)})
    schema = ["opt0-0", "opt0-1", "opt0-2"]
    records = [
        {
            "state": [0.1, 0.2, 0.3, 0.4],
            "questions": ["first?", "second?"],
            "decision_type": "choice",
            "options": [schema, schema],
            "labels": [0, 1],
        }
    ]
    with pytest.raises(ValueError, match="ONE question"):
        run_curriculum(
            model,
            _loader(records, batch_size=1),
            {},
            curriculum=CurriculumConfig(stage_a_epochs=1, stage_b_epochs=0),
        )


def test_decision_type_mismatch_is_reported():
    model = TinyCurriculumChoice(option_vocab={"opt0-%d" % j: j for j in range(NUM_OPTIONS)})
    records = [
        {
            "state": [0.1, 0.2, 0.3, 0.4],
            "questions": ["which class?"],
            "decision_type": "score",
            "options": [["opt0-0", "opt0-1", "opt0-2"]],
            "labels": [0],
        }
    ]
    with pytest.raises(ValueError, match="decision_type"):
        run_curriculum(
            model,
            _loader(records, batch_size=1),
            {},
            curriculum=CurriculumConfig(
                stage_a_epochs=1, stage_b_epochs=0, decision_type="choice"
            ),
        )

# ---------------------------------------------------------------------------
# 11. Stage B must RE-ENABLE the option pathway (the Stage-B flag regression)
# ---------------------------------------------------------------------------
#
# The defect this section pins. `_CurriculumModelView.condition_options` used
# to be assigned exactly ONCE, at construction, from `stage_a_epochs < 1`, and
# never updated when the stage flipped to B. Every Stage-B training step
# therefore still called the model with `condition_options=False`, and that is
# what gates the residual: `DecisionHead._residual_is_active(False)` is False
# whatever the scale buffer holds, so the lambda ramp was recorded and applied
# to a buffer the forward never read. `bilinear_weight` and the whole
# `schema_encoder` kept `grad is None` for the whole run. A "staged"
# curriculum that never opens the second stage is a single-stage run with
# bookkeeping, and a curriculum that cannot be shown to open its second stage
# is a claim, not a mechanism.
#
# Three independent kinds of evidence are used, because any one of them alone
# has an innocent reading:
#
#   1. the flag the model was ACTUALLY invoked with, per call, traced on the
#      real `JevXForDecision` with a forward pre-hook;
#   2. gradient accumulations on the option-pathway parameters, traced with
#      `register_post_accumulate_grad_hook`, which fires only on a real
#      accumulation - so "the hook fired" means "it was in the graph";
#   3. the parameters' own values, which must differ from their init.

#: A schema of three DISTINCT options. Distinct matters: with a repeated or
#: single-option schema the option rows collapse (the duplicate-option tie
#: fires) and the interaction residual has nothing to act on, which would
#: make every assertion below vacuously true.
REGRESSION_SCHEMA = ["hello", "cat", "sat"]


def _regression_records() -> list:
    """12 one-question records over the three-distinct-option schema above."""
    return [
        {
            "state": state,
            "questions": ["pick one"],
            "decision_type": "choice",
            "options": [list(REGRESSION_SCHEMA)],
            "labels": [index % 3],
        }
        for index, state in enumerate(STATES * 3)
    ]


def _real_option_pathway_parameters(model) -> dict:
    """The option-pathway parameters of a REAL `JevXForDecision`, by role.

    The interaction residual (`bilinear_weight`) and every parameter of the
    schema pathway the option rows are encoded by (`schema_encoder`). Both are
    reachable only on a call that passes `condition_options=True`.

    Not all of the schema encoder's parameters can be reached on a given
    config: the default `method="projection"` encodes an option by hashing its
    tokens into `text_encoder.hash_embedding`, and `text_encoder.transform.*`
    belongs to the `method="backbone"` path. So the gradient assertions below
    are made per parameter where the default path puts it in the graph
    (`bilinear_weight` and `hash_embedding.weight`) and collectively over the
    schema group, rather than assuming a module layout.
    """
    parameters = {
        "heads.choice.bilinear_weight": model.heads["choice"].bilinear_weight
    }
    for name, parameter in model.schema_encoder.named_parameters():
        parameters["schema_encoder." + name] = parameter
    return parameters


#: The option-pathway parameters the DEFAULT schema path can place in the
#: graph (see `_real_option_pathway_parameters`).
REGRESSION_IN_GRAPH = (
    "heads.choice.bilinear_weight",
    "schema_encoder.text_encoder.hash_embedding.weight",
)


def _trace_calls(model, calls: list):
    """Record `(condition_options, interaction_scale)` of every model call.

    The scale is read at call time so a call can be attributed to a stage from
    the run itself: Stage A pins the scale at 0.0 on every call, so
    `scale == 0.0` means Stage A and `scale > 0` means Stage B. That is what
    lets the test check the flag per call without trusting the schedule.
    """

    def hook(_module, _args, kwargs):
        calls.append(
            (
                None
                if "condition_options" not in kwargs
                else bool(kwargs["condition_options"]),
                read_interaction_scale(model),
            )
        )

    return model.register_forward_pre_hook(hook, with_kwargs=True)


def _trace_grads(parameters: dict, seen: dict):
    """Count real gradient accumulations per parameter (hook => in the graph)."""
    norms = {name: 0.0 for name in parameters}

    def make(name):
        def hook(grad):
            seen[name] = seen.get(name, 0) + 1
            norms[name] = max(norms[name], float(grad.detach().norm()))

        return hook

    handles = [
        parameter.register_post_accumulate_grad_hook(make(name))
        for name, parameter in parameters.items()
    ]
    return handles, norms

def test_stage_b_calls_the_real_wrapper_with_condition_options_true(
    tiny_bert, tiny_tokenizer
):
    """Stage B calls the model with `condition_options=True` and trains it.

    On a real `JevXForDecision` over a three-distinct-option schema, a 1+1
    curriculum: every Stage-A call is state-only, every Stage-B call carries
    the flag, the option-pathway parameters are in the graph (their gradient
    hooks fire) and they have MOVED from their initialisation. Under the
    frozen-flag defect all four statements fail, the first one first.

    `lambda_start` is 0.5 rather than the shipped 0.0 so that the stage a
    call belongs to is readable from the scale the run itself carries: Stage A
    pins the scale at 0.0 on every call, so `scale == 0.0` is exactly Stage A
    and `scale > 0` exactly Stage B. The attribution therefore does not
    assume the fix works.
    """
    model = _wrapper(tiny_bert, tiny_tokenizer)
    records = _regression_records()
    train_loader = _loader(records, batch_size=4)
    eval_loaders = {"cal": _loader(records, batch_size=4, shuffle=False)}
    parameters = _real_option_pathway_parameters(model)
    init = {name: p.detach().clone() for name, p in parameters.items()}
    seen: dict = {}
    handles, norms = _trace_grads(parameters, seen)
    calls: list = []
    handle = _trace_calls(model, calls)
    try:
        history = run_curriculum(
            model,
            train_loader,
            eval_loaders,
            curriculum=CurriculumConfig(
                stage_a_epochs=1,
                stage_b_epochs=1,
                lambda_start=0.5,
                lambda_end=1.0,
                lr=1e-2,
                seed=83,
            ),
        )
    finally:
        handle.remove()
        for grad_handle in handles:
            grad_handle.remove()

    # The call order is Stage A's calls, then Stage B's, with exactly one
    # switch - and the run's own scale agrees with the attribution, so the
    # two are independent evidence. (The exact number of calls per stage is
    # NOT pinned: `HeadTrainer.evaluate` re-runs the model for the logits, so
    # an epoch is steps + 2 * eval_batches calls today. What is pinned is that
    # every one of them carried its stage's flag.)
    steps = len(train_loader)
    flags = [flag for flag, _scale in calls]
    scales = [scale for _flag, scale in calls]
    switches = [i for i in range(1, len(flags)) if flags[i] != flags[i - 1]]
    assert len(switches) <= 1, (
        "the stage flag went back and forth across the run: " + str(calls)
    )
    split = switches[0] if switches else (0 if flags and not flags[0] else len(flags))
    assert flags[:split] == [False] * split, (
        "Stage A must call the model state-only on EVERY call, not just the "
        f"first: {calls}"
    )
    assert flags[split:] == [True] * (len(flags) - split), (
        "Stage B never called the model with condition_options=True, so the "
        "option pathway was still out of the graph: " + str(calls)
    )
    assert split >= steps, (
        f"only {split} Stage-A calls for {steps} training steps: " + str(calls)
    )
    assert len(flags) - split >= steps, (
        f"only {len(flags) - split} Stage-B calls for {steps} training steps: "
        + str(calls)
    )
    assert all(scale == 0.0 for scale in scales[:split]), scales
    assert all(scale > 0.0 for scale in scales[split:]), scales
    assert [r.condition_options for r in history] == [False, True]

    # (2) the option pathway was in the graph, and (3) it moved.
    for name in REGRESSION_IN_GRAPH:
        parameter = parameters[name]
        assert seen.get(name, 0) > 0, (
            name + " never received a gradient in Stage B - the option "
            "pathway was never placed in the graph"
        )
        assert parameter.grad is not None, (
            name + ".grad is None after Stage B: the option pathway never "
            "received a gradient"
        )
        moved = float((parameter.detach() - init[name]).abs().max())
        assert moved > 0.0, (
            f"{name} is bit-for-bit unchanged from its initialisation "
            f"(max|delta| = {moved}); Stage B recorded a lambda ramp but did "
            "not train the pathway it scales"
        )
    assert norms["heads.choice.bilinear_weight"] > 0.0
    schema_moved = sum(
        float((parameters[n].detach() - init[n]).abs().max())
        for n in parameters
        if n.startswith("schema_encoder.")
    )
    assert schema_moved > 0.0, "the whole schema pathway is unchanged"

    # A conditioned decision and a state-only decision now disagree.
    assert not torch.equal(
        _decision_logits(model, [REGRESSION_SCHEMA], condition_options=False),
        _decision_logits(model, [REGRESSION_SCHEMA], condition_options=True),
    ), (
        "Stage B finished at lambda=1.0, yet a conditioned call and a "
        "state-only call return identical logits: the residual is inert"
    )

def test_stage_b_ramp_from_zero_still_opens_the_option_pathway(
    tiny_bert, tiny_tokenizer
):
    """The shipped `lambda_start=0.0` ramp opens the pathway from its 2nd step.

    λ == 0 is legitimately state-only (`_residual_is_active` is False), so the
    first Stage-B step is not expected to touch the option path - that is the
    ramp, not a defect. What must happen is that the REST of the ramp does.
    Before the fix it did not, at any step, because the per-call flag and not
    the scale gates the residual.
    """
    model = _wrapper(tiny_bert, tiny_tokenizer)
    parameters = _real_option_pathway_parameters(model)
    init = {name: p.detach().clone() for name, p in parameters.items()}
    seen: dict = {}
    handles, _norms = _trace_grads(parameters, seen)
    try:
        history = run_curriculum(
            model,
            _loader(_regression_records(), batch_size=4),
            {},
            curriculum=CurriculumConfig(
                stage_a_epochs=1, stage_b_epochs=2, lr=1e-2, seed=89
            ),
        )
    finally:
        for grad_handle in handles:
            grad_handle.remove()

    # 3 train steps x 2 Stage-B epochs = 6 ramp positions, so each record
    # carries the LAST scale of its epoch: 0.0 (Stage A, pinned), 2/5, 1.0.
    assert [r.interaction_scale for r in history] == pytest.approx(
        [0.0, 0.4, 1.0], abs=0.0
    )
    assert [r.interaction_scale_first for r in history] == pytest.approx(
        [0.0, 0.0, 0.6], abs=0.0
    )
    assert [r.condition_options for r in history] == [False, True, True]
    for name in REGRESSION_IN_GRAPH:
        parameter = parameters[name]
        assert seen.get(name, 0) > 0, (
            name + " received no gradient anywhere in a 0 -> 1 ramp: the "
            "residual was gated off for the whole stage"
        )
        moved = float((parameter.detach() - init[name]).abs().max())
        assert moved > 0.0, f"{name} unchanged (max|delta| = {moved})"


def test_history_records_the_stage_and_the_flag_per_epoch(
    tmp_path, tiny_bert, tiny_tokenizer
):
    """The history states, per epoch, the stage AND the flag that epoch ran under.

    Without the flag on the record a history cannot be checked for the thing
    that matters: a run that recorded a lambda ramp while every call was
    state-only looks IDENTICAL in the history to one that really opened the
    pathway. The record carries both, and survives the save/load round trip.
    """
    model = _wrapper(tiny_bert, tiny_tokenizer)
    history = run_curriculum(
        model,
        _loader(_regression_records(), batch_size=4),
        {"cal": _loader(_regression_records(), batch_size=4, shuffle=False)},
        curriculum=CurriculumConfig(
            stage_a_epochs=2, stage_b_epochs=1, lr=1e-3, seed=97
        ),
    )
    assert [r.stage for r in history] == ["a", "a", "b"]
    assert [r.condition_options for r in history] == [False, False, True]
    assert [d["condition_options"] for d in history.to_dict()["records"]] == [
        False,
        False,
        True,
    ]
    restored = CurriculumHistory.load(history.save(tmp_path / "stages.pt"))
    assert restored.to_dict() == history.to_dict()
    assert [r.condition_options for r in restored] == [False, False, True]
    # A record written before the flag was recorded has no such key; it must
    # NOT default to "the pathway was closed", which is the one reading a
    # Stage-B record must never take.
    legacy = EpochRecord.from_dict(
        {
            "stage": "b",
            "epoch": 1,
            "global_epoch": 3,
            "train_loss": 0.5,
            "interaction_scale": 1.0,
            "interaction_scale_first": 0.0,
            "steps": 4,
            "metrics": {},
        }
    )
    assert legacy.condition_options is True


def test_stage_b_trains_the_option_pathway_of_a_1_plus_1_curriculum():
    """The same regression in its cheapest form, on the local fixture.

    `option_embedding.weight` and `head.bilinear_weight` are reachable only
    through a conditioned call, so a 1+1 curriculum that leaves both at
    `grad is None` has not opened Stage B. This is the assertion the ablation
    agent had to make by hand to find the defect.
    """
    train_records = _separable_problem(60, seed=71)
    model = TinyCurriculumChoice(
        option_vocab={"opt0-%d" % j: j for j in range(NUM_OPTIONS)}
    )
    option_parameters = _option_pathway_parameters(model)
    before = {name: p.detach().clone() for name, p in option_parameters.items()}
    history = run_curriculum(
        model,
        _loader(train_records, batch_size=20),
        {},
        curriculum=CurriculumConfig(
            stage_a_epochs=1, stage_b_epochs=1, lr=1e-2, seed=79
        ),
    )
    assert [r.stage for r in history] == ["a", "b"]
    assert [r.condition_options for r in history] == [False, True]
    for name, parameter in option_parameters.items():
        assert parameter.grad is not None, (
            f"{name} got no gradient in a 1+1 curriculum: Stage B did not "
            "re-enable the option pathway"
        )
        moved = float((parameter.detach() - before[name]).abs().max())
        assert moved > 0.0, f"{name} unchanged from init (max|delta| = {moved})"
    # The state pathway is still trained: Stage B is the sum, not a swap.
    for name, parameter in _state_pathway_parameters(model).items():
        assert parameter.grad is not None, f"{name} lost its gradient"
