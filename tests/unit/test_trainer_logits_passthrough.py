# Copyright (c) 2026 ysnrfd
"""Evaluation reuses the forward the step function already paid for.

`HeadTrainer.evaluate` needs logits for its metrics but its step function
returns only a scalar loss, so it re-ran the model for every batch - two
forwards per eval batch. The step-function contract now has an OPTIONAL
`logits` key: `make_supervised_step_fn(..., return_logits=True)` hands back the
exact tensor the loss was computed from, `evaluate` reads it, and the second
forward disappears.

The contract is additive, so what this file pins is that the *absence* of the
key is still a first-class path:

* a step function that DOES return logits -> exactly ONE model forward per eval
  batch, counted with a forward hook (not inferred from a timing);
* a step function that does NOT -> the two-pass path still runs, still produces
  the metrics, and produces the SAME numbers as the passthrough;
* the checkpoint round-trip of the history is untouched, because the key never
  reaches a history record.

Everything is in-memory and offline: no downloads, no network.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn as nn

from jevx.training.data import decision_collate_fn, make_supervised_step_fn
from jevx.training.trainer import HeadTrainer

STATES = ["alpha state", "beta state", "gamma state"]
QUESTION = "pick one"
OPTIONS = ["yes", "no"]


def _batches(count: int = 2) -> list[dict[str, Any]]:
    """`count` collated two-option choice batches."""
    out = []
    for b in range(count):
        out.append(
            decision_collate_fn(
                [
                    {
                        "state": state,
                        "questions": [QUESTION],
                        "decision_type": "choice",
                        "options": [list(OPTIONS)],
                        "labels": [(i + b) % len(OPTIONS)],
                    }
                    for i, state in enumerate(STATES)
                ]
            )
        )
    return out


class _CountingModel(nn.Module):
    """A one-parameter view model that counts every forward it is asked for."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.25))
        self.forwards = 0

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        self.forwards += 1
        base = torch.tensor(
            [[0.1, -0.2], [0.3, 0.4], [-0.5, 0.6]], dtype=torch.float32
        )
        return base[: len(batch["labels"])] * (1.0 + self.weight)


class _ForwardCounter:
    """Counts module forwards through a real hook (not a monkeypatch)."""

    def __init__(self, model: nn.Module) -> None:
        self.count = 0
        self._handle = model.register_forward_hook(self._hook)

    def _hook(self, _module: nn.Module, _args: Any, _output: Any) -> None:
        self.count += 1

    def close(self) -> None:
        self._handle.remove()

    def __enter__(self) -> _ForwardCounter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _without_logits(model: nn.Module) -> Any:
    """The pre-contract step function: the same loss, no `logits` key.

    Built by stripping the key off the shipped step function so the ONLY
    difference between the two paths is the one under test - not the loss, not
    the dispatch, not the batch.
    """
    inner = make_supervised_step_fn(model, return_logits=True)

    def step(batch: Mapping[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        loss, components = inner(batch)
        components.pop("logits", None)
        return loss, components

    return step


def _custom_step(model: nn.Module) -> Any:
    """A user step function with an unrelated loss and no `logits` key."""

    def step(batch: Mapping[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        logits = model(batch)
        loss = logits.square().sum()
        return loss, {"decision_type": "custom", "batch_size": len(batch["labels"])}

    return step


# ---------------------------------------------------------------------------
# the passthrough
# ---------------------------------------------------------------------------


def test_default_step_fn_hands_back_the_logits_it_computed() -> None:
    """`return_logits=True` adds the key; the default contract is untouched."""
    model = _CountingModel()
    batch = _batches(1)[0]
    plain_loss, plain = make_supervised_step_fn(model)(batch)
    assert "logits" not in plain
    _, rich = make_supervised_step_fn(model, return_logits=True)(batch)
    assert isinstance(rich["logits"], torch.Tensor)
    assert rich["logits"].shape == (len(STATES), len(OPTIONS))
    assert not rich["logits"].requires_grad, "the passthrough must be detached"
    # nothing else about the components changed
    assert {k: v for k, v in rich.items() if k != "logits"} == plain
    assert plain_loss.requires_grad


def test_evaluate_runs_exactly_one_forward_per_batch() -> None:
    """One forward per eval batch when the step function offers its logits."""
    model = _CountingModel()
    batches = _batches(3)
    trainer = HeadTrainer(model)
    with _ForwardCounter(model) as counter:
        metrics = trainer.evaluate(batches)
    assert counter.count == len(batches), counter.count
    assert model.forwards == len(batches)
    assert math.isfinite(metrics["val_loss"])
    assert 0.0 <= metrics["val_accuracy"] <= 1.0


def test_evaluate_still_runs_two_forwards_without_the_key() -> None:
    """A step function with no `logits` keeps the historical second forward."""
    model = _CountingModel()
    batches = _batches(3)
    trainer = HeadTrainer(model)
    with _ForwardCounter(model) as counter:
        metrics = trainer.evaluate(batches)
    assert counter.count == len(batches), counter.count

    # the same step function with the key stripped: identical numbers, twice
    # the forwards
    fresh = _CountingModel()
    fresh.load_state_dict(model.state_dict())
    trainer = HeadTrainer(fresh)
    trainer._eval_step = _without_logits(fresh)
    with _ForwardCounter(fresh) as counter:
        fallback = trainer.evaluate(batches)
    assert counter.count == 2 * len(batches), counter.count
    assert fallback == metrics, (fallback, metrics)

    # and an unrelated custom step function still trains and evaluates
    other = HeadTrainer(_CountingModel())
    other._eval_step = _custom_step(other.model)
    with _ForwardCounter(other.model) as counter:
        custom = other.evaluate(batches)
    assert counter.count == 2 * len(batches), counter.count
    assert custom["val_accuracy"] == metrics["val_accuracy"]


def test_passthrough_and_two_pass_agree_bit_for_bit() -> None:
    """The reported numbers must not move - only the forward count does."""
    batches = _batches(4)
    fast_model = _CountingModel()
    slow_model = _CountingModel()
    slow_model.load_state_dict(fast_model.state_dict())

    fast = HeadTrainer(fast_model).evaluate(
        batches, {"accuracy": lambda lg, lb: float((lg.argmax(-1) == lb).float().mean())}
    )
    slow_trainer = HeadTrainer(slow_model)
    slow_trainer._eval_step = _without_logits(slow_model)
    slow = slow_trainer.evaluate(
        batches, {"accuracy": lambda lg, lb: float((lg.argmax(-1) == lb).float().mean())}
    )
    assert set(fast) == set(slow) == {"val_loss", "val_accuracy"}
    for key in fast:
        assert fast[key] == slow[key], (key, fast[key], slow[key])


def test_fit_reuses_the_passthrough_during_its_own_evaluation() -> None:
    """`fit(eval_loader=...)` evaluates through the same one-forward path."""
    model = _CountingModel()
    batches = _batches(2)
    trainer = HeadTrainer(model)
    with _ForwardCounter(model) as counter:
        history = trainer.fit(batches, epochs=1, eval_loader=batches)
    # 2 training forwards + 2 evaluation forwards (not 2 + 4)
    assert counter.count == 4, counter.count
    record = history[0]
    assert set(record) == {"epoch", "loss", "val_loss", "val_accuracy"}


def test_history_round_trip_is_unchanged(tmp_path) -> None:
    """The extra key never reaches a record, so checkpoints round-trip as before."""
    model = _CountingModel()
    batches = _batches(2)
    trainer = HeadTrainer(model)
    history = trainer.fit(batches, epochs=2, eval_loader=batches)
    path = trainer.save_checkpoint(tmp_path / "head.pt")

    fresh = _CountingModel()
    restored = HeadTrainer(fresh).load_checkpoint(path)
    assert restored.epochs == history.epochs
    assert restored.losses == history.losses
    assert [r["val_accuracy"] for r in restored.epochs] == [
        r["val_accuracy"] for r in history.epochs
    ]
    for record in restored.epochs:
        assert "logits" not in record


def test_objective_aware_evaluation_still_sees_its_components() -> None:
    """The objective's auxiliary terms are read off the same components dict."""
    seen: list[Mapping[str, Any]] = []

    def step(batch: Mapping[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        loss = model(batch).square().sum()
        components = {
            "loss": float(loss.detach()),
            "calibration_loss": 0.5,
            "decision_type": "custom",
            "batch_size": len(batch["labels"]),
        }
        seen.append(components)
        return loss, components

    model = _CountingModel()
    trainer = HeadTrainer(model)
    trainer._eval_step = step
    metrics = trainer.evaluate(_batches(2), objective_aware=True)
    assert seen, "the step function never ran"
    assert seen[0]["calibration_loss"] == 0.5
    assert "logits" not in seen[0]
    assert math.isfinite(metrics["val_loss"])
