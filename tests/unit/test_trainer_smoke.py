# Copyright (c) 2026 ysnrfd
"""End-to-end smoke tests for HeadTrainer: REAL optimization on synthetic
separable data, exact checkpoint round-trip, informative NaN failure, and
DecisionObjective lambda arithmetic."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from jevx.training import (
    DecisionObjective,
    HeadTrainer,
    accuracy,
    decision_collate_fn,
    macro_f1,
    make_supervised_step_fn,
)


# ---------------------------------------------------------------------------
# Stand-in decision model: Linear(D, K) over fixed random features
# ---------------------------------------------------------------------------


class TinyChoiceModel(nn.Module):
    """``head(x @ proj)`` where ``proj`` is a frozen random feature map
    (buffer) and ``head`` is the trainable ``Linear(D, K)`` decision head.

    Forward accepts the whole collated batch mapping (the first duck-typed
    call spelling ``model(batch)``) and returns raw logits ``[B, K]``.
    """

    def __init__(self, d: int = 12, num_classes: int = 3, seed: int = 0) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        proj = torch.randn(d, d, generator=generator)
        # Make the fixed feature map well-conditioned-ish (invertible a.s.
        # anyway; scaling keeps logits in a sane range for 50 AdamW steps).
        proj = proj / proj.norm(dim=0, keepdim=True).clamp(min=1e-6)
        self.register_buffer("proj", proj)
        self.head = nn.Linear(d, num_classes)

    def forward(self, batch: dict) -> torch.Tensor:
        x = torch.as_tensor(batch["state"], dtype=torch.float32)
        return self.head(x @ self.proj)  # [B, K] raw logits


def _synthetic_records(
    n: int, d: int = 12, num_classes: int = 3, seed: int = 0, separation: float = 4.0
) -> list[dict]:
    """Class-conditional Gaussian features — linearly separable by
    construction (means differ by ``separation`` along distinct axes)."""
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(n, d, generator=generator)
    labels = torch.randint(0, num_classes, (n,), generator=generator)
    features[torch.arange(n), labels % d] += separation
    records = []
    for x, y in zip(features.tolist(), labels.tolist()):
        records.append(
            {
                "state": x,
                "questions": ["which class?"],
                "decision_type": "choice",
                "options": [[f"c{k}"] for k in range(num_classes)],
                "labels": [int(y)],
            }
        )
    return records


def _loaders(train_records, eval_records, trainer, batch_size: int = 32):
    train_loader = DataLoader(
        train_records,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=decision_collate_fn,
        generator=trainer.generator,
    )
    eval_loader = DataLoader(
        eval_records,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=decision_collate_fn,
    )
    return train_loader, eval_loader


# ---------------------------------------------------------------------------
# Real optimization: loss drops > 50%, accuracy > 0.9
# ---------------------------------------------------------------------------


def test_real_optimization_loss_drops_and_accuracy_high():
    torch.manual_seed(42)
    train_records = _synthetic_records(240, seed=1)
    eval_records = _synthetic_records(120, seed=2)

    model = TinyChoiceModel(d=12, num_classes=3, seed=7)
    trainer = HeadTrainer(model, lr=5e-3, seed=42, grad_clip=1.0)
    train_loader, eval_loader = _loaders(train_records, eval_records, trainer)

    history = trainer.fit(train_loader, epochs=8, eval_loader=eval_loader)

    # ~56 batches x 8 epochs...: 240/32 = 8 batches x 8 epochs = 64 steps,
    # first epochs dominate the drop; require >50% end-to-end.
    assert len(history) == 8
    losses = history.losses
    assert losses[-1] < 0.5 * losses[0], (
        f"loss did not drop by >50%: first={losses[0]:.4f}, last={losses[-1]:.4f}"
    )

    metrics = trainer.evaluate(eval_loader, {"accuracy": accuracy, "macro_f1": macro_f1})
    assert metrics["val_accuracy"] > 0.9, metrics
    assert metrics["val_macro_f1"] > 0.9, metrics

    # Per-epoch records carry the keys the History contract promises.
    for i, record in enumerate(history.epochs, start=1):
        assert record["epoch"] == i
        assert math.isfinite(record["loss"])
        assert "val_loss" in record and "val_accuracy" in record


def test_fit_is_seeded_and_reproducible():
    """Same seed, same data, same loader generator → identical losses."""
    train_records = _synthetic_records(96, seed=3)

    def run() -> list[float]:
        torch.manual_seed(0)
        model = TinyChoiceModel(seed=11)
        trainer = HeadTrainer(model, lr=5e-3, seed=123)
        loader = DataLoader(
            train_records,
            batch_size=32,
            shuffle=True,
            collate_fn=decision_collate_fn,
            generator=trainer.generator,
        )
        return trainer.fit(loader, epochs=3).losses

    assert run() == run()


def test_callback_receives_epoch_records():
    train_records = _synthetic_records(64, seed=4)
    model = TinyChoiceModel(seed=5)
    trainer = HeadTrainer(model, lr=5e-3, seed=1)
    loader = DataLoader(
        train_records, batch_size=32, shuffle=False, collate_fn=decision_collate_fn
    )
    seen: list[dict] = []
    trainer.fit(loader, epochs=2, callback=seen.append)
    assert [r["epoch"] for r in seen] == [1, 2]
    assert all(math.isfinite(r["loss"]) for r in seen)


# ---------------------------------------------------------------------------
# Checkpoint save/load reproduces eval exactly
# ---------------------------------------------------------------------------


def test_checkpoint_roundtrip_reproduces_eval_exactly(tmp_path):
    torch.manual_seed(0)
    train_records = _synthetic_records(160, seed=5)
    eval_records = _synthetic_records(80, seed=6)

    model = TinyChoiceModel(seed=3)
    trainer = HeadTrainer(model, lr=5e-3, seed=9)
    train_loader, eval_loader = _loaders(train_records, eval_records, trainer)
    trainer.fit(train_loader, epochs=4)

    baseline = trainer.evaluate(eval_loader)
    path = trainer.save_checkpoint(tmp_path / "ckpt.pt")
    assert path.exists()

    # Perturb every parameter so a broken load cannot pass by luck.
    with torch.no_grad():
        for param in trainer.model.parameters():
            param.add_(torch.randn_like(param) * 2.0)
    perturbed = trainer.evaluate(eval_loader)
    assert perturbed["val_loss"] != baseline["val_loss"]

    restored_history = trainer.load_checkpoint(path)
    assert len(restored_history) == 4
    after = trainer.evaluate(eval_loader)
    assert after["val_loss"] == baseline["val_loss"]  # bitwise-exact on CPU
    assert after["val_accuracy"] == baseline["val_accuracy"]
    assert trainer.history.losses == [r["loss"] for r in restored_history.epochs]


def test_trainable_only_checkpoint_requires_base_present(tmp_path):
    """save_trainable_only stores only head params + base reference; reload
    works when the base (frozen feature map) is already present, and the
    reloaded model reproduces the original eval exactly."""
    train_records = _synthetic_records(120, seed=7)
    eval_records = _synthetic_records(60, seed=8)

    torch.manual_seed(0)
    model = TinyChoiceModel(seed=4)
    trainer = HeadTrainer(model, lr=5e-3, seed=10)
    train_loader, eval_loader = _loaders(train_records, eval_records, trainer)
    trainer.fit(train_loader, epochs=3)
    baseline = trainer.evaluate(eval_loader)

    path = trainer.save_checkpoint(
        tmp_path / "heads_only.pt",
        save_trainable_only=True,
        config=SimpleNamespace(base_model_name_or_path="local/tiny-features"),
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["save_trainable_only"] is True
    assert payload["base_model_name_or_path"] == "local/tiny-features"
    # Only requires_grad params — no 'proj' buffer key.
    assert set(payload["model_state"]) == {"head.weight", "head.bias"}

    # New model: same base (seeded proj buffer) but a DIFFERENT head init.
    torch.manual_seed(999)
    fresh = TinyChoiceModel(seed=4)
    fresh.head.reset_parameters()
    fresh_trainer = HeadTrainer(fresh, lr=5e-3, seed=10)
    before = fresh_trainer.evaluate(eval_loader)
    assert before["val_loss"] != baseline["val_loss"]  # head really differs

    fresh_trainer.load_checkpoint(path)
    after = fresh_trainer.evaluate(eval_loader)
    assert after["val_loss"] == baseline["val_loss"]
    assert after["val_accuracy"] == baseline["val_accuracy"]


def test_load_checkpoint_missing_file_raises(tmp_path):
    model = TinyChoiceModel()
    trainer = HeadTrainer(model)
    with pytest.raises(FileNotFoundError, match=r"no checkpoint"):
        trainer.load_checkpoint(tmp_path / "absent.pt")


# ---------------------------------------------------------------------------
# NaN loss path raises with epoch/step context
# ---------------------------------------------------------------------------


def test_nan_loss_raises_with_epoch_and_step_info():
    train_records = _synthetic_records(64, seed=9)
    model = TinyChoiceModel(seed=6)
    trainer = HeadTrainer(model, lr=5e-3, seed=2)
    loader = DataLoader(
        train_records, batch_size=32, shuffle=False, collate_fn=decision_collate_fn
    )

    def nan_step(batch):
        return torch.tensor(float("nan"), requires_grad=True), {}

    with pytest.raises(ValueError, match=r"epoch 1, step 1"):
        trainer.fit(loader, epochs=1, loss_fn=nan_step)


def test_nan_loss_mid_epoch_reports_correct_step():
    train_records = _synthetic_records(96, seed=10)
    model = TinyChoiceModel(seed=8)
    trainer = HeadTrainer(model, lr=5e-3, seed=2)
    loader = DataLoader(
        train_records, batch_size=32, shuffle=False, collate_fn=decision_collate_fn
    )
    calls = {"n": 0}

    def nan_on_second_batch(batch):
        calls["n"] += 1
        value = float("nan") if calls["n"] == 2 else 0.0
        return torch.tensor(value, requires_grad=True), {}

    with pytest.raises(ValueError, match=r"epoch 1, step 2"):
        trainer.fit(loader, epochs=1, loss_fn=nan_on_second_batch)


def test_default_step_propagates_model_nan():
    """A model emitting NaN logits must fail loudly through the default step."""
    train_records = _synthetic_records(32, seed=11)
    model = TinyChoiceModel(seed=6)
    with torch.no_grad():
        model.head.weight.fill_(float("nan"))
    trainer = HeadTrainer(model, lr=5e-3, seed=2)
    loader = DataLoader(
        train_records, batch_size=16, shuffle=False, collate_fn=decision_collate_fn
    )
    with pytest.raises(ValueError, match=r"epoch 1, step 1"):
        trainer.fit(loader, epochs=1)


def test_step_function_bad_return_type_raises():
    train_records = _synthetic_records(32, seed=12)
    model = TinyChoiceModel()
    trainer = HeadTrainer(model)
    loader = DataLoader(
        train_records, batch_size=16, shuffle=False, collate_fn=decision_collate_fn
    )
    with pytest.raises(TypeError, match=r"step function"):
        trainer.fit(loader, epochs=1, loss_fn=lambda batch: "not a loss")


# ---------------------------------------------------------------------------
# DecisionObjective lambda math vs hand computation
# ---------------------------------------------------------------------------


def test_objective_lambda_math_matches_hand_computation():
    obj = DecisionObjective(
        lambda_decision=2.0,
        lambda_calibration=3.0,
        lambda_representation=0.5,
        lambda_regularization=1.5,
    )
    total, components = obj.compute(
        1.0,
        calibration_loss=4.0,
        representation_loss=2.0,
        regularization_loss=0.25,
    )
    expected = 2.0 * 1.0 + 3.0 * 4.0 + 0.5 * 2.0 + 1.5 * 0.25
    assert math.isclose(total, expected, rel_tol=1e-12)
    assert math.isclose(components["decision"], 2.0, rel_tol=1e-12)
    assert math.isclose(components["calibration"], 12.0, rel_tol=1e-12)
    assert math.isclose(components["representation"], 1.0, rel_tol=1e-12)
    assert math.isclose(components["regularization"], 0.375, rel_tol=1e-12)
    assert math.isclose(components["total"], expected, rel_tol=1e-12)


def test_objective_defaults_off_auxiliary_terms():
    """Default DecisionObjective: total == decision loss, None terms are 0."""
    obj = DecisionObjective()
    total, components = obj.compute(0.7, calibration_loss=5.0)
    assert math.isclose(total, 0.7, rel_tol=1e-12)
    assert components["calibration"] == 0.0  # lambda is 0 → term OFF
    assert components["representation"] == 0.0
    assert components["regularization"] == 0.0

    # None components count as 0 even with nonzero lambdas.
    obj2 = DecisionObjective(lambda_calibration=4.0)
    total2, components2 = obj2.compute(1.0, calibration_loss=None)
    assert math.isclose(total2, 1.0, rel_tol=1e-12)
    assert components2["calibration"] == 0.0


def test_objective_tensor_terms_keep_grad():
    obj = DecisionObjective(lambda_decision=2.0, lambda_regularization=0.1)
    decision = torch.tensor(3.0, requires_grad=True)
    reg = torch.tensor(10.0, requires_grad=True)
    total, _ = obj.compute(decision, regularization_loss=reg)
    assert isinstance(total, torch.Tensor)
    assert math.isclose(float(total.detach()), 2.0 * 3.0 + 0.1 * 10.0, rel_tol=1e-6)
    total.backward()
    assert decision.grad is not None and math.isclose(
        float(decision.grad), 2.0, rel_tol=1e-6
    )
    assert reg.grad is not None and math.isclose(
        float(reg.grad), 0.1, rel_tol=1e-6
    )


def test_objective_zero_lambda_excludes_term_from_graph():
    """An OFF term must not contribute gradient even if a value is passed."""
    obj = DecisionObjective(lambda_decision=1.0, lambda_calibration=0.0)
    decision = torch.tensor(1.0, requires_grad=True)
    calibration = torch.tensor(100.0, requires_grad=True)
    total, _ = obj.compute(decision, calibration_loss=calibration)
    assert float(total.detach()) == 1.0
    total.backward()
    assert calibration.grad is None  # not in the graph at all
    assert decision.grad is not None


def test_objective_negative_lambda_raises():
    with pytest.raises(ValueError, match=r"lambda_decision"):
        DecisionObjective(lambda_decision=-1.0)
    with pytest.raises(ValueError, match=r"lambda_calibration"):
        DecisionObjective(lambda_calibration=-0.001)


def test_objective_compute_requires_decision_loss():
    with pytest.raises(TypeError, match=r"decision_loss"):
        DecisionObjective().compute(None)


def test_trainer_default_objective_is_plain_decision_loss():
    """HeadTrainer's default objective must not alter the step loss."""
    train_records = _synthetic_records(32, seed=13)
    model = TinyChoiceModel(seed=9)
    trainer = HeadTrainer(model, lr=5e-3, seed=5)
    assert isinstance(trainer.objective, DecisionObjective)
    loader = DataLoader(
        train_records, batch_size=32, shuffle=False, collate_fn=decision_collate_fn
    )
    step = make_supervised_step_fn(model)
    batch = decision_collate_fn([train_records[0], train_records[1]])
    raw_loss, _ = step(batch)
    total, _ = trainer.objective.compute(raw_loss)
    assert float(total.detach()) == pytest.approx(float(raw_loss.detach()))
    trainer.fit(loader, epochs=1)
    assert math.isfinite(trainer.history.losses[0])
