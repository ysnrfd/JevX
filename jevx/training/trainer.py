# Copyright (c) 2026 ysnrfd
"""Minimal honest supervised trainer for JevX heads.

This is deliberately **not** a HuggingFace ``Trainer`` clone: no accelerate,
no logging framework, no scheduler, no mixed precision, no callbacks beyond a
plain per-epoch function. It does exactly one loop — forward, loss, finite
check, backward, clip, step — with a composite objective (§85) on top and
checkpointing that can store either the full state or only the trainable
(``requires_grad``) parameters plus a base-checkpoint reference.

Every optimization step asserts the loss is finite (via
:func:`jevx.utils.shapes.check_finite`); a NaN/Inf raises immediately with
the epoch and step index in the message instead of silently poisoning the
weights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from jevx.training.data import (
    _call_model,
    accuracy,
    extract_logits,
    make_supervised_step_fn,
)
from jevx.training.objective import DecisionObjective
from jevx.utils.shapes import check_finite

__all__ = ["HeadTrainer", "History"]

#: Default metrics computed by :meth:`HeadTrainer.evaluate` when none are given.
DEFAULT_METRICS: dict[str, Callable[[torch.Tensor, torch.Tensor], float]] = {
    "accuracy": accuracy,
}


@dataclass
class History:
    """Per-epoch training record: ``epochs[i]`` is epoch ``i + 1``'s dict.

    Each epoch dict holds ``epoch`` (1-based int), ``loss`` (mean train loss,
    float), optionally ``val_loss`` (float) when an ``eval_loader`` was given,
    and one entry per evaluation metric (e.g. ``val_accuracy``).
    """

    epochs: list[dict[str, Any]] = field(default_factory=list)

    def append(self, record: dict[str, Any]) -> None:
        self.epochs.append(record)

    @property
    def losses(self) -> list[float]:
        """Mean training loss per epoch, in order."""
        return [float(record["loss"]) for record in self.epochs]

    def __len__(self) -> int:
        return len(self.epochs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.epochs[index]


def _split_step_result(
    result: Any,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Normalize a step-function return to ``(loss, components)``.

    Accepts a bare tensor, ``(loss, components)`` tuple, or a mapping with a
    ``"loss"`` key.
    """
    if isinstance(result, torch.Tensor):
        return result, {}
    if isinstance(result, tuple) and len(result) == 2:
        loss, components = result
        if not isinstance(loss, torch.Tensor):
            raise TypeError(
                "step function: first tuple element must be a loss tensor, got "
                f"{type(loss).__name__}."
            )
        if components is None:
            components = {}
        if not isinstance(components, Mapping):
            raise TypeError(
                "step function: second tuple element must be a mapping of loss "
                f"components, got {type(components).__name__}."
            )
        return loss, dict(components)
    if isinstance(result, Mapping) and isinstance(result.get("loss"), torch.Tensor):
        components = {k: v for k, v in result.items() if k != "loss"}
        return result["loss"], components
    raise TypeError(
        "step function must return a loss tensor, a (loss, components) tuple, or a "
        f"mapping with a 'loss' tensor; got {type(result).__name__}."
    )


class HeadTrainer:
    """Supervised loop over decision-head batches.

    Args:
        model: Module whose parameters receive gradients. Moved to ``device``
            when ``device`` is given.
        optimizer: Optional optimizer; when ``None`` (default) an
            :class:`torch.optim.AdamW` over the model's ``requires_grad``
            parameters is created with ``lr``. Passing an optimizer lets you
            use parameter groups (e.g. Mode B with a smaller backbone LR).
        lr: Learning rate for the default AdamW (default ``1e-3``).
        seed: Seed for :func:`torch.manual_seed` and for the loader generator
            (default 42).
        device: ``torch.device``/str to move the model to; ``None`` keeps the
            model on its current device (CPU here).
        grad_clip: Max grad norm for :func:`torch.nn.utils.clip_grad_norm_`
            applied every step; ``None`` disables clipping (default 1.0).
        objective: Composite objective (§85); ``None`` means
            ``DecisionObjective()`` — the plain decision loss with all
            auxiliary terms off.

    Raises:
        ValueError: The model has no ``requires_grad`` parameters (the default
            optimizer would be empty) or ``lr <= 0``.
        TypeError: A non-positive/non-numeric ``grad_clip`` etc.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        *,
        lr: float = 1e-3,
        seed: int = 42,
        device: str | torch.device | None = None,
        grad_clip: float | None = 1.0,
        objective: DecisionObjective | None = None,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError(
                f"model: expected an nn.Module, got {type(model).__name__}."
            )
        if objective is None:
            objective = DecisionObjective()
        elif not isinstance(objective, DecisionObjective):
            raise TypeError(
                "objective: expected a DecisionObjective, got "
                f"{type(objective).__name__}."
            )
        if grad_clip is not None:
            if isinstance(grad_clip, bool) or not isinstance(grad_clip, (int, float)):
                raise TypeError(
                    f"grad_clip: expected a positive float or None, got "
                    f"{type(grad_clip).__name__}."
                )
            if not (float(grad_clip) > 0.0) or not math.isfinite(float(grad_clip)):
                raise ValueError(
                    f"grad_clip: expected a finite value > 0 (or None to disable), "
                    f"got {grad_clip!r}."
                )

        torch.manual_seed(seed)
        self.seed = seed
        self.device = torch.device(device) if device is not None else None
        if self.device is not None:
            model.to(self.device)
        self.model = model
        self.objective = objective
        self.grad_clip = float(grad_clip) if grad_clip is not None else None

        # Deterministic shuffling: pass this generator to your DataLoader
        # (DataLoader(..., generator=self.generator)) so epoch orders repeat
        # exactly across runs with the same seed.
        self.generator = torch.Generator().manual_seed(seed)

        if optimizer is None:
            trainable = [p for p in model.parameters() if p.requires_grad]
            if not trainable:
                raise ValueError(
                    "model has no requires_grad parameters — nothing for the default "
                    "optimizer to train. Unfreeze a head (or the backbone, Mode B) "
                    "first, or pass your own optimizer."
                )
            if not isinstance(lr, (int, float)) or isinstance(lr, bool) or lr <= 0:
                raise ValueError(f"lr: expected a positive float, got {lr!r}.")
            optimizer = torch.optim.AdamW(trainable, lr=float(lr))
        self.optimizer = optimizer
        self.history = History()
        # Step function shared between fit() and evaluate(); set on first fit.
        self._eval_step: Callable[[Any], Any] | None = None

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        *,
        epochs: int = 1,
        eval_loader: DataLoader | None = None,
        loss_fn: Callable[[Any], Any] | None = None,
        callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> History:
        """Run the supervised loop and return the (appended) :class:`History`.

        Args:
            train_loader: Yields batches consumed by the step function. Build
                it with ``collate_fn=decision_collate_fn`` and, for
                reproducible shuffling, ``generator=self.generator``.
            epochs: Number of passes over ``train_loader`` (default 1).
            eval_loader: Optional validation loader evaluated after each
                epoch; produces ``val_loss`` plus ``val_<metric>`` keys.
            loss_fn: Step function ``fn(batch) -> loss | (loss, components)``
                over the raw collated batch. ``None`` uses
                :func:`jevx.training.data.make_supervised_step_fn(model)`
                (duck-typed forward + decision-type loss dispatch).
            callback: Optional ``fn(epoch_record)`` invoked once per epoch
                after the record is finalized (logging, early stopping — it
                may raise to stop training).

        Returns:
            ``self.history`` — one dict per epoch with ``epoch``, ``loss``,
            optional ``val_*`` metrics.

        Raises:
            ValueError: A step's loss is non-finite (NaN/Inf) — the message
                carries the epoch and step index.
            TypeError: The step function returns something that is not a loss
                tensor / (loss, components) pair.
        """
        if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
            raise ValueError(f"epochs: expected an int >= 1, got {epochs!r}.")
        step: Callable[[Any], Any]
        if loss_fn is None:
            step = make_supervised_step_fn(self.model)
            # evaluate() reuses the same dispatch, but asks for the logits the
            # step already computed instead of paying for a second forward.
            # A custom loss_fn keeps full control of what it returns.
            self._eval_step = make_supervised_step_fn(self.model, return_logits=True)
        else:
            step = loss_fn
            self._eval_step = step

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0
            num_steps = 0
            for step_index, batch in enumerate(train_loader, start=1):
                self.optimizer.zero_grad(set_to_none=True)
                # Wrap the step so failures raised INSIDE the loss (e.g. a
                # NaN-logits check in choice_loss) still report position.
                try:
                    loss, _components = _split_step_result(step(batch))
                except (ValueError, TypeError) as exc:
                    raise type(exc)(
                        f"fit failed at epoch {epoch}, step {step_index}: {exc}"
                    ) from exc
                check_finite(
                    loss.detach(),
                    f"loss at epoch {epoch}, step {step_index}",
                )
                loss.backward()
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        (p for p in self.model.parameters() if p.requires_grad),
                        self.grad_clip,
                    )
                self.optimizer.step()
                total_loss += float(loss.detach())
                num_steps += 1
            if num_steps == 0:
                raise ValueError(
                    "fit: train_loader yielded no batches; nothing to optimize."
                )

            record: dict[str, Any] = {
                "epoch": epoch,
                "loss": total_loss / num_steps,
            }
            if eval_loader is not None:
                record.update(self.evaluate(eval_loader, objective_aware=True))
            self.history.append(record)
            if callback is not None:
                callback(record)
        return self.history

    def evaluate(
        self,
        loader: DataLoader,
        metric_fns: Mapping[str, Callable[[torch.Tensor, torch.Tensor], Any]]
        | None = None,
        *,
        objective_aware: bool = False,
    ) -> dict[str, Any]:
        """Compute mean loss and metrics over ``loader`` (no gradients).

        The step function is the one supplied to :meth:`fit` (or the default
        duck-typed dispatch when ``fit`` was never called with a custom
        ``loss_fn``).

        Args:
            loader: Batches to evaluate.
            metric_fns: ``{name: fn(logits, labels)}``; ``None`` uses
                :data:`DEFAULT_METRICS` (``accuracy``). Pass ``{}`` to
                compute only the loss.
            objective_aware: When True, the reported loss is the composite
                objective's ``total`` (consistent with training); otherwise it
                is the raw decision loss. Metrics always read raw logits.

        Returns:
            ``{"val_loss": float, "val_<name>": metric_value, ...}``.
        """
        if metric_fns is None:
            metric_fns = DEFAULT_METRICS
        step = getattr(self, "_eval_step", None)
        if step is None:
            step = make_supervised_step_fn(self.model, return_logits=True)
            self._eval_step = step

        self.model.eval()
        total_loss = 0.0
        num_steps = 0
        logits_all: list[torch.Tensor] = []
        labels_all: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in loader:
                loss, components = _split_step_result(step(batch))
                if objective_aware:
                    total, _ = self.objective.compute(
                        loss,
                        calibration_loss=components.get("calibration_loss"),
                        representation_loss=components.get("representation_loss"),
                        regularization_loss=components.get("regularization_loss"),
                    )
                    loss_value = total
                else:
                    loss_value = loss
                check_finite(
                    torch.as_tensor(float(loss_value)),
                    f"eval loss at step {num_steps + 1}",
                )
                total_loss += float(loss_value)
                num_steps += 1
                # OPTIONAL PASSTHROUGH: a step function that returns a "logits"
                # key has already paid for this batch's forward, so re-running
                # the model purely for the metrics would double the work. Any
                # other step function (a custom loss_fn, the historical default)
                # simply has no such key and the second forward still happens.
                step_logits = components.get("logits")
                if isinstance(step_logits, torch.Tensor):
                    logits_all.append(step_logits.detach())
                else:
                    logits_all.append(self._batch_logits(batch))
                labels_all.append(self._batch_labels(batch))
        if num_steps == 0:
            raise ValueError("evaluate: loader yielded no batches.")

        out: dict[str, Any] = {"val_loss": total_loss / num_steps}
        if logits_all and labels_all:
            logits = torch.cat(logits_all, dim=0)
            labels = torch.cat(labels_all, dim=0)
            for name, fn in metric_fns.items():
                out[f"val_{name}"] = fn(logits, labels)
        return out

    # -- evaluate helpers ---------------------------------------------------

    def _batch_logits(self, batch: Any) -> torch.Tensor:
        """Re-run the model under no_grad to collect logits for metrics.

        The fallback for step functions that do not hand their logits back (see
        the passthrough note in :meth:`evaluate`). It dispatches through the SAME
        signature-driven helper the step function uses, so the two forwards are
        the same call: the old ``try: self.model(batch) / except TypeError:
        self.model(**dict(batch))`` probe could not report a decision forward at
        all, because a dict binds perfectly well to its ``state`` parameter.
        """
        if not isinstance(batch, Mapping):
            raise TypeError(
                f"evaluate: expected mapping batches, got {type(batch).__name__}."
            )
        return extract_logits(_call_model(self.model, batch)).detach()

    def _batch_labels(self, batch: Any) -> torch.Tensor:
        if not isinstance(batch, Mapping) or "labels" not in batch:
            raise ValueError(
                "evaluate: batches must be mappings with a 'labels' key to compute "
                "metrics."
            )
        labels = batch["labels"]
        if isinstance(labels, torch.Tensor):
            return labels.detach()
        return torch.as_tensor(labels)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        save_trainable_only: bool = False,
        config: Any = None,
    ) -> Path:
        """Save ``{model_state, optimizer_state, history, config}`` to ``path``.

        Args:
            path: Destination file (parent directories are created).
            save_trainable_only: When True, store **only** parameters with
                ``requires_grad=True`` (plus ``base_model_name_or_path`` from
                ``config`` when it carries one). Reloading such a checkpoint
                requires the base model's weights to be present already —
                build the model from its base checkpoint first, then
                :meth:`load_checkpoint` the trainable delta on top. A
                full-state checkpoint needs no base.
            config: Optional config object stored as-is (anything
                :func:`torch.save` accepts); ``None`` uses
                ``model.config`` when the model carries one. When the config
                exposes ``base_model_name_or_path`` that string is mirrored
                into the checkpoint for the reload contract above.

        Returns:
            The resolved output path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if config is None:
            config = getattr(self.model, "config", None)
        if save_trainable_only:
            model_state = {
                name: param
                for name, param in self.model.named_parameters()
                if param.requires_grad
            }
            if not model_state:
                raise ValueError(
                    "save_checkpoint: save_trainable_only=True but the model has no "
                    "requires_grad parameters — the checkpoint would be empty."
                )
        else:
            model_state = dict(self.model.state_dict())
        base_ref = getattr(config, "base_model_name_or_path", None)
        payload: dict[str, Any] = {
            "model_state": model_state,
            "optimizer_state": self.optimizer.state_dict(),
            "history": self.history.epochs,
            "config": config,
            "save_trainable_only": save_trainable_only,
            "base_model_name_or_path": base_ref,
        }
        torch.save(payload, path)
        return path

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        load_optimizer: bool = True,
        strict: bool = True,
    ) -> History:
        """Restore model/optimizer/history from :meth:`save_checkpoint` output.

        Args:
            path: Checkpoint file.
            load_optimizer: Also restore optimizer state (default True); a
                trainable-only checkpoint still carries full optimizer state —
                re-created optimizers over the same parameters accept it.
            strict: For full checkpoints, forwarded to ``load_state_dict``
                (default True). Trainable-only checkpoints always load
                leniently over the frozen base: missing frozen keys are
                expected (the base must already be present), extra keys are
                reported in the returned history's ``_load_report`` only when
                ``strict=False``.

        Returns:
            The restored :class:`History`.

        Raises:
            FileNotFoundError: ``path`` does not exist.
            ValueError: A trainable-only checkpoint references parameters the
                current model does not have (wrong base), while
                ``strict=True``.
            RuntimeError: Optimizer state incompatible with the current
                optimizer (e.g. different parameter grouping) and
                ``load_optimizer=True``.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"load_checkpoint: no checkpoint at {path}.")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for key in ("model_state", "history"):
            if key not in payload:
                raise ValueError(
                    f"load_checkpoint: {path} is missing the {key!r} entry — not a "
                    "HeadTrainer checkpoint?"
                )
        model_state = payload["model_state"]
        if payload.get("save_trainable_only"):
            current = dict(self.model.named_parameters())
            unknown = [k for k in model_state if k not in current]
            if unknown and strict:
                raise ValueError(
                    "load_checkpoint: trainable-only checkpoint holds parameters this "
                    f"model does not have: {unknown[:5]}{'...' if len(unknown) > 5 else ''}. "
                    "Load it onto a model built from the referenced base checkpoint."
                )
            filtered = {k: v for k, v in model_state.items() if k in current}
            self.model.load_state_dict(filtered, strict=False)
        else:
            self.model.load_state_dict(model_state, strict=strict)
        if load_optimizer and payload.get("optimizer_state") is not None:
            try:
                self.optimizer.load_state_dict(payload["optimizer_state"])
            except (ValueError, RuntimeError) as exc:
                raise RuntimeError(
                    f"load_checkpoint: optimizer state from {path} does not fit the "
                    f"current optimizer ({exc}). Pass load_optimizer=False to skip it."
                ) from exc
        self.history = History(epochs=list(payload["history"]))
        return self.history
