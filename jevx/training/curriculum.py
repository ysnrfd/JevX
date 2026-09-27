# Copyright (c) 2026 ysnrfd
"""Staged (curriculum) training for JevX decision heads.

Motivation
----------
A JevX decision head scores option `k` as

    z_k = s^T W[:, k] + b[k]  +  lambda * r_k(s, e_k)

with `s` the fused state, `W`/`b` the additive STATE pathway and `r_k` the
option/schema INTERACTION residual. When the whole head is trained from step 0
the interaction term is already at full strength while the state pathway is
still at its initialization, so the first batches see a loss dominated by the
interaction branch and the additive path never gets a clean gradient (EXP-H:
first-batch loss 178-239 against 2.08 for a plain probe). On a question that
carries no information the interaction term can only add variance.

Stage semantics
---------------
:func:`run_curriculum` trains the head in two stages.

* **Stage A** (`condition_options=False`) trains ONLY the state pathway.
  Every training step calls the model with the option/schema pathway forced
  off, so the option encoder, the schema projection and the interaction
  parameters are outside the autograd graph and keep `grad is None`. The
  interaction scale is pinned to `0.0` for the whole stage, so the stage ends
  with the state pathway fitted and the interaction term still exactly zero.
* **Stage B** (`condition_options=True`) re-enables the option/schema pathway
  and ramps the interaction scale from `lambda_start` to `lambda_end` over the
  stage, so the newly opened pathway grows into a state pathway that already
  solves the task instead of fighting it.

The per-call flag and the scale are applied by the SAME object,
:class:`_StageController`, which drives the model view once per epoch and again
before every training step. So the flag in force is a pure function of the
stage - `False` on every Stage-A call, `True` on every Stage-B call - exactly
like the scale, and every :class:`EpochRecord` records the flag its epoch ran
under (`EpochRecord.condition_options`). Nothing else may set the flag: a flag
that outlives its stage is what would silently turn Stage B back into Stage A.

The lambda schedule
-------------------
The scale is a PURE function of the step's position in Stage B. With
`steps_per_epoch` optimizer steps per epoch and `total_steps =
stage_b_epochs * steps_per_epoch`:

    step_index = epoch * steps_per_epoch + step        (both 0-based)
    progress   = 0.0                if total_steps == 1
                 step_index / (total_steps - 1)        otherwise
    shape      = progress                            (schedule="linear")
                 0.5 - 0.5 * cos(pi * progress)       (schedule="cosine")
    lambda     = lambda_start + (lambda_end - lambda_start) * shape

Consequences the tests pin: the FIRST Stage-B step always uses `lambda_start`
and the LAST always uses `lambda_end`; with `total_steps == 1` the single step
uses `lambda_start`. Nothing in the ramp depends on the seed, on the data or
on the optimizer, so the recorded sequence is reproducible by hand.

Determinism
-----------
:func:`run_curriculum` calls `torch.manual_seed(curriculum.seed)` and re-seeds
any `torch.Generator` reachable from the loaders it is given, so the batch
order is fixed from the first epoch whatever the caller did. Two runs with the
same seed and the same freshly-constructed model produce identical histories.

What is reused (nothing here forks the training stack)
------------------------------------------------------
* `jevx.training.data.make_supervised_step_fn` for batch -> logits -> loss
  dispatch (label tensorization, the `[B, 1] -> [B]` squeeze and the
  decision-type loss table), wrapped in a thin view that injects the stage's
  `condition_options` flag;
* `jevx.training.trainer.HeadTrainer` for the optimizer loop itself (grad
  clipping, the finite-loss guard with epoch/step context, and `evaluate`),
  driven one epoch at a time so the per-step ramp can be interleaved;
* `jevx.training.data.accuracy` / `macro_f1` for the reported metrics;
* the JevX call convention `model(state=..., questions=..., decision_type=...,
  options=..., condition_options=...)`.

No-test-leakage guard
---------------------
:func:`run_curriculum` refuses any evaluation split whose name (the
`eval_loaders` key, or a `split`/`name` attribute on the loader or its dataset)
contains `test`. Curriculum evaluation exists to select a model on
CAL/validation; scoring the test split here would leak it into the ramp's
hyper-parameters and into every number in the history. There is no override: a
test split is measured once, by the evaluation harness, not by the trainer.

Batch contract
--------------
Loaders are built with `jevx.training.data.DecisionDataset` +
`decision_collate_fn`, and this module adds two restrictions needed to speak
the model's forward contract from a per-record batch:

1. every record carries exactly ONE question and one label (a multi-question
   record is the same as several single-question records);
2. the question texts are identical across a batch (the model's raw-text
   `questions` argument is one list shared by the whole state batch).

Both are checked, with the offending batch reported, instead of being silently
coerced.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from jevx.training.data import (
    accuracy,
    extract_logits,
    macro_f1,
    make_supervised_step_fn,
)
from jevx.training.trainer import HeadTrainer

__all__ = [
    "CURRICULUM_FORMAT",
    "LAMBDA_SCHEDULES",
    "STAGE_A",
    "STAGE_B",
    "CurriculumConfig",
    "CurriculumHistory",
    "EpochRecord",
    "interaction_scale",
    "read_interaction_scale",
    "run_curriculum",
]

#: Key stamped into every :meth:`CurriculumHistory.save` payload.
CURRICULUM_FORMAT = "jevx-training-curriculum-v1"

#: Stage identifiers used in :attr:`EpochRecord.stage`.
STAGE_A = "a"
STAGE_B = "b"

#: Accepted values of :attr:`CurriculumConfig.lambda_schedule`.
LAMBDA_SCHEDULES: tuple[str, ...] = ("linear", "cosine")

#: Batch keys that are training-only and are NOT forwarded to the model.
_TRAINING_ONLY_KEYS: frozenset = frozenset(
    {"labels", "option_mask", "split", "num_options"}
)

#: Substring that marks a split as a test split in the no-test-leakage guard.
_TEST_SPLIT_TOKEN = "test"

#: Attributes inspected (on the loader and on its dataset) for a split tag.
_SPLIT_TAG_ATTRS: tuple = ("split", "split_name", "name")


# ---------------------------------------------------------------------------
# The ramp
# ---------------------------------------------------------------------------


def interaction_scale(
    step_index: int,
    total_steps: int,
    *,
    lambda_start: float = 0.0,
    lambda_end: float = 1.0,
    schedule: str = "linear",
) -> float:
    """Interaction scale at one Stage-B step: the pure ramp, no state.

    Args:
        step_index: 0-based position of the step inside Stage B, computed as
            `epoch * steps_per_epoch + step` with both indices 0-based (see
            the module docstring).
        total_steps: Number of optimizer steps in Stage B,
            `stage_b_epochs * steps_per_epoch`.
        lambda_start: Scale applied at `step_index == 0`.
        lambda_end: Scale applied at the last step (`step_index ==
            total_steps - 1`).
        schedule: `"linear"` (default) or `"cosine"`.

    Returns:
        The scale to push into `DecisionHead.set_interaction_scale`.

    Raises:
        ValueError: Unknown `schedule`, `total_steps < 1`, or `step_index`
            outside `[0, total_steps)`.
    """
    if schedule not in LAMBDA_SCHEDULES:
        raise ValueError(
            f"schedule: expected one of {list(LAMBDA_SCHEDULES)}, got {schedule!r}."
        )
    if (
        not isinstance(total_steps, int)
        or isinstance(total_steps, bool)
        or total_steps < 1
    ):
        raise ValueError(f"total_steps: expected an int >= 1, got {total_steps!r}.")
    if (
        not isinstance(step_index, int)
        or isinstance(step_index, bool)
        or not 0 <= step_index < total_steps
    ):
        raise ValueError(
            f"step_index: expected an int in [0, {total_steps}), got {step_index!r}."
        )
    progress = 0.0 if total_steps == 1 else step_index / (total_steps - 1)
    if schedule == "linear":
        shape = progress
    else:  # "cosine"
        shape = 0.5 - 0.5 * math.cos(math.pi * progress)
    return float(lambda_start) + (float(lambda_end) - float(lambda_start)) * shape


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CurriculumConfig:
    """Hyper-parameters of a two-stage curriculum run.

    Args:
        stage_a_epochs: Epochs of state-only training (schema conditioning
            off, interaction scale pinned to 0.0). `0` skips Stage A.
        stage_b_epochs: Epochs of staged interaction training with the scale
            ramped over the stage. `0` skips Stage B.
        lambda_start: Interaction scale on the first Stage-B step.
        lambda_end: Interaction scale on the last Stage-B step.
        lambda_schedule: `"linear"` or `"cosine"`.
        lr: AdamW learning rate, shared by both stages (the run is one
            optimization, not two).
        weight_decay: AdamW weight decay.
        grad_clip: Max grad norm (`None` disables clipping).
        seed: Seeds the global RNG and the loader generators.
        decision_type: The decision type the batches are expected to carry
            (`"choice"`, `"score"`, `"noul"`, ...). It selects the loss used
            by the supervised step and, when the model exposes a head with
            that `head_name`, the only head that receives the Stage-B scale.
    """

    stage_a_epochs: int = 4
    stage_b_epochs: int = 4
    lambda_start: float = 0.0
    lambda_end: float = 1.0
    lambda_schedule: str = "linear"
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float | None = 1.0
    seed: int = 42
    decision_type: str = "choice"

    def __post_init__(self) -> None:
        """Validate every field eagerly, naming the offending one."""
        for name in ("stage_a_epochs", "stage_b_epochs", "seed"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(
                    f"{name}: expected an int, got {type(value).__name__}."
                )
        if self.stage_a_epochs < 0 or self.stage_b_epochs < 0:
            raise ValueError(
                "stage_a_epochs/stage_b_epochs: expected non-negative ints, got "
                f"{self.stage_a_epochs}/{self.stage_b_epochs}."
            )
        if self.stage_a_epochs + self.stage_b_epochs < 1:
            raise ValueError(
                "a curriculum with 0 epochs in both stages trains nothing; set "
                "stage_a_epochs >= 1 or stage_b_epochs >= 1."
            )
        for name in ("lambda_start", "lambda_end"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(
                    f"{name}: expected a float, got {type(value).__name__}."
                )
            if not math.isfinite(float(value)):
                raise ValueError(f"{name}: expected a finite value, got {value!r}.")
        if self.lambda_schedule not in LAMBDA_SCHEDULES:
            raise ValueError(
                f"lambda_schedule: expected one of {list(LAMBDA_SCHEDULES)}, got "
                f"{self.lambda_schedule!r}."
            )
        if isinstance(self.lr, bool) or not isinstance(self.lr, (int, float)):
            raise TypeError(f"lr: expected a float, got {type(self.lr).__name__}.")
        if not math.isfinite(float(self.lr)) or float(self.lr) <= 0.0:
            raise ValueError(f"lr: expected a finite value > 0, got {self.lr!r}.")
        if isinstance(self.weight_decay, bool) or not isinstance(
            self.weight_decay, (int, float)
        ):
            raise TypeError(
                f"weight_decay: expected a float, got "
                f"{type(self.weight_decay).__name__}."
            )
        if not math.isfinite(float(self.weight_decay)) or float(self.weight_decay) < 0.0:
            raise ValueError(
                f"weight_decay: expected a finite value >= 0, got "
                f"{self.weight_decay!r}."
            )
        if self.grad_clip is not None:
            if isinstance(self.grad_clip, bool) or not isinstance(
                self.grad_clip, (int, float)
            ):
                raise TypeError(
                    f"grad_clip: expected a positive float or None, got "
                    f"{type(self.grad_clip).__name__}."
                )
            if not math.isfinite(float(self.grad_clip)) or float(self.grad_clip) <= 0.0:
                raise ValueError(
                    f"grad_clip: expected a finite value > 0 (or None to disable), "
                    f"got {self.grad_clip!r}."
                )
        if not isinstance(self.decision_type, str) or not self.decision_type:
            raise ValueError(
                f"decision_type: expected a non-empty str, got "
                f"{self.decision_type!r}."
            )

    @classmethod
    def from_config(cls, source: Any = None, **overrides: Any) -> "CurriculumConfig":
        """Build a config from a mapping, namespace or config object.

        Any attribute named after a :class:`CurriculumConfig` field is read
        off `source`; everything else is IGNORED (a `JevXConfig` carries
        hundreds of unrelated fields and only the handful below are curriculum
        knobs). Fields absent from `source` keep their default, and explicit
        `overrides` always win over `source`.

        Args:
            source: Mapping, :class:`CurriculumConfig`, or any object (e.g.
                an argparse namespace) exposing the field names; `None` builds
                the defaults.
            **overrides: Explicit values, applied last.

        Returns:
            A validated :class:`CurriculumConfig`.
        """
        known = {f.name for f in fields(cls)}
        values: dict = {}
        if source is not None:
            if isinstance(source, CurriculumConfig):
                candidate = {f.name: getattr(source, f.name) for f in fields(source)}
            elif isinstance(source, Mapping):
                candidate = dict(source)
            else:
                candidate = {
                    name: getattr(source, name)
                    for name in known
                    if hasattr(source, name)
                }
            values = {key: value for key, value in candidate.items() if key in known}
        values.update(overrides)
        return cls(**values)

    def to_dict(self) -> dict:
        """Plain-dict view (the form :meth:`from_config` reads back)."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


@dataclass
class EpochRecord:
    """One epoch of one stage.

    Args:
        stage: `STAGE_A` or `STAGE_B`.
        epoch: 1-based epoch index within its stage.
        global_epoch: 1-based epoch index across the whole curriculum.
        train_loss: Mean training loss over the epoch.
        interaction_scale: The scale in force at the epoch's LAST training
            step, which is therefore the scale its evaluation ran under.
        interaction_scale_first: The scale of the epoch's FIRST training step.
        steps: Number of optimizer steps in the epoch.
        metrics: `{split}_loss`, `{split}_accuracy`, `{split}_macro_f1` per
            evaluation split.
        condition_options: The per-call `condition_options` flag the epoch's
            training steps AND its evaluation ran under: `False` in Stage A
            (the option/schema pathway is out of the graph), `True` in Stage B
            (it is in the graph, scaled by `interaction_scale`). Recorded so
            the stage claim is inspectable from the history alone.
    """

    stage: str
    epoch: int
    global_epoch: int
    train_loss: float
    interaction_scale: float
    interaction_scale_first: float
    steps: int
    metrics: dict = field(default_factory=dict)
    condition_options: bool = False

    def to_dict(self) -> dict:
        """JSON-shaped view of this record."""
        return {
            "stage": self.stage,
            "epoch": self.epoch,
            "global_epoch": self.global_epoch,
            "train_loss": self.train_loss,
            "interaction_scale": self.interaction_scale,
            "interaction_scale_first": self.interaction_scale_first,
            "steps": self.steps,
            "metrics": dict(self.metrics),
            "condition_options": bool(self.condition_options),
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "EpochRecord":
        """Rebuild a record from :meth:`to_dict` output."""
        return cls(
            stage=str(data["stage"]),
            epoch=int(data["epoch"]),
            global_epoch=int(data["global_epoch"]),
            train_loss=float(data["train_loss"]),
            interaction_scale=float(data["interaction_scale"]),
            interaction_scale_first=float(data["interaction_scale_first"]),
            steps=int(data["steps"]),
            metrics={str(k): float(v) for k, v in dict(data.get("metrics", {})).items()},
            # A payload written before the flag was recorded has no key here.
            # Defaulting to True keeps such a record from claiming the pathway
            # was closed, which is the one reading a Stage-B record must never
            # take.
            condition_options=bool(data.get("condition_options", True)),
        )


@dataclass
class CurriculumHistory:
    """Per-epoch records for a whole curriculum run, plus its config.

    Args:
        records: Epoch records in execution order (Stage A first).
        curriculum: The config the run used (`None` when unknown, e.g. after
            loading a history that was saved without one).
    """

    records: list = field(default_factory=list)
    curriculum: Any = None

    def append(self, record: EpochRecord) -> None:
        """Add one epoch record."""
        self.records.append(record)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator:
        return iter(self.records)

    def __getitem__(self, index: int) -> EpochRecord:
        return self.records[index]

    @property
    def stage_a(self) -> list:
        """The Stage-A records, in order."""
        return [record for record in self.records if record.stage == STAGE_A]

    @property
    def stage_b(self) -> list:
        """The Stage-B records, in order."""
        return [record for record in self.records if record.stage == STAGE_B]

    def to_dict(self) -> dict:
        """Self-contained dict: `{"format", "curriculum", "records"}`."""
        return {
            "format": CURRICULUM_FORMAT,
            "curriculum": None if self.curriculum is None else self.curriculum.to_dict(),
            "records": [record.to_dict() for record in self.records],
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "CurriculumHistory":
        """Rebuild a history from :meth:`to_dict` output.

        Raises:
            ValueError: The payload carries a different `format`.
        """
        payload_format = data.get("format")
        if payload_format != CURRICULUM_FORMAT:
            raise ValueError(
                f"history format: expected {CURRICULUM_FORMAT!r}, got "
                f"{payload_format!r} — not a JevX curriculum history."
            )
        curriculum = data.get("curriculum")
        return cls(
            records=[EpochRecord.from_dict(item) for item in data["records"]],
            curriculum=(
                None if curriculum is None else CurriculumConfig.from_config(curriculum)
            ),
        )

    def save(self, path) -> Path:
        """Write the history to `path` (parent directories created).

        Same convention as `HeadTrainer.save_checkpoint`: a `torch.save` dict
        stamped with a format key. Only the history and the config are stored —
        the MODEL keeps its own checkpoint format (the interaction scale rides
        the model state dict as a buffer, so `save_pretrained` /
        `from_pretrained` carry it without help from here).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.to_dict(), path)
        return path

    @classmethod
    def load(cls, path) -> "CurriculumHistory":
        """Read a history written by :meth:`save`.

        Raises:
            FileNotFoundError: `path` does not exist.
            ValueError: The payload is not a curriculum history.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"load: no curriculum history at {path}.")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping) or "records" not in payload:
            raise ValueError(
                f"load: {path} is not a curriculum history payload (keys: "
                f"{sorted(payload) if isinstance(payload, Mapping) else type(payload).__name__})."
            )
        return cls.from_dict(payload)


# ---------------------------------------------------------------------------
# The model view: one collated batch -> one model call -> logits
# ---------------------------------------------------------------------------


def _logits_from_output(output: Any) -> torch.Tensor:
    """Stack one model call's outputs into the logits a loss wants.

    Two output spellings are supported, in this order:

    * an object with a `decisions` sequence (JevXOutput), whose entries are
      stacked in their documented flat order `b * m + j` (state-major) so the
      rows line up with the batch's labels;
    * anything `jevx.training.data.extract_logits` understands (a bare
      tensor, or an object/mapping carrying a logits attribute).
    """
    decisions = getattr(output, "decisions", None)
    if decisions is None:
        return extract_logits(output)
    rows: list = []
    for index, decision in enumerate(decisions):
        if decision is None:
            raise ValueError(
                f"model output: pair {index} of {len(decisions)} has no decision "
                "(None) - the batch produced a (state, question) pair with no "
                "answer, so no logits can be stacked."
            )
        rows.append(extract_logits(decision))
    if not rows:
        raise ValueError(
            "model output: 'decisions' is empty - the batch produced no "
            "(state, question) pair."
        )
    return torch.cat(rows, dim=0)


def _forward_kwargs(batch: Mapping, condition_options: bool) -> dict:
    """Translate one collated batch into model keyword arguments.

    Training-only keys (labels, option_mask, split, num_options) are dropped:
    the supervised step reads the labels itself. The per-record questions list
    is squeezed to the shared form the model expects, after checking that
    every record in the batch agrees and carries exactly one question, and
    the per-record `[B][m][K]` schema nest is flattened to the model's
    `[B*m]` list of schemas (state-major, `b * m + j`). The stage's
    `condition_options` flag is the only argument invented here.

    Raises:
        TypeError: `batch` is not a mapping.
        ValueError: A required key is missing, a record carries more than one
            question or label, or the batch mixes question texts.
    """
    if not isinstance(batch, Mapping):
        raise TypeError(
            f"batch: expected a mapping from decision_collate_fn, got "
            f"{type(batch).__name__}."
        )
    missing = [key for key in ("state", "questions") if key not in batch]
    if missing:
        raise ValueError(
            f"batch: missing required key(s) {missing}; build the loader with "
            f"collate_fn=decision_collate_fn over records holding "
            f"{['state', 'questions', 'decision_type', 'options', 'labels']}."
        )
    questions = batch["questions"]
    if not isinstance(questions, Sequence) or len(questions) == 0:
        raise ValueError(
            f"batch['questions']: expected a non-empty sequence, got "
            f"{type(questions).__name__}."
        )
    if isinstance(questions[0], (str, dict)):
        shared: Any = list(questions)
    else:
        first = list(questions[0])
        if len(first) != 1:
            raise ValueError(
                f"batch['questions']: record 0 carries {len(first)} question(s); "
                "curriculum training is defined for exactly ONE question per "
                "record (split a multi-question record into one record per "
                "question) so that one record is one decision."
            )
        for index, entry in enumerate(questions):
            if list(entry) != first:
                raise ValueError(
                    f"batch['questions']: record {index} asks {list(entry)} but "
                    f"record 0 asks {first}; the model's raw-text 'questions' "
                    "argument is shared by the whole state batch, so a batch "
                    "must carry one question. Split the loader by question."
                )
        shared = first
    labels = batch.get("labels")
    if labels is not None and isinstance(labels, Sequence):
        for index, entry in enumerate(labels):
            if isinstance(entry, Sequence) and len(entry) != 1:
                raise ValueError(
                    f"batch['labels']: record {index} carries {len(entry)} "
                    "label(s); curriculum training is defined for exactly one "
                    "label per record (see batch['questions'])."
                )
    kwargs = {
        key: value for key, value in batch.items() if key not in _TRAINING_ONLY_KEYS
    }
    kwargs["questions"] = shared
    # A per-record [B][m][K] schema nest becomes the model's flat [B*m] list of
    # schemas, in the documented state-major order b * m + j (records outer,
    # questions inner), so decision pair i lines up with labels row i.
    options = kwargs.get("options")
    if options is not None:
        kwargs["options"] = [
            schema for record in options for schema in record
        ]
    kwargs["condition_options"] = condition_options
    return kwargs


class _CurriculumModelView(nn.Module):
    """Stage-aware view of a model, presented to HeadTrainer.

    Holding the model as a child keeps parameters(), to(), train() and eval()
    pointed at the real model, so the trainer's optimizer, device resolution
    and mode switches stay exact. `forward` turns one collated batch into one
    model call carrying the current stage's `condition_options` flag and
    returns the stacked logits.

    Args:
        model: The model being trained.
        condition_options: Fallback flag for a call made before
            :class:`~_StageController` has placed a stage. :func:`run_curriculum`
            never relies on it: it constructs the view without one and the
            controller sets the real per-stage value before the first batch.
    """

    def __init__(self, model: nn.Module, condition_options: bool = True) -> None:
        super().__init__()
        self.model = model
        self.condition_options = bool(condition_options)

    def set_condition_options(self, flag: bool) -> bool:
        """Put the per-call stage flag in force; returns the flag now set.

        This is the ONLY way the stage's flag reaches :meth:`forward`, and its
        only caller is :meth:`_StageController.apply_stage` - once per epoch
        (so the evaluation that closes the epoch runs under the same flag as
        its training steps) and again before every training step. Keeping the
        write in one place is what makes "Stage B called the model with
        `condition_options=True`" a checkable property rather than a comment.
        """
        self.condition_options = bool(flag)
        return self.condition_options

    def forward(self, batch: Mapping) -> torch.Tensor:
        """Run one stage-flagged model call and stack its logits."""
        kwargs = _forward_kwargs(batch, self.condition_options)
        return _logits_from_output(self.model(**kwargs))


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------


def _split_tags(loader: Any) -> list:
    """Split names declared by a loader or its dataset (split / name)."""
    tags: list = []
    for holder in (loader, getattr(loader, "dataset", None)):
        if holder is None:
            continue
        for attr in _SPLIT_TAG_ATTRS:
            value = getattr(holder, attr, None)
            if isinstance(value, str) and value:
                tags.append(value)
    return tags


def _assert_no_test_split(eval_loaders: Mapping) -> None:
    """Refuse a test split as an evaluation split (no-test-leakage).

    A split counts as a test split when the `eval_loaders` key, or a
    `split`/`split_name`/`name` attribute on the loader or on its dataset,
    contains `test` (case-insensitive).

    Raises:
        TypeError: `eval_loaders` is not a mapping.
        ValueError: A split looks like a test split. The message names the
            split and states the no-test-leakage rule; there is no override.
    """
    if not isinstance(eval_loaders, Mapping):
        raise TypeError(
            f"eval_loaders: expected a mapping of split name -> DataLoader, got "
            f"{type(eval_loaders).__name__}."
        )
    for key, loader in eval_loaders.items():
        for tag in [str(key)] + _split_tags(loader):
            if _TEST_SPLIT_TOKEN in tag.lower():
                raise ValueError(
                    f"eval split {tag!r} is a test split - run_curriculum refuses "
                    "it (no-test-leakage): curriculum evaluation is for model "
                    "selection on CAL/validation only, and a test score here "
                    "would leak the test split into the interaction-scale ramp "
                    "and into every number in the history. Evaluate the test "
                    "split once, from the evaluation harness, after the "
                    "curriculum is fixed."
                )


def _conditioning_heads(model: nn.Module) -> list:
    """Submodules exposing the interaction-scale control surface.

    Duck-typed on `set_interaction_scale` plus a readable
    `get_interaction_scale`, so any head implementation (registered or not)
    is picked up without importing the decision package.
    """
    heads: list = []
    for module in model.modules():
        if not callable(getattr(module, "set_interaction_scale", None)):
            continue
        try:
            getattr(module, "get_interaction_scale")
        except AttributeError:  # pragma: no cover - setter without a getter
            continue
        heads.append(module)
    return heads


def read_interaction_scale(model: nn.Module) -> float:
    """First conditioning head's current interaction scale.

    Args:
        model: Model to read from.

    Returns:
        The scale as a python float (a 0-dim tensor buffer is accepted).

    Raises:
        ValueError: The model exposes no conditioning head.
    """
    heads = _conditioning_heads(model)
    if not heads:
        raise ValueError(
            "read_interaction_scale: the model exposes no submodule with "
            "set_interaction_scale/get_interaction_scale, so there is no "
            "interaction scale to read."
        )
    value = getattr(heads[0], "get_interaction_scale")
    if callable(value):
        value = value()
    if isinstance(value, torch.Tensor):
        value = float(value.reshape(()))
    return float(value)


def _assert_conditioning_api(model: nn.Module) -> None:
    """Check forward(..., condition_options=...) before anything is trained.

    The flag is part of BOTH stages: Stage A passes False on every step (that
    is what makes it state-only) and Stage B passes True, so a model that
    cannot take it cannot run a curriculum at all.

    Raises:
        TypeError: The model's forward cannot take the flag.
    """
    try:
        signature = inspect.signature(model.forward)
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return
    parameters = signature.parameters
    accepts = "condition_options" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if not accepts:
        raise TypeError(
            f"run_curriculum: needs forward(..., condition_options=...) - Stage A "
            f"passes condition_options=False on every step (that is what makes it "
            f"state-only) and Stage B passes True; {type(model).__name__}.forward "
            "does not accept it."
        )


def _loader_length(loader: Any, name: str) -> int:
    """Number of batches in a map-style loader (needed to place the ramp)."""
    try:
        length = len(loader)
    except TypeError as exc:
        raise TypeError(
            f"{name}: the curriculum needs a map-style DataLoader (it needs "
            "len(loader) to place the interaction-scale ramp); got "
            f"{type(loader).__name__}."
        ) from exc
    if length < 1:
        raise ValueError(f"{name}: yielded no batches; nothing to train.")
    return int(length)


def _reseed_loaders(seed: int, *loaders: Any) -> None:
    """Re-seed the global RNG and every loader generator, for determinism."""
    torch.manual_seed(int(seed))
    for loader in loaders:
        generator = getattr(loader, "generator", None)
        if isinstance(generator, torch.Generator):
            generator.manual_seed(int(seed))


# ---------------------------------------------------------------------------
# The stage controller: which flag, which scale, when
# ---------------------------------------------------------------------------


class _StageController:
    """Owns the per-step stage flag and the per-step interaction scale.

    Both live here and nowhere else, so each is always a pure function of the
    stage: the scale of (epoch, step, total Stage-B steps), the flag of the
    stage alone (False in Stage A, True in Stage B). The flag reaches the model
    view through :meth:`apply_stage` - once per epoch, so the evaluation that
    closes an epoch runs under the same flag as its training steps, and again
    before every training step, so no step can be trained under the previous
    stage's flag.
    """

    def __init__(
        self,
        model: nn.Module,
        view: _CurriculumModelView,
        curriculum: CurriculumConfig,
        steps_per_epoch: int,
    ) -> None:
        self._view = view
        self._curriculum = curriculum
        self._steps_per_epoch = int(steps_per_epoch)
        self._total_steps = int(curriculum.stage_b_epochs) * self._steps_per_epoch
        heads = _conditioning_heads(model)
        if curriculum.stage_b_epochs > 0 and not heads:
            raise ValueError(
                "run_curriculum: Stage B needs at least one decision head that "
                "exposes set_interaction_scale/get_interaction_scale, and the "
                f"model exposes none ({type(model).__name__}). Set "
                "stage_b_epochs=0 to run the state-only stage on its own."
            )
        targeted = [
            head
            for head in heads
            if getattr(head, "head_name", "") == curriculum.decision_type
        ]
        # A model may carry several heads (choice/score/noul); the curriculum
        # drives the one its batches are for, or all of them when none matches.
        self._heads = targeted or heads
        self.training = True
        self.current_scale = 0.0
        self.epoch_scale_first: Any = None
        self._stage = STAGE_A
        self._epoch = 1
        self._step_in_epoch = 0
        self._checked_decision_type = False

    @property
    def condition_options(self) -> bool:
        """The per-call `condition_options` flag the CURRENT stage requires.

        A pure function of the stage: Stage A is state-only (False), Stage B
        has the option/schema pathway in the graph (True). The ramp alone
        cannot express the difference - `DecisionHead._combine` drops the
        residual whenever the per-call flag is False, whatever the scale
        buffer says - so the flag, not the scale, is what opens Stage B.
        """
        return self._stage != STAGE_A

    def apply_stage(self) -> bool:
        """Push the current stage's flag into the model view; return it.

        Called from :meth:`begin_epoch` (so an epoch's evaluation inherits its
        training flag) and from :meth:`advance` (so a step inherits its stage's
        flag). Both routes write the same value, so "every call in stage S ran
        with the stage-S flag" holds by construction rather than by review.
        """
        return self._view.set_condition_options(self.condition_options)

    def begin_epoch(self, stage: str, epoch: int) -> None:
        """Start a new epoch: reset the within-epoch step counter."""
        self._stage = stage
        self._epoch = int(epoch)
        self._step_in_epoch = 0
        self.epoch_scale_first = None
        self.apply_stage()

    def advance(self) -> float:
        """Apply this step's flag and scale; return the scale applied.

        Stage A forces `condition_options=False` and pins the scale to 0.0;
        Stage B forces `condition_options=True` and walks the documented ramp
        with step_index = epoch * steps_per_epoch + step (both 0-based).
        """
        self.apply_stage()
        if self._stage == STAGE_A:
            value = 0.0
        else:
            index = (self._epoch - 1) * self._steps_per_epoch + self._step_in_epoch
            value = interaction_scale(
                index,
                self._total_steps,
                lambda_start=self._curriculum.lambda_start,
                lambda_end=self._curriculum.lambda_end,
                schedule=self._curriculum.lambda_schedule,
            )
        for head in self._heads:
            head.set_interaction_scale(value)
        if self.epoch_scale_first is None:
            self.epoch_scale_first = value
        self.current_scale = value
        self._step_in_epoch += 1
        return value

    def check_decision_type(self, batch: Mapping) -> None:
        """Validate the first batch's decision type against the config.

        Runs once per run: a loader whose records disagree with
        CurriculumConfig.decision_type would otherwise train (and be reported)
        under a different loss than the one configured.
        """
        if self._checked_decision_type:
            return
        self._checked_decision_type = True
        expected = self._curriculum.decision_type
        value = batch.get("decision_type") if isinstance(batch, Mapping) else None
        if value is None:
            return
        found = value[0] if isinstance(value, Sequence) and value else value
        if isinstance(found, str) and found != expected:
            raise ValueError(
                f"decision_type: the batches are labelled {found!r} but the "
                f"curriculum is configured for {expected!r}. One curriculum run "
                "trains one decision type; split the loader."
            )

    def record_scale(self) -> tuple:
        """(interaction_scale, interaction_scale_first) for the epoch record."""
        first = self.epoch_scale_first
        if first is None:  # pragma: no cover - fit() rejects empty epochs first
            first = self.current_scale
        return self.current_scale, first


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def run_curriculum(
    model: nn.Module,
    train_loader: DataLoader,
    eval_loaders: Mapping,
    *,
    curriculum: CurriculumConfig,
    loss_fn: Callable = None,
) -> CurriculumHistory:
    """Train a JevX decision head with the two-stage curriculum.

    Stage A (stage_a_epochs epochs) calls the model with
    `condition_options=False` on every training step: the option/schema
    pathway is out of the graph, its parameters keep `grad is None`, and the
    interaction scale is pinned to 0.0. Stage B (stage_b_epochs epochs)
    re-enables it and ramps the scale from `lambda_start` to `lambda_end` over
    the stage, per optimizer step (see :func:`interaction_scale` for the exact
    formula). One optimizer and one learning rate serve both stages; the run
    is a single optimization, not two.

    Args:
        model: The model to train. Its forward must accept the state /
            questions / decision_type / options / condition_options keyword
            spellings, and it must expose its decision head(s) through
            `set_interaction_scale` / `get_interaction_scale` for Stage B.
        train_loader: Map-style loader over the training split (build it with
            `DecisionDataset` + `decision_collate_fn`). Every record carries
            ONE question and ONE label; see the module docstring.
        eval_loaders: `{split_name: loader}` for the CAL/validation splits
            scored after every epoch. A split whose name (or split attribute)
            contains `test` is refused: no-test-leakage.
        curriculum: The run's hyper-parameters.
        loss_fn: Optional `loss_fn(logits, labels) -> Tensor` REPLACING the
            decision-type dispatch (choice_loss / score_loss / noul_loss,
            selected by `curriculum.decision_type`). This is the same hook
            `make_supervised_step_fn` takes, so the option mask is NOT
            forwarded to it.

    Returns:
        A `CurriculumHistory` with one record per epoch of each stage.

    Raises:
        TypeError: Bad argument types, or a model whose forward cannot take
            `condition_options` while Stage B is configured.
        ValueError: A test split in `eval_loaders`, an empty loader, a model
            with no conditioning head for Stage B, a batch whose decision
            type disagrees with the config, or a batch breaking the
            one-question-per-record contract.
    """
    if not isinstance(model, nn.Module):
        raise TypeError(
            f"model: expected a torch.nn.Module, got {type(model).__name__}."
        )
    if not isinstance(curriculum, CurriculumConfig):
        raise TypeError(
            f"curriculum: expected a CurriculumConfig, got "
            f"{type(curriculum).__name__}."
        )
    if loss_fn is not None and not callable(loss_fn):
        raise TypeError(
            f"loss_fn: expected a callable loss_fn(logits, labels) or None, got "
            f"{type(loss_fn).__name__}."
        )
    # Refuse a test split before touching the model or the optimizer.
    _assert_no_test_split(eval_loaders)

    steps_per_epoch = _loader_length(train_loader, "train_loader")
    for split, loader in eval_loaders.items():
        _loader_length(loader, f"eval_loaders[{split!r}]")
    _assert_conditioning_api(model)

    _reseed_loaders(
        curriculum.seed,
        train_loader,
        *(loader for loader in eval_loaders.values()),
    )

    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable:
        raise ValueError(
            "run_curriculum: the model has no requires_grad parameters - "
            "nothing for the optimizer to train. Unfreeze a head (or the "
            "backbone) first."
        )
    optimizer = torch.optim.AdamW(
        trainable, lr=curriculum.lr, weight_decay=curriculum.weight_decay
    )

    # The stage flag is NOT a construction-time constant - it belongs to the
    # stage, so the controller writes it at the start of every epoch and
    # before every training step, and records it per epoch on the EpochRecord.
    # Building the view without a flag is therefore safe: the first
    # controller.begin_epoch(...) below puts the Stage-A flag in force before
    # any batch reaches forward().
    view = _CurriculumModelView(model)
    controller = _StageController(model, view, curriculum, steps_per_epoch)
    supervised = make_supervised_step_fn(view, loss_fn)

    def step(batch: Mapping) -> tuple:
        """One optimizer step's loss, with the stage flag and scale applied."""
        if controller.training:
            controller.check_decision_type(batch)
            controller.advance()
        loss, components = supervised(batch)
        components["interaction_scale"] = controller.current_scale
        return loss, components

    trainer = HeadTrainer(
        view, optimizer, seed=curriculum.seed, grad_clip=curriculum.grad_clip
    )
    history = CurriculumHistory(curriculum=curriculum)
    global_epoch = 0
    for stage, epochs in (
        (STAGE_A, curriculum.stage_a_epochs),
        (STAGE_B, curriculum.stage_b_epochs),
    ):
        for epoch in range(1, int(epochs) + 1):
            global_epoch += 1
            controller.begin_epoch(stage, epoch)
            controller.training = True
            trainer.fit(train_loader, epochs=1, loss_fn=step)
            controller.training = False
            metrics: dict = {}
            for split, loader in eval_loaders.items():
                evaluated = trainer.evaluate(
                    loader, {"accuracy": accuracy, "macro_f1": macro_f1}
            )
                for key, value in evaluated.items():
                    if not key.startswith("val_"):  # pragma: no cover
                        raise ValueError(
                            "HeadTrainer.evaluate returned the unprefixed key "
                            f"{key!r}; expected 'val_*'."
                        )
                    metrics[f"{split}_{key[len('val_'):]}"] = float(value)
            scale, scale_first = controller.record_scale()
            history.append(
                EpochRecord(
                    stage=stage,
                    epoch=epoch,
                    global_epoch=global_epoch,
                    train_loss=float(trainer.history.epochs[-1]["loss"]),
                    interaction_scale=float(scale),
                    interaction_scale_first=float(scale_first),
                    steps=steps_per_epoch,
                    metrics=metrics,
                    # Read the flag off the VIEW, not off the controller:
                    # the view holds what every call in this epoch was
                    # actually made with (training steps and the evaluation
                    # alike), so the record is evidence rather than a
                    # restatement of the stage the controller intended.
                    condition_options=view.condition_options,
                )
            )
            controller.training = True
    return history
