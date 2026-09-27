# Copyright (c) 2026 ysnrfd
"""Decision-task data plumbing: dataset, collate, supervised step function.

Call pattern (no downloads, no tensorization in the loader)::

    ds = DecisionDataset([
        {"state": "...", "questions": ["q?"], "decision_type": "choice",
         "options": [["a"], ["b"], ["c"]], "labels": 1},
        ...
    ])
    loader = DataLoader(ds, batch_size=32, shuffle=True,
                        collate_fn=decision_collate_fn,
                        generator=trainer.generator)   # deterministic shuffle
    step = make_supervised_step_fn(model)              # -> fn(batch)
    loss, components = step(batch)                     # raw python batch in

The collate keeps **raw python structures** (lists/dicts/str) — tensor
conversion happens inside the step function, on the model's device, so the
DataLoader stays device-agnostic and picklable.

``make_supervised_step_fn`` dispatches on the model's **declared forward
signature** (see the ``_call_model`` helper):

1. A forward declaring the JevX decision keys (``state``/``questions``/
   ``decision_type``/``options``) is called with the collated dict as keyword
   arguments — one key per parameter. It must NOT be probed positionally:
   a dict IS a valid ``state=`` for such a forward, so ``model(batch)``
   neither raises nor signals anything, it just runs with the whole batch
   bound to the state.
2. A forward taking the whole collated dict in ONE positional parameter
   (the view-wrapper convention, e.g. ``_CurriculumModelView``) is called
   as ``model(batch)``.
3. Anything else is tried as ``model(**batch)`` and retried
   ``model(batch)`` on ``TypeError``; both failures are reported together.
4. Logits are extracted from the output: a bare tensor, an object/dict with
   ``.logits``/``["logits"]`` (e.g. ``DecisionOutput``).
5. Unless a custom ``loss_fn(logits, labels)`` is supplied, the loss is
   chosen by the batch's ``decision_type``: ``"choice"`` →
   :func:`jevx.training.losses.choice_loss`, ``"score"`` → ``score_loss``
   (mode ``"classification"`` by default), ``"noul"`` → ``noul_loss``,
   ``"multilabel"`` → ``multilabel_loss``. A batch-level ``option_mask``
   (when present) is forwarded to the option-wise losses.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from jevx.training.losses import (
    choice_loss,
    multilabel_loss,
    noul_loss,
    score_loss,
)

__all__ = [
    "DecisionDataset",
    "accuracy",
    "confusion_matrix",
    "decision_collate_fn",
    "extract_logits",
    "macro_f1",
    "make_supervised_step_fn",
]

_REQUIRED_KEYS: tuple[str, ...] = (
    "state",
    "questions",
    "decision_type",
    "options",
    "labels",
)

_LOSS_BY_TYPE: dict[str, Callable[..., torch.Tensor]] = {
    "choice": choice_loss,
    "score": score_loss,
    "noul": noul_loss,
    "multilabel": multilabel_loss,
}


class DecisionDataset(Dataset):
    """Dataset over ``list[dict]`` decision records.

    Each record must contain:

    * ``state`` — the raw state ``S`` (any picklable python object);
    * ``questions`` — ``list[str]`` of question texts;
    * ``decision_type`` — ``str`` (e.g. ``"choice"``);
    * ``options`` — ``list[list[str | dict]]``, the schema ``C`` per question;
    * ``labels`` — ``list`` of ground-truth labels.

    Validation is structural only (keys present, outer types sane) — semantic
    validation of labels happens in the loss at step time, where the logit
    width is known.

    Args:
        records: The source list; each element is kept as-is (not copied).

    Raises:
        TypeError: ``records`` is not a list, or an element is not a dict, or
            ``questions``/``options``/``labels`` have wrong container types.
        ValueError: A record misses a required key or ``questions`` and
            ``options`` differ in length.
    """

    def __init__(self, records: list[dict[str, Any]]) -> None:
        if not isinstance(records, list):
            raise TypeError(
                f"records: expected a list of dicts, got {type(records).__name__}."
            )
        for i, record in enumerate(records):
            if not isinstance(record, dict):
                raise TypeError(
                    f"records[{i}]: expected a dict, got {type(record).__name__}."
                )
            missing = [key for key in _REQUIRED_KEYS if key not in record]
            if missing:
                raise ValueError(
                    f"records[{i}]: missing required key(s) {missing}; every record "
                    f"needs {list(_REQUIRED_KEYS)}."
                )
            if not isinstance(record["questions"], list):
                raise TypeError(
                    f"records[{i}].questions: expected a list[str], got "
                    f"{type(record['questions']).__name__}."
                )
            if not isinstance(record["options"], list):
                raise TypeError(
                    f"records[{i}].options: expected a list (schema C), got "
                    f"{type(record['options']).__name__}."
                )
            if not isinstance(record["labels"], list):
                raise TypeError(
                    f"records[{i}].labels: expected a list, got "
                    f"{type(record['labels']).__name__}."
                )
            if len(record["questions"]) != len(record["options"]):
                raise ValueError(
                    f"records[{i}]: {len(record['questions'])} question(s) but "
                    f"{len(record['options'])} schema(s); they must pair 1:1."
                )
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return record ``index`` unchanged (raw python structures)."""
        return self.records[index]


def decision_collate_fn(batch: Sequence[Mapping[str, Any]]) -> dict[str, list[Any]]:
    """Collate raw records into ``{key: [value_per_sample, ...]}``.

    Every top-level key becomes a list ordered like ``batch``; values stay
    raw python (str / list / dict) — no padding, no tensors. This keeps
    variable-``K`` schemas intact; padding/masking is the model/loss's job
    downstream.

    Args:
        batch: Non-empty sequence of mappings with identical key sets.

    Returns:
        Dict of lists, one entry per key.

    Raises:
        ValueError: ``batch`` is empty, or the records' key sets disagree.
    """
    if not batch:
        raise ValueError("decision_collate_fn: cannot collate an empty batch.")
    keys = set(batch[0].keys())
    for i, record in enumerate(batch[1:], start=1):
        if set(record.keys()) != keys:
            raise ValueError(
                f"decision_collate_fn: record {i} has keys {sorted(record.keys())} "
                f"but record 0 has {sorted(keys)}; all records in a batch must agree."
            )
    return {key: [record[key] for record in batch] for key in sorted(keys)}


def _model_device(model: torch.nn.Module) -> torch.device:
    """Device of the model's first parameter (``cpu`` for parameterless models)."""
    for param in model.parameters():
        return param.device
    return torch.device("cpu")


def extract_logits(output: Any) -> torch.Tensor:
    """Pull the ``[B, K]`` (or ``[B]``) logits out of a model output.

    Accepts, in order: a bare tensor; an object with a ``.logits`` attribute
    (``DecisionOutput``, HF ``ModelOutput`` subclasses); a mapping with a
    ``"logits"`` key.

    Raises:
        TypeError: None of the accepted spellings matched; the message
            describes what the output actually was.
    """
    if isinstance(output, torch.Tensor):
        return output
    logits = getattr(output, "logits", None)
    if isinstance(logits, torch.Tensor):
        return logits
    if isinstance(output, Mapping):
        value = output.get("logits")
        if isinstance(value, torch.Tensor):
            return value
    raise TypeError(
        "extract_logits: could not find logits — expected a tensor, an object "
        "with `.logits`, or a mapping with a 'logits' key; got "
        f"{type(output).__name__}."
    )


def _decision_type_of(batch: Mapping[str, Any]) -> str:
    """Read ``decision_type`` from a collated batch, checking uniformity.

    The collate stores it as a 1-element-per-sample list; a bare ``str`` is
    also accepted (hand-built batches).
    """
    if "decision_type" not in batch:
        raise ValueError(
            "batch has no 'decision_type' key; loss dispatch needs it (or pass an "
            "explicit loss_fn to make_supervised_step_fn)."
        )
    value = batch["decision_type"]
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and value:
        first = value[0]
        if any(item != first for item in value):
            raise ValueError(
                f"batch decision_type is not uniform: {list(value)}; a batch must "
                "contain a single decision type (split the loader by type)."
            )
        if not isinstance(first, str):
            raise TypeError(
                f"decision_type: expected str entries, got "
                f"{type(first).__name__}."
            )
        return first
    raise ValueError("batch decision_type is empty; cannot dispatch a loss.")


def _labels_tensor(batch: Mapping[str, Any], device: torch.device) -> torch.Tensor:
    """Tensorize ``batch['labels']`` on the model's device."""
    if "labels" not in batch:
        raise ValueError(
            "batch has no 'labels' key; supervised training needs ground truth."
        )
    labels = batch["labels"]
    if isinstance(labels, torch.Tensor):
        return labels.to(device)
    try:
        return torch.as_tensor(labels, device=device)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"labels: could not build a tensor from {type(labels).__name__} "
            f"({exc}); labels must be uniformly numeric for loss computation."
        ) from exc


#: Forward parameters that identify the shipped decision forward: a model
#: declaring all four takes a collated batch as keyword arguments.
_JEVX_FORWARD_KEYS: frozenset[str] = frozenset(
    {"state", "questions", "decision_type", "options"}
)


def _call_model(model: torch.nn.Module, batch: Mapping[str, Any]) -> Any:
    """Call ``model`` once, in the ONE call spelling its forward declares.

    Signature-driven on purpose. The obvious duck-typed spelling — try
    ``model(batch)``, fall back to ``model(**batch)`` on ``TypeError`` — is
    UNSAFE for a decision forward: its first parameter is ``state`` and a
    dict is a perfectly valid ``state`` value, so ``model(batch)`` does not
    raise, does not fail the argument count, and simply runs with the whole
    collated batch bound to the state. The failure then surfaces far
    downstream (inside the backbone's input preparation) as an error that
    never mentions the call spelling — and a model that *does* get that far
    trains on a bogus state instead of raising. The declared signature is
    the only honest dispatch.

    Branches, in order:

    1. ``forward`` declares the JevX decision keys (``state``, ``questions``, 
       ``decision_type``, ``options``) — the collated dict is splatted onto
       them, one key per parameter;
    2. ``forward`` takes a single POSITIONAL parameter (the view-wrapper
       convention: ``_CurriculumModelView``, featurized trainer stand-ins),
       or names that parameter ``batch`` — the dict is passed positionally;
    3. anything else — ``model(**batch)`` first, ``model(batch)`` retried on
       ``TypeError`` (the historical behaviour, both failures reported).

    A forward whose signature cannot be inspected (C callables, some mocks)
    falls through to branch 3, which is exactly what it did before.

    Returns:
        Whatever ``model`` returned, from whichever spelling it accepted.

    Raises:
        TypeError: branch 1 or 2 is chosen from the signature and the model
            rejects it; branch 3 with both spellings rejected (the message
            carries both errors).
    """
    try:
        parameters = [
            parameter
            for name, parameter in inspect.signature(model.forward).parameters.items()
            if name != "self"
        ]
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        parameters = None
    if parameters is not None:
        declared = {parameter.name for parameter in parameters}
        if _JEVX_FORWARD_KEYS <= declared:
            return model(**dict(batch))
        single_positional = (
            len(parameters) == 1
            and parameters[0].kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        )
        if "batch" in declared or single_positional:
            return model(batch)
    try:
        return model(**dict(batch))
    except TypeError as keyword_error:
        try:
            return model(batch)
        except TypeError as positional_error:
            raise TypeError(
                "step: model forward rejected both call spellings — "
                f"model(**batch) failed with {keyword_error!r}; "
                f"model(batch) failed with {positional_error!r}. The model must "
                "accept the collated dict positionally or as keyword arguments."
            ) from positional_error


def make_supervised_step_fn(
    model: torch.nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    *,
    return_logits: bool = False,
) -> Callable[[Mapping[str, Any]], tuple[torch.Tensor, dict[str, Any]]]:
    """Build ``step(batch) -> (loss, components)`` for supervised training.

    The returned function is what :meth:`jevx.training.trainer.HeadTrainer.fit`
    consumes by default.

    Args:
        model: Any module; called in the one spelling its ``forward``
            declares — keyword arguments for a JevX decision forward, the
            whole collated dict positionally for a view wrapper, keyword-
            first with a positional retry otherwise (see the ``_call_model`` helper).
        loss_fn: Optional custom ``loss_fn(logits, labels) -> Tensor``. When
            given it fully replaces the decision-type dispatch (the
            ``option_mask`` is NOT forwarded — fold it in inside your
            closure if needed).
        return_logits: Add the ``"logits"`` key (a detached ``[B, K]`` tensor,
            the exact tensor the loss was computed from) to the returned
            ``components``. OPT-IN and off by default, so the components a
            caller sees are unchanged. It exists so an evaluation pass can
            reuse the forward it already paid for instead of running the model
            a second time to collect metric inputs:
            :meth:`jevx.training.trainer.HeadTrainer.evaluate` reads the key
            when it is present and skips that second forward; when it is
            absent, evaluation falls back to re-running the model. The
            ``logits`` value is detached, so it never extends the autograd
            graph.

    Returns:
        ``step(batch)`` returning ``(loss, components)``: ``loss`` is a scalar
        tensor with grad enabled; ``components`` is a flat dict with at least
        ``{"loss": float, "decision_type": str, "batch_size": int}`` — ready
        to merge into a trainer history record. Labels shaped ``[B, 1]``
        (one question per record) are squeezed to ``[B]`` for every
        decision type except ``multilabel``. With ``return_logits=True`` it
        also carries ``"logits"``.

    Raises:
        TypeError: The model rejected the call spelling its signature
            selected (a decision forward is never probed positionally — a
            dict is a valid ``state=``), or neither spelling works for a
            duck-typed forward, or the output carries no logits, or a custom
            ``loss_fn`` returns a non-tensor.
        ValueError: Required batch keys are missing, the batch mixes decision
            types, or the ``decision_type`` has no registered loss.
    """

    def step(batch: Mapping[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        if not isinstance(batch, Mapping):
            raise TypeError(
                f"step: expected a mapping batch, got {type(batch).__name__}; "
                "build loaders with collate_fn=decision_collate_fn."
            )
        device = _model_device(model)
        output = _call_model(model, batch)
        logits = extract_logits(output).to(device)
        # Resolve the decision type first: it decides whether a [B, 1] label
        # column collapses to [B] (all option-wise/binary types) or stays
        # rectangular (multilabel).
        decision_type = None if loss_fn is not None else _decision_type_of(batch)
        # Labels keep their natural dtype (long for classification, float for
        # score-expectation targets); each loss validates/coerces as needed.
        labels = _labels_tensor(batch, device)
        if (
            labels.dim() == 2
            and labels.shape[1] == 1
            and decision_type != "multilabel"
        ):
            # Records store labels per question ([y]); one question per record
            # collates to [B, 1] — the losses want [B] (matches
            # RegressionHead's squeeze(-1) convention).
            labels = labels.squeeze(1)

        if loss_fn is not None:
            loss = loss_fn(logits, labels)
            if not isinstance(loss, torch.Tensor):
                raise TypeError(
                    "loss_fn must return a torch.Tensor, got "
                    f"{type(loss).__name__}."
                )
            decision_type = "custom"
            components_extra: dict[str, Any] = {}
        else:
            loss_func = _LOSS_BY_TYPE.get(decision_type)
            if loss_func is None:
                raise ValueError(
                    f"no loss registered for decision_type {decision_type!r}; "
                    f"known types: {sorted(_LOSS_BY_TYPE)}. Pass loss_fn=... to "
                    "make_supervised_step_fn to override dispatch."
                )
            option_mask = batch.get("option_mask")
            if option_mask is not None and not isinstance(option_mask, torch.Tensor):
                option_mask = torch.as_tensor(option_mask, device=device).bool()
            elif isinstance(option_mask, torch.Tensor):
                option_mask = option_mask.to(device)
            if decision_type in ("choice", "score"):
                loss = loss_func(logits, labels, option_mask)
            elif decision_type == "multilabel":
                loss = loss_func(logits, labels, option_mask)
            else:  # noul takes no option mask
                loss = loss_func(logits, labels)
            components_extra = {"option_count": int(logits.shape[-1])}

        batch_size = int(labels.shape[0]) if labels.dim() else 1
        components: dict[str, Any] = {
            "loss": float(loss.detach()),
            "decision_type": decision_type,
            "batch_size": batch_size,
        }
        components.update(components_extra)
        if return_logits:
            # The loss was computed from this exact tensor, so handing it back
            # costs a reference and saves the consumer a whole forward.
            components["logits"] = logits.detach()
        return loss, components

    return step


# ---------------------------------------------------------------------------
# Metrics (pure torch, no sklearn)
# ---------------------------------------------------------------------------


def _predictions(logits: torch.Tensor) -> torch.Tensor:
    """Class predictions ``[B]`` from logits: argmax when ``K >= 2``,
    threshold at 0 for a single score column (Noul)."""
    if logits.dim() == 2 and logits.shape[1] >= 2:
        return logits.argmax(dim=-1)
    squeezed = logits.squeeze(-1) if logits.dim() == 2 else logits
    if squeezed.dim() != 1:
        raise ValueError(
            f"logits: expected [B, K] or [B], got shape {tuple(logits.shape)}."
        )
    return (squeezed >= 0).long()


def _labels_1d(labels: torch.Tensor | Any, batch_size: int, name: str) -> torch.Tensor:
    """Normalize metric targets to integer ``[B]``.

    Collated records carry one label per question (``[B, 1]`` after
    ``as_tensor``); metrics need ``[B]`` — without this squeeze,
    ``pred == labels`` silently broadcasts to ``[B, B]`` and every metric
    becomes garbage.
    """
    labels = torch.as_tensor(labels) if not isinstance(labels, torch.Tensor) else labels
    if labels.dim() == 2 and labels.shape[1] == 1:
        labels = labels.squeeze(1)
    if labels.dim() != 1:
        raise ValueError(
            f"{name}: expected integer targets [B] (or [B, 1]), got shape "
            f"{tuple(labels.shape)}; per-label multilabel targets are not "
            "supported by these classification metrics."
        )
    if labels.shape[0] != batch_size:
        raise ValueError(
            f"{name}: expected {batch_size} entries to match logits, got "
            f"{labels.shape[0]}."
        )
    return labels.long()


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Classification accuracy: ``mean(1[argmax logits == label])``.

    Args:
        logits: Raw scores ``[B, K]`` (K >= 2) or a single score ``[B]``/
            ``[B, 1]`` (thresholded at 0, matching the sigmoid at 0.5).
        labels: Integer targets ``[B]``.

    Returns:
        Python float in ``[0, 1]``.
    """
    labels = _labels_1d(labels, logits.shape[0], "accuracy.labels")
    return float((_predictions(logits) == labels).float().mean())


def macro_f1(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Macro-averaged F1 over the classes present in ``labels``.

    Each class ``c`` contributes ``F1_c = 2·P·R / (P + R)`` (0 when
    ``P + R == 0``); the macro average is the unweighted mean over the
    classes in ``labels`` — classes absent from the ground truth are not
    invented (they would be undefined without a declared label space).

    Args:
        logits: Raw scores ``[B, K]`` or ``[B]``/``[B, 1]``.
        labels: Integer targets ``[B]``.

    Returns:
        Python float in ``[0, 1]``.
    """
    labels = _labels_1d(labels, logits.shape[0], "macro_f1.labels")
    pred = _predictions(logits)
    classes = labels.unique()
    if classes.numel() == 0:
        raise ValueError("macro_f1: cannot compute over zero labels.")
    f1s: list[float] = []
    for cls in classes.tolist():
        tp = int(((pred == cls) & (labels == cls)).sum())
        fp = int(((pred == cls) & (labels != cls)).sum())
        fn = int(((pred != cls) & (labels == cls)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        f1s.append(f1)
    return float(sum(f1s) / len(f1s))


def confusion_matrix(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Confusion matrix ``[C, C]``: rows = true class, cols = predicted.

    ``C = max(labels) + 1`` (labels define the observed class set; extra
    predicted-only columns beyond that width widen the matrix so no predicted
    class is dropped).

    Args:
        logits: Raw scores ``[B, K]`` or ``[B]``/``[B, 1]``.
        labels: Integer targets ``[B]``.

    Returns:
        ``torch.long`` tensor ``[C, C]``.
    """
    labels = _labels_1d(labels, logits.shape[0], "confusion_matrix.labels")
    pred = _predictions(logits)
    num_classes = int(max(int(labels.max()), int(pred.max()))) + 1
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for true, guess in zip(labels.tolist(), pred.tolist()):
        matrix[true, guess] += 1
    return matrix
