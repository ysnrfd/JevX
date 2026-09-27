# Copyright (c) 2026 ysnrfd
"""Task-level conditioning gate: conditioned vs state-only scoring.

Motivation (EXP-H, ``benchmarks/run_head_quality.py``): the option/schema
pathway is a strict superset of a state-only linear probe, so it can only help
when the option text carries information the state does not already encode. On
frozen features where it does not (RAVDESS: bypass 0.2407 vs full stack 0.1353
pre-fix) the option pathway *costs* accuracy while still helping NLL. The gate
decides, **per task** (never per sample — a per-sample switch was explicitly
rejected as an overfitting risk), which of the two scoring modes to use.

Measurement
-----------
``ΔNLL = NLL_conditioned − NLL_state_only`` on ONE split, with the SAME
weights, the SAME batches (materialized once, see below), the SAME conditions,
in ``eval()`` mode under ``torch.no_grad()``. The only difference between the
two passes is the ``condition_options`` flag passed to
:meth:`jevx.modeling_jevx.JevXForDecision.forward` (``True`` = option
pathway on, ``False`` = state-only scoring).

* **Loss**: :mod:`jevx.training.losses` — ``choice_loss`` / ``score_loss``
  (classification mode) for option-wise types, ``noul_loss`` for Noul,
  ``multilabel_loss`` for MultiLabel. A caller-supplied ``loss_fn`` overrides
  the dispatch (contract below).
* **Unit**: NLL is **per example**, not per token. A decision example is one
  ``(state, question)`` pair scored by one distribution over its ``K`` valid
  options, so the per-example value is the masked negative log-likelihood of
  the gold option. There is no token axis in the decision objective.
* **Aggregation**: within a batch, the example-weighted mean
  (``sum(nll_i)/n``); across batches, the example-weighted mean again
  (``sum(n_b · nll_b)/sum(n_b)``), so every example counts once. Per-batch
  values are kept for the τ protocol.

Decision rule
-------------
``rule="state_only_if_delta_nll_exceeds"``: if ``ΔNLL > τ + tie_tolerance`` the
option pathway is judged not useful for this task and scoring becomes
``state_only``; otherwise scoring stays ``conditioned``.

τ protocol (independent of the reported evaluation)
---------------------------------------------------
τ comes from the calibration split ALONE, via the one-standard-error rule::

    d_b     = mean NLL_conditioned − mean NLL_state_only   (batch b)
    SE      = std(d_b, ddof=1) / sqrt(B)                    (B calibration batches)
    τ       = z · SE                                         (z = 1.0 by default)
    switch  ⟺  ΔNLL > τ + tie_tolerance

Rationale: under the null hypothesis "conditioning changes nothing", the
per-batch ``d_b`` are exchangeable draws with mean ≈ 0 and standard error
``SE``. Demanding ``ΔNLL > z·SE`` before declaring the option pathway harmful is
a one-sided test at ``z`` standard errors: with ``z = 1`` an effect that is
smaller than the measurement noise of its own calibration split NEVER switches
the task to ``state_only``. The rule is deterministic (no sampling, no grid
search, no tuning against any evaluation number) and needs no held-out
evaluation data, so there is no test leakage by construction.

Assumptions (all of them, stated because the rule is only as good as they are):

1. Calibration BATCHES are exchangeable and roughly independent draws from the
   task's example distribution; ``SE`` is the ordinary i.i.d. standard error of
   the batch mean. Correlated batches (e.g. sorted-by-label loaders) inflate
   the true uncertainty — batch size is the knob that controls it.
2. The two passes are side-effect free: eval mode + ``no_grad`` + a materialized
   batch list, so a repeated ``decide`` on the same loader and weights returns
   bit-identical numbers.
3. The decision is TASK-level. The measured effect is assumed to be a property
   of the task, not of individual examples; no per-sample switching is
   performed, because a per-sample rule fitted on one split is exactly the
   overfitting failure this module exists to avoid.
4. Equal batch sizes make the unweighted batch mean and the example-weighted
   mean coincide; with unequal batch sizes the unweighted ``SE`` is an
   approximation (the mean is still example-weighted).

No-test-leakage guard
---------------------
The threshold is a property of the calibration protocol, so it must never be
fitted on the split that is reported. :func:`assert_split_not_leakage` raises
``ValueError`` for any split name containing a test-like token — the tokens of
:data:`LEAKY_SPLIT_TOKENS` (``test``, ``tests``, ``testing``, ``testset``,
``testsplit``, ``holdout``, ``eval``, ``evals``, ``evaluation``, ``final``).
Matching is by whole token, so ``RAVDESS_test`` and ``test`` are rejected while
``contest`` is not; the token set is overridable per gate
(``ConditioningGateConfig.forbidden_split_tokens``). The same guard runs in
:meth:`ConditioningGate.measure` and :meth:`ConditioningGate.decide`, and at
config construction.

Usage
-----
::

    gate = ConditioningGate(model, ConditioningGateConfig(split_name="calibration"))
    decision = gate.decide(calibration_loader)      # "conditioned" | "state_only"
    flag = gate.condition_options_for(model)        # pass to forward(condition_options=)
    apply_gate(model, gate)                         # -> bool, sets config.conditioning_enabled

Interaction with training / curriculum
--------------------------------------
The gate is applied AFTER the decision head has been trained (Stage B). A
curriculum that trains the option pathway must run with conditioning ON —
``conditioning_enabled=True`` — because the shipped head is a superset: a head
trained state-only never learns an option interaction for the gate to keep.
The gate's verdict is then a statement about the *trained* model on a
calibration split, and flipping ``conditioning_enabled`` afterwards changes
**only the option pathway** at inference: the state pathway, the projections and
every learned weight are untouched, so no retraining is implied. If a curriculum
itself toggles ``conditioning_enabled`` between stages, call
:meth:`ConditioningGate.decide` after the final stage (or re-decide after any
curriculum-driven flip) so the flag that training leaves behind and the flag the
gate reports cannot disagree.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch

from jevx.training.losses import (
    choice_loss,
    multilabel_loss,
    noul_loss,
    score_loss,
)

__all__ = [
    "ConditioningGate",
    "ConditioningGateConfig",
    "GateMeasurement",
    "TauFit",
    "apply_gate",
    "assert_split_not_leakage",
    "fit_tau",
    "measure_delta_nll",
    "measure_modes",
]

#: Decision types the built-in loss dispatch understands.
SUPPORTED_DECISION_TYPES: tuple[str, ...] = (
    "choice",
    "score",
    "noul",
    "multilabel",
)

#: Split-name tokens that name a REPORTED EVALUATION split. The gate refuses to
#: measure or threshold on any of them (see the module docstring).
LEAKY_SPLIT_TOKENS: tuple[str, ...] = (
    "test",
    "tests",
    "testing",
    "testset",
    "testsplit",
    "holdout",
    "eval",
    "evals",
    "evaluation",
    "final",
)

#: Human-readable name of the τ protocol recorded in ``tau_source``.
TAU_PROTOCOL_NAME = "one_se"


# ---------------------------------------------------------------------------
# Split guard
# ---------------------------------------------------------------------------


def _split_tokens(split_name: str) -> tuple[str, ...]:
    """Lower-case alphanumeric tokens of ``split_name`` (order preserved)."""
    lowered = split_name.lower()
    current: list[str] = []
    tokens: list[str] = []
    for char in lowered:
        if char.isalnum():
            current.append(char)
        else:
            if current:
                tokens.append("".join(current))
                current = []
    if current:
        tokens.append("".join(current))
    return tuple(tokens)


def assert_split_not_leakage(
    split_name: str,
    *,
    forbidden_tokens: Sequence[str] = LEAKY_SPLIT_TOKENS,
) -> None:
    """Reject a TEST split name (no-test-leakage policy).

    Args:
        split_name: The split the caller is about to measure / fit τ on.
        forbidden_tokens: Tokens that mark a reported evaluation split;
            defaults to :data:`LEAKY_SPLIT_TOKENS`.

    Raises:
        TypeError: ``split_name`` is not a string, or ``forbidden_tokens`` is
            not a sequence of strings.
        ValueError: ``split_name`` carries a forbidden token (case-insensitive,
            whole-token match: ``"RAVDESS_test"`` and ``"test"`` are rejected,
            ``"contest"`` is not). The message states the no-test-leakage rule.
    """
    if not isinstance(split_name, str):
        raise TypeError(
            f"split_name: expected a str, got {type(split_name).__name__}."
        )
    if isinstance(forbidden_tokens, str) or not isinstance(
        forbidden_tokens, (list, tuple, frozenset, set)
    ):
        raise TypeError(
            "forbidden_tokens: expected a sequence of str tokens, got "
            f"{type(forbidden_tokens).__name__}."
        )
    tokens = _split_tokens(split_name)
    forbidden = {str(token).lower() for token in forbidden_tokens}
    hits = sorted({token for token in tokens if token in forbidden})
    if hits:
        raise ValueError(
            f"split_name={split_name!r} names a TEST/EVALUATION split (token(s) {hits}) — "
            "refusing to measure ΔNLL or fit tau on it: the threshold would then be tuned "
            "on the reported evaluation data (no-test-leakage policy). Point split_name at "
            "a calibration/validation split that is held out from the reported evaluation."
        )


# ---------------------------------------------------------------------------
# τ protocol
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TauFit:
    """Result of the one-standard-error threshold fit.

    Attributes:
        tau: The threshold, ``z · SE`` (>= 0).
        standard_error: ``std(d_b, ddof=1) / sqrt(B)`` over calibration batches.
        z: Multiplier applied to the standard error.
        num_batches: Number of calibration batches the fit saw.
        source: Machine-readable provenance string, e.g.
            ``"one_se(z=1.0, se=0.0123, batches=8)"``.
    """

    tau: float
    standard_error: float
    z: float
    num_batches: int
    source: str


def fit_tau(
    per_batch_delta: Sequence[float],
    *,
    z: float = 1.0,
    min_batches: int = 2,
) -> TauFit:
    """Fit τ by the one-standard-error rule on calibration batches.

    ``τ = z · SE``, where ``SE = std(d_b, ddof=1) / sqrt(B)`` is the standard
    error of the mean of the per-batch ΔNLL values ``d_b`` (see the module
    docstring for the full protocol, its rationale and its assumptions).

    With fewer than ``min_batches`` batches the standard error is not estimable
    from between-batch spread; it is then set to ``0.0`` and ``source`` says so
    (``"one_se(z=..., se=0.0, batches=1, se_unestimable)"``), which makes the
    rule degenerate to "any positive ΔNLL switches" — callers that need a real
    noise floor should supply more batches (or a caller-set ``tau``).

    The fit is deterministic: pure arithmetic over the given values, no
    sampling and no search over evaluation results.

    Args:
        per_batch_delta: Per-batch ``NLL_conditioned − NLL_state_only``.
        z: Standard-error multiplier; ``1.0`` (one standard error) by default.
        min_batches: Batch count below which the standard error is declared
            unestimable.

    Returns:
        A :class:`TauFit`.

    Raises:
        TypeError: A non-sequence argument, or a non-numeric entry.
        ValueError: ``per_batch_delta`` is empty, holds a non-finite value, or
            ``z`` / ``min_batches`` are invalid.
    """
    if isinstance(per_batch_delta, (str, bytes)) or not isinstance(
        per_batch_delta, (list, tuple)
    ):
        raise TypeError(
            "per_batch_delta: expected a sequence of floats, got "
            f"{type(per_batch_delta).__name__}."
        )
    if len(per_batch_delta) == 0:
        raise ValueError(
            "per_batch_delta: empty — the standard error of the calibration mean is "
            "undefined without at least one batch."
        )
    values: list[float] = []
    for i, value in enumerate(per_batch_delta):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"per_batch_delta[{i}]: expected a float, got {type(value).__name__}."
            )
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(
                f"per_batch_delta[{i}]: expected a finite value, got {value!r}."
            )
        values.append(value)
    if isinstance(z, bool) or not isinstance(z, (int, float)):
        raise TypeError(f"z: expected a float, got {type(z).__name__}.")
    z = float(z)
    if not math.isfinite(z) or z < 0.0:
        raise ValueError(f"z: expected a finite value >= 0, got {z!r}.")
    if isinstance(min_batches, bool) or not isinstance(min_batches, int):
        raise TypeError(
            f"min_batches: expected an int, got {type(min_batches).__name__}."
        )
    if min_batches < 1:
        raise ValueError(f"min_batches: expected an int >= 1, got {min_batches}.")

    num_batches = len(values)
    mean = math.fsum(values) / num_batches
    if num_batches >= min_batches:
        variance = math.fsum((value - mean) ** 2 for value in values) / (
            num_batches - 1
        )
        standard_error = math.sqrt(variance) / math.sqrt(num_batches)
        note = ""
    else:
        standard_error = 0.0
        note = ", se_unestimable"
    tau = z * standard_error
    source = (
        f"{TAU_PROTOCOL_NAME}(z={z:g}, se={standard_error:.6g}, "
        f"batches={num_batches}{note})"
    )
    return TauFit(
        tau=tau,
        standard_error=standard_error,
        z=z,
        num_batches=num_batches,
        source=source,
    )


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateMeasurement:
    """Everything one paired measurement produced.

    Attributes:
        delta_nll: ``NLL_conditioned − NLL_state_only``, example-weighted over
            the whole split (positive ⇒ conditioning made the NLL worse).
        conditioned_nll / state_only_nll: The two example-weighted NLLs.
        conditioned_accuracy / state_only_accuracy: Example-weighted top-1
            (option-wise and Noul) or per-label (MultiLabel) accuracy of the
            two modes on the same split.
        per_batch_delta: Per-batch ΔNLL, in loader order — the input to
            :func:`fit_tau`.
        batch_sizes: Example count per batch, in loader order.
        num_samples: Total examples measured.
        num_batches: Total batches measured.
        decision_type: The decision type the loss was dispatched on.
    """

    delta_nll: float
    conditioned_nll: float
    state_only_nll: float
    conditioned_accuracy: float
    state_only_accuracy: float
    per_batch_delta: tuple[float, ...]
    batch_sizes: tuple[int, ...]
    num_samples: int
    num_batches: int
    decision_type: str


def _as_list(value: Any, name: str) -> list[Any]:
    """Coerce a per-sample container to a list, rejecting scalars."""
    if isinstance(value, (str, bytes)) or isinstance(value, Mapping):
        raise TypeError(
            f"{name}: expected a sequence of per-sample values, got "
            f"{type(value).__name__}."
        )
    if not isinstance(value, (list, tuple)):
        raise TypeError(
            f"{name}: expected a list/tuple of per-sample values, got "
            f"{type(value).__name__}."
        )
    return list(value)


def _question_text(question: Any, index: int) -> str:
    """Read a question's text (``str`` or ``{"text": ...}``)."""
    if isinstance(question, str):
        return question
    if isinstance(question, Mapping) and isinstance(question.get("text"), str):
        return str(question["text"])
    raise TypeError(
        f"questions[{index}]: expected a str or a mapping with a 'text' key, got "
        f"{type(question).__name__}."
    )


@dataclass(frozen=True)
class _Group:
    """One ``model(...)`` call's worth of work inside a collated batch.

    ``JevXForDecision.forward`` takes ``m`` questions, each with ONE schema
    shared across the whole state batch, so records are grouped by
    ``(question text, schema)`` — two records asking the same question with
    different option sets cannot share a call, and pretending they can would
    silently score one of them against the other's options.
    """

    question_texts: tuple[str, ...]
    schema: list[Any]
    states: list[Any]
    targets: list[list[int]]


def _group_batch(batch: Mapping[str, Any]) -> list[_Group]:
    """Split a collated batch into forward-call groups (one per question set).

    Records are grouped by their ``(question text, schema)`` key in
    first-appearance order, which keeps the output order ``b·m + j``
    (state-major) inside every group.
    """
    for key in ("state", "questions", "options", "labels"):
        if key not in batch:
            raise ValueError(
                f"batch has no {key!r} key; the gate needs the collated decision "
                "records (state/questions/options/labels) — build loaders with "
                "collate_fn=jevx.training.decision_collate_fn."
            )
    states = _as_list(batch["state"], "batch['state']")
    questions = _as_list(batch["questions"], "batch['questions']")
    options = _as_list(batch["options"], "batch['options']")
    labels = _as_list(batch["labels"], "batch['labels']")
    size = len(states)
    for name, values in (
        ("questions", questions),
        ("options", options),
        ("labels", labels),
    ):
        if len(values) != size:
            raise ValueError(
                f"batch['{name}'] has {len(values)} entries but batch['state'] has "
                f"{size}; a collated batch must have one entry per sample."
            )

    groups: dict[tuple[Any, ...], _Group] = {}
    for row, state in enumerate(states):
        row_questions = _as_list(questions[row], f"batch['questions'][{row}]")
        row_options = _as_list(options[row], f"batch['options'][{row}]")
        row_labels = _as_list(labels[row], f"batch['labels'][{row}]")
        texts = tuple(
            _question_text(question, j) for j, question in enumerate(row_questions)
        )
        if len(texts) == 0:
            raise ValueError(
                f"batch['questions'][{row}] is empty; a decision needs m >= 1 question."
            )
        if len(row_options) != len(texts) or len(row_labels) != len(texts):
            raise ValueError(
                f"sample {row}: {len(texts)} question(s) but {len(row_options)} "
                f"schema(s) and {len(row_labels)} label(s); they must pair 1:1."
            )
        # repr() of the schema is the grouping key: options may be str/dict/
        # number, and only exact equality may share a call.
        key = tuple(
            (texts[j], repr(row_options[j])) for j in range(len(texts))
        )
        group = groups.get(key)
        if group is None:
            group = _Group(
                question_texts=texts,
                schema=list(row_options),
                states=[],
                targets=[],
            )
            groups[key] = group
        group.states.append(state)
        group.targets.append([int(label) for label in row_labels])
    return list(groups.values())


def _valid_width(decision: Any) -> int | None:
    """Number of VALID options behind a decision's logits, or ``None``.

    The modeling layer pads the option axis on the right and records the
    schema's real width in ``schema_metadata["option_keys"]``; the mask-aware
    probabilities (exactly 0 at invalid positions) are the fallback.
    """
    metadata = getattr(decision, "schema_metadata", None)
    if isinstance(metadata, Mapping):
        keys = metadata.get("option_keys")
        if isinstance(keys, (list, tuple)) and keys:
            return len(keys)
    probabilities = getattr(decision, "probabilities", None)
    if isinstance(probabilities, torch.Tensor) and probabilities.numel():
        return int((probabilities.reshape(-1) > 0).sum())
    return None


def _row_nll_and_hit(
    decision: Any,
    target: int | Sequence[int],
    decision_type: str,
    loss_fn: Callable[..., torch.Tensor] | None,
) -> tuple[float, float]:
    """Per-example NLL and (fractionally) correct score for ONE decision."""
    logits = getattr(decision, "logits", None)
    if not isinstance(logits, torch.Tensor):
        raise TypeError(
            "decision: expected a DecisionOutput carrying a `logits` tensor, got "
            f"{type(decision).__name__}; the gate reads the head logits, not "
            "probabilities, so it can use the task's own masked loss."
        )
    row = logits.detach()
    if row.dim() == 1:
        row = row.unsqueeze(0)  # [1, K] or [1]
    row = row.to(dtype=torch.float32)
    if decision_type == "noul":
        target_value = torch.as_tensor(target, dtype=torch.float32).reshape(1)
        if loss_fn is not None:
            per_sample = loss_fn(row, target_value)
        else:
            per_sample = noul_loss(row, target_value, reduction="none")
        nll = float(per_sample.detach().reshape(-1)[0])
        correct = float((row.reshape(-1)[0] >= 0.0) == bool(int(target_value[0])))
        return nll, correct
    if decision_type == "multilabel":
        target_tensor = torch.as_tensor(target, dtype=torch.float32).reshape(
            1, -1
        )
        if loss_fn is not None:
            per_example = loss_fn(row, target_tensor, None)
        else:
            per_example = multilabel_loss(row, target_tensor, None, reduction="mean")
        per_example = torch.as_tensor(per_example).detach().reshape(-1)
        if per_example.numel() != 1:
            raise TypeError(
                "loss_fn: a per-example loss for multilabel must return one scalar "
                f"per example, got {int(per_example.numel())}."
            )
        with torch.no_grad():
            hits = ((row >= 0.0).to(target_tensor.dtype) == target_tensor).to(
                torch.float32
            )
        return float(per_example[0]), float(hits.mean())

    width = _valid_width(decision)
    logits_width = int(row.shape[-1])
    if width is None or width <= 0:
        width = logits_width
    if width > logits_width:
        raise ValueError(
            f"decision: schema reports {width} valid option(s) but the logit width is "
            f"{logits_width}; the option mask cannot be reconstructed."
        )
    mask = torch.zeros(1, logits_width, dtype=torch.bool)
    mask[0, :width] = True
    target_tensor = torch.as_tensor([int(target)], dtype=torch.long)
    if loss_fn is not None:
        per_sample = loss_fn(row, target_tensor, mask)
    elif decision_type == "score":
        per_sample = score_loss(
            row, target_tensor, mask, mode="classification", reduction="none"
        )
    else:
        per_sample = choice_loss(row, target_tensor, mask, reduction="none")
    per_sample = torch.as_tensor(per_sample).detach().reshape(-1)
    if per_sample.numel() != 1:
        raise TypeError(
            "loss_fn: must return ONE loss per example (reduction='none'), got "
            f"{int(per_sample.numel())}."
        )
    with torch.no_grad():
        masked = row.masked_fill(~mask, torch.finfo(row.dtype).min)
        prediction = int(masked.argmax(dim=-1)[0])
    return float(per_sample[0]), float(prediction == int(target))


def _supports_condition_options(model: Any) -> bool:
    """Whether ``model.forward`` can accept ``condition_options=``."""
    forward = getattr(model, "forward", None)
    if forward is None:
        return False
    try:
        signature = inspect.signature(forward)
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return True
    parameters = signature.parameters
    if "condition_options" in parameters:
        return True
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def _require_condition_options(model: Any) -> None:
    """Fail loudly (and actionably) when the flag is not on the forward."""
    if _supports_condition_options(model):
        return
    raise TypeError(
        f"{type(model).__name__}.forward does not accept `condition_options=`; the gate "
        "needs JevXForDecision.forward(..., condition_options: Optional[bool] = None) "
        "to switch the option pathway for a single call. Either pass a model that "
        "exposes it, or set model.config.conditioning_enabled directly and measure the "
        "two modes with two models."
    )


def _materialize(loader: Iterable[Any], max_batches: int | None) -> list[Any]:
    """Drain ``loader`` ONCE so both passes see identical batches."""
    if max_batches is not None:
        if isinstance(max_batches, bool) or not isinstance(max_batches, int):
            raise TypeError(
                f"max_batches: expected an int or None, got {type(max_batches).__name__}."
            )
        if max_batches < 1:
            raise ValueError(f"max_batches: expected an int >= 1, got {max_batches}.")
    batches: list[Any] = []
    for batch in loader:
        batches.append(batch)
        if max_batches is not None and len(batches) >= max_batches:
            break
    return batches


def _check_decision_type(batch: Any, decision_type: str) -> None:
    """Reject a batch whose own decision type disagrees with the argument."""
    if not isinstance(batch, Mapping):
        raise TypeError(
            f"batch: expected a mapping (build loaders with "
            f"collate_fn=jevx.training.decision_collate_fn), got {type(batch).__name__}."
        )
    value = batch.get("decision_type")
    if value is None:
        return
    types = [value] if isinstance(value, str) else _as_list(value, "decision_type")
    seen = {str(item) for item in types}
    if len(seen) > 1:
        raise ValueError(
            f"batch decision_type is not uniform: {sorted(seen)}; split the loader by "
            "decision type before measuring."
        )
    (only,) = seen
    if only != decision_type:
        raise ValueError(
            f"batch decision_type={only!r} does not match decision_type="
            f"{decision_type!r} passed to the gate; measure each decision type on its "
            "own loader (the gate is per task AND per decision type)."
        )


def _pass_over(
    model: Any,
    batches: Sequence[Mapping[str, Any]],
    *,
    condition_options: bool,
    decision_type: str,
    loss_fn: Callable[..., torch.Tensor] | None,
) -> list[tuple[list[float], list[float], int]]:
    """One evaluation pass; returns ``[(nll_i), (hit_i), n]`` per batch."""
    per_batch: list[tuple[list[float], list[float], int]] = []
    for batch in batches:
        _check_decision_type(batch, decision_type)
        nlls: list[float] = []
        hits: list[float] = []
        for group in _group_batch(batch):
            questions_in_call = len(group.question_texts)
            output = model(
                state=list(group.states),
                questions=list(group.question_texts),
                options=[list(schema) for schema in group.schema],
                decision_type=decision_type,
                condition_options=condition_options,
            )
            decisions = getattr(output, "decisions", None)
            if decisions is None or isinstance(decisions, torch.Tensor):
                raise TypeError(
                    "model output: expected a JevXOutput with a list of "
                    "DecisionOutput in `decisions`, got "
                    f"{type(decisions).__name__}; the gate needs one entry per "
                    "(state, question) pair."
                )
            decisions = list(decisions)
            expected = len(group.states) * questions_in_call
            if len(decisions) != expected:
                raise ValueError(
                    f"model output: `decisions` has {len(decisions)} entries but the call "
                    f"asked {expected} (B={len(group.states)} states x "
                    f"m={questions_in_call} questions); the gate reads the "
                    "state-major b*m+j layout."
                )
            for local in range(len(group.states)):
                for j, target in enumerate(group.targets[local]):
                    # decisions are state-major over the states PASSED IN, so the
                    # pair index is the position inside this group, not in the batch.
                    pair = local * questions_in_call + j
                    nll, hit = _row_nll_and_hit(
                        decisions[pair], target, decision_type, loss_fn
                    )
                    nlls.append(nll)
                    hits.append(hit)
        per_batch.append((nlls, hits, len(nlls)))
    return per_batch


def _weighted_mean(values: Sequence[float], weights: Sequence[int]) -> float:
    """``sum(w_i v_i)/sum(w_i)`` — every example counts exactly once."""
    total_weight = sum(weights)
    if total_weight == 0:
        raise ValueError(
            "measurement: the split produced zero examples; a mean over an empty set is "
            "undefined."
        )
    return math.fsum(w * v for v, w in zip(values, weights)) / total_weight


class _EvalMode:
    """Context manager: ``model.eval()`` + ``torch.no_grad()``, mode restored."""

    def __init__(self, model: Any) -> None:
        self._model = model
        self._was_training: bool | None = None

    def __enter__(self) -> "_EvalMode":
        self._was_training = bool(getattr(self._model, "training", False))
        if hasattr(self._model, "eval"):
            self._model.eval()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._was_training and hasattr(self._model, "train"):
            self._model.train()
        return None


def measure_modes(
    model: Any,
    loader: Iterable[Any],
    *,
    loss_fn: Callable[..., torch.Tensor] | None = None,
    decision_type: str = "choice",
    split_name: str | None = "calibration",
    forbidden_split_tokens: Sequence[str] = LEAKY_SPLIT_TOKENS,
    max_batches: int | None = None,
) -> GateMeasurement:
    """Measure both scoring modes on ONE split and return the paired result.

    The loader is drained ONCE and the resulting batch list is replayed for
    ``condition_options=True`` and then ``condition_options=False``, so both
    passes see the same partitioning in the same order (no shuffle drift) with
    the same weights, in eval mode under ``no_grad``. ``Delta NLL =
    NLL_conditioned − NLL_state_only``; the aggregation is documented in the
    module docstring (per example, then example-weighted over batches).

    Args:
        model: A model whose ``forward`` accepts ``state=``, ``questions=``,
            ``options=``, ``decision_type=`` and ``condition_options=``
            (``JevXForDecision`` after the conditioning flag landed).
        loader: Any iterable of collated decision batches.
        loss_fn: Optional ``loss_fn(logits, target, option_mask)`` (and
            ``loss_fn(logits, target)`` for Noul) returning ONE loss per
            example. Replaces the :mod:`jevx.training.losses` dispatch.
        decision_type: ``"choice"`` (default), ``"score"``, ``"noul"`` or
            ``"multilabel"``.
        split_name: Name of the split being measured; checked against the
            no-test-leakage guard. ``None`` skips the check.
        forbidden_split_tokens: Overrides :data:`LEAKY_SPLIT_TOKENS`.
        max_batches: Measure only the first N batches (smoke runs).

    Returns:
        A :class:`GateMeasurement`.

    Raises:
        TypeError: The model does not accept ``condition_options``, a batch is
            not a mapping, or ``loss_fn`` misbehaves.
        ValueError: A test split was named, the decision type is unknown, a
            batch disagrees with ``decision_type``, or the split is empty.
    """
    if decision_type not in SUPPORTED_DECISION_TYPES:
        raise ValueError(
            f"decision_type: expected one of {list(SUPPORTED_DECISION_TYPES)}, got "
            f"{decision_type!r}."
        )
    if split_name is not None:
        assert_split_not_leakage(split_name, forbidden_tokens=forbidden_split_tokens)
    if loss_fn is not None and not callable(loss_fn):
        raise TypeError(
            f"loss_fn: expected a callable, got {type(loss_fn).__name__}."
        )
    _require_condition_options(model)

    batches = _materialize(loader, max_batches)
    if not batches:
        raise ValueError(
            "measure_modes: the loader yielded no batches; a gate decision needs at "
            "least one calibration batch."
        )
    for batch in batches:
        _check_decision_type(batch, decision_type)

    with _EvalMode(model), torch.no_grad():
        conditioned = _pass_over(
            model,
            batches,
            condition_options=True,
            decision_type=decision_type,
            loss_fn=loss_fn,
        )
        state_only = _pass_over(
            model,
            batches,
            condition_options=False,
            decision_type=decision_type,
            loss_fn=loss_fn,
        )

    sizes = [size for _, _, size in conditioned]
    if any(size != other for size, (_, _, other) in zip(sizes, state_only)):
        raise RuntimeError(  # pragma: no cover - defensive, batches are shared
            "measure_modes: the two passes disagree on the number of examples; the "
            "loader must be a materialized list shared by both passes."
        )
    if sum(sizes) == 0:
        raise ValueError(
            "measure_modes: the calibration split produced zero decision examples."
        )

    conditioned_nlls = [sum(nlls) / size for nlls, _, size in conditioned]
    state_only_nlls = [sum(nlls) / size for nlls, _, size in state_only]
    per_batch_delta = tuple(
        cond - plain for cond, plain in zip(conditioned_nlls, state_only_nlls)
    )
    conditioned_nll = _weighted_mean(conditioned_nlls, sizes)
    state_only_nll = _weighted_mean(state_only_nlls, sizes)
    return GateMeasurement(
        delta_nll=conditioned_nll - state_only_nll,
        conditioned_nll=conditioned_nll,
        state_only_nll=state_only_nll,
        conditioned_accuracy=_weighted_mean(
            [sum(hits) / size for _, hits, size in conditioned], sizes
        ),
        state_only_accuracy=_weighted_mean(
            [sum(hits) / size for _, hits, size in state_only], sizes
        ),
        per_batch_delta=per_batch_delta,
        batch_sizes=tuple(sizes),
        num_samples=sum(sizes),
        num_batches=len(sizes),
        decision_type=decision_type,
    )


def measure_delta_nll(
    model: Any,
    loader: Iterable[Any],
    *,
    loss_fn: Callable[..., torch.Tensor] | None = None,
    decision_type: str = "choice",
    split_name: str | None = "calibration",
    forbidden_split_tokens: Sequence[str] = LEAKY_SPLIT_TOKENS,
    max_batches: int | None = None,
) -> float:
    """``ΔNLL = NLL_conditioned − NLL_state_only`` on one split.

    Two evaluation passes over the SAME materialized batches, same weights,
    same conditions, eval mode + ``no_grad``; see :func:`measure_modes` for the
    loss/aggregation contract and the module docstring for the protocol.

    Args:
        model: Model accepting ``forward(..., condition_options=...)``.
        loader: Iterable of collated decision batches.
        loss_fn: Optional per-example loss override.
        decision_type: Loss dispatch type (default ``"choice"``).
        split_name: Split name checked by the no-test-leakage guard.
        forbidden_split_tokens: Overrides :data:`LEAKY_SPLIT_TOKENS`.
        max_batches: Measure only the first N batches.

    Returns:
        The example-weighted NLL difference (positive = conditioning hurt).

    Raises:
        TypeError / ValueError: As :func:`measure_modes`.
    """
    return measure_modes(
        model,
        loader,
        loss_fn=loss_fn,
        decision_type=decision_type,
        split_name=split_name,
        forbidden_split_tokens=forbidden_split_tokens,
        max_batches=max_batches,
    ).delta_nll


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ConditioningGateConfig:
    """Gate configuration, including its own verdict.

    Attributes:
        tau: Threshold on ΔNLL, ``>= 0``. ``None`` (default) means "use the
            one-standard-error protocol" (:func:`fit_tau`). A float given HERE
            is authoritative, used verbatim and reported with
            ``tau_source="explicit"`` — use it only for a threshold justified
            outside any evaluation split. The field doubles as the
            "threshold in force" report: after :meth:`ConditioningGate.decide`
            it holds the fitted value; use :meth:`ConditioningGate.set_tau` to
            pin or un-pin.
        rule: Decision rule name. Only ``"state_only_if_delta_nll_exceeds"``
            is implemented.
        split_name: Name of the calibration/validation split. Any name carrying
            a test-like token is rejected (no-test-leakage).
        min_samples: Minimum number of calibration EXAMPLES required to let the
            gate switch. Below it the measurement is too small to justify a
            task-level behaviour change, so the gate keeps ``conditioned`` and
            says so in ``tau_source``.
        tie_tolerance: Slack around τ. A ``|ΔNLL − τ| <= tie_tolerance``
        reading is a tie and resolves to ``conditioned`` (do not change the
            shipped behaviour on a coin flip).
        decision: ``"pending"`` until :meth:`ConditioningGate.decide` runs,
            then ``"state_only"`` or ``"conditioned"``.
        delta_nll: The measured ΔNLL (``None`` before a measurement).
        tau_source: Provenance of the threshold used by the last decision.
        z: Standard-error multiplier for the τ protocol (default ``1.0``).
        decision_type: Decision type the loader batches carry.
        forbidden_split_tokens: Tokens that mark a reported evaluation split.
    """

    tau: float | None = None
    rule: str = "state_only_if_delta_nll_exceeds"
    split_name: str = "calibration"
    min_samples: int = 32
    tie_tolerance: float = 0.0
    decision: str = "pending"
    delta_nll: float | None = None
    tau_source: str | None = None
    z: float = 1.0
    decision_type: str = "choice"
    forbidden_split_tokens: tuple[str, ...] = field(
        default_factory=lambda: LEAKY_SPLIT_TOKENS
    )

    def __post_init__(self) -> None:
        if self.rule != "state_only_if_delta_nll_exceeds":
            raise ValueError(
                f"rule: expected 'state_only_if_delta_nll_exceeds', got {self.rule!r}; "
                "it is the only implemented rule (see the module docstring)."
            )
        if self.decision not in ("pending", "state_only", "conditioned"):
            raise ValueError(
                "decision: expected one of ['pending', 'state_only', 'conditioned'], "
                f"got {self.decision!r}."
            )
        if self.tau is not None:
            if isinstance(self.tau, bool) or not isinstance(self.tau, (int, float)):
                raise TypeError(
                    f"tau: expected a float or None, got {type(self.tau).__name__}."
                )
            self.tau = float(self.tau)
            if not math.isfinite(self.tau) or self.tau < 0.0:
                raise ValueError(
                    f"tau: expected a finite value >= 0, got {self.tau!r}."
                )
        if self.decision_type not in SUPPORTED_DECISION_TYPES:
            raise ValueError(
                f"decision_type: expected one of {list(SUPPORTED_DECISION_TYPES)}, got "
                f"{self.decision_type!r}."
            )
        if isinstance(self.tau_source, str) and not self.tau_source:
            raise ValueError("tau_source: expected a non-empty string or None.")
        assert_split_not_leakage(
            self.split_name, forbidden_tokens=self.forbidden_split_tokens
        )
        self._validate_numeric()

    def _validate_numeric(self) -> None:
        if isinstance(self.min_samples, bool) or not isinstance(self.min_samples, int):
            raise TypeError(
                f"min_samples: expected an int, got {type(self.min_samples).__name__}."
            )
        if self.min_samples < 0:
            raise ValueError(
                f"min_samples: expected an int >= 0, got {self.min_samples}."
            )
        if isinstance(self.tie_tolerance, bool) or not isinstance(
            self.tie_tolerance, (int, float)
        ):
            raise TypeError(
                "tie_tolerance: expected a float, got "
                f"{type(self.tie_tolerance).__name__}."
            )
        self.tie_tolerance = float(self.tie_tolerance)
        if not math.isfinite(self.tie_tolerance) or self.tie_tolerance < 0.0:
            raise ValueError(
                "tie_tolerance: expected a finite value >= 0, got "
                f"{self.tie_tolerance!r}."
            )
        if isinstance(self.z, bool) or not isinstance(self.z, (int, float)):
            raise TypeError(f"z: expected a float, got {type(self.z).__name__}.")
        self.z = float(self.z)
        if not math.isfinite(self.z) or self.z < 0.0:
            raise ValueError(f"z: expected a finite value >= 0, got {self.z!r}.")
        if isinstance(self.forbidden_split_tokens, (str, bytes)) or not isinstance(
            self.forbidden_split_tokens, (list, tuple, set, frozenset)
        ):
            raise TypeError(
                "forbidden_split_tokens: expected a sequence of str tokens, got "
                f"{type(self.forbidden_split_tokens).__name__}."
            )
        self.forbidden_split_tokens = tuple(
            str(token) for token in self.forbidden_split_tokens
        )
        if self.delta_nll is not None:
            if isinstance(self.delta_nll, bool) or not isinstance(
                self.delta_nll, (int, float)
            ):
                raise TypeError(
                    f"delta_nll: expected a float or None, got "
                    f"{type(self.delta_nll).__name__}."
                )
            self.delta_nll = float(self.delta_nll)
            if not math.isfinite(self.delta_nll):
                raise ValueError(
                    f"delta_nll: expected a finite value, got {self.delta_nll!r}."
                )


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


class ConditioningGate:
    """Task-level switch between conditioned and state-only scoring.

    The gate measures ΔNLL on a calibration split, derives τ from the
    one-standard-error protocol on that same split (never from a reported
    evaluation split — see the module docstring) and records a single verdict
    for the whole task. It is deliberately not a per-sample switch: a rule
    fitted per example is the overfitting failure this module exists to avoid.

    Args:
        model: The model to measure. It is only read (eval/no_grad) and never
            mutated by the gate; :func:`apply_gate` is what changes a model.
        config: Optional :class:`ConditioningGateConfig`.
        loss_fn: Optional per-example loss override forwarded to
            :func:`measure_modes`.
    """

    def __init__(
        self,
        model: Any = None,
        config: ConditioningGateConfig | None = None,
        *,
        loss_fn: Callable[..., torch.Tensor] | None = None,
    ) -> None:
        if config is None:
            config = ConditioningGateConfig()
        if not isinstance(config, ConditioningGateConfig):
            raise TypeError(
                "config: expected a ConditioningGateConfig, got "
                f"{type(config).__name__}."
            )
        if loss_fn is not None and not callable(loss_fn):
            raise TypeError(
                f"loss_fn: expected a callable, got {type(loss_fn).__name__}."
            )
        self._model = model
        self.config = config
        self.loss_fn = loss_fn
        self.measurement: GateMeasurement | None = None
        # A threshold the CALLER pinned is authoritative; a threshold this
        # module fitted is written to config.tau too (so it is reportable) and
        # must be RE-FIT on the next decision, never mistaken for a pinned one.
        self._explicit_tau = config.tau

    # -- properties ---------------------------------------------------------

    @property
    def model(self) -> Any:
        """The bound model (``None`` for a gate restored from a dict)."""
        return self._model

    @property
    def decision(self) -> str:
        """The current verdict: ``pending``, ``state_only`` or ``conditioned``."""
        return self.config.decision

    @property
    def delta_nll(self) -> float | None:
        """The last measured ΔNLL (``None`` before a measurement)."""
        return self.config.delta_nll

    @property
    def tau(self) -> float | None:
        """The threshold in force for the last decision, if any."""
        return self.config.tau

    def _require_model(self) -> Any:
        if self._model is None:
            raise ValueError(
                "the gate has no model bound — construct it as "
                "ConditioningGate(model, config) or call ConditioningGate(model, "
                "gate.config) before measure()/decide()."
            )
        return self._model

    # -- measurement --------------------------------------------------------

    def measure(self, loader: Iterable[Any], *, max_batches: int | None = None):
        """Measure ΔNLL (and both modes' NLL/accuracy) on the calibration split.

        Same protocol as :func:`measure_modes`; also runs the no-test-leakage
        guard on ``config.split_name``. The result is cached in
        ``self.measurement`` and ``config.delta_nll``.

        Args:
            loader: Iterable of collated decision batches.
            max_batches: Measure only the first N batches.

        Returns:
            The :class:`GateMeasurement`.

        Raises:
            ValueError: ``config.split_name`` names a test split (message states
                the no-test-leakage rule), or the loader is empty.
        """
        model = self._require_model()
        measurement = measure_modes(
            model,
            loader,
            loss_fn=self.loss_fn,
            decision_type=self.config.decision_type,
            split_name=self.config.split_name,
            forbidden_split_tokens=self.config.forbidden_split_tokens,
            max_batches=max_batches,
        )
        self.measurement = measurement
        self.config.delta_nll = measurement.delta_nll
        return measurement

    def set_tau(self, tau: float | None) -> None:
        """Pin (or, with ``None``, un-pin) an externally justified threshold.

        Only for a τ justified OUTSIDE any evaluation split (a preregistered
        value, a domain prior, a threshold copied from a different task). The
        default — ``None`` — keeps the one-standard-error protocol, which is
        what makes the gate independent of the reported evaluation.

        Args:
            tau: A finite value ``>= 0``, or ``None`` to return to the
                protocol.

        Returns:
            ``None``.

        Raises:
            TypeError / ValueError: As :class:`ConditioningGateConfig` validates
                ``tau``.
        """
        if tau is None:
            self._explicit_tau = None
            self.config.tau = None
            self.config.tau_source = None
            return
        if isinstance(tau, bool) or not isinstance(tau, (int, float)):
            raise TypeError(f"tau: expected a float or None, got {type(tau).__name__}.")
        value = float(tau)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"tau: expected a finite value >= 0, got {value!r}.")
        self._explicit_tau = value
        self.config.tau = value

    def fit_threshold(self) -> float:
        """Resolve τ for the last measurement (pinned value or the protocol).

        A τ pinned through :meth:`set_tau` (or a non-``None`` ``tau`` in the
        config at construction) is used verbatim and reported as
        ``tau_source="explicit"``; otherwise ``z · SE`` is fitted on the
        calibration batches and written to ``config.tau`` so the threshold in
        force is always reportable.

        Returns:
            The threshold the next :meth:`decide` compares against.

        Raises:
            ValueError: No measurement has been taken yet.
        """
        if self.measurement is None:
            raise ValueError(
                "fit_threshold: no measurement yet — call measure(loader) first."
            )
        if self._explicit_tau is not None:
            self.config.tau = self._explicit_tau
            self.config.tau_source = "explicit"
            return self._explicit_tau
        fit = fit_tau(
            self.measurement.per_batch_delta,
            z=self.config.z,
        )
        self.config.tau = fit.tau
        self.config.tau_source = fit.source
        return fit.tau

    def decide(self, loader: Iterable[Any], *, max_batches: int | None = None) -> str:
        """Measure, fit τ and return the task-level verdict.

        Rule ``state_only_if_delta_nll_exceeds``:
        ``state_only`` iff ``ΔNLL > τ + tie_tolerance``, else ``conditioned``
        (a tie keeps the shipped behaviour). τ is the pinned
        ``config.tau`` when there is one, otherwise ``z · SE`` from the
        one-standard-error protocol on the calibration batches. Below
        ``config.min_samples`` examples the gate refuses to switch and keeps
        ``conditioned``, recording why in ``tau_source``.

        Args:
            loader: Iterable of collated decision batches.
            max_batches: Measure only the first N batches.

        Returns:
            ``"state_only"`` or ``"conditioned"`` (stored in ``config.decision``).

        Raises:
            ValueError: Test split named, empty loader, or no model bound.
        """
        measurement = self.measure(loader, max_batches=max_batches)
        tau = self.fit_threshold()
        if measurement.num_samples < self.config.min_samples:
            self.config.decision = "conditioned"
            self.config.tau_source = (
                f"min_samples_unmet(n={measurement.num_samples} < "
                f"{self.config.min_samples}); kept=conditioned; base_tau="
                f"{self.config.tau_source}"
            )
            return self.config.decision
        if measurement.delta_nll > tau + self.config.tie_tolerance:
            self.config.decision = "state_only"
        else:
            self.config.decision = "conditioned"
        return self.config.decision

    def condition_options_for(self, model: Any = None) -> bool:
        """The ``condition_options=`` flag that reproduces the chosen mode.

        Args:
            model: Optional model, accepted so a call site can read
                ``model(..., condition_options=gate.condition_options_for(model))``
                in one line and so the error message can name it. The model is
                NOT mutated here — :func:`apply_gate` does that.

        Returns:
            ``True`` for ``"conditioned"``, ``False`` for ``"state_only"``.

        Raises:
            RuntimeError: No decision has been made yet (the gate must not
                silently pass ``None`` to the forward, which would fall back to
                the config default and hide an undecided gate).
        """
        if self.config.decision == "pending":
            where = (
                f" on {type(model).__name__}" if model is not None else ""
            )
            raise RuntimeError(
                f"condition_options_for: the gate has no decision yet{where} — call "
                "decide(calibration_loader) first (an undecided gate would pass the "
                "config default and hide the missing verdict)."
            )
        return self.config.decision == "conditioned"

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Plain-data snapshot of the configuration (JSON-serializable)."""
        payload: dict[str, Any] = {
            "tau": self.config.tau,
            "rule": self.config.rule,
            "split_name": self.config.split_name,
            "min_samples": self.config.min_samples,
            "tie_tolerance": self.config.tie_tolerance,
            "decision": self.config.decision,
            "delta_nll": self.config.delta_nll,
            "tau_source": self.config.tau_source,
            "z": self.config.z,
            "decision_type": self.config.decision_type,
            "forbidden_split_tokens": list(self.config.forbidden_split_tokens),
        }
        if self.measurement is not None:
            payload["measurement"] = {
                "delta_nll": self.measurement.delta_nll,
                "conditioned_nll": self.measurement.conditioned_nll,
                "state_only_nll": self.measurement.state_only_nll,
                "conditioned_accuracy": self.measurement.conditioned_accuracy,
                "state_only_accuracy": self.measurement.state_only_accuracy,
                "per_batch_delta": list(self.measurement.per_batch_delta),
                "batch_sizes": list(self.measurement.batch_sizes),
                "num_samples": self.measurement.num_samples,
                "num_batches": self.measurement.num_batches,
                "decision_type": self.measurement.decision_type,
            }
        return payload

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        model: Any = None,
        *,
        loss_fn: Callable[..., torch.Tensor] | None = None,
    ) -> "ConditioningGate":
        """Rebuild a gate from :meth:`to_dict` (measurement restored too).

        Args:
            payload: A mapping produced by :meth:`to_dict`.
            model: Optional model to bind.
            loss_fn: Optional per-example loss override (never serialized).

        Returns:
            The restored gate.

        Raises:
            TypeError / ValueError: Malformed payload (unknown keys are kept
                out; invalid values are rejected by the config).
        """
        if not isinstance(payload, Mapping):
            raise TypeError(
                f"payload: expected a mapping, got {type(payload).__name__}."
            )
        known = {
            "tau",
            "rule",
            "split_name",
            "min_samples",
            "tie_tolerance",
            "decision",
            "delta_nll",
            "tau_source",
            "z",
            "decision_type",
            "forbidden_split_tokens",
        }
        unknown = sorted(set(payload) - known - {"measurement"})
        if unknown:
            raise ValueError(
                f"payload: unknown key(s) {unknown}; expected any of {sorted(known)} "
                "plus 'measurement'."
            )
        fields = {key: payload[key] for key in known if key in payload}
        tokens = fields.get("forbidden_split_tokens")
        if tokens is not None:
            fields["forbidden_split_tokens"] = tuple(tokens)
        # A τ this module FITTED must not come back pinned: a restored gate has
        # to re-fit on the next calibration split, while a τ the caller pinned
        # stays authoritative. ``config.tau`` is re-set afterwards so the
        # threshold in force is still reported.
        source = payload.get("tau_source")
        pinned = isinstance(source, str) and source.startswith("explicit")
        in_force = fields.get("tau")
        if not pinned:
            fields["tau"] = None
        gate = cls(model, ConditioningGateConfig(**fields), loss_fn=loss_fn)
        if not pinned and in_force is not None:
            gate.config.tau = float(in_force)
        measurement = payload.get("measurement")
        if measurement is not None:
            if not isinstance(measurement, Mapping):
                raise TypeError(
                    "payload['measurement']: expected a mapping, got "
                    f"{type(measurement).__name__}."
                )
            gate.measurement = GateMeasurement(
                delta_nll=float(measurement["delta_nll"]),
                conditioned_nll=float(measurement["conditioned_nll"]),
                state_only_nll=float(measurement["state_only_nll"]),
                conditioned_accuracy=float(measurement["conditioned_accuracy"]),
                state_only_accuracy=float(measurement["state_only_accuracy"]),
                per_batch_delta=tuple(
                    float(value) for value in measurement["per_batch_delta"]
                ),
                batch_sizes=tuple(int(value) for value in measurement["batch_sizes"]),
                num_samples=int(measurement["num_samples"]),
                num_batches=int(measurement["num_batches"]),
                decision_type=str(measurement["decision_type"]),
            )
        return gate

    def summary(self) -> str:
        """A deterministic, human-readable report of the gate's state."""
        config = self.config
        lines = [
            "ConditioningGate "
            f"(task-level; rule={config.rule}; split={config.split_name})",
            f"  decision      : {config.decision}",
            f"  delta_nll     : {_fmt(config.delta_nll)} "
            "(NLL_conditioned - NLL_state_only, per example)",
            f"  tau           : {_fmt(config.tau)}",
            f"  tau_source    : {config.tau_source}",
            f"  condition_options flag : {config.decision == 'conditioned'}"
            if config.decision != "pending"
            else "  condition_options flag : (undecided)",
            f"  min_samples   : {config.min_samples}",
            f"  tie_tolerance : {config.tie_tolerance:g}",
        ]
        if self.measurement is not None:
            m = self.measurement
            lines.extend(
                [
                    f"  samples/batches: {m.num_samples}/{m.num_batches}"
                    f" (decision_type={m.decision_type})",
                    f"  NLL   conditioned={m.conditioned_nll:.6f} "
                    f"state_only={m.state_only_nll:.6f}",
                    f"  acc   conditioned={m.conditioned_accuracy:.6f} "
                    f"state_only={m.state_only_accuracy:.6f}",
                    "  tau protocol   : tau = z * SE(per-batch delta), "
                    f"z={config.z:g}"
                    f" (one standard error; independent of any evaluation split)",
                ]
            )
        else:
            lines.append("  measurement    : (none yet — call measure/decide)")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ConditioningGate(decision={self.config.decision!r}, "
            f"split={self.config.split_name!r}, "
            f"delta_nll={_fmt(self.config.delta_nll)})"
        )


def _fmt(value: float | None) -> str:
    """Format an optional float for summaries (``None`` stays ``None``)."""
    return "None" if value is None else f"{value:.6f}"


def apply_gate(
    model: Any,
    gate: ConditioningGate,
    *,
    sync_interaction_scale: bool = False,
) -> bool:
    """Apply a decided gate to a model, returning the flag it set.

    Sets ``model.config.conditioning_enabled`` to the gate's verdict, which is
    the *default* the forward falls back to when ``condition_options`` is left
    as ``None``; an explicit per-call ``condition_options=`` still wins. Use
    :meth:`ConditioningGate.condition_options_for` when a single call must be
    pinned explicitly (e.g. to A/B both modes on one model).

    The gate is TASK-level and is meant to be applied AFTER the decision head
    has been trained (Stage B, and after any curriculum stage that toggled the
    option pathway). Flipping the flag afterwards changes ONLY the option
    pathway at inference: the state pathway, the projections and every learned
    weight are untouched, so no retraining is implied — but the flipped mode
    was never trained as the active path, which is exactly why the calibration
    measurement (not a test-split score) decides it.

    Args:
        model: A model exposing a mutable ``config``.
        gate: A decided :class:`ConditioningGate`.
        sync_interaction_scale: When ``True`` and the model exposes the
            interaction-scale API (``set_interaction_scale``), the scale is set
            to ``1.0`` for ``conditioned`` and ``0.0`` for ``state_only`` so
            the two mechanisms cannot disagree. Requires a head that provides
            the method; raises when asked for on a model that does not.

    Returns:
        The boolean written into ``model.config.conditioning_enabled``.

    Raises:
        TypeError: ``model``/``gate`` types are wrong, or the model has no
            mutable ``config``.
        RuntimeError: The gate is still ``pending``, or
            ``sync_interaction_scale=True`` on a model without
            ``set_interaction_scale``.
    """
    if not isinstance(gate, ConditioningGate):
        raise TypeError(
            f"gate: expected a ConditioningGate, got {type(gate).__name__}."
        )
    if gate.decision == "pending":
        raise RuntimeError(
            "apply_gate: the gate is still 'pending' — call decide(calibration_loader) "
            "first; applying an undecided gate would set a flag no measurement "
            "supports."
        )
    config = getattr(model, "config", None)
    if config is None:
        raise TypeError(
            f"apply_gate: {type(model).__name__} has no `config`; the gate is applied by "
            "writing config.conditioning_enabled."
        )
    flag = gate.condition_options_for(model)
    setattr(config, "conditioning_enabled", flag)
    if sync_interaction_scale:
        setter = None
        for name, module in getattr(model, "named_modules", lambda: [])():
            candidate = getattr(module, "set_interaction_scale", None)
            if callable(candidate):
                setter = candidate
                break
        if setter is None:
            raise RuntimeError(
                "apply_gate: sync_interaction_scale=True but no module of "
                f"{type(model).__name__} exposes set_interaction_scale."
            )
        setter(1.0 if flag else 0.0)
    return flag
