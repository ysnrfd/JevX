# Copyright (c) 2026 ysnrfd
"""Numerically stable tensor primitives and calibration metrics.

Contracts used across JevX:

* All softmax-family helpers compute in float32 when the input is fp16/bf16
  and cast back to the input dtype (probabilities are consumed downstream in
  the model's working dtype).
* Masked softmax treats ``mask=False`` positions as *impossible*: their logit
  is set to ``torch.finfo(dtype).min`` (never ``-inf``, which produces NaN
  gradients) and their probability is forced to exactly ``0.0`` while the
  valid positions of each row sum to 1 (per-question normalization only —
  never across rows).
* Calibration metrics (NLL / ECE / MCE / Brier) operate on probabilities of
  shape ``[N, K]`` plus integer labels ``[N]``. ``K == 1`` is interpreted as a
  single sigmoid probability ``P(y=1)`` with labels in ``{0, 1}``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .shapes import assert_shape

__all__ = [
    "BATCH_EQUIVALENCE_SAFETY",
    "JEVX_DECISION_ACCUMULATION_STAGES",
    "LOW_PRECISION_DTYPES",
    "batch_equivalence_tolerance",
    "brier_score",
    "expected_calibration_error",
    "masked_softmax",
    "maximum_calibration_error",
    "negative_log_likelihood",
    "one_hot",
    "stable_log_softmax",
    "stable_softmax",
]

#: Per-dtype safety factor of the batched-vs-solo (T7-ii) tolerance rule.
#: Derivation (reports/NUMERICAL_EQUIVALENCE_STUDY.md): the smallest power of
#: two that is at least 3x the WORST ratio observed between a measured
#: batched-vs-solo deviation and the bare "eps * sqrt(T * stages) * scale"
#: term, over every cell of the sweep. float32 is governed by the real
#: 27.6M-parameter scenario walk (worst ratio 2.45); bfloat16/float16 are
#: governed by the low-precision pooling cells (worst ratio 0.089).
BATCH_EQUIVALENCE_SAFETY: dict[str, float] = {
    "float32": 8.0,
    "float64": 8.0,
    "bfloat16": 0.5,
    "float16": 0.5,
}

#: Sequential accumulation stages a JevX call adds ON TOP OF the backbone
#: layers: state projection, fusion context, the shared per-option formula,
#: the decision head, and the probability map. Combined with L backbone
#: layers this is the "stages" argument of batch_equivalence_tolerance() for
#: a full JevXForDecision call.
JEVX_DECISION_ACCUMULATION_STAGES = 5

LOW_PRECISION_DTYPES: tuple[torch.dtype, ...] = (torch.float16, torch.bfloat16)

_DEFAULT_PROBS_DTYPE = torch.float32


def _checked_float(logits: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"{name}: expected a torch.Tensor, got {type(logits).__name__}.")
    if not logits.is_floating_point():
        raise TypeError(f"{name}: expected a floating-point tensor, got {logits.dtype}.")
    return logits


def stable_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax that upcasts fp16/bf16 inputs to fp32, then casts back.

    fp16/bf16 softmax is prone to overflow in ``exp``; computing in fp32 and
    downcasting keeps the result finite and normalized for realistic logits
    (|logit| up to ~1e4 in fp32). The returned tensor has the input dtype.

    Args:
        logits: Floating-point tensor, any shape.
        dim: Reduction dimension.

    Returns:
        Probabilities with the same shape and dtype as ``logits``.
    """
    logits = _checked_float(logits, "logits")
    compute = logits.float() if logits.dtype in LOW_PRECISION_DTYPES else logits
    probs = F.softmax(compute, dim=dim)
    return probs.to(logits.dtype) if probs.dtype != logits.dtype else probs


def stable_log_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Log-softmax with the same fp16/bf16 upcast policy as :func:`stable_softmax`.

    Args:
        logits: Floating-point tensor, any shape.
        dim: Reduction dimension.

    Returns:
        Log-probabilities, same shape/dtype as ``logits``; rows sum to 0 on
        the probability scale (``logsumexp == 0``) up to dtype precision.
    """
    logits = _checked_float(logits, "logits")
    compute = logits.float() if logits.dtype in LOW_PRECISION_DTYPES else logits
    out = F.log_softmax(compute, dim=dim)
    return out.to(logits.dtype) if out.dtype != logits.dtype else out


def masked_softmax(
    logits: torch.Tensor,
    mask: torch.Tensor | None = None,
    dim: int = -1,
) -> torch.Tensor:
    """Mask-aware softmax for variable-K batches (``True`` = valid option).

    Invalid positions receive ``torch.finfo(dtype).min`` before the softmax —
    *not* ``-inf``, which yields NaN gradients — so after the softmax their
    probability is exactly ``0.0`` and the valid entries of each row sum to 1
    (within dtype precision). Normalization is strictly per row.

    Args:
        logits: Floating-point scores ``[B, K]`` (unmasked raw logits).
        mask: Optional boolean tensor with the same shape as ``logits``.
            ``None`` means every position is valid.
        dim: Dimension spanning the options (default ``-1``).

    Returns:
        Probabilities with the same shape/dtype as ``logits``.

    Raises:
        TypeError: ``mask`` is not a boolean tensor.
        ValueError: mask shape differs from logits, or a row is fully masked
            out (softmax over an empty set is undefined).
    """
    logits = _checked_float(logits, "logits")
    if mask is None:
        return stable_softmax(logits, dim=dim)
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"mask: expected a torch.Tensor, got {type(mask).__name__}.")
    if mask.dtype != torch.bool:
        raise TypeError(
            f"mask: expected a torch.bool tensor (True = valid option), got {mask.dtype}; "
            "convert it with `.bool()`."
        )
    if mask.shape != logits.shape:
        raise ValueError(
            f"mask: expected the same shape as logits {tuple(logits.shape)} "
            f"but got {tuple(mask.shape)}."
        )
    if not bool(mask.any(dim=dim).all()):
        raise ValueError(
            "masked_softmax: every row must keep at least one valid option, "
            "but a row is fully masked out; softmax over an empty option set is undefined."
        )

    compute = logits.float() if logits.dtype in LOW_PRECISION_DTYPES else logits
    filled = compute.masked_fill(~mask, torch.finfo(compute.dtype).min)
    probs = F.softmax(filled, dim=dim)
    # Guarantee: invalid positions are exactly zero, independent of underflow.
    probs = probs.masked_fill(~mask, 0.0)
    return probs.to(logits.dtype) if probs.dtype != logits.dtype else probs


def one_hot(
    labels: torch.Tensor,
    num_classes: int,
    dtype: torch.dtype = _DEFAULT_PROBS_DTYPE,
) -> torch.Tensor:
    """One-hot encode integer class labels.

    Args:
        labels: Integer tensor ``[N]`` (or any shape) of class indices.
        num_classes: Number of classes ``K``.
        dtype: Dtype of the output (default float32).

    Returns:
        Tensor of shape ``(*labels.shape, K)``.

    Raises:
        ValueError: ``num_classes < 1`` or a label lies outside ``[0, K)``.
    """
    if num_classes < 1:
        raise ValueError(f"one_hot: num_classes must be >= 1, got {num_classes}.")
    if not isinstance(labels, torch.Tensor):
        raise TypeError(f"labels: expected a torch.Tensor, got {type(labels).__name__}.")
    labels = labels.long()
    if labels.numel() > 0:
        lo, hi = int(labels.min()), int(labels.max())
        if lo < 0 or hi >= num_classes:
            raise ValueError(
                f"one_hot: labels must lie in [0, {num_classes}), "
                f"but found min={lo}, max={hi}."
            )
    return F.one_hot(labels, num_classes=num_classes).to(dtype)


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------


def _validate_probs_labels(
    probs: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    assert_shape(probs, (None, None), "probs")
    if not probs.is_floating_point():
        raise TypeError(f"probs: expected a floating-point tensor, got {probs.dtype}.")
    assert_shape(labels, (None,), "labels")
    if probs.shape[0] != labels.shape[0]:
        raise ValueError(
            f"labels: expected one label per sample — probs has {probs.shape[0]} samples "
            f"but labels has {labels.shape[0]} entries."
        )
    if probs.shape[1] < 1:
        raise ValueError(f"probs: expected at least 1 class, got K={probs.shape[1]}.")
    k = probs.shape[1]
    labels_l = labels.long()
    if labels_l.numel() > 0:
        lo, hi = int(labels_l.min()), int(labels_l.max())
        if k == 1:
            if lo < 0 or hi > 1:
                raise ValueError(
                    f"labels: single-column probabilities (K=1) imply binary labels in {{0, 1}}, "
                    f"but found min={lo}, max={hi}."
                )
        elif lo < 0 or hi >= k:
            raise ValueError(
                f"labels: must lie in [0, {k}) for K={k}-class probabilities, "
                f"but found min={lo}, max={hi}."
            )


def _confidence_and_correct(
    probs: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (confidence [N], correctness [N]) as float64.

    For K >= 2: confidence = max_k p_k, predicted = argmax_k p_k.
    For K == 1 (single sigmoid probability of class 1): confidence =
    max(p, 1-p), predicted = (p >= 0.5).
    """
    probs64 = probs.double()
    if probs.shape[1] == 1:
        p = probs64[:, 0].clamp(0.0, 1.0)
        conf = torch.maximum(p, 1.0 - p)
        correct = ((p >= 0.5).long() == labels.long()).double()
    else:
        conf, pred = probs64.max(dim=-1)
        correct = (pred == labels.long()).double()
    return conf, correct


def _bin_indices(conf: torch.Tensor, n_bins: int) -> torch.Tensor:
    """Equal-width bins over [0, 1]: bin i covers [i/n, (i+1)/n), last closed.

    ``conf == 1`` maps to the last bin via clamping.
    """
    return (conf * n_bins).long().clamp(max=n_bins - 1)


def _check_n_bins(n_bins: int) -> None:
    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError(f"n_bins: expected an int >= 1, got {n_bins!r}.")


def expected_calibration_error(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error: ``sum_b |B_b|/N * |acc(B_b) - conf(B_b)|``.

    Empty bins contribute nothing. See :func:`_bin_indices` for bin edges.

    Args:
        probs: Class probabilities ``[N, K]`` (already normalized; for K=1 a
            single sigmoid probability).
        labels: Integer labels ``[N]`` (``{0,1}`` when K == 1).
        n_bins: Number of equal-width confidence bins (default 15).

    Returns:
        ECE as a Python float in ``[0, 1]``.
    """
    _validate_probs_labels(probs, labels)
    _check_n_bins(n_bins)
    conf, correct = _confidence_and_correct(probs, labels)
    if conf.numel() == 0:
        raise ValueError("expected_calibration_error: cannot evaluate on an empty batch.")
    bins = _bin_indices(conf, n_bins)
    total = conf.shape[0]
    ece = 0.0
    for b in range(n_bins):
        sel = bins == b
        count = int(sel.sum())
        if count == 0:
            continue
        gap = abs(float(correct[sel].mean()) - float(conf[sel].mean()))
        ece += (count / total) * gap
    return float(ece)


def maximum_calibration_error(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> float:
    """Maximum Calibration Error: ``max_b |acc(B_b) - conf(B_b)|`` over non-empty bins.

    Same binning as :func:`expected_calibration_error`.

    Returns:
        MCE as a Python float in ``[0, 1]``.
    """
    _validate_probs_labels(probs, labels)
    _check_n_bins(n_bins)
    conf, correct = _confidence_and_correct(probs, labels)
    if conf.numel() == 0:
        raise ValueError("maximum_calibration_error: cannot evaluate on an empty batch.")
    bins = _bin_indices(conf, n_bins)
    mce = 0.0
    for b in range(n_bins):
        sel = bins == b
        if not bool(sel.any()):
            continue
        gap = abs(float(correct[sel].mean()) - float(conf[sel].mean()))
        mce = max(mce, gap)
    return float(mce)


def negative_log_likelihood(
    probs: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    """Mean ``-log p(label)`` (cross-entropy) in nats.

    For K == 1 the single column is read as ``P(y=1)`` and the binary
    cross-entropy ``-[y log p + (1-y) log(1-p)]`` is used. Probabilities are
    clamped to ``[eps, 1]`` before the log to stay finite at exact zeros.

    Returns:
        NLL as a Python float (computed in float64).
    """
    _validate_probs_labels(probs, labels)
    if probs.shape[1] == 1:
        # Binary branch needs both p and 1-p bounded away from 0, otherwise
        # p == 1 gives 0 * log(0) = NaN when the label is 1.
        p = probs.double()[:, 0].clamp(min=eps, max=1.0 - eps)
        y = labels.double()
        nll = -(y * p.log() + (1.0 - y) * (1.0 - p).log()).mean()
    else:
        picked = probs.double().clamp(min=eps, max=1.0)
        picked = picked.gather(1, labels.long().unsqueeze(1)).squeeze(1)
        nll = -picked.log().mean()
    return float(nll)


def brier_score(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """Mean squared error against the one-hot target: ``E[sum_k (p_k - y_k)^2]``.

    For K == 1 this reduces to the binary Brier score ``E[(p - y)^2]``.

    Returns:
        Brier score as a Python float (computed in float64).
    """
    _validate_probs_labels(probs, labels)
    probs64 = probs.double()
    if probs.shape[1] == 1:
        return float(((probs64[:, 0] - labels.double()) ** 2).mean())
    target = one_hot(labels.long(), probs.shape[1], dtype=torch.float64)
    return float(((probs64 - target) ** 2).sum(dim=-1).mean())


def _resolve_dtype(dtype: "torch.dtype | str") -> torch.dtype:
    """Accept a torch dtype or its name (``"float32"``); reject anything else.

    A string is resolved with a single ``getattr(torch, name)`` lookup, which
    already covers the short aliases ``"half"`` and ``"bfloat16"`` (both are
    ``torch.dtype`` attributes); anything else in the namespace that is not
    a dtype (``"tensor"`, ``"nn"`, ...) is rejected, as is a name torch does
    not define at all.

    Raises:
        TypeError: ``dtype`` is neither a ``torch.dtype`` nor a dtype name.
    """
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        resolved = getattr(torch, dtype, None)
        if isinstance(resolved, torch.dtype):
            return resolved
    raise TypeError(
        f"dtype: expected a torch.dtype or a dtype name, got {type(dtype).__name__}."
    )


def batch_equivalence_tolerance(
    dtype: "torch.dtype | str",
    seq_len: int,
    *,
    scale: float = 1.0,
    stages: int = 1,
    safety: float | None = None,
) -> float:
    """Bound on the batched-vs-solo (T7-ii) deviation for one configuration.

    The rule is

        tol = safety(dtype) * eps(dtype) * sqrt(seq_len * stages) * scale

    where ``eps(dtype)`` is the machine epsilon of the *compute* dtype and
    ``safety(dtype)`` is the measured constant in
    :data:`BATCH_EQUIVALENCE_SAFETY`. It replaces the bare "1e-5" that was
    previously hard-coded in tests and reports: a fixed number is not a
    bound, because the same forward pass evaluated over differently shaped
    tensors (batch width) rounds differently, and the rounding grows with
    both the reduction width and the number of sequential accumulation
    stages (PROOFS.md T7-ii, Remark 7.2; A7.4(b)).

    Where the numbers come from (reports/NUMERICAL_EQUIVALENCE_STUDY.md,
    seeded sweep; see that file for the full grid):

    * pooled ``H_S`` over 96 cells (pooling in {cls, masked_mean,
      last_valid, attention_pool} x {float32, bfloat16} x padding width
      {4, 8, 16, 32, 48, 64} x {1, 2} layers): worst observed deviation
      4.768e-07 (float32) and 3.906e-03 (bfloat16); the worst ratio to the
      bare term is 0.77 (float32) and 0.089 (bfloat16).
    * the real 1500-state scenario walk over a 27.6M-parameter, 6-layer
      backbone (benchmarks/results/real_scenario.json): the worst
      probability deviation was 1.0967254638671875e-05, ratio 2.45 to the
      bare term with T = 128 and stages = 6 + 5 = 11. That cell -- not the
      synthetic one -- is what sets the float32 safety factor of 8.0.
    * no argmax flip was observed in any cell of either study.

    Args:
        dtype: Compute dtype (a ``torch.dtype`` or its name). Use the dtype
            the backbone and the head actually run in, not the dtype the
            tokenizer produces.
        seq_len: The padded sequence width ``T`` that differs between the
            two runs. This is the reduction length whose *ordering* differs,
            i.e. the width of the LONGER of the two calls, not the number of
            valid tokens.
        scale: Magnitude the deviation is measured against. ``1.0``
            (default) is right for probabilities, which live in [0, 1]; for
            unbounded quantities (logits, hidden states) pass the largest
            absolute value seen, e.g. ``float(t.abs().max())``.
        stages: Number of sequential accumulation stages. Use
            ``num_backbone_layers + JEVX_DECISION_ACCUMULATION_STAGES`` for
            a full ``JevXForDecision`` call, or ``1`` when comparing a single
            primitive.
        safety: Override the per-dtype safety factor. Use it to tighten or
            loosen deliberately; the default is the measured one.

    Returns:
        A non-negative float tolerance.

    Raises:
        TypeError: ``dtype`` is neither a torch dtype nor a dtype name.
        ValueError: ``dtype`` resolves to a torch dtype with no measured
            safety factor (the message names the supported ones);
            ``seq_len``/``stages`` are not positive ints, or ``scale``/
            ``safety`` is not positive and finite.

    Example:
        >>> import torch
        >>> from jevx.utils.numerical import batch_equivalence_tolerance
        >>> tol = batch_equivalence_tolerance(torch.float32, 128, stages=11)
        >>> tol > 1.0967254638671875e-05   # the recorded scenario deviation
        True
    """
    resolved = _resolve_dtype(dtype)
    for name, value in (("seq_len", seq_len), ("stages", stages)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(
                f"{name}: expected a positive int, got {value!r}."
            )
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise TypeError(f"scale: expected a real number, got {type(scale).__name__}.")
    scale = float(scale)
    if not scale > 0.0 or scale == float("inf"):
        raise ValueError(f"scale: expected a positive finite float, got {scale!r}.")
    if safety is None:
        dtype_name = str(resolved).removeprefix("torch.")
        try:
            factor = BATCH_EQUIVALENCE_SAFETY[dtype_name]
        except KeyError:
            raise ValueError(
                f"dtype: no measured safety factor for {resolved} (dtype name "
                f"{dtype_name!r}); batch_equivalence_tolerance is calibrated for "
                f"{sorted(BATCH_EQUIVALENCE_SAFETY)} only - pass safety=<float> "
                "to override, or use one of the supported dtypes."
            ) from None
    else:
        factor = safety
    if isinstance(factor, bool) or not isinstance(factor, (int, float)):
        raise TypeError(f"safety: expected a real number or None, got {type(factor).__name__}.")
    factor = float(factor)
    if not factor > 0.0 or factor == float("inf"):
        raise ValueError(f"safety: expected a positive finite float, got {factor!r}.")
    return (
        factor
        * float(torch.finfo(resolved).eps)
        * ((seq_len * stages) ** 0.5)
        * scale
    )
