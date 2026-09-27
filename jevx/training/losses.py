# Copyright (c) 2026 ysnrfd
"""Typed training losses for JevX decision tasks (§85 loss primitives).

All losses are pure functions over raw head logits:

* ``choice_loss``    — masked cross-entropy over mutually exclusive options
  (Choice, and the classification view of Score);
* ``score_loss``     — cross-entropy over ordinal levels and/or MSE on the
  expected score ``s = Σ_k k·p_k`` (``mode="classification" | "expectation"
  | "hybrid"``);
* ``noul_loss``      — binary cross-entropy with logits for Noul/Threshold;
* ``multilabel_loss``— EXPERIMENTAL per-label BCE for MultiLabel.

Numerical policy: every loss delegates to the native stable primitives
(:func:`torch.nn.functional.cross_entropy`, :func:`F.binary_cross_entropy_with_logits`,
:func:`jevx.utils.masked_softmax`) — no hand-rolled ``log(softmax)`` anywhere.
Invalid (masked) options are excluded from the softmax denominator by filling
their logit with ``torch.finfo(dtype).min`` (the codebase-wide convention —
never ``-inf``, which yields NaN gradients), then applying native
``cross_entropy``; gradients at masked positions are exactly zero because
their softmax mass underflows to 0.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from jevx.utils.numerical import masked_softmax
from jevx.utils.shapes import assert_shape, check_finite

__all__ = [
    "choice_loss",
    "multilabel_loss",
    "noul_loss",
    "score_loss",
]

VALID_REDUCTIONS: tuple[str, ...] = ("mean", "sum", "none")

_SCORE_MODES: tuple[str, ...] = ("classification", "expectation", "hybrid")


def _validate_reduction(reduction: str) -> None:
    if reduction not in VALID_REDUCTIONS:
        raise ValueError(
            f"reduction: expected one of {list(VALID_REDUCTIONS)}, got {reduction!r}."
        )


def _reduce(values: torch.Tensor, reduction: str) -> torch.Tensor:
    """Reduce per-sample values ``[B]`` according to ``reduction``."""
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    return values


def _check_mask(
    option_mask: torch.Tensor, logits: torch.Tensor
) -> torch.Tensor:
    """Validate ``option_mask [B, K]`` bool against ``logits [B, K]``."""
    if not isinstance(option_mask, torch.Tensor):
        raise TypeError(
            f"option_mask: expected a torch.Tensor, got {type(option_mask).__name__}."
        )
    if option_mask.dtype != torch.bool:
        raise TypeError(
            f"option_mask: expected a torch.bool tensor (True = valid option), got "
            f"{option_mask.dtype}; convert it with `.bool()`."
        )
    assert_shape(option_mask, tuple(logits.shape), "option_mask")
    if not bool(option_mask.any(dim=-1).all()):
        row = int((~option_mask.any(dim=-1)).nonzero(as_tuple=False)[0])
        raise ValueError(
            f"option_mask: row {row} has no valid option; cross-entropy over an empty "
            "option set is undefined."
        )
    return option_mask


def _integral_targets(target: torch.Tensor, name: str = "target") -> torch.Tensor:
    """Return ``target`` as ``torch.long``, rejecting non-integral floats.

    JSON-produced labels often arrive as floats (``1.0``); those are accepted
    because they are exactly integral, but ``1.5`` as a class index is not.
    """
    if not isinstance(target, torch.Tensor):
        raise TypeError(f"{name}: expected a torch.Tensor, got {type(target).__name__}.")
    if target.is_floating_point() and not bool(torch.equal(target, target.round())):
        lo = float(target.min()) if target.numel() else 0.0
        hi = float(target.max()) if target.numel() else 0.0
        raise ValueError(
            f"{name}: expected integer class indices, but found non-integral values "
            f"(min={lo}, max={hi})."
        )
    return target.long()


def _validate_choice_target(
    target: torch.Tensor,
    logits: torch.Tensor,
    option_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Validate ``target [B]`` against width ``K`` and the option mask."""
    target = _integral_targets(target)
    assert_shape(target, (logits.shape[0],), "target")
    k = logits.shape[1]
    if target.numel() > 0:
        lo, hi = int(target.min()), int(target.max())
        if lo < 0 or hi >= k:
            raise ValueError(
                f"target: class indices must lie in [0, {k}) for {k} logits, "
                f"but found min={lo}, max={hi}."
            )
    if option_mask is not None:
        valid = option_mask.gather(1, target.unsqueeze(1)).squeeze(1)
        if not bool(valid.all()):
            row = int((~valid).nonzero(as_tuple=False)[0])
            tgt = int(target[row])
            raise ValueError(
                f"target: row {row} labels option {tgt}, which the option_mask marks "
                "invalid; targets must reference a valid option."
            )
    return target


def _masked_logits(
    logits: torch.Tensor, option_mask: torch.Tensor | None
) -> torch.Tensor:
    """Fill invalid logits with ``finfo.min`` so native CE excludes them."""
    if option_mask is None:
        return logits
    return logits.masked_fill(~option_mask, torch.finfo(logits.dtype).min)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def choice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    option_mask: torch.Tensor | None = None,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Masked categorical cross-entropy for Choice decisions.

    Args:
        logits: Raw option scores ``[B, K]`` (unmasked — invalid columns may
            hold arbitrary finite values; they are excluded before the
            softmax denominator).
        target: Class indices ``[B]`` in ``[0, K)``; every target must point
            at a valid (unmasked) option when ``option_mask`` is given.
            Float tensors are accepted iff exactly integral.
        option_mask: Optional ``[B, K]`` bool (``True`` = valid option) for
            variable-``K`` batches; ``None`` means all ``K`` options valid.
        reduction: ``"mean"`` (default), ``"sum"`` or ``"none"``
            (per-sample loss ``[B]``).

    Returns:
        Scalar tensor (mean/sum) or ``[B]`` tensor (none).

    Raises:
        TypeError: A non-tensor input, or a non-bool ``option_mask``.
        ValueError: Shape mismatch, non-integral/out-of-range target, a
            target naming a masked-out option, an all-masked row, or an
            unknown ``reduction``.
    """
    _validate_reduction(reduction)
    logits = assert_shape(check_finite(_float_tensor(logits, "logits"), "logits"),
                          (None, None), "logits")
    if logits.shape[1] < 1:
        raise ValueError(f"choice_loss: need at least 1 option, got K={logits.shape[1]}.")
    if option_mask is not None:
        _check_mask(option_mask, logits)
    target = _validate_choice_target(target, logits, option_mask)
    per_sample = F.cross_entropy(_masked_logits(logits, option_mask), target,
                                 reduction="none")
    return _reduce(per_sample, reduction)


def score_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    option_mask: torch.Tensor | None = None,
    *,
    mode: str = "classification",
    expectation_weight: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Loss for Score decisions over ``K`` ordinal levels ``0..K-1``.

    Modes (levels are identified with ``{0, ..., K-1}``, Definition 3):

    * ``"classification"`` — cross-entropy on the level index
      (identical to :func:`choice_loss`).
    * ``"expectation"`` — MSE on the expected score
      ``s = Σ_k k·p_k`` with ``p = masked_softmax(logits)``;
      ``target`` is a real value in ``[0, K-1]``.
    * ``"hybrid"`` — ``CE + expectation_weight · MSE``; ``target`` must be an
      integral level (it serves both terms).

    Args:
        logits: Level scores ``[B, K]``.
        target: Class indices ``[B]`` (classification/hybrid) or real scores
            ``[B]`` in ``[0, K-1]`` (expectation).
        option_mask: Optional ``[B, K]`` bool mask of valid levels.
        mode: One of ``"classification"``, ``"expectation"``, ``"hybrid"``.
        expectation_weight: Non-negative weight on the expectation MSE term
            (used only in ``"hybrid"`` mode).
        reduction: ``"mean"`` (default), ``"sum"`` or ``"none"``.

    Returns:
        Scalar tensor (mean/sum) or ``[B]`` tensor (none).

    Raises:
        TypeError / ValueError: As :func:`choice_loss`, plus unknown
            ``mode``, negative/non-finite ``expectation_weight``, or an
            expectation target outside ``[0, K-1]``.
    """
    _validate_reduction(reduction)
    if mode not in _SCORE_MODES:
        raise ValueError(
            f"score_loss: mode must be one of {list(_SCORE_MODES)}, got {mode!r}."
        )
    if not isinstance(expectation_weight, (int, float)) or isinstance(
        expectation_weight, bool
    ):
        raise TypeError(
            f"expectation_weight: expected a float, got "
            f"{type(expectation_weight).__name__}."
        )
    expectation_weight = float(expectation_weight)
    if not (expectation_weight >= 0.0) or expectation_weight == float("inf"):
        raise ValueError(
            f"expectation_weight: expected a finite value >= 0, got "
            f"{expectation_weight!r}."
        )
    logits = assert_shape(check_finite(_float_tensor(logits, "logits"), "logits"),
                          (None, None), "logits")
    k = logits.shape[1]
    if k < 1:
        raise ValueError(f"score_loss: need at least 1 level, got K={k}.")
    if option_mask is not None:
        _check_mask(option_mask, logits)

    per_sample: torch.Tensor | None = None

    if mode in ("classification", "hybrid"):
        target_ce = _validate_choice_target(target, logits, option_mask)
        per_sample = F.cross_entropy(_masked_logits(logits, option_mask),
                                     target_ce, reduction="none")
    else:  # expectation only — target is a real score, not an index
        target = _float_tensor(target, "target")
        assert_shape(target, (logits.shape[0],), "target")
        if target.numel() > 0:
            lo, hi = float(target.min()), float(target.max())
            if lo < 0.0 or hi > float(k - 1):
                raise ValueError(
                    f"target: expectation targets must lie in [0, {k - 1}] "
                    f"(the Score level range), but found min={lo}, max={hi}."
                )

    if mode in ("expectation", "hybrid"):
        probs = masked_softmax(logits, option_mask)  # [B, K], rows sum to 1
        levels = torch.arange(k, dtype=probs.dtype, device=probs.device)  # [K]
        expected = (probs * levels).sum(dim=-1)  # [B]
        target_e = target.to(dtype=logits.dtype)
        mse = (expected - target_e).pow(2)
        per_sample = mse if per_sample is None else (
            per_sample + expectation_weight * mse
        )

    assert per_sample is not None  # mode dispatch guarantees a term
    return _reduce(per_sample, reduction)


def noul_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Binary cross-entropy with logits for Noul/Threshold decisions.

    Uses :func:`F.binary_cross_entropy_with_logits` (the ReLU-trick stable
    primitive), so inputs of ``±1e4`` stay finite.

    Args:
        logits: Raw binary scores ``[B]`` or ``[B, 1]``.
        target: Binary labels ``[B]`` in ``{0, 1}`` (float ``0.0``/``1.0``
            accepted).
        reduction: ``"mean"`` (default), ``"sum"`` or ``"none"``.

    Returns:
        Scalar tensor (mean/sum) or ``[B]`` tensor (none).

    Raises:
        TypeError / ValueError: Wrong rank, non-binary target, or unknown
            ``reduction``.
    """
    _validate_reduction(reduction)
    logits = check_finite(_float_tensor(logits, "logits"), "logits")
    if logits.dim() == 2 and logits.shape[1] == 1:
        logits = logits.squeeze(-1)
    assert_shape(logits, (None,), "logits")
    target = _integral_targets(target)
    assert_shape(target, (logits.shape[0],), "target")
    if target.numel() > 0:
        lo, hi = int(target.min()), int(target.max())
        if lo < 0 or hi > 1:
            raise ValueError(
                f"target: Noul labels must be in {{0, 1}}, but found min={lo}, max={hi}."
            )
    per_sample = F.binary_cross_entropy_with_logits(
        logits, target.to(dtype=logits.dtype), reduction="none"
    )
    return _reduce(per_sample, reduction)


def multilabel_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """EXPERIMENTAL: per-label binary cross-entropy for MultiLabel tasks.

    Each label is an independent Bernoulli (Definition 6c — rows deliberately
    do not sum to 1). Masked-out labels contribute exactly zero loss.

    Args:
        logits: Raw label scores ``[B, K]``.
        target: Binary multi-hot targets ``[B, K]`` in ``{0, 1}``
            (float accepted iff exactly 0/1).
        mask: Optional ``[B, K]`` bool mask of *valid labels* (``True`` =
            evaluate this label); masked entries in ``target`` are ignored
            (and need not be binary).
        reduction: ``"mean"`` (default — mean over VALID labels), ``"sum"``
            (sum over valid labels), or ``"none"`` (``[B, K]`` with masked
            entries zeroed).

    Returns:
        Scalar tensor (mean/sum) or ``[B, K]`` tensor (none).

    Raises:
        TypeError / ValueError: Shape mismatch, non-binary unmasked target,
            non-bool mask, or unknown ``reduction``.
    """
    _validate_reduction(reduction)
    logits = assert_shape(check_finite(_float_tensor(logits, "logits"), "logits"),
                          (None, None), "logits")
    target = _float_tensor(target, "target")
    assert_shape(target, tuple(logits.shape), "target")
    if mask is not None:
        mask = _check_mask(mask, logits)
        checked = target[mask]
    else:
        checked = target
    if checked.numel() > 0 and not bool(((checked == 0) | (checked == 1)).all()):
        lo, hi = float(checked.min()), float(checked.max())
        raise ValueError(
            f"target: MultiLabel targets must be in {{0, 1}}, but found min={lo}, "
            f"max={hi} at unmasked positions."
        )
    per_element = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    )
    if mask is not None:
        per_element = per_element.masked_fill(~mask, 0.0)
    if reduction == "mean":
        if mask is not None:
            count = mask.sum()
            if int(count) == 0:
                raise ValueError(
                    "multilabel_loss: the mask has no valid labels; mean over an "
                    "empty set is undefined."
                )
            return per_element.sum() / count.to(per_element.dtype)
        return per_element.mean()
    if reduction == "sum":
        return per_element.sum()
    return per_element


def _float_tensor(value: object, name: str) -> torch.Tensor:
    """Coerce ``value`` to a floating-point tensor, rejecting non-tensors."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name}: expected a torch.Tensor, got {type(value).__name__}.")
    if not value.is_floating_point():
        return value.float()
    return value
