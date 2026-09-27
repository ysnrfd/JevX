# Copyright (c) 2026 ysnrfd
"""Runtime validation of extracted representations and decisions (audit Q-003).

:class:`ValidationEngine` is the guardrail component named in
``docs/ARCHITECTURE.md`` §2 (L5), ``docs/PROOFS.md`` and
``docs/MATHEMATICAL_SPECIFICATION.md`` (axioms D1.x, invariants D8.I /
(11.3)): it re-checks the invariants the pipeline relies on *at the point
they are produced*, so a regression in an adapter, head or calibrator
surfaces as an actionable ``ValueError`` at the boundary instead of as
silently wrong decisions.

What is checked (the fix-wave contract, docs/API.md §3):

* :meth:`ValidationEngine.validate_representation` — called from
  ``JevXForDecision.forward`` after extraction: a pooled state vector
  ``H_S`` exists, ranks/dtypes are sane, every tensor is finite, and
  sequence ``[B, T, D]`` / ``attention_mask`` ``[B, T]`` align with the
  pooled batch when present.
* :meth:`ValidationEngine.validate_decision` — called from ``forward``
  after each head output: shape sanity + finiteness of ``logits`` /
  ``probabilities`` / ``confidence``; per-type probability invariants
  (**choice/score** rows form a simplex, ``sum_i p_i = 1 ± tol`` per
  question — never across questions; **noul** is one column strictly
  inside ``(0, 1)``); ``confidence`` in ``[0, 1]``; ``predicted_index``
  inside the option range (``{0, 1}`` for noul); bool ``abstained`` flag.

Both methods return ``None`` on success and raise ``ValueError`` (shape /
value / invariant violations) or ``TypeError`` (wrong object types) with
messages that name the offending tensor, the value found, and carry the
caller's ``context`` prefix (reuse of :mod:`jevx.utils.shapes` helpers:
``check_finite`` / ``assert_batch_alignment``).

The checks are read-only and RNG-free: disabling them
(``JevXConfig.runtime_validation = False``) cannot change any output —
the healthy-call outputs are bit-for-bit identical (tested).

Documented exception (docs/CONFIDENCE.md §1): ``confidence_method
= "energy"`` returns RAW, unbounded energies before ``engine.fit()``.
Models configured with energy therefore construct this engine with
``allow_unbounded_confidence=True`` so the documented pre-fit behavior
keeps working end-to-end; every other confidence method is checked
against ``[0, 1]``.
"""

from __future__ import annotations

from typing import Any, List

import torch

from .utils.shapes import assert_batch_alignment, check_finite

__all__ = ["ValidationEngine"]

#: Slack for probability/confidence range checks: softmax/sigmoid outputs
#: can land a few ULPs outside ``[0, 1]`` after rounding.
_RANGE_TOL = 1e-6


def _name(context: str, base: str) -> str:
    """Prefix ``base`` with the caller ``context`` (empty context → base)."""
    return f"{context}: {base}" if context else base


def _sum_tolerance(dtype: torch.dtype) -> float:
    """Absolute tolerance for a probability row summing to 1.

    Scales with the dtype: fp32 rows from ``masked_softmax`` miss 1 by a
    few ULPs (~1e-6 for realistic K), while half-precision rows (the
    Qwen3.5 hybrid battery runs bf16) carry a much larger rounding budget.
    """
    return max(1e-5, 8.0 * float(torch.finfo(dtype).eps))


class ValidationEngine:
    """Runtime invariant checks for JevX forward boundaries (audit Q-003).

    Stateless: construct once (``JevXForDecision`` keeps an instance as
    ``model.validator``) and call the two methods from any boundary that
    needs the L5 guardrail. Never mutates its inputs and never draws from
    any RNG, so enabling/disabling validation cannot change results.

    Args:
        allow_unbounded_confidence: Skip the ``confidence ∈ [0, 1]``
            range check. Set for ``confidence_method="energy"`` models,
            whose documented pre-fit value is a raw, unbounded energy
            (docs/CONFIDENCE.md §1); fitted energy maps into ``[0, 1]``
            anyway, so the check would pass for them either way.

    Example:
        >>> import torch
        >>> from jevx.validation import ValidationEngine
        >>> from jevx.modeling_outputs import DecisionOutput
        >>> engine = ValidationEngine()
        >>> probs = torch.tensor([[0.25, 0.75]])
        >>> decision = DecisionOutput(
        ...     logits=torch.zeros(1, 2),
        ...     probabilities=probs,
        ...     confidence=torch.tensor([0.75]),
        ...     predicted_index=torch.tensor([1]),
        ...     schema_metadata={"decision_type": "choice"},
        ... )
        >>> engine.validate_decision(decision)  # returns None, no raise
    """

    def __init__(self, *, allow_unbounded_confidence: bool = False) -> None:
        self.allow_unbounded_confidence = bool(allow_unbounded_confidence)

    # -- representation -------------------------------------------------------

    def validate_representation(self, rep: Any, *, context: str = "") -> None:
        """Validate one extracted state representation (audit Q-003).

        Contract: after ``adapter.extract_representation`` succeeds, the
        representation must expose a finite ``[B, D]`` pooled state ``H_S``
        (the vector fusion consumes) with optional, aligned sequence
        ``[B, T, D]`` and validity mask ``[B, T]`` companions.

        Args:
            rep: A :class:`jevx.adapters.Representation` (duck-typed via
                its ``.pooled`` attribute) or a bare pooled tensor
                ``[B, D]``.
            context: Prefix for every error message (the call site, e.g.
                ``"forward: state extraction"``).

        Returns:
            None — all invariants hold.

        Raises:
            TypeError: ``rep`` is neither a Representation nor a tensor,
                or a field is not a tensor.
            ValueError: Missing pooled vector, wrong rank/dtype, non-
                finite entries, sequence/mask misalignment, or batch-size
                disagreement; the message names the field, the shape
                found, and the invariant.
        """
        if isinstance(rep, torch.Tensor):
            self._validate_pooled(rep, _name(context, "state tensor"))
            return
        if not hasattr(rep, "pooled"):
            raise TypeError(
                _name(
                    context,
                    "validate_representation: expected a Representation (with a "
                    f".pooled [B, D] tensor) or a bare state tensor, got "
                    f"{type(rep).__name__}.",
                )
            )
        layout = getattr(rep, "layout", None)
        if layout is not None and layout not in ("bd", "btd"):
            raise ValueError(
                _name(
                    context,
                    f"representation layout: expected 'bd' or 'btd', got {layout!r}.",
                )
            )
        pooled = rep.pooled
        sequence = getattr(rep, "sequence_output", None)
        mask = getattr(rep, "attention_mask", None)
        if pooled is None:
            raise ValueError(
                _name(
                    context,
                    "the adapter produced no pooled state vector H_S — JevX fusion "
                    "consumes a [B, D] state; check config.pooling_method and the "
                    "capability report (representation_field) of this backbone.",
                )
            )
        self._validate_pooled(pooled, _name(context, "pooled state H_S"))
        if sequence is not None:
            if not isinstance(sequence, torch.Tensor):
                raise TypeError(
                    _name(
                        context,
                        f"sequence_output: expected a torch.Tensor or None, got "
                        f"{type(sequence).__name__}.",
                    )
                )
            if not sequence.is_floating_point():
                raise ValueError(
                    _name(
                        context,
                        f"sequence_output: expected a floating-point tensor, got "
                        f"dtype {sequence.dtype}.",
                    )
                )
            if sequence.ndim != 3 or int(sequence.shape[-1]) < 1:
                raise ValueError(
                    _name(
                        context,
                        "sequence_output: expected an unpooled [B, T, D] state "
                        f"sequence, got shape {tuple(sequence.shape)}.",
                    )
                )
            check_finite(sequence, _name(context, "sequence_output H_S^seq"))
        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                raise TypeError(
                    _name(
                        context,
                        f"attention_mask: expected a torch.Tensor or None, got "
                        f"{type(mask).__name__}.",
                    )
                )
            if mask.ndim != 2:
                raise ValueError(
                    _name(
                        context,
                        "attention_mask: expected a [B, T] token-validity mask, got "
                        f"shape {tuple(mask.shape)}.",
                    )
                )
            if sequence is not None and tuple(mask.shape) != tuple(sequence.shape[:2]):
                raise ValueError(
                    _name(
                        context,
                        f"attention_mask: shape {tuple(mask.shape)} does not align "
                        f"with sequence_output {tuple(sequence.shape[:2])} — the mask "
                        "indexes the sequence's [B, T] axes.",
                    )
                )
        assert_batch_alignment(
            pooled=pooled, sequence_output=sequence, attention_mask=mask
        )

    @staticmethod
    def _validate_pooled(pooled: Any, name: str) -> None:
        """Type/rank/finiteness checks shared by both representation entries."""
        if not isinstance(pooled, torch.Tensor):
            raise TypeError(
                f"{name}: expected a torch.Tensor, got {type(pooled).__name__}."
            )
        if not pooled.is_floating_point():
            raise ValueError(
                f"{name}: expected a floating-point tensor, got dtype {pooled.dtype}."
            )
        if pooled.ndim != 2:
            raise ValueError(
                f"{name}: expected a [B, D] state tensor, got shape "
                f"{tuple(pooled.shape)}."
            )
        if int(pooled.shape[0]) < 1 or int(pooled.shape[1]) < 1:
            raise ValueError(
                f"{name}: expected a non-empty [B, D] tensor, got shape "
                f"{tuple(pooled.shape)}."
            )
        check_finite(pooled, name)

    # -- decisions ------------------------------------------------------------

    def validate_decision(self, decision_or_output: Any, *, context: str = "") -> None:
        """Validate one decision (or every decision of a forward output).

        Contract: a decision produced by ``JevXForDecision.forward`` must
        carry shape-sane, finite tensors whose values respect the
        per-type probability invariants —
        **choice/score**: rows form a simplex (``sum_i p_i = 1 ± tol``,
        masked columns contribute exactly 0 — per question, never across
        questions, axiom (11.3)); **noul**: a single column strictly
        inside ``(0, 1)`` (T4 / head clamp); plus ``confidence ∈ [0, 1]``
        and ``predicted_index`` inside the option range (``{0, 1}`` for
        noul).

        Args:
            decision_or_output: A :class:`jevx.DecisionOutput`, a
                :class:`jevx.JevXOutput` (its ``decisions`` list is
                validated entry by entry), or a non-empty list/tuple of
                decisions.
            context: Prefix for every error message (the call site, e.g.
                ``"forward: decision pair 0 (choice)"``).

        Returns:
            None — all invariants hold.

        Raises:
            TypeError: Not a decision-like object (no ``.logits``), or a
                field has the wrong type/dtype.
            ValueError: Shape, finiteness, range or per-type invariant
                violations; the message names the tensor, the value found
                and the invariant that failed.
        """
        decisions = self._collect_decisions(decision_or_output, context)
        for position, decision in enumerate(decisions):
            scope = f"{context}[{position}]" if (context and len(decisions) > 1) else context
            self._validate_one(decision, scope)

    @staticmethod
    def _collect_decisions(decision_or_output: Any, context: str) -> List[Any]:
        """Normalize the accepted inputs to a non-empty list of decisions."""
        if isinstance(decision_or_output, (list, tuple)):
            if len(decision_or_output) == 0:
                raise ValueError(
                    _name(context, "validate_decision: got an empty decision list.")
                )
            return list(decision_or_output)
        if hasattr(decision_or_output, "decisions"):
            inner = decision_or_output.decisions
            if isinstance(inner, torch.Tensor):
                raise TypeError(
                    _name(
                        context,
                        "validate_decision: stacked-tensor `decisions` are not "
                        "supported — validate the DecisionOutput entries instead.",
                    )
                )
            if not isinstance(inner, (list, tuple)):
                raise TypeError(
                    _name(
                        context,
                        "validate_decision: output.decisions must be a list of "
                        f"DecisionOutput, got {type(inner).__name__}.",
                    )
                )
            if len(inner) == 0:
                raise ValueError(
                    _name(context, "validate_decision: output carries no decisions.")
                )
            return list(inner)
        if hasattr(decision_or_output, "logits"):
            return [decision_or_output]
        raise TypeError(
            _name(
                context,
                "validate_decision: expected a DecisionOutput, a JevXOutput or a "
                f"list of decisions, got {type(decision_or_output).__name__}.",
            )
        )

    def _validate_one(self, decision: Any, context: str) -> None:
        """Run every check for a single DecisionOutput-like object."""
        logits = getattr(decision, "logits", None)
        if not isinstance(logits, torch.Tensor):
            raise TypeError(
                _name(
                    context,
                    f"decision logits: expected a torch.Tensor, got "
                    f"{type(logits).__name__}.",
                )
            )
        if not logits.is_floating_point():
            raise ValueError(
                _name(
                    context,
                    f"decision logits: expected a floating-point tensor, got dtype "
                    f"{logits.dtype}.",
                )
            )
        if logits.ndim != 2 or int(logits.shape[0]) < 1 or int(logits.shape[1]) < 1:
            raise ValueError(
                _name(
                    context,
                    "decision logits: expected a non-empty [n, K] tensor, got shape "
                    f"{tuple(logits.shape)}.",
                )
            )
        n, k = int(logits.shape[0]), int(logits.shape[1])
        check_finite(logits, _name(context, "decision logits"))

        probs = getattr(decision, "probabilities", None)
        if probs is not None:
            if not isinstance(probs, torch.Tensor):
                raise TypeError(
                    _name(
                        context,
                        "decision probabilities: expected a torch.Tensor or None, "
                        f"got {type(probs).__name__}.",
                    )
                )
            if tuple(probs.shape) != tuple(logits.shape):
                raise ValueError(
                    _name(
                        context,
                        f"decision probabilities: expected the logits shape "
                        f"{tuple(logits.shape)} (same distribution space), got "
                        f"{tuple(probs.shape)}.",
                    )
                )
            if not probs.is_floating_point():
                raise ValueError(
                    _name(
                        context,
                        f"decision probabilities: expected a floating-point tensor, "
                        f"got dtype {probs.dtype}.",
                    )
                )
            check_finite(probs, _name(context, "decision probabilities"))
            # Value reads only — never backprop through validation (also
            # avoids requires_grad->scalar warnings during training).
            probs = probs.detach()
            p_min, p_max = float(probs.min()), float(probs.max())
            if p_min < -_RANGE_TOL or p_max > 1.0 + _RANGE_TOL:
                raise ValueError(
                    _name(
                        context,
                        "decision probabilities must lie in [0, 1] — found "
                        f"min={p_min:.6g}, max={p_max:.6g} in shape "
                        f"{tuple(probs.shape)}; a value outside [0, 1] means the "
                        "probability map or a calibrator broke the range contract.",
                    )
                )

        schema_meta = getattr(decision, "schema_metadata", None)
        dtype = schema_meta.get("decision_type") if isinstance(schema_meta, dict) else None

        if dtype in ("choice", "score"):
            if probs is None:
                raise ValueError(
                    _name(
                        context,
                        f"{dtype!r} decision carries probabilities=None — every "
                        f"{dtype} decision must expose its distribution (shape "
                        f"{tuple(logits.shape)}).",
                    )
                )
            tol = _sum_tolerance(probs.dtype)
            row_sums = probs.sum(dim=-1)
            deviation = (row_sums - 1.0).abs()
            bad = deviation > tol
            if bool(bad.any()):
                first = int(bad.nonzero(as_tuple=False)[0])
                raise ValueError(
                    _name(
                        context,
                        f"{dtype} probabilities must form a simplex — each question's "
                        f"row sums to 1 ± {tol:g} (masked columns contribute 0; "
                        f"per-question normalization, D8.I/(11.3)) — row {first} sums "
                        f"to {float(row_sums[first]):.6g} (off by "
                        f"{float(deviation[first]):.3g}); shape "
                        f"{tuple(probs.shape)}.",
                    )
                )
        elif dtype == "noul":
            if probs is None:
                raise ValueError(
                    _name(
                        context,
                        "noul decision carries probabilities=None — a noul decision "
                        "must expose its single sigmoid probability.",
                    )
                )
            if int(probs.shape[-1]) != 1:
                raise ValueError(
                    _name(
                        context,
                        "noul probabilities: expected exactly ONE sigmoid column "
                        f"(K=1), got shape {tuple(probs.shape)}.",
                    )
                )
            n_min, n_max = float(probs.min()), float(probs.max())
            if n_min <= 0.0 or n_max >= 1.0:
                raise ValueError(
                    _name(
                        context,
                        "noul probabilities must lie strictly inside (0, 1) "
                        "(T4 / the head's eps-clamp invariant) — found "
                        f"min={n_min:.6g}, max={n_max:.6g}.",
                    )
                )

        conf = getattr(decision, "confidence", None)
        if conf is not None:
            if not isinstance(conf, torch.Tensor):
                raise TypeError(
                    _name(
                        context,
                        f"decision confidence: expected a torch.Tensor or None, got "
                        f"{type(conf).__name__}.",
                    )
                )
            if conf.ndim != 1 or int(conf.shape[0]) != n:
                raise ValueError(
                    _name(
                        context,
                        f"decision confidence: expected shape [{n}] (one value per "
                        f"logits row), got {tuple(conf.shape)}.",
                    )
                )
            if not conf.is_floating_point():
                raise ValueError(
                    _name(
                        context,
                        f"decision confidence: expected a floating-point tensor, got "
                        f"dtype {conf.dtype}.",
                    )
                )
            check_finite(conf, _name(context, "decision confidence"))
            conf = conf.detach()
            if not self.allow_unbounded_confidence:
                c_min, c_max = float(conf.min()), float(conf.max())
                if c_min < -_RANGE_TOL or c_max > 1.0 + _RANGE_TOL:
                    raise ValueError(
                        _name(
                            context,
                            "decision confidence must lie in [0, 1] — found "
                            f"min={c_min:.6g}, max={c_max:.6g}. A value outside "
                            "[0, 1] is the documented pre-fit value of "
                            "confidence_method='energy' only (docs/CONFIDENCE.md §1): "
                            "fit the confidence engine first "
                            "(engine.fit(calibration_logits)) or pick another method.",
                        )
                    )

        idx = getattr(decision, "predicted_index", None)
        if idx is not None:
            if not isinstance(idx, torch.Tensor):
                raise TypeError(
                    _name(
                        context,
                        f"decision predicted_index: expected a torch.Tensor or None, "
                        f"got {type(idx).__name__}.",
                    )
                )
            if idx.dtype == torch.bool or idx.is_floating_point() or idx.is_complex():
                raise ValueError(
                    _name(
                        context,
                        "decision predicted_index: expected an integer index tensor, "
                        f"got dtype {idx.dtype}.",
                    )
                )
            if idx.ndim != 1 or int(idx.shape[0]) != n:
                raise ValueError(
                    _name(
                        context,
                        f"decision predicted_index: expected shape [{n}] (one index "
                        f"per logits row), got {tuple(idx.shape)}.",
                    )
                )
            idx = idx.detach()
            # noul thresholds into {0, 1}; tag-less single-column decisions may
            # be either a K=1 choice (index 0) or a noul (index 0/1); every
            # other head contracts predicted_index into [0, K).
            upper = 1 if (dtype == "noul" or (dtype is None and k == 1)) else k - 1
            i_min, i_max = int(idx.min()), int(idx.max())
            if i_min < 0 or i_max > upper:
                type_note = f", decision_type={dtype!r}" if dtype else ""
                raise ValueError(
                    _name(
                        context,
                        f"decision predicted_index out of range: expected values in "
                        f"[0, {upper}] (K={k}{type_note}), found min={i_min}, "
                        f"max={i_max}.",
                    )
                )

        score = getattr(decision, "expected_score", None)
        if score is not None:
            if not isinstance(score, torch.Tensor):
                raise TypeError(
                    _name(
                        context,
                        f"decision expected_score: expected a torch.Tensor or None, "
                        f"got {type(score).__name__}.",
                    )
                )
            if score.is_floating_point():
                if score.ndim != 1 or int(score.shape[0]) != n:
                    raise ValueError(
                        _name(
                            context,
                            f"decision expected_score: expected shape [{n}], got "
                            f"{tuple(score.shape)}.",
                        )
                    )
                check_finite(score, _name(context, "decision expected_score"))

        abstained = getattr(decision, "abstained", None)
        if abstained is not None:
            if not isinstance(abstained, torch.Tensor):
                raise TypeError(
                    _name(
                        context,
                        f"decision abstained: expected a torch.Tensor or None, got "
                        f"{type(abstained).__name__}.",
                    )
                )
            if abstained.dtype != torch.bool:
                raise ValueError(
                    _name(
                        context,
                        "decision abstained: expected a torch.bool mask, got dtype "
                        f"{abstained.dtype}.",
                    )
                )
            if abstained.ndim != 1 or int(abstained.shape[0]) != n:
                raise ValueError(
                    _name(
                        context,
                        f"decision abstained: expected shape [{n}], got "
                        f"{tuple(abstained.shape)}.",
                    )
                )
