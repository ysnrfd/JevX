# Copyright (c) 2026 ysnrfd
"""Executable equivalence checkers (formalizing spec §70–§71).

Every checker here returns a **plain dict of measured values** — no
assertions about pass/fail live inside (structural contract violations such
as comparing tensors on different devices raise ``ValueError``; the
*numerical verdict* is a returned value the caller decides on and records in
a :class:`jevx.conversion.report.ConversionReport`).

Checker semantics:

* :func:`check_base_equivalence` — **functional equivalence of the wrap**:
  ``jevx.base_forward(**inputs)`` vs an independently obtained
  ``base_model(**inputs)`` under ``eval()`` + ``no_grad``. The reference may
  be an ``nn.Module`` (called) or a precomputed reference output (e.g. a
  snapshot taken *before* wrapping, which is what makes the check sensitive
  to a wrap that mutated weights).
* :func:`check_representation_equivalence` — **state-representation
  correctness** (§71): ``adapter.extract_representation`` vs an independent
  reference computed from a raw backbone forward (manual fp32 pooling, the
  pattern of ``tests/integration/test_qwen35_hybrid.py`` item 2). Tolerance
  philosophy there: report the measured error, a bound of ``2 * ulp`` at the
  observed magnitude (floor ``1e-6``), and the rationale — the stored pooled
  tensor can only be as precise as its dtype.
* :func:`check_decision_roundtrip` — **same-call determinism**: calling a
  model twice with identical arguments must produce bitwise-identical
  outputs in eval mode (the local prerequisite for any behavioral claim).
* :func:`check_functional_equivalence` — orchestration entry: derives probe
  inputs when possible, and records an explicit *skip with reason* when no
  inputs or no independent reference are available (skipped != passed).

Relative-error guard: elementwise ``|a-b| / max(|b|, dtype.tiny)`` so an
identically-zero reference cannot divide by zero.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import fields as _dataclass_fields, is_dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from jevx.utils.shapes import assert_same_device_dtype

__all__ = [
    "PROBE_TEXT",
    "check_base_equivalence",
    "check_decision_roundtrip",
    "check_functional_equivalence",
    "check_representation_equivalence",
    "derive_probe_inputs",
]

#: Fixed short text used when a tokenizer is available (deterministic probe).
PROBE_TEXT = "hello world"

#: Pooling strategies with an independent reference implementation below.
_MANUAL_POOL_STRATEGIES = ("masked_mean", "encoder_pool", "mean", "cls", "last_valid")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _flatten_tensors(obj: Any, prefix: str = "") -> Dict[str, torch.Tensor]:
    """Named-tensor view of a (possibly nested) model output.

    Recurses through mappings/``ModelOutput`` (both are ``dict``-likes),
    lists/tuples and dataclasses; non-tensor leaves (strings, metadata
    dicts' scalar values, ``None``) are skipped. Keys are dotted paths.
    """
    out: Dict[str, torch.Tensor] = {}
    if obj is None:
        return out
    if isinstance(obj, torch.Tensor):
        out[prefix or "<root>"] = obj
        return out
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            out.update(_flatten_tensors(value, f"{prefix}.{key}" if prefix else str(key)))
        return out
    if isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            out.update(_flatten_tensors(value, f"{prefix}.{i}"))
        return out
    if is_dataclass(obj) and not isinstance(obj, type):
        for f in _dataclass_fields(obj):
            out.update(_flatten_tensors(getattr(obj, f.name, None), f"{prefix}.{f.name}"))
        return out
    if hasattr(obj, "items") and callable(obj.items):  # ModelOutput-like
        try:
            for key, value in obj.items():
                out.update(
                    _flatten_tensors(value, f"{prefix}.{key}" if prefix else str(key))
                )
        except Exception:  # noqa: BLE001 - exotic output; nothing to extract
            return out
    return out


def _common_tensors(
    a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]
) -> list[Tuple[str, torch.Tensor, torch.Tensor]]:
    """Pair up tensors present in both outputs; raise when there is nothing to compare."""
    common = sorted(set(a) & set(b))
    if not common:
        raise ValueError(
            "equivalence check: the two outputs share no tensor fields to compare "
            f"(left: {sorted(a)[:8] or 'none'}, right: {sorted(b)[:8] or 'none'})."
        )
    pairs = []
    for name in common:
        left, right = a[name], b[name]
        if tuple(left.shape) != tuple(right.shape):
            raise ValueError(
                f"equivalence check: tensor {name!r} has shape {tuple(left.shape)} on one "
                f"side and {tuple(right.shape)} on the other — not comparable."
            )
        assert_same_device_dtype(
            left, right, names=[f"measured.{name}", f"reference.{name}"]
        )
        pairs.append((name, left, right))
    return pairs


def _diff_metrics(pairs: list[Tuple[str, torch.Tensor, torch.Tensor]]) -> Dict[str, Any]:
    """max_abs / mean_abs / rel / bitwise over paired tensors (metrics in float64)."""
    max_abs = 0.0
    mean_abs = 0.0
    rel = 0.0
    total_count = 0
    per_tensor: Dict[str, float] = {}
    bitwise = True
    for name, left, right in pairs:
        if not torch.equal(left, right):
            bitwise = False
        ld = left.detach().double()
        rd = right.detach().double()
        diff = (ld - rd).abs()
        tensor_max = float(diff.max()) if diff.numel() else 0.0
        per_tensor[name] = tensor_max
        max_abs = max(max_abs, tensor_max)
        mean_abs += float(diff.sum())
        total_count += diff.numel()
        # Guarded elementwise relative error (zero reference -> huge but finite).
        denom = rd.abs().clamp_min(torch.finfo(torch.float64).tiny)
        rel = max(rel, float((diff / denom).max()) if diff.numel() else 0.0)
    if total_count:
        mean_abs /= total_count
    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rel": rel,
        "bitwise": bool(bitwise),
        "per_tensor_max_abs": per_tensor,
        "n_tensors": len(pairs),
    }


def _eval_no_grad(models: list[nn.Module]) -> Callable[[], None]:
    """Put modules in eval under no_grad; returns a restorer of prior state."""
    was_training = [bool(m.training) for m in models]
    for m in models:
        m.eval()

    def restore() -> None:
        for m, was in zip(models, was_training):
            if was:
                m.train()

    return restore


def _ulp(value: float, dtype: torch.dtype) -> float:
    """Unit in the last place of ``value`` in ``dtype`` (``eps * 2^floor(log2|v|)``).

    ``torch.finfo(dtype).eps`` equals the spacing at 1.0 for every dtype
    torch exposes here (fp32 ``2**-23``, bf16 ``0.0078125 == 2**-7``), which
    is the same quantity ``bf16_ulp`` computes in the Qwen3.5 hybrid test.
    """
    v = abs(float(value))
    if v == 0.0 or not math.isfinite(v):
        return 0.0
    return float(torch.finfo(dtype).eps) * (2.0 ** math.floor(math.log2(v)))


def _skip(name: str, reason: str) -> Dict[str, Any]:
    """A skipped check dict — ``passed=None`` + reason, ready for ``add_check``."""
    return {
        "name": name,
        "value": None,
        "tolerance": None,
        "passed": None,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# probe-input derivation
# ---------------------------------------------------------------------------


def derive_probe_inputs(
    model: nn.Module,
    *,
    tokenizer: Any = None,
    max_len: int = 8,
    seed: int = 0,
) -> Optional[Dict[str, torch.Tensor]]:
    """Cheap, deterministic probe inputs for a base model (or wrapper child).

    Ladder (first success wins; ``None`` means "cannot derive" — the caller
    records an explicit skip, never a fabricated pass):

    1. ``tokenizer`` (if given) tokenizes the fixed :data:`PROBE_TEXT`;
    2. otherwise, if ``forward`` declares ``input_ids`` and the config
       carries a ``vocab_size``, seeded random token ids of length
       ``min(max_len, learned position table)`` are built — the
       "tokenizer-less probe".

    Returns:
        Kwargs dict for ``model(**inputs)``, or ``None``.
    """
    if not isinstance(model, nn.Module):
        raise TypeError(f"model: expected an nn.Module, got {type(model).__name__}.")
    if tokenizer is not None:
        try:
            encoded = tokenizer(PROBE_TEXT, return_tensors="pt")
            if isinstance(encoded, Mapping) and "input_ids" in encoded:
                return {k: v for k, v in dict(encoded).items() if isinstance(v, torch.Tensor)}
        except Exception:  # noqa: BLE001 - fall through to the id probe
            pass
    try:
        sig = inspect.signature(model.forward)
        declares_ids = "input_ids" in sig.parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        declares_ids = False
    if not declares_ids:
        return None
    config = getattr(model, "config", None)
    vocab = getattr(config, "vocab_size", None)
    if isinstance(vocab, dict):  # wrapper embedding a base config dict
        vocab = vocab.get("vocab_size")
    if not isinstance(vocab, int) or isinstance(vocab, bool) or vocab <= 1:
        return None
    length = max_len
    for key in ("max_position_embeddings", "n_positions", "max_seq_len"):
        limit = getattr(config, key, None)
        if isinstance(limit, dict):
            limit = limit.get(key)
        if isinstance(limit, int) and not isinstance(limit, bool) and limit > 0:
            length = min(length, limit)
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, vocab, (1, max(1, length)), generator=generator)
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


# ---------------------------------------------------------------------------
# checkers
# ---------------------------------------------------------------------------


def check_base_equivalence(
    jevx_model: Any,
    base_model: Any,
    inputs: Mapping[str, Any],
    *,
    atol: float = 0.0,
) -> Dict[str, Any]:
    """Measure ``jevx.base_forward(**inputs)`` vs ``base_model(**inputs)`` (§70).

    Semantic contract: *functional equivalence of the wrap* — the backbone
    embedded in the JevX wrapper computes the same function as the reference
    on identical inputs. Both sides run under ``eval()`` + ``no_grad``.

    Args:
        jevx_model: A ``JevXForDecision`` (anything exposing ``base_forward``).
        base_model: The reference — an ``nn.Module`` (called with
            ``**inputs``) or an already-computed reference output (e.g. a
            pre-wrap snapshot, which is what detects a wrap that mutated
            weights).
        inputs: Model kwargs; must be a non-empty mapping.
        atol: Absolute tolerance for ``passed`` (default ``0.0`` — bitwise
            equality is the bar for a weight-preserving wrap).

    Returns:
        Dict with ``value``/``max_abs`` (worst-case absolute error),
        ``mean_abs``, ``rel`` (guarded), ``bitwise``, ``tolerance``,
        ``passed`` (= ``max_abs <= atol``), ``n_tensors``, ``device``,
        ``dtype`` and ``reference`` (what the reference side was).

    Raises:
        TypeError / ValueError: Structural contract violations only — empty
            inputs, missing ``base_forward``, no overlapping tensor fields,
            or tensors living on different devices / dtypes / shapes. The
            numerical verdict itself is returned, never raised.
    """
    if not isinstance(inputs, Mapping) or not inputs:
        raise ValueError(
            f"inputs: expected a non-empty mapping of model kwargs, got {inputs!r}."
        )
    forward = getattr(jevx_model, "base_forward", None)
    if not callable(forward):
        raise TypeError(
            f"jevx_model: expected an object with a callable base_forward(), got "
            f"{type(jevx_model).__name__}."
        )
    reference_is_module = isinstance(base_model, nn.Module)
    reference_name = (
        f"{type(base_model).__name__}(module)"
        if reference_is_module
        else type(base_model).__name__
    )
    modules = [jevx_model] if isinstance(jevx_model, nn.Module) else []
    if reference_is_module:
        modules.append(base_model)
    restore = _eval_no_grad(modules) if modules else (lambda: None)
    try:
        with torch.no_grad():
            measured = forward(**inputs)
            reference = (
                base_model(**inputs) if callable(base_model) else base_model
            )
    finally:
        restore()
    pairs = _common_tensors(_flatten_tensors(measured), _flatten_tensors(reference))
    metrics = _diff_metrics(pairs)
    atol = float(atol)
    first = pairs[0][1]
    return {
        "value": metrics["max_abs"],
        "max_abs": metrics["max_abs"],
        "mean_abs": metrics["mean_abs"],
        "rel": metrics["rel"],
        "bitwise": metrics["bitwise"],
        "tolerance": atol,
        "passed": bool(metrics["max_abs"] <= atol),
        "n_tensors": metrics["n_tensors"],
        "per_tensor_max_abs": metrics["per_tensor_max_abs"],
        "device": str(first.device),
        "dtype": str(first.dtype),
        "reference": reference_name,
    }


def check_representation_equivalence(
    jevx_model: Any,
    inputs: Mapping[str, Any],
    *,
    pool_fn: Optional[Callable[..., torch.Tensor]] = None,
    atol: Optional[float] = None,
) -> Dict[str, Any]:
    """Measure adapter-extracted ``H_S`` vs an independent reference (§71).

    Semantic contract: *state-representation correctness* — the adapter's
    :meth:`~jevx.adapters.base.BaseAdapter.extract_representation` on the
    embedded backbone agrees with a reference computed independently from a
    raw ``base_forward``:

    * ``pool_fn(raw_output, inputs) -> Tensor`` when supplied by the caller;
    * else the raw field the adapter selected (``pooler_output``/semantic
      fields are compared against that field directly);
    * else an independent fp32 implementation of the effective pooling
      strategy (``masked_mean``/``mean``/``cls``/``last_valid``);
    * else an explicit **skip with reason** (no reference implementation for
      that strategy — skipped is not passed).

    Tolerance (Qwen3.5 hybrid test item-2 philosophy): measured ``value``
    against ``max(2 * ulp(magnitude), 1e-6)`` — the stored pooled tensor
    cannot be more precise than its dtype, so two-ulp is the quantization
    floor, not a fudge factor; ``rationale`` spells this out in the result.

    Returns:
        Dict with ``value`` (max abs error), ``mean_abs``, ``tolerance``,
        ``passed``, ``bitwise``, ``field``, ``strategy``, ``reference`` and
        ``rationale`` — or the skip dict (``passed=None`` + ``reason``).
    """
    if not isinstance(inputs, Mapping) or not inputs:
        raise ValueError(
            f"inputs: expected a non-empty mapping of model kwargs, got {inputs!r}."
        )
    backbone = getattr(jevx_model, "model", None)
    adapter = getattr(jevx_model, "adapter", None)
    config = getattr(jevx_model, "config", None)
    forward = getattr(jevx_model, "base_forward", None)
    if backbone is None or adapter is None or not callable(forward):
        raise TypeError(
            "jevx_model: expected a JevX wrapper exposing .model, .adapter and "
            f".base_forward(); got {type(jevx_model).__name__}."
        )
    strategy = getattr(config, "pooling_method", None)
    restore = (
        _eval_no_grad([jevx_model]) if isinstance(jevx_model, nn.Module) else (lambda: None)
    )
    try:
        with torch.no_grad():
            rep = adapter.extract_representation(backbone, dict(inputs), strategy=strategy)
            raw = forward(**inputs)
    finally:
        restore()
    if rep.pooled is None:
        return _skip(
            "representation_equivalence",
            "adapter.extract_representation returned no pooled tensor — state "
            "representation equivalence NOT measured",
        )

    # --- resolve the independent reference ---------------------------------
    reference: Optional[torch.Tensor] = None
    reference_name = ""
    effective = strategy or getattr(adapter, "default_pooling", None)
    if pool_fn is not None:
        reference = pool_fn(raw, inputs)
        reference_name = getattr(pool_fn, "__name__", "pool_fn")
    elif rep.field in ("pooler_output", "sentence_embedding", "text_embeds", "image_embeds"):
        reference = getattr(raw, rep.field, None)
        reference_name = f"raw_output.{rep.field}"
    elif effective in _MANUAL_POOL_STRATEGIES:
        sequence = getattr(raw, rep.field, None)
        if sequence is None:
            sequence = getattr(raw, "last_hidden_state", None)
        if sequence is None:
            sequence = getattr(raw, "encoder_last_hidden_state", None)
        if sequence is not None:
            mask = inputs.get("attention_mask")
            reference = _manual_pool(sequence, mask, effective)
            reference_name = f"independent fp32 {effective} over raw {rep.field}"
    if reference is None:
        return _skip(
            "representation_equivalence",
            f"no independent reference implementation for field={rep.field!r} "
            f"strategy={effective!r} (pass pool_fn=...) — state representation "
            "equivalence NOT measured",
        )

    if tuple(reference.shape) != tuple(rep.pooled.shape):
        raise ValueError(
            f"representation check: reference shape {tuple(reference.shape)} != "
            f"adapter pooled shape {tuple(rep.pooled.shape)} — the reference does not "
            "compute the same quantity."
        )
    measured = rep.pooled.detach().double()
    ref = reference.detach().to(device=measured.device).double()
    diff = (measured - ref).abs()
    value = float(diff.max()) if diff.numel() else 0.0
    mean_abs = float(diff.mean()) if diff.numel() else 0.0
    bitwise = bool(torch.equal(rep.pooled, reference.to(dtype=rep.pooled.dtype)))
    if atol is None:
        magnitude = max(
            float(measured.abs().max()) if measured.numel() else 0.0,
            float(ref.abs().max()) if ref.numel() else 0.0,
        )
        tolerance = max(2.0 * _ulp(magnitude, rep.pooled.dtype), 1e-6)
        rationale = (
            "2-ulp dtype quantization bound at the observed magnitude (stored pooled "
            "tensors round to their dtype), floor 1e-6 — tolerance philosophy of "
            "tests/integration/test_qwen35_hybrid.py item 2"
        )
    else:
        tolerance = float(atol)
        rationale = "caller-supplied atol"
    return {
        "name": "representation_equivalence",
        "value": value,
        "max_abs": value,
        "mean_abs": mean_abs,
        "tolerance": tolerance,
        "passed": bool(value <= tolerance),
        "bitwise": bitwise,
        "field": rep.field,
        "strategy": effective,
        "reference": reference_name,
        "rationale": rationale,
    }


def _manual_pool(
    sequence: torch.Tensor, mask: Optional[torch.Tensor], strategy: str
) -> torch.Tensor:
    """Independent fp32 reference pooling (deliberately NOT ``jevx.pool``).

    Mask-based contract (math spec Definition 4, invariant P4.3): every
    mask-aware strategy is driven by the validity mask, never by a padding
    convention. With the validity set ``I_b = {t : mask[b, t] = 1}``:

    * ``cls``            -> ``X[b, t1]``, ``t1 = min I_b`` (first valid position);
    * ``last_valid``     -> ``X[b, i_b]``, ``i_b = max I_b`` (last valid position);
    * ``masked_mean``/``encoder_pool`` -> mean over ``I_b``; ``mean`` is
      unmasked by definition.

    Left- and right-padded rows holding the same content therefore pool to
    the same vector. Indexing by POSITION instead (``seq[:, 0]``, ``count - 1``) is
    the right-padding assumption this reference must not make: on left-padded
    input it measures a padded position, so the checker would compare the
    adapter against a wrong baseline. ``mask=None`` means every position is
    valid, which degenerates to position 0 / the final position. A
    fully-masked row is undefined (axiom D1.1) and cannot reach here — the
    adapter's own pooling raises on it first.
    """
    seq = sequence.float()
    if mask is None:
        mask = torch.ones(seq.shape[:2], dtype=torch.bool, device=seq.device)
    bool_mask = mask.to(torch.bool)
    positions = torch.arange(seq.shape[1], device=seq.device)
    if strategy == "cls":
        # t1 = min I_b: argmin of the position index, invalid positions
        # pushed past the end of the sequence.
        first = torch.where(bool_mask, positions, seq.shape[1]).argmin(dim=-1)
        return seq.gather(
            1, first.view(-1, 1, 1).expand(-1, 1, seq.shape[-1])
        ).squeeze(1)
    if strategy == "last_valid":
        # i_b = max I_b: argmax of the position index, invalid positions
        # pushed below zero; valid positions increase strictly, so the
        # maximum is the last valid position.
        last = torch.where(bool_mask, positions, -1).argmax(dim=-1)
        return seq.gather(
            1, last.view(-1, 1, 1).expand(-1, 1, seq.shape[-1])
        ).squeeze(1)
    if strategy == "mean":
        return seq.mean(dim=1)
    if strategy in ("masked_mean", "encoder_pool"):
        m = bool_mask.unsqueeze(-1)
        return (seq * m).sum(dim=1) / m.sum(dim=1).clamp(min=1).float()
    raise ValueError(f"_manual_pool: unsupported strategy {strategy!r}.")


def check_decision_roundtrip(
    model: Any,
    *args: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Same call twice -> identical outputs (determinism prerequisite).

    Semantic contract: in ``eval()`` a model is a pure function — two
    identical calls must produce bitwise-identical tensors. This is the
    *local* determinism floor any behavioral-equivalence claim rests on; it
    is NOT by itself evidence that two different systems agree.

    Returns:
        Dict with ``value`` (max abs difference between the two runs),
        ``tolerance`` (``0.0``), ``passed`` (bitwise identity), ``bitwise``,
        ``n_tensors`` and ``per_tensor_max_abs``.
    """
    if not hasattr(model, "__call__"):
        raise TypeError(f"model: expected a callable module, got {type(model).__name__}.")
    restore = _eval_no_grad([model]) if isinstance(model, nn.Module) else (lambda: None)
    try:
        with torch.no_grad():
            first = model(*args, **kwargs)
            second = model(*args, **kwargs)
    finally:
        restore()
    pairs = _common_tensors(_flatten_tensors(first), _flatten_tensors(second))
    metrics = _diff_metrics(pairs)
    return {
        "name": "decision_roundtrip",
        "value": metrics["max_abs"],
        "max_abs": metrics["max_abs"],
        "mean_abs": metrics["mean_abs"],
        "tolerance": 0.0,
        "passed": bool(metrics["bitwise"]),
        "bitwise": metrics["bitwise"],
        "n_tensors": metrics["n_tensors"],
        "per_tensor_max_abs": metrics["per_tensor_max_abs"],
    }


def check_functional_equivalence(
    jevx_model: Any,
    base_model: Any = None,
    inputs: Optional[Mapping[str, Any]] = None,
    *,
    atol: float = 0.0,
    tokenizer: Any = None,
) -> Dict[str, Any]:
    """Orchestrate a functional-equivalence measurement (converter entry point).

    Resolution ladder — each failure mode is an explicit **skip with
    reason** (skipped != passed), never an exception and never a fabricated
    pass:

    1. ``inputs`` missing -> :func:`derive_probe_inputs` against
       ``jevx_model.model`` (base config) with ``tokenizer``;
    2. still missing -> skip ("no featurizer and probe inputs not derivable");
    3. ``base_model`` missing -> skip ("no independent reference supplied");
    4. otherwise -> :func:`check_base_equivalence` and its measured dict.

    Returns:
        The checker dict, or ``{"name", "value": None, "tolerance": None,
        "passed": None, "reason": ...}``.
    """
    name = "functional_equivalence"
    base = getattr(jevx_model, "model", None)
    if inputs is None:
        try:
            inputs = derive_probe_inputs(
                base if isinstance(base, nn.Module) else jevx_model, tokenizer=tokenizer
            )
        except Exception as exc:  # noqa: BLE001 - derivation is best effort
            return _skip(
                name,
                f"probe input derivation failed ({type(exc).__name__}: {exc}) — "
                "functional equivalence NOT measured",
            )
        if inputs is None:
            return _skip(
                name,
                "no featurizer available and probe inputs are not derivable from the "
                "backbone config (needs forward(input_ids=...) + vocab_size) — "
                "functional equivalence NOT measured",
            )
    if base_model is None:
        return _skip(
            name,
            "no independent base-model reference supplied (an in-memory module is "
            "snapshotted before wrapping; an id needs its base checkpoint reloaded) — "
            "functional equivalence NOT measured",
        )
    result = check_base_equivalence(jevx_model, base_model, inputs, atol=atol)
    result["name"] = name
    return result
