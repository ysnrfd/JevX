# Copyright (c) 2026 ysnrfd
"""Honest weight transplant: only mappings that are TRUE for the actual heads.

This module implements the "head grafting (possible, narrow)" branch of the
representation-reuse-vs-transplant distinction (``docs/ARCHITECTURE.md`` §5):
when — and only when — the source checkpoint contains a compatible task head,
its tensors may be transplanted as a **warm start** into a JevX module with a
genuine counterpart. Anything else is reported as *not mapped*; the module
never silently random-initializes while claiming a transplant.

What the mappings are derived from (the actual JevX head parameterizations):

* ``ChoiceHead`` (``jevx/decisions/choice.py``) scores
  ``z_k = f(fused, e_k)`` via the interaction modes of
  ``jevx/decisions/generic.py::_interact`` — ``dot`` (parameter-free),
  ``bilinear`` ``z_k = fused^T W e_k`` (one ``[D, D]`` parameter shared
  across options) or ``mlp``. It has **no option-independent linear term of
  ``fused`` and no bias**: a pretrained classifier's per-class weight rows
  have no counterpart, so a classifier -> ChoiceHead transplant is
  honestly refused (no-mapping report), not improvised.
* ``ScoreHead`` with ``num_levels=K`` scores ``z = level_proj(fused)`` with
  ``level_proj: Linear(D -> K)`` (``score.py``) — the option-independent
  linear map a sequence-classification head parameterizes.
* ``NoulHead`` scores ``logit = w . fused + b`` with ``proj: Linear(D -> 1)``
  (``noul.py``).
* ``MultiLabelHead``/``RegressionHead`` expose ``proj: Linear`` analogously
  (``generic.py``).

Source heads supported: an ``nn.Linear`` itself, or a module exposing
``.classifier`` / ``.score`` / ``.lm_head`` as an ``nn.Linear``
(the ``BertForSequenceClassification``/``RobertaForSequenceClassification``
``score``/``classifier`` and causal-LM ``lm_head`` spellings). Anything else
(e.g. a nested head such as T5's ``classification_head.out_proj``, or a head
with an unrecognizable name) yields an unmapped report listing its state
keys — never a guess.

Mapping plans (``mapping="auto"`` picks the strongest TRUE one):

============  =========================================================
plan          applies when / what is written
============  =========================================================
``linear``    ``K_src == K_tgt`` and ``D_src == D_tgt``: weight and bias
              copied exactly (``target.weight = source.weight`` rows).
``binary_gauge``  ``K_src == 2``, ``K_tgt == 1`` and ``D`` matches:
              ``target.weight = row1 - row0``, ``target.bias = b1 - b0`` —
              exact under softmax/sigmoid gauge invariance:
              ``sigmoid(z1 - z0) == softmax([z0, z1])[1]``.
``bias_only`` classes match but widths don't (or gauge without matching
              ``D``): only the bias is carried — see the probabilistic
              rationale below.
``none``      class counts disagree with no gauge available (or the target
              has no option-independent linear): NOTHING is written.
============  =========================================================

Bias-only rationale (documented probabilistic claim, NOT equivalence): for a
softmax/sigmoid classifier the bias is the option-independent logit offset;
at the training distribution's mean feature it approximates ``log p(y)`` (up
to the feature-mean term), so copying it carries the source model's class
priors into the target as a warm start. The scoring matrix stays fresh
because its rows cannot align across mismatched widths — the report says so
in ``unmapped_source`` + ``warnings``.

Mode assignment (consumed by :func:`jevx.conversion.converter.convert`):

* every parameter of the target module written -> ``HEAD_ATTACHMENT``
  (the source head now fully parameterizes the target);
* a strict subset written -> ``WEIGHT_TRANSPLANT`` (partial warm start);
* nothing written -> ``REPRESENTATION_REUSE`` (honest "no transplant").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .report import ConversionMode, ConversionReport, EquivalenceKind

__all__ = [
    "CompatReport",
    "probe_head_compat",
    "transplant_classifier_head",
    "transplant_state_encoder_proj",
]

#: Attribute spellings of a pretrained classification head that are supported
#: as the transplant SOURCE, in probe order.
_SOURCE_ATTRS: Tuple[str, ...] = ("classifier", "score", "lm_head")

#: Attribute spellings of an option-independent linear scorer on the TARGET,
#: in probe order (``level_proj`` = ScoreHead fixed-level schema, ``proj`` =
#: Noul/MultiLabel/Regression heads; the rest accept a raw HF head as target).
_TARGET_ATTRS: Tuple[str, ...] = ("level_proj", "proj", "classifier", "score", "out_proj")

#: Explicitly supported mapping plans (``mapping=`` argument).
_MAPPING_PLANS = ("auto", "linear", "binary_gauge", "bias_only", "none")


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------


def _find_source_linear(module: nn.Module) -> Optional[Tuple[str, nn.Linear]]:
    """Locate the source classifier's ``nn.Linear`` (see module docstring)."""
    if isinstance(module, nn.Linear):
        return ("self", module)
    for attr in _SOURCE_ATTRS:
        child = getattr(module, attr, None)
        if isinstance(child, nn.Linear):
            return (attr, child)
    return None


def _find_target_linear(target: nn.Module) -> Optional[Tuple[str, nn.Linear]]:
    """Locate the target's option-independent linear scorer, if one exists."""
    if isinstance(target, nn.Linear):
        return ("self", target)
    for attr in _TARGET_ATTRS:
        child = getattr(target, attr, None)
        if isinstance(child, nn.Linear):
            return (attr, child)
    return None


def _describe_target_miss(target: nn.Module) -> str:
    """Why a target has no classifier-compatible linear (ChoiceHead case)."""
    looked = "/".join(_TARGET_ATTRS)
    return (
        f"target {type(target).__name__} exposes no option-independent linear scorer "
        f"(looked for {looked}); interaction heads such as ChoiceHead score options "
        "through bilinear/mlp parameters shared across options — a per-class "
        "classifier weight has no counterpart there, so no mapping is invented"
    )


# ---------------------------------------------------------------------------
# compatibility probe
# ---------------------------------------------------------------------------


@dataclass
class CompatReport:
    """Shape-level compatibility of a source classifier and a target scorer.

    Args:
        source: Human label of the probed source (``ClassName(attr)``).
        target: Human label of the probed target.
        source_attr / target_attr: Attribute the scoring ``Linear`` was found
            under (``None`` when not found).
        source_shape / target_shape: Weight shapes ``[out_features, in_features]``.
        source_out_features / target_out_features: ``K`` (class/level count).
        source_in_features / target_in_features: ``D`` (state width).
        source_bias / target_bias: Whether the linear carries a bias.
        compatible: True iff a full ``linear`` plan is possible (matching
            ``K``, matching ``D``, and the bias can be carried).
        reasons: Human-readable mismatch explanations (empty iff compatible).
    """

    source: str
    target: str
    source_attr: Optional[str] = None
    target_attr: Optional[str] = None
    source_shape: Optional[Tuple[int, ...]] = None
    target_shape: Optional[Tuple[int, ...]] = None
    source_out_features: Optional[int] = None
    target_out_features: Optional[int] = None
    source_in_features: Optional[int] = None
    target_in_features: Optional[int] = None
    source_bias: bool = False
    target_bias: bool = False
    compatible: bool = False
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict of the probe result."""
        return {
            "source": self.source,
            "target": self.target,
            "source_attr": self.source_attr,
            "target_attr": self.target_attr,
            "source_shape": list(self.source_shape) if self.source_shape else None,
            "target_shape": list(self.target_shape) if self.target_shape else None,
            "source_out_features": self.source_out_features,
            "target_out_features": self.target_out_features,
            "source_in_features": self.source_in_features,
            "target_in_features": self.target_in_features,
            "source_bias": self.source_bias,
            "target_bias": self.target_bias,
            "compatible": self.compatible,
            "reasons": list(self.reasons),
        }


def probe_head_compat(source_head: nn.Module, target_head: nn.Module) -> CompatReport:
    """Probe shape compatibility of a source head against a JevX target head.

    Strict diagnostic API (unlike :func:`transplant_classifier_head`, which
    reports instead of raising): this probe raises for garbage input so a
    caller notices immediately.

    Raises:
        TypeError: Either side is not an ``nn.Module``.
        ValueError: The source exposes no classifier-like ``nn.Linear``
            (message names the supported spellings ``classifier`` /
            ``score`` / ``lm_head``).
    """
    if not isinstance(source_head, nn.Module):
        raise TypeError(
            f"probe_head_compat: source_head must be an nn.Module, got "
            f"{type(source_head).__name__}."
        )
    if not isinstance(target_head, nn.Module):
        raise TypeError(
            f"probe_head_compat: target_head must be an nn.Module, got "
            f"{type(target_head).__name__}."
        )
    source = _find_source_linear(source_head)
    if source is None:
        raise ValueError(
            f"probe_head_compat: source_head {type(source_head).__name__} exposes no "
            "classifier-like scoring nn.Linear; supported sources are an nn.Linear "
            "itself or a module with a `.classifier` / `.score` / `.lm_head` "
            "nn.Linear attribute (nested heads such as T5's "
            "classification_head.out_proj are not auto-detected)."
        )
    s_attr, s_lin = source
    t_match = _find_target_linear(target_head)
    source_label = f"{type(source_head).__name__}({s_attr})"
    if t_match is None:
        return CompatReport(
            source=source_label,
            target=f"{type(target_head).__name__}",
            source_attr=s_attr,
            source_shape=tuple(s_lin.weight.shape),
            source_out_features=s_lin.out_features,
            source_in_features=s_lin.in_features,
            source_bias=s_lin.bias is not None,
            target_attr=None,
            compatible=False,
            reasons=[_describe_target_miss(target_head)],
        )
    t_attr, t_lin = t_match
    report = CompatReport(
        source=source_label,
        target=f"{type(target_head).__name__}({t_attr})",
        source_attr=s_attr,
        target_attr=t_attr,
        source_shape=tuple(s_lin.weight.shape),
        target_shape=tuple(t_lin.weight.shape),
        source_out_features=s_lin.out_features,
        target_out_features=t_lin.out_features,
        source_in_features=s_lin.in_features,
        target_in_features=t_lin.in_features,
        source_bias=s_lin.bias is not None,
        target_bias=t_lin.bias is not None,
    )
    if s_lin.out_features != t_lin.out_features:
        report.reasons.append(
            f"class count mismatch: source K={s_lin.out_features} != target "
            f"K={t_lin.out_features} — rows cannot align without an equal number of "
            "classes (and no binary gauge applies unless 2 -> 1)"
        )
    if s_lin.in_features != t_lin.in_features:
        report.reasons.append(
            f"state width mismatch: source D={s_lin.in_features} != target "
            f"D={t_lin.in_features} — weight rows live in different spaces"
        )
    if s_lin.bias is not None and t_lin.bias is None:
        report.reasons.append("source has a bias but the target linear is bias-free")
    report.compatible = not report.reasons
    return report


# ---------------------------------------------------------------------------
# classifier-head transplant
# ---------------------------------------------------------------------------


def transplant_classifier_head(
    source_head: nn.Module,
    target_head: nn.Module,
    *,
    mapping: str = "auto",
) -> ConversionReport:
    """Transplant a pretrained classifier into a JevX head — honestly.

    Semantic contract: this writes **only** tensors whose mapping is true
    for the target's actual parameterization (see module docstring), then
    reports exactly what happened. It never writes into a backbone, never
    guesses a mapping, and never claims equivalence (``equivalence`` stays
    ``NONE`` — transplanting parameters into a differently-shaped pipeline
    is a warm start, not measured agreement).

    Args:
        source_head: ``nn.Linear``, or a module with ``.classifier`` /
            ``.score`` / ``.lm_head`` (``nn.Linear``).
        target_head: A JevX decision head exposing an option-independent
            linear (``ScoreHead(num_levels=...)``, ``NoulHead``,
            ``MultiLabelHead``, ``RegressionHead``) or any module with a
            recognized linear attribute. A ChoiceHead-style interaction head
            is honestly refused.
        mapping: ``"auto"`` (default; strongest true plan), ``"linear"``,
            ``"binary_gauge"``, ``"bias_only"`` or ``"none"`` (force
            refusal). An explicitly requested plan that does not apply
            yields a no-mapping report rather than a wrong write.

    Returns:
        A :class:`~jevx.conversion.report.ConversionReport` whose
        ``mode`` is ``HEAD_ATTACHMENT`` (target fully written),
        ``WEIGHT_TRANSPLANT`` (subset written) or ``REPRESENTATION_REUSE``
        (nothing written); ``mapped_params`` keys are relative to
        ``target_head``; ``checks`` contains a measured
        ``transplant_tensor_identity`` entry whenever tensors were written.
    """
    if mapping not in _MAPPING_PLANS:
        raise ValueError(
            f"mapping: expected one of {list(_MAPPING_PLANS)}, got {mapping!r}."
        )
    if not isinstance(source_head, nn.Module) or not isinstance(target_head, nn.Module):
        raise TypeError(
            "transplant_classifier_head: source_head and target_head must be "
            f"nn.Modules, got {type(source_head).__name__} / "
            f"{type(target_head).__name__}."
        )

    source_state = dict(source_head.state_dict())
    # Head-transplant inventory: a source task head typically embeds a full
    # copy of its backbone (e.g. `bert.*` under BertForSequenceClassification
    # whose `base_model_prefix` is "bert"). Those tensors are NOT head
    # tensors — transplanting the head never carries a second backbone, and
    # the conversion reuses the backbone from its own source. They are
    # excluded from `unmapped_source` (with an explicit warning) so the
    # inventory stays about the head; nothing about them is hidden.
    backbone_prefix = getattr(source_head, "base_model_prefix", None)
    excluded_backbone: List[str] = []
    if isinstance(backbone_prefix, str) and backbone_prefix:
        prefix = backbone_prefix + "."
        excluded_backbone = sorted(k for k in source_state if k.startswith(prefix))
    inventory = {k: v for k, v in source_state.items() if k not in set(excluded_backbone)}

    def report(
        *,
        mode: ConversionMode,
        mapped: Dict[str, str],
        unmapped: List[str],
        fresh: List[str],
        warnings: List[str],
        source_label: str,
        target_label: str,
        checks: Optional[List[Dict[str, Any]]] = None,
    ) -> ConversionReport:
        merged = list(warnings)
        if excluded_backbone:
            merged.append(
                f"source backbone copy excluded from the head-transplant inventory: "
                f"{len(excluded_backbone)} tensor(s) under {backbone_prefix + '.'!r} — "
                "transplant grafts the head only; the conversion reuses the backbone "
                "from its own source, never from the head module"
            )
        return ConversionReport(
            mode=mode,
            equivalence=EquivalenceKind.NONE,
            source=source_label,
            target=target_label,
            mapped_params=mapped,
            unmapped_source=unmapped,
            new_params=fresh,
            checks=list(checks or []),
            warnings=merged,
        )

    source_label = type(source_head).__name__
    target_label = type(target_head).__name__

    source = _find_source_linear(source_head)
    if source is None:
        # Honest refusal: nothing recognizable to transplant. List whatever
        # state keys exist so the caller sees the full inventory as unmapped.
        keys = sorted(inventory) or [f"<{source_label} has no state>"]
        return report(
            mode=ConversionMode.REPRESENTATION_REUSE,
            mapped={},
            unmapped=keys,
            fresh=sorted(target_head.state_dict()),
            warnings=[
                f"source {source_label} exposes no classifier-like scoring nn.Linear "
                "(supported: an nn.Linear itself, or .classifier/.score/.lm_head as "
                "nn.Linear) — nothing was transplanted; the JevX head stays freshly "
                "initialized (representation reuse, NOT a transplant)"
            ],
            source_label=source_label,
            target_label=target_label,
        )

    s_attr, s_lin = source
    s_prefix = "" if s_attr == "self" else f"{s_attr}."
    s_weight_key = f"{s_prefix}weight"
    s_bias_key = f"{s_prefix}bias"

    t_match = _find_target_linear(target_head)
    if t_match is None:
        return report(
            mode=ConversionMode.REPRESENTATION_REUSE,
            mapped={},
            unmapped=sorted(inventory),
            fresh=sorted(target_head.state_dict()),
            warnings=[
                _describe_target_miss(target_head)
                + " — nothing was transplanted (representation reuse, NOT a transplant)"
            ],
            source_label=f"{type(source_head).__name__}({s_attr})",
            target_label=target_label,
        )

    t_attr, t_lin = t_match
    t_prefix = "" if t_attr == "self" else f"{t_attr}."
    t_weight_key = f"{t_prefix}weight"
    t_bias_key = f"{t_prefix}bias"
    source_label = f"{type(source_head).__name__}({s_attr})"
    target_label = f"{type(target_head).__name__}({t_attr})"

    k_s, d_s = s_lin.out_features, s_lin.in_features
    k_t, d_t = t_lin.out_features, t_lin.in_features
    has_s_bias = s_lin.bias is not None
    has_t_bias = t_lin.bias is not None

    # --- choose the strongest TRUE plan ------------------------------------
    plan: Optional[str]
    if mapping == "none":
        plan = None
    elif mapping in ("linear", "binary_gauge", "bias_only"):
        plan = mapping  # validated against applicability below
    else:  # auto
        if k_s == k_t and d_s == d_t:
            plan = "linear"
        elif k_s == 2 and k_t == 1:
            plan = "binary_gauge" if d_s == d_t else "bias_only"
        elif k_s == k_t:
            plan = "bias_only"
        else:
            plan = None

    # Applicability guards for explicitly requested plans (never write a
    # wrong mapping just because the caller named it).
    plan_error = ""
    if plan == "linear" and not (k_s == k_t and d_s == d_t):
        plan_error = (
            f"linear mapping needs matching K and D (source {k_s}x{d_s}, target "
            f"{k_t}x{d_t})"
        )
        plan = None
    elif plan == "binary_gauge" and not (k_s == 2 and k_t == 1 and d_s == d_t):
        plan_error = (
            f"binary_gauge needs source K=2 -> target K=1 with matching D "
            f"(source {k_s}x{d_s}, target {k_t}x{d_t})"
        )
        plan = None
    elif plan == "bias_only" and not (k_s == k_t or (k_s == 2 and k_t == 1)):
        plan_error = (
            f"bias_only needs matching class counts (or a 2 -> 1 binary gauge); "
            f"source K={k_s}, target K={k_t}"
        )
        plan = None

    warnings: List[str] = []
    mapped: Dict[str, str] = {}
    written: Dict[str, torch.Tensor] = {}  # target key -> expected value (source dtype)

    if plan is None:
        if mapping == "none":
            reason = "mapping='none' requested — nothing was transplanted"
        elif plan_error:
            reason = f"{plan_error} — nothing was transplanted"
        else:
            reason = (
                f"class counts disagree with no gauge available (source K={k_s}, "
                f"target K={k_t}) — no mapping exists, so nothing was transplanted"
            )
        warnings.append(
            f"{reason}; the JevX head stays freshly initialized — this is "
            "representation reuse, NOT a transplant"
        )
        return report(
            mode=ConversionMode.REPRESENTATION_REUSE,
            mapped={},
            unmapped=sorted(inventory),
            fresh=sorted(target_head.state_dict()),
            warnings=warnings,
            source_label=source_label,
            target_label=target_label,
        )

    if not has_t_bias and plan != "none":
        # A bias-carrying source cannot write its bias; degrade to weight-only
        # for 'linear' (still true), otherwise refuse.
        if plan == "linear":
            warnings.append(
                "target linear is bias-free — source bias was NOT transplanted"
            )
            has_s_bias_for_write = False
        else:
            return report(
                mode=ConversionMode.REPRESENTATION_REUSE,
                mapped={},
                unmapped=sorted(inventory),
                fresh=sorted(target_head.state_dict()),
                warnings=[
                    f"{plan} mapping requires a target bias but "
                    f"{target_label} is bias-free — nothing was transplanted "
                    "(representation reuse, NOT a transplant)"
                ],
                source_label=source_label,
                target_label=target_label,
            )
    else:
        has_s_bias_for_write = has_s_bias

    # --- write the plan -----------------------------------------------------
    with torch.no_grad():
        if plan == "linear":
            expected_w = s_lin.weight.detach().clone()
            t_lin.weight.copy_(expected_w)
            written[t_weight_key] = expected_w
            mapped[s_weight_key] = t_weight_key
            if has_s_bias_for_write and has_t_bias:
                expected_b = s_lin.bias.detach().clone()
                t_lin.bias.copy_(expected_b)
                written[t_bias_key] = expected_b
                mapped[s_bias_key] = t_bias_key
            warnings.append(
                "linear mapping: target weight/bias copied from the source classifier "
                "rows verbatim — a warm start of the same parameterization, still "
                "subject to validation (ARCHITECTURE §5.3)"
            )
        elif plan == "binary_gauge":
            expected_w = (s_lin.weight[1] - s_lin.weight[0]).detach().clone()
            t_lin.weight.copy_(expected_w.unsqueeze(0))
            written[t_weight_key] = expected_w.unsqueeze(0)
            mapped[s_weight_key] = t_weight_key
            expected_b = (s_lin.bias[1] - s_lin.bias[0]).detach().clone().reshape(1)
            t_lin.bias.copy_(expected_b)
            written[t_bias_key] = expected_b
            mapped[s_bias_key] = t_bias_key
            warnings.append(
                "binary_gauge mapping: target weight/bias = source row1 - row0 "
                "(softmax/sigmoid gauge invariance: sigmoid(z1 - z0) == "
                "softmax([z0, z1])[1]) — exact for a K=2 -> K=1 head, and a warm "
                "start only for the surrounding pipeline"
            )
        elif plan == "bias_only" and k_s == 2 and k_t == 1:
            expected_b = (s_lin.bias[1] - s_lin.bias[0]).detach().clone().reshape(1)
            t_lin.bias.copy_(expected_b)
            written[t_bias_key] = expected_b
            mapped[s_bias_key] = t_bias_key
            warnings.append(
                "bias_only (binary gauge) mapping: target bias = source b1 - b0 — the "
                "class-prior offset under sigmoid gauge invariance; the weight row "
                "stays fresh (widths differ or the gauge plan was requested alone). "
                "Probabilistic rationale: a classifier bias approximates log p(y) up "
                "to the feature-mean term, so this carries class priors as a warm "
                "start — NOT an equivalence claim"
            )
        elif plan == "bias_only":
            expected_b = s_lin.bias.detach().clone()
            t_lin.bias.copy_(expected_b)
            written[t_bias_key] = expected_b
            mapped[s_bias_key] = t_bias_key
            warnings.append(
                f"bias_only mapping: classes match (K={k_s}) but state widths do not "
                f"(source D={d_s} != target D={d_t}) — only the bias was carried; the "
                "weight rows live in different spaces and stay freshly initialized. "
                "Probabilistic rationale: a classifier bias approximates log p(y) up "
                "to the feature-mean term, so this carries class priors as a warm "
                "start — NOT an equivalence claim"
            )
        else:  # pragma: no cover - guarded above; keep the exhaustiveness obvious
            raise RuntimeError(f"transplant: unhandled plan {plan!r}.")

    # --- report -------------------------------------------------------------
    target_state = dict(target_head.state_dict())
    unmapped = sorted(k for k in inventory if k not in mapped)
    fresh = sorted(k for k in target_state if k not in written)
    if unmapped:
        warnings.append(
            f"{len(unmapped)} source tensor(s) had no counterpart and were left "
            f"unmapped: {unmapped}"
        )
    if fresh:
        warnings.append(
            f"{len(fresh)} target tensor(s) remain freshly initialized: {fresh}"
        )
    # Measured identity of every written tensor (post-write, exactness).
    checks: List[Dict[str, Any]] = []
    worst = 0.0
    for t_key, expected in written.items():
        actual = target_state[t_key].detach()
        diff = float((actual.double() - expected.detach().double()).abs().max())
        worst = max(worst, diff)
    checks.append(
        {
            "name": "transplant_tensor_identity",
            "value": worst,
            "tolerance": 0.0,
            "passed": bool(worst == 0.0),
        }
    )
    all_written = not fresh
    return report(
        mode=(
            ConversionMode.HEAD_ATTACHMENT
            if all_written
            else ConversionMode.WEIGHT_TRANSPLANT
        ),
        mapped=mapped,
        unmapped=unmapped,
        fresh=fresh,
        warnings=warnings,
        source_label=source_label,
        target_label=target_label,
        checks=checks,
    )


# ---------------------------------------------------------------------------
# generic tensor-copy transplant
# ---------------------------------------------------------------------------


def transplant_state_encoder_proj(
    source_proj_like: nn.Module,
    target: nn.Module,
    *,
    key_map: Optional[Dict[str, str]] = None,
) -> ConversionReport:
    """Copy source tensors into ``target`` where keys AND shapes match exactly.

    Semantic contract: a generic, structure-preserving tensor copy — for
    each source state-dict key, the tensor is written iff ``target`` has the
    same key with the same shape (optionally re-keyed through ``key_map``).
    No suffix guessing, no reshapes: anything unmatched is reported in
    ``unmapped_source`` with the reason. **The backbone is protected**: when
    ``target`` is a JevX wrapper (has a child module named ``model``),
    target keys under ``model.`` are never written even when they match —
    the backbone is transplanted by identity at load time, never here.

    Uses ``load_state_dict(..., strict=False)``-style diffs to build the
    report: ``mapped_params`` = loaded keys, ``unmapped_source`` = source
    keys not loaded (+ reasons in ``warnings``), ``new_params`` = target
    keys absent from the loaded payload (fresh).

    Args:
        source_proj_like: Module whose state dict supplies tensors (e.g. a
            pooler or a ``Projection``).
        target: Module to write into.
        key_map: Optional ``{source key -> target key}`` renames applied
            before matching (e.g. ``{"dense.weight": "proj.weight"}``).

    Returns:
        ConversionReport: mode ``HEAD_ATTACHMENT``/``WEIGHT_TRANSPLANT`` when
        something was written (full/partial), ``REPRESENTATION_REUSE`` when
        nothing was; ``equivalence`` always ``NONE`` (a copy is not a
        measurement of behavioral agreement).
    """
    if not isinstance(source_proj_like, nn.Module) or not isinstance(target, nn.Module):
        raise TypeError(
            "transplant_state_encoder_proj: both sides must be nn.Modules, got "
            f"{type(source_proj_like).__name__} / {type(target).__name__}."
        )
    key_map = dict(key_map or {})
    source_state = dict(source_proj_like.state_dict())
    target_state = dict(target.state_dict())
    protects_backbone = isinstance(getattr(target, "model", None), nn.Module)

    mapped: Dict[str, str] = {}
    unmapped: List[str] = []
    payload: Dict[str, torch.Tensor] = {}
    warnings: List[str] = []
    protected = 0
    for s_key, tensor in source_state.items():
        t_key = key_map.get(s_key, s_key)
        if protects_backbone and t_key.startswith("model."):
            protected += 1
            unmapped.append(s_key)
            continue
        if t_key not in target_state:
            unmapped.append(s_key)
            continue
        if tuple(target_state[t_key].shape) != tuple(tensor.shape):
            unmapped.append(s_key)
            warnings.append(
                f"shape mismatch for {s_key!r}: source {tuple(tensor.shape)} != "
                f"target {tuple(target_state[t_key].shape)} — not copied"
            )
            continue
        payload[t_key] = tensor
        mapped[s_key] = t_key
    if protected:
        warnings.append(
            f"backbone protection: {protected} source tensor(s) matched keys under "
            "'model.' and were deliberately NOT copied — the embedded backbone is "
            "never overwritten by transplant"
        )

    if payload:
        # strict=False diff-based application (missing/unexpected reported).
        incompat = target.load_state_dict(payload, strict=False)
        missing = list(getattr(incompat, "missing_keys", []) or [])
        fresh = sorted(k for k in missing if k in target_state)
    else:
        fresh = sorted(target_state)

    checks: List[Dict[str, Any]] = []
    if payload:
        worst = 0.0
        after = dict(target.state_dict())
        for t_key, original in payload.items():
            diff = float(
                (after[t_key].detach().double() - original.detach().double()).abs().max()
            )
            worst = max(worst, diff)
        checks.append(
            {
                "name": "transplant_tensor_identity",
                "value": worst,
                "tolerance": 0.0,
                "passed": bool(worst == 0.0),
            }
        )
    if unmapped:
        warnings.append(
            f"{len(unmapped)} source tensor(s) found no exact target key/shape match "
            f"and were left unmapped: {sorted(unmapped)}"
        )
    if not mapped:
        mode = ConversionMode.REPRESENTATION_REUSE
        warnings.append(
            "no tensor was copied — representation reuse, NOT a transplant"
        )
    elif not fresh:
        mode = ConversionMode.HEAD_ATTACHMENT
    else:
        mode = ConversionMode.WEIGHT_TRANSPLANT
    return ConversionReport(
        mode=mode,
        equivalence=EquivalenceKind.NONE,
        source=type(source_proj_like).__name__,
        target=type(target).__name__,
        mapped_params=mapped,
        unmapped_source=sorted(unmapped),
        new_params=fresh,
        checks=checks,
        warnings=warnings,
    )
