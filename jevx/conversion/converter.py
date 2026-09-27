# Copyright (c) 2026 ysnrfd
"""``convert()`` — orchestrate a conversion and report exactly what happened.

This is the §63 factory flow wrapped in honesty bookkeeping: it reuses
:meth:`jevx.modeling_jevx.JevXForDecision.from_pretrained` for id/path
inputs and mirrors ``_wrap_base_model``'s ``JevXConfig`` construction for
in-memory modules (rather than duplicating the loader logic), then records
the operation in a :class:`~jevx.conversion.report.ConversionReport`.

What the report means (semantic contract):

* **Mode** — ``REPRESENTATION_REUSE`` for a fresh wrap (backbone weights
  reused verbatim, JevX heads freshly initialized); ``WEIGHT_TRANSPLANT`` /
  ``HEAD_ATTACHMENT`` when a ``source_head`` was supplied and
  :func:`~jevx.conversion.transplant.transplant_classifier_head` actually
  mapped tensors (subset vs full target parameterization). When the source
  head cannot be mapped, the mode stays ``REPRESENTATION_REUSE`` and the
  report says so — a failed graft is never dressed up as a transplant.
  ``ARCHITECTURAL_CONVERSION`` exists for taxonomy completeness but is
  **never auto-assigned**: this package performs no structural rewrites.
* **Equivalence** — with ``equivalence_check=True`` a probe input is derived
  (attached tokenizer -> fixed text; else a seeded tokenizer-less id probe
  from the backbone config; caller ``probe_inputs=`` wins) and functional
  equivalence of the wrap is *measured* via
  :func:`~jevx.conversion.equivalence.check_functional_equivalence`
  against an independent reference: a **pre-wrap snapshot** of the supplied
  module (sensitive to a wrap that mutated weights), or, for id/path
  inputs, a second load of the base checkpoint through the documented
  Auto-model chain. When no reference or no featurizer exists the check is
  recorded as **skipped with a reason** — skipped is not passed, and no
  exception is raised for a non-measurement (load/construction failures do
  raise).

Argument partitioning for the in-memory path mirrors the factory exactly:
``JevXConfig`` fields are config overrides, ``tokenizer``/``processor``/
``adapter``/``trust_remote_code`` are factory arguments, backbone-loader
kwargs (``dtype``, ``revision``, ...) are rejected with a message naming
the groups (an in-memory module is embedded directly — there is nothing to
load), and anything else raises ``TypeError`` listing all three groups.
"""

from __future__ import annotations

from dataclasses import fields as _config_dataclass_fields
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM

from jevx.configuration_jevx import JevXConfig
from jevx.modeling_jevx import (
    _BACKBONE_LOAD_KWARGS,
    _HIDDEN_WIDTH_KEYS,
    _hidden_width,
    JevXForDecision,
)

from .equivalence import (
    check_functional_equivalence,
    derive_probe_inputs,
)
from .report import ConversionMode, ConversionReport, EquivalenceKind
from .transplant import transplant_classifier_head

__all__ = ["convert"]

#: Factory kwargs accepted alongside the model id (mirror of the factory's
#: documented factory group).
_FACTORY_FIELDS: Tuple[str, ...] = (
    "trust_remote_code",
    "tokenizer",
    "processor",
    "adapter",
)


def _resolve_probe(
    base_model: nn.Module,
    tokenizer: Any,
    probe_inputs: Optional[Mapping[str, Any]],
) -> Optional[Dict[str, torch.Tensor]]:
    """Probe-input ladder: explicit mapping > tokenizer text > seeded ids."""
    if probe_inputs is not None:
        if not isinstance(probe_inputs, Mapping) or not probe_inputs:
            raise ValueError(
                f"probe_inputs: expected a non-empty mapping of model kwargs, got "
                f"{probe_inputs!r}."
            )
        return dict(probe_inputs)
    try:
        return derive_probe_inputs(base_model, tokenizer=tokenizer)
    except Exception:  # noqa: BLE001 - derivation is best effort; skip records it
        return None


def _reload_base_reference(
    model_id: Union[str, Path], kwargs: Dict[str, Any]
) -> Tuple[Optional[nn.Module], list]:
    """Independently reload the base checkpoint (factory's documented chain).

    Returns ``(model, attempts)``; ``model is None`` means every Auto class
    failed and ``attempts`` holds the recorded errors for the skip reason.
    """
    loader_kwargs = {k: kwargs[k] for k in kwargs if k in _BACKBONE_LOAD_KWARGS}
    loader_kwargs["trust_remote_code"] = bool(kwargs.get("trust_remote_code", False))
    attempts = []
    for auto_cls in (AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM):
        try:
            return auto_cls.from_pretrained(model_id, **loader_kwargs), attempts
        except Exception as exc:  # noqa: BLE001 - documented fallback chain
            attempts.append(f"{auto_cls.__name__}: {type(exc).__name__}: {exc}")
    return None, attempts


def _skip_check(name: str, reason: str) -> Dict[str, Any]:
    """A skipped check entry (``passed=None`` + reason)."""
    return {"name": name, "value": None, "tolerance": None, "passed": None, "reason": reason}


def convert(
    model_or_id: Union[str, Path, nn.Module],
    *,
    decision_type: Optional[str] = None,
    equivalence_check: bool = True,
    source_head: Optional[nn.Module] = None,
    probe_inputs: Optional[Mapping[str, torch.Tensor]] = None,
    **from_pretrained_kwargs: Any,
) -> Tuple[JevXForDecision, ConversionReport]:
    """Wrap a base checkpoint (or in-memory module) and report the conversion.

    Args:
        model_or_id: A model id / local path (routed through
            ``JevXForDecision.from_pretrained`` — the §63 factory chain), or
            an in-memory ``nn.Module`` with an HF-style ``config`` (embedded
            directly, mirroring ``_wrap_base_model``'s config construction).
            The module is embedded **by reference** (factory convention) and
            ends up in ``eval()`` with its backbone frozen per
            ``config.freeze_base_model`` — pass ``copy.deepcopy(...)`` if you
            need the original left mutable.
        decision_type: Pin ``config.decision_type`` (id path forwards it to
            the factory; module path sets it on the built config).
        equivalence_check: Measure functional equivalence of the wrap on
            probe inputs (see module docstring). Failures to *derive a
            probe* or to obtain an *independent reference* are recorded as
            skipped checks with reasons — never raised, never faked.
        source_head: Optional pretrained classifier head to graft into the
            wrapper's decision heads via
            :func:`~jevx.conversion.transplant.transplant_classifier_head`.
            Candidate heads are tried in a documented ladder (pinned
            ``decision_type``/``config.decision_type`` defaulting to
            ``"choice"``, then the wrapper's heads in registration order);
            the first head that yields a TRUE mapping wins and the ladder
            taken is recorded in ``warnings``. If none maps, the mode stays
            ``REPRESENTATION_REUSE`` with warnings listing every attempt and
            the full unmapped source inventory.
        probe_inputs: Explicit probe batch (e.g. ``{"input_ids": ...}``)
            overriding tokenizer/config derivation — the "supplied probe"
            for models without a cheap derivable input.
        **from_pretrained_kwargs: Id-path: forwarded verbatim to the
            factory (config fields, loader kwargs, factory kwargs — the
            factory partitions them). Module-path: ``JevXConfig`` fields are
            config overrides; factory fields are honored; backbone-loader
            kwargs and unknowns raise ``TypeError`` naming the groups.

    Returns:
        ``(model, report)`` — the wrapped ``JevXForDecision`` and its
        honesty-checked :class:`~jevx.conversion.report.ConversionReport`.

    Raises:
        TypeError: Malformed arguments (not an id/module, unknown kwargs,
            loader kwargs on the module path).
        ValueError / Exception: Load or construction failures propagate
            unchanged. Equivalence problems never raise — they are reported.
    """
    # ------------------------------------------------------------------ id path
    if isinstance(model_or_id, (str, Path)):
        kwargs = dict(from_pretrained_kwargs)
        trust_remote_code = bool(kwargs.get("trust_remote_code", False))
        user_config = kwargs.get("config")
        if isinstance(user_config, JevXConfig):
            is_reload = True
        else:
            _, is_reload = JevXForDecision._peek_config(
                model_or_id, trust_remote_code, kwargs
            )
        wrapper = JevXForDecision.from_pretrained(
            model_or_id, decision_type=decision_type, **kwargs
        )
        source_label = str(model_or_id)
        loader_kwargs = kwargs
        path_equivalence_kind = "id"
        probe = (
            _resolve_probe(
                wrapper.model, getattr(wrapper, "tokenizer", None), probe_inputs
            )
            if equivalence_check
            else None
        )
    # ------------------------------------------------------------- module path
    elif isinstance(model_or_id, nn.Module):
        if isinstance(model_or_id, JevXForDecision):
            raise TypeError(
                "convert: got an already-wrapped JevXForDecision — reload such a "
                "checkpoint with JevXForDecision.from_pretrained instead of "
                "converting it again."
            )
        model = model_or_id
        config_obj = getattr(model, "config", None)
        if config_obj is None:
            raise TypeError(
                f"convert: in-memory module {type(model).__name__} has no `.config` — "
                "an HF-style config is required to build JevXConfig (pass a model id "
                "or a PreTrainedModel-style module)."
            )
        kwargs = dict(from_pretrained_kwargs)
        # Partition exactly like _wrap_base_model: config fields | factory
        # fields | loader kwargs (meaningless for an in-memory module) | else.
        config_fields = {f.name for f in _config_dataclass_fields(JevXConfig)}
        factory_kwargs = {k: kwargs.pop(k) for k in _FACTORY_FIELDS if k in kwargs}
        loader_hit = sorted(k for k in kwargs if k in _BACKBONE_LOAD_KWARGS)
        if loader_hit:
            raise TypeError(
                f"convert: backbone loader kwargs {loader_hit} only apply to the "
                "id/path path — an in-memory module is embedded directly (convert "
                "its dtype with .to(...) before calling, or pass the id instead)."
            )
        config_overrides = {k: kwargs.pop(k) for k in list(kwargs) if k in config_fields}
        if "config" in kwargs:
            user_config = kwargs.pop("config")
            if not isinstance(user_config, JevXConfig):
                raise TypeError(
                    f"convert: config= must be a JevXConfig, got "
                    f"{type(user_config).__name__}."
                )
        else:
            user_config = None
        if kwargs:
            raise TypeError(
                f"convert: unexpected keyword argument(s) {sorted(kwargs)}. "
                f"JevX config fields: {sorted(config_fields)}; backbone loader "
                f"fields (id path only): {list(_BACKBONE_LOAD_KWARGS)}; factory "
                f"fields: {list(_FACTORY_FIELDS)}."
            )
        width = _hidden_width(config_obj)
        if width is None:
            raise ValueError(
                f"convert: cannot determine hidden width of {type(model).__name__} "
                f"from config keys {list(_HIDDEN_WIDTH_KEYS)}; pass "
                "decision_hidden_size= explicitly."
            )
        # Pre-wrap probe + reference snapshot: this is what makes the
        # equivalence check sensitive to a wrap that mutated backbone weights.
        probe: Optional[Dict[str, torch.Tensor]] = None
        reference_outputs: Any = None
        probe_error: Optional[str] = None
        if equivalence_check:
            tokenizer = factory_kwargs.get("tokenizer")
            probe = _resolve_probe(model, tokenizer, probe_inputs)
            if probe is not None:
                model.eval()
                try:
                    with torch.no_grad():
                        reference_outputs = model(**probe)
                except Exception as exc:  # noqa: BLE001 - recorded, never raised
                    probe_error = f"{type(exc).__name__}: {exc}"
                    reference_outputs = None
        if user_config is not None:
            config = user_config
            if decision_type is not None:
                config.decision_type = decision_type
        else:
            backbone_config = config_obj.to_diff_dict()
            backbone_config["architectures"] = [type(model).__name__]
            name_or_path = getattr(model, "name_or_path", None) or getattr(
                config_obj, "_name_or_path", None
            )
            config_kwargs: Dict[str, Any] = {
                "base_model_name_or_path": str(name_or_path) if name_or_path else None,
                "base_model_type": getattr(config_obj, "model_type", None),
                "backbone_config": backbone_config,
                "decision_hidden_size": width,
                "question_hidden_size": width,
                "schema_hidden_size": width,
            }
            config_kwargs.update(config_overrides)
            config = JevXConfig(**config_kwargs)
            if decision_type is not None:
                config.decision_type = decision_type
        wrapper = JevXForDecision(config, backbone=model, **factory_kwargs)
        wrapper.eval()
        name_or_path = getattr(config, "base_model_name_or_path", None)
        source_label = type(model).__name__ + (f" ({name_or_path})" if name_or_path else "")
        loader_kwargs = {}
        path_equivalence_kind = "module"
        is_reload = False
    else:
        raise TypeError(
            "convert: expected a model id (str/Path) or an in-memory nn.Module, got "
            f"{type(model_or_id).__name__}."
        )

    # ---------------------------------------------------------------- transplant
    mode = ConversionMode.REPRESENTATION_REUSE
    mapped_params: Dict[str, str] = {}
    unmapped_source = []
    transplant_checks: list = []
    warnings: list = []
    if source_head is not None:
        candidates: list = []
        primary = decision_type if isinstance(decision_type, str) else None
        if primary is None:
            primary = getattr(wrapper.config, "decision_type", None)
        if not isinstance(primary, str):
            primary = "choice"  # documented default decision type
        for name in (primary, *tuple(wrapper.heads.keys())):
            if isinstance(name, str) and name in wrapper.heads and name not in candidates:
                candidates.append(name)
        transplant_result: Optional[ConversionReport] = None
        failures = []
        attempt_inventories: list = []
        for name in candidates:
            result = transplant_classifier_head(source_head, wrapper.heads[name])
            attempt_inventories.append(set(result.unmapped_source))
            if result.mapped_params:
                transplant_result = result
                target_head_name = name
                if failures:
                    warnings.append(
                        f"transplant ladder: no mapping in {failures} before "
                        f"{name!r} mapped — candidate order documented on convert()"
                    )
                break
            failures.append(f"heads[{name}]")
            warnings.extend(
                f"transplant attempt on heads[{name}]: {w}" for w in result.warnings
            )
        if transplant_result is None:
            # Nothing mapped: keep the honest source inventory (union over the
            # attempts — each lists the full unmapped head tensors).
            unmapped_source = sorted(set().union(*attempt_inventories)) if attempt_inventories else []
            warnings.append(
                f"source head {type(source_head).__name__} could not be transplanted "
                f"into any candidate head {candidates} — JevX heads remain freshly "
                "initialized: representation reuse, NOT a transplant"
            )
        else:
            mode = transplant_result.mode
            mapped_params = {
                s_key: f"heads.{target_head_name}.{t_key}"
                for s_key, t_key in transplant_result.mapped_params.items()
            }
            unmapped_source = list(transplant_result.unmapped_source)
            transplant_checks = list(transplant_result.checks)
            warnings.extend(transplant_result.warnings)
            warnings.append(
                f"head graft: {transplant_result.source} -> "
                f"heads[{target_head_name}] ({transplant_result.target}), "
                f"mode={mode.value} — a warm start, not an equivalence claim"
            )

    # ---------------------------------------------------------------- equivalence
    functional_result: Optional[Dict[str, Any]] = None
    if equivalence_check:
        try:
            if path_equivalence_kind == "module":
                if probe is None:
                    functional_result = check_functional_equivalence(
                        wrapper, None, None, tokenizer=factory_kwargs.get("tokenizer")
                    )
                elif reference_outputs is None:
                    functional_result = _skip_check(
                        "functional_equivalence",
                        f"pre-wrap reference snapshot unavailable ({probe_error}) — "
                        "functional equivalence NOT measured",
                    )
                else:
                    functional_result = check_functional_equivalence(
                        wrapper, reference_outputs, probe
                    )
            else:  # id / path
                if probe is None:
                    functional_result = check_functional_equivalence(
                        wrapper,
                        None,
                        None,
                        tokenizer=getattr(wrapper, "tokenizer", None),
                    )
                elif is_reload:
                    functional_result = _skip_check(
                        "functional_equivalence",
                        "checkpoint is already a JevX save (model_type='jevx'): the "
                        "reload restores its own weights, and no independent "
                        "base-model reference exists for comparison — functional "
                        "equivalence NOT measured",
                    )
                else:
                    reference, attempts = _reload_base_reference(
                        model_or_id, loader_kwargs
                    )
                    if reference is None:
                        functional_result = _skip_check(
                            "functional_equivalence",
                            "independent base reload failed on the documented chain "
                            f"(AutoModel -> AutoModelForCausalLM -> AutoModelForSeq2SeqLM): "
                            f"{attempts} — functional equivalence NOT measured",
                        )
                    else:
                        functional_result = check_functional_equivalence(
                            wrapper,
                            reference,
                            probe,
                            tokenizer=getattr(wrapper, "tokenizer", None),
                        )
        except Exception as exc:  # noqa: BLE001 - equivalence never raises
            functional_result = _skip_check(
                "functional_equivalence",
                f"equivalence probe raised {type(exc).__name__}: {exc} — functional "
                "equivalence NOT measured",
            )

    # -------------------------------------------------------------- assemble
    if is_reload:
        new_params: list = []
        warnings.append(
            "reload path: parameter provenance follows the checkpoint (backbone "
            "restored verbatim, heads as saved); new_params not enumerated"
        )
    else:
        jevx_keys = [
            k for k in wrapper.state_dict() if not k.startswith("model.")
        ]
        mapped_targets = set(mapped_params.values())
        new_params = sorted(k for k in jevx_keys if k not in mapped_targets)

    report = ConversionReport(
        mode=mode,
        equivalence=EquivalenceKind.NONE,
        source=source_label,
        target=type(wrapper).__name__,
        mapped_params=mapped_params,
        unmapped_source=sorted(unmapped_source),
        new_params=new_params,
        checks=list(transplant_checks),
        warnings=warnings,
    )
    if functional_result is not None:
        if functional_result.get("passed") is None:
            report.add_check(
                "functional_equivalence",
                None,
                None,
                None,
                reason=functional_result.get("reason", "not measured"),
            )
        else:
            report.add_check(
                "functional_equivalence",
                functional_result["value"],
                functional_result["tolerance"],
                functional_result["passed"],
            )
            if functional_result["passed"]:
                report.claim_equivalence(EquivalenceKind.FUNCTIONAL)
            else:
                report.warnings.append(
                    "base_forward diverged from the independent reference by "
                    f"{functional_result['value']:.3e} > atol "
                    f"{functional_result['tolerance']:.3e} — functional equivalence "
                    "NOT claimed"
                )
    return wrapper, report
