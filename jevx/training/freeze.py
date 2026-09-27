# Copyright (c) 2026 ysnrfd
"""Backbone freeze/unfreeze control for training Mode A / Mode B (§29).

Mode A (default): backbone ``θ`` frozen — ``requires_grad=False`` on every
backbone parameter; only adapters/encoders/fusion/heads ``φ`` train.
Mode B (opt-in): backbone unfrozen (fully, or layer-by-layer via
:func:`unfreeze_last_n_layers`).

Everything here is **duck-typed**: this module never imports
``jevx.modeling_jevx`` (or any model class) at module import time, so it can
be used with the JevX wrapper, a bare HF model, or a plain ``nn.Module``
without import cycles. Freezing only flips ``requires_grad`` — switching a
module to ``eval()`` mode (BN/dropout statistics) is the caller's/model's
job, not this module's.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch.nn as nn

__all__ = [
    "count_parameters",
    "freeze_backbone",
    "freeze_module",
    "unfreeze_backbone",
    "unfreeze_last_n_layers",
    "unfreeze_module",
]

#: Conventional per-layer container attribute paths, checked in this order:
#: GPT-2 style ``model.h``, Llama/GPT-NeoX style ``model.layers``,
#: BERT/encoder style ``model.encoder.layer``, decoder style
#: ``model.decoder.layers``.
_LAYER_PATHS: tuple[tuple[str, ...], ...] = (
    ("layers",),
    ("h",),
    ("encoder", "layer"),
    ("decoder", "layers"),
)


def _iter_params(module: nn.Module) -> Iterable[nn.Parameter]:
    if not isinstance(module, nn.Module):
        raise TypeError(
            f"expected an nn.Module, got {type(module).__name__}."
        )
    return module.parameters()


def _set_requires_grad(module: nn.Module, flag: bool) -> nn.Module:
    """Set ``requires_grad=flag`` on every parameter of ``module``."""
    for param in _iter_params(module):
        param.requires_grad_(flag)
    return module


def freeze_module(module: nn.Module) -> nn.Module:
    """Freeze every parameter of ``module`` (``requires_grad=False``).

    Args:
        module: Any ``nn.Module`` (applies recursively to children).

    Returns:
        ``module`` itself, for chaining (e.g. ``freeze_module(enc).eval()``).
    """
    return _set_requires_grad(module, False)


def unfreeze_module(module: nn.Module) -> nn.Module:
    """Unfreeze every parameter of ``module`` (``requires_grad=True``).

    Args:
        module: Any ``nn.Module``.

    Returns:
        ``module`` itself, for chaining.
    """
    return _set_requires_grad(module, True)


def _resolve_backbone_child(model: nn.Module) -> nn.Module:
    """Find the backbone child by duck-typing, without touching model classes.

    Preference order (the ``trainable_parameters()``-complement path is
    handled by the callers before this helper runs):

    1. A direct child named ``"model"`` or ``"backbone"`` (the HF wrapper
       convention: ``model.model`` on ``ForXxx`` classes, or an explicit
       ``.backbone`` attribute).
    2. Otherwise: an informative error naming both mechanisms — never a
       blanket ``model.parameters()`` freeze, which would wrongly freeze the
       heads too.

    Returns:
        The backbone ``nn.Module``.
    """
    if not isinstance(model, nn.Module):
        raise TypeError(f"expected an nn.Module, got {type(model).__name__}.")

    for name in ("model", "backbone"):
        child = getattr(model, name, None)
        if isinstance(child, nn.Module):
            return child

    raise ValueError(
        f"cannot locate the backbone of {type(model).__name__}: it defines neither "
        "`trainable_parameters()` nor a child module named 'model' or 'backbone'. "
        "Freeze the intended submodule explicitly with freeze_module(...) instead."
    )


def _has_trainable_parameters_api(model: nn.Module) -> bool:
    return isinstance(model, nn.Module) and callable(
        getattr(model, "trainable_parameters", None)
    )


def _trainable_ids(model: nn.Module) -> set[int]:
    """Identity set of parameters yielded by ``model.trainable_parameters()``."""
    ids: set[int] = set()
    for param in model.trainable_parameters():  # type: ignore[attr-defined]
        ids.add(id(param))
    return ids


def freeze_backbone(model: nn.Module) -> nn.Module:
    """Freeze the backbone ``θ`` (Mode A), leaving heads/adapters trainable.

    Resolution (duck-typed, no imports of model classes):

    * If ``model.trainable_parameters()`` exists, freeze every parameter it
      does NOT yield (the complement = backbone). Parameters it yields keep
      their current ``requires_grad`` state untouched.
    * Else freeze the direct child named ``"model"`` or ``"backbone"``.
    * Else raise ``ValueError`` explaining both mechanisms.

    Args:
        model: Wrapper model (or a module whose child *is* the backbone).

    Returns:
        ``model`` itself.
    """
    if not isinstance(model, nn.Module):
        raise TypeError(f"expected an nn.Module, got {type(model).__name__}.")
    if _has_trainable_parameters_api(model):
        keep = _trainable_ids(model)
        for param in model.parameters():
            if id(param) not in keep:
                param.requires_grad_(False)
        return model
    return freeze_module(_resolve_backbone_child(model))


def unfreeze_backbone(model: nn.Module) -> nn.Module:
    """Unfreeze the backbone ``θ`` (Mode B), leaving the resolution split intact.

    Mirror of :func:`freeze_backbone`: the complement of
    ``trainable_parameters()`` if that method exists, else the child named
    ``"model"``/``"backbone"``, else an informative error. Heads keep their
    current flags (they are expected to already be trainable).

    Args:
        model: Wrapper model (or a module whose child *is* the backbone).

    Returns:
        ``model`` itself.
    """
    if not isinstance(model, nn.Module):
        raise TypeError(f"expected an nn.Module, got {type(model).__name__}.")
    if _has_trainable_parameters_api(model):
        keep = _trainable_ids(model)
        for param in model.parameters():
            if id(param) not in keep:
                param.requires_grad_(True)
        return model
    return unfreeze_module(_resolve_backbone_child(model))


def _find_layer_list(model: nn.Module) -> tuple[str, list[nn.Module]]:
    """Locate the conventional per-layer ``ModuleList``/list of modules.

    Checks ``model.layers``, ``model.h``, ``model.encoder.layer`` and
    ``model.decoder.layers`` (and the same four one level down under
    ``model.model.*``, covering ``ForCausalLM`` wrappers that nest the base
    model under ``.model``).

    Returns:
        ``(dotted_path, layers)`` where ``layers`` is a non-empty sequence of
        ``nn.Module`` objects.

    Raises:
        ValueError: No known layer-list attribute exists — the message lists
            every path that was tried.
    """
    tried: list[str] = []
    roots: list[tuple[str, nn.Module]] = [("", model)]
    inner = getattr(model, "model", None)
    if isinstance(inner, nn.Module):
        roots.append(("model.", inner))
    for prefix, root in roots:
        for path in _LAYER_PATHS:
            obj = root
            ok = True
            for attr in path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    ok = False
                    break
            dotted = prefix + ".".join(path)
            if not ok:
                tried.append(dotted)
                continue
            if isinstance(obj, nn.ModuleList):
                layers = list(obj)
            elif isinstance(obj, (list, tuple)) and all(
                isinstance(layer, nn.Module) for layer in obj
            ):
                layers = list(obj)
            else:
                tried.append(f"{dotted} (found, but not a layer list)")
                continue
            if not layers:
                tried.append(f"{dotted} (empty)")
                continue
            return dotted, layers
    raise ValueError(
        f"cannot find a conventional layer list on {type(model).__name__}: tried "
        f"{tried}. Expected one of `layers`, `h`, `encoder.layer`, `decoder.layers` "
        "(directly or under `.model`). Unfreeze layers explicitly with "
        "unfreeze_module(...) instead."
    )


def unfreeze_last_n_layers(model: nn.Module, n: int) -> nn.Module:
    """Mode B helper: unfreeze the last ``n`` layers of the backbone stack.

    Layer lists are found by convention (``layers``/``h``/``encoder.layer``/
    ``decoder.layers``, also under ``.model``). The final ``n`` layers — and
    only those — are switched to ``requires_grad=True``; everything else keeps
    its current flags.

    Args:
        model: Model (or backbone) holding the layer list.
        n: Number of trailing layers to unfreeze; ``n == 0`` is a no-op.

    Returns:
        ``model`` itself.

    Raises:
        TypeError: ``n`` is not an ``int`` (``bool`` excluded).
        ValueError: ``n < 0``, ``n > len(layers)``, or no layer list exists.
    """
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError(f"n: expected an int, got {type(n).__name__}.")
    if n < 0:
        raise ValueError(f"n: must be >= 0, got {n}.")
    if not isinstance(model, nn.Module):
        raise TypeError(f"expected an nn.Module, got {type(model).__name__}.")
    _, layers = _find_layer_list(model)
    if n > len(layers):
        raise ValueError(
            f"n: requested the last {n} layers but {type(model).__name__} only has "
            f"{len(layers)}."
        )
    for layer in layers[len(layers) - n:]:
        unfreeze_module(layer)
    return model


def count_parameters(module: nn.Module) -> dict[str, int]:
    """Count parameters by ``requires_grad`` state.

    Args:
        module: Any ``nn.Module`` (recursed over children).

    Returns:
        ``{"trainable": ..., "frozen": ..., "total": ...}`` where each value
        is the number of scalar elements (``param.numel()``, not the number
        of parameter tensors) — this matches how optimizer memory and
        FLOP-relevant weights are usually reported. ``trainable + frozen ==
        total`` by construction.

    Raises:
        TypeError: ``module`` is not an ``nn.Module``.
    """
    trainable = frozen = total = 0
    for param in _iter_params(module):
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
        else:
            frozen += count
    return {"trainable": trainable, "frozen": frozen, "total": total}
