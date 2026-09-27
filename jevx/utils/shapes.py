# Copyright (c) 2026 ysnrfd
"""Explicit tensor validators whose errors tell the caller exactly what is wrong.

Every helper raises immediately (fail-fast, before any matmul) and carries the
logical *name* of the offending tensor (e.g. ``"fused"``, ``"option_mask"``)
so shape bugs surface at the boundary instead of deep inside a kernel.
"""

from __future__ import annotations

from typing import Any

import torch

__all__ = [
    "assert_shape",
    "assert_batch_alignment",
    "assert_same_device_dtype",
    "check_finite",
    "has_inf",
    "has_nan",
]


def assert_shape(
    tensor: Any,
    expected_dims: tuple[int | None, ...] | None,
    name: str,
) -> torch.Tensor:
    """Assert ``tensor`` is a :class:`torch.Tensor` with the expected shape.

    Args:
        tensor: Object that must be a tensor.
        expected_dims: Sequence of expected dimensions where ``None`` is a
            wildcard for "any size", e.g. ``(None, 3, None)``. ``None`` (the
            scalar value) skips the shape check entirely and only verifies
            tensor-ness.
        name: Human-readable tensor name used in the error message.

    Returns:
        The input tensor, so the call can be chained.

    Raises:
        TypeError: ``tensor`` is not a :class:`torch.Tensor`.
        ValueError: rank or a concrete dimension does not match.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name}: expected a torch.Tensor, got {type(tensor).__name__}.")
    if expected_dims is None:
        return tensor

    actual = tuple(tensor.shape)
    if len(actual) != len(expected_dims):
        raise ValueError(
            f"{name}: expected shape {expected_dims} (rank {len(expected_dims)}) "
            f"but got shape {actual} (rank {len(actual)})."
        )
    for axis, (got, want) in enumerate(zip(actual, expected_dims)):
        if want is not None and got != want:
            raise ValueError(
                f"{name}: expected shape {expected_dims} but got {actual} "
                f"(dimension {axis} expected {want}, got {got})."
            )
    return tensor


def assert_batch_alignment(**named_tensors: torch.Tensor | None) -> int:
    """Assert every provided tensor shares the same batch size (dimension 0).

    Keyword arguments map a logical name to a tensor; ``None`` values are
    skipped so optional inputs can be forwarded directly.

    Args:
        **named_tensors: Mapping ``name -> tensor`` (or ``None``).

    Returns:
        The shared batch size ``B``.

    Raises:
        ValueError: No tensors were provided, a tensor has rank 0, or batch
            sizes disagree (the message names both sides of the mismatch).
    """
    present = {name: tensor for name, tensor in named_tensors.items() if tensor is not None}
    if not present:
        raise ValueError("assert_batch_alignment: no tensors provided to compare.")

    for name, tensor in present.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name}: expected a torch.Tensor, got {type(tensor).__name__}.")
        if tensor.dim() < 1:
            raise ValueError(
                f"{name}: expected at least 1 dimension (a batch dim), got a 0-dim scalar."
            )

    first_name, first = next(iter(present.items()))
    batch = first.shape[0]
    for name, tensor in present.items():
        if tensor.shape[0] != batch:
            raise ValueError(
                f"batch size mismatch: {first_name} has batch size {batch} but "
                f"{name} has batch size {tensor.shape[0]}; "
                f"all of {sorted(present)} must share dimension 0."
            )
    return batch


def has_nan(tensor: torch.Tensor) -> bool:
    """Return ``True`` iff ``tensor`` contains at least one NaN (float tensors)."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"has_nan: expected a torch.Tensor, got {type(tensor).__name__}.")
    if not tensor.is_floating_point() and not tensor.is_complex():
        return False
    return bool(torch.isnan(tensor).any())


def has_inf(tensor: torch.Tensor) -> bool:
    """Return ``True`` iff ``tensor`` contains +Inf or -Inf.

    Float AND complex tensors are inspected (``torch.isinf`` is defined for
    both, and a complex entry is infinite when EITHER component is); integer
    and bool tensors cannot hold an infinity and return ``False``, mirroring
    :func:`has_nan`.

    Raises:
        TypeError: ``tensor`` is not a :class:`torch.Tensor`.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"has_inf: expected a torch.Tensor, got {type(tensor).__name__}.")
    if not tensor.is_floating_point() and not tensor.is_complex():
        return False
    return bool(torch.isinf(tensor).any())


def check_finite(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """Assert every entry of ``tensor`` is finite (no NaN, +Inf or -Inf).

    Args:
        tensor: Floating-point tensor to inspect.
        name: Logical name reported in the error.

    Returns:
        The input tensor for chaining.

    Raises:
        TypeError: ``tensor`` is not a floating-point tensor.
        ValueError: The tensor holds non-finite values; the message reports
            separate NaN / +Inf / -Inf counts and the first offending index.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name}: expected a torch.Tensor, got {type(tensor).__name__}.")
    if not tensor.is_floating_point():
        raise TypeError(
            f"{name}: expected a floating-point tensor to check finiteness, got {tensor.dtype}."
        )
    nan_mask = torch.isnan(tensor)
    pos_mask = torch.isposinf(tensor)
    neg_mask = torch.isneginf(tensor)
    bad = nan_mask | pos_mask | neg_mask
    if bool(bad.any()):
        first = tuple(int(i) for i in torch.nonzero(bad, as_tuple=False)[0].tolist())
        raise ValueError(
            f"{name}: tensor contains non-finite values: "
            f"{int(nan_mask.sum())} NaN, {int(pos_mask.sum())} +Inf, "
            f"{int(neg_mask.sum())} -Inf; first offending entry at index {first} "
            f"of shape {tuple(tensor.shape)}."
        )
    return tensor


def assert_same_device_dtype(*tensors: torch.Tensor, names: list[str] | None = None) -> None:
    """Assert all tensors live on the same device with the same dtype.

    Args:
        tensors: Two or more tensors to compare (``None`` is not accepted).
        names: Optional parallel list of logical names for error messages;
            defaults to ``tensor0``, ``tensor1``, ...

    Raises:
        TypeError: Fewer than two arguments, or an argument is not a tensor.
        ValueError: dtypes or devices disagree (each mismatch type has its own
            message containing ``dtype`` or ``device`` respectively).
    """
    if len(tensors) < 2:
        raise TypeError(
            f"assert_same_device_dtype: expected at least two tensors, got {len(tensors)}."
        )
    for i, tensor in enumerate(tensors):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"argument {i}: expected a torch.Tensor, got {type(tensor).__name__}."
            )
    if names is None:
        names = [f"tensor{i}" for i in range(len(tensors))]
    if len(names) != len(tensors):
        raise ValueError(
            f"assert_same_device_dtype: got {len(names)} names for {len(tensors)} tensors."
        )

    ref_name, ref = names[0], tensors[0]
    for name, tensor in zip(names[1:], tensors[1:]):
        if tensor.dtype != ref.dtype:
            raise ValueError(
                f"dtype mismatch: {ref_name} has dtype {ref.dtype} but "
                f"{name} has dtype {tensor.dtype}; all tensors must share one dtype."
            )
        if tensor.device != ref.device:
            raise ValueError(
                f"device mismatch: {ref_name} is on device {ref.device} but "
                f"{name} is on device {tensor.device}; all tensors must share one device."
            )
