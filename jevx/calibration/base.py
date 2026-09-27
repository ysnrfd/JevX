# Copyright (c) 2026 ysnrfd
"""Calibrator base class, registry and the :class:`CalibrationEngine` facade.

Schema/type invariant (tested): every calibrator maps a tensor ``x`` of
shape ``[N, K]`` to a tensor of the SAME shape, dtype and class count —
calibration changes *values* only, never the decision schema, the type of
the decision, or ``K``. ``input_type`` tells :meth:`BaseCalibrator.calibrate`
which scale the caller is on:

* ``"logits"`` — temperature/vector return transformed logits; isotonic
  returns a logit representative ``log p'`` (softmax of it reproduces the
  calibrated probabilities exactly, up to an irrelevant additive shift).
* ``"probs"``  — every method returns probabilities (temperature/vector go
  through ``softmax(scale * log p + bias)``, which is well defined because
  softmax is invariant to the additive constant in ``log p``).

Fit protocol: ``fit(logits, labels)`` on a CALIBRATION split — never the
test split. Serialization: ``save(path)``/``load(path)`` via ``torch.save``
of tensor state only (plus string metadata).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, ClassVar, Literal, Self

import torch
import torch.nn as nn

from jevx.utils.numerical import (
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    negative_log_likelihood,
    stable_softmax,
)

__all__ = [
    "CALIBRATOR_REGISTRY",
    "BaseCalibrator",
    "IdentityCalibrator",
    "available_calibrators",
    "build_calibrator",
    "get_calibrator",
    "register_calibrator",
]

CALIBRATOR_REGISTRY: dict[str, type["BaseCalibrator"]] = {}

_INPUT_TYPE = Literal["logits", "probs"]
_SAVE_FORMAT = "jevx-calibration-v1"


def register_calibrator(name: str) -> Callable[[type[BaseCalibrator]], type[BaseCalibrator]]:
    """Class decorator registering a calibrator under ``name``."""

    def decorator(cls: type[BaseCalibrator]) -> type[BaseCalibrator]:
        if name in CALIBRATOR_REGISTRY and CALIBRATOR_REGISTRY[name] is not cls:
            raise ValueError(
                f"calibrator {name!r} is already registered to "
                f"{CALIBRATOR_REGISTRY[name].__name__}."
            )
        CALIBRATOR_REGISTRY[name] = cls
        setattr(cls, "calibrator_name", name)
        return cls

    return decorator


def available_calibrators() -> list[str]:
    """Sorted names of registered calibrators."""
    return sorted(CALIBRATOR_REGISTRY)


def get_calibrator(name: str) -> type["BaseCalibrator"]:
    """Look up a calibrator class; unknown names raise with the roster."""
    if not isinstance(name, str):
        raise TypeError(f"calibrator name must be a str, got {type(name).__name__}.")
    if name not in CALIBRATOR_REGISTRY:
        raise ValueError(
            f"unknown calibration method {name!r}; registered methods are "
            f"{available_calibrators()}. Register new ones with "
            f"@register_calibrator({name!r})."
        )
    return CALIBRATOR_REGISTRY[name]


def build_calibrator(name: str, **kwargs: Any) -> "BaseCalibrator":
    """Instantiate the calibrator registered under ``name``."""
    return get_calibrator(name)(**kwargs)


def validate_fit_inputs(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shared fit-time validation: ``logits [N, K]`` float, ``labels [N]`` long.

    Labels must lie in ``[0, K)`` for ``K >= 2`` (multiclass) or in ``{0, 1}``
    for ``K == 1`` (single sigmoid logit). Returns the tensors unchanged.
    """
    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"logits: expected a torch.Tensor, got {type(logits).__name__}.")
    if logits.ndim != 2:
        raise ValueError(f"logits: expected shape [N, K], got {tuple(logits.shape)}.")
    if not logits.is_floating_point():
        raise TypeError(f"logits: expected a floating-point tensor, got {logits.dtype}.")
    if not isinstance(labels, torch.Tensor):
        raise TypeError(f"labels: expected a torch.Tensor, got {type(labels).__name__}.")
    if labels.ndim != 1:
        raise ValueError(f"labels: expected shape [N], got {tuple(labels.shape)}.")
    if logits.shape[0] != labels.shape[0]:
        raise ValueError(
            f"labels: expected one label per sample — logits has {logits.shape[0]} samples "
            f"but labels has {labels.shape[0]} entries."
        )
    k = logits.shape[1]
    if k < 1:
        raise ValueError(f"logits: expected at least 1 class, got K={k}.")
    labels_l = labels.long()
    if labels_l.numel() > 0:
        lo, hi = int(labels_l.min()), int(labels_l.max())
        if k == 1:
            if lo < 0 or hi > 1:
                raise ValueError(
                    f"labels: K=1 logits imply binary labels in {{0, 1}}, "
                    f"but found min={lo}, max={hi}."
                )
        elif lo < 0 or hi >= k:
            raise ValueError(
                f"labels: must lie in [0, {k}) for K={k} logits, "
                f"but found min={lo}, max={hi}."
            )
    return logits, labels_l


class BaseCalibrator(nn.Module):
    """Abstract calibrator: ``P' = Cal(P)`` without touching schema or type.

    Subclasses implement :meth:`fit`, :meth:`calibrate_logits` and (optionally)
    :meth:`calibrate_probs` (the default routes probabilities through
    ``log`` -> :meth:`calibrate_logits` -> ``softmax``).
    """

    calibrator_name: ClassVar[str] = ""

    def __init__(self) -> None:
        super().__init__()
        # Persisted flag: survives torch.save/load of state_dict.
        self.register_buffer("_fitted", torch.tensor(False))

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has been called at least once."""
        return bool(self._fitted)

    # -- fitting -----------------------------------------------------------

    def fit(self, logits: torch.Tensor, labels: torch.Tensor) -> Self:
        """Fit calibrator parameters on a CALIBRATION split (never test).

        Args:
            logits: Raw logits ``[N, K]``.
            labels: Ground-truth class indices ``[N]``.

        Returns:
            self (chainable).
        """
        raise NotImplementedError

    # -- applying ----------------------------------------------------------

    @torch.no_grad()
    def calibrate_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Transform logits ``[N, K]`` -> calibrated logits ``[N, K]``."""
        raise NotImplementedError

    @torch.no_grad()
    def calibrate_probs(self, probs: torch.Tensor) -> torch.Tensor:
        """Transform probabilities ``[N, K]`` -> calibrated probabilities.

        Default: read ``log p`` as a logit representative (softmax is
        invariant to its additive constant), calibrate, softmax back.
        """
        log_p = probs.clamp_min(torch.finfo(probs.dtype).tiny).log()
        return stable_softmax(self.calibrate_logits(log_p), dim=-1)

    def calibrate(self, x: torch.Tensor, input_type: _INPUT_TYPE = "logits") -> torch.Tensor:
        """Dispatch on the input scale; output scale matches ``input_type``.

        Args:
            x: ``[N, K]`` logits (default) or probabilities.
            input_type: ``"logits"`` or ``"probs"``.

        Returns:
            Calibrated tensor on the SAME scale as the input, same shape,
            same dtype, same ``K`` (schema invariant).
        """
        if input_type == "logits":
            return self.calibrate_logits(x)
        if input_type == "probs":
            return self.calibrate_probs(x)
        raise ValueError(
            f"input_type: expected 'logits' or 'probs', got {input_type!r}."
        )

    def calibrate_valid_block(
        self, x: torch.Tensor, input_type: _INPUT_TYPE = "logits"
    ) -> torch.Tensor:
        """Calibrate a block of VALID-only columns ``[n, k]`` (audit Q-012).

        The block never contains padded columns, so every shape-agnostic
        calibrator (`none` / `temperature` / `isotonic`) behaves exactly as
        with the full padded row: isotonic's top-label protocol reads
        `max` over the block, which now only contains real options.
        K-shaped calibrators (per-class parameters, e.g. `vector`) override
        this hook to align their parameters with the leading valid schema
        positions.

        Args:
            x: Valid-column slice(s) ``[n, k]`` (`k >= 1`).
            input_type: ``"logits"`` or ``"probs"`` (as :meth:`calibrate`).

        Returns:
            Calibrated ``[n, k]`` — same shape/dtype/`k` (schema invariant).
        """
        return self.calibrate(x, input_type=input_type)

    # -- evaluation --------------------------------------------------------

    @torch.no_grad()
    def evaluate(
        self, logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 15
    ) -> dict[str, float]:
        """Metrics of the CALIBRATED predictive probabilities.

        Args:
            logits: Raw logits ``[N, K]`` (the calibrator is applied first).
            labels: Labels ``[N]``.
            n_bins: Bins for ECE/MCE (default 15).

        Returns:
            ``{"nll", "ece", "mce", "brier"}`` — Python floats (nats / [0,1] /
            nats / Brier units).
        """
        logits, labels = validate_fit_inputs(logits, labels)
        if logits.shape[1] == 1:
            probs = torch.sigmoid(self.calibrate_logits(logits))
        else:
            probs = stable_softmax(self.calibrate_logits(logits), dim=-1)
        return {
            "nll": negative_log_likelihood(probs, labels, eps=1e-12),
            "ece": expected_calibration_error(probs, labels, n_bins=n_bins),
            "mce": maximum_calibration_error(probs, labels, n_bins=n_bins),
            "brier": brier_score(probs, labels),
        }

    # -- serialization -----------------------------------------------------

    def load_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        strict: bool = True,
        assign: bool = False,
    ) -> Any:
        """Restore state, tolerating fit-data-shaped parameters/buffers.

        Some tensors depend on the calibration data (isotonic breakpoints:
        one pair per distinct confidence; vector scale/bias: one entry per
        class), so their saved shape can differ from the freshly constructed
        default. Such entries are reallocated to the saved shape before the
        standard strict load copies the saved values in.
        """
        state_dict = dict(state_dict)
        for name, value in state_dict.items():
            if not isinstance(value, torch.Tensor):
                continue
            buffer = self._buffers.get(name)
            if buffer is not None and buffer.shape != value.shape:
                self._buffers[name] = torch.empty_like(value)
                continue
            param = self._parameters.get(name) if hasattr(self, "_parameters") else None
            if param is not None and param.shape != value.shape:
                self._parameters[name] = nn.Parameter(
                    torch.empty_like(value), requires_grad=param.requires_grad
                )
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def save(self, path: str | Path) -> None:
        """Persist tensor state via ``torch.save`` (tensors + string metadata)."""
        payload = {
            "format": _SAVE_FORMAT,
            "calibrator": self.calibrator_name,
            "state_dict": self.state_dict(),
        }
        torch.save(payload, Path(path))

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> "BaseCalibrator":
        """Rebuild a calibrator written by :meth:`save` and restore its state."""
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("format") != _SAVE_FORMAT:
            raise ValueError(
                f"{path}: not a JevX calibration file (missing format marker "
                f"{_SAVE_FORMAT!r})."
            )
        calibrator = build_calibrator(payload["calibrator"], **kwargs)
        calibrator.load_state_dict(payload["state_dict"])
        return calibrator


@register_calibrator("none")
class IdentityCalibrator(BaseCalibrator):
    """No-op calibration: ``Cal(x) = x`` (the default; metrics still work)."""

    def fit(self, logits: torch.Tensor, labels: torch.Tensor) -> Self:
        validate_fit_inputs(logits, labels)
        self._fitted.fill_(True)
        return self

    @torch.no_grad()
    def calibrate_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return logits

    @torch.no_grad()
    def calibrate_probs(self, probs: torch.Tensor) -> torch.Tensor:
        return probs


class CalibrationEngine:
    """Facade: method name -> calibrator instance; fit / apply / evaluate.

    Example:
        >>> engine = CalibrationEngine("temperature")
        >>> engine.fit(cal_logits, cal_labels)          # CALIBRATION split
        >>> calibrated = engine.apply(test_logits)      # same [N, K], same type
        >>> metrics = engine.evaluate(test_logits, test_labels)

    The engine never changes the schema: ``apply`` preserves shape, dtype
    and the number of classes for every registered method.
    """

    def __init__(self, method: str = "none", **method_kwargs: Any) -> None:
        self.method = method
        self.calibrator: BaseCalibrator = build_calibrator(method, **method_kwargs)

    @classmethod
    def from_config(cls, config: Any) -> "CalibrationEngine":
        """Build from an object exposing ``calibration_method`` (JevXConfig)."""
        return cls(getattr(config, "calibration_method", "none"))

    def fit(self, logits: torch.Tensor, labels: torch.Tensor) -> "CalibrationEngine":
        """Fit the underlying calibrator on a CALIBRATION split (never test).

        Returns:
            self (chainable).
        """
        self.calibrator.fit(logits, labels)
        return self

    def calibrate(self, x: torch.Tensor, input_type: _INPUT_TYPE = "logits") -> torch.Tensor:
        """Apply calibration; see :meth:`BaseCalibrator.calibrate`."""
        return self.calibrator.calibrate(x, input_type=input_type)

    def apply(self, x: torch.Tensor, input_type: _INPUT_TYPE = "logits") -> torch.Tensor:
        """Alias of :meth:`calibrate` (``P' = Cal(P)``)."""
        return self.calibrate(x, input_type=input_type)

    @torch.no_grad()
    def apply_valid_columns(
        self,
        x: torch.Tensor,
        option_mask: torch.Tensor | None,
        input_type: _INPUT_TYPE = "logits",
    ) -> torch.Tensor:
        """Calibrate per-row over VALID columns only — variable-K safe (audit Q-012).

        Design choice (D8.I, "calibrate the question's own distribution"):
        each row is sliced to its valid columns, the equal-width slices are
        stacked into one ``[n_k, k]`` block per distinct valid width `k`,
        the block is calibrated, and the results are scattered back. Padded
        columns keep their ORIGINAL raw values — they are re-masked by
        ``head.build_output(..., option_mask)`` before anything reads them,
        so nothing downstream can observe them.

        Why slicing (vs. mask-filling): isotonic's top-label protocol takes
        ``max` over the LAST axis, so an arbitrary padded logit (0 for
        dot/bilinear fusion, MLP bias for `mlp`, ±1e6 under adversarial
        inputs) could win the "top" and silently no-op or misplace the
        correction for that row (audit Q-012). Slicing makes padded columns
        unrepresentable in the calibration computation.

        Args:
            x: Padded logits/probs ``[N, K_max]``.
            option_mask: Bool ``[N, K_max]`` (`True` = valid option) with
                at least one valid entry per row, or `None` (= no padding;
                falls back to :meth:`apply`).
            input_type: ``"logits"`` or ``"probs"``.

        Returns:
            ``[N, K_max]`` on the same scale as `x` — valid columns
            calibrated, padded columns untouched (same shape/dtype/K).

        Raises:
            TypeError: `x`/`option_mask` are not tensors, or the mask is
                not bool.
            ValueError: Wrong rank, shape mismatch, or a row with no valid
                column (messages name the tensor and row).
        """
        if option_mask is None:
            return self.apply(x, input_type=input_type)
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x: expected a torch.Tensor, got {type(x).__name__}.")
        if x.ndim != 2:
            raise ValueError(f"x: expected shape [N, K], got {tuple(x.shape)}.")
        if not isinstance(option_mask, torch.Tensor):
            raise TypeError(
                f"option_mask: expected a torch.Tensor, got {type(option_mask).__name__}."
            )
        if option_mask.dtype != torch.bool:
            raise TypeError(
                f"option_mask: expected a torch.bool tensor, got {option_mask.dtype}; "
                "convert it with `.bool()`."
            )
        if tuple(option_mask.shape) != tuple(x.shape):
            raise ValueError(
                f"option_mask: expected the same shape as x {tuple(x.shape)}, got "
                f"{tuple(option_mask.shape)}."
            )
        counts = option_mask.sum(dim=-1)
        if not bool((counts >= 1).all()):
            bad = int((counts == 0).nonzero(as_tuple=False)[0])
            raise ValueError(
                f"option_mask: row {bad} has no valid column — calibration needs at "
                "least one valid option per question."
            )
        out = x.clone()
        for k in torch.unique(counts):
            k_int = int(k)
            rows = (counts == k).nonzero(as_tuple=False).squeeze(-1)
            # nonzero scans row-major over the sub-mask -> [n_k * k, 2]
            # (local_row, col) pairs; take the column axis to get the [n_k, k]
            # valid column indices in schema order.
            idx = (
                option_mask[rows]
                .nonzero(as_tuple=False)
                .view(rows.numel(), k_int, 2)[..., 1]
            )
            block = torch.gather(x[rows], 1, idx)
            calibrated = self.calibrator.calibrate_valid_block(
                block, input_type=input_type
            )
            out[rows] = out[rows].scatter(1, idx, calibrated)
        return out

    def evaluate(
        self, logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 15
    ) -> dict[str, float]:
        """Calibrated metrics ``{"nll", "ece", "mce", "brier"}``."""
        return self.calibrator.evaluate(logits, labels, n_bins=n_bins)

    @property
    def is_fitted(self) -> bool:
        """Whether the underlying calibrator has been fitted."""
        return self.calibrator.is_fitted

    def state_dict(self) -> dict[str, Any]:
        """Tensor state of the underlying calibrator."""
        return self.calibrator.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> "CalibrationEngine":
        """Restore state produced by :meth:`state_dict` (returns self)."""
        self.calibrator.load_state_dict(state_dict)
        return self

    def save(self, path: str | Path) -> None:
        """Persist via the calibrator's ``torch.save`` format."""
        self.calibrator.save(path)

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> "CalibrationEngine":
        """Load an engine saved by :meth:`save`."""
        engine = cls.__new__(cls)
        engine.calibrator = BaseCalibrator.load(path, **kwargs)
        engine.method = engine.calibrator.calibrator_name
        return engine

    def __repr__(self) -> str:
        return (
            f"CalibrationEngine(method={self.method!r}, "
            f"fitted={self.is_fitted})"
        )
