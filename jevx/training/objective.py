# Copyright (c) 2026 ysnrfd
"""Composite training objective (§85).

The default objective is the plain decision loss: every auxiliary term is
OFF (``lambda = 0``). Multi-term objectives (calibration-aware losses,
representation regularizers, weight decay terms) are *experimental knobs* —
they are off by default because combining them without experimental
justification makes attribution impossible. Enable them per experiment and
report the lambdas used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

__all__ = ["DecisionObjective", "l2_regularization"]

_COMPONENT_NAMES: tuple[str, ...] = (
    "decision",
    "calibration",
    "representation",
    "regularization",
)


def _validate_lambda(name: str, value: Any) -> float:
    """Return ``value`` as a float, requiring a finite number ``>= 0``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name}: expected a float >= 0, got {type(value).__name__}.")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name}: expected a finite value >= 0, got {value!r}.")
    return value


@dataclass
class DecisionObjective:
    """Weighted sum of the decision loss and optional auxiliary losses (§85)::

        total = λ_dec · decision
              + λ_cal · calibration        # default 0 (OFF)
              + λ_rep · representation     # default 0 (OFF)
              + λ_reg · regularization     # default 0 (OFF)

    Defaults: only the decision term is active — the objective equals the
    decision loss. Auxiliary losses passed as ``None`` are treated as 0 (they
    are *absent*, not "free": a zero-weight term contributes nothing even if a
    value is supplied).

    Args:
        lambda_decision: Weight on the primary decision loss (>= 0).
        lambda_calibration: Weight on the calibration term (>= 0, default 0).
        lambda_representation: Weight on the representation term (>= 0,
            default 0).
        lambda_regularization: Weight on the regularization term (>= 0,
            default 0).

    Raises:
        TypeError / ValueError: Any lambda is not a finite number >= 0.
    """

    lambda_decision: float = 1.0
    lambda_calibration: float = 0.0
    lambda_representation: float = 0.0
    lambda_regularization: float = 0.0

    def __post_init__(self) -> None:
        self.lambda_decision = _validate_lambda("lambda_decision", self.lambda_decision)
        self.lambda_calibration = _validate_lambda(
            "lambda_calibration", self.lambda_calibration
        )
        self.lambda_representation = _validate_lambda(
            "lambda_representation", self.lambda_representation
        )
        self.lambda_regularization = _validate_lambda(
            "lambda_regularization", self.lambda_regularization
        )

    @property
    def lambdas(self) -> dict[str, float]:
        """Mapping ``component name -> lambda`` in ``_COMPONENT_NAMES`` order."""
        return {
            "decision": self.lambda_decision,
            "calibration": self.lambda_calibration,
            "representation": self.lambda_representation,
            "regularization": self.lambda_regularization,
        }

    def compute(
        self,
        decision_loss: torch.Tensor | float,
        *,
        calibration_loss: torch.Tensor | float | None = None,
        representation_loss: torch.Tensor | float | None = None,
        regularization_loss: torch.Tensor | float | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Combine the terms into ``(total, components)``.

        Args:
            decision_loss: Primary loss (tensor or python number); required.
            calibration_loss: Optional calibration term; ``None`` = 0.
            representation_loss: Optional representation term; ``None`` = 0.
            regularization_loss: Optional regularization term; ``None`` = 0.

        Returns:
            ``(total, components)`` where ``total`` is the weighted sum (a
            tensor if any *active* term is a tensor, else a float) and
            ``components`` maps each of ``decision``/``calibration``/
            ``representation``/``regularization`` to its **weighted**
            contribution (tensor or float), plus the key ``"total"`` mirroring
            ``total``. Inactive (zero-lambda or ``None``) components are the
            plain number ``0``.

        Raises:
            TypeError / ValueError: ``decision_loss`` is ``None`` or not a
                number/tensor, or a lambda became invalid after construction
                (the dataclass is mutable).
        """
        lambdas = self.lambdas  # re-validates via _validate_lambda? no — see below
        # Lambdas are re-checked here because the dataclass is mutable after
        # __post_init__ (users may assign fields directly).
        for name, value in lambdas.items():
            lambdas[name] = _validate_lambda(f"lambda_{name}", value)

        if decision_loss is None:
            raise TypeError(
                "compute: decision_loss is required (got None); the composite "
                "objective has no meaning without its primary term."
            )
        values: dict[str, Any] = {
            "decision": decision_loss,
            "calibration": calibration_loss,
            "representation": representation_loss,
            "regularization": regularization_loss,
        }
        for name in _COMPONENT_NAMES[1:]:
            if values[name] is not None and not isinstance(
                values[name], (torch.Tensor, int, float)
            ):
                raise TypeError(
                    f"compute: {name}_loss must be a tensor, a number, or None; "
                    f"got {type(values[name]).__name__}."
                )
        if isinstance(decision_loss, bool) or not isinstance(
            decision_loss, (torch.Tensor, int, float)
        ):
            raise TypeError(
                "compute: decision_loss must be a tensor or number, got "
                f"{type(decision_loss).__name__}."
            )

        components: dict[str, Any] = {}
        total: Any = 0.0
        for name in _COMPONENT_NAMES:
            lam = lambdas[name]
            value = values[name]
            weighted: Any = 0.0
            if lam != 0.0 and value is not None:
                weighted = lam * value
            components[name] = weighted
            if isinstance(weighted, torch.Tensor) or isinstance(total, torch.Tensor):
                total = total + weighted
            else:
                total = total + float(weighted)
        components["total"] = total
        return total, components


def l2_regularization(module: torch.nn.Module, weight: float) -> torch.Tensor:
    """L2 penalty ``weight · Σ_p ‖p‖²`` over all parameters of ``module``.

    Args:
        module: Whose parameters are penalized (all of them, regardless of
            ``requires_grad`` — freezing does not exempt weights from an
            explicit penalty; skip frozen groups by passing a submodule).
        weight: Non-negative coefficient.

    Returns:
        Scalar tensor (on the device/dtype of the first parameter; a
        float32 CPU scalar if the module has no parameters — the value is 0).

    Raises:
        TypeError: ``module`` is not an ``nn.Module`` or ``weight`` is not a
            number.
        ValueError: ``weight`` is negative or non-finite.
    """
    if not isinstance(module, torch.nn.Module):
        raise TypeError(
            f"l2_regularization: expected an nn.Module, got {type(module).__name__}."
        )
    weight = _validate_lambda("weight", weight)
    total: torch.Tensor | None = None
    for param in module.parameters():
        sq = param.pow(2).sum()
        total = sq if total is None else total + sq
    if total is None:
        return torch.zeros((), dtype=torch.float32)
    return weight * total
