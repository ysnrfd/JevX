# Copyright (c) 2026 ysnrfd
"""Temperature scaling: one positive scalar ``T`` shared by all classes."""

from __future__ import annotations

import math
from typing import Self

import torch

from jevx.utils.numerical import stable_softmax

from .base import BaseCalibrator, register_calibrator, validate_fit_inputs

__all__ = ["TemperatureCalibrator"]

# Log-spaced search range and refinement schedule (deterministic, no RNG).
_T_MIN = 1e-2
_T_MAX = 1e2
_GRID = 400
_REFINE_ROUNDS = 6


def _nll_at_temperature(
    logits: torch.Tensor, labels: torch.Tensor, temperature: float
) -> float:
    """NLL of ``softmax(logits / T)`` (sigmoid for K == 1), in float64."""
    scaled = logits.double() / temperature
    if logits.shape[1] == 1:
        p = torch.sigmoid(scaled[:, 0]).clamp(1e-12, 1.0 - 1e-12)
        y = labels.double()
        return float((-(y * p.log() + (1.0 - y) * (1.0 - p).log())).mean())
    picked = torch.softmax(scaled, dim=-1).gather(
        1, labels.unsqueeze(1)
    ).squeeze(1)
    return float(-picked.clamp_min(1e-12).log().mean())


@register_calibrator("temperature")
class TemperatureCalibrator(BaseCalibrator):
    """Guo et al. temperature scaling: ``Cal(l) = l / T`` with scalar ``T > 0``.

    Mathematical contract:

    * ``softmax`` of ``l / T`` is monotonically increasing in each ``l_k`` /
      ``T`` for ``T > 0``, so **temperature preserves argmax** exactly.
    * ``T`` is parameterized implicitly positive (fit optimizes ``log T``).

    Fit: deterministic grid search over ``T in [1e-2, 1e2]`` (log-spaced,
    ``_GRID`` points) minimizing NLL, followed by ``_REFINE_ROUNDS`` halving
    refinements around the best point — no RNG, no optimizer state, so the
    result is bit-reproducible on a given platform. (A convex-in-log-T NLL
    makes the grid optimum essentially exact for well-posed data.)
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("log_temperature", torch.tensor(0.0, dtype=torch.float64))

    @property
    def temperature(self) -> torch.Tensor:
        """The fitted scalar ``T > 0`` (float64 buffer, 1.0 before fit)."""
        return self.log_temperature.exp()

    def fit(self, logits: torch.Tensor, labels: torch.Tensor) -> Self:
        """Fit ``T`` by minimizing NLL on ``logits``/``labels``.

        Args:
            logits: Calibration-split logits ``[N, K]``.
            labels: Calibration-split labels ``[N]``.

        Returns:
            self.
        """
        logits, labels = validate_fit_inputs(logits, labels)
        data = logits.detach().to(torch.float64)

        # Phase 1: dense log-space grid over [1e-2, 1e2].
        grid = torch.logspace(
            math.log10(_T_MIN), math.log10(_T_MAX), _GRID, dtype=torch.float64
        )
        nlls = torch.tensor(
            [_nll_at_temperature(data, labels, float(t)) for t in grid]
        )
        best = int(nlls.argmin())
        t_best = float(grid[best])

        # Phase 2: shrinking-window refinement around the grid optimum.
        half_width = float(grid[1] / grid[0]) ** (_GRID / (2 * _REFINE_ROUNDS))
        for _ in range(_REFINE_ROUNDS):
            lo, hi = t_best / half_width, t_best * half_width
            local = torch.logspace(
                math.log10(lo), math.log10(hi), 41, dtype=torch.float64
            )
            local_nlls = torch.tensor(
                [_nll_at_temperature(data, labels, float(t)) for t in local]
            )
            t_best = float(local[int(local_nlls.argmin())])
            half_width = half_width**0.5

        self.log_temperature.fill_(math.log(t_best))
        self._fitted.fill_(True)
        return self

    @torch.no_grad()
    def calibrate_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """``l / T`` — same shape and dtype as ``logits``; argmax preserved."""
        t = self.temperature.to(device=logits.device, dtype=logits.dtype)
        return logits / t

    @torch.no_grad()
    def calibrate_probs(self, probs: torch.Tensor) -> torch.Tensor:
        """``softmax(log p / T)`` — valid because softmax ignores constants."""
        log_p = probs.clamp_min(torch.finfo(probs.dtype).tiny).log()
        t = self.temperature.to(device=probs.device, dtype=log_p.dtype)
        return stable_softmax(log_p / t, dim=-1)
