# Copyright (c) 2026 ysnrfd
"""Vector scaling: per-class affine logits ``l'_k = scale_k * l_k + bias_k``."""

from __future__ import annotations

from typing import Self

import torch
import torch.nn as nn

from jevx.utils.numerical import stable_softmax

from .base import BaseCalibrator, _INPUT_TYPE, register_calibrator, validate_fit_inputs

__all__ = ["VectorCalibrator"]


@register_calibrator("vector")
class VectorCalibrator(BaseCalibrator):
    """Per-class affine (vector) scaling of logits.

    Mathematical contract::

        Cal(l)_k = scale_k * l_k + bias_k,   scale, bias in R^K

    For ``K == 1`` (a single Noul/Binary logit) this reduces to the scalar
    scale+bias form on that logit, fitted against binary cross-entropy.

    Fit: full-batch LBFGS on NLL with identity initialization
    (``scale=1, bias=0``). LBFGS is deterministic (no sampling, no momentum
    noise), so repeated fits on the same inputs give identical parameters.
    The ``seed`` argument is applied inside ``torch.random.fork_rng`` so
    any future stochastic element stays reproducible **without** mutating the
    caller's global RNG stream.

    Note: unlike temperature scaling, per-class scales need not preserve
    argmax in pathological cases; nothing is clamped or reordered — the raw
    fitted affine map is applied.
    """

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def _loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        calibrated = logits * self.scale + self.bias
        if logits.shape[1] == 1:
            p = torch.sigmoid(calibrated[:, 0]).clamp(1e-12, 1.0 - 1e-12)
            y = labels.to(logits.dtype)
            return -(y * p.log() + (1.0 - y) * (1.0 - p).log()).mean()
        return torch.nn.functional.nll_loss(
            torch.log_softmax(calibrated, dim=-1), labels
        )

    def fit(
        self, logits: torch.Tensor, labels: torch.Tensor, seed: int = 0
    ) -> Self:
        """Fit ``scale``/``bias`` by minimizing NLL (deterministic LBFGS).

        Args:
            logits: Calibration-split logits ``[N, K]``.
            labels: Calibration-split labels ``[N]``.
            seed: Seed applied to a **forked** RNG before optimizing, so the
                fit is reproducible without perturbing the caller's global
                random stream. LBFGS itself is RNG-free; the seed is kept so
                any future stochastic element stays reproducible.

        Returns:
            self.
        """
        logits, labels = validate_fit_inputs(logits, labels)
        # Determinism, not entropy: LBFGS is RNG-free, so the seed is scoped
        # to a forked generator. torch.manual_seed(seed) would re-seed the
        # CALLER's global stream as a side effect, changing every downstream
        # random draw (data-loader shuffles, dropout, ...). fork_rng makes
        # the fit reproducible without touching global RNG state.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed)

        # Identity init, resized to [K] so each class gets its own parameter.
        k = logits.shape[1]
        self.scale = nn.Parameter(torch.ones(k))
        self.bias = nn.Parameter(torch.zeros(k))

        data = logits.detach().to(
            dtype=torch.float32, device=self.scale.device
        )
        target = labels.long().to(self.scale.device)

        optimizer = torch.optim.LBFGS(
            [self.scale, self.bias],
            lr=1.0,
            max_iter=200,
            tolerance_grad=1e-9,
            tolerance_change=1e-11,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = self._loss(data, target)
            loss.backward()
            return loss

        optimizer.step(closure)
        self._fitted.fill_(True)
        return self

    def _ensure_width(self, k: int) -> None:
        if not self.is_fitted:
            return
        if self.scale.numel() != k:
            raise ValueError(
                f"VectorCalibrator: fitted for {self.scale.numel()} classes but received "
                f"{k}; calibration must not change the number of classes (re-fit on the "
                "matching K)."
            )

    @torch.no_grad()
    def calibrate_valid_block(
        self, x: torch.Tensor, input_type: _INPUT_TYPE = "logits"
    ) -> torch.Tensor:
        """Valid-column block ``[n, k]`` with per-class parameters aligned (Q-012).

        The per-class ``scale``/``bias`` vectors are indexed by SCHEMA
        position; a variable-K row's valid columns are its LEADING `k`
        positions (`mask_from_num_options` keeps valid options leading), so
        a calibrator fitted at the batch's `K_max` applies its first `k`
        parameters to that row's block. A calibrator fitted NARROWER than
        `k` cannot invent parameters — :meth:`_ensure_width` raises the
        actionable re-fit error instead.
        """
        k = x.shape[1]
        if not self.is_fitted or self.scale.numel() == k:
            return self.calibrate(x, input_type=input_type)
        if self.scale.numel() < k:
            self._ensure_width(k)  # raises: fitted narrower than this block
        scale = self.scale[:k].to(device=x.device, dtype=x.dtype)
        bias = self.bias[:k].to(device=x.device, dtype=x.dtype)
        if input_type == "logits":
            return x * scale + bias
        log_p = x.clamp_min(torch.finfo(x.dtype).tiny).log()
        return stable_softmax(log_p * scale + bias, dim=-1)

    @torch.no_grad()
    def calibrate_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """``l * scale + bias`` — same shape/dtype as ``logits`` (identity before fit)."""
        self._ensure_width(logits.shape[1])
        if not self.is_fitted:
            return logits
        scale = self.scale.to(device=logits.device, dtype=logits.dtype)
        bias = self.bias.to(device=logits.device, dtype=logits.dtype)
        return logits * scale + bias

    @torch.no_grad()
    def calibrate_probs(self, probs: torch.Tensor) -> torch.Tensor:
        """``softmax(scale * log p + bias)`` — softmax is shift-invariant."""
        self._ensure_width(probs.shape[1])
        if not self.is_fitted:
            return probs
        log_p = probs.clamp_min(torch.finfo(probs.dtype).tiny).log()
        scale = self.scale.to(device=probs.device, dtype=log_p.dtype)
        bias = self.bias.to(device=probs.device, dtype=log_p.dtype)
        return stable_softmax(log_p * scale + bias, dim=-1)
