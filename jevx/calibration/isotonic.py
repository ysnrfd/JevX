# Copyright (c) 2026 ysnrfd
"""Top-label isotonic regression fitted with the pool-adjacent-violators (PAVA)
algorithm — implemented here, no sklearn dependency."""

from __future__ import annotations

from typing import Self

import torch

from jevx.utils.numerical import stable_softmax

from .base import BaseCalibrator, register_calibrator, validate_fit_inputs

__all__ = ["IsotonicCalibrator", "pava_isotonic"]


def pava_isotonic(
    x: torch.Tensor, y: torch.Tensor, weights: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted isotonic (nondecreasing) regression via PAVA.

    Adjacent violators are pooled left-to-right; pooling is weighted by
    ``weights`` (e.g. tie-group sizes). Fully deterministic.

    Args:
        x: Sorted, strictly increasing inputs ``[M]`` (the caller groups
            ties first — this function assumes unique ``x``).
        y: Targets ``[M]`` to be made nondecreasing.
        weights: Optional positive weights ``[M]`` (default all ones).

    Returns:
        ``(x, y_fit)`` where ``y_fit`` is nondecreasing and has the same
        shape as ``y``.
    """
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError(
            f"pava_isotonic: x and y must be 1-D of equal length, got "
            f"{tuple(x.shape)} and {tuple(y.shape)}."
        )
    m = x.numel()
    if m == 0:
        return x, y
    if weights is None:
        weights = torch.ones_like(y)
    x64 = x.to(torch.float64)
    y64 = y.to(torch.float64)
    w64 = weights.to(torch.float64)

    block_w: list[float] = []
    block_v: list[float] = []
    block_n: list[int] = []
    for i in range(m):
        w_cur = float(w64[i])
        v_cur = float(y64[i])
        n_cur = 1
        # Pool with the left neighbour while monotonicity is violated.
        while block_v and v_cur < block_v[-1]:
            w_left = block_w.pop()
            v_left = block_v.pop()
            n_left = block_n.pop()
            total = w_left + w_cur
            v_cur = (w_left * v_left + w_cur * v_cur) / total
            w_cur = total
            n_cur += n_left
        block_w.append(w_cur)
        block_v.append(v_cur)
        block_n.append(n_cur)

    y_fit = torch.empty(m, dtype=torch.float64, device=y.device)
    pos = 0
    for v, n in zip(block_v, block_n):
        y_fit[pos : pos + n] = v
        pos += n
    return x64, y_fit


@register_calibrator("isotonic")
class IsotonicCalibrator(BaseCalibrator):
    """Top-label isotonic calibration: monotone map ``confidence -> accuracy``.

    Fit (on a CALIBRATION split): probabilities ``p = softmax(l)`` (sigmoid
    for K == 1); confidence ``c = max_k p_k``; target ``r = 1[argmax p == y]``;
    ties in ``c`` are grouped (mean target, weighted PAVA) so the learned map
    is a well-defined monotone function of ``c``; the result is stored as
    piecewise-linear breakpoints ``(x_points, y_points)`` (float64).

    Predict: linear interpolation of the breakpoints at each confidence
    (clamped outside the fitted range). For multiclass, the TOP-label
    probability is replaced by the calibrated value ``q = g(c)`` and the
    remaining mass ``1 - q`` is redistributed proportionally over the other
    classes, preserving shape, dtype and ``K``. For K == 1 the binary
    probability is corrected symmetrically (``p`` or ``1-p`` calibrated
    according to the predicted side).

    Contract notes: ``g`` is nondecreasing with ``g(c) in [0, 1]``; unlike
    temperature scaling this transform can, in principle, reorder close
    classes — argmax preservation is NOT guaranteed (documented, tested only
    for temperature).
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "x_points", torch.empty(0, dtype=torch.float64)
        )
        self.register_buffer(
            "y_points", torch.empty(0, dtype=torch.float64)
        )

    def fit(self, logits: torch.Tensor, labels: torch.Tensor) -> Self:
        """Fit the monotone confidence map with weighted PAVA.

        Args:
            logits: Calibration-split logits ``[N, K]``.
            labels: Calibration-split labels ``[N]``.

        Returns:
            self.
        """
        logits, labels = validate_fit_inputs(logits, labels)
        data = logits.detach().to(torch.float64)
        k = data.shape[1]
        if k == 1:
            p = torch.sigmoid(data[:, 0])
            pred = (p >= 0.5).to(torch.long)
            conf = torch.maximum(p, 1.0 - p)
        else:
            probs = torch.softmax(data, dim=-1)
            conf, pred = probs.max(dim=-1)
        correct = (pred == labels.long()).to(torch.float64)

        # Stable sort groups equal confidences adjacently; then average ties
        # so each distinct confidence maps to exactly one fitted value.
        order = torch.argsort(conf, stable=True)
        xs, ys = conf[order], correct[order]
        xu, inv_counts = torch.unique_consecutive(xs, return_inverse=True)
        # Mean target per distinct confidence (weighted tie groups).
        sums = torch.zeros_like(xu).scatter_add(0, inv_counts, ys)
        counts = torch.zeros_like(xu).scatter_add(
            0, inv_counts, torch.ones_like(ys)
        )
        yu = sums / counts.clamp_min(1.0)

        x_fit, y_fit = pava_isotonic(xu, yu, weights=counts)
        self.x_points = x_fit.to(torch.float64)
        self.y_points = y_fit.to(torch.float64)
        self._fitted.fill_(True)
        return self

    def _require_fitted(self) -> None:
        if not self.is_fitted or self.x_points.numel() == 0:
            raise RuntimeError(
                "IsotonicCalibrator: call fit(logits, labels) on a calibration split "
                "before applying the calibration."
            )

    @staticmethod
    def _interpolate(x_points: torch.Tensor, y_points: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Piecewise-linear interpolation of monotone breakpoints at ``c``."""
        m = x_points.numel()
        if m == 1:
            return torch.full_like(c, float(y_points[0]))
        idx = torch.searchsorted(x_points, c, right=False).clamp(1, m - 1)
        x0, x1 = x_points[idx - 1], x_points[idx]
        y0, y1 = y_points[idx - 1], y_points[idx]
        denom = (x1 - x0).clamp_min(torch.finfo(torch.float64).eps)
        t = ((c - x0) / denom).clamp(0.0, 1.0)
        return y0 + t * (y1 - y0)

    @torch.no_grad()
    def calibrate_probs(self, probs: torch.Tensor) -> torch.Tensor:
        """Apply the fitted map; same shape/dtype/``K`` as ``probs``."""
        self._require_fitted()
        xp = self.x_points.to(probs.device)
        yp = self.y_points.to(probs.device)
        p = probs.to(torch.float64)
        k = p.shape[1]

        if k == 1:
            raw = p[:, 0].clamp(0.0, 1.0)
            above = raw >= 0.5
            conf = torch.where(above, raw, 1.0 - raw)
            q = self._interpolate(xp, yp, conf).clamp(0.0, 1.0)
            calibrated = torch.where(above, q, 1.0 - q)
            return calibrated.unsqueeze(-1).to(probs.dtype)

        conf, top = p.max(dim=-1)
        q = self._interpolate(xp, yp, conf).clamp(0.0, 1.0)
        rest = (1.0 - conf).clamp_min(1e-12)
        scale = (1.0 - q) / rest
        out = p * scale.unsqueeze(-1)
        out.scatter_(1, top.unsqueeze(1), q.unsqueeze(1))
        return out.to(probs.dtype)

    @torch.no_grad()
    def calibrate_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Return a logit representative of the calibrated probabilities.

        For ``K >= 2`` that is ``log p'`` (``softmax`` of it reproduces
        ``calibrate_probs(softmax(logits))`` up to an additive constant);
        for ``K == 1`` it is the log-odds ``logit(p')`` so ``sigmoid`` of it
        reproduces the calibrated probability. Shape, dtype and ``K`` are
        unchanged either way.
        """
        self._require_fitted()
        # Use the same input pipeline as `apply(..., "probs")` (stable
        # softmax on the caller's scale) so both paths see identical
        # probabilities and stay consistent up to output rounding.
        if logits.shape[1] == 1:
            p = torch.sigmoid(logits)
            p_cal = self.calibrate_probs(p).to(torch.float64)[:, 0]
            p_cal = p_cal.clamp(1e-12, 1.0 - 1e-12)
            log_odds = (p_cal / (1.0 - p_cal)).log()
            return log_odds.to(logits.dtype).unsqueeze(-1)
        probs = stable_softmax(logits, dim=-1)
        calibrated = self.calibrate_probs(probs)
        return calibrated.clamp_min(torch.finfo(logits.dtype).tiny).log().to(logits.dtype)
