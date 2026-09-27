# Copyright (c) 2026 ysnrfd
"""Confidence engine: registry of confidence measures over decision outputs.

All methods return a per-question confidence ``[B]``. Values lie in ``[0, 1]``
whenever the measure is mathematically bounded; the two documented
exceptions/edge rules:

* ``energy`` BEFORE :meth:`ConfidenceEngine.fit` returns the raw energy
  ``E(x) = -logsumexp(logits)`` (unbounded, more negative = more confident)
  together with a :class:`warnings.WarningMessage`; after ``fit(logits)`` the
  engine applies the affine map ``(hi - E) / (hi - lo)`` fitted on that
  calibration set, yielding ``[0, 1]`` (degenerate ``hi == lo`` -> 0.5).
* A WIDTH-1 probability column is read as the BINARY pair ``(p, 1 - p)``:
  confidence = ``max(p, 1 - p) in [0.5, 1]`` (spec Def 9(e): "Noul/Threshold
  with a single sigmoid: ``c_max = max(p, 1-p)``" — the same rule
  ``jevx.utils.numerical`` uses for ECE/NLL confidence). A genuine K=1
  *choice* row carries ``p == 1`` (softmax over a single column), so it still
  gets exactly 1; for width >= 2 every method dispatches unchanged. (Audit
  Q-002: the old "short-circuit to 1" made ``confidence_threshold``
  abstention a silent no-op for every noul decision.)
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Callable, ClassVar

import torch
import torch.nn as nn

from jevx.utils.shapes import assert_shape, check_finite

__all__ = [
    "CONFIDENCE_REGISTRY",
    "ConfidenceEngine",
    "available_confidence_methods",
    "get_confidence_method",
    "register_confidence_method",
]

CONFIDENCE_REGISTRY: dict[str, type[nn.Module]] = {}

_TINY = torch.finfo(torch.float32).tiny


def register_confidence_method(name: str) -> Callable[[type[nn.Module]], type[nn.Module]]:
    """Class decorator registering a confidence method under ``name``."""

    def decorator(cls: type[nn.Module]) -> type[nn.Module]:
        if name in CONFIDENCE_REGISTRY and CONFIDENCE_REGISTRY[name] is not cls:
            raise ValueError(
                f"confidence method {name!r} is already registered to "
                f"{CONFIDENCE_REGISTRY[name].__name__}."
            )
        CONFIDENCE_REGISTRY[name] = cls
        setattr(cls, "method_name", name)
        return cls

    return decorator


def available_confidence_methods() -> list[str]:
    """Sorted names of every registered confidence method."""
    return sorted(CONFIDENCE_REGISTRY)


def get_confidence_method(name: str) -> type[nn.Module]:
    """Look up a registered method class; unknown names raise with the roster."""
    if not isinstance(name, str):
        raise TypeError(f"confidence method name must be a str, got {type(name).__name__}.")
    if name not in CONFIDENCE_REGISTRY:
        raise ValueError(
            f"unknown confidence method {name!r}; registered methods are "
            f"{available_confidence_methods()}. Register new ones with "
            f"@register_confidence_method({name!r})."
        )
    return CONFIDENCE_REGISTRY[name]


# ---------------------------------------------------------------------------
# Shared feature computations (per-question, mask aware)
# ---------------------------------------------------------------------------


def _max_probability(probs: torch.Tensor) -> torch.Tensor:
    return probs.max(dim=-1).values


def _margin(probs: torch.Tensor) -> torch.Tensor:
    """top1 - top2; for width 1 the margin degenerates to 1 (top2 = 0)."""
    k = probs.shape[-1]
    if k == 1:
        return torch.ones(probs.shape[0], dtype=probs.dtype, device=probs.device)
    top2 = probs.topk(k=2, dim=-1).values
    return top2[:, 0] - top2[:, 1]


def _entropy_confidence(
    probs: torch.Tensor, option_mask: torch.Tensor | None
) -> torch.Tensor:
    """``1 - H(p)/log(K_eff)`` with ``K_eff`` the per-row valid option count.

    K_eff == 1 rows get confidence 1 by definition (log 1 == 0 would divide
    by zero). Invalid (masked) entries are exactly 0 and contribute nothing
    to H. For binary questions K_eff == 2 this is the standard normalized
    entropy ``1 - H(p)/log 2``.
    """
    p = probs.double()
    h = -(p * p.clamp_min(_TINY).log()).sum(dim=-1)  # 0 * log 0 == 0
    if option_mask is not None:
        k_eff = option_mask.sum(dim=-1).to(torch.long)
    else:
        k_eff = torch.full(
            (p.shape[0],), p.shape[-1], dtype=torch.long, device=p.device
        )
    conf = torch.ones_like(h)
    active = k_eff > 1
    if bool(active.any()):
        norm = k_eff[active].to(h.dtype).log()
        conf[active] = 1.0 - h[active] / norm
    return conf.clamp(0.0, 1.0).to(probs.dtype)


def _validate_probs(
    probabilities: torch.Tensor, option_mask: torch.Tensor | None
) -> None:
    assert_shape(probabilities, (None, None), "probabilities")
    if not probabilities.is_floating_point():
        raise TypeError(
            f"probabilities: expected a floating-point tensor, got {probabilities.dtype}."
        )
    check_finite(probabilities, "probabilities")
    if option_mask is not None:
        if option_mask.dtype != torch.bool:
            raise TypeError(
                f"option_mask: expected a torch.bool tensor, got {option_mask.dtype}; "
                "convert it with `.bool()`."
            )
        if tuple(option_mask.shape) != tuple(probabilities.shape):
            raise ValueError(
                f"option_mask: expected the same shape as probabilities "
                f"{tuple(probabilities.shape)} but got {tuple(option_mask.shape)}."
            )


# ---------------------------------------------------------------------------
# Method implementations
# ---------------------------------------------------------------------------


class _ConfidenceMethod(nn.Module):
    """Base class for registered confidence methods."""

    method_name: ClassVar[str] = ""

    def forward(
        self,
        probabilities: torch.Tensor,
        *,
        logits: torch.Tensor | None = None,
        option_mask: torch.Tensor | None = None,
        fused: torch.Tensor | None = None,
    ) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError


@register_confidence_method("max_probability")
class MaxProbabilityConfidence(_ConfidenceMethod):
    """``conf = max_k p_k`` — directly in ``[0, 1]``.

    Width-1 columns never reach a method: :class:`ConfidenceEngine` applies
    the binary ``max(p, 1 - p)`` rule first (module docstring).
    """

    def forward(
        self,
        probabilities: torch.Tensor,
        *,
        logits: torch.Tensor | None = None,
        option_mask: torch.Tensor | None = None,
        fused: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return _max_probability(probabilities).clamp(0.0, 1.0)


@register_confidence_method("margin")
class MarginConfidence(_ConfidenceMethod):
    """``conf = p_(1) - p_(2)`` (top-1 minus top-2 probability) in ``[0, 1]``.

    For a mask leaving one valid option with the rest at exactly 0 this
    degenerates to ``1 - 0 = 1``. (A width-1 column never reaches a method:
    the engine's binary ``max(p, 1 - p)`` rule applies first — module
    docstring.)
    """

    def forward(
        self,
        probabilities: torch.Tensor,
        *,
        logits: torch.Tensor | None = None,
        option_mask: torch.Tensor | None = None,
        fused: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return _margin(probabilities).clamp(0.0, 1.0)


@register_confidence_method("entropy")
class EntropyConfidence(_ConfidenceMethod):
    """Normalized-entropy confidence ``1 - H(p)/log(K_eff)`` in ``[0, 1]``.

    Uniform over the valid options -> 0; one-hot -> 1; K_eff == 1 -> 1.
    """

    def forward(
        self,
        probabilities: torch.Tensor,
        *,
        logits: torch.Tensor | None = None,
        option_mask: torch.Tensor | None = None,
        fused: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return _entropy_confidence(probabilities, option_mask)


@register_confidence_method("energy")
class EnergyConfidence(_ConfidenceMethod):
    """Energy score ``E(x) = -logsumexp(logits)`` with affine normalization.

    Sign convention: LOWER (more negative) energy = more in-distribution =
    more confident, so the fitted map is ``conf = (hi - E) / (hi - lo)`` with
    ``lo``/``hi`` the min/max energy on the calibration set passed to
    :meth:`fit`. Before fitting, ``forward`` returns the RAW energy (not in
    ``[0, 1]``) and emits a warning telling the caller to fit first.

    Spec cross-reference (audit Q-035 — CODE SIDE IS CONSISTENT, no change):
    Def 9.4 defines ``c_eng(z) = -E = logsumexp(z/T_e)`` and itself carries
    the matching "Implementation sign note [FACT]": JevX returns the raw E
    pre-fit (the NEGATIVE of c_eng — more negative = more confident, this
    docstring's contract) and the affine ``(hi - E)/(hi - lo)`` post-fit,
    whose ordering matches c_eng.
    tests/unit/test_confidence.py::test_energy_unfitted_returns_raw_with_warning
    pins ``raw == -logsumexp``; flipping the pre-fit sign would contradict
    the spec's own sign note and break the pinned contract — any further
    reconciliation belongs in docs (CONFIDENCE.md §1), not here.

    Masked-out options (``option_mask``) are pushed to ``finfo.min`` before
    the logsumexp, so invalid columns do not leak into the energy.

    Fit state is an explicit ``_fitted`` bool buffer in the state_dict (the
    calibrator convention, ``jevx/calibration/base.py``); ``energy_lo``/
    ``energy_hi`` hold the fitted extremes. A checkpoint written before
    ``_fitted`` existed loads unchanged: the flag is derived from the
    ``energy_lo`` NaN sentinel (see `_load_from_state_dict`).
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("energy_lo", torch.tensor(float("nan")))
        self.register_buffer("energy_hi", torch.tensor(float("nan")))
        # Persisted fit flag, following the calibrator convention
        # (jevx/calibration/base.py:149): an explicit bool buffer is part of
        # the state_dict instead of a NaN sentinel, so "never fitted" is a
        # value rather than a magic number. energy_lo/energy_hi still start at
        # NaN and still round-trip through old checkpoints.
        self.register_buffer("_fitted", torch.tensor(False))

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has run at least once (persisted ``_fitted``)."""
        return bool(self._fitted)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Load state, accepting pre-``_fitted`` (energy-buffers-only) dicts.

        Checkpoints written before the flag existed carry only
        ``energy_lo``/``energy_hi``, whose ``energy_lo`` NaN sentinel was
        the old fit marker. When ``_fitted`` is absent the flag is therefore
        DERIVED from that buffer (not-NaN -> fitted) and the synthetic miss is
        dropped from ``missing_keys`` so a strict load succeeds unchanged.
        """
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        if f"{prefix}_fitted" in state_dict:
            return
        key = f"{prefix}_fitted"
        if strict and key in missing_keys:
            missing_keys.remove(key)
        lo = state_dict.get(f"{prefix}energy_lo")
        fitted = lo is not None and not bool(torch.isnan(torch.as_tensor(lo)))
        self._fitted.fill_(fitted)

    @staticmethod
    def _energy(logits: torch.Tensor, option_mask: torch.Tensor | None) -> torch.Tensor:
        if option_mask is not None:
            logits = logits.masked_fill(
                ~option_mask, torch.finfo(logits.dtype).min
            )
        return -torch.logsumexp(logits, dim=-1)

    def fit(
        self, logits: torch.Tensor, option_mask: torch.Tensor | None = None
    ) -> "EnergyConfidence":
        """Fit the affine ``[0, 1]`` normalization on a calibration batch.

        Args:
            logits: Raw energies source ``[N, K]`` (from the calibration
                split — never the test split).
            option_mask: Optional ``[N, K]`` validity mask.

        Returns:
            self (buffers ``energy_lo``/``energy_hi`` and ``_fitted``
            updated).
        """
        check_finite(logits, "logits")
        e = self._energy(logits, option_mask)
        self.energy_lo.copy_(e.min().detach())
        self.energy_hi.copy_(e.max().detach())
        self._fitted.fill_(True)
        return self

    def forward(
        self,
        probabilities: torch.Tensor,
        *,
        logits: torch.Tensor | None = None,
        option_mask: torch.Tensor | None = None,
        fused: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if logits is None:
            raise ValueError(
                "energy confidence requires raw `logits=` — probabilities discard the logit "
                "scale that the energy score is defined on."
            )
        check_finite(logits, "logits")
        energy = self._energy(logits, option_mask)
        if not self.is_fitted:
            warnings.warn(
                "ConfidenceEngine(method='energy'): returning RAW energy "
                f"E = -logsumexp(logits) (range {float(energy.min()):.3f} .. "
                f"{float(energy.max()):.3f}), which is NOT in [0, 1]; call "
                "engine.fit(calibration_logits) to enable the affine [0, 1] mapping.",
                stacklevel=2,
            )
            return energy
        lo, hi = float(self.energy_lo), float(self.energy_hi)
        if math.isclose(lo, hi, rel_tol=0.0, abs_tol=1e-12):
            # No spread on the calibration set: no discriminative information.
            return torch.full_like(energy, 0.5)
        return ((hi - energy) / (hi - lo)).clamp(0.0, 1.0)


@register_confidence_method("learned")
class LearnedConfidence(_ConfidenceMethod):
    """Learned confidence: small MLP over ``[fused, max_prob, H_norm, margin]``.

    Features ``x = [fused ; max_k p_k ; 1 - H/log K ; p_(1)-p_(2)]`` in
    ``[B, D + 3]`` are mapped through ``MLP(D+3 -> hidden -> 1)`` and a
    sigmoid, giving ``[B]`` in ``(0, 1)``. The module exposes normal
    parameters — training is optional and left to the caller (loss: e.g. MSE
    against correctness). Requires ``fused=`` at forward time.

    Args:
        fused_dim: Width ``D`` of the fused state.
        hidden_size: MLP hidden width (default 32).
    """

    def __init__(self, fused_dim: int, hidden_size: int = 32) -> None:
        super().__init__()
        if fused_dim < 1:
            raise ValueError(f"fused_dim must be >= 1, got {fused_dim}.")
        self.fused_dim = fused_dim
        self.mlp = nn.Sequential(
            nn.Linear(fused_dim + 3, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        probabilities: torch.Tensor,
        *,
        logits: torch.Tensor | None = None,
        option_mask: torch.Tensor | None = None,
        fused: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if fused is None:
            raise ValueError(
                "learned confidence requires `fused=` ([B, D] state) — its input features are "
                "[fused, max_prob, entropy, margin]."
            )
        assert_shape(fused, (probabilities.shape[0], self.fused_dim), "fused")
        feats = torch.cat(
            [
                fused,
                _max_probability(probabilities).unsqueeze(-1),
                _entropy_confidence(probabilities, option_mask).unsqueeze(-1),
                _margin(probabilities).unsqueeze(-1),
            ],
            dim=-1,
        )  # [B, D + 3]
        return torch.sigmoid(self.mlp(feats).squeeze(-1))


# ---------------------------------------------------------------------------
# Engine facade
# ---------------------------------------------------------------------------


class ConfidenceEngine(nn.Module):
    """Config-driven facade: ``confidence_method`` name -> concrete method.

    Example:
        >>> engine = ConfidenceEngine("entropy")
        >>> conf = engine(probs, option_mask=mask)   # [B] in [0, 1]

    For ``"energy"``, call :meth:`fit` on a calibration batch first to get
    ``[0, 1]`` outputs; for ``"learned"``, train ``engine.method_module``
    parameters yourself and pass ``fused=`` at forward time.

    Args:
        method: Registered method name (see
            :func:`available_confidence_methods`).
        **method_kwargs: Forwarded to the method constructor (e.g.
            ``fused_dim=`` for ``"learned"``).
    """

    def __init__(self, method: str = "max_probability", **method_kwargs: Any) -> None:
        super().__init__()
        cls = get_confidence_method(method)
        self.method_name = method
        self.method_module: nn.Module = cls(**method_kwargs)

    @classmethod
    def from_config(cls, config: Any) -> "ConfidenceEngine":
        """Build from an object exposing ``confidence_method`` (e.g. JevXConfig)."""
        return cls(getattr(config, "confidence_method", "max_probability"))

    @property
    def method_class(self) -> type[nn.Module]:
        """The registered class backing this engine."""
        return get_confidence_method(self.method_name)

    def fit(
        self, logits: torch.Tensor, option_mask: torch.Tensor | None = None
    ) -> "ConfidenceEngine":
        """Fit method parameters that need data (currently only ``energy``).

        ``max_probability``/``margin``/``entropy``/``learned`` need no fit
        and ignore the inputs; ``energy`` fits its affine ``[0, 1]``
        normalization on ``logits`` (pass a CALIBRATION split, never test).

        Returns:
            self, for chaining.
        """
        fit_fn = getattr(self.method_module, "fit", None)
        if callable(fit_fn):
            fit_fn(logits, option_mask)
        return self

    def forward(
        self,
        probabilities: torch.Tensor,
        *,
        logits: torch.Tensor | None = None,
        option_mask: torch.Tensor | None = None,
        fused: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute per-question confidence.

        Args:
            probabilities: ``[B, K]`` (mask-aware; invalid rows entries 0).
            logits: ``[B, K]`` raw logits — REQUIRED for ``"energy"``.
            option_mask: ``[B, K]`` bool mask (``True`` = valid option).
            fused: ``[B, D]`` fused state — REQUIRED for ``"learned"``.

        Returns:
            Confidence ``[B]`` in ``[0, 1]`` (see module docstring for the
            raw-energy exception before fitting).
        """
        _validate_probs(probabilities, option_mask)
        if probabilities.shape[-1] == 1:
            # Single column = the binary pair (p, 1 - p) — spec Def 9(e)
            # ("Noul/Threshold with a single sigmoid: c_max = max(p, 1-p)"),
            # identical to the K==1 branch of
            # jevx.utils.numerical._confidence_and_correct (ECE/NLL path).
            # A genuine K=1 *choice* row has p == 1 (softmax over one column),
            # so max(p, 1 - p) still returns exactly 1 there; a noul row with
            # p near 0.5 now gets ~0.5, which is what makes Def-10
            # confidence_threshold abstention fire for noul decisions
            # (audit Q-002: the previous hard-wired 1.0 was a silent no-op).
            p = probabilities[:, 0].clamp(0.0, 1.0)
            return torch.maximum(p, 1.0 - p)
        return self.method_module(
            probabilities, logits=logits, option_mask=option_mask, fused=fused
        )
