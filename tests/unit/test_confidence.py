# Copyright (c) 2026 ysnrfd
"""Unit tests for the confidence engine: ranges, edge cases, entropy/margin
properties and registry dispatch."""

from __future__ import annotations

import math
import warnings

import pytest
import torch

from jevx.confidence import (
    CONFIDENCE_REGISTRY,
    ConfidenceEngine,
    available_confidence_methods,
    get_confidence_method,
    register_confidence_method,
)
from jevx.utils.numerical import stable_softmax


def _random_probs(batch: int, k: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return stable_softmax(torch.randn(batch, k))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_contains_all_methods():
    assert set(available_confidence_methods()) == {
        "max_probability",
        "margin",
        "entropy",
        "energy",
        "learned",
    }


def test_registry_dispatch_and_unknown_error():
    for name in ("max_probability", "margin", "entropy", "energy"):
        engine = ConfidenceEngine(name)
        assert engine.method_name == name
        assert engine.method_class is get_confidence_method(name)
    with pytest.raises(ValueError, match="unknown confidence method"):
        get_confidence_method("vibes")
    with pytest.raises(ValueError, match="entropy"):
        get_confidence_method("vibes")  # message lists the roster


def test_custom_method_registration_with_cleanup():
    name = "unit_test_probe_conf"
    try:
        @register_confidence_method(name)
        class Probe:
            def forward(self, probabilities, **kwargs):
                return torch.zeros(probabilities.shape[0])

        assert get_confidence_method(name) is Probe
        with pytest.raises(ValueError, match="already registered"):
            register_confidence_method("entropy")(Probe)
    finally:
        CONFIDENCE_REGISTRY.pop(name, None)


def test_engine_from_config():
    class Cfg:
        confidence_method = "margin"

    assert ConfidenceEngine.from_config(Cfg()).method_name == "margin"
    # missing attribute falls back to the default
    assert ConfidenceEngine.from_config(object()).method_name == "max_probability"


# ---------------------------------------------------------------------------
# Ranges
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["max_probability", "margin", "entropy"])
@pytest.mark.parametrize("k", [2, 5, 8])
def test_range_for_differentiable_methods(method, k):
    probs = _random_probs(16, k, seed=k)
    conf = ConfidenceEngine(method)(probs)
    assert conf.shape == (16,)
    assert torch.isfinite(conf).all()
    assert (conf >= 0.0).all() and (conf <= 1.0).all()


def test_max_probability_matches_formula():
    probs = _random_probs(8, 4, seed=1)
    conf = ConfidenceEngine("max_probability")(probs)
    assert torch.allclose(conf, probs.max(dim=-1).values, atol=1e-7)


# ---------------------------------------------------------------------------
# Width-1 edge: the binary max(p, 1-p) rule (spec Def 9(e), audit Q-002)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method", ["max_probability", "margin", "entropy", "energy", "learned"]
)
def test_width_one_binary_rule(method):
    # EDITED (bug-encoding assertion): the old version pinned
    # ``conf == 1.0`` for a width-1 column with p = 0.7 — that hard-wired
    # short-circuit is audit Q-002 (Def-10 abstention never fired for noul).
    # Correct rule (spec Def 9(e) / ECE path): conf = max(p, 1-p).
    kwargs = {"fused_dim": 3} if method == "learned" else {}
    fused4 = torch.randn(4, 3) if method == "learned" else None
    engine = ConfidenceEngine(method, **kwargs)

    probs = torch.ones(4, 1) * 0.7
    conf = engine(probs, fused=fused4)
    assert conf.shape == (4,)
    assert torch.equal(conf, torch.full((4,), 0.7))  # max(0.7, 0.3) = 0.7

    # the Q-002 probe value: p = 0.4954 -> 0.5046, NOT 1.0
    near = torch.full((1, 1), 0.4954)
    conf_near = engine(near, fused=torch.randn(1, 3) if method == "learned" else None)
    assert float(conf_near[0]) == pytest.approx(0.5046, abs=1e-6)

    # a genuine K=1 CHOICE row carries p == 1 -> still exactly 1
    one = torch.ones(4, 1)
    conf_one = engine(one, fused=fused4)
    assert torch.equal(conf_one, torch.ones(4))


# ---------------------------------------------------------------------------
# Entropy properties
# ---------------------------------------------------------------------------


def test_entropy_of_uniform_is_zero_confidence():
    for k in (2, 3, 7, 20):
        probs = torch.full((1, k), 1.0 / k, dtype=torch.float64)
        conf = ConfidenceEngine("entropy")(probs)
        assert float(conf) == pytest.approx(0.0, abs=1e-6)


def test_entropy_of_one_hot_is_one_confidence():
    for k in (2, 3, 7, 20):
        probs = torch.zeros(1, k)
        probs[0, k // 2] = 1.0
        conf = ConfidenceEngine("entropy")(probs)
        assert float(conf) == pytest.approx(1.0, abs=1e-6)


def test_entropy_binary_standard_normalized_form():
    p = 0.2
    probs = torch.tensor([[p, 1 - p]], dtype=torch.float64)
    h = -(p * math.log(p) + (1 - p) * math.log(1 - p))
    expected = 1.0 - h / math.log(2.0)
    conf = ConfidenceEngine("entropy")(probs)
    assert float(conf) == pytest.approx(expected, abs=1e-9)


def test_entropy_decreases_with_more_uniformity():
    sharp = torch.tensor([[0.97, 0.02, 0.01]], dtype=torch.float64)
    flat = torch.tensor([[0.34, 0.33, 0.33]], dtype=torch.float64)
    engine = ConfidenceEngine("entropy")
    assert float(engine(sharp)) > float(engine(flat))


def test_entropy_uses_per_row_valid_counts_with_mask():
    # two valid options, uniformly split -> confidence 0 for that row
    probs = torch.tensor([[0.5, 0.5, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    mask = torch.tensor(
        [[True, True, False, False], [True, False, False, False]]
    )
    conf = ConfidenceEngine("entropy")(probs, option_mask=mask)
    assert float(conf[0]) == pytest.approx(0.0, abs=1e-6)  # K_eff = 2, uniform
    assert float(conf[1]) == pytest.approx(1.0, abs=1e-6)  # K_eff = 1 -> 1


# ---------------------------------------------------------------------------
# Margin properties
# ---------------------------------------------------------------------------


def test_margin_values():
    engine = ConfidenceEngine("margin")
    probs = torch.tensor(
        [
            [0.6, 0.4, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.7, 0.2, 0.1, 0.0],
            [0.25, 0.25, 0.25, 0.25],
        ],
        dtype=torch.float64,
    )
    conf = engine(probs)
    assert float(conf[0]) == pytest.approx(0.2, abs=1e-9)
    assert float(conf[1]) == pytest.approx(1.0, abs=1e-9)
    assert float(conf[2]) == pytest.approx(0.5, abs=1e-9)
    assert float(conf[3]) == pytest.approx(0.0, abs=1e-9)
    # sharper distribution -> larger margin
    sharp = torch.tensor([[0.99, 0.01]], dtype=torch.float64)
    mild = torch.tensor([[0.55, 0.45]], dtype=torch.float64)
    assert float(engine(sharp)) > float(engine(mild))


def test_margin_with_single_valid_option_is_one():
    probs = torch.tensor([[1.0, 0.0, 0.0]])
    mask = torch.tensor([[True, False, False]])
    conf = ConfidenceEngine("margin")(probs, option_mask=mask)
    assert float(conf) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Energy
# ---------------------------------------------------------------------------


def test_energy_requires_logits():
    engine = ConfidenceEngine("energy")
    with pytest.raises(ValueError, match="logits"):
        engine(_random_probs(4, 3, seed=0))


def test_energy_unfitted_returns_raw_with_warning():
    engine = ConfidenceEngine("energy")
    logits = torch.randn(6, 4)
    probs = stable_softmax(logits)
    with pytest.warns(UserWarning, match="energy"):
        raw = engine(probs, logits=logits)
    expected = -torch.logsumexp(logits, dim=-1)
    assert torch.allclose(raw, expected, atol=1e-5)
    assert (raw <= 0).all()  # raw energies are non-positive here, not [0, 1]


def test_energy_fitted_maps_to_unit_interval():
    torch.manual_seed(0)
    logits = torch.randn(50, 5)
    probs = stable_softmax(logits)
    engine = ConfidenceEngine("energy").fit(logits)
    conf = engine(probs, logits=logits)
    assert (conf >= 0.0).all() and (conf <= 1.0).all()
    # affine map anchored on the calibration extremes
    assert float(conf.min()) == pytest.approx(0.0, abs=1e-6)
    assert float(conf.max()) == pytest.approx(1.0, abs=1e-6)


def test_energy_confidence_ordering_matches_energy_sign():
    # lower energy = more confident: a peaked row must beat a flat row
    peaked = torch.tensor([[20.0, 0.0, 0.0]])
    flat = torch.tensor([[0.0, 0.0, 0.0]])
    logits = torch.cat([peaked, flat], dim=0)
    probs = stable_softmax(logits)
    engine = ConfidenceEngine("energy").fit(logits)
    conf = engine(probs, logits=logits)
    assert float(conf[0]) > float(conf[1])


def test_energy_degenerate_calibration_set_gives_half():
    logits = torch.ones(4, 3)  # identical rows -> zero energy spread
    engine = ConfidenceEngine("energy").fit(logits)
    conf = engine(stable_softmax(logits), logits=logits)
    assert torch.allclose(conf, torch.full((4,), 0.5), atol=1e-6)


def test_energy_masked_options_do_not_leak():
    # A masked-out huge logit must not dominate the logsumexp
    logits = torch.tensor([[0.0, 0.0, 500.0]])
    mask = torch.tensor([[True, True, False]])
    engine = ConfidenceEngine("energy")
    with pytest.warns(UserWarning, match="energy"):
        raw = engine(stable_softmax(logits), logits=logits, option_mask=mask)
    expected = -torch.logsumexp(logits[:, :2], dim=-1)
    assert torch.allclose(raw, expected, atol=1e-4)


# ---------------------------------------------------------------------------
# Energy fit state (explicit ``_fitted`` buffer, NaN-sentinel back-compat)
# ---------------------------------------------------------------------------


def test_energy_fit_state_is_an_explicit_bool_buffer_in_the_state_dict():
    """G13: fit state is a persisted bool, not the NaN sentinel on energy_lo."""
    engine = ConfidenceEngine("energy")
    method = engine.method_module
    assert "_fitted" in dict(method.named_buffers())
    assert method._fitted.dtype is torch.bool
    assert method.is_fitted is False
    # the fresh buffers are still the unfitted NaN placeholders
    assert bool(torch.isnan(method.energy_lo)) and bool(torch.isnan(method.energy_hi))
    assert method.state_dict()["_fitted"].dtype is torch.bool
    assert not bool(method.state_dict()["_fitted"])

    method.fit(torch.randn(8, 4))
    assert method.is_fitted is True
    assert bool(method.state_dict()["_fitted"])
    assert not bool(torch.isnan(method.energy_lo))
    # is_fitted no longer tracks the NaN-ness of energy_lo
    method.energy_lo.fill_(float("nan"))
    assert method.is_fitted is True


def test_energy_state_dict_round_trip_preserves_fit_state():
    """G13: a saved+loaded engine keeps the affine mapping bit-for-bit."""
    torch.manual_seed(7)
    calibration = torch.randn(40, 5)
    probe = torch.randn(6, 5)
    probs = stable_softmax(probe)
    source = ConfidenceEngine("energy").fit(calibration)
    assert source.method_module.is_fitted

    restored = ConfidenceEngine("energy")
    assert restored.method_module.is_fitted is False
    restored.load_state_dict(source.state_dict())  # strict=True
    assert restored.method_module.is_fitted is True
    assert torch.equal(
        restored.method_module.energy_lo, source.method_module.energy_lo
    )
    assert torch.equal(
        restored.method_module.energy_hi, source.method_module.energy_hi
    )
    assert torch.equal(restored(probs, logits=probe), source(probs, logits=probe))


def test_energy_legacy_state_dict_without_fitted_still_loads_strictly():
    """G13: a pre-``_fitted`` checkpoint (two energy buffers) loads unchanged.

    The flag is DERIVED from the old NaN sentinel on ``energy_lo``, and the
    synthetic missing key is cleared so ``strict=True`` does not fail.
    """
    torch.manual_seed(11)
    calibration = torch.randn(30, 4)
    probe = torch.randn(5, 4)
    probs = stable_softmax(probe)
    source = ConfidenceEngine("energy").fit(calibration)

    # exactly what an old checkpoint carried: the two energy buffers only
    legacy = {
        name: value
        for name, value in source.method_module.state_dict().items()
        if name in ("energy_lo", "energy_hi")
    }
    assert set(legacy) == {"energy_lo", "energy_hi"}

    restored = ConfidenceEngine("energy").method_module
    restored.load_state_dict(legacy)  # strict=True must not raise
    assert restored.is_fitted is True

    # the same legacy payload loaded through the owning engine (prefixed keys)
    nested = ConfidenceEngine("energy")
    nested.load_state_dict(
        {f"method_module.{k}": v for k, v in legacy.items()}
    )  # strict=True must not raise
    assert nested.method_module.is_fitted is True
    assert torch.equal(nested(probs, logits=probe), source(probs, logits=probe))


def test_energy_legacy_unfitted_state_dict_stays_unfitted():
    """G13: a legacy dict whose energy_lo is still NaN loads as NOT fitted."""
    legacy = {
        "energy_lo": torch.tensor(float("nan")),
        "energy_hi": torch.tensor(float("nan")),
    }
    restored = ConfidenceEngine("energy")
    restored.method_module.load_state_dict(legacy)
    assert restored.method_module.is_fitted is False
    logits = torch.randn(4, 3)
    probs = stable_softmax(logits)
    with pytest.warns(UserWarning, match="energy"):
        raw = restored(probs, logits=logits)
    assert torch.allclose(raw, -torch.logsumexp(logits, dim=-1), atol=1e-5)


# ---------------------------------------------------------------------------
# Learned
# ---------------------------------------------------------------------------


def test_learned_requires_fused_and_returns_unit_range():
    probs = _random_probs(8, 4, seed=2)
    engine = ConfidenceEngine("learned", fused_dim=6, hidden_size=8)
    with pytest.raises(ValueError, match="fused"):
        engine(probs)
    fused = torch.randn(8, 6)
    conf = engine(probs, fused=fused)
    assert conf.shape == (8,)
    assert (conf > 0.0).all() and (conf < 1.0).all()


def test_learned_parameters_get_gradients():
    probs = _random_probs(8, 4, seed=3)
    engine = ConfidenceEngine("learned", fused_dim=5)
    fused = torch.randn(8, 5, requires_grad=True)
    conf = engine(probs, fused=fused)
    loss = conf.sum()
    loss.backward()
    grads = [p.grad for p in engine.parameters() if p.grad is not None]
    assert grads, "the learned MLP must expose trainable parameters"
    assert all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)
    assert fused.grad is not None and torch.isfinite(fused.grad).all()


def test_learned_fused_dim_validated():
    engine = ConfidenceEngine("learned", fused_dim=5)
    probs = _random_probs(4, 3, seed=4)
    with pytest.raises(ValueError, match="fused"):
        engine(probs, fused=torch.randn(4, 2))


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_engine_validates_inputs():
    engine = ConfidenceEngine("entropy")
    with pytest.raises(ValueError, match="probabilities"):
        engine(torch.randn(4, 3, 5))  # wrong rank
    bad = torch.randn(4, 3)
    bad[0, 0] = float("inf")
    with pytest.raises(ValueError, match="probabilities"):
        engine(bad)
    with pytest.raises(TypeError, match="torch.bool"):
        engine(torch.rand(4, 3), option_mask=torch.ones(4, 3, dtype=torch.uint8))
    with pytest.raises(ValueError, match="same shape"):
        engine(torch.rand(4, 3), option_mask=torch.ones(4, 2, dtype=torch.bool))
