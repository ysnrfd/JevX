# Copyright (c) 2026 ysnrfd
"""Unit tests for the calibration engine: temperature/vector/isotonic fitting,
hand-computed metrics, schema invariance and serialization round trips."""

from __future__ import annotations

import math

import pytest
import torch

from jevx.calibration import (
    CalibrationEngine,
    available_calibrators,
    pava_isotonic,
)
from jevx.utils.numerical import stable_softmax


def _overconfident_batch(seed: int = 0, n: int = 4000, k: int = 4, t_true: float = 2.5):
    """logits = raw * T_true + noise; labels sampled from softmax(raw).

    The population-optimal temperature under the `logits / T` convention is
    exactly T_true (the observed logits are over-scaled by T_true), and the
    stochastic labels keep the MLE finite (deterministic argmax labels would
    degenerate to T -> 0).
    """
    g = torch.Generator().manual_seed(seed)
    raw = torch.randn(n, k, dtype=torch.float64, generator=g)
    noise = torch.randn(n, k, dtype=torch.float64, generator=g)
    logits = (raw * t_true + 0.05 * noise).float()
    # Inverse-CDF sampling from softmax(raw) with the local generator
    # (torch.distributions' .sample() does not accept a generator kwarg).
    cdf = torch.softmax(raw, dim=-1).cumsum(dim=-1)
    u = torch.rand(n, dtype=torch.float64, generator=g)
    labels = torch.searchsorted(cdf, u.unsqueeze(-1), right=False).squeeze(-1)
    labels = labels.clamp_max(k - 1)  # guard fp cumsum drift at the last class
    return logits, labels.long()


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------


def test_temperature_recovers_ground_truth_scale():
    t_true = 2.5
    logits, labels = _overconfident_batch(t_true=t_true)
    engine = CalibrationEngine("temperature").fit(logits, labels)
    fitted = float(engine.calibrator.temperature)
    assert fitted > 0
    assert fitted == pytest.approx(t_true, rel=0.15), (
        f"fitted T={fitted} should be within tolerance of T_true={t_true}"
    )


def test_temperature_reduces_nll():
    logits, labels = _overconfident_batch()
    before = CalibrationEngine("none").evaluate(logits, labels)
    after = CalibrationEngine("temperature").fit(logits, labels).evaluate(logits, labels)
    assert after["nll"] <= before["nll"]
    assert after["nll"] < before["nll"]  # strictly better on this dataset


def test_temperature_preserves_argmax():
    torch.manual_seed(0)
    logits, labels = _overconfident_batch(n=500)
    engine = CalibrationEngine("temperature").fit(logits, labels)
    calibrated = engine.apply(logits)
    assert torch.equal(calibrated.argmax(-1), logits.argmax(-1))
    # also for an extreme temperature
    engine.calibrator.log_temperature.fill_(math.log(0.05))
    calibrated = engine.apply(logits)
    assert torch.equal(calibrated.argmax(-1), logits.argmax(-1))


def test_temperature_grid_fit_is_deterministic():
    logits, labels = _overconfident_batch(n=800)
    t1 = float(CalibrationEngine("temperature").fit(logits, labels).calibrator.temperature)
    t2 = float(CalibrationEngine("temperature").fit(logits, labels).calibrator.temperature)
    assert t1 == t2  # bit-identical: no RNG, no optimizer state


# ---------------------------------------------------------------------------
# Vector / isotonic shapes
# ---------------------------------------------------------------------------


def test_vector_fit_and_apply_shapes():
    torch.manual_seed(0)
    logits, labels = _overconfident_batch(n=600, k=5)
    engine = CalibrationEngine("vector").fit(logits, labels)
    scale, bias = engine.calibrator.scale, engine.calibrator.bias
    assert scale.shape == (5,) and bias.shape == (5,)
    out = engine.apply(logits)
    assert out.shape == logits.shape
    assert out.dtype == logits.dtype
    probs = stable_softmax(logits)
    out_p = engine.apply(probs, input_type="probs")
    assert out_p.shape == probs.shape
    assert torch.allclose(out_p.sum(-1), torch.ones(600), atol=1e-5)


def test_vector_binary_single_logit():
    torch.manual_seed(0)
    logits = torch.randn(400, 1)
    labels = torch.distributions.Categorical(
        logits=torch.cat([-logits, logits], dim=-1)
    ).sample()
    engine = CalibrationEngine("vector").fit(logits, labels)
    assert engine.calibrator.scale.shape == (1,)
    assert engine.calibrator.bias.shape == (1,)
    out = engine.apply(logits)
    assert out.shape == (400, 1)
    metrics = engine.evaluate(logits, labels)
    assert metrics["nll"] <= CalibrationEngine("none").evaluate(logits, labels)["nll"]


def test_isotonic_fit_and_apply_shapes():
    torch.manual_seed(0)
    logits, labels = _overconfident_batch(n=500, k=3)
    engine = CalibrationEngine("isotonic").fit(logits, labels)
    assert engine.calibrator.x_points.numel() >= 1
    assert engine.calibrator.x_points.shape == engine.calibrator.y_points.shape

    out_logits = engine.apply(logits)
    assert out_logits.shape == logits.shape
    assert out_logits.dtype == logits.dtype

    probs = stable_softmax(logits)
    out_probs = engine.apply(probs, input_type="probs")
    assert out_probs.shape == probs.shape
    assert (out_probs >= 0).all()
    assert torch.allclose(out_probs.sum(-1), torch.ones(500), atol=1e-5)
    # logits and probs paths agree up to output rounding
    back = stable_softmax(out_logits)
    assert torch.allclose(back, out_probs, atol=1e-3)


def test_isotonic_requires_fit():
    engine = CalibrationEngine("isotonic")
    with pytest.raises(RuntimeError, match="fit"):
        engine.apply(torch.randn(4, 3))


def test_isotonic_monotone_map_and_ties():
    calibrator = CalibrationEngine("isotonic").calibrator
    # hand-made calibration set: 3 distinct confidences with tied groups
    logits = torch.tensor(
        [
            [2.0, 0.0, 0.0],  # conf .89 correct
            [2.0, 0.0, 0.0],  # conf .89 wrong
            [0.4, 0.0, 0.0],  # conf .59 wrong-ish
            [0.4, 0.0, 0.0],  # conf .59 correct
            [0.0, 0.0, 0.0],  # conf 1/3 correct
            [0.0, 0.0, 0.0],  # conf 1/3 wrong
        ]
    )
    labels = torch.tensor([0, 1, 1, 0, 0, 1])
    calibrator.fit(logits, labels)
    # ties: one breakpoint per distinct confidence
    assert calibrator.x_points.numel() == 3
    # fitted values are means of the tied groups, nondecreasing
    y = calibrator.y_points
    assert y[0] <= y[1] <= y[2]
    probs = stable_softmax(logits)
    calibrated = calibrator.calibrate_probs(probs)
    assert torch.allclose(calibrated.sum(-1), torch.ones(6), atol=1e-5)


def test_isotonic_all_tied_confidences():
    logits = torch.zeros(20, 3)
    labels = torch.randint(0, 3, (20,), generator=torch.Generator().manual_seed(3))
    engine = CalibrationEngine("isotonic").fit(logits, labels)
    assert engine.calibrator.x_points.numel() == 1  # single tie group
    out = engine.apply(logits)
    assert out.shape == (20, 3)
    probs = stable_softmax(logits)
    out_p = engine.apply(probs, input_type="probs")
    assert torch.allclose(out_p.sum(-1), torch.ones(20), atol=1e-5)


def test_pava_isotonic_known_values():
    x = torch.tensor([0.1, 0.2, 0.3, 0.4])
    _, y1 = pava_isotonic(x, torch.tensor([0.9, 0.1, 0.8, 0.2]))
    assert torch.allclose(y1, torch.full((4,), 0.5, dtype=torch.float64), atol=1e-12)
    _, y2 = pava_isotonic(x, torch.tensor([0.1, 0.8, 0.7, 0.9]))
    assert torch.allclose(
        y2, torch.tensor([0.1, 0.75, 0.75, 0.9], dtype=torch.float64), atol=1e-12
    )
    # weighted pooling
    _, y3 = pava_isotonic(
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([1.0, 0.0, 1.0]),
        weights=torch.tensor([1.0, 10.0, 1.0]),
    )
    assert torch.allclose(
        y3,
        torch.tensor([(1.0 * 1.0 + 10.0 * 0.0) / 11.0] * 2 + [1.0], dtype=torch.float64),
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# Metrics against hand computation
# ---------------------------------------------------------------------------


def test_ece_mce_nll_brier_hand_computed():
    probs = torch.tensor(
        [
            [0.4, 0.35, 0.25],   # conf 0.4, correct
            [0.4, 0.35, 0.25],   # conf 0.4, wrong
            [0.9, 0.05, 0.05],   # conf 0.9, correct
            [0.9, 0.05, 0.05],   # conf 0.9, wrong
        ],
        dtype=torch.float64,
    )
    labels = torch.tensor([0, 1, 0, 2])
    logits = probs.log()  # softmax(log p) == p

    metrics = CalibrationEngine("none").evaluate(logits, labels, n_bins=2)

    # Hand ECE with 2 equal-width bins: bin0 holds conf {0.4, 0.4}
    # (acc 1/2), bin1 holds {0.9, 0.9} (acc 1/2):
    #   ECE = (2/4)*|0.5 - 0.4| + (2/4)*|0.5 - 0.9| = 0.05 + 0.20 = 0.25
    #   MCE = max(0.1, 0.4) = 0.4
    assert metrics["ece"] == pytest.approx(0.25, abs=1e-9)
    assert metrics["mce"] == pytest.approx(0.40, abs=1e-9)

    # NLL picks probability AT THE TRUE LABEL:
    # rows give p[y] = 0.4, 0.35, 0.9, 0.05
    hand_nll = -(math.log(0.4) + math.log(0.35) + math.log(0.9) + math.log(0.05)) / 4
    assert metrics["nll"] == pytest.approx(hand_nll, abs=1e-9)

    s1 = (0.4 - 1) ** 2 + 0.35**2 + 0.25**2
    s2 = 0.4**2 + (0.35 - 1) ** 2 + 0.25**2
    s3 = (0.9 - 1) ** 2 + 0.05**2 + 0.05**2
    s4 = 0.9**2 + 0.05**2 + (0.05 - 1) ** 2
    assert metrics["brier"] == pytest.approx((s1 + s2 + s3 + s4) / 4, abs=1e-9)

    # same metrics through the class API with hand-given probs
    from jevx.utils.numerical import (
        brier_score,
        expected_calibration_error,
        maximum_calibration_error,
        negative_log_likelihood,
    )

    assert expected_calibration_error(probs, labels, n_bins=2) == pytest.approx(0.25, abs=1e-9)
    assert maximum_calibration_error(probs, labels, n_bins=2) == pytest.approx(0.4, abs=1e-9)
    assert negative_log_likelihood(probs, labels) == pytest.approx(hand_nll, abs=1e-9)
    assert brier_score(probs, labels) == pytest.approx((s1 + s2 + s3 + s4) / 4, abs=1e-9)


def test_ece_default_bins_and_perfect_calibration():
    # A perfectly calibrated set: all confidences equal accuracy by
    # construction — half correct at conf 0.5 in the same bin.
    probs = torch.full((100, 2), 0.5)
    labels = torch.zeros(100, dtype=torch.long)
    labels[50:] = 1  # argmax is 0; exactly half "correct"... argmax of [0.5,0.5] = 0
    metrics = CalibrationEngine("none").evaluate(
        torch.zeros(100, 2), labels, n_bins=15
    )
    # confidence of [0.5, 0.5] is 0.5, accuracy is 0.5 -> ECE 0
    assert metrics["ece"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["mce"] == pytest.approx(0.0, abs=1e-9)


def test_ece_confidence_one_lands_in_last_bin():
    # conf == 1.0 must clamp into the final bin rather than index past it
    probs = torch.tensor([[1.0, 0.0]] * 10, dtype=torch.float64)
    labels = torch.tensor([0] * 5 + [1] * 5)
    from jevx.utils.numerical import expected_calibration_error

    ece = expected_calibration_error(probs, labels, n_bins=15)
    assert ece == pytest.approx(0.5, abs=1e-9)


# ---------------------------------------------------------------------------
# Schema invariance: shape / dtype / class count never change
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["none", "temperature", "vector", "isotonic"])
def test_calibration_preserves_schema(method):
    torch.manual_seed(0)
    logits, labels = _overconfident_batch(n=300, k=4)
    engine = CalibrationEngine(method).fit(logits, labels)

    out = engine.apply(logits)
    assert out.shape == logits.shape, "calibration must not change shape"
    assert out.dtype == logits.dtype, "calibration must not change dtype"
    assert out.shape[1] == 4, "calibration must not change the number of classes"
    assert torch.isfinite(out).all()

    probs = stable_softmax(logits)
    out_p = engine.apply(probs, input_type="probs")
    assert out_p.shape == probs.shape
    assert out_p.dtype == probs.dtype
    assert out_p.shape[1] == 4


@pytest.mark.parametrize("method", ["none", "temperature", "vector", "isotonic"])
def test_calibration_round_trips_exact(method, tmp_path):
    torch.manual_seed(0)
    logits, labels = _overconfident_batch(n=300, k=4)
    engine = CalibrationEngine(method).fit(logits, labels)

    path = tmp_path / f"{method}.pt"
    engine.save(path)
    restored = CalibrationEngine.load(path)

    assert restored.method == method
    assert restored.is_fitted == engine.is_fitted
    assert torch.equal(engine.apply(logits), restored.apply(logits))
    assert torch.equal(
        engine.apply(stable_softmax(logits), "probs"),
        restored.apply(stable_softmax(logits), "probs"),
    )
    # state dicts themselves round trip
    for key, value in engine.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])


# ---------------------------------------------------------------------------
# Validation / registry / facade
# ---------------------------------------------------------------------------


def test_identity_calibrator_returns_input_untouched():
    torch.manual_seed(0)
    logits = torch.randn(10, 3)
    engine = CalibrationEngine("none")
    engine.fit(logits, torch.randint(0, 3, (10,)))
    assert torch.equal(engine.apply(logits), logits)


def test_fit_input_validation():
    logits = torch.randn(10, 3)
    with pytest.raises(ValueError, match="must lie in"):
        CalibrationEngine("temperature").fit(logits, torch.full((10,), 7))
    with pytest.raises(ValueError, match="one label per sample"):
        CalibrationEngine("temperature").fit(logits, torch.zeros(4, dtype=torch.long))
    with pytest.raises(ValueError, match=r"\[N, K\]"):
        CalibrationEngine("temperature").fit(torch.randn(3, 10, 3), torch.zeros(3, dtype=torch.long))
    with pytest.raises(ValueError, match=r"\{0, 1\}"):
        CalibrationEngine("temperature").fit(
            torch.randn(5, 1), torch.full((5,), 2, dtype=torch.long)
        )


def test_unknown_method_informative_error():
    with pytest.raises(ValueError, match="unknown calibration method"):
        CalibrationEngine("bogus-method")
    with pytest.raises(ValueError, match="temperature"):
        CalibrationEngine("bogus-method")


def test_registry_contents():
    assert set(available_calibrators()) == {"none", "temperature", "vector", "isotonic"}


def test_engine_from_config():
    from jevx.configuration_jevx import JevXConfig

    for method in ("none", "temperature", "vector", "isotonic"):
        cfg = JevXConfig(calibration_method=method)
        engine = CalibrationEngine.from_config(cfg)
        assert engine.method == method


def test_evaluate_keys_and_types():
    torch.manual_seed(0)
    logits, labels = _overconfident_batch(n=200, k=3)
    metrics = CalibrationEngine("none").evaluate(logits, labels)
    assert set(metrics) == {"nll", "ece", "mce", "brier"}
    for key, value in metrics.items():
        assert isinstance(value, float), key
        assert math.isfinite(value), key


def test_unknown_input_type():
    engine = CalibrationEngine("none")
    with pytest.raises(ValueError, match="input_type"):
        engine.apply(torch.randn(2, 2), input_type="raw")  # type: ignore[arg-type]
