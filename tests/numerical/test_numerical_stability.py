# Copyright (c) 2026 ysnrfd
"""Section 38 — numerical stability of the head stack under extremes.

Contract under test (docs/MATHEMATICAL_SPECIFICATION.md section 6 + PROOFS.md
Remarks 1.2/3.2/4.1):

* the FULL head stack (Choice/Score/Noul heads + ConfidenceEngine, through
  the real ``check_finite`` guard) stays finite on +/-1e4 and +/-1e6 logits;
* temperature scaling of extreme logits stays finite (no inf/NaN division);
* isotonic (PAVA) and vector (LBFGS) fits on near-duplicate logits produce
  no NaN and apply cleanly;
* fp16/bf16 inputs produce finite probabilities close to the fp32 path
  (atol 1e-2: documented cost of input rounding to the low-precision dtype
  plus output downcast — JevX computes in fp32 and casts back);
* the stable log-softmax survives logits where the naive exp formula
  overflows to inf/NaN — both sides are demonstrated in one test.

All randomness is seeded (``torch.manual_seed`` / local generators); no
network access.
"""

from __future__ import annotations

import pytest
import torch

from jevx.calibration import CalibrationEngine
from jevx.confidence import ConfidenceEngine
from jevx.decisions import ChoiceHead, NoulHead, ScoreHead
from jevx.utils.numerical import (
    masked_softmax,
    stable_log_softmax,
    stable_softmax,
)
from jevx.utils.shapes import check_finite

SEED = 20260924

#: Rows covering +/-1e4 / +/-1e6 magnitudes plus flat/zero rows.
def _extreme_rows(magnitude: float) -> torch.Tensor:
    return torch.tensor(
        [
            [magnitude, -magnitude, 0.0, 5.0],
            [-magnitude, -magnitude, magnitude, -magnitude],
            [0.0, 0.0, 0.0, 0.0],
            [5.0, magnitude, -magnitude, 0.0],
            [-magnitude, 4.0, magnitude, -magnitude],
        ],
        dtype=torch.float32,
    )


@pytest.mark.parametrize("magnitude", [1e4, 1e6])
def test_full_head_stack_finite_on_extreme_logits(magnitude):
    """Choice + Score + Noul heads and every ConfidenceEngine method on
    +/-1e4 / +/-1e6 logits: all probabilities/confidences finite (verified
    with the REAL check_finite guard) and in their declared ranges."""
    torch.manual_seed(SEED + 70_000)
    logits = _extreme_rows(magnitude)  # [5, 4]

    # -- Choice: simplex membership survives saturation (exact 0/1 allowed by
    # Remark 1.2 underflow; rows still normalize) --------------------------
    choice = ChoiceHead(8, interaction="bilinear")
    choice_out = choice.build_output(logits)
    probs = choice_out.probabilities
    check_finite(probs, "choice.probabilities")
    assert torch.allclose(probs.sum(dim=-1), torch.ones(5), atol=1e-6)
    assert bool((probs >= 0.0).all())
    assert bool((choice_out.predicted_index >= 0).all())
    assert bool((choice_out.predicted_index < 4).all())

    # -- Score: expectation stays finite and inside the box ----------------
    score = ScoreHead(8, num_levels=4)
    score_out = score.build_output(logits)
    check_finite(score_out.probabilities, "score.probabilities")
    check_finite(score_out.expected_score, "score.expected_score")
    s = score_out.expected_score
    assert bool((s >= 0.0 - 1e-4).all()) and bool((s <= 3.0 + 1e-4).all())

    # -- Noul: finite and strictly interior (T4 clamp invariant) -----------
    noul = NoulHead(8)
    noul_out = noul.build_output(logits[:, :1])
    check_finite(noul_out.probabilities, "noul.probabilities")
    assert bool((noul_out.probabilities > 0.0).all())
    assert bool((noul_out.probabilities < 1.0).all())

    # -- Confidence: every method finite; [0,1] methods in [0,1] ------------
    confidences = {}
    for method in ("max_probability", "margin", "entropy"):
        engine = ConfidenceEngine(method)
        conf = engine(probs)
        check_finite(conf, f"confidence[{method}]")
        assert bool((conf >= 0.0).all()) and bool((conf <= 1.0).all())
        confidences[method] = conf
    # energy: fit on the extreme batch (spread across rows) then apply
    energy_engine = ConfidenceEngine("energy").fit(logits)
    conf = energy_engine(probs, logits=logits)
    check_finite(conf, "confidence[energy]")
    assert bool((conf >= 0.0).all()) and bool((conf <= 1.0).all())
    confidences["energy"] = conf
    # learned: sigmoid output in (0,1)
    learned = ConfidenceEngine("learned", fused_dim=8, hidden_size=16)
    conf = learned(probs, fused=torch.randn(5, 8, generator=torch.Generator().manual_seed(SEED)))
    check_finite(conf, "confidence[learned]")
    assert bool((conf > 0.0).all()) and bool((conf < 1.0).all())
    # Width-1 binary rule (spec Def 9(e), audit Q-002): conf = max(p, 1-p)
    # in [0.5, 1], finite. EDITED (bug-encoding assertion): the old version
    # pinned the hard-wired 1.0 short-circuit that disabled noul abstention.
    conf_k1 = ConfidenceEngine("margin")(noul_out.probabilities)
    check_finite(conf_k1, "confidence[k=1]")
    expected_k1 = torch.maximum(
        noul_out.probabilities[:, 0], 1.0 - noul_out.probabilities[:, 0]
    )
    assert torch.equal(conf_k1, expected_k1)
    assert bool((conf_k1 >= 0.5).all()) and bool((conf_k1 <= 1.0).all())
    print(
        f"[measure] head stack @ ±{magnitude:.0e}: all outputs finite; "
        f"score range [{float(s.min()):.4f}, {float(s.max()):.4f}] in [0, 3]"
    )


def test_temperature_scaling_of_extreme_logits_stays_finite():
    """A fitted temperature calibrator applied to +/-1e4 / +/-1e6 logits
    (and an extreme T=1e-2) yields finite logits and finite softmax output."""
    torch.manual_seed(SEED + 71_000)
    engine = CalibrationEngine("temperature")
    cal_logits = torch.randn(512, 4) * 2.0
    cal_labels = torch.randint(0, 4, (512,))
    engine.fit(cal_logits, cal_labels)
    assert engine.is_fitted
    t = float(engine.calibrator.temperature)
    assert t > 0

    for magnitude in (1e4, 1e6):
        logits = _extreme_rows(magnitude)
        scaled = engine.apply(logits)
        check_finite(scaled, f"temperature.calibrate_logits(±{magnitude:.0e})")
        probs = stable_softmax(scaled)
        check_finite(probs, f"temperature.softmax(±{magnitude:.0e})")
        assert torch.allclose(probs.sum(dim=-1), torch.ones(5), atol=1e-6)

    # extreme temperature: T = 1e-2 -> logits / T at 1e8, still finite
    engine.calibrator.log_temperature.fill_(torch.log(torch.tensor(1e-2)))
    scaled = engine.apply(_extreme_rows(1e6))
    check_finite(scaled, "temperature.calibrate_logits(T=1e-2, ±1e6)")
    probs = stable_softmax(scaled)
    check_finite(probs, "temperature.softmax(T=1e-2, ±1e6)")
    assert torch.allclose(probs.sum(dim=-1), torch.ones(5), atol=1e-6)
    print(
        f"[measure] temperature scaling: fitted T = {t:.4f}; T=1e-2 on ±1e6 "
        f"-> max |logit| = {float(scaled.abs().max()):.3e}, probs finite"
    )


@pytest.mark.parametrize("noise", [0.0, 1e-7])
def test_isotonic_and_vector_fits_on_near_duplicate_logits_no_nan(noise):
    """Isotonic (PAVA) and vector (LBFGS) fits on near-duplicate logits
    (all rows within 1e-7 of each other, or identical): no NaN anywhere,
    isotonic map nondecreasing in [0,1], applications finite and normalized."""
    torch.manual_seed(SEED + 72_000)
    n, k = 400, 4
    base = torch.tensor([[0.5, -0.3, 0.8, 1.1]])
    logits = base.expand(n, k).clone()
    if noise > 0.0:
        logits = logits + torch.randn(n, k) * noise
    labels = torch.randint(0, k, (n,))

    # isotonic
    iso = CalibrationEngine("isotonic").fit(logits, labels)
    xp, yp = iso.calibrator.x_points, iso.calibrator.y_points
    assert bool(torch.isfinite(xp).all()) and bool(torch.isfinite(yp).all()), (
        "PAVA produced non-finite breakpoints on near-duplicate logits"
    )
    if yp.numel() > 1:
        assert bool((yp[1:] - yp[:-1] >= -1e-12).all()), "isotonic map must be nondecreasing"
    assert bool((yp >= 0.0).all()) and bool((yp <= 1.0).all())
    iso_out = iso.apply(logits)
    check_finite(iso_out, "isotonic.apply(logits)")
    iso_probs = iso.apply(stable_softmax(logits), input_type="probs")
    check_finite(iso_probs, "isotonic.apply(probs)")
    assert torch.allclose(iso_probs.sum(dim=-1), torch.ones(n), atol=1e-5)
    assert bool((iso_probs >= 0.0).all()) and bool((iso_probs <= 1.0).all())

    # vector
    vec = CalibrationEngine("vector").fit(logits, labels)
    scale, bias = vec.calibrator.scale, vec.calibrator.bias
    assert bool(torch.isfinite(scale).all()) and bool(torch.isfinite(bias).all()), (
        "LBFGS produced non-finite parameters on near-duplicate logits"
    )
    vec_out = vec.apply(logits)
    check_finite(vec_out, "vector.apply(logits)")
    vec_probs = stable_softmax(vec_out)
    check_finite(vec_probs, "vector.softmax(apply(logits))")
    assert torch.allclose(vec_probs.sum(dim=-1), torch.ones(n), atol=1e-5)
    print(
        f"[measure] near-duplicate fits (noise={noise:g}): isotonic breakpoints "
        f"finite ({xp.numel()} pts), vector scale in "
        f"[{float(scale.detach().min()):.4f}, {float(scale.detach().max()):.4f}], no NaN"
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_inputs_finite_close_to_fp32_path(dtype):
    """fp16/bf16 logits -> finite probabilities within atol 1e-2 of the fp32
    path. Documented caveats: (i) storing the logits in the low-precision
    dtype already rounds them (bf16 of 1000 is 1008), (ii) the output is cast
    back to the input dtype (7-11 mantissa bits). JevX computes in fp32
    internally (upcast), so no exp overflow occurs at ±1e3."""
    x32 = torch.tensor([[8.0, -8.0, 0.0], [1000.0, -1000.0, 1.0]])
    x_low = x32.to(dtype)
    ref_probs = torch.softmax(x32, dim=-1)

    probs = stable_softmax(x_low)
    assert probs.dtype == dtype, "output must keep the input dtype"
    check_finite(probs, f"stable_softmax({dtype})")
    err = float((probs.float() - ref_probs).abs().max())
    assert err <= 1e-2, f"{dtype}: probs off the fp32 path by {err:.3e} (tol 1e-2)"

    log_probs = stable_log_softmax(x_low)
    assert log_probs.dtype == dtype
    check_finite(log_probs, f"stable_log_softmax({dtype})")
    recomputed = torch.softmax(log_probs.float(), dim=-1)
    err_log = float((recomputed - ref_probs).abs().max())
    assert err_log <= 1e-2, f"{dtype}: exp(log_softmax) off the fp32 path by {err_log:.3e}"

    masked = masked_softmax(x_low, torch.tensor([[True, True, True]]).expand(2, 3))
    check_finite(masked, f"masked_softmax({dtype})")
    assert torch.allclose(masked.float().sum(dim=-1), torch.ones(2), atol=1e-2)
    print(
        f"[measure] {dtype}: max |p_low - p_fp32| = {err:.3e}, "
        f"exp(log p) err = {err_log:.3e} (tol 1e-2)"
    )


def test_stable_log_softmax_vs_naive_exp_formula():
    """Demonstrate BOTH sides: the naive formula exp(z)/sum(exp(z))
    overflows to inf and yields inf/inf = NaN at z ~ 1000 (fp32 exp overflows
    at ~88.7), while stable_log_softmax stays finite and normalized; on
    moderate logits the two agree to 1e-6."""
    # naive overflow demonstration
    z_hot = torch.tensor([[1000.0, 999.0, 1001.0]])
    naive_exp = torch.exp(z_hot)
    assert bool(torch.isinf(naive_exp).all()), "fp32 exp(1000) must overflow to +inf"
    naive_p = naive_exp / naive_exp.sum(dim=-1, keepdim=True)
    assert bool(torch.isnan(naive_p).all()), "inf/inf must be NaN — the naive formula fails"

    # our stable path on the SAME logits
    log_p = stable_log_softmax(z_hot)
    check_finite(log_p, "stable_log_softmax(1000-scale logits)")
    p = torch.exp(log_p)
    assert torch.allclose(p.sum(dim=-1), torch.ones(1), atol=1e-6), (
        "stable log-softmax must stay normalized where the naive form is NaN"
    )
    # the mass concentrates on the largest logit
    assert int(p.argmax(dim=-1)) == 2

    # agreement regime: naive formula is fine for moderate logits
    z_cold = torch.tensor([[1.0, 2.0, 3.0]])
    naive_cold = torch.exp(z_cold) / torch.exp(z_cold).sum(dim=-1, keepdim=True)
    stable_p = torch.exp(stable_log_softmax(z_cold))
    assert torch.allclose(naive_cold, stable_p, atol=1e-6), (
        "where the naive formula is computable it must agree with ours"
    )
    print(
        "[measure] log-softmax vs naive: exp(1000) = inf -> naive p = NaN; "
        f"stable p = {[round(float(v), 6) for v in p.flatten()]}, finite & sums to 1"
    )
