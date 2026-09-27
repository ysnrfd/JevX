# Copyright (c) 2026 ysnrfd
"""Executable numerical verification of PROOFS.md Theorems T1-T8 (Phase 24 + §27 + §68).

Each test numerically instantiates the *statement* of one theorem against the
REAL implementation (``jevx.decisions`` heads, ``jevx.representations``
pooling/fusion, ``jevx.calibration``, ``ConfidenceEngine``,
``JevXForDecision``) — plus direct formula checks where the theorem is about a
primitive. The paper proofs stay in ``docs/PROOFS.md``; this file only checks
that the implementation SATISFIES each theorem's statement. It never
re-derives the mathematics.

Seed policy: ``torch.manual_seed(SEED + trial)`` per property trial so any
failure is reproducible; 100 trials per property check unless noted. No
network access.

Tolerance policy — a tolerance may reflect ONLY the theorem's own stated
assumptions/caveats (fp underflow, fp non-associativity); each widening is
documented at its assertion and in ``reports/MATHEMATICAL_VERIFICATION.md``:

* fp32 underflow (PROOFS Remark 1.2): ``exp`` gaps ≳ 104 underflow to exact
  0, so exact-0 probabilities are asserted as the *documented caveat* where
  the gap is huge, and strict positivity where the gap is within fp range.
  Masked entries are *intentionally* exact 0 via finfo-min (masked_softmax).
* fp non-associativity / kernel reduction order (Remarks 3.2, 7.2): batched
  vs looped results are compared with a MEASURED tolerance (T7(ii) is an
  [EMPIRICAL] claim template); bit-equality is never asserted across shapes.
"""

from __future__ import annotations

import copy

import pytest
import torch

from jevx import (
    ConfidenceEngine,
    JevXConfig,
    JevXForDecision,
    coverage_at_risk,
    risk_coverage_curve,
)
from jevx.calibration import CalibrationEngine
from jevx.decisions import (
    ChoiceHead,
    NoulHead,
    ScoreHead,
    mask_from_num_options,
    project_to_schema,
)
from jevx.representations import available_fusions, build_fusion, pool
from jevx.training.losses import choice_loss
from jevx.utils.numerical import brier_score, stable_softmax

SEED = 20260924
TRIALS = 100
HIDDEN = 32  # matches the tiny_bert fixture's hidden_size


def _build_wrapper(backbone, tokenizer, **overrides) -> JevXForDecision:
    """Deterministically wrap a deep-copied backbone in eval mode."""
    torch.manual_seed(SEED)  # wrapper-side parameter draws are reproducible
    config_kwargs = dict(
        decision_hidden_size=HIDDEN,
        question_hidden_size=HIDDEN,
        schema_hidden_size=HIDDEN,
    )
    config_kwargs.update(overrides)
    wrapper = JevXForDecision(
        JevXConfig(**config_kwargs),
        backbone=copy.deepcopy(backbone),
        tokenizer=tokenizer,
    )
    wrapper.eval()
    return wrapper


# ===========================================================================
# T1 — Softmax validity (PROOFS.md T1, Remarks 1.1/1.2)
# ===========================================================================


def test_t1_softmax_random_validity_and_reference_match():
    """T1(i)(ii) on 100 random cases: strict p_i > 0, rows sum to 1 within
    1e-6, and stable_softmax is bit-identical to the torch fp32 reference."""
    max_sum_err = 0.0
    for trial in range(TRIALS):
        torch.manual_seed(SEED + trial)
        k = int(torch.randint(1, 17, (1,)))
        kind = trial % 4
        if kind == 0:
            logits = torch.rand(k) * 90.0 - 45.0  # gaps <= 90 < 104: no underflow
        elif kind == 1:
            logits = torch.randn(k) * 0.5
        elif kind == 2:
            logits = torch.randn(k) * 10.0  # typical gaps ~40 < 104
        else:
            logits = torch.randn(k) * 1e-3
        p = stable_softmax(logits)
        ref = torch.softmax(logits, dim=-1)
        assert torch.equal(p, ref), (
            f"trial {trial}: stable_softmax must match the torch fp32 softmax reference"
        )
        assert (p > 0.0).all(), f"trial {trial}: T1(i) p_i > 0 violated: {p}"
        err = float((p.sum() - 1.0).abs())
        max_sum_err = max(max_sum_err, err)
        assert err <= 1e-6, f"trial {trial}: T1(ii) sum != 1: err={err}"
        # masked variant: rows still normalize, masked entries exact 0 (finfo-min)
        if k >= 2:
            from jevx.utils.numerical import masked_softmax

            mask = torch.ones(k, dtype=torch.bool)
            mask[k - 1] = False
            pm = masked_softmax(logits.unsqueeze(0), mask.unsqueeze(0))
            assert float(pm[0, : k - 1].sum()) == pytest.approx(1.0, abs=1e-6)
            assert pm[0, k - 1] == 0.0  # documented: masked entries are exact 0
    print(f"[measure] T1 random: max |sum p - 1| = {max_sum_err:.3e} over {TRIALS} trials")


def test_t1_adversarial_magnitude_1e6_validity_and_reference():
    """Adversarial logits at magnitude ±1e6 with moderate spread: validity,
    strict positivity (gaps stay < 104), reference match, per shift c=±1e6."""
    max_sum_err = 0.0
    # 25 seeded draws x 4 center magnitudes = 100 adversarial cases (>=100 rule)
    for trial in range(25):
        torch.manual_seed(SEED + 500 + trial)
        base = torch.randn(8) * (5.0 if trial % 2 else 0.05)
        for center in (1e6, -1e6, 1e5, -1e5):
            z = base + center  # fp32 rounding of the sum: ulp(1e6)=0.0625 (Remark 1.2)
            p = stable_softmax(z)
            ref = torch.softmax(z, dim=-1)
            assert torch.equal(p, ref)
            assert (p > 0.0).all(), f"trial {trial} center {center}: p not strictly positive"
            err = float((p.sum() - 1.0).abs())
            max_sum_err = max(max_sum_err, err)
            assert err <= 1e-6, f"trial {trial} center {center}: sum err {err}"
    print(f"[measure] T1 adversarial ±1e6: max |sum p - 1| = {max_sum_err:.3e}")


def test_t1_extreme_spread_documents_fp_underflow_caveat():
    """Extreme spread (±1e6 gaps): rows still sum to 1 and match the fp32
    reference bit-for-bit; entries with gap ≲ 104 are strictly positive;
    entries with gap >> 104 are EXACTLY 0 — the documented fp-underflow caveat
    of PROOFS Remark 1.2 (RM value > 0; FP value = 0)."""
    z = torch.tensor(
        [
            [0.0, -30.0, -60.0, -90.0],  # gaps <= 90 -> strict positivity holds
            [1e6, -1e6, 0.0, 5.0],  # gaps ~1e6 -> exact 0 (underflow caveat)
            [1e6, 1e6 - 80.0, -1e6, 5.0],  # gap 80 -> >0; gap 2e6 -> 0
        ]
    )
    p = stable_softmax(z)
    ref = torch.softmax(z, dim=-1)
    assert torch.equal(p, ref), "implementation must equal the fp32 reference"
    assert torch.allclose(p.sum(dim=-1), torch.ones(3), atol=1e-6), "T1(ii)"
    # within fp range: strict positivity (RM claim visible at fp32)
    assert bool((p[0] > 0.0).all()), f"gaps <= 90 must stay positive: {p[0]}"
    assert p[2, 1] > 0.0, "gap 80 must stay positive in fp32"
    # far beyond fp range: documented exact-0 underflow (Remark 1.2)
    assert p[1, 1] == 0.0 and p[1, 2] == 0.0 and p[1, 3] == 0.0, (
        "gap >> 104 underflows to exact 0 — this IS the documented caveat"
    )
    assert p[2, 0] == 1.0 - float(p[2, 1]), "mass concentrates on the max"
    print(
        "[measure] T1 extreme spread: row sums = "
        + ", ".join(f"{float(s):.12f}" for s in p.sum(dim=-1))
    )


def test_t1_shift_invariance():
    """Remark 1.1: softmax(z + c) == softmax(z).

    (a) c in {±1e3, ±1e6} with half-integer z: z + c is EXACTLY representable
    in fp32, so the RM identity is observable — assert within 1e-6 (expected
    bit-equal after max-subtraction).
    (b) arbitrary random z shifted by ±1e6: z + c ROUNDS (ulp(1e6) = 0.0625,
    half-ulp 0.03125 per component); PROOFS Remark 1.2 documents that the fp
    identity is not bit-exact — tolerance 0.07 = e^(2·0.03125) − 1, derived
    from the stated rounding only.
    """
    gen = torch.Generator().manual_seed(SEED + 600)
    z_exact = (torch.randint(-128, 129, (32, 6), generator=gen).float()) * 0.5
    max_err_exact = 0.0
    for c in (1e6, -1e6, 1e3, -1e3):
        shifted = z_exact + torch.tensor(c)  # exact for half-integers |z| <= 1000
        err = float((stable_softmax(z_exact) - stable_softmax(shifted)).abs().max())
        max_err_exact = max(max_err_exact, err)
        assert err <= 1.6e-6, f"c={c}: shift invariance broken at fp32, err={err}"
    gen2 = torch.Generator().manual_seed(SEED + 601)
    z_rand = torch.randn(64, 8, generator=gen2) * 20.0
    max_err_rounded = 0.0
    for c in (1e6, -1e6):
        err = float(
            (stable_softmax(z_rand) - stable_softmax(z_rand + torch.tensor(c))).abs().max()
        )
        max_err_rounded = max(max_err_rounded, err)
        # fp rounding of z+c (Remark 1.2): bound e^(2 * 0.5*ulp(1e6)) - 1 = 0.0645
        assert err <= 0.07, f"c={c}: shift error {err} exceeds the documented fp bound"
    print(
        f"[measure] T1 shift invariance: exact-shift max err = {max_err_exact:.3e}, "
        f"rounded ±1e6-shift max err = {max_err_rounded:.3e} (tol 0.07, Remark 1.2)"
    )


# ===========================================================================
# T2 — Choice type safety (PROOFS.md T2)
# ===========================================================================


def _random_keys(trial: int, k: int) -> list:
    """Schema keys: strings / duplicates / negative ints / floats (D3.2
    uniqueness is validated upstream — projection must only guarantee
    membership of whatever list it is handed)."""
    style = trial % 4
    if style == 0:
        return [f"trial{trial}_opt{j}" for j in range(k)]
    if style == 1:
        pool_keys = ["dup", "dup", -7, -7, 3.5, "dup", 0]
        return [pool_keys[j % len(pool_keys)] for j in range(k)]  # duplicates present
    if style == 2:
        return [-(j + 1) for j in range(k)]  # negative integer keys
    return [float(j) + 0.25 for j in range(k)]  # float keys


def test_t2_argmax_in_range_and_schema_membership_100_schemas():
    """T2(a)(b): 100 random schemas (K 1..12, keys incl. duplicates/negative/
    float): predicted_index = argmax always in [0, K) and project_to_schema
    returns an element of the keys (membership)."""
    for trial in range(TRIALS):
        torch.manual_seed(SEED + 10_000 + trial)
        k = int(torch.randint(1, 13, (1,)))
        d = int(torch.randint(4, 17, (1,)))
        b = 3
        head = ChoiceHead(d, interaction=("dot", "bilinear", "mlp")[trial % 3])
        fused = torch.randn(b, d)
        options = torch.randn(b, k, d)
        out = head(fused, options)
        idx = out.predicted_index
        assert bool((idx >= 0).all()) and bool((idx < k).all()), (
            f"trial {trial}: T2(a) argmax escaped [0, {k}): {idx}"
        )
        keys = _random_keys(trial, k)
        projected = project_to_schema(idx, keys)  # T2(b)/(c): the Project step
        assert len(projected) == b
        for chosen in projected:
            assert chosen in keys, f"trial {trial}: {chosen!r} not in schema C={keys}"


def test_t2_masked_argmax_never_selects_invalid_slot_100_trials():
    """T2: masked coordinates are never selected even when every invalid raw
    logit is LARGER than every valid one (adversarial construction)."""
    max_sum_err = 0.0
    for trial in range(TRIALS):
        torch.manual_seed(SEED + 20_000 + trial)
        k_max = int(torch.randint(2, 13, (1,)))
        b, d = 3, 8
        counts = torch.randint(1, k_max, (b,))  # always >=1 invalid slot per row
        mask = mask_from_num_options(counts, k_max)
        head = ChoiceHead(d, interaction=("dot", "bilinear", "mlp")[trial % 3])
        logits = torch.randn(b, k_max) * 4.0
        valid_max = logits.masked_fill(~mask, float("-inf")).max(dim=-1).values
        boost = (valid_max + 5.0 + (trial % 5)).unsqueeze(-1)
        if trial % 10 == 0:  # occasional extreme: invalid logit at +1e6
            boost = (valid_max + 1e6).unsqueeze(-1)
        logits = torch.where(mask, logits, boost.expand_as(logits))
        # construction sanity: every invalid logit strictly exceeds its row's valid max
        invalid_min = logits.masked_fill(mask, float("inf")).min(dim=-1).values
        assert bool((invalid_min > valid_max).all()), "adversarial construction broken"

        out = head.build_output(logits, mask)
        rows = torch.arange(b)
        assert bool(mask[rows, out.predicted_index].all()), (
            f"trial {trial}: T2 argmax selected an invalid slot: {out.predicted_index}"
        )
        assert bool((out.predicted_index >= 0).all()) and bool(
            (out.predicted_index < k_max).all()
        )
        assert bool((out.probabilities[~mask] == 0.0).all()), "masked probs must be exact 0"
        err = float(
            (out.probabilities.sum(dim=-1) - torch.ones(b)).abs().max()
        )
        max_sum_err = max(max_sum_err, err)
        assert err <= 1e-6, f"trial {trial}: valid probabilities must sum to 1, err={err}"
    print(
        f"[measure] T2 masked argmax: 100 trials, max |sum p - 1| = {max_sum_err:.3e}, "
        "0 invalid selections"
    )


# ===========================================================================
# T3 — Score boundedness (PROOFS.md T3, Remark 3.2)
# ===========================================================================


def test_t3_expected_score_in_level_box_100_trials():
    """T3: expected_score = sum_i l_i p_i lies in [min I, max I] = [0, K-1]
    over 100 random cases including extreme logits (fused/options scaled to
    1e6). Tolerance 1e-4 — the head's own documented fp guard (Remark 3.2:
    the computed sum can leave the box by O(gamma_K); the implementation
    asserts, never clamps)."""
    max_violation = 0.0
    for trial in range(TRIALS):
        torch.manual_seed(SEED + 30_000 + trial)
        k = int(torch.randint(1, 13, (1,)))
        d = int(torch.randint(4, 17, (1,)))
        scale = 10.0 ** float(torch.randint(-1, 7, (1,)))  # 0.1 .. 1e6
        if trial % 2 == 0:
            head = ScoreHead(d, num_levels=k)
            out = head(torch.randn(4, d) * scale)  # fixed-level schema
        else:
            head = ScoreHead(d, interaction=("dot", "bilinear")[trial % 2])
            out = head(torch.randn(4, d) * scale, torch.randn(4, k, d) * scale)
        s = out.expected_score.detach()  # detach: avoid requires_grad->scalar warning
        assert torch.isfinite(s).all(), f"trial {trial}: non-finite score"
        lower_viol = max(0.0, float(-s.min()))
        upper_viol = max(0.0, float(s.max() - (k - 1)))
        violation = max(lower_viol, upper_viol)
        max_violation = max(max_violation, violation)
        assert violation <= 1e-4, (
            f"trial {trial}: T3 box [0, {k - 1}] escaped by {violation} (s in "
            f"[{float(s.min())}, {float(s.max())}])"
        )
        # probabilities themselves remain a simplex point (T1 via T3 premise)
        assert torch.allclose(
            out.probabilities.sum(dim=-1), torch.ones(4), atol=1e-6
        ), f"trial {trial}: level distribution off the simplex"
    # crafted extreme logits straight through the real build_output path
    for mag in (1e4, 1e6):
        head = ScoreHead(4, num_levels=4)
        for row in ([mag, -mag, 0.0, 5.0], [-mag, -mag, mag, -mag], [0.0] * 4):
            out = head.build_output(torch.tensor([row], dtype=torch.float32))
            s = float(out.expected_score.detach())
            assert 0.0 - 1e-4 <= s <= 3.0 + 1e-4, f"crafted row {row}: s={s}"
            max_violation = max(max_violation, max(0.0, -s, s - 3.0))
    print(
        f"[measure] T3: max box violation = {max_violation:.3e} over {TRIALS} trials "
        "+ crafted ±1e4/±1e6 rows (guard tol 1e-4)"
    )


def test_t3_uniform_probs_give_exact_center():
    """T3 with uniform p: expected score is exactly the center (K-1)/2
    within 1e-6 (fp32: 1/K is inexact for most K — documented)."""
    max_err = 0.0
    for k in range(1, 13):
        head = ScoreHead(8, num_levels=k)
        out = head.build_output(torch.zeros(2, k))  # equal logits -> uniform p
        assert torch.allclose(
            out.probabilities, torch.full((2, k), 1.0 / k), atol=1e-6
        ), f"K={k}: equal logits must give uniform probabilities"
        target = (k - 1) / 2.0
        err = float((out.expected_score - torch.tensor(target)).abs().max())
        max_err = max(max_err, err)
        assert err <= 1e-6, f"K={k}: center {target} missed by {err}"
    print(f"[measure] T3 uniform: max |s - (K-1)/2| = {max_err:.3e} for K=1..12")


# ===========================================================================
# T4 — Binary validity (PROOFS.md T4, Remark 4.1)
# ===========================================================================


def test_t4_noul_strict_interior_for_finite_extreme_logits():
    """T4: 0 < sigma(z) < 1 for every finite z — checked at z in
    {0, ±1e3, ±1e4, ±1e6}, i.e. including the fp32 saturation regime.

    PROOFS Remark 4.1: raw fp32 sigmoid saturates to exact 0/1 for these
    magnitudes, so NoulHead must apply the sanctioned [eps, 1-eps] clamp
    (implementation invariant) — stored probabilities must never be exactly
    0 or 1 for a finite fp32-representable logit."""
    head = NoulHead(8)
    zs = [0.0, 1e3, -1e3, 1e4, -1e4, 1e6, -1e6]
    logits = torch.tensor([[z] for z in zs], dtype=torch.float32)
    out = head.build_output(logits)
    p = out.probabilities
    assert torch.isfinite(p).all(), "T4 premise: finite outputs"
    assert bool((p > 0.0).all()), (
        "T4 violated: stored p is exactly 0 for finite z — "
        + str({z: float(v) for z, v in zip(zs, p.flatten())})
    )
    assert bool((p < 1.0).all()), (
        "T4 violated: stored p is exactly 1 for finite z — "
        + str({z: float(v) for z, v in zip(zs, p.flatten())})
    )
    assert set(out.predicted_index.tolist()) <= {0, 1}
    # direction sanity: positive logits -> p >= 0.5 >= negative logits
    assert float(p[1, 0]) >= 0.5 >= float(p[2, 0])
    assert float(p[3, 0]) >= float(p[1, 0])  # monotone in z (T4 step 2)
    # also through the real forward path with extreme fused inputs
    torch.manual_seed(SEED + 40_000)
    fwd = head(torch.randn(16, 8) * 1e4)
    assert bool((fwd.probabilities > 0.0).all()) and bool(
        (fwd.probabilities < 1.0).all()
    ), "forward path must also stay strictly interior"
    print(
        "[measure] T4 noul: p(0)=%.9f p(1e3)=%.9f p(1e6)=%.9f "
        "p(-1e3)=%.3e p(-1e6)=%.3e (strictly interior)"
        % (float(p[0, 0]), float(p[1, 0]), float(p[5, 0]), float(p[2, 0]), float(p[6, 0]))
    )


# ===========================================================================
# T5 — Log score is strictly proper (PROOFS.md T5)
# ===========================================================================


def test_t5_log_score_strictly_proper_monte_carlo():
    """T5 via Monte Carlo (n = 200_000, seeded), losses through the REAL
    ``choice_loss``:

    * E_p[-log q] is STRICTLY smaller for q = p than for every q != p;
    * E_p[-log q] - H(p) ~= analytic KL(p||q) >= 0 within MC error;
    * E_p[-log p] ~= H(p) (KL(p||p) = 0).
    """
    gen = torch.Generator().manual_seed(SEED + 50_000)
    k = 5
    base = torch.rand(k, generator=gen) ** 2 + 0.05  # skewed random simplex point
    p = (base / base.sum()).double()
    n = 200_000

    # sample y ~ p by inverse-CDF (seeded)
    cdf = p.cumsum(dim=0)
    u = torch.rand(n, dtype=torch.float64, generator=gen)
    y = torch.searchsorted(cdf, u, right=False).clamp(max=k - 1)

    # candidate forecasts: p itself, two perturbations, the uniform q
    shift = torch.randn(k, dtype=torch.float64, generator=gen) * 0.9
    q_shift = (torch.log(p) + shift).softmax(dim=0)
    # 50/50 mixture: KL(p||mix) ~= 0.09 for this seed — comfortably above the
    # 0.05 separation floor asserted below (MC SE of the loss gap is ~0.004).
    q_mix = 0.5 * p + 0.5 / k
    q_uniform = torch.full((k,), 1.0 / k, dtype=torch.float64)
    candidates = {"p": p, "shift": q_shift, "mixture": q_mix, "uniform": q_uniform}

    def analytic(q: torch.Tensor) -> tuple[float, float]:
        h = float(-(p * p.log()).sum())
        kl = float((p * (p.log() - q.log())).sum())
        return h, kl

    # construction guard: the seeded perturbations must be strongly distinct
    kl_values = {name: analytic(q)[1] for name, q in candidates.items()}
    assert kl_values["p"] == pytest.approx(0.0, abs=1e-12)
    for name in ("shift", "mixture", "uniform"):
        assert kl_values[name] >= 0.05, (
            f"construction guard: KL(p||{name})={kl_values[name]:.4f} too small "
            "to separate from MC noise — change the seed/perturbation"
        )

    expected: dict[str, float] = {}
    for name, q in candidates.items():
        logits_q = q.log().view(1, k).expand(n, -1)  # softmax(log q) = q
        per_sample = choice_loss(logits_q, y, reduction="none")  # REAL implementation
        expected[name] = float(per_sample.mean())
        assert torch.isfinite(per_sample).all()

    # (b)+(c): strict propriety — E[loss(p)] strictly smallest
    for name in ("shift", "mixture", "uniform"):
        assert expected["p"] < expected[name], (
            f"T5(c) violated: E[loss(p)]={expected['p']:.6f} not < "
            f"E[loss({name})]={expected[name]:.6f}"
        )

    # (a): E[loss(q)] = H(p) + KL(p||q) within MC error (n=2e5 -> SE ~0.007)
    tol = 0.03  # ~4 standard errors of the sample mean
    h = analytic(p)[0]
    assert abs(expected["p"] - h) <= tol, (
        f"E[loss(p)] - H(p) = {expected['p'] - h:+.5f} should be ~= KL(p||p)=0"
    )
    for name, q in candidates.items():
        _, kl = analytic(q)
        assert kl >= -1e-12, f"analytic KL(p||{name}) negative: {kl}"
        gap = expected[name] - h
        assert abs(gap - kl) <= tol, (
            f"T5(a) for q={name}: E-H = {gap:.5f} vs KL = {kl:.5f} (tol {tol})"
        )
    print(
        "[measure] T5 MC (n=200000): "
        + ", ".join(f"E[loss({n_})]={v:.4f}" for n_, v in expected.items())
        + f", H(p)={h:.4f}, KL="
        + ", ".join(f"{n_}:{kl_values[n_]:.4f}" for n_ in candidates)
        + f", max decomposition err = "
        + f"{max(abs(expected[n_] - h - kl_values[n_]) for n_ in candidates):.4f}"
    )


def test_t5_choice_loss_temperature_grid_argmin_near_one():
    """T5(c) through the REAL ``choice_loss``: grid-searching the
    temperature-scaled forecast q_T = softmax(z/T) on a labelled batch whose
    empirical frequencies match p = softmax(z) gives argmin_T ~= 1 within one
    grid step (grid includes T=1; the population minimizer is exactly T=1)."""
    gen = torch.Generator().manual_seed(SEED + 51_000)
    k = 4
    z = torch.randn(1, k, generator=gen).double() * 1.5
    p = torch.softmax(z, dim=-1).squeeze(0)
    # labels with (near-)exact frequencies p: CE(p, q_T) = population risk
    m = 200_000
    counts = (p * m).round().long()
    counts[int(counts.argmax())] += m - int(counts.sum())  # exact total
    labels = torch.repeat_interleave(torch.arange(k), counts)

    grid = torch.linspace(0.5, 1.5, 21, dtype=torch.float64)  # step 0.05, includes 1.0
    n_labels = int(labels.numel())
    logits_batch = z.expand(n_labels, -1)  # [n, K]: one logit vector per label
    losses = [
        float(choice_loss(logits_batch / float(t), labels, reduction="mean"))
        for t in grid
    ]
    assert bool(torch.isfinite(torch.tensor(losses)).all())
    best = int(min(range(len(grid)), key=lambda i: losses[i]))
    t_star = float(grid[best])
    step = 0.05
    assert abs(t_star - 1.0) <= step + 1e-9, (
        f"T5(c): grid argmin T*={t_star} not within one grid step of 1 "
        f"(losses={losses})"
    )
    print(
        f"[measure] T5 temperature grid: T* = {t_star:.4f}, grid step = {step}, "
        f"loss(T*) = {losses[best]:.6f}, loss(T=1) = {losses[10]:.6f}"
    )


# ===========================================================================
# T6 — Brier decomposition and optimality (PROOFS.md T6)
# ===========================================================================


def test_t6_brier_decomposition_hand_built_example():
    """T6(a) on the hand-built 8-sample / 2-class example with known bins
    (forecasts constant within each bin — assumption A6.2): BS = Rel - Res + Unc
    exactly (fp64, tolerance 1e-9), with the known closed-form values."""
    q = torch.tensor(
        [
            [0.9, 0.1],
            [0.9, 0.1],  # bin 1: n=2, forecast (0.9, 0.1)
            [0.6, 0.4],
            [0.6, 0.4],
            [0.6, 0.4],  # bin 2: n=3, forecast (0.6, 0.4)
            [0.3, 0.7],
            [0.3, 0.7],
            [0.3, 0.7],  # bin 3: n=3, forecast (0.3, 0.7)
        ],
        dtype=torch.float64,
    )
    y = torch.tensor([0, 0, 0, 1, 1, 1, 1, 0])
    n = 8
    # Brier via the REAL implementation (computes in float64 internally)
    bs = brier_score(q, y)

    # Rel / Res / Unc exactly per PROOFS.md (all in fp64)
    bin_slices = (slice(0, 2), slice(2, 5), slice(5, 8))
    w = torch.tensor([2 / 8, 3 / 8, 3 / 8], dtype=torch.float64)
    q_bar = torch.stack([q[s].mean(dim=0) for s in bin_slices])
    p_bar_k = torch.stack(
        [
            torch.nn.functional.one_hot(y[s], num_classes=2).to(torch.float64).mean(dim=0)
            for s in bin_slices
        ]
    )
    p_bar = (w.unsqueeze(1) * p_bar_k).sum(dim=0)  # climatology
    rel = float((w * ((q_bar - p_bar_k) ** 2).sum(dim=1)).sum())
    res = float((w * ((p_bar_k - p_bar) ** 2).sum(dim=1)).sum())
    unc = float(1.0 - (p_bar**2).sum())

    assert rel >= 0.0 and res >= 0.0 and unc >= 0.0, "Rel/Res/Unc must be nonnegative"
    residual = abs(bs - (rel - res + unc))
    assert residual <= 1e-9, f"T6(a) decomposition identity broken: residual {residual}"
    # known closed forms for this construction
    assert abs(bs - 0.3925) <= 1e-12, f"BS should be 0.3925, got {bs}"
    assert abs(unc - 0.5) <= 1e-12, f"Unc should be 1/2, got {unc}"
    assert abs(res - 1 / 6) <= 1e-12, f"Res should be 1/6, got {res}"
    assert abs(rel - 71 / 1200) <= 1e-12, f"Rel should be 71/1200, got {rel}"
    assert rel + unc >= res - 1e-12, "corollary Rel + Unc >= Res"
    print(
        f"[measure] T6 decomposition: BS={bs:.12f} Rel={rel:.12f} Res={res:.12f} "
        f"Unc={unc:.12f}, |BS-(Rel-Res+Unc)| = {residual:.3e}"
    )


def test_t6_brier_uniquely_minimized_at_truth_numeric():
    """T6(b): for true p = (0.7, 0.3), the exact expected Brier score
    E[||q - o||^2] (evaluated through the REAL brier_score, weighted by p)
    is minimized at q = p on a 0.001 grid; minimum value = 1 - ||p||^2."""
    p = torch.tensor([0.7, 0.3], dtype=torch.float64)

    def exact_expected(q_row: torch.Tensor) -> float:
        # E_{Y~p} Brier(q, e_Y) computed by summing the two outcome scores
        e0 = brier_score(q_row.unsqueeze(0), torch.tensor([0]))
        e1 = brier_score(q_row.unsqueeze(0), torch.tensor([1]))
        return float(p[0]) * e0 + float(p[1]) * e1

    grid = torch.linspace(0.0, 1.0, 1001, dtype=torch.float64)
    losses = [exact_expected(torch.tensor([float(t), 1.0 - float(t)])) for t in grid]
    best = int(min(range(len(grid)), key=lambda i: losses[i]))
    t_star = float(grid[best])
    assert abs(t_star - 0.7) <= 0.001 + 1e-12, (
        f"T6(b): argmin q = ({t_star}, {1 - t_star}) should be within one grid "
        f"step of p = (0.7, 0.3)"
    )
    min_value = losses[best]
    unc = float(1.0 - (p**2).sum())
    assert abs(min_value - unc) <= 1e-9, (
        f"T6(b): minimum {min_value} should equal Unc(p) = {unc}"
    )
    print(
        f"[measure] T6 optimality: argmin q1 = {t_star:.4f} (grid step 0.001), "
        f"min = {min_value:.9f}, Unc(p) = {unc:.9f}"
    )


# ===========================================================================
# T7 — Parallel-state equivalence on the REAL model (PROOFS.md T7, A7.1-A7.5)
# ===========================================================================

T7_STATE = "hello world foo bar"
T7_QUESTIONS = ["q one", "q two is this", "which option", "is it safe"]
T7_OPTIONS = [["a", "b"], ["c", "d", "e"], ["f", "g", "h", "i"], ["j", "k"]]


@pytest.fixture()
def t7_wrapper(tiny_bert, tiny_tokenizer) -> JevXForDecision:
    return _build_wrapper(tiny_bert, tiny_tokenizer)


def test_t7a_batched_equals_sequential_single_question_calls(t7_wrapper) -> None:
    """T7(ii): m=4 batched vs 4 sequential single-question calls — per-question
    logits/probs allclose with MEASURED max_abs_error < 1e-5.

    RM equality (T7-i) holds only up to fp kernel/reduction-order differences
    (A7.4b relaxed; Remark 7.2) — fp bit-equality across shapes is NOT claimed.
    """
    w = t7_wrapper
    assert [len(o) for o in T7_OPTIONS] == [2, 3, 4, 2]  # variable K in one batch
    with torch.no_grad():
        batched = w.choice(T7_STATE, T7_QUESTIONS, options=T7_OPTIONS)
    assert len(batched.decisions) == 4
    max_err = 0.0
    for j, opts in enumerate(T7_OPTIONS):
        k = len(opts)
        with torch.no_grad():
            solo = w.choice(T7_STATE, [T7_QUESTIONS[j]], options=[opts])
        b_logits = batched.decisions[j].logits[0, :k]
        s_logits = solo.decisions[0].logits[0]
        b_probs = batched.decisions[j].probabilities[0, :k]
        s_probs = solo.decisions[0].probabilities[0]
        err = max(
            float((b_logits - s_logits).abs().max()),
            float((b_probs - s_probs).abs().max()),
        )
        max_err = max(max_err, err)
        assert err < 1e-5, f"question {j}: batched vs sequential diverged by {err:.3e}"
        assert int(batched.decisions[j].predicted_index) == int(
            solo.decisions[0].predicted_index
        ), f"question {j}: discrete decision flipped without a margin argument"
    print(f"[measure] T7(a) m=4 batched vs sequential: max_abs_error = {max_err:.3e} (tol 1e-5)")


def test_t7b_single_backbone_forward_per_batched_call(t7_wrapper) -> None:
    """T7/A7.2: H_S = E_M(S) computed exactly once — one backbone forward per
    wrapper call, counted by a real forward hook (not a declared constant)."""
    w = t7_wrapper
    counter = {"n": 0}

    def hook(module, args, output):
        counter["n"] += 1

    handle = w.model.register_forward_hook(hook)
    try:
        with torch.no_grad():
            out = w.choice(T7_STATE, T7_QUESTIONS, options=T7_OPTIONS)
    finally:
        handle.remove()
    assert len(out.decisions) == 4
    assert counter["n"] == 1, f"m=4 batched call ran the backbone {counter['n']} times, not 1"
    print(f"[measure] T7(b) backbone forward count = {counter['n']} for m=4")


def test_t7c_schema_perturbation_leaves_other_questions_unchanged(t7_wrapper) -> None:
    """P11.1/T7-A7.3 (functional independence): perturbing q2's schema leaves
    q1's logits/probs unchanged < 1e-7 (in practice: bit-identical)."""
    w = t7_wrapper
    options_b = [T7_OPTIONS[0], ["x", "y", "z"], T7_OPTIONS[2], T7_OPTIONS[3]]
    with torch.no_grad():
        base = w.choice(T7_STATE, T7_QUESTIONS, options=T7_OPTIONS)
        perturbed = w.choice(T7_STATE, T7_QUESTIONS, options=options_b)
    logits_err = float(
        (base.decisions[0].logits - perturbed.decisions[0].logits).abs().max()
    )
    probs_err = float(
        (base.decisions[0].probabilities - perturbed.decisions[0].probabilities).abs().max()
    )
    assert logits_err < 1e-7, f"q1 logits changed by {logits_err:.3e} when q2's schema changed"
    assert probs_err < 1e-7, f"q1 probs changed by {probs_err:.3e} when q2's schema changed"
    print(
        f"[measure] T7(c) q2-schema perturbation -> q1: "
        f"logits err = {logits_err:.3e}, probs err = {probs_err:.3e} (tol 1e-7)"
    )


def test_t7d_state_batch_b2_vs_two_b1_calls(t7_wrapper) -> None:
    """T7(ii) for the state axis: B=2 batch (rows of different length ->
    token padding) vs two B=1 calls — row-wise allclose, measured < 1e-5.
    Padding differences only affect fp reduction order (Lemma 7.1 says padding
    itself is exact; residual error comes from kernel shape selection)."""
    w = t7_wrapper
    states = ["hello world", "the cat sat a b c foo bar baz"]
    options = [["a", "b", "c"]]
    with torch.no_grad():
        batched = w.choice(states, ["q one"], options=options)
        singles = [w.choice(s, ["q one"], options=options) for s in states]
    assert len(batched.decisions) == 2
    max_err = 0.0
    for b in range(2):
        err = max(
            float((batched.decisions[b].logits - singles[b].decisions[0].logits).abs().max()),
            float(
                (batched.decisions[b].probabilities - singles[b].decisions[0].probabilities)
                .abs()
                .max()
            ),
        )
        max_err = max(max_err, err)
        assert err < 1e-5, f"state row {b}: B=2 vs B=1 diverged by {err:.3e}"
        assert int(batched.decisions[b].predicted_index) == int(
            singles[b].decisions[0].predicted_index
        )
    print(f"[measure] T7(d) B=2 vs two B=1: max_abs_error = {max_err:.3e} (tol 1e-5)")


# ===========================================================================
# T8 — Typed-output invariant through the REAL forward (PROOFS.md T8, D8.I)
# ===========================================================================


def test_t8_random_forward_outputs_in_declared_type_space(tiny_bert, tiny_tokenizer) -> None:
    """100 random DecisionOutput objects produced by the actual
    ``JevXForDecision.forward``: Choice on the simplex, Score in the box
    [0, K-1], Noul strictly in (0, 1) — including variable-K masked batches."""
    w = _build_wrapper(tiny_bert, tiny_tokenizer)
    torch.manual_seed(SEED + 80_000)
    max_simplex_err = 0.0
    max_box_violation = 0.0
    for trial in range(TRIALS):
        variant = trial % 4
        with torch.no_grad():
            if variant == 0:  # single choice, random K in 1..12
                k = int(torch.randint(1, 13, (1,)))
                keys = [f"t{trial}_o{j}" for j in range(k)]
                out = w.choice(T7_STATE, [f"question {trial}"], options=[keys])
                dec = out.decisions[0]
                probs = dec.probabilities
                assert bool((probs >= 0.0).all())
                err = float((probs.sum() - 1.0).abs())
                max_simplex_err = max(max_simplex_err, err)
                assert err <= 1e-6, f"trial {trial}: simplex violated, err={err}"
                idx = int(dec.predicted_index[0])
                assert 0 <= idx < k
                assert dec.predicted_value in keys, "T2/T8: decision must be in C"
            elif variant == 1:  # batched m=2 with different K -> masked padding
                k1 = int(torch.randint(1, 4, (1,)))
                k2 = int(torch.randint(4, 9, (1,)))
                keys1 = [f"t{trial}_a{j}" for j in range(k1)]
                keys2 = [f"t{trial}_b{j}" for j in range(k2)]
                out = w.choice(
                    T7_STATE, ["which one", "rate this"],
                    options=[keys1, keys2],
                )
                for dec, keys, k in (
                    (out.decisions[0], keys1, k1),
                    (out.decisions[1], keys2, k2),
                ):
                    probs = dec.probabilities  # padded width k2
                    assert bool((probs >= 0.0).all())
                    assert bool((probs[0, k:] == 0.0).all()), "masked entries exact 0"
                    err = float((probs[0, :k].sum() - 1.0).abs())
                    max_simplex_err = max(max_simplex_err, err)
                    assert err <= 1e-6, f"trial {trial}: simplex violated, err={err}"
                    idx = int(dec.predicted_index[0])
                    assert 0 <= idx < k, f"trial {trial}: idx {idx} outside [0,{k})"
                    assert dec.predicted_value in keys
            elif variant == 2:  # score, random K levels
                k = int(torch.randint(2, 8, (1,)))
                levels = [str(i) for i in range(k)]
                out = w.score(T7_STATE, [f"score {trial}"], options=[levels])
                dec = out.decisions[0]
                assert torch.isfinite(dec.expected_score).all()
                s = float(dec.expected_score[0])
                violation = max(0.0, -s, s - (k - 1))
                max_box_violation = max(max_box_violation, violation)
                assert violation <= 1e-4, f"trial {trial}: score {s} outside [0,{k - 1}]"
                err = float((dec.probabilities.sum() - 1.0).abs())
                max_simplex_err = max(max_simplex_err, err)
                assert err <= 1e-6
            else:  # noul: strictly interior
                out = w.noul(T7_STATE, [f"noul {trial}"])
                dec = out.decisions[0]
                p = dec.probabilities
                assert torch.isfinite(p).all()
                assert bool((p > 0.0).all()), f"trial {trial}: noul p = {float(p)}"
                assert bool((p < 1.0).all()), f"trial {trial}: noul p = {float(p)}"
                assert int(dec.predicted_index[0]) in (0, 1)
    print(
        f"[measure] T8 forward: {TRIALS} trials, max simplex err = {max_simplex_err:.3e}, "
        f"max box violation = {max_box_violation:.3e}"
    )


def _fit_engine(wrapper: JevXForDecision, logits: torch.Tensor, labels: torch.Tensor) -> None:
    wrapper.calibration_engine.fit(logits, labels)


@pytest.fixture()
def temp_wrapper(tiny_bert, tiny_tokenizer) -> JevXForDecision:
    """Wrapper with a FITTED temperature calibrator (D8.I composition)."""
    w = _build_wrapper(tiny_bert, tiny_tokenizer, calibration_method="temperature")
    gen = torch.Generator().manual_seed(SEED + 81_000)
    _fit_engine(
        w,
        torch.randn(512, 3, generator=gen) * 2.0,
        torch.randint(0, 3, (512,), generator=gen),
    )
    assert w.calibration_engine.is_fitted
    return w


@pytest.fixture()
def vec3_wrapper(tiny_bert, tiny_tokenizer) -> JevXForDecision:
    """Wrapper with a FITTED vector calibrator at K=3 (width-bound by D8.I)."""
    w = _build_wrapper(tiny_bert, tiny_tokenizer, calibration_method="vector")
    gen = torch.Generator().manual_seed(SEED + 81_001)
    _fit_engine(
        w,
        torch.randn(512, 3, generator=gen) * 2.0,
        torch.randint(0, 3, (512,), generator=gen),
    )
    assert w.calibration_engine.is_fitted
    return w


@pytest.fixture()
def vec1_wrapper(tiny_bert, tiny_tokenizer) -> JevXForDecision:
    """Wrapper with a FITTED vector calibrator at K=1 (noul logits)."""
    w = _build_wrapper(tiny_bert, tiny_tokenizer, calibration_method="vector")
    gen = torch.Generator().manual_seed(SEED + 81_002)
    _fit_engine(
        w,
        torch.randn(512, 1, generator=gen),
        torch.randint(0, 2, (512,), generator=gen),
    )
    assert w.calibration_engine.is_fitted
    return w


def test_t8_calibration_then_project_preserves_membership(
    temp_wrapper, vec3_wrapper, vec1_wrapper
) -> None:
    """T8 composition clause + D8.I: Cal is applied between G_type and
    Pi_T (real forward path); 100 random trials over temperature/vector x
    choice/noul — calibrated outputs stay in the SAME type space: choice
    on the simplex, noul strictly in (0, 1)."""
    torch.manual_seed(SEED + 82_000)
    max_simplex_err = 0.0
    plan = (
        [("temp", "choice")] * 40
        + [("vec3", "choice")] * 25
        + [("temp", "noul")] * 20
        + [("vec1", "noul")] * 15
    )
    wrappers = {"temp": temp_wrapper, "vec3": vec3_wrapper, "vec1": vec1_wrapper}
    for trial, (method, kind) in enumerate(plan):
        w = wrappers[method]
        with torch.no_grad():
            if kind == "choice":
                k = 3 if method == "vec3" else int(torch.randint(1, 13, (1,)))
                keys = [f"c{trial}_o{j}" for j in range(k)]
                out = w.choice(T7_STATE, [f"cal choice {trial}"], options=[keys])
                dec = out.decisions[0]
                assert dec.calibration_metadata is not None
                assert dec.calibration_metadata["applied"] is True, (
                    "calibration must actually run (fitted engine)"
                )
                probs = dec.probabilities
                assert bool((probs >= 0.0).all())
                err = float((probs.sum() - 1.0).abs())
                max_simplex_err = max(max_simplex_err, err)
                assert err <= 1e-6, (
                    f"trial {trial} ({method}): calibrated choice off the simplex, err={err}"
                )
                assert dec.predicted_value in keys
            else:
                out = w.noul(T7_STATE, [f"cal noul {trial}"])
                dec = out.decisions[0]
                assert dec.calibration_metadata["applied"] is True
                p = dec.probabilities
                assert torch.isfinite(p).all()
                assert bool((p > 0.0).all()), (
                    f"trial {trial} ({method}): calibrated noul p not > 0: {float(p)}"
                )
                assert bool((p < 1.0).all()), (
                    f"trial {trial} ({method}): calibrated noul p not < 1: {float(p)}"
                )
                assert int(dec.predicted_index[0]) in (0, 1)
    print(
        f"[measure] T8 calibration-then-project: 100 trials "
        f"(40 temp-choice, 25 vector-choice, 20 temp-noul, 15 vector-noul), "
        f"max simplex err = {max_simplex_err:.3e}"
    )


# ===========================================================================
# PLUS — structural claims beyond T1-T8: pooling / fusion / risk-coverage
# ===========================================================================


@pytest.mark.parametrize("strategy", ["masked_mean", "last_valid", "attention_pool"])
@pytest.mark.parametrize("pad_side", ["left", "right"])
def test_plus_masked_pooling_perturbation_invariance(strategy, pad_side) -> None:
    """P4.3 padding invariance: perturbing PADDED hidden values (finite, huge)
    leaves the pooled output BITWISE unchanged, for left and right padding,
    for masked_mean / last_valid / attention_pool."""
    torch.manual_seed(SEED + 90_000)
    b, t_valid, t_pad, d = 2, 5, 3, 16
    valid = torch.randn(b, t_valid, d)
    seq = torch.zeros(b, t_valid + t_pad, d)
    mask = torch.zeros(b, t_valid + t_pad, dtype=torch.bool)
    if pad_side == "right":
        seq[:, :t_valid] = valid
        mask[:, :t_valid] = True
    else:
        seq[:, t_pad:] = valid
        mask[:, t_pad:] = True

    baseline = pool(seq, mask, strategy).detach()
    perturbed = seq.clone()
    gen = torch.Generator().manual_seed(SEED + 90_001)
    perturbed[~mask] = torch.randn(int((~mask).sum()), d, generator=gen) * 1e6
    out = pool(perturbed, mask, strategy).detach()
    assert torch.equal(baseline, out), (
        f"{strategy}/{pad_side}: pooled output changed under padded-value "
        f"perturbation (max diff {float((baseline - out).abs().max()):.3e})"
    )
    # Sanity (not the required claim): the same logical content pooled from
    # the compact block must match — within fp summation-tree differences
    # (left-padding changes the reduction order of .sum; PROOFS Remark 7.2
    # fp non-associativity — the BITWISE claim above is only about perturbing
    # padded values WITHIN one layout).
    ref = pool(valid, torch.ones(b, t_valid, dtype=torch.bool), strategy).detach()
    err = float((baseline - ref).abs().max())
    assert err <= 1e-6, f"{strategy}/{pad_side}: compact vs padded block differ by {err:.3e}"


@pytest.mark.parametrize("kind", sorted(available_fusions()))
def test_plus_fusion_masked_option_invariance(kind) -> None:
    """D5.1/A7.3 masking correctness: perturbing MASKED option features leaves
    the valid logits unchanged < 1e-7 (in practice bit-identical: masked rows
    are zeroed by masked_fill after the row-wise maps)."""
    torch.manual_seed(SEED + 91_000)
    b, k_max, d = 3, 5, 8
    fusion = build_fusion(kind, d)
    fusion.eval()
    state = torch.randn(b, d)
    question = torch.randn(b, d)
    options = torch.randn(b, k_max, d)
    counts = torch.tensor([3, 1, 5])
    mask = mask_from_num_options(counts, k_max)

    out_base = fusion(state, question, options, mask)
    head = ChoiceHead(d, interaction="bilinear")
    logits_base = head.compute_logits(
        out_base.fused, out_base.option_features
    ).detach()

    options_perturbed = options.clone()
    gen = torch.Generator().manual_seed(SEED + 91_001)
    options_perturbed[~mask] = (
        torch.randn(int((~mask).sum()), d, generator=gen) * 1e4
    )
    out_pert = fusion(state, question, options_perturbed, mask)
    logits_pert = head.compute_logits(
        out_pert.fused, out_pert.option_features
    ).detach()

    # mechanism: masked option rows are exactly 0 in BOTH runs
    assert bool((out_base.option_features[~mask] == 0.0).all())
    assert bool((out_pert.option_features[~mask] == 0.0).all())
    # the claim: valid logits do not move
    diff = (logits_base - logits_pert).abs().masked_fill(~mask, 0.0)
    max_diff = float(diff.max())
    assert max_diff < 1e-7, f"{kind}: masked-option perturbation moved valid logits by {max_diff:.3e}"
    # (and valid option features are bitwise unchanged — Lemma 7.1 at fp32)
    feat_diff = (
        (out_base.option_features.detach() - out_pert.option_features.detach())
        .abs()
        .masked_fill(~mask.unsqueeze(-1), 0.0)
    )
    assert float(feat_diff.max()) == 0.0, f"{kind}: valid option features moved"
    print(f"[measure] fusion masked-option invariance [{kind}]: valid logit max diff = {max_diff:.3e}")


def test_plus_risk_coverage_coverage_monotone_and_car_consistent() -> None:
    """Section 62 / Definition 10 (structural only):

    * coverage along the curve is non-decreasing (and equals i/n);
    * coverage_at_risk(alpha) in [0,1] and consistent with the returned curve;
    * risk itself is NOT asserted monotone — the spec explicitly disclaims
      that (one deterministic nonmonotone example is asserted instead).
    100 random trials x 4 alpha values.
    """
    # deterministic demonstration that risk is non-monotone (so we must not check it)
    conf = torch.tensor([0.9, 0.4, 0.8, 0.1])
    correct = torch.tensor([True, False, True, True])
    cov, risk = risk_coverage_curve(conf, correct)
    # ranking: 0.9(T), 0.8(T), 0.4(F), 0.1(T) -> risk 0, 0, 1/3, 1/4
    assert torch.allclose(risk, torch.tensor([0.0, 0.0, 1 / 3, 0.25]), atol=1e-6)
    assert risk[3] < risk[2], "risk must be allowed to be non-monotone (Def 10)"
    assert abs(coverage_at_risk(conf, correct, 0.0) - 0.5) <= 1e-6
    assert abs(coverage_at_risk(conf, correct, 0.1) - 0.5) <= 1e-6
    assert abs(coverage_at_risk(conf, correct, 0.25) - 1.0) <= 1e-6
    assert abs(coverage_at_risk(conf, correct, 0.33) - 1.0) <= 1e-6

    gen = torch.Generator().manual_seed(SEED + 92_000)
    alphas = (0.0, 0.2, 0.5, 1.0)
    checked = 0
    for trial in range(TRIALS):
        n = int(torch.randint(1, 60, (1,), generator=gen))
        confidences = torch.rand(n, generator=gen)
        correct_mask = torch.rand(n, generator=gen) < torch.rand(1, generator=gen).item()
        cov, risk = risk_coverage_curve(confidences, correct_mask)
        assert cov.shape == risk.shape == (n,)
        # coverage non-decreasing, exactly i/n
        if n > 1:
            assert bool((cov[1:] - cov[:-1] >= 0).all()), "coverage must be non-decreasing"
        expected_cov = torch.arange(1, n + 1, dtype=torch.float32) / n
        assert torch.allclose(cov, expected_cov, atol=1e-6)
        # risk is an empirical rate in [0,1] (NOT checked for monotonicity)
        assert bool((risk >= 0.0).all()) and bool((risk <= 1.0).all())
        for alpha in alphas:
            car = coverage_at_risk(confidences, correct_mask, alpha)
            assert 0.0 <= car <= 1.0, f"coverage_at_risk({alpha}) = {car} outside [0,1]"
            # consistency with the returned curve: max coverage whose risk <= alpha
            feasible = risk <= alpha + 1e-6
            expected_car = float(cov[feasible].max()) if bool(feasible.any()) else 0.0
            assert abs(car - expected_car) <= 1e-6, (
                f"trial {trial} alpha={alpha}: coverage_at_risk={car} vs curve {expected_car}"
            )
            checked += 1
    print(
        f"[measure] risk-coverage: {TRIALS} trials x {len(alphas)} alphas = {checked} "
        "coverage_at_risk checks, coverage monotone, risk deliberately NOT checked"
    )
