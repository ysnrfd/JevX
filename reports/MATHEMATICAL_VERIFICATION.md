# JevX Mathematical Verification — Executable Instantiation of PROOFS.md T1–T8

**Phase 24 + §27-executable + §68.** Every row below corresponds to a test that was
EXECUTED in this repository (not proposed): the paper proof lives in `docs/PROOFS.md`,
and the executable check instantiates the theorem's *statement* against the real
implementation (`jevx.decisions` heads, `jevx.representations` pooling/fusion,
`jevx.calibration`, `ConfidenceEngine`, `JevXForDecision`, `jevx.training.losses`,
`jevx.utils.numerical`). The suite checks that the implementation **satisfies** each
theorem; it does not re-derive the mathematics.

## Environment

| Item | Value |
|---|---|
| Date | 2026-09-24 |
| OS / platform | Windows 11 (win32 10.0.26100 x64), CPU-only |
| Python | 3.14.3 (MSC v.1944 64 bit) |
| torch | 2.14.0+cpu |
| transformers | 5.16.1 |
| pytest | 9.0.3 |
| Network | not used; all fixtures in-memory (`tests/conftest.py` tiny_bert / tiny_tokenizer) |

## Seed policy

`torch.manual_seed(20260924 + trial)` at the top of every property trial (base seed
`SEED = 20260924`); Monte-Carlo sampling uses dedicated local
`torch.Generator().manual_seed(...)` instances; wrapper construction is preceded by
`torch.manual_seed(SEED)` so head/fusion/projection draws are reproducible;
calibration fits are RNG-free by design (temperature = deterministic grid search,
isotonic = deterministic PAVA, vector = seeded LBFGS). Every property check is
deterministic: re-running reproduces the exact numbers below.

## Tolerance policy

A tolerance may reflect **only** the theorem's own stated assumptions/caveats
(fp underflow, fp non-associativity); every widening is documented at its assertion:

* **fp underflow (PROOFS Remark 1.2):** `exp` gaps ≳ 104 underflow to exact 0 in
  binary32. T1 asserts *strict* positivity where the gap is within fp range and
  asserts the *exact-0* value where the gap is ≫ 104 (the documented caveat);
  masked entries are intentionally exact 0 via `finfo.min` (`masked_softmax`).
* **Shift-invariance rounding (Remark 1.2):** `z + 1e6` itself rounds
  (ulp(1e6) = 0.0625), so the arbitrary-`z` shift check uses the derived bound
  e^(2·0.03125) − 1 ≈ 0.0645 → tol 0.07; the exactly-representable-shift check
  (half-integer `z`, c = ±1e3/±1e6) uses tol 1e-6 (measured 0 — bit-equal).
* **fp non-associativity / kernel reduction order (Remarks 3.2, 7.2):** T7 batched vs
  sequential/state-batch comparisons use a *measured* tolerance 1e-5 (T7-ii is an
  [EMPIRICAL] claim template); bit-equality across shapes is never asserted.
  The pooling compact-vs-padded *sanity* uses 1e-6 for the same reason (the required
  bitwise claim is only about perturbing padded values within one layout).
* **Score box (Remark 3.2 + head guard):** T3/T8 box tolerance 1e-4 = the head's own
  documented implementation guard (ScoreHead asserts, never clamps).
* **Noul clamp (Remark 4.1):** strict `(0,1)` for stored noul probabilities is held
  by the sanctioned `[eps, 1−eps]` clamp (implementation invariant, bias ≤
  `finfo(dtype).eps` = 1.19e-07) — see "Implementation bug fixed" below.
* **MC error (T5):** decomposition tolerance 0.03 ≈ 4 standard errors of the sample
  mean at n = 200 000 (σ of −log q_y ≤ ~3); the construction guard requires every
  perturbation to have analytic KL ≥ 0.05 so the strict-inequality sign is separated
  from MC noise by > 20 SE.

## Test inventory

| File | Tests (parametrized cases) |
|---|---|
| `tests/numerical/test_theorem_verification.py` | 33 — T1×4, T2×2, T3×2, T4×1, T5×2, T6×2, T7×4, T8×2, PLUS×14 (pooling 3×2, fusion 7, risk–coverage 1) |
| `tests/numerical/test_numerical_stability.py` | 8 — head-stack×2, temperature×1, near-duplicate fits×2, fp16/bf16×2, naive-vs-stable×1 |
| **Total** | **41** (all executed; ≥100 random trials per property check unless the theorem specifies a fixed construction) |

## Theorem → proof location → executed test → status → measured error

| Theorem | Paper proof | Executed test(s) | Status | Measured max error / value |
|---|---|---|---|---|
| **T1** softmax validity | `docs/PROOFS.md` §T1, Remarks 1.1–1.2 | `test_t1_softmax_random_validity_and_reference_match` (100 trials); `test_t1_adversarial_magnitude_1e6_validity_and_reference` (25×4 = 100 cases at ±1e5/±1e6); `test_t1_extreme_spread_documents_fp_underflow_caveat`; `test_t1_shift_invariance` | **Verified (caveated)** | max \|Σp−1\| = **1.192e-07** (tol 1e-6); `stable_softmax` ≡ `torch.softmax` fp32 (bit-equal, every trial); strict p>0 wherever gap ≤ ~100; gaps ≫104 → exact 0 as *documented* (Remark 1.2); shift: exact-shift err **0.000e+00**, arbitrary-z ±1e6-shift err **1.017e-02** ≤ 0.07 (fp rounding of z+c, Remark 1.2) |
| **T2** choice type safety | §T2 (A2.1–A2.5) | `test_t2_argmax_in_range_and_schema_membership_100_schemas` (K 1–12; keys incl. duplicates/negative/float); `test_t2_masked_argmax_never_selects_invalid_slot_100_trials` (every invalid raw logit strictly largest, incl. +1e6) | **Verified** | argmax escapes: **0**; schema-membership failures: **0** (300 projections); invalid-slot selections: **0**/100 trials; masked probs exactly 0; max \|Σp−1\| = **2.384e-07** |
| **T3** score boundedness | §T3, Remark 3.2 | `test_t3_expected_score_in_level_box_100_trials` (incl. inputs scaled to 1e6 + crafted ±1e4/±1e6 rows); `test_t3_uniform_probs_give_exact_center` | **Verified** | max box violation = **0.000e+00** (asserted at the head's documented 1e-4 guard); uniform center \|s−(K−1)/2\| = **2.384e-07** ≤ 1e-6 (K=1..12) |
| **T4** binary validity | §T4, Remark 4.1 | `test_t4_noul_strict_interior_for_finite_extreme_logits` (z ∈ {0, ±1e3, ±1e4, ±1e6} + forward path at fused·1e4) | **Verified AFTER implementation fix (see below — flagged loudly)** | p(0)=0.500000000, p(1e3)=p(1e6)=**0.999999881**, p(−1e3)=p(−1e6)=**1.192e-07** — strictly interior for every finite fp32-representable z; clamp bias ≤ eps = 1.19e-07 |
| **T5** log score strictly proper | §T5 | `test_t5_log_score_strictly_proper_monte_carlo` (n = 200 000 seeded, losses via real `choice_loss`); `test_t5_choice_loss_temperature_grid_argmin_near_one` (grid 0.5–1.5, step 0.05) | **Verified** | strict E[loss(p)] < E[loss(q)] for all q≠p (E(p)=**1.2334** vs 1.3210/1.6094/1.7036; min KL = **0.0864**); max \|E−(H+KL)\| = **0.0020** ≤ 0.03 (4 SE); H(p)=1.2354; analytic KL ≥ 0 (0.0864…0.4671); grid argmin **T\* = 1.0000** (within step 0.05; T\* on-grid) |
| **T6** Brier decomposition + optimality | §T6 (A6.2, A6.3) | `test_t6_brier_decomposition_hand_built_example` (8 samples, 2 classes, 3 known bins, forecasts constant per bin); `test_t6_brier_uniquely_minimized_at_truth_numeric` (real `brier_score`, 1001-point grid) | **Verified** | \|BS−(Rel−Res+Unc)\| = **0.000e+00** ≤ 1e-9; known values exact to 1e-12: BS=0.3925, Rel=71/1200, Res=1/6, Unc=1/2; Rel,Res,Unc ≥ 0; grid argmin q₁ = **0.7000** (=p, step 0.001), min = **0.420000000** = Unc(p) |
| **T7** parallel-state equivalence | §T7 (A7.1–A7.5, Remarks 7.2–7.3), real `JevXForDecision` on tiny_bert | `test_t7a_batched_equals_sequential_single_question_calls`; `test_t7b_single_backbone_forward_per_batched_call`; `test_t7c_schema_perturbation_leaves_other_questions_unchanged`; `test_t7d_state_batch_b2_vs_two_b1_calls` | **Verified (T7-ii measured, as specified)** | (a) m=4 batched vs 4×sequential max_abs_error = **2.235e-08** < 1e-5 (logits+probs; decisions equal); (b) backbone forward-hook count = **1**; (c) q2-schema perturbation → q1 logits/probs err = **0.000e+00** < 1e-7; (d) B=2 vs two B=1 max_abs_error = **3.725e-08** < 1e-5. fp caveat: cross-shape bit-equality not claimed (Remark 7.2) |
| **T8** typed-output invariant | §T8 + Corollary 8.1 (D8.I) | `test_t8_random_forward_outputs_in_declared_type_space` (100 real `forward` calls: 25 single-choice K 1–12, 25 variable-K masked batches, 25 score, 25 noul); `test_t8_calibration_then_project_preserves_membership` (100 trials: 40 temperature-choice, 25 vector-choice, 20 temperature-noul, 15 vector-noul — fitted engines, real forward) | **Verified** | max simplex err = **1.192e-07** ≤ 1e-6; max score-box violation = **0.000e+00**; noul p strictly in (0,1) in all 50 noul trials (25 raw + 25 calibrated); masked entries exact 0; `predicted_value ∈ C` in every choice trial; calibration actually applied (`applied=True`) |

## Structural claims beyond T1–T8 (PLUS)

| Claim | Specification | Executed test | Status | Measured |
|---|---|---|---|---|
| Masked pooling padding invariance (P4.3) | `MATHEMATICAL_SPECIFICATION.md` §4 | `test_plus_masked_pooling_perturbation_invariance` (masked_mean / last_valid / attention_pool × left/right pad = 6 cases) | **Verified** | perturbing padded hidden values (randn·1e6) → pooled output **bitwise unchanged** (`torch.equal`, diff = 0) in all 6; compact-vs-padded ≤ **2.98e-08** ≤ 1e-6 (summation-tree caveat, Remark 7.2) |
| Fusion masked-option invariance (D5.1 / A7.5) | §5 + §11 (P11.3) | `test_plus_fusion_masked_option_invariance` (all 7 registered fusions: add, additive, bilinear, concat, dot, gated, mlp) | **Verified** | perturbing masked option inputs by ±1e4 → valid logits max diff = **0.000e+00** (tol 1e-7); valid option features bitwise unchanged; masked features exactly 0 |
| Risk–coverage structural properties | §10 (Eqs. 10.2–10.4) | `test_plus_risk_coverage_coverage_monotone_and_car_consistent` (100 random trials × 4 α = 400 checks + deterministic non-monotone example) | **Verified (caveated)** | coverage non-decreasing, equals i/n (max dev ≤ 1e-7); `coverage_at_risk ∈ [0,1]` and curve-consistent to ≤ **1e-6** on 400/400 checks; **risk monotonicity deliberately NOT checked** (example asserts risk 1/3 → 1/4 decrease, per the spec's explicit non-claim) |

## §38 numerical stability (executed)

| Check | Executed test | Status | Measured |
|---|---|---|---|
| Full head stack on ±1e4/±1e6 logits → all outputs finite | `test_full_head_stack_finite_on_extreme_logits[1e4]`, `[1e6]` (Choice/Score/Noul `build_output` + 5 confidence methods, all through real `check_finite`) | **Verified AFTER implementation fix** | all finite; scores in [0.0000, 2.0000] ⊂ [0,3]; confidences in range |
| Temperature scaling of extreme logits finite | `test_temperature_scaling_of_extreme_logits_stays_finite` | **Verified** | fitted T = 454.8778; T=1e-2 on ±1e6 → \|logit\| = 1.000e+08, probs finite, Σ = 1±1e-6 |
| Isotonic/vector fits on near-duplicate logits → no NaN | `test_isotonic_and_vector_fits_on_near_duplicate_logits_no_nan[0.0]`, `[1e-07]` | **Verified** | isotonic: 1 / 360 finite breakpoints, nondecreasing in [0,1]; vector: scale ∈ [0.6465, 0.9306] / [0.6818, 0.9447], finite; applications finite, rows Σ = 1±1e-5 |
| fp16/bf16 inputs → finite, close to fp32 path (documented atol 1e-2: input rounding + output downcast) | `test_low_precision_inputs_finite_close_to_fp32_path[torch.float16]`, `[torch.bfloat16]` | **Verified** | fp16 max \|p−p_fp32\| = **1.528e-04**; bf16 = **3.355e-04** (both ≪ 1e-2) |
| Stable log-softmax vs naive exp formula | `test_stable_log_softmax_vs_naive_exp_formula` | **Verified** | naive: exp(1000) = **inf** → p = **NaN** (both demonstrated); stable: p = [0.244728, 0.090031, 0.665241], finite, Σ = 1; agreement ≤ 1e-6 on moderate logits |

## Implementation bug found and FIXED (flagged loudly)

* **File:** `jevx/decisions/noul.py` — `NoulHead.build_output` (sigmoid line ~71 + class
  docstring, edit applied 2026-09-24).
* **Cause:** raw `torch.sigmoid(logits)` saturates to *exactly* `1.0` for z ≳ 17 and
  exactly `0.0` for z ≲ −17 in binary32 (`1 + e^{−z}` rounds to 1.0), so stored noul
  probabilities violated T4's strict `(0,1)` for every finite fp32-representable logit
  in the saturation regime. Observed on first run: p(±1e3) = {1.0, 0.0},
  p(±1e4) = {1.0, 0.0}, p(±1e6) = {1.0, 0.0}; the same root cause failed the §38
  head-stack test at ±1e4/±1e6 (3 failing tests, first run 13 failed / 28 passed).
* **Fix (minimal):** clamp the stored probability to
  `[finfo(dtype).eps, 1 − finfo(dtype).eps]` inside `build_output` and update the class
  docstring. This is exactly the implementation invariant **sanctioned by
  PROOFS.md Remark 4.1** ("if strict (0,1) membership of stored outputs is required,
  clamps to [ε, 1−ε] with a declared ε"); the induced bias on p is bounded by
  ε = 1.1920929e-07 (fp32). No check was weakened to achieve green: T4 and the §38
  head-stack checks assert strict interiority exactly as specified.
* **Blast radius checked:** no existing test asserts saturation to exact 0/1
  (`test_decisions.py`, `test_property_invariants.py`, e2e/matrix noul checks all use
  moderate logits or inclusive bounds) — full suite green before (727 passed) and after
  (768 passed = 727 + 41 new).
* Other failures in the first run (10) were bugs in *my new tests* (KL construction
  guard too weak, a [1,K] vs [N] shape mismatch, a mask-broadcast mistake, an
  over-strict compact-vs-padded sanity), each fixed on the test side without
  weakening any theorem check.

## What this does NOT verify

1. **Accuracy / calibration quality are empirical, not theorems.** Nothing here measures
   ECE, NLL, Brier, or task accuracy on data; T5/T6 certify *propriety of the scores*
   (the unique truth-telling optimum of the functional), never that a fitted JevX model
   is accurate or calibrated. No benchmark claim is made anywhere in this report.
2. **Only FUNCTIONAL question independence, not statistical independence.** T7(c)/P11.1
   certify graph separation (perturbing q2's schema cannot change q1's output). The
   outputs D_j, D_k remain statistically dependent through the shared state H_S,
   shared parameters, and correlated inputs (P11.2) — no independence of random
   variables is tested or implied.
3. **No fp bit-equality claim.** T7(ii) is reported as a measured tolerance
   (2.235e-08 / 3.725e-08 on this configuration only); batched vs looped results are
   not claimed bit-identical (Remark 7.2 — fp addition is not associative, kernel
   reduction order differs by shape). Cross-process determinism is also not asserted.
4. **No distribution-free calibration guarantee.** Fitting temperature/vector/isotonic
   on a calibration split carries sampling error and distribution assumptions; D8.I is
   a *type/schema-preservation* invariant (verified here), never a statement that
   calibrated probabilities are reliable on unseen data (Def. 8 [STATISTICAL]).
5. **Type safety ≠ semantic correctness** (Remark 2.1, Corollary 8.2): every T2/T8 result
   constrains the *codomain* (the decision lies in the declared type/schema); decisions
   may still be wrong for the task, and confidence values remain deterministic functions
   of logits until separately validated.

## Pytest summaries (verbatim)

`python -m pytest tests/numerical -q`:

```
.........................................                                [100%]
41 passed in 8.48s
```

`python -m pytest -q` (full suite; executed as the same command with stdout redirected
to a file because in-session shell calls are capped at 120 s):

```
........................................................................ [  9%]
........................................................................ [ 18%]
........................................................................ [ 28%]
........................................................................ [ 37%]
..................s..................................................... [ 46%]
........................................................................ [ 56%]
........................................................................ [ 65%]
........................................................................ [ 74%]
........................................................................ [ 84%]
........................................................................ [ 93%]
.................................................                        [100%]
============================== warnings summary ===============================
tests/integration/test_matrix_text.py::test_matrix_text_family[deberta-v2]
  C:\Users\ysnrfd\AppData\Roaming\Python\Python314\site-packages\torch\jit\_script.py:1485: FutureWarning: `torch.jit.script` is not supported in Python 3.14+ and may break. Please switch to `torch.compile` or `torch.export`.
    warnings.warn(

tests/integration/test_matrix_vision_audio.py::test_matrix_vision_audio_family[ast]
  C:\Users\ysnrfd\AppData\Roaming\Python\Python314\site-packages\transformers\audio_utils.py:723: UserWarning: At least one mel filter has all zero values. The value for `num_mel_filters` (128) may be set too high. Or the value for `num_frequency_bins` (257) may be set too low.
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
768 passed, 1 skipped, 2 warnings in 233.31s (0:03:53)
```

(Both pre-existing warnings originate in third-party code (`torch.jit`, transformers
audio filters) and in other suites; the new `tests/numerical` suite emits zero
warnings.)
