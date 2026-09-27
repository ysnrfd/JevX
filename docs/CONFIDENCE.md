# Confidence — implementation and measured results

(A) what the code does (verified in `jevx/confidence/` + tests), (B) what was
*measured* (quoted from [reports/ABLATIONS.md](../reports/ABLATIONS.md) EXP-6).
API: [docs/API.md](API.md) §12; risk–coverage utilities §13. Interpretation is
governed by docs/LIMITATIONS.md §5: **confidence ≠ correctness** — confidence
functionals are deterministic functions of logits and are uninformative until
validated.

## 1. Methods (fact)

Registry (`available_confidence_methods`):
`['energy', 'entropy', 'learned', 'margin', 'max_probability']`. Every method
returns per-question confidence `[B]`. Universal edge rules (module
docstring of `jevx/confidence/base.py`):

* **Single-column (noul) rows: confidence = `max(p, 1 − p) ∈ [0.5, 1]`** —
  the binary pair (p, 1−p) of spec Def. 9(e), so a threshold in (0.5, 1]
  can actually fire for noul decisions. Genuine K=1 *choice* rows (where
  `p == 1` because there is only one option) still yield exactly 1.
  This also matches the ECE/NLL confidence path (`jevx/utils/numerical.py`
  K==1 branch).
* Values lie in `[0, 1]` whenever the measure is mathematically bounded —
  with the one documented exception below (energy before fit).

| Method | Formula | Fit needed | Notes |
|---|---|---|---|
| `max_probability` (default) | `max_k p_k` | no | directly on a probability scale |
| `margin` | `p_(1) − p_(2)` | no | width-1 (or mask leaving 1 option) → 1 |
| `entropy` | `1 − H(p)/log K_eff` | no | `K_eff` = per-row valid option count; masked entries contribute 0; K_eff=1 → 1 |
| `energy` | `E = −logsumexp(logits)`, then `(hi − E)/(hi − lo)` | **yes** — `engine.fit(calibration_logits)` stores `energy_lo/hi` from a **calibration** split | **sign convention (spec 9.4 vs code):** spec 9.4 defines `c_eng = −E = logsumexp(z)` (larger = more confident); the code returns **raw `E = −logsumexp(logits)` before fit** (more negative = more confident; NOT in [0, 1]) **plus a UserWarning**, and only after fit the affine map `(hi − E)/(hi − lo) ∈ [0, 1]` (larger = more confident again). Pre-fit value and spec `c_eng` are exact negatives of each other — both sides document their sign; see spec 9.4. Requires `logits=` (else `ValueError`); degenerate `hi == lo` → 0.5; masked options pushed to `finfo.min` first (they cannot leak into the energy) |
| `learned` | MLP over `[fused, max_prob, H_norm, margin]` → sigmoid | training is caller-owned | requires `fused=` at forward (else `ValueError`) |

Validation (fact): `probabilities` must be a finite 2-D float tensor, and
`option_mask` (when given) must be bool and shape-identical — TypeErrors/
ValueErrors name the tensor (`_validate_probs`). Fit state for energy is scalar
buffers, so it rides the model's regular state_dict (no sidecar needed).

Where it runs (fact): confidence is computed **after** calibration, per
decision-type group, inside `JevXForDecision.forward`; `confidence_threshold`
then turns `confidence < threshold` rows into `abstained=True` with cleared
predicted fields (probabilities untouched).

## 2. Measured — EXP-6 confidence ablation (SINGLE seed 0, n=872)

Same seed-0 test logits; energy fitted on the calibration logits (never test);
`learned` was **not run** (would add a second training stage outside the shared
protocol — recorded as not-run rather than approximated). Metrics:
ECE-of-confidence (15 bins), AURC (mean of the risk array from
`risk_coverage_curve`), coverage at risk 0.1 / 0.2 (`coverage_at_risk`, 0-1
loss). Quoted verbatim from reports/ABLATIONS.md EXP-6 (source JSON
`benchmarks/results/exp6_confidence.json`):

| Confidence | ECE-of-conf ↓ | AURC ↓ | coverage@risk 0.1 ↑ | coverage@risk 0.2 ↑ | mean conf |
|---|---|---|---|---|---|
| **max_probability** | **0.0371** | 0.1837 | 0.1674 | 0.4931 | 0.7383 |
| margin | 0.2356 | 0.1837 | 0.1674 | 0.4931 | 0.4766 |
| entropy | 0.4794 | 0.1837 | 0.1674 | 0.4931 | 0.2327 |
| energy | 0.3433 | 0.2898 | 0.0000 | 0.0700 | 0.3780 |

Rankings by measurement (single seed): ECE-of-confidence —
max_probability → margin → energy → entropy; AURC and coverage@0.1 —
max_probability = margin = **tie** → energy.

## 3. The K=2 tie (observed, with the reason measured)

The exact **three-way tie** in AURC and coverage across max_probability,
margin and entropy is not a rounding artifact (ABLATIONS.md): for **K=2** all
three are strictly monotone transforms of `max(p, 1−p)`, so they induce the
*same ranking* of the 872 examples — and risk–coverage metrics depend only on
the ranking. Their ECE-of-confidence values differ (0.0371 vs 0.2356 vs 0.4794)
because those depend on the confidence *values*, and only max_probability lives
on a probability scale (spec §9: scores are not probabilities of correctness
unless separately calibrated). **A multi-class task is required to separate
these rankings** — no such claim is made here.

**Energy measured strictly worse on this task:** AURC 0.2898 vs 0.1837;
coverage@0.1 = 0.0000 (even the single most energy-confident example exceeded
10% risk) — observed for one affine fit on 1000 calibration rows, single
seed (ABLATIONS.md).

## 4. Risk–coverage utilities (fact)

`risk_coverage_curve(confidences, correct_mask) → (coverage, risk)` — sorts by
confidence descending with a **stable** sort (ties keep original order →
deterministic), coverage `γ_i = (i+1)/n`, risk = empirical error rate among
covered rows. `coverage_at_risk(confidences, correct_mask, max_risk)` returns
the largest coverage whose risk stays ≤ `max_risk` (`max_risk ∉ [0,1]` →
TypeError/ValueError; `n == 0 → 0.0`). Risk monotonicity is a documented
**non-claim** — tested as such (reports/MATHEMATICAL_VERIFICATION.md PLUS
row: risk deliberately not checked; the test asserts a decreasing example).

## 5. Honest interpretation

* At risk ≤ 0.1 the best confidence covered **16.7%** of the test set; at
  risk ≤ 0.2, **49.3%** — bounded by the model's own ~28.8% error rate
  (acc 0.7122), sample estimates on n=872 (ABLATIONS.md).
* Thresholds κ tuned on one split do not transport for free; all
  risk–coverage numbers are sample estimates with sampling error
  (LIMITATIONS §5/§10).
* Nothing here supports "confidence guarantees correctness" — that sentence is
  false and never claimed (LIMITATIONS §5).

## 6. `learned` measured (EXP-6b, appended 2026-09-24 — §2's "not run" note is superseded, not edited)

On 2026-09-24 the `learned` method was trained and evaluated
(`benchmarks/results/exp6b_learned_confidence.json`; full protocol in
reports/ABLATIONS.md **EXP-6b**): the SST-2 EXP-6 pipeline was re-run once
under the identical seed-0 recipe (re-trained heads reproduced the published
seed-0 test accuracy 0.7122), `ConfidenceEngine("learned", fused_dim=128,
hidden_size=32)` — MLP 131→32 (tanh) →1→ sigmoid, 4,257 params — was fit with
BCE + Adam (lr 1e-3, 100 epochs, batch 64, seed 0) against calibration-split
correctness `1[argmax(cal_logits) == cal_label]` on the 1,000-row calibration
split **only** (no early stopping, no tuning, test never fit), and all five
confidence methods were then scored on the **same** test logits, with the four
EXP-6 baseline rows reproduced bit-for-bit (max abs diff 0.0 across all six
fields). Measured on n=872 (SINGLE seed 0): `learned` ECE-of-conf **0.0910**
(second, behind max_probability 0.0371; ahead of margin 0.2356, energy 0.3433,
entropy 0.4794), AURC **0.1983** (worse than the K=2 monotone trio 0.1837,
better than energy 0.2898), coverage@risk 0.1 **0.1571**, coverage@risk 0.2
**0.4484**, mean confidence 0.6554; its in-sample calibration-split
diagnostics (ECE 0.0478, AURC 0.1459) degraded to the test values, i.e. the
fit did not carry over in full — one draw, no selection on test, and no
reading beyond it (LIMITATIONS §5: confidence ≠ correctness).


## 7. `learned` on a SECOND dataset — the §6 ordering does NOT replicate (EXP-6c, appended 2026-09-26)

Section 6's SST-2 row was one draw with one learned-head seed. EXP-6c re-ran the
same protocol on **ag_news** (K = 4) with **five** learned-head seeds
(`benchmarks/results/exp6c_learned_agnews.json`; full protocol and deviations
D1–D4 in reports/ABLATIONS.md **EXP-6c**; the featurized head stack, the
logits+fused collector, the learned trainer and the metrics are imported from
`benchmarks/run_exp6b.py`, so the executed code objects are the reference
implementation's). Test accuracy 0.8127, n = 1,500, all five methods on the
same test logits, `learned` = mean±std over seeds {0,1,2,3,4}:
**ECE-of-conf 0.0375 ± 0.0095** (max_probability 0.0532, margin 0.0750,
entropy 0.0817, energy 0.4639) and **AURC 0.0688 ± 0.0013** (entropy 0.0706,
max_probability 0.0713, margin 0.0724, energy 0.1744),
**coverage@risk 0.1 0.7644 ± 0.0072**; `coverage@risk 0.2` measured 1.0000 for
all five methods and carries no information on this dataset (error rate 0.1873
is already under the 0.2 budget). **The ordering changes, and both halves of
it change:** `learned` went second→first on ECE-of-confidence (5/5 seeds, its
per-seed range [0.0282, 0.0487] lying entirely below max_probability's 0.0532)
and fourth→first on AURC (4/5 seeds; entropy's 0.0706 lies inside the `learned`
per-seed range [0.0675, 0.0708], so that particular margin is not separated by
the 1.29e-03 seed std). The AURC half was structural on SST-2 — at K = 2 the
max_probability/margin/entropy tie is an identity, so only fourth place was
available to `learned` — and the ECE half is an empirical reversal between two
datasets. The measured 5-seed spread (±0.0095 ECE) is 5.7× smaller than the
+0.0539 SST-2 deficit, so seed noise does not account for the difference on the
evidence available; the spread is measured on ag_news and is not transferable to
SST-2 by assumption, and the comparison is asymmetric (5 seeds vs 1), so the
direction is a measured fact and the magnitude is not seed-corrected on the
SST-2 side. `energy` is the last-place outlier on both datasets and both
criteria, which is the part that does replicate. §3's K = 2 tie is a property of
**binary** tasks and does not transfer; nothing here supports "confidence
guarantees correctness" (LIMITATIONS §5).
