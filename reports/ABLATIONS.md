# Ablations — JevX Phases 57–61 (pooling / fusion+question / calibration / confidence)

Companion to `DATASET_BENCHMARKS.md` (same environment, splits, protocol,
provenance block — reproduced there once). Every number is copied from
`benchmarks/results/exp{3,4,5,6}_*.json`, written by executed code on
2026-09-24. Language: *measured* / *observed* only (§76).

**Seed counts are labeled on every table.** EXP-1 headline metrics are
3-seed mean±std; every ablation row below is **SINGLE SEED (seed 0)** — one
training run per configuration, so differences between rows of the same table
are single-sample differences and are reported as such, never as
variance-adjusted claims.

Shared protocol: frozen `google/bert_uncased_L-2_H-128_A-2`
(rev `30b0a37…`), SST-2 rev `8d51e7e…`, train 3000 / cal 1000 / test 872,
heads = `HeadTrainer` + `choice_loss`, 4 epochs, lr 1e-3, batch 64,
featurize-once (one backbone pass per split, poolings derived from the same
sequence output), ECE/MCE with 15 bins. Calibrators fit on the 1000-row
calibration split only; metrics on the 872-row test split only.

---

## EXP-3 — Representation/pooling ablation (§57) — SINGLE SEED (0)

All four poolings are derived from the **same** backbone forward per batch
(`bench_common.featurize`), so the comparison isolates the pooling operator.
Baseline JevX config otherwise: fusion=concat. Rows: uncalibrated metrics plus
temperature-calibrated NLL/ECE for reference.

| Pooling | acc | macro-F1 | NLL ↓ | ECE(15) ↓ | NLL (temp) ↓ | ECE (temp) ↓ |
|---|---|---|---|---|---|---|
| native (BERT pooler) | 0.6491 | 0.6292 | 0.6269 | 0.0329 | 0.6219 | 0.0217 |
| cls | 0.7122 | 0.7112 | 0.5683 | 0.0371 | 0.5654 | 0.0233 |
| **masked_mean** | **0.7144** | **0.7135** | **0.5625** | **0.0255** | **0.5620** | 0.0266 |
| last_valid | 0.6904 | 0.6886 | 0.6128 | 0.0645 | 0.5971 | 0.0375 |

**Winner per column (measured, single seed):**

| Column | Winner | Value |
|---|---|---|
| Accuracy | masked_mean | 0.7144 |
| Macro-F1 | masked_mean | 0.7135 |
| NLL | masked_mean | 0.5625 |
| ECE (uncal) | masked_mean | 0.0255 |
| NLL (temp) | masked_mean | 0.5620 |
| ECE (temp) | native | 0.0217 |

**Observed:** `masked_mean` won 5 of 6 columns; `cls` was the closest runner-up
on accuracy (0.7122, Δ=0.0022 ≈ 2 test examples); the native pooler measured
6.3 points below masked_mean on accuracy (0.6491 vs 0.7144) — the largest
spread in this table. After temperature calibration the ECE ordering shifted
slightly (native 0.0217 < cls 0.0233 < masked_mean 0.0266), a reminder that
ECE depends on both binning and the calibrator state. Single seed: these are
one draw each.

---

## EXP-4 — Fusion ablation + question-conditioning control (§58–59) — SINGLE SEED (0)

Same protocol, `pooling=cls`; each row is its own training run. Param counts
are measured (`count_trainable` over the head stack; fusion-only counts sum
`model.fusion.parameters()`).

| Fusion | acc | macro-F1 | NLL ↓ | ECE ↓ | ECE (temp) ↓ | trainable params | fusion params |
|---|---|---|---|---|---|---|---|
| concat | 0.7122 | 0.7112 | 0.5683 | 0.0371 | 0.0233 | 1,247,361 | 65,920 |
| additive | 0.7122 | 0.7118 | 0.5677 | 0.0447 | 0.0309 | 1,230,849 | 49,408 |
| **gated** | **0.7236** | **0.7236** | **0.5607** | 0.0408 | 0.0475 | 1,263,745 | 82,304 |
| bilinear | 0.7110 | 0.7106 | 0.5723 | **0.0346** | 0.0337 | 1,230,849 | 49,408 |
| *control: concat + zeroed question* | 0.7190 | 0.7190 | 0.5649 | 0.0344 | 0.0361 | (same as concat) | (same as concat) |

**Winner per column (measured, single seed):**

| Column | Winner | Value |
|---|---|---|
| Accuracy | gated | 0.7236 |
| Macro-F1 | gated | 0.7236 |
| NLL | gated | 0.5607 |
| ECE (uncal) | bilinear | 0.0346 |
| ECE (temp) | concat | 0.0233 |
| Fewest params | additive = bilinear | 49,408 fusion params |

**Observed:** the four fusions span 0.7110–0.7236 accuracy (Δ=0.0126 ≈ 11 test
examples) in this single-seed run; `gated` measured highest on accuracy/F1/NLL
while carrying the most parameters (82,304 fusion params vs 49,408 for
additive/bilinear). No winner is asserted beyond this draw.

**Question-conditioning control (§59), measured:** replacing the question
embedding with zeros (same concat fusion, same protocol) measured
**0.7190** accuracy vs **0.7122** with the real question — i.e. in this
single-seed, single-question task the question pathway did **not** produce a
measured gain (Δ=+0.0068 in favor of the control, ≈6 examples). This is
consistent with the task structure: there is exactly **one** question, so the
question encoder contributes a constant vector per example rather than
example-dependent information; with a trained fusion the constant can be
absorbed into offsets. The honest reading: on this benchmark the
question-conditioning machinery is exercised and measured, but this task
cannot show its value — a multi-question benchmark would be needed for that,
and no such claim is made here.

---

## EXP-5 — Calibration ablation (§60) — SINGLE SEED (0)

Same trained heads (seed 0 of the EXP-1 protocol); every calibrator is fitted
on the 1000-row calibration split and evaluated on the 872-row test split.
Reliability-bin tables (15 bins each) are stored in
`exp5_calibration.json`; key bins rendered below.

| Calibrator | acc | NLL ↓ | ECE(15) ↓ | MCE(15) ↓ | Brier ↓ |
|---|---|---|---|---|---|
| none | 0.7122 | 0.5683 | 0.0371 | 0.0719 | 0.3866 |
| **temperature** | 0.7122 | **0.5654** | **0.0233** | **0.0478** | **0.3847** |
| vector | **0.7179** | 0.5685 | 0.0374 | 0.0601 | 0.3873 |
| isotonic | 0.7144 | 0.5716 | 0.0397 | 0.6667 | 0.3888 |

**Winner per column (measured, single seed):**

| Column | Winner | Value |
|---|---|---|
| Accuracy | vector | 0.7179 |
| NLL | temperature | 0.5654 |
| ECE | temperature | 0.0233 |
| MCE | temperature | 0.0478 |
| Brier | temperature | 0.3847 |

**Observed:**
- Temperature preserved accuracy exactly (0.7122 → 0.7122) — argmax
  invariance reproduced (§8.1) — while improving all four probability metrics.
- **Vector** scaling changed decisions (0.7122 → 0.7179 accuracy, +5 examples)
  and **isotonic** changed them too (→ 0.7144): non-order-preserving
  calibrators can move the argmax (§8.2). Both directions are reported; the
  changed decisions are the documented semantics of fitting these families,
  not a silent side effect.
- Isotonic's MCE measured **0.6667** — an order of magnitude worse than every
  other row. The stored reliability bins show why: its coarse step map puts a
  large confidence gap in a small high-confidence bin (observed on this fit;
  n_cal=1000 → few distinct breakpoints; the LIMITATIONS §4 caveat about small
  calibration sets making isotonic fits coarse is visible in the data).

### Reliability key bins (temperature, seed 0, 15 bins; full table in JSON)

| bin | range | n | mean conf | accuracy | gap (acc−conf) |
|---|---|---|---|---|---|
| 7 | 0.467–0.533 | 49 | 0.5181 | 0.5306 | +0.0125 |
| 8 | 0.533–0.600 | 121 | 0.5658 | 0.5537 | −0.0121 |
| 9 | 0.600–0.667 | 158 | 0.6325 | 0.6582 | +0.0257 |
| 10 | 0.667–0.733 | 153 | 0.7000 | 0.7124 | +0.0124 |
| 11 | 0.733–0.800 | 174 | 0.7680 | 0.7414 | −0.0266 |
| 12 | 0.800–0.867 | 129 | 0.8269 | 0.7907 | −0.0362 |
| 13 | 0.867–0.933 | 58 | 0.9048 | 0.9310 | +0.0262 |
| 14 | 0.933–1.000 | 30 | 0.9522 | 1.0000 | +0.0478 |

Counts sum to 872 (all test rows); largest measured bin gap 0.0478 — these
eight populated bins are the whole distribution (bins 0–6 empty: no test
example received confidence < 0.467 after temperature scaling, observed).

---

## EXP-6 — Confidence ablation (§61) — SINGLE SEED (0)

Confidence functionals computed on the same seed-0 test logits;
`energy` is fitted on the calibration logits (never test) per §9's fit
protocol. `learned` was **not run**: it requires training an extra module,
which would introduce a second training stage outside the shared protocol —
recorded as not-run rather than approximated. Metrics: ECE-of-confidence
(confidence vs empirical accuracy, 15 bins), AURC (mean of the risk array
from `risk_coverage_curve`, n=872 coverage points), coverage at risk 0.1 / 0.2
(`coverage_at_risk`, 0-1 loss).

| Confidence | ECE-of-conf ↓ | AURC ↓ | coverage@risk 0.1 ↑ | coverage@risk 0.2 ↑ | mean conf |
|---|---|---|---|---|---|
| **max_probability** | **0.0371** | **0.1837** | **0.1674** | **0.4931** | 0.7383 |
| margin | 0.2356 | 0.1837 | 0.1674 | 0.4931 | 0.4766 |
| entropy | 0.4794 | 0.1837 | 0.1674 | 0.4931 | 0.2327 |
| energy | 0.3433 | 0.2898 | 0.0000 | 0.0700 | 0.3780 |

**Rankings BY MEASUREMENT (single seed):**

| Criterion | Ranking (best → worst) |
|---|---|
| ECE-of-confidence | max_probability (0.0371) → margin (0.2356) → energy (0.3433) → entropy (0.4794) |
| AURC | max_probability = margin = entropy (tie, 0.1837) → energy (0.2898) |
| coverage@0.1 | max_probability = margin = entropy (tie, 0.1674) → energy (0.0000) |

**Observed:**
- The exact three-way **tie** in AURC and coverage across max_probability,
  margin and entropy is not a rounding artifact: for K=2 all three are
  strictly monotone transforms of `max(p, 1−p)`, so they induce the *same*
  ranking of the 872 examples, and risk–coverage metrics depend only on the
  ranking. Their ECE-of-confidence values differ (0.0371 vs 0.2356 vs 0.4794)
  because those metrics depend on the confidence *values*, and only
  max_probability lives on a probability scale (spec §9: scores are not
  probabilities of correctness unless separately calibrated).
- **energy** measured a strictly worse ranking on this task (AURC 0.2898 vs
  0.1837; coverage@0.1 = 0.0000 — even the single most energy-confident
  example exceeded 10% risk): its affine [0,1] map was fit on 1000
  calibration rows and did not order errors as well here (observed, single
  fit).
- Coverage context: at risk ≤ 0.1 the best confidence covers only 16.7% of
  the test set; at risk ≤ 0.2 it covers 49.3%. The base error rate is
  ~28.8% (acc 0.7122), so low-risk coverage is inherently limited by the
  model's accuracy — these are sample estimates on n=872 (§10 caveat).

---

## Seed-count and honesty summary

| Experiment | Seed count | Result file |
|---|---|---|
| EXP-1 (headline) | **3 seeds** (0,1,2), mean±std | `exp1_jevx_quality.json` |
| EXP-2 (B1/B2/B4) | **3 seeds** (0,1,2), mean±std | `exp2_baselines.json` |
| EXP-2 (B3, latency) | n=1 / timed-100 | `exp2_baselines.json` |
| EXP-3 (pooling) | **single seed 0** | `exp3_pooling.json` |
| EXP-4 (fusion + control) | **single seed 0** | `exp4_fusion.json` |
| EXP-5 (calibration) | **single seed 0** | `exp5_calibration.json` |
| EXP-6 (confidence) | **single seed 0** | `exp6_confidence.json` |

Every ablation winner above is a single-draw observation on n=872 test
examples; no statistical-significance claim is made for any row difference,
and none should be read into these tables (§54, §76).

---

## EXP-6b — `learned` confidence, MEASURED (SINGLE seed 0) — appended 2026-09-24

**Append-only section: the EXP-6 text above (which recorded `learned` as
not-run) is left untouched; this section supersedes it with executed
measurements. Source: `benchmarks/results/exp6b_learned_confidence.json`,
written by `python benchmarks/run_exp6b.py` on 2026-09-24.**

### Pairing (stated explicitly)

The full SST-2 pipeline was re-run **once** under the EXP-6 recipe (same
`stanfordnlp/sst2` rev `8d51e7e…`, same seed-0 stratified train 3,000 / cal
1,000 carve, test = full 872-row official validation split, featurize-once
`cls` features, `HeadTrainer` + `choice_loss`, 4 epochs, lr 1e-3, batch 64,
head model built immediately after `torch.manual_seed(0)` exactly as
`run_quality.run_exp6` does). Re-trained heads: train loss 0.6509 → 0.5318,
test accuracy **0.7122** — the same seed-0 accuracy the published EXP-1/EXP-5
rows are built on. **All five confidence methods were then computed from the
same test logits in that single run**, so the comparison is *paired*. As a
check, the four published EXP-6 rows were re-computed from these logits and
compared field-by-field against `exp6_confidence.json`: **max abs diff = 0.0
for all four methods across all six fields** (`pairing_check_vs_exp6` in the
JSON) — the baselines below are bit-identical to the EXP-6 table above.

### Learned-training protocol (fit on CALIBRATION split only — recorded exactly)

* **Inputs** `x = [fused ; max_prob ; 1 − H/log K ; margin]` on the 1,000-row
  calibration split, where `fused` is the 128-d concat-fusion state `H` of the
  trained JevX head stack (computed by the same `state_proj` → question →
  schema → fusion modules `model.forward` uses; a subclass exposes it without
  changing any math — verified by the built-in equivalence check, max-abs
  logit diff 2.38e-6 vs the full `model.choice()` path).
* **Targets** `y = 1[ argmax(cal_logits) == cal_label ]` (calibration-split
  correctness).
* **Module** `ConfidenceEngine("learned", fused_dim=128, hidden_size=32)` —
  the library class **unmodified** (`MLP 131 → 32 (tanh) → 1 → sigmoid`,
  4,257 trainable parameters).
* **Optimization** BCE on the sigmoid output; Adam, lr 1e-3; **100 epochs**,
  batch 64, per-epoch shuffle via `torch.Generator().manual_seed(0)`;
  `torch.manual_seed(0)` before module construction. Per-epoch mean BCE
  0.6416 → 0.5159. Schedule fixed a priori; **no hyperparameter search on any
  split, no early stopping, test never fit or tuned.**

### Results (SINGLE seed 0, n=872 test rows; same metric definitions as EXP-6)

| Confidence | ECE-of-conf ↓ | AURC ↓ | coverage@risk 0.1 ↑ | coverage@risk 0.2 ↑ | mean conf |
|---|---|---|---|---|---|
| **max_probability** | **0.0371** | **0.1837** | **0.1674** | **0.4931** | 0.7383 |
| **learned (this run)** | **0.0910** | 0.1983 | 0.1571 | 0.4484 | 0.6554 |
| margin | 0.2356 | 0.1837 | 0.1674 | 0.4931 | 0.4766 |
| entropy | 0.4794 | 0.1837 | 0.1674 | 0.4931 | 0.2327 |
| energy | 0.3433 | 0.2898 | 0.0000 | 0.0700 | 0.3780 |

(The four non-learned rows are bit-identical to the EXP-6 table — paired
reproduction verified above; `learned` is the only new row.)

**Rankings BY MEASUREMENT (single seed):**

| Criterion | Ranking (best → worst) |
|---|---|
| ECE-of-confidence | max_probability (0.0371) → **learned (0.0910)** → margin (0.2356) → energy (0.3433) → entropy (0.4794) |
| AURC | max_probability = margin = entropy (tie, 0.1837) → **learned (0.1983)** → energy (0.2898) |
| coverage@0.1 | max_probability = margin = entropy (tie, 0.1674) → **learned (0.1571)** → energy (0.0000) |
| coverage@0.2 | max_probability = margin = entropy (tie, 0.4931) → **learned (0.4484)** → energy (0.0700) |

**Observed (measured):**
- `learned` measured **second-best ECE-of-confidence** (0.0910), behind
  max_probability (0.0371) and ahead of margin (0.2356), energy (0.3433) and
  entropy (0.4794) — the only row besides max_probability on a plausibly
  probability-like scale, though its sigmoid output is not a calibrated
  probability of correctness (spec §9, LIMITATIONS §5).
- On every risk–coverage metric `learned` measured **worse than the K=2
  monotone trio** (AURC 0.1983 vs 0.1837; coverage@0.1 0.1571 vs 0.1674;
  coverage@0.2 0.4484 vs 0.4931) and **better than energy on all four**. In
  this draw max_probability measured better than `learned` on all four
  metrics; no claim beyond this draw is made.
- **Cal-to-test gap recorded:** on the calibration split (in-sample for
  `learned`, since it was fit there) its ECE-of-conf measured 0.0478 and AURC
  0.1459; on test these became 0.0910 and 0.1983 — the fit does not carry
  over in full, which is why the cal-side numbers are stored as a diagnostic
  only and never reported as the result.
- This is the second training stage EXP-6 deliberately avoided: the learned
  MLP adds a cal-side fit outside the shared head protocol. It is reported as
  a **separately labeled, single-seed, single-draw** row on n=872; nothing
  here supports "confidence guarantees correctness" (LIMITATIONS §5).


---

## 7. EXP-H: choice-head v2 — causal ablation (APPENDED 2026-09-24)

**APPEND-ONLY.** Nothing above this line is edited. Source:
``benchmarks/run_head_quality.py`` / ``benchmarks/results/head_quality_*.json``.
Head **v1** = as shipped (option interaction only). Head **v2** = additive
state-only linear term + the unchanged interaction residual, zero-initialized
so an untrained head is bit-identical to v1.

### 7.1 What each cell changes (one variable at a time)

| cell | the single thing changed vs the cell above it |
|---|---|
| P1 / P2 | reference probes (B1 / B2), no JevX stack at all |
| P3 / P3b / P3c | JevX v1 head, interaction = dot / bilinear / mlp |
| P4 | question+schema pathway **bypassed** (fixed state-independent option basis) |
| P5 | **state-pathway initialization** = closed-form least-squares probe |
| P6 | head **v2** (the fix) |

### 7.2 Measured causal ranking (ag_news; synthetic control in brackets)

| rank | cell | ag_news acc | synthetic acc | what it isolates |
|---|---|---|---|---|
| 1 | P2 MLP probe | 0.8373 | 0.9616 | — |
| 2 | **P5 probe-initialized state pathway** | **0.8300** | 0.9427 | **initialization scale** |
| 3 | P4 state-only | 0.8287 | **0.9533** | fusion / question noise |
| 4 | P3c v1 mlp | 0.8271 | 0.9469 | interaction parameterization |
| 5 | P1 linear probe | 0.8262 | 0.9382 | — |
| 6 | **P6 head v2 (the fix)** | 0.8258 | **0.9482** | missing additive path |
| 7 | P3 v1 dot | 0.8249 | 0.9427 | interaction parameterization |
| 8 | P3b v1 bilinear (shipped) | 0.8233 | 0.9476 | interaction parameterization |

### 7.3 The ablation verdict

- **The option-interaction parameterization is NOT the cause.** P4 (same
  interaction, question/schema removed) *beats* P3b on ag_news, and the
  synthetic control — which has no text, no question and no schema at all —
  still shows the same ordering, so nothing about the question pathway is
  responsible.
- **Initialization scale IS a cause.** P5 changes *only* the starting values
  of ``state_proj`` and is the best JevX cell on ag_news (0.8300, +0.67 pt over
  the shipped head) with 3x lower seed variance (std 0.0047 vs 0.0133).
- **The missing additive state-only path IS a real structural gap**, and v2
  fixes it: on the synthetic control v2 (0.9482) beats v1 (0.9476) and the
  linear probe (0.9382) with the best JevX NLL (0.1641 vs 0.1742).
- **v2 does not close the gap on ag_news** (0.8258 vs P1 0.8262, P2 0.8373).
  A zero-initialized additive pathway must still learn its scale within a
  4-epoch / lr-1e-3 budget, which is precisely the constraint P5 removes by
  starting from the closed-form solution. Recorded as a limitation.
- **Headroom remains.** P4 on the synthetic control (0.9533) is well above
  every full-stack JevX cell, so the fusion + question pathway still costs
  real accuracy even after the head fix.


### 7.4 The audio cell contradicts the text diagnosis (recorded, not smoothed over)

Running the same grid on RAVDESS (`head_quality_ravdess.json`, EXP-A protocol,
frozen wav2vec2 features) gives a **different ordering**, and it is the more
important result of the two:

| cell | RAVDESS acc |
|---|---|
| P1 linear probe | 0.3467 |
| P2 MLP probe | 0.3547 |
| P3b JevX v1, bilinear (shipped) | 0.1680 |
| **P4 JevX v1, question+schema BYPASSED** | **0.2407** |
| P5 JevX v1, probe-initialized | 0.1380 |
| P6 JevX v2 (the fix) | 0.1353 |

- P4 is again the best JevX cell, by **+7.3 pt** over the shipped head:
  removing ONLY the question/schema pathway rescues a degenerate head.
- P5 does **not** help here (0.1380), so initialization scale is not the
  binding constraint on audio.
- **Head v2 does not repair EXP-A** (0.1353 vs 0.1680 shipped — inside seed
  noise). This is a negative result and is reported as one.

**Cross-dataset conclusion:** the head's missing additive path is real and
is fixed (proven by the equivalence test, and measured on the synthetic
control), but it is **not** what makes the shipped head lose on the real
benchmarks. On text the deficit is dominated by state-pathway initialization
scale; on audio it is dominated by the option/schema pathway, whose option
rows are bare class names hashed into an embedding table and then mixed into
the state by the fusion option gate. The next owner of ``jevx/questions/`` /
``jevx/representations/fusion.py`` should attack the option pathway, not the head.

---

## 8. EXP-H follow-up: the option-pathway SCALE defect (APPENDED 2026-09-25)

**APPEND-ONLY.** Nothing above this line is edited. Source:
`benchmarks/run_head_quality.py` (cells P7-P15 added), results under
`benchmarks/results/expH/`. The finding closed here is the one 7.4 left open:
on AUDIO the damage comes from the **option/schema pathway**, and the next step
was named explicitly (attack the option pathway, not the head).

### 8.1 The gap, reproduced first (3 seeds, cached features, no re-featurization)

A full re-run of the P1-P6 grid on RAVDESS reproduced the published numbers
**exactly**, cell for cell and seed for seed
(`benchmarks/results/expH/head_quality_ravdess.json`):

| cell | acc | macro-F1 | NLL | per-seed |
|---|---|---|---|---|
| P1 linear probe | 0.3467 +/- 0.0181 | 0.2971 | 1.9176 | 0.3540 / 0.3260 / 0.3600 |
| P2 MLP probe | 0.3547 +/- 0.0258 | 0.2993 | 1.7364 | 0.3760 / 0.3260 / 0.3620 |
| P3 JevX v1, dot | 0.1100 +/- 0.0381 | 0.0266 | 2.3303 | 0.1320 / 0.1320 / 0.0660 |
| **P3b JevX v1, bilinear (shipped head)** | **0.1680 +/- 0.0360** | 0.0680 | 2.0337 | 0.1320 / 0.2040 / 0.1680 |
| P3c JevX v1, mlp | 0.1333 +/- 0.0012 | 0.0295 | 2.0760 | 0.1340 / 0.1320 / 0.1340 |
| **P4 question+schema pathway BYPASSED** | **0.2407 +/- 0.0535** | 0.1794 | 2.1580 | 0.2260 / 0.1960 / 0.3000 |
| P5 probe-initialized state pathway | 0.1380 +/- 0.0087 | 0.0375 | 2.2759 | 0.1340 / 0.1320 / 0.1480 |
| P6 JevX v2 (state-linear probe pathway) | 0.1353 +/- 0.0058 | 0.0339 | 2.3097 | 0.1420 / 0.1320 / 0.1320 |

**P4 - P3b = +7.27 pt** and **P4 - P6 = +10.54 pt**; the majority-class rate of the
test split is 0.1340 and chance is 0.1250, so every full-stack cell except P3b sits
on the floor. The effect size is confirmed before anything was changed.

### 8.2 What the diagnosis actually is (measured, not inferred)

The one number that separates the cells is the **epoch-1 train loss**
(chance = ln 8 = 2.079):

| cell | train loss, first epoch | last epoch |
|---|---|---|
| P1 linear probe | 2.06-2.07 | 1.92 |
| P4 option pathway bypassed | 6.92-8.36 | 1.91-2.28 |
| P3c v1 mlp | 2.14-2.19 | 2.07-2.08 |
| **P3b v1 bilinear (shipped)** | **178.2-225.9** | 2.25-2.43 |
| P6 v2 | 178.3-226.0 | 2.23-2.41 |

The full-stack cells **explode at step 0 and collapse back to the uniform
distribution**. Instrumenting the initialization directly (seed 0, 768-d frozen
wav2vec2 features, one 64-row batch) localizes it to the option pathway:

| quantity (at initialization) | value |
|---|---|
| `o_k` norm (schema row after `schema_proj`) | 26.64 |
| `c` norm (fused context) | 19.68 |
| `e_k` norm (option feature) | 12.64 |
| across-k (option-discriminative) std of `e_k` | 0.258 |
| across-batch (state-leak) std of `e_k` | 0.077 |
| gate `g = sigmoid(W_g c + b_g)` | mean 0.503, no coordinate < 0.1 |
| **head logits max abs z, std z** | **21.65, 4.28** (uniform NLL 2.079) |
| first-batch gradient L2, by module | question_proj **1115**, question_encoder **675**, fusion **534**, schema_proj **353**, schema_encoder **334**, state_proj **300**, **heads 204** |

`Projection` ends every option row with a `LayerNorm`, so every option arrives at
norm `sqrt(D) = 27.7` **whatever the option says** - and those rows are
*learnable*, so the optimizer can inflate them. The head's option-interaction
residual `r_k = s^T W_b e_k` is a **product of two full-scale 768-d vectors**, so
it starts saturated (`std(z) = 4.28` against a reference of 2.079), cross-entropy
gradients vanish, and the module that actually has to classify - the head -
receives the **smallest** gradient in the whole stack (204 against 1115 for the
question pathway). On RAVDESS the option rows are also bare class *names* hashed
into an embedding table, so the branch carries almost no information but a
full-size magnitude: information-poor, magnitude-rich. **That is the defect: the
option row was carrying a magnitude where only an identity belongs.**

### 8.3 The candidate table

Every cell = the shipped v2 head + the full question/schema stack, ONE variable
changed per row, 3 seeds, identical protocol. Candidates were run as a *patch on
the option formula only*, on top of the then-shipped code, so each row is a
one-variable comparison against P15 (0.1353). Artifacts:
`expH/head_quality_*.json` (P7, P9, P10, P12, P13, P14), `expH/gaps/` (P8, P11,
P3b), `expH/landed/` (P13/P15 on the LANDED library).

| cell | the single change | RAVDESS | ag_news | synthetic |
|---|---|---|---|---|
| P1 / P2 | reference probes (no JevX stack) | 0.3467 / 0.3547 | 0.8262 / 0.8373 | 0.9382 / 0.9616 |
| P4 | question+schema pathway bypassed | 0.2407 | 0.8287 | 0.9533 |
| **P15 = P6 = before the fix** | shipped `e_k = g * (W_o o_k) + (1-g) * c` | **0.1353** | **0.8258** | **0.9482** |
| P7 (candidate a) | `W_o, b_o` zero-initialized (neutral option pathway) | 0.1107 | 0.8209 | 0.9487 |
| P11 (a + b2) | zero-init + LayerNorm on `e_k` | 0.1413 | 0.8231 | 0.9496 |
| P12 (candidate a, at the residual) | learnable gain `alpha`, `alpha(0)=0`, on `r_k` | 0.2107 | 0.8224 | 0.9473 |
| P8 (candidate b1) | LayerNorm on the projected rows `W_o o_k` | 0.1473 | 0.8271 | 0.9516 |
| P9 (candidate b2) | LayerNorm on the fused row `e_k` | 0.1113 | 0.8260 | 0.9491 |
| P10 (candidate c) | `Dropout(0.5)` on the option branch, train only | 0.1453 | 0.8193 | 0.9298 |
| P14 | `e_k = W_o o_k` (option row fully state-independent) | 0.1360 | 0.8284 | 0.9502 |
| **P13 (candidate b3) - LANDED** | **`o_k / max(norm(o_k), eps)` before `W_o`** | **0.2220** | **0.8284** | **0.9462** |

Mean +/- sample std over seeds {0,1,2} are in the JSONs; the table gives the means
for readability. P13's std on ag_news is 0.0126 against P15's 0.0200.

### 8.4 Why the obvious candidates failed, and why this one worked

* **(a) zero-initialized option projection (P7) is DEGENERATE, not merely
  ineffective - 0.1107, *below* the 0.1340 majority rate.** With `W_o = 0` every
  `e_k` collapses to the same `(1 - g) * c`, so all K options become bit-identical
  duplicates; `ChoiceHead._tie_duplicate_options` then averages all K columns, so
  `dL/dz_k = 0` for every k and the gradient to **every** parameter is exactly
  zero - the model can never leave the uniform distribution. Measured directly:
  first-batch NLL `2.0794 = ln 8` exactly, first-batch gradient L2 `0.0` for the
  head and for the option path. Neutrality at the FEATURE level is therefore
  self-contradictory: an uninformative option pathway is exactly one whose
  options are indistinguishable. This is a structural result, not a tuning
  failure, and it is pinned by
  `tests/unit/test_option_pathway.py::test_neutral_option_pathway_collapses_to_the_duplicate_tie`.
* **(a) moved to the residual (P12) works - 0.2107, +7.5 pt - but was NOT
  shipped**: it needs a head change, the gain is one scalar, and it works by
  *suppressing* the option pathway for the whole budget (`alpha` grows about lr
  per step, so after 44 steps it is about 0.04 and the residual stays near zero).
  Its RAVDESS NLL is 2.9019 - **worse** than the 2.3097 it was meant to fix - so
  it buys accuracy by making the head *less* calibrated. Recorded as measured.
* **(b) LayerNorm (P8/P9) makes the saturation WORSE, and the measurement says
  why**: `o_k` is already at the LayerNorm scale, so normalizing again raises it.
  Init-time `max abs z`: shipped **21.65** -> P8 **30.53** -> P9 **49.14**;
  first-batch NLL 8.29 -> 14.64 -> 16.92; P9's RAVDESS NLL is 4.78. The brief's
  candidate (b) is therefore *rejected as literally specified* (a LayerNorm on the
  option features) and is replaced by its scale-free variant.
* **(c) dropout (P10) is within seed noise on audio (+1.0 pt, std 0.0196) and
  clearly negative on the synthetic control** (0.9298 vs 0.9482, -1.8 pt, the
  worst cell in the table) and on ag_news (0.8193). Rejected.
* **(b3) the unit-direction option row (P13) is the winner**: it removes exactly
  the magnitude 8.2 identified as the defect, keeps the gate, the context
  fallback, the cross-attention shape and every invariant, and is a strict
  *scale* superset (every direction stays reachable and `W_o` can still grow the
  branch). The option branch's contribution drops from norm ~12.6 to ~0.3, and the
  shipped head's first-epoch loss drops from 178-239 into the P4 range.

### 8.5 Before -> after, on the LANDED library (3 seeds each)

`P15` = `option_normalize=False` (the pre-fix formula, reproduced from the shipped
flag, not a re-implementation) and `P13` = the shipped `option_normalize=True`.
Cells `p13_unit_option` / `p15_option_unnormalized` in
`benchmarks/results/expH/landed/head_quality_*.json`; P15 reproduces the
published pre-fix numbers exactly (RAVDESS 0.1353 +/- 0.0058, NLL 2.3097).

| dataset | acc before | acc after | delta acc | F1 before -> after | NLL before -> after |
|---|---|---|---|---|---|
| **RAVDESS (audio, 8-way)** | 0.1353 +/- 0.0058 | **0.2220 +/- 0.0442** | **+8.67 pt** | 0.0339 -> 0.1255 | 2.3097 -> **2.0787** |
| ag_news (text, 4-way) | 0.8258 +/- 0.0200 | 0.8284 +/- 0.0126 | +0.26 pt | 0.8229 -> 0.8263 | 0.5092 -> 0.5001 |
| synthetic control | 0.9482 +/- 0.0034 | 0.9462 +/- 0.0021 | -0.20 pt | 0.9483 -> 0.9461 | 0.1641 -> 0.1796 |

RAVDESS references on the same features: P1 0.3467, P2 0.3547, P4 (option pathway
**removed**) 0.2407, majority 0.1340, chance 0.1250. The fix recovers **+8.67 pt**
- more than the +7.27 pt the bypass recovered - and reaches a **lower NLL than
the bypass** (2.0787 vs 2.1580) and than the shipped head (2.3097).

One further measured consequence, not a claim about the shipped default: with the
option pathway normalized, the **v1** head (P3b, `state_pathway=False`) goes from
0.1680 to **0.2520 +/- 0.0521** on RAVDESS - i.e. on audio the additive state
pathway of head v2 is no longer what is missing; the option pathway was. The
shipped default keeps v2 (0.2220). Both are reported; neither was re-tuned.

### 8.6 Verdict, stated without smoothing

* **Audio closed.** The option/schema pathway was the cause and it is fixed:
  +8.67 pt and -0.23 NLL on RAVDESS under one identical protocol, 3 seeds, with
  the bypass (P4) exceeded and its NLL beaten.
* **Text did not regress.** ag_news +0.26 pt with seed std falling from 0.0200 to
  0.0126, and NLL improving - both inside seed noise, i.e. neutral.
* **The synthetic control moved -0.20 pt** (0.9482 -> 0.9462; seed std 0.0021),
  with NLL 0.1641 -> 0.1796. That is a small, honestly-reported cost, about one
  seed std on accuracy, and it is the price of the fix. It is not claimed away.
* **Two candidates from the brief's space are rejected by measurement**
  (feature-level zero-init is provably degenerate; LayerNorm-on-option-features
  makes the saturation worse), one is a wash (dropout), one works but was not
  shipped because it needs a head change and degrades NLL (residual gain). The
  shipped design is the scale-free variant of the same idea: the option row
  carries **identity, not magnitude**.


## 9. JEVX-NEXT ABLATION MATRIX: nonlinear state path x option residual x gate x curriculum (APPENDED 2026-09-26)

Full report with every per-seed number, the convergence series, the
per-module gradient norms, the gate table and the latency table:
`reports/HEAD_ABLATION_MATRIX.md`. Written by `benchmarks/run_head_ablation.py`
on 2026-09-26. **9 cells x 4 datasets x 5 seeds = 180 cell-runs, 0 failures,
2062.3 s (34 min 22 s)**, one identical protocol per cell (same frozen
features, same splits, same seeds, same AdamW/lr/4 epochs/batch 64/choice_loss).
Cells: `A` linear+interaction, `B` mlp + interaction **OFF**, `C` mlp +
interaction, `D` C+gate, `E` C+curriculum (2+2 epochs), `F` C+gate+curriculum;
references `P1` linear probe, `P2` MLP probe, `P4` state-only bypass.
"Interaction OFF" is enforced explicitly twice (`config.conditioning_enabled
= False` AND `set_interaction_scale(0.0)`, both asserted) — never by relying on
a config default, since a fresh model maps `conditioning_scale_init == 0.0` onto
a head lambda of 1.0 by design.

### 9.1 Test accuracy, 5 seeds (mean±std); P1/P2 in the same table

| cell | synthetic (D=128,K=4) | ag_news (D=128,K=4) | sst2 (D=128,K=2) | ravdess (D=768,K=8) |
|---|---|---|---|---|
| P1 linear | 0.9384±0.0021 | 0.8255±0.0033 | **0.7000±0.0046** | 0.3472±0.0162 |
| **P2 MLP** | **0.9616±0.0022** | **0.8361±0.0031** | 0.6966±0.0164 | **0.3632±0.0217** |
| P4 bypass | 0.9523±0.0035 | 0.8316±0.0149 | 0.6975±0.0184 | 0.2720±0.0572 |
| A linear+int | 0.9445±0.0031 | 0.8327±0.0108 | 0.6929±0.0256 | 0.2104±0.0351 |
| B mlp, no int | 0.9505±0.0035 | 0.8328±0.0082 | 0.7014±0.0143 | 0.3072±0.0511 |
| C mlp+int | 0.9453±0.0071 | 0.8313±0.0123 | 0.7000±0.0122 | 0.2376±0.0274 |
| D +gate | 0.9453±0.0071 | 0.8313±0.0123 | 0.7002±0.0127 | 0.2376±0.0274 |
| E +curr | 0.9508±0.0023 | 0.8321±0.0062 | 0.7007±0.0135 | 0.2812±0.0579 |
| F +both | 0.9497±0.0026 | 0.8332±0.0081 | 0.7007±0.0135 | 0.3068±0.0511 |

Paired per-seed deltas (same features, same seeds; "k/5" = seeds the left cell wins):

| comparison | synthetic | ag_news | sst2 | ravdess |
|---|---|---|---|---|
| B − P2 (mlp state vs MLP probe) | **−1.11 pt, 0/5** | −0.33 pt, 1/5 | +0.48 pt, 2/5 | **−5.60 pt, 0/5** |
| C − B (does the option residual add information?) | −0.52 pt, 1/5 | −0.15 pt, 2/5 | −0.14 pt, 1/5 | **−6.96 pt, 0/5** |
| D − C (gate) | 0.00, 0/5 | 0.00, 0/5 | +0.02, 1/5 | 0.00, 0/5 |
| E − C (curriculum) | +0.55, 3/5 | +0.08, 2/5 | +0.07, 1/5 | **+4.36, 4/5** |
| F − C (gate+curriculum) | +0.44, 3/5 | +0.19, 3/5 | +0.07, 1/5 | **+6.92, 5/5** (NLL 5/5) |
| F − E | −0.11, 0/5 | +0.11, 3/5 | 0.00, 0/5 | **+2.56, 4/5** (NLL 5/5) |
| A − C (linear vs mlp state) | −0.08, 1/5 | +0.13, 3/5 | −0.71, 1/5 | −2.72, 1/5 |

### 9.2 The three questions, answered with the numbers

1. **Is B ≈ P2? No.** The MLP probe is ahead of the MLP-state-pathway head on
   **all 5 seeds** on synthetic (−1.11 pt) and on ravdess (−5.60 pt); on ag_news
   they are within one seed std; on sst2 the MLP probe does not separate from
   the *linear* probe either (P1 0.7000 vs P2 0.6966), so that cell is a null.
   The missing nonlinear state pathway is therefore **not** the dominant factor.
   What the numbers implicate instead: B's state pathway is 49 536 params
   (D=128) / 787 456 (D=768) against P2's 17 028 / 596 744, and B's whole
   decision stack is **76x / 24x** P2's budget with 98.8% / 95.8% of it in the
   question encoder, schema encoder and fusion — modules that encode a quantity
   constant for a whole dataset and are refit from scratch in 4 epochs. The
   signature is the first-epoch train loss on ravdess: 3.2286 (C) and 6.8857 (A)
   against 2.0642 (B) and 2.0451 (P2).
2. **Is C > B? No — C ≤ B on all four datasets.** With one fixed question and
   one fixed schema every option row `e_k` is constant across batch rows, so
   `s^T A e_k` is a constant vector and the residual only reparameterises the
   per-class weight matrix. Measured cost: **−6.96 pt and +0.1083 NLL on
   ravdess (0/5 seeds)**, −0.52 to −0.14 pt elsewhere. The one place it pays is
   likelihood on the synthetic control (NLL 0.1799 vs 0.1860, better on 3/5),
   which is invisible at the argmax. A gain only where options are informative
   would be a valid result; here the options carry nothing the state path does
   not already have.
3. **D > C and F vs E/C?** The gate is a **no-op on 3 of 4 datasets**: D's
   per-seed accuracies are bit-identical to C's on synthetic, ag_news and
   ravdess (gate flips 0/5 there, all `conditioned`, because on a model whose
   residual was *trained* ΔNLL is −0.09…−0.25, i.e. conditioning does help on
   calibration). Gate+curriculum reaches **F − C = +6.92 pt and −0.1086 NLL on
   ravdess, 5/5 seeds on both**, and **+0.19 pt / −0.0109 NLL on ag_news
   (3/5)**; the three text/synthetic deltas otherwise sit inside one seed std
   and flip sign across seeds, so no winner is declared from them. Flip
   frequency **D 2/20, F 12/20**, measured on the calibration split only (τ by
   the one-standard-error protocol on the same split, `split_name="calibration"`
   passes the shipped no-test-leakage guard); the test split is scored only
   after `apply_gate` writes the verdict.

### 9.3 Optimization-inefficiency evidence (gradient norms, seed 0, epoch 1 → final)

Group L2 norm, max over the epoch's steps, un-clipped, pre-`optimizer.step`:

| cell | dataset | state_pathway | option_interaction | schema | fusion | option_pathway | ratio state:option |
|---|---|---|---|---|---|---|---|
| A linear+int | synthetic | 0.278 → 0.445 | 11.638 → 11.559 | 18.610 → 18.611 | 5.938 → 6.446 | 22.738 → 22.837 | 1 : 82 |
| A linear+int | ag_news | 0.238 → 0.367 | 11.471 → 11.350 | 18.606 → 18.589 | 5.575 → 5.859 | 22.558 → 22.555 | 1 : 95 |
| A linear+int | sst2 | 0.114 → 0.161 | 11.361 → 11.275 | 18.607 → 18.597 | 5.335 → 5.504 | 22.445 → 22.434 | 1 : 196 |
| A linear+int | ravdess | 0.231 → 0.318 | 27.739 → 27.712 | 47.658 → 47.757 | 30.987 → 31.381 | 63.253 → 63.510 | 1 : 274 |
| B mlp, no int | ravdess | 16.144 → 16.543 | - | - | 21.911 → 22.371 | 21.911 → 22.371 | 1 : 1.3 |
| C mlp+int | ravdess | 16.164 → 16.458 | 27.766 → 27.820 | 47.666 → 47.801 | 31.027 → 31.551 | 63.290 → 63.674 | 1 : 3.8 |

The **linear** state pathway is starved by two orders of magnitude and does not
grow over the budget — the hypothesis' "joint-optimization inefficiency" is
confirmed *as a description of that pathway*. It is not the binding constraint:
cell B, which has the healthiest state gradient in the table, is still 5.60 pt
below P2 on ravdess.

### 9.4 Parameter budget and latency

| | D=128 (K=4) | D=768 (K=8) |
|---|---|---|
| P1 / P2 trainable | 516 / 17 028 | 6 152 / 596 744 |
| state pathway linear → mlp | 33 024 → 49 536 | 196 864 → 787 456 |
| interaction (bilinear) | 16 384 | 589 824 |
| schema / question / fusion | 574 080 / 574 080 / 65 920 | 4 919 040 / 4 919 040 / 2 361 600 |
| **stack total, A/P4 → B..F** | 1 280 385 → **1 296 897** | 13 579 265 → **14 169 857** |
| option_pathway share | 50.6% | 55.6% |
| mlp-state extra cost as share of stack | 1.3% | 4.2% |

Batch-1 latency, 20 warmup + 120 timed iters, seed 0 (pooled features
precomputed, so the frozen backbone is **not** in these numbers):

| cell | synthetic e2e / head-only | ag_news e2e / head-only | sst2 e2e / head-only | ravdess e2e / head-only |
|---|---|---|---|---|
| P1 | 0.053 / - | 0.030 / - | 0.034 / - | 0.032 / - |
| P2 | 0.146 / - | 0.222 / - | 0.110 / - | 0.312 / - |
| A | 4.999 / 1.069 | 4.444 / 1.006 | 6.543 / 1.734 | 9.774 / 2.403 |
| B | 4.594 / 1.324 | 4.182 / 1.105 | 5.606 / 2.037 | 7.550 / 2.077 |
| C | 5.136 / 1.160 | 4.969 / 1.123 | 4.864 / 1.425 | 9.500 / 2.149 |
| D | 4.931 / 1.146 | 4.815 / 1.118 | 3.765 / 1.132 | 14.823 / 3.007 |
| E | 4.823 / 1.126 | 4.871 / 1.128 | 4.543 / 1.085 | 11.685 / 2.704 |
| F | 4.124 / 1.109 | 4.057 / 1.264 | 4.317 / 1.056 | 9.350 / 2.017 |

ms/example. The stack costs 23x–35x the MLP probe at D=128 and 23x–30x at
D=768. Per-iteration std is comparable to the mean on this 4-core box, so these
are an order of magnitude, not a ranking between cells. sst2's P4 (10.827±18.874)
is the largest-variance reading in the matrix and is reported as measured.

### 9.5 Two findings about the landed code

* **No state-only defect.** On all four datasets the state-only forward
  (`condition_options=False`) leaves **0** rows with all K logits equal (spread
  means 3.144 / 1.773 / 1.311 / 0.792 on synthetic / ag_news / sst2 / ravdess),
  and the two routes to the state-only scoring function — the per-call flag and
  `set_interaction_scale(0.0)` — agree to **max abs diff 0.0**. The degenerate
  duplicate-option tie is confirmed fixed on the real stack. The obsolete
  "real-wrapper Stage A is degenerate" caveat in
  `tests/unit/test_curriculum.py` has been replaced with a real assertion that
  Stage A's state-only decisions discriminate the options; the assertion was
  mutation-checked (re-arming the tie in-process makes it fail).
* **`run_curriculum`'s Stage B does not re-enable the option pathway**
  (reported to the owner; the file is outside this task's ownership).
  `_CurriculumModelView.condition_options` is assigned once at construction from
  `stage_a_epochs < 1` (`curriculum.py:647`, `:1001`) and never updated per
  stage, so with any `stage_a_epochs >= 1` the model is called
  `condition_options=False` in **both** stages and the per-call flag overrides
  the λ the ramp writes into the head. Measured three ways: the ramp records
  `0→0→0.48→1.00` and `get_interaction_scale()` reads 1.0; the option
  pathway's grad norm is absent from E's and F's epoch-1 *and* final buckets;
  and after a full 2+2 curriculum `bilinear_weight` is bit-for-bit unchanged
  (`max|Δ| = 0.0`, `grad is None`), as is every `schema_encoder` gradient.
  Consequence: cells E and F are "state-only for all four epochs plus an inert
  ramp", E is a near-replicate of B (synthetic 0.9508 vs 0.9505, sst2 0.7007 vs
  0.7014), and the gate's high flip rate on F (12/20) versus D (2/20) is a
  consequence of scoring through an *untrained* residual. The hypothesis' claim
  that the option pathway needs a staged ramp is therefore **untested** by this
  matrix.

### 9.6 Verdict, stated without smoothing

* **The JevX-Next configuration does not reach the MLP probe on three of four
  datasets** (synthetic −1.08 pt, ag_news −0.29 pt, ravdess −5.64 pt against the
  best JevX-Next cell in each) and on the fourth (sst2) the MLP probe does not
  separate from the linear probe either, so there is no probe gap there.
* **Neither named cause is the dominant one.** The nonlinear state pathway is
  real but small (1.3%–4.2% of the stack's parameters, and the MLP variant is
  already within noise of the linear one on ag_news/sst2). The interaction
  residual carries no information on top of the state path on any of these four
  datasets and costs up to 6.96 pt. The dominant cost is the encoder + fusion
  bulk of the stack relative to a probe of the same features.
* **What did pay, on the data:** the adaptive gate's *detection* — 12/20 flips,
  +2.56 pt and −0.1001 NLL on ravdess (4/5 and 5/5 seeds), and never worse than
  −0.11 pt anywhere. The curriculum cannot be credited from this run (§9.5).
* No ranking is declared between cells whose per-seed ranges overlap, and no
  claim is made about any dataset, backbone or task outside the four executed.

---

## 10. Curriculum Stage-B defect FIXED, and cells E/F re-run (appended 2026-09-26)

The §9.5 finding — `run_curriculum`’s Stage B never re-enabled the option
pathway — has been fixed in `jevx/training/curriculum.py` and cells **E** and
**F** have been re-executed. Full tables, protocol and per-seed numbers:
`reports/HEAD_ABLATION_MATRIX.md` §11 and
`benchmarks/results/head_ablation_curriculum_fixed.json`. This section is the
summary; the other report is the source.

### 10.1 The fix, in one paragraph

`_CurriculumModelView.condition_options` was a construction-time constant
(`curriculum.py:647`, `:1001` at the time), so with any `stage_a_epochs >= 1`
every Stage-B step still called the model with `condition_options=False` and
`DecisionHead._combine` dropped the residual whatever λ the ramp wrote into the
buffer. `_StageController` now owns the flag next to the scale
(`curriculum.py:889` `condition_options`, `:900` `apply_stage`), writes it into
the view through `_CurriculumModelView.set_condition_options()` (`:670`) once
per epoch (`:916`) and before every training step (`:925`), and
`EpochRecord.condition_options` (`:398`) records the flag each epoch ran under,
read back off the view. No public signature changed. Four regression tests were
added (`tests/unit/test_curriculum.py:1400`, `:1513`, `:1561`, `:1608`) and all
four were re-run against the defect restored in place: all four fail.

### 10.2 Old (defective) vs new (fixed), test accuracy, 5 seeds, mean±std

| dataset | E old | E **new** | F old | F **new** |
|---|---|---|---|---|
| synthetic | 0.9508±0.0023 | 0.9500±0.0024 | 0.9497±0.0026 | 0.9505±0.0024 |
| ag_news | 0.8321±0.0062 | 0.8316±0.0080 | 0.8332±0.0081 | 0.8319±0.0082 |
| sst2 | 0.7007±0.0135 | 0.7014±0.0138 | 0.7007±0.0135 | 0.7011±0.0142 |
| **ravdess** | **0.2812±0.0579** | **0.3280±0.0364** | 0.3068±0.0511 | 0.3268±0.0360 |

The option pathway is now measurably in the graph: on ravdess the final-epoch
`option_interaction` / `schema` gradient norms are **27.783 / 47.697** in the
fixed E/F buckets against **absent** in the superseded ones, and the epoch-1
buckets are unchanged (Stage A is state-only in both).

### 10.3 The curriculum question, answered with the fixed run

* **vs C** (mlp + interaction, no curriculum): **+9.04 pt (E) / +8.92 pt (F) on
  ravdess, 5/5 seeds, non-overlapping 5-seed ranges**; +0.47 / +0.52 (synthetic),
  +0.03 / +0.05 (ag_news), +0.14 / +0.11 (sst2) — all inside one seed std with
  overlapping ranges and sign flips across seeds, so no effect is declared there.
* **vs B** (mlp, no interaction): a dead heat on synthetic, ag_news and sst2
  (|Δ| ≤ 0.12 pt, overlapping ranges); on ravdess **+2.08 pt (E, 3/5) and
  +1.96 pt (F, 2/5)** with overlapping ranges — a mean, not a seed-consistent
  win. The curriculum recovers most of what the interaction costs on ravdess
  (C is 6.96 pt below B) and does not pass B there.
* **It does not hurt anywhere measured.** Worst paired delta: ag_news E − B
  **−0.12 pt** (1/5); worst NLL: synthetic E − C **+0.0042** (worse). One number
  moved the wrong way and is stated: ravdess E’s mean **train** loss rose
  1.7076 → 1.7756 while its test accuracy rose 0.2812 → 0.3280 and its test NLL
  fell 1.8638 → 1.7517.
* **Gate flips on F: 12/20 → 9/20**, and on ravdess 5/5 → 1/5, because the
  "conditioned" mode is now scoring through a *trained* residual. The §9.5
  reading that F flipped often *because* its residual was untrained no longer
  holds.

**§9.6’s "the curriculum cannot be credited from this run" is superseded.** The
staged-ramp claim is now tested, and it pays on one dataset of the four. Q1
(B vs P2) and Q2 (C vs B) were measured with no curriculum in any cell and are
**restated unchanged**: B is 1.11 / 0.33 / 0.48 / **5.60** pt below P2 on
synthetic / ag_news / sst2 / ravdess, and C ≤ B on all four (−0.52 / −0.15 /
−0.14 / **−6.96** pt).



## EXP-6c — `learned` confidence on ag_news (second dataset) — appended 2026-09-26

**Append-only: every EXP-6 / EXP-6b number above is left exactly as it was.**
This section adds a second dataset to the learned-confidence row. Source:
`benchmarks/results/exp6c_learned_agnews.json`, written by
`python benchmarks/run_exp6c.py` on 2026-09-26.

### The question this run asks

EXP-6b measured `learned` **second** on ECE-of-confidence and **fourth** on
AURC on SST-2 — from **one** draw with **one** learned-head seed (0). EXP-6c
re-runs the identical protocol on **ag_news** (K = 4, a multi-class task, where
the K = 2 three-way monotone tie does *not* apply) and trains the learned head
under **five** seeds (0, 1, 2, 3, 4), so the row is reported as mean±std and
the SST-2 single-draw number has a seed-spread reference to be read against.

### Protocol (ids, splits, seeds — recorded exactly)

| Item | Value |
|---|---|
| dataset | `fancyzhx/ag_news`, revision `eb185aade064a813bc0b7f42de02595523103ca4` |
| question / options | "What is the topic of this news article?" / World, Sports, Business, Sci/Tech (label order = ClassLabel order; no remapping) |
| backbone | `google/bert_uncased_L-2_H-128_A-2` rev `30b0a37ccaaa32f332884b96992754e246e48c5f`, frozen, pooling `cls`, fusion `concat`, max_length 128 |
| splits | train 3,000 + cal 1,000 stratified carve from the official **train** split (`bc.make_splits`, seed 0, train∩cal = 0); test = **first 1,500** rows of the official **test** split (7,600 available) |
| split label counts | train 750/750/750/750, cal 250/250/250/250, test 378/401/336/385 |
| head stack | trained **once**, seed 0: `HeadTrainer` + `choice_loss`, 4 epochs, lr 1e-3, batch 64, constructed immediately after `torch.manual_seed(0)` — the EXP-6b / EXP-7a recipe, unchanged |
| learned head | `ConfidenceEngine("learned", fused_dim=128, hidden_size=32)` → MLP 131→32 (tanh)→1→sigmoid, **4,257** trainable parameters, library class unmodified |
| learned fit | CALIBRATION split only (n = 1,000); targets `1[argmax(cal_logits) == cal_label]`; inputs `[fused(128) ; max_prob ; 1−H/logK ; margin]` |
| learned optimization | BCE on the sigmoid output; Adam lr 1e-3; 100 epochs; batch 64; per-epoch shuffle via `torch.Generator().manual_seed(s)`; `torch.manual_seed(s)` before construction; seeds **0,1,2,3,4** |
| test usage | final metrics only — never fit, never tuned, no early stopping |
| pairing | one head, one set of test logits: all four reference methods **and** all five learned seeds are scored on the **same** logits |
| metrics | ECE-of-confidence (15 bins), AURC (`risk_coverage_curve` mean), `coverage_at_risk` 0.1 / 0.2 (0-1 loss) — the EXP-6/6b definitions, unchanged |
| wall time | **113.37 s** total (dataset load 27.5 s, train+collect 29.0 s, cache spot-check 4.9 s, 5× learned training 51.0 s) |

Re-trained head: train loss 0.6067 → 0.3799, **test accuracy 0.8127** (cal
0.8320) — the ag_news figure the published EXP-7 rows are built on.

### Parity with the reference implementation, and the deviations

`benchmarks/run_exp6c.py` **imports** `FusedFeaturizedJevX`,
`collect_logits_and_fused`, `train_learned_confidence` and
`evaluate_confidence` from `benchmarks/run_exp6b.py` rather than re-typing
them, so the code objects executed are the reference implementation's own; the
ag_news id/revision/question/options come from `benchmarks/run_exp7.py` and
the feature cache from `benchmarks/run_head_quality.py`. Four deviations from
EXP-6b, all recorded in the JSON:

* **D1 — features read from the featurize-once cache**
  (`benchmarks/cache/head_quality/ag_news_features.npz`) instead of being
  re-featurized in-run. Not taken on trust: the first **64** test rows were
  re-featurized in this run and compared against the cache — **max abs
  difference 0.0** (`cache_spot_check`).
* **D2 — the cache's splits were checked, not assumed.** A fresh
  `bc.make_splits(train_labels, 3000, 1000, seed=0)` on the pinned dataset
  reproduces the cached train/cal/test labels: **all three match**, train∩cal = 0
  (`split_check`).
* **D3 — the equivalence check is restricted** to test rows whose *untruncated*
  token length is ≤ 128 (1,489 of 1,500 eligible; 11 truncated), because
  `featurize` truncates at `max_length` while `model.choice()` never does, so
  truncated rows are not comparable. On those 16 rows the featurized head stack
  matches the full `model.choice()` path to **2.563e-06** max abs logit
  difference. Same restriction `run_exp7.py` applies and records.
* **D4 — five learned seeds (0,1,2,3,4) instead of one.** Every other protocol
  element is unchanged.

A dataset property, **not** a deviation: ag_news is K = 4, so
max_probability, margin and entropy are not monotone transforms of one another
(that identity holds only at K = 2) and their risk–coverage rows are free to
separate. Whether they do is a measured outcome, reported below.

### Results — all five methods on the SAME 1,500 test logits

`learned` is mean±std over seeds {0,1,2,3,4}; the four reference methods are
deterministic functions of the same logits and are single rows.

| Confidence | ECE-of-conf ↓ | AURC ↓ | coverage@risk 0.1 ↑ | coverage@risk 0.2 ↑ | mean conf |
|---|---|---|---|---|---|
| **learned (5 seeds)** | **0.0375 ± 0.0095** | **0.0688 ± 0.0013** | **0.7644 ± 0.0072** | 1.0000 ± 0.0000 | 0.8361 ± 0.0150 |
| max_probability | 0.0532 | 0.0713 | 0.7200 | 1.0000 | 0.8640 |
| margin | 0.0750 | 0.0724 | 0.7220 | 1.0000 | 0.7557 |
| entropy | 0.0817 | 0.0706 | 0.7273 | 1.0000 | 0.7310 |
| energy | 0.4639 | 0.1744 | 0.0467 | 1.0000 | 0.3507 |

### `learned` per seed (n = 1,500 test rows; BCE is the per-epoch mean train loss)

| seed | ECE-of-conf ↓ | AURC ↓ | cov@0.1 ↑ | cov@0.2 ↑ | mean conf | BCE epoch 1 → 100 |
|---|---|---|---|---|---|---|
| 0 | 0.0313 | 0.0708 | 0.7580 | 1.0000 | 0.8231 | 0.5692 → 0.3197 |
| 1 | 0.0469 | 0.0686 | 0.7607 | 1.0000 | 0.8487 | 0.6680 → 0.3193 |
| 2 | 0.0282 | 0.0691 | 0.7667 | 1.0000 | 0.8217 | 0.6076 → 0.3216 |
| 3 | 0.0324 | 0.0675 | 0.7607 | 1.0000 | 0.8321 | 0.6591 → 0.3235 |
| 4 | 0.0487 | 0.0678 | 0.7760 | 1.0000 | 0.8549 | 0.5726 → 0.3213 |
| **mean±std** | **0.0375 ± 0.0095** | **0.0688 ± 0.0013** | **0.7644 ± 0.0072** | **1.0000 ± 0.0000** | 0.8361 ± 0.0150 | — |

### Rankings BY MEASUREMENT

| Criterion | ag_news (this run, 5 learned seeds) | SST-2 (EXP-6b, 1 learned seed) |
|---|---|---|
| ECE-of-conf | **learned (0.0375)** → max_probability (0.0532) → margin (0.0750) → entropy (0.0817) → energy (0.4639) | max_probability (0.0371) → **learned (0.0910)** → margin (0.2356) → energy (0.3433) → entropy (0.4794) |
| AURC | **learned (0.0688)** → entropy (0.0706) → max_probability (0.0713) → margin (0.0724) → energy (0.1744) | max_probability = margin = entropy (tie, 0.1837) → **learned (0.1983)** → energy (0.2898) |
| coverage@0.1 | **learned (0.7644)** → entropy (0.7273) → margin (0.7220) → max_probability (0.7200) → energy (0.0467) | max_probability = margin = entropy (tie, 0.1674) → **learned (0.1571)** → energy (0.0000) |
| coverage@0.2 | all five tie at 1.0000 (see below) | max_probability = margin = entropy (tie, 0.4931) → **learned (0.4484)** → energy (0.0700) |

**Verdict: the SST-2 ordering does NOT replicate — it changes, and both halves
of it change.**

* **ECE-of-confidence: second → first.** `learned` measured 0.0375 ± 0.0095
  against max_probability 0.0532. The per-seed range is **[0.0282, 0.0487]**,
  entirely below 0.0532, and `learned` measured first on this criterion in
  **5 of 5 seeds** — the sign of the SST-2 gap (learned 0.0539 *worse* than
  max_probability) is reversed.
* **AURC: fourth → first.** `learned` measured 0.0688 ± 0.0013, ahead of
  entropy (0.0706), max_probability (0.0713) and margin (0.0724), and first in
  **4 of 5 seeds** (entropy leads at seed 0, 0.0706 vs 0.0708). The
  separations differ per comparator and are stated separately: the `learned`
  per-seed AURC range is **[0.0675, 0.0708]**, and **entropy's 0.0706 lies
  inside it**, so the mean lead over entropy is **not** separated by the seed
  spread; max_probability (0.0713) and margin (0.0724) sit above that range,
  but by 0.0005 and 0.0016 — margins smaller than the gap the mean column
  reports. The honest statement is narrower than the mean column suggests:
  **`learned` is at the top of the AURC field on the mean of 5 seeds, its
  seed std is 1.29e-03, and no per-comparator AURC gap on this dataset is
  cleanly separated by that spread.**
* **Why the AURC half could not have replicated, and the ECE half is a real
  change.** On SST-2 the three monotone methods are *provably* tied in AURC —
  for K = 2, max_probability, margin and entropy are strictly monotone
  transforms of `max(p, 1−p)` and risk–coverage metrics depend only on the
  induced ranking, so the top three AURC slots are occupied by identity, not
  by measurement, and the only slot `learned` could take was fourth. That tie
  is gone here (0.0713 / 0.0724 / 0.0706 — three distinct values), exactly as
  the K = 2 argument predicts, and `learned` leads. So "behind the K = 2
  monotone trio on AURC" was substantially a structural property of a binary
  task, not evidence about the learned head. The ECE-of-confidence flip is
  **not** of that kind: ECE depends on the confidence *values* and the SST-2
  values were both real measurements, so that reversal is an empirical change
  between two datasets.
* **What replicates.** `energy` is the outlier on both datasets and on both
  criteria — last on ECE-of-confidence (0.4639 here vs 0.3433 on SST-2) and
  last on AURC (0.1744 vs 0.2898), with coverage@0.1 collapsing to 0.0467
  (0.0000 on SST-2). The `learned` head is trainable and converges on both
  datasets (100-epoch terminal BCE 0.3197-0.3235 here, 0.5159 on SST-2).

### How the seed spread compares with the single-seed SST-2 number

The ag_news spread over 5 seeds is **± 0.0095 ECE-of-confidence, ± 0.0013
AURC, ± 0.0072 coverage@0.1**. The SST-2 deficits the EXP-6b row is built on
are **+0.0539** ECE-of-confidence and **+0.0146** AURC. The measured spread is
**5.7× smaller than the SST-2 ECE gap** and **11× smaller than the SST-2 AURC
gap**, so — *on the evidence available* — the SST-2 ordering is not explained
by seed noise in a head whose spread is this tight. Two limits on that
sentence are stated rather than smoothed: (a) the spread is measured on
ag_news and is not transferable to SST-2 by assumption; (b) the comparison is
asymmetric — 5 seeds here, 1 there — so the *direction* of the difference is
a measured fact while its magnitude is not seed-corrected on the SST-2 side.
Re-running EXP-6b with seeds 1–4 would close (b) and is the single next step
that would make the two datasets symmetric.

### One degeneracy recorded on this dataset

`coverage@risk 0.2` measured **1.0000 for all five methods**: at a 0.8127 test
accuracy the base error rate is 0.1873, below the 0.2 budget, so the budget
never binds and the criterion carries no information here. Its 5-way "tie" is
an artifact of the accuracy level, not a property of the confidence methods.
The informative risk–coverage criterion on this dataset is the 0.1 budget.

### Cal-to-test gap, recorded as in EXP-6b

On the calibration split (in-sample for `learned`) the per-seed ECE-of-conf
ran 0.0253–0.0352 and AURC 0.0422–0.0434; on test these became 0.0375 ± 0.0095
and 0.0688 ± 0.0013. The fit again does not carry over in full, so the cal-side
numbers stay a diagnostic and are never the reported row.

**Scope.** This is one dataset (ag_news), one backbone, one head-stack seed and
five learned-head seeds. It establishes that the SST-2 ordering is **not**
stable across datasets; it does not establish which ordering is "the" one —
that would need a third dataset and a seed-corrected SST-2 re-run (§54, §76).
Nothing here supports "confidence guarantees correctness" (LIMITATIONS §5).


---

