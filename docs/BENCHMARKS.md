# Benchmarks — index and digest of the four benchmark reports

Index page. Every table below is quoted from a report produced by executed
code on 2026-09-24 on this host; source JSONs live in
`benchmarks/results/*.json`. **No number appears here that is absent from its
source report.** Re-run commands and CLI args:
[docs/REPRODUCIBILITY.md](REPRODUCIBILITY.md) §4.

## Hardware context (shared by all four reports)

Windows 11 (10.0.26100), Intel64 Family 6 Model 78 Stepping 3
(i5-6300U class), 4 logical cores, ~8 GB RAM (8,476 MB psutil), Python
3.14.3, torch 2.14.0+cpu (threads=2 during runs), transformers 5.16.1,
**no GPU**, float32. Concurrent agents shared the host during the latency/
memory runs: cells with IQR/median > 0.25 were flagged noisy and retried once;
differences below ~1 ms are treated as noise (reports/LATENCY_BENCHMARKS.md).

---

## 1. Quality — reports/DATASET_BENCHMARKS.md

Protocol: SST-2 (`stanfordnlp/sst2` rev `8d51e7e…`), backbone
`google/bert_uncased_L-2_H-128_A-2` (rev `30b0a37…`, 4,385,920 frozen params,
Mode A / `freeze_base_model=True`), train 3,000 / calibration 1,000 / test 872
(official validation split in full), heads-only training (1,247,361 trainable
JevX params, measured), epochs 4, lr 1e-3, batch 64 — chosen a priori, no
hyperparameter search on any split. Heads fit ONLY on train; calibrators ONLY
on calibration; test ONLY for final metrics (§2 of that report).

### Headline — EXP-1, 3 seeds {0,1,2}, mean ± std (DATASET_BENCHMARKS §3)

| Model / variant | acc | macro-F1 | NLL ↓ | Brier ↓ | ECE(15) ↓ | MCE(15) ↓ |
|---|---|---|---|---|---|---|
| JevX uncalibrated | 0.7011 ± 0.0104 | 0.6998 ± 0.0107 | 0.5759 ± 0.0067 | 0.3940 ± 0.0064 | 0.0320 ± 0.0068 | 0.0717 ± 0.0008 |
| **JevX + temperature** | 0.7011 ± 0.0104 | 0.6998 ± 0.0107 | **0.5742 ± 0.0080** | **0.3926 ± 0.0072** | **0.0234 ± 0.0003** | **0.0517 ± 0.0133** |
| B1 linear probe | 0.7003 ± 0.0063 | 0.7001 ± 0.0064 | 0.5773 ± 0.0029 | 0.3932 ± 0.0017 | 0.0286 ± 0.0088 | 0.0678 ± 0.0199 |
| B2 MLP probe | **0.7091 ± 0.0046** | **0.7089 ± 0.0047** | 0.5806 ± 0.0041 | 0.3945 ± 0.0023 | 0.0445 ± 0.0057 | 0.0951 ± 0.0195 |
| B4 = B1 + temperature | 0.7003 ± 0.0063 | 0.7001 ± 0.0064 | 0.5777 ± 0.0034 | 0.3932 ± 0.0017 | 0.0291 ± 0.0079 | 0.0671 ± 0.0165 |
| B3 distilgpt2 zero-shot (n=1, not comparable) | 0.6170 | 0.5979 | 0.6537 | 0.4619 | 0.0235 | 0.8779 |

Measured reading (DATASET_BENCHMARKS §4): B2's accuracy edge over JevX
(0.7091 vs 0.7011 ≈ 7 of 872 examples) has **overlapping seed ranges** — not
distinguishable at n=872 with 3 seeds; what separates these measurements is
probability quality, where JevX+temperature measured best NLL/Brier/ECE/MCE of
the same-backbone set. Accuracy ~0.70 vs 0.509 majority baseline. Batch-1
latency rows in the same report: full `.choice` 12.256 ± 2.235 ms (single
text ×100), 14.272 ± 3.682 ms (100 distinct texts), head stack only
5.960 ± 3.055 ms, B1 linear head only 0.108 ± 0.093 ms.

## 2. Ablations — reports/ABLATIONS.md (every row SINGLE seed 0)

| Ablation | Headline observation (measured, single draw, n=872) |
|---|---|
| EXP-3 pooling | `masked_mean` won 5 of 6 columns (acc 0.7144); native pooler 0.6491 — largest spread (Δ=6.3 pts) |
| EXP-4 fusion + question control | fusions span 0.7110–0.7236 acc; `gated` highest (0.7236) with most fusion params (82,304); zeroed-question control 0.7190 vs real question 0.7122 — question pathway showed no gain on this single-question task (stated, not over-read) |
| EXP-5 calibration | temperature best NLL/ECE/MCE/Brier; vector best acc (0.7179); isotonic MCE 0.6667 (instability) — see [CALIBRATION.md](CALIBRATION.md) |
| EXP-6 confidence | max_probability best ECE-of-conf (0.0371); AURC three-way tie at K=2 (0.1837); energy worse (0.2898) — see [CONFIDENCE.md](CONFIDENCE.md) |

Seed table (ABLATIONS.md): EXP-1/EXP-2 = 3 seeds; EXP-3…EXP-6 = single seed 0.
No row difference is a variance-adjusted claim.

## 3. Latency — reports/LATENCY_BENCHMARKS.md

Model `hf-internal-testing/tiny-random-bert` (state 66 tokens) unless noted;
medians over ≥30 timed iterations after ≥10 warmup; `time.perf_counter`;
instrumentation external (public internals called separately — model code not
modified).

**Stage breakdown of one `.choice` call (m=1, K=4), median ms:**

| Stage | median ms | | Stage | median ms |
|---|---:|---|---|---:|
| preprocess (tokenizer) | 3.115 | | fusion F | 0.917 |
| **backbone forward + pool** | **15.135** | | choice head G | 1.102 |
| state projection | 0.127 | | calibration + confidence | 0.170 |
| question encode | 1.509 | | **full `.choice` call** | **22.749** |
| schema encode (K=4) | 2.176 | | | |

Decision layer total ≈ **6.0 ms** vs 15.135 ms backbone (head itself ≈ 2.2 ms,
~7× cheaper than the backbone pass) — the §88 decomposition, measured.

**T(m): shared-state vs naive (medians, ms; cross-check sshleifer/tiny-gpt2
showed the same ordering):**

| m | shared | naive | speedup |
|---:|---:|---:|---:|
| 1 | 21.27 | 19.80 | 0.93× |
| 2 | 21.16 | 55.07 | 2.60× |
| 4 | 29.37 | 100.97 | 3.44× |
| 8 | 62.64 | 315.33 | 5.03× |
| 16 | 94.75 | 530.78 | 5.60× |
| 32 | 92.20 | 1020.07 | **11.06×** |

OLS on shared medians: **slope 2.52 ms/question, intercept 27.1 ms,
R² = 0.752** — good direction, explicitly *not tight* (residuals reported,
m=16 noisy, m=32 non-monotone under concurrent load). Naive mode grows
~32 ms/question (19.8 → 1020 ms). Peak RSS flat in both modes (490.6–491.2 MB).

Load: warm-cache `JevX.from_pretrained` **3,955.8 ms** median (n=3), cold
cache 5,688.5 ms (includes download). Batch throughput (bert): B=1 37.1 →
B=8 130.0 pairs/s (directional only — cells noisy); tiny-gpt2 B=8 245.2 pairs/s.

## 4. Memory — reports/MEMORY_BENCHMARKS.md

`hf-internal-testing/tiny-random-bert`; MB = 10^6 bytes; RSS via ctypes
`GetProcessMemoryInfo` fallback; tracemalloc = Python allocator only.

| Scenario | python-side peak MB | RSS delta MB |
|---|---:|---:|
| model load | 3.03 | **15.89** |
| steady forward, batch 1 (×50) | 0.054 | **0.008** |
| m=32 shared single call | 0.104 | 0.037 |
| m=32 as 32 separate calls | 0.050 | 0.045 |
| CUDA / optimizer memory | **N/A (CPU-only, inference-only run)** | — |

No forward-time growth observed (8 KB over 50 calls = allocator noise).
Final process RSS 493.50 MB; baseline ~470 MB is torch/transformers itself.

---

## 5. What these reports do NOT claim

Scoped exactly as their sources scope them: no sub-millisecond precision on the
noisy shared host; all ablation rows are single-draw observations on n=872;
quality numbers cover one frozen backbone on one dataset (SST-2) and do not
extend beyond this host/protocol (§76 language of the reports); no
distribution-free or cross-domain claim; GPU/quantized/cross-device paths
untested here (docs/LIMITATIONS.md §1).

---

## 6. Extension runs (appended 2026-09-24, after §5 — append-only)

Two runs extended the scope §5 states: **EXP-7** repeated the EXP-1/EXP-2
protocol on a second real dataset, `fancyzhx/ag_news` rev
`eb185aade064a813bc0b7f42de02595523103ca4` (id verified via the Hub API before
download; 3 seeds {0,1,2}, train 3,000 / cal 1,000 stratified carve from
official train, test = first 1,500 of the official 7,600-row test split, same
backbone/hyperparams, wall time 163.62 s, `failures: {}`) — headline
mean ± std: JevX uncal **0.8216 ± 0.0128** acc / ECE 0.0623, JevX+temperature
0.8216 ± 0.0128 / **ECE 0.0262** (best ECE and MCE of the same-backbone set),
JevX+isotonic 0.8202 ± 0.0115 / 0.0359, B1 linear 0.8262 ± 0.0010 / 0.0543,
B4 = B1+temp 0.8262 ± 0.0010 / 0.0289, B2 MLP **0.8349 ± 0.0112** (best
accuracy/F1/NLL/Brier) / 0.0397 — full tables in
[DATASET_BENCHMARKS.md](../reports/DATASET_BENCHMARKS.md) §8; **EXP-6b**
measured the `learned` confidence method on the SST-2 EXP-6 setup (SINGLE
seed 0, MLP fit on the calibration split only, all five methods on the same
test logits with the four EXP-6 baselines reproduced bit-for-bit): learned
ECE-of-conf 0.0910, AURC 0.1983, coverage@0.1 0.1571, coverage@0.2 0.4484 —
[ABLATIONS.md](../reports/ABLATIONS.md) EXP-6b. These append-only sections
supersede §5's "one dataset / `learned` not run" scope note; every number in
§§1–5 above remains as originally recorded.


## 7. Modality quality — reports/MODALITY_BENCHMARKS.md (append-only, 2026-09-25)

Two extension runs brought the §1 quality protocol to real vision and audio
data (same §2 discipline: splits carved with `bench_common`, calibrators on
the calibration split only, frozen backbones featurize-once, B1/B2/B4 on
identical features, seeds {0,1,2}): **EXP-V** = `facebook/deit-tiny-patch16-224`
(rev `b3428f18…`, 5,561,472 frozen params) × `frgfm/imagenette`
imagenette2-160 (9,469/3,925 official rows indexed; splits 800/300/500; wall
593.03 s, failures {}) — JevX uncal **0.9087 ± 0.0257** acc, JevX+temperature
0.9087 with the set's best ECE(15) 0.0207, B1 0.9467, **B2 0.9673 (best
acc/F1/NLL/Brier)**, B4 0.9467/ECE 0.0221; **EXP-A** = 
`superb/wav2vec2-base-superb-ks` (rev `372e0486…`, 94,371,712 params) ×
`Codec-SUPERB/RAVDESS` (rev `46aa36c5…`, 1,440 clips, 8 classes, splits
700/240/500, wall 3111.21 s, failures {}) — JevX **0.1340 = the majority
baseline, explicitly flagged below the credible bar** while B1 0.3433 and B2
0.3607 on the identical features. The report also records the exact API
errors for every rejected candidate (401s, dataset-script removal), one fixed
adapter bug (`jevx/adapters/audio.py` batched-audio padding), equivalence
0.0 rows, and batch-1 latency for both modalities; full tables and JSON:
[MODALITY_BENCHMARKS.md](../reports/MODALITY_BENCHMARKS.md),
`benchmarks/results/{vision_*,audio_*}.json`. Every number in this section
is quoted from that report; §1–§6 remain as originally recorded.


## 8. Multi-question regime + conditioning-cache cost — reports/MULTI_QUESTION.md (append-only, 2026-09-26)

Sections 1-7 are the single-question benchmarks; this is the first evaluation of
the regime the question/schema encoders actually exist for. Script
`benchmarks/run_multi_question.py`; full report and every deviation:
[MULTI_QUESTION.md](../reports/MULTI_QUESTION.md);
JSON: `benchmarks/results/multi_question_*.json`.

**The `constant` conditioning cache.** `conditioning_mode="constant"` is
bitwise-equal to `"full"` on the cold call, the warm memo hit and after a
content change — confirmed on two independent wrappers (tiny in-memory BERT +
`tests/conftest.py` tokenizer, and the real
`google/bert_uncased_L-2_H-128_A-2`), max abs diff 0.0 in all six cells, and
identical to 4 decimals on every metric over 5 seeds on ABSA. It is
inference-only (`_condition_cache_enabled()` gates on `self.training` and
`torch.is_grad_enabled()`), so it buys latency and never accuracy. Batch-1,
m=4 questions, K=3 options, 20 warmup + 120 timed iters, arms timed round-robin
on a shared box: state-only (`"off"`) 9.84 ms (tiny) / 11.08 ms (real backbone),
`constant`-HIT 7.41 / 7.30, `constant`-MISS 8.52 / 8.79, `full` 11.00 / 12.06.
**The program's `<= 1.15x` criterion is MET: `constant`-HIT/`off` = 0.753 and
0.659.** The hit arm beats the state-only arm because `"off"` does not skip the
encoders — it zeroes the option residual and still pays the full question +
schema encoding. A new hazard was measured and NOT fixed (`jevx/**` out of
scope): the memo is keyed on content, not weights, so eval → one AdamW step →
eval *without* `clear_condition_cache()` serves stale H_Q/H_C (max abs diff
**4.1556e+04**).

**Multi-question grid — 11 questions, 4 datasets, 5 seeds (0-4).** ABSA
SemEval-2014 Task 4 (`xinyixiuxiu/semeval2014_restaurant` rev
`709b40b7…`, `…_laptop` rev `fed294ac…`; 4-13 REAL aspect-polarity questions
per state); RAVDESS `Codec-SUPERB/RAVDESS` rev `46aa36c5…` (4 REAL
filename-encoded questions: emotion, gender, intensity, statement-vs-question);
ag_news `fancyzhx/ag_news` rev `eb185aad…` (1 REAL topic + 3 DERIVED
thresholds); Yelp `Yelp/yelp_review_full` rev `c1f9ee93…` (1 REAL star rating
+ 1 DERIVED detractor threshold; aspect questions OMITTED because that dataset
ships no aspect label and hand-writing one would be hand-annotating a
benchmark). Established frozen-protocol backbones, featurize-once, stratified
state-level splits seed 0. **Decisive question: does any conditioning mode beat
`off` on any question with NON-OVERLAPPING 5-seed accuracy ranges? NO — 35
paired comparisons, 0 wins.** The single non-overlapping comparison is a **loss
of 16.92 pp** (RAVDESS gender, `full` 0.7296 ± 0.0687 vs `off` 0.8988 ± 0.0535).
Largest positive delta anywhere is +9.72 pp (RAVDESS intensity) with ranges
overlapping by 9.2 pp and a 10.6 pp baseline std — **a win inside one std is not
a win**. Pooled over all 11 questions: `p2_mlp` 0.7357 > `state_only` 0.7294 >
`off` 0.7199 > `full` 0.7046 > `p1_linear` 0.6898. Measured gradient norms:
the conditioning half is **80.8-89.7 %** of the trainable parameters and
receives **exactly zero** gradient at step 1 in the `off`/`state_only` arms;
its largest gradient is `question_proj` at step 1, and `state_proj` dominates
by the final step. **Recommendation: `conditioning_mode="off"` becomes the
single-question default**, on the numbers above; `state_only` (question rows
zeroed too, a reference this wave added, not a shipped mode) is a further ~1 pp.
**Shared state:** with 4+ questions on ONE state the backbone forward-hook count
is **exactly 1** for m = 1..5, and the m-question call's per-question logits
agree with the per-question calls to 3.58e-07 (worst of 5) with argmax 5/5 — the
multi-question regime is a shared-parameter regime, not a different
computation. Deviations (RAVDESS at 2 epochs and head width 256, ag_news/Yelp at
1500/500/1000 splits inside the 3000/1000/1500 caps, `constant` not re-run on
three datasets) and an explicit "what this does NOT establish" section are in the
report; no metric was recomputed or omitted.
