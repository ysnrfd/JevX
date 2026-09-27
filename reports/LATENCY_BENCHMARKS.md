# JevX Latency Benchmarks — Phases 55-56, 88 (real CPU measurements)

## Environment (every number below comes from the executed run)

| Field | Value |
|---|---|
| Timestamp (UTC) | 2026-09-24T11:11:23 (bert) / 2026-09-24T11:22:11 (gpt2 result files) |
| OS / platform | Windows-11-10.0.26100-SP0 |
| CPU | Intel64 Family 6 Model 78 Stepping 3 (i5-6300U class), 4 logical cores |
| RAM | ~8.5 GB (8,476 MB total, per psutil) |
| Python | 3.14.3 |
| torch | 2.14.0+cpu (CPU-only; torch threads = 2 during the run) |
| transformers | 5.16.1 |
| GPU | **none** — CPU-only build, no CUDA/MPS/XPU device |
| Model (primary) | hf-internal-testing/tiny-random-bert (state tokenized to **66 tokens**, target 64) |
| Model (cross-check) | sshleifer/tiny-gpt2 (state **69 tokens**) |
| Method | `time.perf_counter`, warmup >= 10 iters, >= 30 timed iters per cell, **median + p95 + IQR** |
| Noise policy | **Concurrent agents share this host.** Cells with IQR/median > 0.25 are flagged `noisy` and retried once (the less-noisy batch is kept). **No sub-millisecond precision is claimed** — treat differences below ~1 ms as noise. Raw JSON: `benchmarks/results/latency_*_{stamp}.json` + `*_latest.json`. |

Instrumentation is **external**: each stage is timed by separate calls to the
public internals (`adapter.prepare_inputs` / `extract_representation` /
`question_encoder` / `schema_encoder` / `fusion` / `heads["choice"]` /
`calibration+confidence`). The model code was **not** modified for timing —
no `return_timings` flag exists or was needed.

---

## (a) + (e) Stage breakdown of one `.choice` call (state ~66 tok, m=1, K=4)

tiny-random-bert, milliseconds (median over >=30 iters):

| # | Stage | median ms | p95 ms | IQR ms | noisy? |
|---|-------|----------:|-------:|-------:|:------:|
| 1 | preprocess (tokenizer) | 3.115 | 6.635 | 4.031 | yes (retried) |
| 2 | **backbone forward + adapter pool** | **15.135** | 26.796 | 4.802 | yes (retried) |
| 3 | state projection | 0.127 | 0.147 | 0.008 | no |
| 4 | question encode (E_Q + proj) | 1.509 | 3.468 | 0.305 | no |
| 5 | schema encode (E_C + proj, K=4) | 2.176 | 3.218 | 0.259 | no (retried) |
| 6 | fusion F | 0.917 | 1.135 | 0.132 | no |
| 7 | choice head G | 1.102 | 1.712 | 0.170 | no |
| 8 | calibration + confidence | 0.170 | 0.317 | 0.077 | yes (retried) |
| 0 | **full .choice call (reference)** | **22.749** | 34.414 | 4.586 | no (retried) |

**Top-3 costliest stages:** (1) backbone forward + adapter pool **15.1 ms**,
(2) tokenizer preprocessing **3.1 ms**, (3) schema encode **2.2 ms**.

### The tiny-head-vs-backbone point (§88), measured

The entire JevX decision layer — state projection (0.13) + question encode
(1.51) + schema encode (2.18) + fusion (0.92) + head (1.10) + calibration/
confidence (0.17) = **6.0 ms** — costs ~40% of the 15.1 ms backbone forward,
and the head itself (fusion + G + confidence ≈ **2.2 ms**) is **~7x cheaper
than the backbone pass**. Stage medians do not sum exactly to the full-call
median (24.3 vs 22.7 ms) because each cell carries its own call overhead and
noise; the full-call row is the authoritative end-to-end number.

### Cross-check: sshleifer/tiny-gpt2 (decoder family), same shape

| Stage | median ms | Stage | median ms |
|---|---:|---|---:|
| preprocess | 1.167 | fusion | 1.012 |
| **backbone + pool** | **12.949** | choice head | 1.029 |
| state proj | 0.124 | cal + confidence | 0.275 |
| question encode | 1.326 | **full call** | **32.609** |
| schema encode | 3.658 | | |

Same ordering: backbone dominates; gpt2's Python-side overheads are noisier
(most gpt2 cells flagged noisy under concurrent load).

---

## (b) Model load — `JevX.from_pretrained` cold vs warm cache

| Cell | median | p95 | IQR | n | notes |
|---|---:|---:|---:|--:|---|
| warm cache (pre-populated temp cache dir) | **3955.8 ms** | 4051.7 | 119.6 | 3 | relative IQR 3.0% — stable |
| cold cache (fresh empty temp cache dir) | **5688.5 ms** | 5848.8 | 178.1 | 2 | **includes the weight+tokenizer download** (network OK) |

First load in the latency process (default HF cache, includes capability
detection + wrapper construction): **8.4 s** (tiny-random-bert), **6.1 s**
(tiny-gpt2). Cold ≈ warm + ~1.7 s of download for this tiny checkpoint;
warm-cache reload cost is dominated by transformers' load/inspect machinery,
not by JevX's wrapper construction.

---

## (c) T(m): shared-state one-call vs naive m separate requests (§56)

One state (66 tok), K=4, m questions. **Shared** = one `.choice` call with m
questions; **naive** = m separate `.choice` calls (each re-runs tokenizer +
backbone). medians in ms.

| m | shared median | shared p95 | shared IQR | naive median | speedup |
|--:|---:|---:|---:|---:|---:|
| 1 | 21.27 | 27.83 | 3.87 | 19.80 | 0.93x |
| 2 | 21.16 | 73.02 | 6.49 † | 55.07 | 2.60x |
| 4 | 29.37 | 36.01 | 5.61 | 100.97 | 3.44x |
| 8 | 62.64 | 77.74 | 11.42 | 315.33 | 5.03x |
| 16 | 94.75 | 160.13 | 29.18 † | 530.78 | 5.60x |
| 32 | 92.20 | 220.26 | 16.18 | 1020.07 | **11.06x** |

† cell flagged noisy (IQR/median > 0.25 before retry); retried batch kept.

**Marginal cost (OLS on shared medians): slope = 2.52 ms/question,
intercept 27.1 ms, R² = 0.752.**

### Does C_total ≈ C_state + Σ C_question hold — as measured?

Reported honestly, no unmeasured claims:

* The linear model T(m) = 27.1 + 2.52·m explains **R² = 0.75** of the
  measured medians — good direction, but **not tight**: residuals vs the
  T(1)+(m-1)·slope form are −2.6 ms (m=2), +0.5 (m=4), **+23.7 (m=8)**,
  **+35.7 (m=16)**, −7.2 (m=32). The m=8/16 overshoot and the m=32
  non-monotonicity (92.2 < 94.8) are consistent with concurrent-host noise
  (m=16 flagged noisy), not with a clean linear law.
* What **is** measured cleanly: shared-mode growth is **far sub-linear in
  the naive cost** — naive grows ~linearly at ~32 ms/question (19.8 → 1020 ms
  from m=1 → 32) while shared grows at ~2.5 ms/question fitted, and the
  speedup widens monotonically 2.6x → 11.1x.
* The m=1 row (0.93x) shows the shared path carries ~1.5 ms of per-call
  schema/question-list plumbing overhead vs a single-question call — within
  noise of parity.
* Peak RSS is essentially **flat in both modes** (see below): the shared-state
  claim is about compute reuse, and no memory penalty was observed either way.

**Peak RSS during both modes** (psutil sampler thread, 5 ms interval):

| m | shared peak RSS MB | naive peak RSS MB | RSS before MB |
|--:|---:|---:|---:|
| 1 | 490.64 | 490.68 | 490.61 |
| 2 | 490.74 | 490.75 | 490.71 |
| 4 | 490.78 | 490.80 | 490.75 |
| 8 | 490.81 | 490.80 | 490.78 |
| 16 | 490.92 | 491.01 | 490.83 |
| 32 | 491.24 | 491.19 | 491.08 |

Both modes stay within **0.6 MB** of the pre-sweep baseline across the whole
sweep (tiny model; torch CPU allocator stats are N/A — no CUDA allocator).

---

## (d) Batch throughput: B states x 1 question

tiny-random-bert:

| B | median ms | p95 ms | IQR ms | throughput (pairs/s) | noisy? |
|--:|---:|---:|---:|---:|:--:|
| 1 | 26.97 | 41.67 | 9.69 | 37.1 | yes |
| 2 | 31.55 | 37.67 | 6.52 | 63.4 | no |
| 4 | 62.06 | 146.99 | 19.02 | 64.5 | yes |
| 8 | 61.55 | 102.24 | 21.69 | **130.0** | yes |

tiny-gpt2 cross-check: B=1 35.9 pairs/s → B=2 81.6 → B=4 110.5 →
**B=8 245.2 pairs/s**.

Throughput rises with B (one backbone pass serves B states), but the cells
are noisy under concurrent load: B=2 and B=4 medians are nearly identical on
bert (63 vs 64 pairs/s) — treat the B-sweep as directional (batches help),
not as a precise scaling law.

---

## Reproducing

```
python benchmarks/run_latency.py        # ~4 min; overwrites latency_*.json only
python benchmarks/run_latency.py --quick  # smoke run
```
