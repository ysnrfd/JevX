# Real-Modality Quality Benchmarks — EXP-V (vision) + EXP-A (audio)

**Status:** all numbers below were produced by executed code on 2026-09-25 and
are stored verbatim in `benchmarks/results/{vision_*,audio_*}.json`. No table
cell is hand-written. Language follows §76: *measured* / *observed*, no claim
extends beyond this host, this split, this protocol. This report closes the
modality gap noted in [DATASET_BENCHMARKS.md](DATASET_BENCHMARKS.md) (which
covers text/SST-2 only) and the quality-metric pointers in
[REAL_MODEL_VALIDATION.md](REAL_MODEL_VALIDATION.md) (which run no quality
metrics).

---

## 1. Environment & reproduction block (§87)

| Item | Value |
|---|---|
| Machine | Windows 11 (10.0.26100), Intel64 Family 6 Model 78 Stepping 3, 4 logical CPUs (2 physical), ~8 GB RAM; concurrent agents shared the host during all runs |
| Python | 3.14.3 |
| torch | 2.14.0+cpu (`cuda_available=False`, `torch_threads=2`) |
| transformers | 5.16.1 |
| datasets | 4.4.1 |
| huggingface_hub | 1.30.0 |
| numpy | 2.3.5 |
| PIL / soundfile / scipy | 12.0.0 / 0.14.0 / 1.16.3 |
| dtype | float32 throughout |
| EXP-V backbone | `facebook/deit-tiny-patch16-224`, revision `b3428f18dcc7b543470d07f14b4a4157815d1880` (API sha, identical to `config._commit_hash` after load), 5,561,472 frozen params (measured) |
| EXP-V dataset | `frgfm/imagenette` revision `a8a496123b40fa739da2acf9b0dae339d30c7bae`, config `imagenette2-160`; image archive = the S3 tarball the pinned loader script downloads: 99,003,388 bytes, sha256 `64d0c4859f35a461889e0147755a999a48b49bf38a7e0f9bd27003f10db02fe5` |
| EXP-A backbone | `superb/wav2vec2-base-superb-ks`, revision `372e0486cd83e6f0c05c20a27262e9ca09450d24` (API sha = loaded `_commit_hash`), `pytorch_model.bin` 378,369,297 bytes, 94,371,712 params measured (≤100M ✓) |
| EXP-A dataset | `Codec-SUPERB/RAVDESS` revision `46aa36c5fdeb1b6d63b746b74fde528a540044b2`, 1,440 wav files ≈ 120 MB on the Hub, 16 kHz PCM-16, 8 emotion classes |
| Training seeds | both experiments `{0, 1, 2}` (3 seeds) |
| Hyperparams | epochs=4, lr=1e-3 (HeadTrainer AdamW default — chosen a priori, **no hyperparameter search on any split**), head batch=64; featurize batch: vision 32, audio 16; split seed=0 |
| JevX config | `freeze_base_model=True` (Mode A), `fusion_type="concat"` (default), decision hidden = backbone width (vision 192; audio 768); pooling: vision `"cls"` (see §4), audio `None` → adapter `masked_mean` (wav2vec2 has no pooler) |
| Entry points | `python benchmarks/run_modality.py --modality vision` and `python benchmarks/run_modality.py --modality audio --audio-feat-batch 16` |
| Wall time | EXP-V **593.03 s** total; EXP-A **3111.21 s (featurize 2594.68 s of it)** (both `failures: {}`); **combined experiment wall 3704.24 s (61.7 min)**; development: vision smoke 99.45 s, audio smoke 1345.93 s |
| Result files | `benchmarks/results/vision_jevx_quality.json`, `vision_baselines.json`, `vision_baselines_latency.json`, `vision_run_summary.json`, `audio_jevx_quality.json`, `audio_baselines.json`, `audio_baselines_latency.json`, `audio_run_summary.json` |

---

## 2. Ids verified via the Hub API **before** the run (exact errors for rejects)

### 2.1 Backbones

| role | id | API | revision sha | size / params | decision |
|---|---|---|---|---|---|
| EXP-V primary | `facebook/deit-tiny-patch16-224` | GET 200 | `b3428f18dcc7b543470d07f14b4a4157815d1880` | `pytorch_model.bin` 22,950,127 B; 5,561,472 params measured | **used** (≤30M ✓) |
| EXP-V candidate | `google/vit-tiny-patch16-224` | **401** `{"error":"Invalid username or password."}` on `/api/models/`, `/tree/main`, `/resolve/main/config.json` (no token on this host) | — | — | rejected (API blocked) |
| EXP-A primary | `superb/wav2vec2-base-superb-ks` | GET 200 | `372e0486cd83e6f0c05c20a27262e9ca09450d24` | `pytorch_model.bin` 378,369,297 B; 94,371,712 params measured (≤100M ✓) | **used** |
| EXP-A alt | `MIT/ast-finetuned-audioset-10-10-0.4593` | GET 200 | `f826b80d28226b62986cc218e5cec390b1096902` | 692,880,325 B repo total | not needed (wav2vec2 verified first) |

### 2.2 Datasets

| role | id | API | revision sha | size | decision |
|---|---|---|---|---|---|
| EXP-V | `frgfm/imagenette` (apache-2.0 tag) | GET 200 | `a8a496123b40fa739da2acf9b0dae339d30c7bae` | loader repo ≈ 2 MB + data archive 99,003,388 B (≤250 MB ✓) | **used** (see 2.3 for the loader-script block and the workaround) |
| EXP-A | `Codec-SUPERB/RAVDESS` (no license tag returned; recorded as-is, not inferred) | GET 200 | `46aa36c5fdeb1b6d63b746b74fde528a540044b2` | 1,440 wavs, tree total 120,024,806 B (≤250 MB ✓) | **used** |

### 2.3 Rejected candidates — exact API/runner errors (attempt order)

1. `google/vit-tiny-patch16-224` → **HTTP 401** `{"error":"Invalid username or password."}` (three endpoints tried) → fell back to the second backbone named in the brief, `facebook/deit-tiny-patch16-224` (GET 200).
2. `frgfm/imagenette` via `datasets.load_dataset` → `RuntimeError: Dataset scripts are no longer supported, but found imagenette.py`; the datasets-server parquet endpoint fails identically (HTTP 500, `cause_message: "Dataset scripts are no longer supported, but found imagenette.py"`). **Workaround (recorded, not hidden):** the run downloads the *exact* S3 tarball the pinned script downloads (`imagenette2-160.tgz`, 99,003,388 B, sha256 in §1) plus the pinned-revision `metadata/train.txt` + `val.txt` (sha256 recorded per file in the JSON), and applies the script's own `_LABEL_MAP` (WNID order) for labels. Data identity is pinned by content hash rather than by a Hub dataset revision alone.
3. `beans/oxford-flower102` → **HTTP 401** `{"error":"Invalid username or password."}` → not used.
4. `PolyAI/minds11` → **HTTP 401** `{"error":"Invalid username or password."}` → not used.
5. `esc50` (canonical id) → **HTTP 401** `{"error":"Invalid username or password."}`; mirrors checked: `ashraq/esc50` (GET 200 but 773,325,074 B > 250 MB budget), `mskov/ESC50` (1.55 GB), `yangwang825/esc50` (646 MB zip) → all over budget → not used.
6. `hf-internal-testing/librispeech_asr_dummy` → GET 200 but only 73 rows < the ≥300-usable-clips rule → not used (size reject, not an API error).
7. RAVDESS mirrors: `confit/ravdess-parquet` (3.25 GB), `xbgoose/ravdess` (325 MB), `MahiA/RAVDESS` (409 MB) over budget; `Codec-SUPERB/RAVDESS` (120 MB, 1,440 clips, 8 classes, 16 kHz) satisfies every constraint → **used**.

After these two verification passes no further candidate was needed: both chosen pairs satisfy ≤250 MB, ≥3 classes, ≥300 usable clips, ≤3 s clips after the documented trim — **no reduced-n run was required**.

---

## 3. Protocol (mirrors the SST-2 text protocol — reports/DATASET_BENCHMARKS.md §2)

> **Heads are fit ONLY on the train split. Calibrators (temperature, isotonic)
> are fit ONLY on the calibration split. The test split is used ONLY for the
> final reported metrics — never for fitting, never for tuning, never for
> selection. No hyperparameter (lr, epochs, batch sizes, pooling, fusion) was
> selected on the basis of test performance; lr/epochs were fixed a priori at
> the HeadTrainer library defaults (lr 1e-3, epochs 4).**

| | EXP-V (vision) | EXP-A (audio) |
|---|---|---|
| Decision framing | 10-class choice over imagenette labels; question *"Which category does this image belong to?"*; schema = the script's `ClassLabel` names in WNID order | 8-class choice over RAVDESS emotions; question *"Which emotion is spoken in this audio?"*; schema = `neutral, calm, happy, sad, angry, fearful, disgusted, surprised` (filename-prefix labels, no remapping) |
| train | 800, stratified carve from the **official train split** (seed 0), disjoint | 700, stratified carve from the corpus pool (seed 0), disjoint |
| calibration | 300, stratified carve from the remainder of official train (seed 0+1), disjoint | 240, stratified carve from the remainder (seed 0+1), disjoint |
| test | **official validation split**, stratified subsample n=500 (seed 0) — never fitted on | the untouched **disjoint remainder** of the pool, n=500 — never fitted on |
| label balance (train/cal/test) | 72–84 / 27–32 / 46–53 per class | 47–94 / 16–32 / 33–67 per class |
| backbone | frozen DeiT-tiny, featurize-once (Mode A) | frozen wav2vec2-base, featurize-once (Mode A) |
| features | `extract_representation(strategy="cls")` — CLS token (why: §4) | `extract_representation(strategy=None)` → no pooler → adapter `masked_mean` over frames (family convention) |
| heads | `HeadTrainer` + `choice_loss`, 4 epochs, lr 1e-3, batch 64; JevX trainable params 2,018,497 (measured) | same protocol; JevX trainable params 13,382,401 (measured) |
| metrics | Accuracy / Macro-F1 / NLL / Brier / ECE(15) / MCE(15) — `bench_common.metrics_from_logits` / `calibrated_metrics` | same |
| baselines | B1 linear probe, B2 MLP probe, B4 = B1 logits + temperature on the **identical** cached features and splits; B3 (generative) is text-only and does not apply to modalities | same |
| seeds | 3 ({0,1,2}), mean ± sample std | same |
| invariants | asserted on **every** eval batch: finite logits, probs ∈ [0,1], Σp = 1 (tol 1e-4), K = schema size; plus full `.choice` invariants (schema membership, confidence ∈ [0,1]) on e2e probes | same |
| equivalence | full `model.choice()` vs cached `FeaturizedJevX` logits on 16 test examples × seed: **max_abs_diff = 0.0** (all seeds) | **max_abs_diff = 0.0** (all seeds) |

Split construction reuses `benchmarks/bench_common.py` verbatim
(`stratified_indices` / `make_splits`, largest-remainder quotas + seeded
`np.random.RandomState`); train/cal/test disjointness is asserted at
construction. Calibration fits use `CalibrationEngine` on calibration logits
only; test logits are never passed to any fit call (`calibrated_metrics`
rejects that by design).

---

## 4. EXP-V — JevX over frozen DeiT-tiny × Imagenette-160 (3 seeds)

**Backbone pooling note (measured deviation from "model-native"):** the
transformers load report for `facebook/deit-tiny-patch16-224` marks
`pooler.dense.weight` / `pooler.dense.bias` as **MISSING** — the checkpoint
ships no pooler weights, so the "native" `pooler_output` would be a randomly
initialized projection (drawn per reload, i.e. seed-inconsistent across the
per-seed model reloads). The primary pooling is therefore `"cls"` — the token
the checkpoint's own classifier consumes — recorded in every result file
(`provenance.pooling`, `benchmarks/run_modality.py:629`). Feature tensors are
stateless across reloads, so the featurize-once cache is exact for all seeds
(equivalence 0.0 confirms it).

Train losses fell in every seed (0.725→0.194, 0.689→0.042, 0.731→0.115).
Majority-class baseline of the test split: 53/500 = 0.106; chance = 0.100.

### 4.1 Headline: metric × variant (mean ± sample std over 3 seeds)

| Model / variant | Accuracy | Macro-F1 | NLL ↓ | Brier ↓ | ECE(15) ↓ | MCE(15) ↓ |
|---|---|---|---|---|---|---|
| JevX uncalibrated | 0.9087 ± 0.0257 | 0.9107 ± 0.0247 | 0.6326 ± 0.1207 | 0.1541 ± 0.0385 | 0.0736 ± 0.0196 | 0.5567 ± 0.0622 |
| JevX + temperature | 0.9087 ± 0.0257 | 0.9107 ± 0.0247 | 0.2991 ± 0.0630 | 0.1307 ± 0.0313 | **0.0207 ± 0.0072** | 0.4803 ± 0.1208 |
| JevX + isotonic | 0.9120 ± 0.0269 | 0.9142 ± 0.0255 | 0.5890 ± 0.1208 | 0.1388 ± 0.0327 | 0.0292 ± 0.0049 | 0.4679 ± 0.1958 |
| B1 linear probe | 0.9467 ± 0.0046 | 0.9470 ± 0.0046 | 0.2900 ± 0.0161 | 0.1058 ± 0.0049 | 0.1376 ± 0.0109 | 0.3923 ± 0.0362 |
| **B2 MLP probe** | **0.9673 ± 0.0042** | **0.9676 ± 0.0041** | **0.1085 ± 0.0033** | **0.0488 ± 0.0017** | 0.0260 ± 0.0042 | 0.6096 ± 0.2727 |
| B4 = B1 + temperature | 0.9467 ± 0.0046 | 0.9470 ± 0.0046 | 0.1601 ± 0.0046 | 0.0730 ± 0.0020 | 0.0221 ± 0.0042 | 0.3985 ± 0.0560 |

### 4.2 Per-seed raw rows (verbatim from `vision_jevx_quality.json` / `vision_baselines.json`)

| seed | JevX acc | JevX F1 | JevX NLL | +temp NLL | +temp ECE | iso acc | B1 acc | B2 acc | B4 NLL |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.8900 | 0.8914 | 0.7376 | 0.3569 | 0.0285 | 0.8900 | 0.9440 | 0.9720 | 0.1576 |
| 1 | 0.9380 | 0.9386 | 0.5007 | 0.2320 | 0.0142 | 0.9420 | 0.9440 | 0.9640 | 0.1573 |
| 2 | 0.8980 | 0.9022 | 0.6594 | 0.3085 | 0.0196 | 0.9040 | 0.9520 | 0.9660 | 0.1654 |

### 4.3 Observed (measured) — stated without hedging

- **Every model is far above chance (0.100) and the majority baseline
  (0.106)** — the heads learned real signal through the frozen backbone; no
  near-chance result to flag.
- **The B2 MLP probe beats JevX on accuracy outright**: 0.9673 ± 0.0042 vs
  0.9087 ± 0.0257 (≈ 5.9 points, ≈ 29 of 500 test examples), seed ranges do
  **not** overlap (0.9631–0.9715 vs 0.8830–0.9344). On NLL and Brier likewise
  B2 is best. This is reported as measured: on this near-domain 10-class task,
  the plain MLP probe over the same frozen features is the stronger predictor;
  JevX's choice head stack does not close that gap at 800 training examples.
- **Probability quality is where JevX competes**: JevX+temperature measured
  the best ECE(15) of all six models (0.0207 vs 0.0221 for B4, 0.0260 for B2)
  and cut its own NLL from 0.6326 → 0.2991 and ECE from 0.0736 → 0.0207.
  Uncalibrated JevX is the weakest calibrator of the trained set (NLL 0.6326,
  MCE 0.5567) — stated, not buried.
- **Temperature preserved accuracy bit-identically in all 3 seeds**
  (0.8900 = 0.8900, 0.9380 = 0.9380, 0.8980 = 0.8980) — the argmax-preservation
  property reproduced on real image data.
- **Isotonic changed decisions in 2 of 3 seeds, both directions reported**:
  seed 0 unchanged (0.8900 → 0.8900), seed 1 up (0.9380 → 0.9420), seed 2 up
  (0.8980 → 0.9040); its NLL stayed worse than temperature in every seed.
- **B1's raw probabilities are poorly calibrated (ECE 0.1376 ± 0.0109)**;
  fitting temperature on the calibration split (B4) removes that (0.0221) —
  the same calibration story the text benchmark saw, reproduced on images.
- **Featurize-once is exact**: full-path vs cached logits max_abs_diff = 0.0
  for seeds 0–2; invariants held on all 39 eval batches of the JevX run
  (max |Σp − 1| = 2.4e-7) and all 117 batches of the combined run.
- Seed spread of JevX accuracy (0.8900–0.9380) is wider than the probes'
  (B1: 0.9440–0.9520) — head-init variance is visible at this sample size.

### 4.4 Latency (batch 1, warmup 10 + timed 100, `torch.no_grad`; shared host)

| Path | mean ms | std ms | min | max | Source |
|---|---|---|---|---|---|
| JevX `.choice` end-to-end (image processor + frozen DeiT + heads), one image ×100 | 439.425 | 366.527 | 176.420 | 1883.012 | `vision_baselines_latency.json` |
| **B1 linear head only** (cached 192-d feature → Linear) | 0.2566 | 0.3508 | 0.1109 | 2.6037 | `vision_baselines_latency.json` |

The e2e row was measured while a concurrent agent's test process shared the
host (std is large; min 176 ms is the closer-to-idle estimate). No
sub-millisecond precision is claimed for the head row either (its std exceeds
its mean; min 0.111 ms).

---

## 5. EXP-A — JevX over frozen wav2vec2-base × RAVDESS (3 seeds)

**Backbone.** `superb/wav2vec2-base-superb-ks` (a
`Wav2Vec2ForSequenceClassification` checkpoint on `facebook/wav2vec2-base`);
the wrap drops the task heads (load report: `classifier.*` / `projector.*` /
`layer_weights` UNEXPECTED, all ignorable) leaving 94,371,712 frozen
parameters (≤100M ✓). Features: `pooling_method=None` → the audio adapter's
default `masked_mean` over frames (wav2vec2 ships no pooler; this is the
family convention the adapter documents). Decision hidden width 768;
JevX trainable heads **13,382,401** parameters (measured).

**Data.** 1,440 clips, 8 emotion classes from filename prefixes
(neutral 96; all others 192 each), 16 kHz PCM-16, durations measured
2.936–5.272 s → each clip **trimmed to its first 3.0 s** (48,000 samples) to
satisfy the ≤3 s contract; exactly 2 clips are shorter than 3.0 s
(46,981 and 47,514 samples) — those two are what made the batched-processor
bug in §7.1 reachable. Train losses fell sharply but converged to about the
uniform level: 280.336 → 2.176, 219.703 → 2.563, 180.667 → 4.129 (uniform
reference for K=8: ln 8 ≈ 2.079 — observed, not interpreted away).
Chance = 0.125; majority-class baseline of the test split = 67/500 = 0.134.

### 5.1 Headline: metric × variant (mean ± sample std over 3 seeds)

| Model / variant | Accuracy | Macro-F1 | NLL ↓ | Brier ↓ | ECE(15) ↓ | MCE(15) ↓ |
|---|---|---|---|---|---|---|
| JevX uncalibrated | 0.1340 ± 0.0000 | 0.0318 ± 0.0038 | 2.6280 ± 0.9586 | 0.9655 ± 0.1604 | 0.1520 ± 0.1992 | 0.4023 ± 0.2489 |
| JevX + temperature | 0.1340 ± 0.0000 | 0.0318 ± 0.0038 | 2.0676 ± 0.0103 | 0.8717 ± 0.0032 | 0.0285 ± 0.0250 | 0.2263 ± 0.3672 |
| JevX + isotonic | 0.1347 ± 0.0321 | 0.0568 ± 0.0130 | 2.5734 ± 0.7062 | 1.0027 ± 0.2238 | 0.2283 ± 0.3043 | 0.5145 ± 0.3015 |
| B1 linear probe | 0.3433 ± 0.0136 | 0.2956 ± 0.0234 | 1.9161 ± 0.0017 | 0.8332 ± 0.0009 | 0.1655 ± 0.0183 | 0.3414 ± 0.0510 |
| **B2 MLP probe** | **0.3607 ± 0.0266** | **0.3037 ± 0.0506** | **1.7337 ± 0.0115** | **0.7774 ± 0.0037** | 0.0818 ± 0.0318 | 0.3681 ± 0.0906 |
| B4 = B1 + temperature | 0.3433 ± 0.0136 | 0.2956 ± 0.0234 | 1.8153 ± 0.0219 | 0.7961 ± 0.0077 | **0.0678 ± 0.0190** | 0.3184 ± 0.1797 |

### 5.2 Per-seed raw rows (verbatim from `audio_jevx_quality.json` / `audio_baselines.json`)

| seed | JevX acc | JevX F1 | JevX NLL | +temp NLL | +temp ECE | iso acc | B1 acc | B2 acc | B4 NLL |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.1340 | 0.0295 | 3.7347 | 2.0626 | 0.0198 | 0.1320 | 0.3480 | 0.3740 | 1.7909 |
| 1 | 0.1340 | 0.0362 | 2.0887 | 2.0794 | 0.0089 | 0.1680 | 0.3280 | 0.3300 | 1.8330 |
| 2 | 0.1340 | 0.0295 | 2.0606 | 2.0607 | 0.0566 | 0.1040 | 0.3540 | 0.3780 | 1.8221 |

### 5.3 Observed (measured) — stated without hedging, including the below-bar result

- **Explicitly below the credible bar: JevX accuracy is 0.1340 ± 0.0000 —
  exactly the test majority baseline (67/500 = 0.1340) and barely above
  chance (0.125) — in all three seeds, uncalibrated and temperature
  bit-identical, with macro-F1 0.0318, i.e. a degenerate single-class
  predictor.** This is a weak JevX head/backbone pairing on this task, and it
  is not hidden.
- **The rest of the pipeline demonstrably ran:** probability invariants held
  on every eval batch (36 JevX + 108 baseline batches, max |Σp − 1| =
  2.4e-7), cached-vs-full-path equivalence was **0.0 for all three seeds**,
  train loss decreased from 280.34/219.70/180.67 to 2.18/2.56/4.13, and —
  decisively — **B1/B2 on the identical frozen features and identical splits
  measured 0.3433 / 0.3607**, ≈2.6× the majority baseline. The features
  carry usable emotion signal; the JevX choice-head training under the fixed
  a-priori protocol (13,382,401 trainable parameters at width 768 fitted from
  scratch on n=700, epochs 4, lr 1e-3 — the same defaults as EXP-V, chosen
  before any run) converged to a near-uniform solution (final train losses
  ≈ ln 8). **No test-informed retuning was performed** (protocol §3 forbids
  it); the result stands as measured: on 8-class RAVDESS emotion, this
  head/protocol pairing is weak.
- **Temperature's low ECE (0.0285) is not a quality win here and must not be
  read as one:** it scales an almost non-discriminative predictor's
  probabilities (seed 0: NLL 3.7347 → 2.0626 ≈ ln 8 while accuracy stays
  0.1340). It is calibration of a majority/uniform predictor, stated as such.
- Isotonic changed decisions in every seed (0.1320 / 0.1680 / 0.1040 —
  down in seeds 0 and 2, up in seed 1; both directions reported) but remained near the
  bar; its MCE is the least stable (0.5145 ± 0.3015).
- **B2 wins every discriminative metric** (acc 0.3607, F1 0.3037, NLL
  1.7337, Brier 0.7774); B4 has the best ECE among the baselines
  (0.0678). Even the probes fit the train split only mildly (B1 loss
  2.06 → 1.92, B2 2.05 → 1.71) — 8-class emotion from 3 s clips at n=700 is
  hard for every model here; JevX is the one that failed to beat the
  majority baseline.
- Featurize-once cost 2594.68 s of the 3111.21 s wall; invariants and
  equivalence as above; `failures: {}`.

### 5.4 Latency (batch 1, warmup 10 + timed 100, `torch.no_grad`; shared host)

| Path | mean ms | std ms | min | max | Source |
|---|---|---|---|---|---|
| JevX `.choice` end-to-end (soundfile + feature extractor + frozen wav2vec2 + heads), one clip ×100 | 2302.151 | 803.838 | 1395.669 | 7617.896 | `audio_baselines_latency.json` |
| **B1 linear head only** (cached 768-d feature → Linear) | 0.2779 | 0.3103 | 0.1374 | 2.6749 | `audio_baselines_latency.json` |

Same shared-host caveats as §4.4: concurrent agent load inflates the e2e row
(min 1396 ms is the closer-to-idle estimate); no sub-millisecond precision is
claimed for the head row.



---

## 6. Interpretation scope / what these numbers do NOT claim

- Measured on this host, these splits, these seeds; no distribution-free or
  cross-domain claim; no statement about other backbones or larger training
  budgets (§76).
- Calibration statements are aggregate reliability properties of
  probabilities on the stated test split; they are not per-decision
  correctness claims (docs/PROOFS.md non-claims 1–2).
- Latency numbers come from a shared, concurrently loaded host; std is
  reported and no ranking closer than the stated noise is asserted.
- The two experiments are not compared to each other (different modalities,
  datasets, and K); each is compared only against its own same-feature
  baselines.

## 7. Bugs found and fixed

1. **`jevx/adapters/audio.py:35` (`_PaddedAudioProcessor`) and
   `jevx/adapters/audio.py:75` (`AudioAdapter._call_processor`) — cause:**
   the generic processor route calls `processor(payload, return_tensors="pt")`,
   but wav2vec2-style feature extractors default to `padding=False`; a BATCH of
   variable-length waveforms therefore fails with `ValueError: Unable to convert
   output 'input_values' ... inhomogeneous shape ... You can try: 1. Use
   padding=True` (probe on transformers 5.16.1). This dataset makes it
   reachable in practice: 2 of the 1,440 RAVDESS clips are shorter than 3.0 s
   (46,981 and 47,514 samples after the ≤3 s trim) while the rest trim to
   48,000 — any batch mixing them dies. **Fix (minimal, inside the sanctioned
   file):** an `AudioAdapter` override of `_call_processor` wraps the processor
   in a proxy that injects `padding=True` (retried once without it for
   callables that reject the keyword); `padding=True` pads with zeros to the
   batch max and, for this checkpoint's config (`return_attention_mask: true`),
   emits the input-length `attention_mask` the adapter's frame-mask resolver
   and masked-mean pooling convention require. Vision/state-less routing and
   single-payload calls are unchanged (verified: single clip output identical
   shape, non-audio states take the untouched generic path). **Verification:**
   `python -m pytest tests/unit -q` → **798 passed, 1 skipped in 90.84s** after
   the edit; the modality smoke test (12 synthetic images + 12 synthetic
   waveforms) passes; the audio benchmark run then completed with
   `failures: {}`.
2. **`benchmarks/run_modality.py:345` (`strip_root` / metadata key
   normalization) — cause:** imagenette metadata lines are prefixed
   `imagenette2-160/` while tar members use the same prefix; the first
   implementation mangled keys to `train/imagenette2-160/train/...`, so
   extraction reported `missing 80 rows` (caught by the first smoke run before
   any full run). **Fix:** normalize both sides to `train/<wnid>/<file>`;
   index now reports 9,469 train / 3,925 val / 0 unmatched.
3. **Finding (not a bug, protocol decision):** the DeiT checkpoint has no
   trained pooler (§4) — model-native pooling would be a per-reload random
   projection; `pooling_method="cls"` chosen and recorded.

## 8. Artifacts & reproduction

| File | Contents |
|---|---|
| `benchmarks/run_modality.py` | single entry point for EXP-V + EXP-A (splits, featurize-once, training, calibration, baselines, latency, invariants) |
| `benchmarks/results/vision_jevx_quality.json` | per-seed + aggregate JevX vision, equivalence, provenance, invariants |
| `benchmarks/results/vision_baselines.json` | per-seed B1/B2/B4 vision + aggregate |
| `benchmarks/results/vision_baselines_latency.json` | vision latency rows |
| `benchmarks/results/vision_run_summary.json` | failures {} , context timings, wall 593.03 s |
| `benchmarks/results/audio_*.json` | same four files for EXP-A |
| `tests/integration/test_modality_quality_smoke.py` | 12-sample synthetic image + waveform end-to-end choice run, invariants only, no network (2 tests, seconds) |

Reproduce from the repo root:

```
python benchmarks/run_modality.py --modality vision
python benchmarks/run_modality.py --modality audio --audio-feat-batch 16
python -m pytest tests/integration/test_modality_quality_smoke.py -q
python -m pytest tests/unit -q
```

First vision run re-downloads 99 MB (S3 archive) if the cache is cleared; the
audio run re-downloads ≈120 MB (Hub snapshot, pinned revision). Cached runs
reuse `benchmarks/.cache/` and the Hub cache.


---

## 9. EXP-H: choice-head v2 — appended 2026-09-24 (APPEND-ONLY)

Nothing above this line is edited. The head-quality diagnosis and the fix
are specified in ``reports/DATASET_BENCHMARKS.md`` §9 (head **v1** = as
shipped, head **v2** = additive state-only linear term + the unchanged
option-interaction residual, zero-initialized so an untrained head is
bit-identical to v1).

### 9.1 The audio finding (RAVDESS, EXP-A) — what was actually at fault

The published EXP-A result put the JevX choice head at **0.1340** —
exactly the majority-class rate — with **zero** seed variance, while B1 /
B2 on the *same* frozen wav2vec2 features reached 0.3433 / 0.3607. A head
sitting at the majority baseline with *identical* accuracy across three
seeds is not an under-performing head, it is a **degenerate one**: it is
predicting a constant.

The controlled re-run (``benchmarks/run_head_quality.py --datasets ravdess``,
identical EXP-A splits/protocol/seeds, features featurized ONCE and cached
under ``benchmarks/cache/head_quality/``) isolates the cause. Under one
identical protocol (700 train / 240 cal / 500 test, 3 seeds, 4 epochs, lr 1e-3,
batch 64):

| cell | accuracy | macro-F1 | NLL | seed std |
|---|---|---|---|---|
| P1 linear probe (published B1 = 0.3433) | 0.3467 | 0.2971 | 1.9176 | 0.0181 |
| P2 MLP probe (published B2 = 0.3607) | 0.3547 | 0.2993 | 1.7364 | 0.0258 |
| P3 JevX v1, dot | 0.1100 | 0.0266 | 2.3303 | 0.0381 |
| P3b JevX v1, bilinear (SHIPPED default) | 0.1680 | 0.0680 | 2.0337 | 0.0360 |
| P3c JevX v1, mlp | 0.1333 | 0.0295 | 2.0760 | 0.0012 |
| **P4 JevX v1, question+schema pathway BYPASSED** | **0.2407** | **0.1794** | 2.1580 | 0.0535 |
| P5 JevX v1, probe-initialized state pathway | 0.1380 | 0.0375 | 2.2759 | 0.0087 |
| P6 JevX v2 (the fix) | 0.1353 | 0.0339 | 2.3097 | 0.0058 |

**This overturns the working hypothesis, and that is the finding.** Every
full-stack JevX cell lands at or below the 0.1340 majority-class rate — the
head is degenerate, exactly as the published number said. But **P4, which keeps
the identical shipped interaction parameterization and only BYPASSES the
question/schema pathway, reaches 0.2407 — +7.3 pt over the shipped head and
well clear of the majority floor.** So on audio the head is **not** at fault:
the damage comes from the option-embedding pathway, not the scoring function.

P5 (probe-initialized state pathway) does **not** rescue audio (0.1380),
confirming again that initialization scale is not the binding constraint here.

**The fix therefore does NOT repair EXP-A, and this is reported as a negative
result rather than dressed up.** Head v2 moves RAVDESS by +0.7 pt (0.1333 →
0.1353 for the matching interaction), which is inside the seed noise
(std 0.001–0.006) and is not a win. The measured, actionable lever is the
option/schema pathway P4 removes: 0.1320 → 0.2407 by bypassing it alone.

Concrete next step, **not** implemented here (out of this agent''s file
ownership — it lives in ``jevx/questions/encoder.py`` / ``jevx/questions/schema.py`` /
``jevx/representations/fusion.py``, and the fusion option-gate formula is a
normative spec clause): on a text schema the option rows are learned and carry
class identity, but on an AUDIO/VISION schema the options are bare class
*names* hashed into an embedding table, and the shared fusion option formula
``e_k = g (*) (W_o o_k) + (1 - g) (*) c`` then mixes that low-information row in
with the state, which is what collapses the head. Damping the option-gate
contribution, or giving the head a state-independent option path, is the
indicated change — flagged for the owner of those files, not silently applied
here.

### 9.2 Vision (EXP-V, imagenette) — NOT re-run in this pass

EXP-V was left out deliberately: its frozen deit-tiny features are not
cached anywhere in the repo, so reproducing them costs another full
featurization pass, and the CPU budget on this host was already spent on
RAVDESS (≈45 min of the ≈2 h this section''s evidence took). The vision
number published in §4.1 (JevX 0.9087 vs B2 0.9673) is therefore **unchanged
and un-refuted** by this work; the head change is not claimed to affect it.
Because RAVDESS and ag_news were measured under one identical protocol, the
conclusions above stand on those two without the vision cell.


---

## 10. EXP-H follow-up: the option pathway on RAVDESS (APPENDED 2026-09-25)

**APPEND-ONLY.** Nothing above this line is edited, and the published EXP-A
numbers in 5.1-5.3 stand as measured. This section records the closure of the
concrete next step 9.1 handed over, plus the new head comparison it produced.
Source: `benchmarks/run_head_quality.py` (cells P7-P15), results under
`benchmarks/results/expH/` (candidates) and `benchmarks/results/expH/landed/`
(the shipped library, before/after). Same EXP-A protocol throughout: 700 train /
240 cal / 500 test, 3 seeds, 4 epochs, lr 1e-3, batch 64, AdamW, grad-clip 1.0,
masked cross-entropy, frozen wav2vec2 features featurized once and cached.

### 10.1 What was handed over, and what it turned out to be

9.1 left this: on an AUDIO schema the option rows are bare class names hashed
into an embedding table, the fusion option formula mixes that low-information row
into the state, and damping that contribution (or giving the head a
state-independent option path) is the indicated change.

The first half of that diagnosis was right and the second half was wrong, and the
measurement says which is which: the option rows do carry state, but the damage
is not the state leak (the across-k option signal is 3.3x the across-batch leak).
It is the **magnitude**. `schema_proj` is `LayerNorm(Linear(...))`, so every
option row arrives at norm `sqrt(768) = 27.7` whatever the option says, and those
rows are learnable. The head's residual `r_k = s^T W_b e_k` is then a product of
two full-scale 768-d vectors: at initialization, on these features,
`max abs z = 21.65`, `std(z) = 4.28` against a uniform NLL of 2.079. The
cross-entropy is saturated **before the first optimizer step**, the epoch-1 train
loss of every full-stack cell starts at 178-239 (the same stack with the option
pathway bypassed: 8.4), and the gradient that reaches the head - the only module
that can classify - is the smallest in the stack (L2 204, against 1115 for the
question projection).

### 10.2 RAVDESS head comparison, before and after (3 seeds each)

`before` is the shipped option formula (`option_normalize=False`); `after` is the
landed one (`option_normalize=True`), which is what
`Fusion.__init__(hidden_size, option_normalize=True)` now builds by default.

| head / cell | acc before | acc after | macro-F1 after | NLL before | NLL after | seed std acc |
|---|---|---|---|---|---|---|
| **shipped v2 head (default)** | 0.1353 +/- 0.0058 | **0.2220 +/- 0.0442** | 0.1255 | 2.3097 | **2.0787** | 0.0058 -> 0.0442 |
| v1 head (`state_pathway=False`) | 0.1680 +/- 0.0360 | **0.2520 +/- 0.0521** | 0.1813 | 2.0337 | **1.8956** | 0.0360 -> 0.0521 |
| P4 option pathway fully removed | 0.2407 +/- 0.0535 | unchanged (definitionally) | 0.1794 | 2.1580 | - | - |
| P1 linear probe (reference) | 0.3467 +/- 0.0181 | unchanged | 0.2971 | 1.9176 | - | - |
| P2 MLP probe (reference) | 0.3547 +/- 0.0258 | unchanged | 0.2993 | 1.7364 | - | - |
| majority class / chance | 0.1340 / 0.1250 | unchanged | - | - | - | - |

Three things this table says, all of them measured:

1. **The fix beats the bypass.** The shipped head goes 0.1353 -> 0.2220 (+8.67 pt)
   while the bypass that motivated the whole investigation only reaches 0.2407
   (+7.27 pt) - and the fixed head has a **lower** NLL than the bypass
   (2.0787 vs 2.1580). Keeping the schema pathway is not the problem; giving it a
   scale it is allowed to have is.
2. **The v1 head now beats the v2 head on audio** (0.2520 vs 0.2220). Head v2's
   additive state-only pathway was introduced to repair audio in EXP-H, and with
   the option pathway fixed it is no longer what audio was missing. The shipped
   default is **not** changed here: v2 stays the default because it is a strict
   superset of a linear probe and remains the better choice on the other
   benchmarks, and because switching the default on the strength of one dataset
   would be exactly the test-informed retuning 3 forbids. Both arms are reported.
3. **The gap to the probes is NOT closed.** 0.2220 against P1 0.3467 and P2 0.3547
   is still a 12.5 pt deficit, and 0.2520 is still 9.5 pt short. What closed is
   the part attributable to the option pathway; the remainder is the state-
   projection + fusion depth trained from scratch on n=700 for 4 epochs, which
   lives in files outside this change.

### 10.3 Cross-modality cost of the same change (3 seeds, identical protocol)

| dataset | modality | acc before | acc after | NLL before | NLL after |
|---|---|---|---|---|---|
| RAVDESS | audio | 0.1353 | **0.2220** | 2.3097 | **2.0787** |
| ag_news | text | 0.8258 | 0.8284 | 0.5092 | 0.5001 |
| synthetic Gaussian control | none | 0.9482 | 0.9462 | 0.1641 | 0.1796 |

ag_news is neutral (+0.26 pt, seed std 0.0200 -> 0.0126, NLL improved). The
synthetic control costs 0.20 pt of accuracy and 0.0155 of NLL - the one real
price of the change, reported as measured rather than smoothed. **Vision
(EXP-V) is not re-run**: its deit-tiny features are not cached in the repo, the
published 4.1 numbers (JevX 0.9087 vs B2 0.9673) are unchanged and un-refuted by
this work, and no claim is made about them.

### 10.4 What was tried and rejected (full table in `reports/ABLATIONS.md` 8.3)

Measured as one-variable patches of the option formula, 3 seeds, all three
datasets: zero-initialized option projection **0.1107** on RAVDESS (provably
degenerate - it collapses every option into a duplicate, the duplicate-tie then
zeroes the gradient to every parameter, first-batch NLL pinned at exactly ln 8);
LayerNorm on the option features **0.1473 / 0.1113** (it raises the already-large
norm and makes the initial saturation worse: `max abs z` 21.65 -> 30.53 / 49.14);
option-pathway dropout **0.1453** (noise on audio, -1.8 pt on the synthetic
control); a zero-init gain on the interaction residual **0.2107** (works, but
needs a head change and makes RAVDESS NLL *worse* at 2.9019, so it was not
shipped). The landed design is the scale-free variant: the option row enters as a
unit **direction**.

---

## 11. SUPERSESSION — the §5 EXP-A audio numbers re-run under the current default (APPENDED 2026-09-26)

**APPEND-ONLY.** Nothing above this line is edited. The published §5.1–§5.3
numbers (JevX uncalibrated **0.1340 ± 0.0000**, NLL 2.6280) stand as measured
in their original run and are **superseded, not corrected**, for the purpose of
"what does the current default score": they were captured under the pre-§5.f
option formula and the pre-v2 head, so the current default's number is a
different experiment and is reported here as a new one.

Source: `benchmarks/results/audio_jevx_quality_v2.json` (new). Re-run command,
verbatim defaults:

```
python benchmarks/run_modality.py --modality audio --seeds 0,1,2
```

i.e. the §3/§5 protocol unchanged — RAVDESS (`Codec-SUPERB/RAVDESS` rev
`46aa36c5fdeb1b6d63b746b74fde528a540044b2`), frozen
`superb/wav2vec2-base-superb-ks` (rev `372e0486cd83e6f0c05c20a27262e9ca09450d24`,
94,371,712 frozen params), 700 train / 240 cal / 500 test, split seed 0, seeds
{0,1,2}, `HeadTrainer` + `choice_loss`, **4 epochs, lr 1e-3, batch 64**,
AdamW, grad-clip 1.0. Wall 3004.63 s (featurize 2510.25 s; JevX stage
261.29 s), `failures: {}`. Environment unchanged: Python 3.14.3,
torch 2.14.0+cpu, transformers 5.16.1, datasets 4.4.1, numpy 2.3.5,
Windows-11-10.0.26100-SP0, CPU only.

### 11.1 The old number, the new number

| metric (uncalibrated, 3 seeds) | OLD — §5.1, `audio_jevx_quality.json` | NEW — §5.f re-run, `audio_jevx_quality_v2.json` | Δ |
|---|---|---|---|
| accuracy | 0.1340 ± 0.0000 | **0.2087 ± 0.0361** | **+0.0747** |
| macro-F1 | 0.0318 ± 0.0038 | **0.1250 ± 0.0338** | +0.0932 |
| NLL ↓ | 2.6280 ± 0.9586 | **1.9690 ± 0.0373** | −0.6590 |
| Brier ↓ | 0.9655 ± 0.1604 | **0.8553 ± 0.0171** | −0.1102 |
| ECE(15) ↓ | 0.1520 ± 0.1992 | **0.0881 ± 0.0323** | −0.0639 |
| MCE(15) ↓ | 0.4023 ± 0.2489 | **0.3674 ± 0.0389** | −0.0349 |
| JevX trainable params | 13,382,401 | **13,579,265** | +196,864 |

Per-seed accuracy **0.1740 / 0.2060 / 0.2460** (OLD: 0.1340 / 0.1340 /
0.1340). Per-seed NLL 2.011 / 1.956 / 1.940 (OLD: 3.735 / 2.089 / 2.061).
Train loss first → last 7.138→2.016, 6.557→2.017, 6.825→2.006 (OLD:
280.336→2.176, 219.703→2.563, 180.667→4.129). Calibrated arms: +temperature
0.2087 / NLL 1.9567 / ECE 0.0560; +isotonic 0.2280 / NLL 2.2602 / ECE 0.0605.
Cached-vs-full-path equivalence `max_abs_diff = 0.0` on all three seeds;
probability invariants on 108 eval batches, max |Σp − 1| = 2.384e-07.

**Protocol identity, verified rather than asserted.** The B1/B2/B4 baselines
measured *in this same run, on the same cached features and the same splits*,
reproduce the published §5.1 values to the digit: B1 0.34333 ± 0.01361,
B2 0.36067 ± 0.02663, B4 0.34333 with NLL 1.81534. The protocol did not move;
only the JevX code did. The v1 file is retained unmodified
(sha256 `8067270c1bdde6d6bb9033bb7b1a9851e77b77aae19e37a0c6054668a1cc623d`),
and the v2 file records the same digest in its own `supersession` block.

**What the delta means for §5.3.** §5.3's central claim — that the JevX arm sat
*exactly* on the majority baseline (67/500 = 0.1340) with zero seed variance
and macro-F1 0.032, i.e. a degenerate single-class predictor — **is no longer
true of the current default.** The current default scores 0.2087 ± 0.0361
(1.56× the 0.1340 majority rate, 0.67× chance, macro-F1 0.1250, real
seed-to-seed spread). The degenerate-predictor diagnosis, the
cross-entropy-saturation diagnosis of §10.1 (epoch-1 train loss 178–239) and
the "this head/protocol pairing is weak" conclusion all still describe the
**old** code, not the shipped one. The gap to the probes is **not** closed:
0.2087 against B1 0.3433 and B2 0.3607 is still a 13.5 pt deficit.

### 11.2 The code change between the two numbers

Two shipped changes, both after the original EXP-A run, both recorded verbatim
in the `supersession` block of `audio_jevx_quality_v2.json`:

1. **`option_normalize` (§5.f) — the option pathway.** `Fusion.__init__`
   now takes `option_normalize: bool = True` as the default
   (`jevx/representations/fusion.py:152`, and every subclass at
   `fusion.py:353, 366, 379, 394, 416, 448`), so each option row is reduced to
   a unit **direction** before `W_o` instead of entering at its raw
   `LayerNorm`-fixed norm. This is the fix §10.1 diagnosed and §10.2 landed
   (RAVDESS 0.1353 → 0.2220 on the head-quality protocol).
2. **The state path (head v2).** The choice head's additive state-only
   pathway is now the default: `JevXConfig.state_pathway`
   (`jevx/configuration_jevx.py:280`, default `"linear"`) with
   `LinearProbePathway` / `MLPStatePathway` (`jevx/decisions/generic.py:238`
   and `:349`), so the head scores `z_k = state_k + λ·r_k` instead of the v1
   interaction-only form. The +196,864 trainable parameters are exactly this
   pathway (768×256 + 256 = 196,864 at the shipped `"linear"` kind with the
   default `state_hidden_size`).

The published v1 run predates both, which is why its number is not the current
default's number. It is superseded **for the "current default" question only**;
as a record of the 2026-09-24 code it is unchanged and un-erroneous.

### 11.3 One correction to the A–F matrix framing, found by this re-run

`reports/MODALITY_BENCHMARKS.md` §9.1 calls the cell with
`state_pathway=False` the "SHIPPED default" head, and the A–F matrix's
column-C cell (`state_pathway="mlp"`, interaction ON) is treated as the
shipped stack in the WAVE-3a grid spec. The **actual shipped config default is
`state_pathway="linear"`** (`jevx/configuration_jevx.py:280`; verified by
constructing `JevXConfig(base_model_type="bert", decision_hidden_size=768)` →
`state_pathway='linear'`, `conditioning_mode='full'`,
`conditioning_enabled=True`). On this protocol the true shipped-default cell is
the A–F matrix's `A_linear_interact` = **0.2104 ± 0.0351** (5 seeds,
`benchmarks/results/head_ablation_ravdess.json`), which agrees with this
re-run's 0.2087 ± 0.0361 (3 seeds) and is 2.7 pt below the mlp cell
(0.2376 ± 0.0274). The `mlp` state pathway is available and mildly better on
RAVDESS; it is **not** the default, and no change to the default is proposed
here (selecting one on a test split is what §3 forbids).

### 11.4 What this supersession does NOT claim

- It does not claim 0.2087 is a better *model*; it claims it is what the
  current default scores on this protocol with these splits and seeds.
- It does not change §4 (EXP-V vision): those numbers were not re-run and are
  unchanged and un-refuted.
- It does not claim the §5.f change caused the whole +7.5 pt on its own. The
  head-quality protocol attributes 0.1353 → 0.2220 to `option_normalize` alone
  with the same head, and the state pathway adds the parameter delta; the split
  between the two is not re-measured here, and the two are not separable
  without a cell this section did not run.
- It does not validate any hyperparameter. lr 1e-3, 4 epochs and batch 64 are
  still the a-priori defaults; see `reports/AUDIO_DIAGNOSTIC.md` §5, where a
  20-epoch budget is measured at 0.5416 ± 0.0138 and is **specified, not
  shipped, and not adopted** (its value was read off the test split).
