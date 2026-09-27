# MULTI-QUESTION benchmark and conditioning-cache cost (WAVE-2)

This is the first multi-question evaluation in this repository. Every other
benchmark under `benchmarks/` asks **one** question per state, which is the
regime in which the question/schema encoders were already measured at 50.6-55.5 %
of the trainable parameters and 18.3-59.3x P2's latency for no measured benefit.
The encoders exist for the **multi-question** regime - `JevXForDecision.forward`
answers m questions about B states in one call - and that regime had never been
run. This report runs it, measures what
`JevXConfig.conditioning_mode="constant"` costs, and states a verdict.

Everything below was executed. Script: `benchmarks/run_multi_question.py`.
Result files: `benchmarks/results/multi_question_*.json`.
Feature cache: `benchmarks/cache/multi_question/`. Nothing under `jevx/`, and
nothing under `docs/` bar one appended pointer paragraph, was modified.

---

## 1. Environment

| field | value |
|---|---|
| python | 3.14.3 |
| platform | Windows-11-10.0.26100-SP0, Intel64 Family 6 Model 78, 4 cores |
| torch | 2.14.0+cpu, `torch.set_num_threads(1)`, `cuda_available=false`, dtype float32 |
| transformers | 5.16.1 |
| datasets | 4.4.1 |
| numpy | 2.3.5 |
| machine load during the run | 69-100 % - other agent processes were live throughout (`run_vision_diagnostic.py --phase featurize`, `run_level4_continue.py --stage quality`, `main.py selftest`) |

`torch.set_num_threads(1)` was forced: on a 4-core box at load 9-11 the default
2 intra-op threads thrash. A 64x128 `nn.Linear` costs 0.23 ms and a
`check_finite` 0.33 ms under the measured load, so the slowness at higher thread
counts is scheduler contention, not arithmetic.

**Contamination of the latency numbers.** PART 1's timings were taken on a
machine shared with other agents. The four arms are therefore timed
**round-robin inside one loop** (`off`, `hit`, `full`, `miss`, `off`, ...) so a
load spike lands on every arm, and the **median** (not the mean) is reported.
The min/p10/median/mean/std of every arm is in section 2.3 so the spread is
visible.

---

## 2. PART 1 - the `constant` conditioning-cache cost

### 2.1 Bitwise equality, independently confirmed

`"constant"` is a **content-addressed memo of H_Q / H_C** that only engages
outside training and outside autograd. Confirmed here on two independent
wrappers, on a **real** `JevXForDecision`:

| wrapper | cold == `full` | warm (memo hit) == `full` | after a content change == `full` | the content change actually moved the logits | max abs diff (all three) |
|---|---|---|---|---|---|
| tiny in-memory BERT + `tests/conftest.py` WordLevel tokenizer | true | true | true | true | 0.0 |
| `google/bert_uncased_L-2_H-128_A-2` + its real WordPiece tokenizer | true | true | true | true | 0.0 |

The last-but-one column is the control: without it, "everything is equal" would
also be true of a memo that never recomputes. The `after-content-change` cell
uses the same question with a different option text, and its logits differ from
the cold cell, so the miss really recomputed.

**This also holds at the metric level, over 5 seeds.** On ABSA the `full` and
`constant` arms produced identical accuracy, macro-F1, NLL, ECE and Brier to 4
decimals (0.7628 +- 0.0230 std, NLL 0.6536, ECE 0.1053, Brier 0.3554, per-seed
0.767 / 0.756 / 0.727 / 0.779 / 0.785 for both), and on RAVDESS emotion the
per-seed accuracies matched seed for seed (0.3020 / 0.3760 / 0.3800 / 0.2900 /
0.3440 at 4 epochs). That is structural, not luck:
`_condition_cache_enabled()` returns False whenever `self.training` **or**
`torch.is_grad_enabled()`, so the memo is **inference-only** and can never
change a training result or a metric. `constant` buys latency, never accuracy.

### 2.2 A stale-memo hazard found while doing this

The memo is keyed on **content, not on weights**. `clear_condition_cache()` is
called automatically by `load_state_dict`, but nothing calls it after an
in-place weight update. Measured on the tiny wrapper:

| protocol | constant == `full`? | max abs diff |
|---|---|---|
| eval -> one AdamW step -> clear the memo -> eval | true | 0.0 |
| eval -> one AdamW step -> eval **without** clearing | **false** | **4.1556e+04** |

The second row is a real, reachable ordering (any eval/train/eval loop that
reuses one model instance) and it serves stale H_Q/H_C. This is a property of
the shipped memo, not something this benchmark introduced. It was **not** fixed
here because `jevx/**` is out of scope for this wave: it is reported, not
patched.

### 2.3 Latency, batch 1, m = 4 questions, K = 3 options, frozen features

20 warmup + 120 timed iterations per arm, round-robin interleaved.

**tiny in-memory BERT + `tests/conftest.py` tokenizer**

| arm | median ms/example | p10 | mean | std | min | max |
|---|---|---|---|---|---|---|
| state-only (`conditioning_mode="off"`) | 9.842 | 6.966 | 9.974 | 2.270 | 6.548 | 15.356 |
| `constant` cache **HIT** | 7.407 | 4.581 | 7.473 | 2.342 | 4.442 | 14.573 |
| `constant` cache **MISS** | 8.518 | 5.776 | 8.573 | 1.857 | 5.538 | 14.235 |
| `full` | 10.996 | 7.765 | 11.020 | 2.315 | 7.353 | 17.553 |

**real backbone `google/bert_uncased_L-2_H-128_A-2`**

| arm | median ms/example | p10 | mean | std | min | max |
|---|---|---|---|---|---|---|
| state-only (`off`) | 11.079 | 7.838 | 10.844 | 2.001 | 7.194 | 14.711 |
| `constant` cache **HIT** | 7.302 | 4.794 | 7.736 | 2.577 | 4.651 | 15.167 |
| `constant` cache **MISS** | 8.789 | 6.248 | 8.974 | 1.984 | 5.994 | 13.523 |
| `full` | 12.060 | 9.171 | 12.038 | 2.045 | 8.137 | 17.323 |

### 2.4 The program's success criterion

> `constant`-hit <= 1.15x the state-only forward

| wrapper | `constant`-hit / `off` (median) | p10 variant | `constant`-miss / `off` | `full` / `off` | criterion met |
|---|---|---|---|---|---|
| tiny in-memory BERT | **0.753** | 0.658 | 0.865 | 1.117 | **YES** |
| real `bert_uncased_L-2_H-128_A-2` | **0.659** | 0.612 | 0.793 | 1.088 | **YES** |

**The criterion is met, and the hit arm is 25-34 % FASTER than the state-only
forward. That is not a paradox, it is a measurement of what `off` actually is.**
`conditioning_mode="off"` zeroes the option-interaction residual but does
**not** skip the encoders: `modeling_jevx.py` computes `option_features` and only
then gates it at the fusion call (`option_features if condition else None`), and
the question rows are passed unconditionally. So `off` still pays the full
question and schema encoding, which is why it lands within 9-12 % of `full`, and
why a cache that skips both encoders beats it. The honest reading:

* the `constant` memo removes the whole conditioning-encoder cost of a call;
* the shipped state-only path does not remove it either.

A like-for-like "no-cache vs cache at equal work" ratio is **not** available
from this table, because the shipped `off` path is not a no-conditioning-encoding
path. That is why the grid adds a `state_only` arm which also bypasses the
question, and that arm is the true state-only reference for the quality
question.

---

## 3. PART 2 - the multi-question suite

### 3.1 Datasets, ids, resolved revisions

Every hub id was resolved through the HF API before download, and the resolved
commit sha is what the code pins. `SemEvalWorkshop/sem_eval_2014_task_14` and
`yuehchn/MAMS` both return **HTTP 401** anonymously from this host and are
therefore unreachable here; the two SemEval-2014 Task 4 mirrors below are the
reachable equivalents and carry the same per-aspect polarity field.

| dataset | hub id | resolved revision | backbone | feature dim |
|---|---|---|---|---|
| ABSA (SemEval-2014 Task 4) restaurant | `xinyixiuxiu/semeval2014_restaurant` | `709b40b7f0a9ed142df433c0d84aaa14d0d2e8c2` | `google/bert_uncased_L-2_H-128_A-2`, `cls`, frozen | 128 |
| ABSA (SemEval-2014 Task 4) laptop | `xinyixiuxiu/semeval2014_laptop` | `fed294ac48b1b9f903afb833baeeb45754fe7e2c` | (same) | 128 |
| RAVDESS (multi) | `Codec-SUPERB/RAVDESS` | `46aa36c5fdeb1b6d63b746b74fde528a540044b2` | `superb/wav2vec2-base-superb-ks`, native pooling, frozen | 768 |
| ag_news | `fancyzhx/ag_news` | `eb185aade064a813bc0b7f42de02595523103ca4` | `google/bert_uncased_L-2_H-128_A-2`, `cls`, frozen | 128 |
| Yelp | `Yelp/yelp_review_full` | `c1f9ee939b7d05667af864ee1cb066393154bf85` | (same) | 128 |

Featurize-once, with the backbone frozen, per the established Mode-A protocol.
RAVDESS features were **reused** from
`benchmarks/cache/head_quality/ravdess_features.npz` after verifying, per split,
that the recomputed split's emotion-label counts are identical to the cached
ones. ag_news and Yelp ship 120k and 650k training rows; a deterministic
stratified pool subsample of 6000 (`bc.stratified_indices(labels, 6000, seed=0)`)
is taken **before** featurization, because the split can use at most
train+cal+test rows.

### 3.2 Per-question label provenance - REAL vs DERIVED, stated per question

| dataset | question | K | options | provenance | why |
|---|---|---|---|---|---|
| ABSA | aspect polarity (per aspect term) | 3 | negative / neutral / positive | **REAL** | SemEval-2014 Task 4 ships a polarity label for every (sentence, aspect term) pair. Nothing is annotated or derived here. The question text is templated on the aspect term, so each state carries one genuinely different question per aspect it contains. |
| RAVDESS | `ravdess_emotion` | 8 | the 8 RAVDESS emotion codes | **REAL** | field 0 of the filename, shipped by the dataset |
| RAVDESS | `ravdess_gender` | 2 | man / woman | **REAL** | field 1 (`spk_NN`); RAVDESS documents odd actors 01-20 as male, even as female |
| RAVDESS | `ravdess_intensity` | 3 | medium / high / low | **REAL** | field 2 (`intensity_NN`): 01=medium, 02=high, 03=low |
| RAVDESS | `ravdess_statement` | 2 | statement / question | **REAL** | field 3 (`statement_NN`): 01=statement, 02=question |
| ag_news | `ag_news_topic` | 4 | World / Sports / Business / Sci/Tech | **REAL** | ag_news's own 4-way annotation |
| ag_news | `ag_news_is_sports` | 2 | no / yes | **DERIVED** | deterministic rule `label == 1` |
| ag_news | `ag_news_is_scitech` | 2 | no / yes | **DERIVED** | deterministic rule `label == 3` |
| ag_news | `ag_news_is_business` | 2 | no / yes | **DERIVED** | deterministic rule `label == 2` |
| Yelp | `yelp_stars` | 5 | 1..5 stars | **REAL** | Yelp's own star field |
| Yelp | `yelp_detractor` | 2 | no / yes | **DERIVED** | deterministic rule `stars == 1`. The brief calls this real; it is a threshold on a REAL dataset label, so it is recorded here as DERIVED-from-a-real-label, which is the distinction that matters. |

### 3.3 Questions per state, and what was OMITTED

| dataset | questions per state | distinct question strings actually sent to the model | states (train/cal/test) |
|---|---|---|---|
| ABSA | 4-13 (mean 4.65; 257 sentences carry >= 4 distinct aspect terms = 27 % of all SemEval-2014 restaurant+laptop sentences) | one per distinct aspect term (hundreds) | 180 / 39 / 38 |
| RAVDESS | exactly 4 | 4 | 700 / 240 / 500 |
| ag_news | exactly 4 | 4 | 1500 / 500 / 1000 |
| Yelp | exactly 2 | 2 | 1500 / 500 / 1000 |

**Yelp aspect questions are OMITTED, deliberately.** `Yelp/yelp_review_full`
ships exactly one label field - the star rating. It carries no aspect
annotation, and no aspect-labelled source is reachable from it. Hand-writing
aspect labels and calling them real would be annotating a benchmark by hand,
which the brief explicitly forbids. The only further obtainable "questions" are
restatements of the same star label ("is it 4 or 5 stars", "at most 2 stars"),
which add no information and would inflate the multi-question appearance
without content. Yelp therefore carries 2 questions, not 4, and says so.

**ag_news's 4 questions are 1 REAL + 3 DERIVED.** The three derived questions
are deterministic thresholds on the single REAL topic label, so they are not
independent information; ag_news's aggregate is a weak instrument and is
reported separately from ABSA and RAVDESS.

### 3.4 Splits and protocol

Splits are over **STATES**, never over (state, question) pairs, so a state can
never contribute to two splits - the multi-question regime would leak its own
answers otherwise. The script asserts train/cal/test disjointness at load time.
Stratification is on the dataset's primary task label (RAVDESS emotion, ag_news
topic, Yelp stars); ABSA has no sentence-level label, so it is stratified on the
binned number of aspect questions per state. Split seed 0, 5 training seeds
(0-4), `lr=1e-3` AdamW, batch 64 (state, question) pairs, `choice_loss`,
ECE/Brier on 15 equal-width bins, all metrics from `jevx.utils.numerical` via
`bench_common`.

Every arm of a cell starts from **bitwise-identical head weights**
(`torch.manual_seed(1234)` before the wrapper is built), so the arms are paired
by construction, not by luck.

**One question per call.** `JevXForDecision.forward` shares one question across
the whole state batch, so a batch is a set of states asking the SAME question;
the option count is uniform and no padded option column is ever produced. (This
matters: a hand-built m-question call with heterogeneous K leaves the padded
option columns carrying real logits, not `-inf` - measured here, max abs 6.76e-1
on the padded cells.) Section 4 shows this protocol is the same computation as
the genuine multi-question forward.

### 3.5 DEVIATIONS, stated explicitly

The box is 4 shared cores at 69-100 % load, and the established 4-epoch
protocol did not fit the wall-clock budget for all four datasets. Every
deviation is listed here and mirrored in the result JSON.

| dataset | deviation | reason (measured) | what it does not establish |
|---|---|---|---|
| ABSA | none: 4 epochs, 5 seeds, 6 arms (incl. `constant`) | - | - |
| RAVDESS | **2 epochs** (not 4) | one (arm, seed) cell costs ~60-100 s at 69-100 % load; 4 questions x 5 arms x 5 seeds x 4 epochs did not fit | the RAVDESS ranking at 4 epochs. The single 4-epoch spot check that did run gave `ravdess_emotion` `full` 0.3384 +- 0.0414 and `constant` identical seed-for-seed, consistent with the 2-epoch result |
| RAVDESS | **head width 256, not the 768 feature width** | at 768 one (arm, seed) cell costs ~270 s | the audio head at full width. The width is IDENTICAL across all four JevX arms here, so the within-dataset comparison is unaffected |
| ag_news, Yelp | **1500/500/1000** splits (the contract states 3000/1000/1500 as UPPER BOUNDS, so this is inside the contract, not a breach) | at 3000 the topic question alone is ~170 s per (arm, seed) | - |
| RAVDESS, ag_news, Yelp | **`constant` arm not run separately** | `constant` is inference-only and bitwise-equal to `full` (section 2.1), measured identical over 5 seeds on ABSA and on RAVDESS emotion | nothing - it is labelled NOT RUN everywhere it appears, never reported as a result |
| all | calibration split unused | the grid trains on train and scores on test; the cal split is carved and size-checked but no calibrator is fitted, because calibration is not the object of study here | any calibration claim |

Wall time: ABSA 935.6 s, RAVDESS 928.4 s, ag_news 1235.9 s, Yelp 1004.1 s, plus
~150 s for the cache-cost phase, ~60 s shared-state, ~90 s self-check and ~900 s
for the gradient-norm pass - about 1.6 h of executed compute, most of it spent
waiting on a contended box.

---

## 4. PART 3 - the shared-state property

`register_forward_hook` on the **real** backbone
(`google/bert_uncased_L-2_H-128_A-2`), one state, m questions in ONE call:

| m | backbone forward hooks | decisions returned |
|---|---|---|
| 1 | **1** | 1 |
| 2 | **1** | 2 |
| 3 | **1** | 3 |
| 4 | **1** | 4 |
| 5 | **1** | 5 |

**With 4+ questions on ONE state the shared-state property still holds exactly:
one backbone forward.** The extra questions cost the question/schema encoders
and the head, not the backbone.

**And the grid's per-question protocol is the same computation.** Comparing the
per-question logits taken from one 5-question call against the same five
questions asked one at a time: **argmax agrees on 5/5**, and the per-question
max-abs logit differences are 1.19e-07, 1.49e-07, 1.49e-07, 3.58e-07, 7.45e-08 -
**worst case 3.58e-07**, four orders of magnitude below the metric resolution and
far under the 1e-05 float-reassociation threshold this script asserts against.
The 5-question call pads every schema to K_max and runs one batched option
projection and duplicate-tie, while the single-question calls run narrower
tensors, so BLAS picks different kernels and float32 re-associates. That is the
same class of difference `bench_common.equivalence_check()` already records. The
multi-question regime is therefore a **shared-parameter** regime, not a
different computation - which is exactly what makes the question/schema
encoders' generalisation testable at all.

### Self-check that the harness is sound

| check | result |
|---|---|
| the frozen-feature path is not a constant function of the weights (a zero feature table must NOT reproduce a real backbone pass) | max abs diff 3.31e-02 - **passed** |
| featurize-once, then replay the same feature through the frozen path | max abs diff **0.0**, bitwise equal |
| `constant` bitwise cells, tiny and real backbone | all true (section 2.1) |

---

## 5. The multi-question grid

11 questions over 4 datasets, 5 seeds each, `full` / `off` / `state_only` /
P1-linear / P2-MLP on identical frozen features, splits and initial weights.
`constant` appears only on ABSA (section 3.5).

### 5.1 Per-question tables

`acc` is accuracy mean +- std over the 5 seeds; the range is min-max over the
same 5 seeds. `constant` rows read **NOT RUN** rather than FAILED - the arm was
deliberately dropped for wall clock, with the reason in 3.5.

#### ABSA - aspect polarity (REAL), 4 epochs

| question | prov | arm | acc mean+-std | acc range | per-seed acc | macro-F1 | NLL | ECE(15) | Brier(15) | trainable |
|---|---|---|---|---|---|---|---|---|---|---|
| absa_aspect_polarity | REAL | full | 0.7628+-0.0230 | [0.727, 0.785] | 0.767, 0.756, 0.727, 0.779, 0.785 | 0.6324 | 0.6536 | 0.1053 | 0.3554 | 1280385 |
| absa_aspect_polarity | REAL | constant | 0.7628+-0.0230 | [0.727, 0.785] | 0.767, 0.756, 0.727, 0.779, 0.785 | 0.6324 | 0.6536 | 0.1053 | 0.3554 | 1280385 |
| absa_aspect_polarity | REAL | off | 0.7174+-0.0348 | [0.657, 0.744] | 0.744, 0.733, 0.733, 0.721, 0.657 | 0.5918 | 0.6354 | 0.0980 | 0.3769 | 1280385 |
| absa_aspect_polarity | REAL | state_only | 0.7221+-0.0258 | [0.680, 0.744] | 0.744, 0.715, 0.733, 0.738, 0.680 | 0.5920 | 0.6258 | 0.1147 | 0.3712 | 1280385 |
| absa_aspect_polarity | REAL | p1_linear | 0.7128+-0.0106 | [0.698, 0.727] | 0.715, 0.698, 0.715, 0.727, 0.709 | 0.4485 | 0.7378 | 0.1961 | 0.4190 | 387 |
| absa_aspect_polarity | REAL | p2_mlp | 0.8000+-0.0052 | [0.791, 0.802] | 0.802, 0.802, 0.791, 0.802, 0.802 | 0.6650 | 0.6277 | 0.1714 | 0.3504 | 16899 |

#### RAVDESS - four REAL filename-encoded questions, 2 epochs

| question | prov | arm | acc mean+-std | acc range | per-seed acc | macro-F1 | NLL | ECE(15) | Brier(15) | trainable |
|---|---|---|---|---|---|---|---|---|---|---|
| ravdess_emotion | REAL | full | 0.2468+-0.0609 | [0.160, 0.328] | 0.160, 0.228, 0.328, 0.264, 0.254 | 0.1733 | 1.8685 | 0.0750 | 0.8256 | 3084801 |
| ravdess_emotion | REAL | constant | NOT RUN (3.5) | | | | | | | |
| ravdess_emotion | REAL | off | 0.2792+-0.0226 | [0.250, 0.308] | 0.288, 0.286, 0.250, 0.264, 0.308 | 0.1947 | 1.8025 | 0.0715 | 0.8046 | 3084801 |
| ravdess_emotion | REAL | state_only | 0.2880+-0.0253 | [0.246, 0.314] | 0.290, 0.292, 0.298, 0.246, 0.314 | 0.2087 | 1.7918 | 0.0497 | 0.8013 | 3084801 |
| ravdess_emotion | REAL | p1_linear | 0.2800+-0.0459 | [0.218, 0.328] | 0.328, 0.322, 0.258, 0.274, 0.218 | 0.2232 | 1.9814 | 0.1176 | 0.8509 | 6152 |
| ravdess_emotion | REAL | p2_mlp | 0.3096+-0.0284 | [0.262, 0.338] | 0.316, 0.262, 0.338, 0.312, 0.320 | 0.2200 | 1.8738 | 0.0999 | 0.8216 | 596744 |
| ravdess_gender | REAL | full | 0.7296+-0.0687 | [0.652, 0.808] | 0.682, 0.712, 0.808, 0.652, 0.794 | 0.7076 | 0.6610 | 0.1743 | 0.4203 | 3084801 |
| ravdess_gender | REAL | constant | NOT RUN (3.5) | | | | | | | |
| ravdess_gender | REAL | off | **0.8988+-0.0535** | [0.832, 0.958] | 0.958, 0.832, 0.910, 0.856, 0.938 | 0.8971 | 0.2579 | 0.0449 | 0.1516 | 3084801 |
| ravdess_gender | REAL | state_only | 0.8864+-0.0947 | [0.754, 0.960] | 0.960, 0.818, 0.754, 0.942, 0.958 | 0.8820 | 0.2574 | 0.0533 | 0.1632 | 3084801 |
| ravdess_gender | REAL | p1_linear | 0.7740+-0.0542 | [0.710, 0.830] | 0.710, 0.808, 0.830, 0.722, 0.800 | 0.7684 | 0.6099 | 0.2111 | 0.4180 | 1538 |
| ravdess_gender | REAL | p2_mlp | 0.8280+-0.0652 | [0.748, 0.892] | 0.862, 0.870, 0.768, 0.748, 0.892 | 0.8234 | 0.4467 | 0.1402 | 0.2804 | 592130 |
| ravdess_intensity | REAL | full | 0.6788+-0.0421 | [0.616, 0.722] | 0.722, 0.616, 0.712, 0.666, 0.678 | 0.6581 | 0.6066 | 0.0850 | 0.4143 | 3084801 |
| ravdess_intensity | REAL | constant | NOT RUN (3.5) | | | | | | | |
| ravdess_intensity | REAL | off | 0.5816+-0.1064 | [0.470, 0.708] | 0.708, 0.470, 0.532, 0.516, 0.682 | 0.4869 | 0.6700 | 0.1465 | 0.4774 | 3084801 |
| ravdess_intensity | REAL | state_only | 0.6492+-0.1032 | [0.470, 0.720] | 0.688, 0.712, 0.656, 0.470, 0.720 | 0.6024 | 0.6456 | 0.1365 | 0.4525 | 3084801 |
| ravdess_intensity | REAL | p1_linear | **0.7004+-0.0067** | [0.692, 0.708] | 0.696, 0.692, 0.708, 0.706, 0.700 | 0.6964 | 0.6285 | 0.1424 | 0.4284 | 2307 |
| ravdess_intensity | REAL | p2_mlp | 0.6744+-0.0533 | [0.606, 0.726] | 0.726, 0.714, 0.696, 0.606, 0.630 | 0.6497 | 0.5965 | 0.0673 | 0.4109 | 592899 |
| ravdess_statement | REAL | full | 0.5432+-0.0947 | [0.458, 0.688] | 0.688, 0.552, 0.458, 0.458, 0.560 | 0.4166 | 0.6810 | 0.1111 | 0.4883 | 3084801 |
| ravdess_statement | REAL | constant | NOT RUN (3.5) | | | | | | | |
| ravdess_statement | REAL | off | 0.6380+-0.0918 | [0.490, 0.712] | 0.706, 0.670, 0.490, 0.712, 0.612 | 0.5713 | 0.6381 | 0.1361 | 0.4464 | 3084801 |
| ravdess_statement | REAL | state_only | **0.6716+-0.1417** | [0.504, 0.886] | 0.716, 0.618, 0.504, 0.886, 0.634 | 0.6232 | 0.5856 | 0.1351 | 0.4063 | 3084801 |
| ravdess_statement | REAL | p1_linear | 0.5696+-0.0372 | [0.522, 0.610] | 0.610, 0.594, 0.540, 0.522, 0.582 | 0.5239 | 0.6671 | 0.0944 | 0.4744 | 1538 |
| ravdess_statement | REAL | p2_mlp | 0.6684+-0.0903 | [0.540, 0.788] | 0.788, 0.640, 0.674, 0.540, 0.700 | 0.6488 | 0.6157 | 0.1288 | 0.4265 | 592130 |

#### ag_news - 1 REAL + 3 DERIVED questions, 2 epochs

| question | prov | arm | acc mean+-std | acc range | per-seed acc | macro-F1 | NLL | ECE(15) | Brier(15) | trainable |
|---|---|---|---|---|---|---|---|---|---|---|
| ag_news_topic | REAL | full | 0.8486+-0.0139 | [0.828, 0.865] | 0.843, 0.828, 0.852, 0.865, 0.855 | 0.8478 | 0.4377 | 0.0324 | 0.2286 | 1280385 |
| ag_news_topic | REAL | constant | NOT RUN (3.5) | | | | | | | |
| ag_news_topic | REAL | off | 0.8568+-0.0100 | [0.843, 0.868] | 0.858, 0.868, 0.843, 0.864, 0.851 | 0.8564 | 0.4116 | 0.0264 | 0.2160 | 1280385 |
| ag_news_topic | REAL | state_only | 0.8574+-0.0109 | [0.842, 0.871] | 0.858, 0.871, 0.842, 0.863, 0.853 | 0.8569 | 0.4126 | 0.0272 | 0.2161 | 1280385 |
| ag_news_topic | REAL | p1_linear | 0.7992+-0.0142 | [0.786, 0.817] | 0.791, 0.786, 0.790, 0.817, 0.812 | 0.7973 | 0.7774 | 0.2738 | 0.4035 | 516 |
| ag_news_topic | REAL | p2_mlp | **0.8580+-0.0052** | [0.853, 0.865] | 0.862, 0.865, 0.853, 0.855, 0.855 | 0.8575 | 0.4370 | 0.0711 | 0.2264 | 17028 |
| ag_news_is_sports | DERIVED | full | 0.9664+-0.0100 | [0.949, 0.973] | 0.973, 0.949, 0.969, 0.973, 0.968 | 0.9560 | 0.1009 | 0.0157 | 0.0526 | 1280385 |
| ag_news_is_sports | DERIVED | constant | NOT RUN (3.5) | | | | | | | |
| ag_news_is_sports | DERIVED | off | 0.9664+-0.0181 | [0.934, 0.976] | 0.973, 0.934, 0.976, 0.975, 0.974 | 0.9564 | 0.1017 | 0.0142 | 0.0538 | 1280385 |
| ag_news_is_sports | DERIVED | state_only | 0.9700+-0.0102 | [0.952, 0.976] | 0.973, 0.952, 0.976, 0.976, 0.973 | 0.9605 | 0.0964 | 0.0148 | 0.0502 | 1280385 |
| ag_news_is_sports | DERIVED | p1_linear | 0.9330+-0.0124 | [0.920, 0.948] | 0.932, 0.948, 0.922, 0.943, 0.920 | 0.9042 | 0.2434 | 0.1096 | 0.1276 | 258 |
| ag_news_is_sports | DERIVED | p2_mlp | **0.9718+-0.0038** | [0.968, 0.976] | 0.968, 0.968, 0.976, 0.975, 0.972 | 0.9619 | 0.0920 | 0.0183 | 0.0463 | 16770 |
| ag_news_is_scitech | DERIVED | full | 0.9068+-0.0110 | [0.888, 0.914] | 0.913, 0.914, 0.888, 0.913, 0.906 | 0.8731 | 0.2473 | 0.0321 | 0.1410 | 1280385 |
| ag_news_is_scitech | DERIVED | constant | NOT RUN (3.5) | | | | | | | |
| ag_news_is_scitech | DERIVED | off | 0.9000+-0.0177 | [0.871, 0.916] | 0.903, 0.916, 0.871, 0.912, 0.898 | 0.8684 | 0.2426 | 0.0220 | 0.1446 | 1280385 |
| ag_news_is_scitech | DERIVED | state_only | 0.9004+-0.0202 | [0.866, 0.916] | 0.907, 0.916, 0.866, 0.913, 0.900 | 0.8694 | 0.2458 | 0.0236 | 0.1466 | 1280385 |
| ag_news_is_scitech | DERIVED | p1_linear | 0.8734+-0.0124 | [0.854, 0.884] | 0.884, 0.884, 0.854, 0.870, 0.875 | 0.8088 | 0.3257 | 0.0728 | 0.1932 | 258 |
| ag_news_is_scitech | DERIVED | p2_mlp | **0.9148+-0.0041** | [0.909, 0.920] | 0.909, 0.914, 0.920, 0.914, 0.917 | 0.8818 | 0.2297 | 0.0197 | 0.1347 | 16770 |
| ag_news_is_business | DERIVED | full | 0.8856+-0.0066 | [0.878, 0.893] | 0.883, 0.893, 0.878, 0.882, 0.892 | 0.8416 | 0.2793 | 0.0312 | 0.1647 | 1280385 |
| ag_news_is_business | DERIVED | constant | NOT RUN (3.5) | | | | | | | |
| ag_news_is_business | DERIVED | off | **0.8904+-0.0019** | [0.888, 0.893] | 0.889, 0.888, 0.893, 0.891, 0.891 | 0.8508 | 0.2674 | 0.0215 | 0.1628 | 1280385 |
| ag_news_is_business | DERIVED | state_only | 0.8886+-0.0027 | [0.886, 0.893] | 0.893, 0.888, 0.889, 0.887, 0.886 | 0.8481 | 0.2677 | 0.0205 | 0.1629 | 1280385 |
| ag_news_is_business | DERIVED | p1_linear | 0.8228+-0.0125 | [0.801, 0.833] | 0.833, 0.801, 0.827, 0.828, 0.825 | 0.7003 | 0.3775 | 0.0589 | 0.2372 | 258 |
| ag_news_is_business | DERIVED | p2_mlp | 0.8920+-0.0043 | [0.887, 0.897] | 0.896, 0.897, 0.887, 0.890, 0.890 | 0.8494 | 0.2697 | 0.0199 | 0.1622 | 16770 |

#### Yelp - 1 REAL + 1 DERIVED question, 2 epochs

| question | prov | arm | acc mean+-std | acc range | per-seed acc | macro-F1 | NLL | ECE(15) | Brier(15) | trainable |
|---|---|---|---|---|---|---|---|---|---|---|
| yelp_stars | REAL | full | 0.3688+-0.0077 | [0.359, 0.378] | 0.378, 0.359, 0.363, 0.371, 0.373 | 0.3374 | 1.4237 | 0.0588 | 0.7270 | 1280385 |
| yelp_stars | REAL | constant | NOT RUN (3.5) | | | | | | | |
| yelp_stars | REAL | off | **0.3836+-0.0146** | [0.363, 0.399] | 0.399, 0.363, 0.385, 0.376, 0.395 | 0.3640 | 1.3993 | 0.0442 | 0.7143 | 1280385 |
| yelp_stars | REAL | state_only | 0.3816+-0.0087 | [0.367, 0.390] | 0.383, 0.367, 0.386, 0.390, 0.382 | 0.3578 | 1.3965 | 0.0396 | 0.7129 | 1280385 |
| yelp_stars | REAL | p1_linear | 0.3186+-0.0277 | [0.274, 0.343] | 0.324, 0.313, 0.339, 0.274, 0.343 | 0.3072 | 1.5232 | 0.0336 | 0.7647 | 645 |
| yelp_stars | REAL | p2_mlp | 0.3568+-0.0072 | [0.345, 0.362] | 0.361, 0.355, 0.361, 0.345, 0.362 | 0.3369 | 1.4461 | 0.0329 | 0.7311 | 17157 |
| yelp_detractor | DERIVED | full | 0.8128+-0.0078 | [0.799, 0.817] | 0.799, 0.817, 0.817, 0.815, 0.816 | 0.6568 | 0.4019 | 0.0432 | 0.2571 | 1280385 |
| yelp_detractor | DERIVED | constant | NOT RUN (3.5) | | | | | | | |
| yelp_detractor | DERIVED | off | 0.8062+-0.0174 | [0.777, 0.820] | 0.809, 0.806, 0.777, 0.820, 0.819 | 0.6117 | 0.4164 | 0.0537 | 0.2681 | 1280385 |
| yelp_detractor | DERIVED | state_only | 0.8078+-0.0113 | [0.796, 0.819] | 0.797, 0.808, 0.796, 0.819, 0.819 | 0.6161 | 0.4117 | 0.0501 | 0.2651 | 1280385 |
| yelp_detractor | DERIVED | p1_linear | 0.8038+-0.0015 | [0.802, 0.806] | 0.804, 0.803, 0.802, 0.806, 0.804 | 0.4821 | 0.4516 | 0.0353 | 0.2881 | 258 |
| yelp_detractor | DERIVED | p2_mlp | **0.8188+-0.0033** | [0.814, 0.823] | 0.819, 0.818, 0.820, 0.814, 0.823 | 0.5884 | 0.4090 | 0.0260 | 0.2600 | 16770 |

### 5.2 Aggregate over questions (mean of the per-question 5-seed mean)

| dataset | full | constant | off | state_only | p1_linear | p2_mlp |
|---|---|---|---|---|---|---|
| absa (1 question, 4 epochs) | 0.7628 | 0.7628 | 0.7174 | 0.7221 | 0.7128 | **0.8000** |
| ravdess (4 questions, 2 epochs) | 0.5496 | NOT RUN | 0.5994 | **0.6238** | 0.5810 | 0.6201 |
| ag_news (4 questions, 2 epochs) | 0.9019 | NOT RUN | 0.9034 | 0.9041 | 0.8571 | **0.9092** |
| yelp (2 questions, 2 epochs) | 0.5908 | NOT RUN | 0.5949 | 0.5947 | 0.5612 | 0.5878 |
| **pooled over all 11 questions** | **0.7046** | - | **0.7199** | **0.7294** | 0.6898 | **0.7357** |

Pooling is the unweighted mean over all 11 questions, so RAVDESS (4) and ag_news
(4) weigh twice a single-question dataset. The ordering is the same on every
dataset except ABSA: **`p2_mlp` > `state_only` > `off` > `full`**.

### 5.3 The decisive comparison, every cell

> does at least one question on at least one dataset show a conditioning mode
> beating `off` / `state_only` with NON-OVERLAPPING 5-seed accuracy ranges?

35 paired comparisons were computed. **0 wins.** Exactly ONE comparison had
non-overlapping 5-seed ranges, and it is a loss.

| dataset | question | conditioned | baseline | cond mean+-std | base mean+-std | delta pp | disjoint | verdict |
|---|---|---|---|---|---|---|---|---|
| absa | aspect polarity | full | off | 0.7628+-0.0230 | 0.7174+-0.0348 | +4.53 | false | TIE |
| absa | aspect polarity | full | state_only | 0.7628+-0.0230 | 0.7221+-0.0258 | +4.07 | false | TIE |
| absa | aspect polarity | constant | off | 0.7628+-0.0230 | 0.7174+-0.0348 | +4.53 | false | TIE |
| absa | aspect polarity | constant | state_only | 0.7628+-0.0230 | 0.7221+-0.0258 | +4.07 | false | TIE |
| absa | aspect polarity | off | state_only | 0.7174+-0.0348 | 0.7221+-0.0258 | -0.47 | false | LOSS |
| ravdess | emotion | full | off | 0.2468+-0.0609 | 0.2792+-0.0226 | -3.24 | false | LOSS |
| ravdess | emotion | full | state_only | 0.2468+-0.0609 | 0.2880+-0.0253 | -4.12 | false | LOSS |
| ravdess | emotion | off | state_only | 0.2792+-0.0226 | 0.2880+-0.0253 | -0.88 | false | LOSS |
| ravdess | gender | full | off | 0.7296+-0.0687 | 0.8988+-0.0535 | **-16.92** | **TRUE** | **LOSS** |
| ravdess | gender | full | state_only | 0.7296+-0.0687 | 0.8864+-0.0947 | -15.68 | false | LOSS |
| ravdess | gender | off | state_only | 0.8988+-0.0535 | 0.8864+-0.0947 | +1.24 | false | TIE |
| ravdess | intensity | full | off | 0.6788+-0.0421 | 0.5816+-0.1064 | **+9.72** | false | TIE |
| ravdess | intensity | full | state_only | 0.6788+-0.0421 | 0.6492+-0.1032 | +2.96 | false | TIE |
| ravdess | intensity | off | state_only | 0.5816+-0.1064 | 0.6492+-0.1032 | -6.76 | false | LOSS |
| ravdess | statement | full | off | 0.5432+-0.0947 | 0.6380+-0.0918 | -9.48 | false | LOSS |
| ravdess | statement | full | state_only | 0.5432+-0.0947 | 0.6716+-0.1417 | -12.84 | false | LOSS |
| ravdess | statement | off | state_only | 0.6380+-0.0918 | 0.6716+-0.1417 | -3.36 | false | LOSS |
| ag_news | topic | full | off | 0.8486+-0.0139 | 0.8568+-0.0100 | -0.82 | false | LOSS |
| ag_news | topic | full | state_only | 0.8486+-0.0139 | 0.8574+-0.0109 | -0.88 | false | LOSS |
| ag_news | topic | off | state_only | 0.8568+-0.0100 | 0.8574+-0.0109 | -0.06 | false | LOSS |
| ag_news | is_sports | full | off | 0.9664+-0.0100 | 0.9664+-0.0181 | -0.00 | false | LOSS |
| ag_news | is_sports | full | state_only | 0.9664+-0.0100 | 0.9700+-0.0102 | -0.36 | false | LOSS |
| ag_news | is_sports | off | state_only | 0.9664+-0.0181 | 0.9700+-0.0102 | -0.36 | false | LOSS |
| ag_news | is_scitech | full | off | 0.9068+-0.0110 | 0.9000+-0.0177 | +0.68 | false | TIE |
| ag_news | is_scitech | full | state_only | 0.9068+-0.0110 | 0.9004+-0.0202 | +0.64 | false | TIE |
| ag_news | is_scitech | off | state_only | 0.9000+-0.0177 | 0.9004+-0.0202 | -0.04 | false | LOSS |
| ag_news | is_business | full | off | 0.8856+-0.0066 | 0.8904+-0.0019 | -0.48 | false | LOSS |
| ag_news | is_business | full | state_only | 0.8856+-0.0066 | 0.8886+-0.0027 | -0.30 | false | LOSS |
| ag_news | is_business | off | state_only | 0.8904+-0.0019 | 0.8886+-0.0027 | +0.18 | false | TIE |
| yelp | stars | full | off | 0.3688+-0.0077 | 0.3836+-0.0146 | -1.48 | false | LOSS |
| yelp | stars | full | state_only | 0.3688+-0.0077 | 0.3816+-0.0087 | -1.28 | false | LOSS |
| yelp | stars | off | state_only | 0.3836+-0.0146 | 0.3816+-0.0087 | +0.20 | false | TIE |
| yelp | detractor | full | off | 0.8128+-0.0078 | 0.8062+-0.0174 | +0.66 | false | TIE |
| yelp | detractor | full | state_only | 0.8128+-0.0078 | 0.8078+-0.0113 | +0.50 | false | TIE |
| yelp | detractor | off | state_only | 0.8062+-0.0174 | 0.8078+-0.0113 | -0.16 | false | LOSS |

**The largest positive delta in the whole grid is +9.72 pp** (`ravdess_intensity`,
`full` over `off`). Its 5-seed ranges are `[0.616, 0.722]` and `[0.470, 0.708]`
- they overlap by 9.2 pp. **That is a win inside one standard deviation, and a
win inside one standard deviation is not a win.** The `off` arm's own std on that
question is 10.6 pp, larger than the effect.

### 5.4 Gradient norms per module - measured, 1 seed, 2 epochs

L2 norm of the accumulated gradient, read after `loss.backward()` and BEFORE
`clip_grad_norm_(1.0)`, at optimizer step 1 and at the final step. **0** marks a
module that received no gradient at all on that step.

| dataset (question) | arm | trainable | conditioning half | step | question_encoder | question_proj | schema_encoder | schema_proj | fusion | heads | state_proj |
|---|---|---|---|---|---|---|---|---|---|---|---|
| absa (aspect polarity) | full | 1 280 385 | 1 148 160 (89.7 %) | 1 | 7.37e-1 | **3.12e+0** | 3.90e-1 | 4.80e-1 | 6.43e-1 | 9.41e-1 | 4.69e-1 |
| | | | | 26 | 7.67e-1 | 2.06e+0 | 3.04e-1 | 3.16e-1 | 1.38e+0 | 8.07e-1 | **3.04e+0** |
| | off | 1 280 385 | 1 148 160 (89.7 %) | 1 | **0** | **0** | **0** | **0** | **0** | 9.31e-1 | **0** |
| | | | | 26 | 6.95e-2 | 2.49e-1 | **0** | **0** | 3.15e-1 | 1.45e+0 | 1.03e+0 |
| | state_only | 1 280 385 | 1 148 160 (89.7 %) | 1 | **0** | **0** | **0** | **0** | **0** | 8.75e-1 | **0** |
| | | | | 26 | **0** | 3.83e-1 | **0** | **0** | 2.15e-1 | 1.17e+0 | 6.83e-1 |
| ravdess (emotion) | full | 3 084 801 | 2 492 928 (80.8 %) | 1 | 6.34e-1 | **1.90e+0** | 4.66e-1 | 4.66e-1 | 6.55e-1 | 8.77e-1 | 3.59e-1 |
| | | | | 22 | 8.00e-1 | 3.03e+0 | 1.84e+0 | 1.88e+0 | **3.85e+0** | 3.19e+0 | 3.30e+0 |
| | off | 3 084 801 | 2 492 928 (80.8 %) | 1 | **0** | **0** | **0** | **0** | **0** | 8.46e-1 | **0** |
| | | | | 22 | 2.05e-1 | 7.11e-1 | **0** | **0** | 8.47e-1 | 3.17e+0 | 3.33e+0 |
| | state_only | 3 084 801 | 2 492 928 (80.8 %) | 1 | **0** | **0** | **0** | **0** | **0** | 6.68e-1 | **0** |
| | | | | 22 | **0** | 1.75e+0 | **0** | **0** | 8.96e-1 | 3.29e+0 | 4.40e+0 |
| ag_news (topic) | full | 1 280 385 | 1 148 160 (89.7 %) | 1 | 1.03e-1 | 4.69e-1 | 2.06e-1 | 4.60e-1 | 2.64e-1 | 3.90e-1 | 1.78e-1 |
| | | | | 48 | 2.06e-1 | 8.51e-1 | 1.35e+0 | 1.86e+0 | 1.16e+0 | 8.17e-1 | **2.99e+0** |
| | off | 1 280 385 | 1 148 160 (89.7 %) | 1 | **0** | **0** | **0** | **0** | **0** | 3.90e-1 | **0** |
| | | | | 48 | 9.34e-2 | 4.41e-1 | **0** | **0** | 3.69e-1 | 1.54e+0 | 1.60e+0 |
| | state_only | 1 280 385 | 1 148 160 (89.7 %) | 1 | **0** | **0** | **0** | **0** | **0** | 3.89e-1 | **0** |
| | | | | 48 | **0** | 1.25e+0 | **0** | **0** | 3.21e-1 | 1.47e+0 | 1.59e+0 |
| yelp (stars) | full | 1 280 385 | 1 148 160 (89.7 %) | 1 | 2.20e-1 | **8.43e-1** | 1.56e-1 | 2.31e-1 | 2.35e-1 | 3.79e-1 | 1.83e-1 |
| | | | | 48 | 2.20e-1 | 9.19e-1 | 8.05e-1 | 1.05e+0 | 9.90e-1 | 9.30e-1 | **3.40e+0** |
| | off | 1 280 385 | 1 148 160 (89.7 %) | 1 | **0** | **0** | **0** | **0** | **0** | 3.66e-1 | **0** |
| | | | | 48 | 3.25e-2 | 1.42e-1 | **0** | **0** | 2.13e-1 | 7.36e-1 | 1.24e+0 |
| | state_only | 1 280 385 | 1 148 160 (89.7 %) | 1 | **0** | **0** | **0** | **0** | **0** | 3.38e-1 | **0** |
| | | | | 48 | **0** | 3.76e-1 | **0** | **0** | 1.77e-1 | 7.47e-1 | 1.08e+0 |

Three things this table establishes that a parameter count cannot:

1. **The conditioning half is 80.8-89.7 % of the trainable parameters** on every
   dataset, and in the `off` and `state_only` arms it receives **exactly zero**
   gradient at step 1 - and in `state_only`, at the final step too. Those
   parameters are trainable, are counted in every budget and memory report in
   this repository, and contribute nothing.
2. **The largest gradient at step 1 is `question_proj`, not `state_proj`.** On
   ABSA it is 3.12e+0 against the head's 9.41e-1 and the state path's 4.69e-1.
   The question pathway is where the loss wants to move first, and it is exactly
   the pathway that produces no measurable accuracy.
3. **The ordering inverts during training.** By the final step `state_proj`
   dominates everywhere (3.04e+0, 3.30e+0, 2.99e+0, 3.40e+0), one to two orders
   of magnitude above the conditioning modules. The conditioning half starts
   with the largest gradient and ends with the smallest.

---

## 6. Verdict

### 6.1 The decisive question, answered

> does at least one question on at least one dataset show a conditioning mode
> beating `off` with NON-OVERLAPPING 5-seed accuracy ranges?

**NO.** 11 questions, 4 datasets, 5 seeds, 35 paired comparisons, **zero wins**.
The only non-overlapping comparison in the entire grid is a **loss of 16.92 pp**
(`ravdess_gender`: `full` 0.7296 +- 0.0687 vs `off` 0.8988 +- 0.0535, ranges
[0.652, 0.808] and [0.832, 0.958]).

This is outcome **(b)** in the program's terms: no win with non-overlapping
ranges, so the recommendation ships with the numbers attached.

### 6.2 The shipped recommendation

**`conditioning_mode="off"` becomes the single-question default.**

The numbers behind it:

| evidence | number |
|---|---|
| accuracy, pooled over 11 questions | `off` 0.7199 vs `full` 0.7046 (**+1.53 pp for `off`**) |
| accuracy on the 4 REAL RAVDESS questions | `off` 0.5994 vs `full` 0.5496 (**+4.98 pp for `off`**) |
| questions where `full` loses to `off` | 6 of 11 outright; the 5 where `full` leads average only +3.1 pp with overlapping ranges |
| worst single-question regression from conditioning | `ravdess_gender` **-16.92 pp** (disjoint ranges) |
| trainable parameters spent on the conditioning half | **80.8-89.7 %** of the stack |
| gradient those parameters receive in the `off` arm | **exactly 0** at step 1 |
| latency cost of the conditioning path | `full` is **1.09-1.12x** the `off` forward and `constant`-hit is 0.66-0.75x (section 2.4), i.e. the whole conditioning stack is pure cost when it does not pay |

And where even `off` is not the best answer: `state_only` - which additionally
bypasses the question, so the fused context stops depending on `q` at all -
scores **0.7294** pooled, above `off`'s 0.7199, and is the best or joint-best
arm on 2 of 4 datasets. The reason is visible in the code and in the gradient
table: `conditioning_mode="off"` zeroes the option residual but still feeds the
question vector into `ConcatFusion`'s `W_cat [s ; q]`, so half the
conditioning stack is still live. `state_only` is not a shipped mode, so the
recommendation is `off` as the shipped default, with the measured note that
`IdentityFusion` (or the `state_only` arm's zeroed question rows) is worth a
further ~1 pp if the question text is known not to matter.

**Do not read this as "the encoders are useless."** What the data says is
narrower and more specific: on frozen features at this budget, the
question-conditioned context is not the mechanism that solves any of these 11
tasks, and on two of them it is actively harmful. It leaves a place for a
different regime - many questions per state where the question is the ONLY thing
distinguishing the required answer, with enough budget to learn the interaction.
This grid is not that regime, and saying otherwise would be marketing.

### 6.3 The `constant` cache

`constant` is a **pure inference-time memo**: bitwise-equal to `full` on cold,
warm and content-changed calls, identical metrics over 5 seeds, and gated off
entirely during training. It satisfies the program's `<= 1.15x` criterion with
margin (0.66-0.75x) and it carries a **newly measured hazard**: the memo is
keyed on content, not on weights, so an eval/train/eval loop that does not call
`clear_condition_cache()` serves stale H_Q/H_C (max abs diff **4.1556e+04** in
the probe). If `constant` is adopted, that call belongs in the training step's
epilogue, not only in `load_state_dict`.

---

## 7. What this does NOT establish

1. **It does not establish that conditioning never helps.** 11 questions, 4
   datasets, 2-4 epochs, frozen 128-dim (text) and 768-dim (audio) tiny
   features. `ravdess_intensity` shows +9.72 pp for `full` - a real effect size,
   but with 5-seed ranges overlapping by 9.2 pp and a 10.6 pp std on the baseline
   arm. A larger budget, a bigger backbone, or a question set where the question
   is the only distinguishing signal could separate it. This grid cannot.
2. **It does not establish anything about RAVDESS at 4 epochs or at the 768
   feature width.** Both were reduced for wall clock (section 3.5). The single
   4-epoch spot check agrees in direction; the 768-width grid was never run.
3. **It does not establish anything about `constant` on RAVDESS, ag_news or
   Yelp.** That arm was not run there. The claim that it equals `full` rests on
   the bitwise proof (two independent wrappers, three call shapes) plus two
   5-seed metric-identity measurements, not on a full grid.
4. **It does not establish anything about calibration.** The cal split is carved
   and size-checked but no calibrator is fitted. Every NLL/ECE/Brier here is
   uncalibrated, and ECE in particular is not a calibration claim.
5. **It does not establish that the encoders generalise across question TYPES.**
   ag_news's three non-topic questions are deterministic thresholds on the topic
   label; Yelp's second question is a threshold on the star label. Those are not
   independent information, so their agreement is weaker evidence than a real
   second annotation would be. The genuinely independent multi-question evidence
   here is ABSA (4-13 distinct aspect questions per state, all REAL) and RAVDESS
   (4 independent filename fields, all REAL).
6. **The latency numbers are contended.** 4 shared cores at 69-100 % load from
   other agents. Round-robin interleaving makes the contention common-mode and
   the medians are stable (arm std 2.0-2.6 ms on 7-12 ms medians), but the
   absolute milliseconds are not a clean-machine measurement and must not be
   quoted as one.
7. **The stale-memo hazard was NOT fixed.** `jevx/**` was out of scope for this
   wave. Section 2.2 reports it; it is still there.
8. **Nothing here speaks to multimodal (vision) conditioning.** The vision
   diagnostic's finding that the question/ConcatFusion half costs 2.36 of 2.76 pp
   is consistent with these numbers but is a separate measurement.

---

## 8. WAVE-2 FOLLOW-UP — confirmation of the shipped default (append-only)

Added 2026-09-24. Everything above is unchanged. This section records what
was changed after the grid, and the confirmation run that checked the change
landed. Script: `benchmarks/run_conditioning_default_confirmation.py`, which
**imports** `run_multi_question` rather than re-implementing the protocol.
Result: `benchmarks/results/conditioning_default_confirmation.json`.

### 8.1 What shipped

| item | change |
|---|---|
| P0-1 | the `"constant"` H_Q/H_C memo is now **weight-keyed**: on every memo-eligible call the two memoized encoders' `torch.Tensor._version` counters are compared and any change drops the memo. §2.2's hazard is closed — see §8.3. |
| P0-2 | `conditioning_mode="off"` now also feeds fusion an **exact-zero question vector**, so `"off"` **is** the `state_only` reference arm rather than something that merely zeroed the option residual (§5.2/§5.3 of the grid above). Shapes are unchanged. |
| DECISION-3 | a fresh `JevXConfig()` is now `conditioning_mode="off"`. Revert in one line: `conditioning_mode: str = "off"` → `conditioning_mode: str = "full"` in `jevx/configuration_jevx.py`. |

### 8.2 Confirmation run — RAVDESS (audio) and ag_news (text), 3 seeds

Same frozen-feature protocol as §3.4/§5: identical splits (seed 0), identical
`torch.manual_seed(1234)` head init, `lr=1e-3` AdamW, batch 64, 2 epochs (the
§3.5 deviation), ECE/Brier on 15 bins, metrics from `jevx.utils.numerical` via
`bench_common`. The `default` arm is built with **no** `conditioning_mode` kwarg
at all. `torch 2.14.0+cpu`, `torch.set_num_threads(1)`, 4 shared cores.

| dataset (question) | arm | per-seed accuracy (0,1,2) | mean | std | NLL | ECE(15) |
|---|---|---|---|---|---|---|
| RAVDESS (`ravdess_emotion`) | **default** (shipped) | 0.2900, 0.2920, 0.2980 | **0.2933** | 0.0042 | 1.7802 | 0.0430 |
| RAVDESS | `full` | 0.1600, 0.2280, 0.3280 | 0.2387 | 0.0845 | 1.8789 | 0.0835 |
| RAVDESS | `state_only_ref` (WAVE-2 arm, seed 0) | 0.2900 | 0.2900 | — | 1.7771 | 0.0693 |
| ag_news (`ag_news_topic`) | **default** (shipped) | 0.8580, 0.8710, 0.8420 | **0.8570** | 0.0145 | 0.4188 | 0.0291 |
| ag_news | `full` | 0.8430, 0.8280, 0.8520 | 0.8410 | 0.0121 | 0.4486 | 0.0327 |
| ag_news | `state_only_ref` (WAVE-2 arm, seed 0) | 0.8580 | 0.8580 | — | 0.4030 | 0.0159 |

**The default reproduces the WAVE-2 `state_only` reference seed for seed** on
all six (dataset, arm, seed) cells where they can be compared:

| cell | this run | WAVE-2 §5.1 (seeds 0-2) |
|---|---|---|
| RAVDESS `state_only` per-seed | 0.290 / 0.292 / 0.298 | 0.290 / 0.292 / 0.298 |
| RAVDESS `full` per-seed | 0.160 / 0.228 / 0.328 | 0.160 / 0.228 / 0.328 |
| ag_news `state_only` per-seed | 0.858 / 0.871 / 0.842 | 0.858 / .871 / 0.842 |
| ag_news `full` per-seed | 0.843 / 0.828 / 0.852 | 0.843 / 0.828 / 0.852 |

The 3-seed default mean is **+0.53 pp** above the published 5-seed
`state_only` reference on RAVDESS and **−0.04 pp** on ag_news — i.e. it is the
same arm, not an approximation of it. That is the expected result and is the
point: P0-2 made the shipped `"off"` bitwise equal to the reference rather than
making it better.

**Default vs `full`.** RAVDESS +5.47 pp and ag_news +1.60 pp for the default.
Neither pair has disjoint 3-seed ranges (`ravdess` default range [0.290, 0.298]
against `full` [0.160, 0.328]; `ag_news` [0.842, 0.871] against [0.828, 0.852]),
so — exactly as §5.3 concluded for the 5-seed grid — this is **not** a
statistically separated win and is not reported as one.

**Invariants on every eval batch.** 20 eval batches over the two datasets and
the two conditioned arms, `config.runtime_validation=True` throughout (the
`ValidationEngine` runs per decision inside `forward`, so a violation raises
before these are read). Measured worst deviations: per-question simplex
error **2.38e-07** (float32 round-off) on both datasets, padded-column
absolute value **exactly 0.0**, all logits finite, every `predicted_index`
inside the valid set, confidence in [0, 1]. **clean = true on every cell.**

### 8.3 P0-1 — the stale-memo hazard is closed

The §2.2 probe (`eval` → one AdamW step → `eval` with **no**
`clear_condition_cache()`), re-run on the real backbones after the fix:

| backbone | max abs logit diff vs a fresh-cache eval | bitwise equal |
|---|---|---|
| `google/bert_uncased_L-2_H-128_A-2` | **0.0** | true |
| `superb/wav2vec2-base-superb-ks` | **0.0** | true |

§2.2 measured **4.1556e+04** on the tiny wrapper. It is now exactly 0 on both
real backbones, and the §7.7 caveat ("the stale-memo hazard was NOT fixed") no
longer applies.

**Cost of the check.** One `int` read per parameter/buffer TENSOR of
`question_encoder` and `schema_encoder` (10 tensors / 266 368 scalars on the
`tiny_bert` fixture), 20 000 iterations, contended 4-core box:
**83.5 µs** for the hand-rolled module walk + version reads, against
161.3 µs for the `parameters()`/`buffers()` form (110.3 µs of which is the
generator/de-duplication memo) and ~3.3 µs for a cached tensor list that was
rejected because it can go stale on parameter REPLACEMENT. The in-suite
measurement printed 57.7 / 72.2 / 89.0 µs on three runs of the same code.
That is ~0.5-0.9 % of the 7-12 ms batch-1 forward of §2.3, and ~2-6 % of the
1.4-3.6 ms a memo hit saves.

### 8.4 P0-2 — `"off"` IS the `state_only` reference, bitwise

| check | `google/bert_uncased_L-2_H-128_A-2` | `superb/wav2vec2-base-superb-ks` |
|---|---|---|
| `JevXConfig().conditioning_mode` | `"off"` | `"off"` |
| `JevXConfig().conditioning_enabled` | `False` | `False` |
| default == explicit `off` (bitwise) | true | true |
| default == WAVE-2 `state_only` reference (bitwise) | **true** | **true** |
| default `state_dict` == explicit `off` `state_dict` | true | true |
| trained default arm == trained `state_only_ref` arm (SHA-256 of the whole float32 test-logit matrix, seed 0) | **identical** | **identical** |

The last row is the strong one: it is not a wiring argument but the two arms'
full trained test-logit matrices hashing to the same value.

### 8.5 What this confirmation run does NOT establish

1. **It is 2 questions, not 11.** One primary REAL question per dataset, so it
   checks that the change landed and did not regress; it cannot re-derive the
   pooled 11-question table of §5.2, which stands on its own 5-seed run.
2. **3 seeds, not 5.** Every per-seed value above matches §5.1 seed for seed,
   which is why the comparison is exact rather than approximate, but the mean
   over 3 seeds is not the published 5-seed mean.
3. **No calibration claim.** The cal split is carved and size-checked; no
   calibrator is fitted, exactly as in the grid. Every NLL/ECE/Brier here is
   uncalibrated.
4. **RAVDESS at head width 256, 2 epochs** — the §3.5 deviation, unchanged.
   The width is identical across the arms compared here.
5. **The box was shared with other agent processes throughout**, as in §1.
