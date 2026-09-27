# Real Scenario Validation — three decision types, one state batch, one backbone forward

Every number below was produced by an executed run of `benchmarks/run_real_scenario.py`
and is stored field-for-field in `benchmarks/results/real_scenario.json`. The console block
in section 9 is copied verbatim from that run.

| | |
|---|---|
| Run | `python benchmarks/run_real_scenario.py --seeds 0,1,2` |
| Total wall time | **306.5 s** (phases 1-8, one process) |
| Phases recorded | 8 of 8; recorded failures: `{}` |
| Result document | `benchmarks/results/real_scenario.json` |
| Run log | `%TEMP%\jevx_real_scenario.log` (outside the repo; override with `JEVX_REAL_SCENARIO_LOG`) |

## 1. Environment

| key | value |
|---|---|
| python | 3.14.3 |
| platform | Windows-11-10.0.26100-SP0 |
| processor | Intel64 Family 6 Model 78 Stepping 3, GenuineIntel |
| cpu_count | 4 |
| torch | 2.14.0+cpu |
| transformers | 5.16.1 |
| datasets | 4.4.1 |
| numpy | 2.3.5 |
| torch_threads | 2 |
| cuda_available | false |
| dtype | float32 |

## 2. Dataset, backbone, splits, seeds

### 2.1 Dataset

| field | value |
|---|---|
| id | `Yelp/yelp_review_full` |
| revision (Hub API `sha`) | `c1f9ee939b7d05667af864ee1cb066393154bf85` |
| license (HF API tag, recorded verbatim) | other (HF API tag license:other) |
| official train / test rows | 650000 / 50000 |
| ClassLabel names | `["1 star","2 star","3 stars","4 stars","5 stars"]` |
| label semantics | class index y in 0..4 == (y+1) stars; Q1 choice label = y, Q2 score label = y (levels 0..4), Q3 noul label = 1[y == 0] |

The id was verified with a Hub API `GET /api/datasets/Yelp/yelp_review_full` before any
download: HTTP 200, `sha` as above, tag `format:parquet` (so `datasets` 4.4.1 loads it with no
loading script). The bare id `yelp_review_full` 307-redirects to `Yelp/yelp_review_full`.

### 2.2 Backbone

| field | value |
|---|---|
| id | `google/bert_uncased_L-2_H-128_A-2` |
| sha expected / observed | `30b0a37ccaaa32f332884b96992754e246e48c5f` / `30b0a37ccaaa32f332884b96992754e246e48c5f` |
| frozen | true — A (featurize-once, heads only) |
| pooling / fusion | `cls` / `concat` |
| max_length | 128 |

### 2.3 Splits (deterministic; split seed 0)

| split | n | source | label counts |
|---|---|---|---|
| train | 4000 | disjoint stratified carve from the official train split (bc.make_splits) | {"0":800,"1":800,"2":800,"3":800,"4":800} |
| cal | 1200 | disjoint from train (asserted in `bc.make_splits`) | {"0":240,"1":240,"2":240,"3":240,"4":240} |
| test | 1500 | official test split, first 1500 rows (cap applied=True) | {"0":278,"1":302,"2":333,"3":345,"4":242} |

1-star (noul positive) rate: train 0.2000, test 0.1853. Training seeds `[0,1,2]` change head initialisation and
batch order only; the split seed is fixed at 0. The test split was never
used for fitting, calibration, model selection, or threshold choice.

### 2.4 Features (featurize ONCE per split)

| split | feature shape | dtype | rows over max_length | longest untruncated row |
|---|---|---|---|---|
| train | `[4000,128]` | float32 | 2012 / 4000 | 1245 tokens |
| cal | `[1200,128]` | float32 | 631 / 1200 | 1329 tokens |
| test | `[1500,128]` | float32 | 741 / 1500 | 1263 tokens |

Featurization wall time: train 22.33 s (all three splits). One backbone
forward per batch per split, so the backbone is run once over the data and never during head training.

## 3. The three questions and the one call

One state batch of 1500 reviews; m = 3 questions;
one `JevXForDecision.decide(...)` call; one backbone forward.

| slot | decision_type | question | options / levels | ground truth |
|---|---|---|---|---|
| Q1 | `choice` (K=5) | What star rating will this reviewer give? | `["1 star","2 star","3 stars","4 stars","5 stars"]` | dataset class index `y` |
| Q2 | `score` (0..4) | On a 0-to-4 scale, how many stars does this review earn? | `["0","1","2","3","4"]` | dataset class index `y` |
| Q3 | `noul` (binary) | Is this a 1-star (detractor) review? | none | `1[y == 0]` |

Q1 options are the dataset's own `ClassLabel` names in `ClassLabel` order, so the class index
**is** the label and no remapping exists anywhere in the pipeline. Q2 re-expresses the same five
stars as 0-based levels 0..4, so its label is the same `y` — it is an ordinal question with a real
label, not a derived one. Q3 asks a genuinely different question about the same state (is this
the worst class?), with a real binary label.

```python
model.decide(
    state=None,
    questions=["What star rating will this reviewer give?",
                "On a 0-to-4 scale, how many stars does this review earn?",
                "Is this a 1-star (detractor) review?"],
    decision_type=['choice', 'score', 'noul'],
    options=[['1 star', '2 star', '3 stars', '4 stars', '5 stars'], ['0','1','2','3','4'], None],
    confidence_threshold=None,
    input_ids=..., attention_mask=...,   # tokenized with truncation=True, max_length=128
)
```

**Why the state is pre-tokenized.** `JevXForDecision.forward` never truncates a state: it raises
when the token length exceeds the backbone's learned position table, and this backbone has
`max_position_embeddings=512` while Yelp reviews run to 1263 tokens. `forward` documents
`input_ids=`/`attention_mask=` as the pre-tokenized spelling, so the scenario tokenizes with the
SAME call `bc.featurize()` makes (`truncation=True, max_length=128`). The live call
and the featurize-once training features therefore see identical tokens, and the raw review text is
still the state being decided about. This was found by running, not by reading: the first
execution of phase 4 raised `state token length 784 exceeds the backbone's learned position limit 512`.

## 4. Phase-by-phase results

| phase | what it did | wall (s) | recorded result |
|---|---|---|---|
| 1 | load pinned dataset, carve disjoint stratified train/cal, cap test, featurize once | 49.99 | features `[[4000,128],[1200,128],[1500,128]]`, truncation table above |
| 2 | train all three heads jointly on the same features (seeds 0,1,2) | 84.27 | 1,247,361 trainable params, loss curves below |
| 3 | fit temperature + isotonic per question type on the CAL split only | 5.54 | T per type below |
| 4 | ONE `decide()` over the 1500-state test batch, invariants asserted live | 35.74 | 16 checks, all passed |
| 5 | correctness battery | 116.7 | 5 / 6 checks passed |
| 6 | quality on TEST, uncalibrated vs temperature vs isotonic, 3 seeds + B1 | 11.15 | tables in section 6 |
| 7 | risk-coverage on Q1 with temperature-calibrated confidences | 1.39 | section 7 |
| 8 | print 10 real decisions | 0.09 | section 9 |

### 4.1 Training (phase 2) - three heads, one feature set, one optimizer

`jevx.training.HeadTrainer` (AdamW, `grad_clip=1.0`), loss dispatch by `jevx.training.make_supervised_step_fn` (choice -> CE, score -> CE, noul -> BCE), 4 epochs, lr 0.001, batch 64. one loader interleaves all three decision types inside every epoch, so state_proj / E_Q / E_C / fusion are updated by all three losses; each type's head sees only its own loss.

| seed | wall (s) | steps/epoch | epoch 1 | epoch 2 | epoch 3 | epoch 4 |
|---|---|---|---|---|---|---|
| 0 | 22.64 | 186 | 1.1942 | 1.0489 | 1.0297 | 1.0200 |
| 1 | 23.57 | 186 | 1.1806 | 1.0515 | 1.0344 | 1.0095 |
| 2 | 37.3 | 186 | 1.1585 | 1.0524 | 1.0284 | 1.0131 |

Per-question loss at epoch 4 (seed 0): choice 1.335871, score 1.355657, noul 0.368551. Trainable parameters: 1,247,361 (the frozen backbone contributes none).

### 4.2 Calibration (phase 3) - fit on the CALIBRATION split only

| question type | cal n | cal logit shape | fitted temperature T |
|---|---|---|---|
| choice | 1200 | `[1200,5]` | 1.0315 |
| score | 1200 | `[1200,5]` | 1.1776 |
| noul | 1200 | `[1200,1]` | 1.0712 |

Isotonic is fitted on the same split and reported in section 6. JevXForDecision holds ONE calibration engine per model, so a mixed decide() call applies a single calibrator to all three types. The scenario therefore fits one engine PER QUESTION TYPE and reports the per-type temperatures; the live decide() call in phase 4 runs the raw (uncalibrated) model.

## 5. Correctness

### 5.1 Invariants asserted live inside the one `decide()` call (phase 4)

| invariant | observed | verdict |
|---|---|---|
| backbone forward-hook count == 1 | **1** | pass |
| `len(decisions) == B*m` | 4500 == 4500 | pass |
| state-major ordering `b*m + j` | first and last 3 entries carry the 3 question types in order | pass |
| choice rows are a simplex (atol 1e-5) | max abs error 1.490e-7 | pass |
| score rows are a simplex (atol 1e-5) | max abs error 1.490e-7 | pass |
| choice argmax in options | all 5 occur: `["1 star","2 star","3 stars","4 stars","5 stars"]` | pass |
| score argmax in levels | all 5 occur: `["0","1","2","3","4"]` | pass |
| `expected_score` in [0, 4] | [0.1179, 3.8499] | pass |
| noul `p` in the open interval (0, 1) | [0.002301, 0.898475] | pass |
| noul rows carry exactly 1 probability | all 1500 rows width 1 | pass |
| `confidence` in [0, 1] | [0.2050, 0.9977] | pass |
| no cross-type contamination | widths 1500x5 / 1500x5 / 1500x1; every decision carries its own slot type | pass |
| `abstained is None` with no threshold | all 4500 | pass |
| `capability_metadata["backbone_forward_calls"] == 1` | 1 | pass |
| featurize-once: live logits vs cached-feature head stack | max abs Q1 logit delta 1.526e-5 over 64 states | recorded |

Wall time of the single call: **35.276 s** for 4500 decisions (1500 states x 3 questions) on 2 torch threads. Recorded check failures: `[]`.

### 5.2 Correctness battery (phase 5) - 5 of 6 passed

| # | check | verdict | measured |
|---|---|---|---|
| a | question independence: perturb Q2 options, Q1/Q3 must not move | **pass** | Q1 delta **0**, Q3 delta **0** (exactly 0); Q2 delta 0.752965 |
| b | batched == 3 solo calls, tolerance 1e-5 | **FAIL** | worst **1.0967e-5** (score); choice 2.980e-6, noul 7.749e-7 |
| c | `runtime_validation` on/off -> bitwise identical | **pass** | bitwise equal, max delta 0 |
| d | save -> load -> same call | **pass** | max abs probability delta **0.0** (bitwise equal) |
| e | same call twice -> bitwise identical | **pass** | bitwise equal, max delta 0.0 |
| f | `confidence_threshold=0.9` on Q1 | **pass** | **1496/1500** Q1 rows abstained; predicted fields cleared on all of them; probability delta vs the unthresholded call **0.0** |

Details worth reading:

* **(a)** Q2's levels were replaced with `["worst","bad","so-so","good","best"]` on the SAME state batch. Q1 and Q3
  probabilities are bit-for-bit unchanged (delta exactly 0.0 on all 1500 states), while Q2 moves by up to 0.7530. Question independence holds, and the perturbation provably reached the model.
* **(c)** Toggling `config.runtime_validation` changes nothing in the returned probabilities (bitwise equal over 64 states x 3 questions): the ValidationEngine is a gate, not a transform.
* **(d)** `save_pretrained()` to a temp directory, then `JevX.from_pretrained()` on that directory, reproduces the original call bitwise (max abs probability delta 0.0).
* **(f)** At threshold 0.9, 1496 of 1500 Q1 rows abstain, 1498 score rows and 729 noul rows abstain (noul is a binary question, so its confidence clears 0.9 far more often). On every abstained row `predicted_index`, `predicted_value` and `expected_score` are `None` while `probabilities` are byte-identical to the unthresholded call.

**On check (b), the one check that did not pass.** The scenario asked for a 1e-5 bound between the one batched call and three per-state calls. Measured worst deviation over 128 states walked one at a time: **1.0967e-5** on the score question, roughly 1e-6 above the requested bound. The same walk was repeated with every solo call padded to `max_length` so both calls hand the backbone the same tensor width; the worst deviation drops to **8.6129e-6**, inside the bound. Both numbers are in the JSON.

The residual is float re-association, not a semantic difference: a 1500-row call pads to the longest state in the batch, a 1-row call pads to that row's own length, so the two passes evaluate the same arithmetic over differently shaped tensors. Argmax, expected score and probability ORDER agreed on every state walked. This is reported as a failed check rather than relaxed after the fact.

## 6. Quality on the TEST split (1500 rows), 3 training seeds

Reported as mean ± std over seeds `[0,1,2]`, scored only on the test split.
Calibration was fit on the calibration split alone. ECE/MCE use 15 equal-width confidence bins.

### 6.1 Q1 — choice over K = 5 star levels

| variant | accuracy | macro-F1 | NLL | Brier | ECE(15) | MCE(15) |
|---|---|---|---|---|---|---|
| JevX, uncalibrated | 0.4000 ± 0.0018 | 0.3976 ± 0.0035 | 1.3672 ± 0.0054 | 0.7045 ± 0.0026 | 0.0304 ± 0.0080 | 0.2239 ± 0.0845 |
| JevX, temperature | 0.4000 ± 0.0018 | 0.3976 ± 0.0035 | 1.3668 ± 0.0052 | 0.7041 ± 0.0023 | 0.0246 ± 0.0093 | 0.2208 ± 0.1415 |
| JevX, isotonic | 0.3920 ± 0.0083 | 0.3932 ± 0.0065 | 1.4027 ± 0.0289 | 0.7062 ± 0.0023 | 0.0336 ± 0.0078 | 0.3666 ± 0.1049 |
| B1 linear probe, uncalibrated | 0.3591 ± 0.0019 | 0.3527 ± 0.0048 | 1.4347 ± 0.0089 | 0.7300 ± 0.0048 | 0.0191 ± 0.0037 | 0.2394 ± 0.0974 |
| B1 linear probe, temperature | 0.3591 ± 0.0019 | 0.3527 ± 0.0048 | 1.4332 ± 0.0099 | 0.7304 ± 0.0053 | 0.0295 ± 0.0014 | 0.4401 ± 0.3797 |
| **majority baseline** (class 1, chosen on TRAIN) | 0.2013 | - | - | - | - | - |
| uniform / chance | 0.2000 | 0.2000 | 1.6094 | 0.8000 | - | - |

Reading of section 6.1:

* JevX reaches 0.4000 accuracy against a train-majority baseline of 0.2013 — **1.99x the majority class**, and 2.00x chance. The seed-to-seed spread is 0.0018, so the gap is not a seed artifact.
* JevX beats the B1 linear probe on the same frozen features by 0.0409 accuracy and 0.0675 NLL (B1 NLL 1.4347 vs JevX 1.3672). The question- and option-aware head stack extracts more from the same representation than a bare linear layer.
* Temperature scaling moves ECE from 0.0304 to 0.0246 and leaves accuracy and macro-F1 **exactly** unchanged (a positive temperature preserves argmax). The fitted T = 1.0315 is close to 1: the uncalibrated model is already close to calibrated, so there is little for a single scalar to fix.
* Isotonic **hurts** here: accuracy 0.3920 and NLL 1.4027 are both worse than the uncalibrated model, and MCE rises to 0.3666. A per-class isotonic map fit on 1200 calibration rows is too flexible for a 5-class problem.
* B1's ECE is *lower* than JevX's uncalibrated ECE (0.0191 vs 0.0304) and its accuracy is lower too. A weaker model is easier to calibrate; ECE on its own is not a quality score.

### 6.2 Q2 — ordinal score, levels 0..4 = the same five stars

| variant | MAE | RMSE | Pearson r |
|---|---|---|---|
| JevX, uncalibrated | 0.8959 ± 0.0093 | 1.1219 ± 0.0155 | 0.5630 ± 0.0101 |
| JevX, temperature | 0.9020 ± 0.0108 | 1.1168 ± 0.0116 | 0.5638 ± 0.0095 |
| JevX, isotonic | 0.9005 ± 0.0106 | 1.1220 ± 0.0109 | 0.5597 ± 0.0086 |

Mean |E[score] − true level| by true level (seed 0, uncalibrated):

| true level | stars | n | mean E[score] | mean abs error | RMSE |
|---|---|---|---|---|---|
| 0 | 1 | 278 | 1.3184 | 1.3184 | 1.5249 |
| 1 | 2 | 302 | 1.7588 | 0.8869 | 1.0495 |
| 2 | 3 | 333 | 2.1745 | 0.6001 | 0.7203 |
| 3 | 4 | 345 | 2.5418 | 0.6284 | 0.8444 |
| 4 | 5 | 242 | 2.7343 | 1.2657 | 1.4874 |

Reading of section 6.2: MAE 0.8959 levels on a 0-4 scale, against 1.60 for a uniformly random guess and 2.00 for always predicting the midpoint; Pearson r = 0.5630. The per-level table shows the error is not uniform: it is smallest in the middle (3 stars: 0.6001, 4 stars: 0.6284) and largest at the two ends (1 star: 1.3184, 5 stars: 1.2657). The head is mean-seeking: at the extremes it predicts 1.3184 for a 0 and 2.7343 for a 4, shrinking toward the middle of the ordinal scale rather than committing. Temperature scaling does not fix this — it is a monotone resharpening of the same distribution, so MAE moves by 0.0061 and r moves by 0.0009.

### 6.3 Q3 — noul, "is this a 1-star (detractor) review?"

| variant | accuracy | AUC | NLL | ECE(15) | MCE(15) |
|---|---|---|---|---|---|
| JevX, uncalibrated | 0.8293 ± 0.0018 | 0.8051 ± 0.0031 | 0.3915 ± 0.0074 | 0.0338 ± 0.0129 | 0.1285 ± 0.0995 |
| JevX, temperature | 0.8293 ± 0.0018 | 0.8051 ± 0.0031 | 0.3857 ± 0.0017 | 0.0210 ± 0.0027 | 0.1008 ± 0.0543 |
| JevX, isotonic | 0.8293 ± 0.0035 | 0.8024 ± 0.0016 | 0.4019 ± 0.0152 | 0.0231 ± 0.0029 | 0.3661 ± 0.2141 |
| baseline: always "no" | 0.8147 | 0.5000 | - | - | - |
| baseline: always "yes" | 0.1853 | 0.5000 | - | - | - |

AUC is the Mann-Whitney U rank statistic with mid-ranks for ties, implemented directly in
`benchmarks/run_real_scenario.py` — no sklearn anywhere in the pipeline. The implementation is pinned by
closed-form cases recorded in the JSON, all passing: `["perfectly separated -> 1","fully reversed -> 0","all scores tied -> 0.5","disjoint blocks (36/36) -> 1","y=[0,0,1,1] on ascending scores (4/4 pairs) -> 1"]`.
An earlier revision of this script paired the sorted rank vector with the *unsorted* label vector and
returned an inflated AUC. It was caught by testing the helper against a brute-force pair count; the
self-check block exists because of that and is written into the artifact so the metric cannot regress
silently. (Section 14.)

Reading of section 6.3: accuracy 0.8293 sits only just above the always-"no" baseline of 0.8147, because 1-star reviews are only 18.5333% of the test split. **Accuracy is the wrong read here; AUC is 0.8051** — the noul head ranks a 1-star review above a non-1-star review 80.5106% of the time against a 50% coin flip. Temperature scaling leaves AUC and accuracy untouched (rescaling one logit preserves both) and improves NLL 0.3915 -> 0.3857 and ECE 0.0338 -> 0.0210. Isotonic degrades NLL and MCE.

## 7. Risk-coverage and selective prediction (Q1)

Confidence = max probability after temperature scaling fit on the calibration split (T = 1.0315).
`risk_coverage_curve` / `coverage_at_risk` are the library functions.

| quantity | value |
|---|---|
| overall accuracy on the test split | 0.3993 |
| AURC (mean risk over all coverage points) | 0.4913 |
| **`coverage_at_risk(0.1)`** | **0.0013** (2 of 1500 states) |
| **`coverage_at_risk(0.2)`** | **0.0147** (22 of 1500 states) |
| AURC, uncalibrated reference | 0.4912 |
| `coverage_at_risk(0.1)`, uncalibrated reference | 0.0013 |
| `coverage_at_risk(0.2)`, uncalibrated reference | 0.0147 |

| coverage | states answered | risk (1 − accuracy) at that coverage |
|---|---|---|
| 0.1 | 150 | 0.3444 |
| 0.2 | 300 | 0.4252 |
| 0.3 | 450 | 0.4789 |
| 0.5 | 750 | 0.5166 |
| 0.7 | 1050 | 0.5543 |
| 0.9 | 1350 | 0.5837 |
| 1 | 1500 | 0.6007 |

Reading of section 7: the risk-coverage curve is **flat, not steep**. At 10% coverage the risk is
already 0.344 against a full-coverage risk of 0.601, so there is
almost no coverage at which the model can promise a 10% or 20% error rate — `coverage_at_risk` returns
0.0013 and 0.0147. This is the correct, and unflattering, read: with a 2-layer 128-hidden backbone and 4000
training reviews, confidence separates right from wrong only weakly, and temperature scaling (T ≈ 1)
barely changes the ordering. AURC is essentially identical calibrated and uncalibrated
(0.4913 vs 0.4912) — calibration moves the probability values, not the ranking, and AURC depends only on the ranking.

**Selective prediction at thresholds taken from the calibration split** (no test labels are used to
choose the threshold; the test split is only *scored* at them):

| cal confidence quantile | threshold τ | test coverage | test accuracy on answered | cal accuracy on answered |
|---|---|---|---|---|
| 0.1 | 0.2725 | 88.5% | 0.4205 | 0.4194 |
| 0.25 | 0.3107 | 73.6% | 0.4438 | 0.4567 |
| 0.5 | 0.3656 | 47.3% | 0.4965 | 0.5150 |
| 0.75 | 0.4910 | 21.8% | 0.5719 | 0.6200 |

This is the useful form of the risk-coverage result. Answering only the top 21.8% most confident
reviews raises accuracy from 0.3993 to 0.5719 — a 43% relative gain — and the calibration split
predicts that gain (0.6200 on cal vs 0.5719 on test), so the threshold transfers. Confidence is useful for
**ranking** which reviews to answer first; it is not useful as an **absolute** promise.

## 8. How the printed decisions were chosen

All rows come from the same live `decide()` call as section 5.1. Selection is deterministic:
3 highest-confidence correct + 3 highest-confidence wrong + 2 rows at the median confidence + the 2 lowest-confidence rows, all taken from the live decide() output; at threshold 0.9 1496/1500 Q1 rows abstained, so the abstained flag is reported per row rather than used to select rows.

raw (uncalibrated) model probabilities; the per-type temperatures are reported in phase 3 / phase 6.

## 9. VERBATIM printed decisions (copied unchanged from the run)

```text

====================================================================================================
REAL DECISIONS — live JevX.decide() over the 1500-row Yelp test batch
  Q1 choice: What star rating will this reviewer give?  options=['1 star', '2 star', '3 stars', '4 stars', '5 stars']
  Q2 score : On a 0-to-4 scale, how many stars does this review earn?  levels=['0', '1', '2', '3', '4']
  Q3 noul  : Is this a 1-star (detractor) review?  (P(yes) = 1-star detractor)
====================================================================================================

[highest-confidence CORRECT] test row 811  true=1 star  abstained_at_0.9=False
  text           : I would love to rate their food, but never got the opportunity. Happened to stop there on a Saturday evening at about 5…
  Q1 choice      : pred=1 star  conf=0.9169  CORRECT  argmax_p=1 star
  Q1 top probs   : 1 star=0.9169, 2 star=0.0610   full=['1 star', '2 star', '3 stars', '4 stars', '5 stars']
  Q1 probs       : [0.9169, 0.061, 0.0087, 0.0042, 0.0093]
  Q2 score       : E[score]=0.1179 (argmax level 0)  conf=0.9030
  Q3 noul        : P(1-star yes)=0.8985  conf=0.8985
  TRUE           : 1 star (class index 0)  | 1-star detractor=True

[highest-confidence CORRECT] test row 391  true=1 star  abstained_at_0.9=False
  text           : My mother wanted to take my wife and I out to dinner for my wife's B-day, we were a party of eight, we were assigned tw…
  Q1 choice      : pred=1 star  conf=0.9096  CORRECT  argmax_p=1 star
  Q1 top probs   : 1 star=0.9096, 2 star=0.0572   full=['1 star', '2 star', '3 stars', '4 stars', '5 stars']
  Q1 probs       : [0.9096, 0.0572, 0.0096, 0.0058, 0.0177]
  Q2 score       : E[score]=0.1426 (argmax level 0)  conf=0.8916
  Q3 noul        : P(1-star yes)=0.8813  conf=0.8813
  TRUE           : 1 star (class index 0)  | 1-star detractor=True

[highest-confidence CORRECT] test row 338  true=1 star  abstained_at_0.9=False
  text           : One of the worst mgmt companies i have ever dealt with. The policies in their building are obviously very outdated but …
  Q1 choice      : pred=1 star  conf=0.9052  CORRECT  argmax_p=1 star
  Q1 top probs   : 1 star=0.9052, 2 star=0.0533   full=['1 star', '2 star', '3 stars', '4 stars', '5 stars']
  Q1 probs       : [0.9052, 0.0533, 0.0103, 0.0069, 0.0243]
  Q2 score       : E[score]=0.1470 (argmax level 0)  conf=0.8972
  Q3 noul        : P(1-star yes)=0.8782  conf=0.8782
  TRUE           : 1 star (class index 0)  | 1-star detractor=True

[highest-confidence WRONG] test row 894  true=3 stars  abstained_at_0.9=False
  text           : As much as I love the idea of saving a poor helpless animal from being put to death this place could use some work if t…
  Q1 choice      : pred=1 star  conf=0.9080  WRONG  argmax_p=1 star
  Q1 top probs   : 1 star=0.9080, 2 star=0.0487   full=['1 star', '2 star', '3 stars', '4 stars', '5 stars']
  Q1 probs       : [0.908, 0.0487, 0.0102, 0.0077, 0.0254]
  Q2 score       : E[score]=0.1830 (argmax level 0)  conf=0.8742
  Q3 noul        : P(1-star yes)=0.8745  conf=0.8745
  TRUE           : 3 stars (class index 2)  | 1-star detractor=False

[highest-confidence WRONG] test row 583  true=2 stars  abstained_at_0.9=True
  text           : Had a layover here recently coming in from an international destination. Customs wasn't too bad, basically waved us thr…
  Q1 choice      : pred=None  (ABSTAINED)  conf=0.8961  WRONG  argmax_p=1 star
  Q1 top probs   : 1 star=0.8961, 2 star=0.0589   full=['1 star', '2 star', '3 stars', '4 stars', '5 stars']
  Q1 probs       : [0.8961, 0.0589, 0.0122, 0.0083, 0.0245]
  Q2 score       : E[score]=0.1839 (argmax level 0)  conf=0.8715
  Q3 noul        : P(1-star yes)=0.8692  conf=0.8692
  TRUE           : 2 stars (class index 1)  | 1-star detractor=False

[highest-confidence WRONG] test row 1130  true=4 stars  abstained_at_0.9=True
  text           : love this place!!! the wait is long and the cashier is slow!!! there is one worker which i think is the owners son he h…
  Q1 choice      : pred=None  (ABSTAINED)  conf=0.8902  WRONG  argmax_p=5 stars
  Q1 top probs   : 5 stars=0.8902, 4 stars=0.0759   full=['1 star', '2 star', '3 stars', '4 stars', '5 stars']
  Q1 probs       : [0.0138, 0.0047, 0.0154, 0.0759, 0.8902]
  Q2 score       : E[score]=3.7897 (argmax level 4)  conf=0.8809
  Q3 noul        : P(1-star yes)=0.0705  conf=0.9295
  TRUE           : 4 stars (class index 3)  | 1-star detractor=False

[median-confidence] test row 1057  true=3 stars  abstained_at_0.9=True
  text           : A little pricey for the lack of amenities but the history of the hotel made it worth one night stay. \nTiny room.\nSimp…
  Q1 choice      : pred=None  (ABSTAINED)  conf=0.3642  WRONG  argmax_p=2 star
  Q1 top probs   : 2 star=0.3642, 3 stars=0.2368   full=['1 star', '2 star', '3 stars', '4 stars', '5 stars']
  Q1 probs       : [0.2136, 0.3642, 0.2368, 0.1267, 0.0588]
  Q2 score       : E[score]=1.5401 (argmax level 1)  conf=0.2960
  Q3 noul        : P(1-star yes)=0.1782  conf=0.8218
  TRUE           : 3 stars (class index 2)  | 1-star detractor=False

[median-confidence] test row 944  true=4 stars  abstained_at_0.9=True
  text           : I had a craving for Mexican food tonight... which happens rarely enough that I should I should indulge. I had no idea w…
  Q1 choice      : pred=None  (ABSTAINED)  conf=0.3642  WRONG  argmax_p=2 star
  Q1 top probs   : 2 star=0.3642, 3 stars=0.2442   full=['1 star', '2 star', '3 stars', '4 stars', '5 stars']
  Q1 probs       : [0.2015, 0.3642, 0.2442, 0.1308, 0.0592]
  Q2 score       : E[score]=1.6173 (argmax level 1)  conf=0.2884
  Q3 noul        : P(1-star yes)=0.1657  conf=0.8343
  TRUE           : 4 stars (class index 3)  | 1-star detractor=False

[lowest-confidence (would ABSTAIN at the threshold)] test row 1470  true=4 stars  abstained_at_0.9=True
  text           : Genny's is a hoot! I'm def Straight but NOT narrow.
  Q1 choice      : pred=None  (ABSTAINED)  conf=0.2088  WRONG  argmax_p=5 stars
  Q1 top probs   : 5 stars=0.2088, 2 star=0.2012   full=['1 star', '2 star', '3 stars', '4 stars', '5 stars']
  Q1 probs       : [0.1914, 0.2012, 0.1998, 0.1989, 0.2088]
  Q2 score       : E[score]=2.1889 (argmax level 4)  conf=0.2596
  Q3 noul        : P(1-star yes)=0.1366  conf=0.8634
  TRUE           : 4 stars (class index 3)  | 1-star detractor=False

[lowest-confidence (would ABSTAIN at the threshold)] test row 1213  true=2 stars  abstained_at_0.9=True
  text           : Expensive and just ok.
  Q1 choice      : pred=None  (ABSTAINED)  conf=0.2088  WRONG  argmax_p=1 star
  Q1 top probs   : 1 star=0.2088, 2 star=0.2081   full=['1 star', '2 star', '3 stars', '4 stars', '5 stars']
  Q1 probs       : [0.2088, 0.2081, 0.1972, 0.1922, 0.1937]
  Q2 score       : E[score]=2.0841 (argmax level 4)  conf=0.2286
  Q3 noul        : P(1-star yes)=0.1624  conf=0.8376
  TRUE           : 2 stars (class index 1)  | 1-star detractor=False

====================================================================================================
```

## 10. Reproduction

```bash
# from the repository root; CPython 3.14.3
python benchmarks/run_real_scenario.py --seeds 0,1,2
```

The result document is rewritten atomically after **every** phase, so an interrupted run leaves a
valid partial JSON rather than a truncated file. Useful flags:

| flag | effect |
|---|---|
| `--quick` | tiny smoke sizes (300/150/200, 2 epochs, 1 seed); runs all 8 phases in ~100 s |
| `--phases 4,5` | run a subset (phases 1-3 must have populated the context in the same process) |
| `--seeds 0` | single seed |
| `--out PATH` | result document path |
| `--fresh-log` | truncate the run log first |
| `JEVX_REAL_SCENARIO_LOG` | run-log path (default `%TEMP%\jevx_real_scenario.log`) |

Determinism: the split seed is fixed at 0 and every split comes from
`benchmarks.bench_common.make_splits` / `stratified_indices` (seeded `numpy.RandomState`, largest-remainder
class quotas). The joint batch sampler is seeded per training seed. Checks (c), (d) and (e) are
**bitwise** reproducible; check (b) is a numerical bound and moves by ~1e-6 with the host BLAS/threading.

Wall-time breakdown of the recorded run (seconds):

| step | s |
|---|---|
| dataset load (offline Arrow cache) | 23.16 |
| text selection for the three splits | 0.63 |
| backbone load | 0.32 |
| featurize train+cal+test | 22.33 |
| train 3 seeds x 3 questions | 84.02 |
| calibration fit | 5.53 |
| the one live `decide()` call | 35.74 |
| correctness battery | 116.7 |
| quality, 3 seeds + B1 | 11.15 |
| risk-coverage | 1.39 |
| **total** | **306.5** |

## 11. Optional hybrid leg (Qwen3.5-0.8B) — skipped, with a measured projection

The brief allows this leg only if >= 25 min of budget remained after phases 1-8. It did not, and the
skip is projected from a measurement rather than a guess: `benchmarks/results/qwen35_jevx_validation.json`
records, on this same host and the same checkpoint snapshot
(`Qwen/Qwen3.5-0.8B`, sha `2fc06364715b967f1860aea9cf38778875588b17`, 852,985,920 backbone parameters,
load 3.4 s), that **featurizing 200 rows took 1244.7 s** and that a 40-row `decide()` smoke took 402.2 s.

| step | rows | measured rate (s/row) | projected |
|---|---|---|---|
| featurize train | 200 | 6.22 | 1244.7 s (measured) |
| featurize test | 100 | 6.22 | ~622 s |
| decide + save/load + invariants | 100 | 10.05 | ~1005 s (measured on 40 rows) |
| **total** | | | **~48 min** |

That is roughly 9x the remaining budget, on 4 CPU cores with a 0.8B backbone at 96 tokens. Running it
would have meant either exceeding the budget or silently shrinking the protocol, so it was skipped
and is reported as skipped. The hybrid leg is already covered separately by
`benchmarks/results/qwen35_jevx_validation.json` and `reports/QWEN35_HYBRID_VALIDATION.md`.

## 12. LIMITATIONS (measured, not hypothetical)

1. **The backbone is 2 layers and 128 hidden units.** `google/bert_uncased_L-2_H-128_A-2` is frozen and
   0.4000 five-way accuracy is the ceiling that follows from that, not from JevX. Nothing here
   separates "the head stack is weak" from "the representation is weak"; the B1 linear probe on the
   same features reaching 0.3591 says the representation carries real but limited signal.
2. **Half the test reviews are truncated.** 741 of 1500 test rows exceed 128
   tokens (longest 1263). Every reported metric — and every printed decision — is computed on the first 128
   tokens. `max_length` was never tuned; it is the brief's budget value.
3. **One calibration engine per model.** `JevXForDecision` holds a single `CalibrationEngine`, so a
   mixed `decide(["choice","score","noul"])` call would apply one calibrator to all three types. This
   scenario therefore fits one engine per question type for the reported numbers and runs the live call
   uncalibrated. The printed probabilities are uncalibrated for that reason, and the per-type T values in
   section 4.2 do not apply to them.
4. **Risk-coverage is flat, so `coverage_at_risk` is near zero** (0.0013 at risk 0.1, 0.0147 at risk 0.2). Selective
   prediction still helps (section 7), but this model cannot promise a 10% error rate at any useful
   coverage. That is a statement about the configuration, not about the API.
5. **Accuracy on Q3 is a misleading headline.** At 0.8293 it looks close to the always-"no"
   baseline of 0.8147. The ranking quality (AUC 0.8051) is the number that
   describes the head.
6. **Check (b) did not pass its 1e-5 bound** (worst 1.097e-5). See section 5.2. The cause is padding-width
   dependent float re-association, and the matched-padding walk (8.613e-6) is inside the bound, but the
   check as specified is recorded as failed.
7. **Three seeds is a small sample for a std.** Seed spread on Q1 accuracy is 0.0018; the comparison against the
   majority baseline and B1 is far larger than that, but the spread itself is estimated from n = 3.
8. **No B2/B3/B4 baselines and no latency or memory numbers** were re-measured here. This document
   covers one scenario end to end; the wider baseline and latency tables live in
   `reports/DATASET_BENCHMARKS.md`, `reports/LATENCY_BENCHMARKS.md` and `reports/MEMORY_BENCHMARKS.md`.
   The B1 linear probe *was* re-run here, on the identical features, splits and seeds.
9. **The text column is not materialized.** The official train split has 650,000 reviews and the
   script selects only the 4000 + 1200 rows the splits use. This is a memory decision, not a data decision: the selected rows are
   exactly the indices `bc.make_splits` returned, in the same order.
10. **The dataset was loaded from the local cache for the recorded run** (`HF_HUB_OFFLINE=1`), because the
    ids and revisions had already been verified against the Hub API. The resolved cache directory is the
    one named by the pinned revision `c1f9ee939b7d05667af864ee1cb066393154bf85`. A cold online run of the same command
    produced the same rows and labels.

## 13. What this scenario does and does not establish

Established by execution, with the numbers above:

* One `decide()` call with three questions of three different decision types over a 1500-state batch performs
  **exactly one backbone forward** (hook count 1) and returns 4500 state-major decisions with per-type invariants intact.
* Questions are independent: perturbing Q2's options leaves Q1 and Q3 bitwise unchanged.
* `save_pretrained` / `from_pretrained` and repeated identical calls are bitwise reproducible;
  `runtime_validation` is a gate, not a transform.
* `confidence_threshold` abstains, clears the predicted fields and leaves probabilities untouched.
* All three question types score against real dataset labels on the same states, and the same three heads
  are trained jointly from one feature set with one optimizer.

Not established:

* That these accuracies are competitive with a fine-tuned transformer. They are not; the backbone is
  deliberately tiny and frozen.
* That confidence can be taken as a promise. Section 7 shows the opposite for this configuration.
* Anything about the generative (decoder) path — the hybrid leg was skipped, section 11.

## 14. Changes made

* **Added** `benchmarks/run_real_scenario.py` (the executable scenario; writes
  `benchmarks/results/real_scenario.json` incrementally).
* **Added** `benchmarks/results/real_scenario.json`.
* **Added** this report.
* **No change under `jevx/**`.** No library bug blocked the scenario, so no fix was made and no
  library behaviour was modified. `python -m pytest tests/unit -q` was re-run against this tree and
  reports **798 passed, 1 skipped** in 27.1 s. A full `python -m pytest -q` (which additionally walks the
  vision, audio and benchmark tests) was started but did not finish inside the run budget, so no claim is
  made about it here.

One bug was found and fixed, and it was in the benchmark script, not the library: the script's own
Mann-Whitney AUC paired the sorted rank vector with the unsorted label vector. It was caught by
testing the helper against a brute-force pair count, corrected, and a closed-form self-check is now
written into the result JSON (section 6.3) so the metric is verifiable without rerunning anything.

