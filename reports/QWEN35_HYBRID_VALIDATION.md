# Qwen3.5-0.8B HYBRID — JevX Real-Execution Validation Report

Date: 2026-09-24. Every PASS below is an executed pass on the real checkpoint
(every PASS executed, §98): `tests/integration/test_qwen35_hybrid.py`,
measured numbers persisted to `benchmarks/results/qwen35_jevx_validation.json`
(written incrementally after each battery item).

## 1. Environment + checkpoint

| Item | Value |
|---|---|
| Python / torch / transformers | 3.14.3 / 2.14.0+cpu / 5.16.1 (site-packages; vendor copy identical version) |
| pytest / psutil / datasets | 9.0.3 / 7.1.3 / 4.4.1 |
| Platform | Windows-11-10.0.26100, Intel64 Family 6 Model 78, 4 logical CPUs, torch threads=2 |
| Checkpoint path | `C:\Users\ysnrfd\.cache\huggingface\hub\models--Qwen--Qwen3.5-0.8B\snapshots\2fc06364715b967f1860aea9cf38778875588b17` |
| `refs/main` sha | `2fc06364715b967f1860aea9cf38778875588b17` |
| Model stats | `model_type=qwen3_5`, architectures `Qwen3_5ForConditionalGeneration`; backbone built by AutoModel chain = **`Qwen3_5Model`** (text+vision wrapper); 852,985,920 backbone params (wrapper total 873,974,849), bf16; hidden 1024, 24 layers, layer_types `[linear,linear,linear,full]×6`, `full_attention_interval=4`, 8Q/2KV heads head_dim 256, vocab 248320, mrope interleaved; single safetensors shard 1,746,942,600 B |
| Load call under test | `JevX.from_pretrained(<explicit snapshot path>, local_files_only=True)` — `local_files_only` IS in `_BACKBONE_LOAD_KWARGS` (whitelist check recorded in JSON); raw reference: `AutoModel.from_pretrained(<snapshot>, local_files_only=True)` |
| Wall time / RSS | JevX wrap 2.504 s (raw AutoModel 0.579 s); RSS after wrap 848.6 MB, process peak 1781.8 MB (single copy; save/load phase briefly holds two copies). Figures quoted in this report are from the JSON-of-record run (`python -m pytest -q`); the standalone targeted run measured 2.378 s / 0.636 s / 1708.9 MB — same behavior. |

## 2. Per-battery table

| # | Test | Measured | Result | Tolerance rationale |
|---|---|---|---|---|
| 1 | LOAD: backbone/adapter/capability/width | backbone `Qwen3_5Model`; adapter `MultimodalAdapter` (name `multimodal`, default pooling `masked_mean`); tokenizer `Qwen2Tokenizer` (padding side right, pad 248044) via `Qwen3VLProcessor`; capability: `jevx_conversion="experimental"`, `representation_field="last_hidden_state"`, `output_class=Qwen3_5ModelOutputWithPast`, decoder=True, pooler=False, processor_required=True; `decision_hidden_size=1024` == `text_config.hidden_size`; layer_types reach JevX intact (24 entries, interval 4); `backbone_config.architectures=["Qwen3_5Model"]`; load 2.504 s (raw 0.579 s) | **PASS** | functional asserts on exact class/width/field values; hybrid key round-trip verified |
| 2 | EXTRACTION vs independent manual pool | field `last_hidden_state`, effective strategy `masked_mean` (config `pooling_method=None` → adapter default; no pooler exists), pooled `[1,1024]` bf16; adapter vs direct-forward `use_cache=False` + own fp32 masked-mean: **5.059e-02** ≤ 2-ulp bf16 bound **2.500e-01** (ulp at pooled max = 0.125); sequence A-vs-C **bitwise 0.0**; cache-vs-no-cache sequence **0.0** | **PASS** | cross-path equality limited by bf16 STORE rounding of the pooled vector (2 ulp at observed magnitude); both decomposed sub-errors recorded |
| 3 | BASE PRESERVATION `base_forward` vs directly-loaded `AutoModel` | **max_abs_error 0.000e+00, bitwise equal** (same inputs, both bf16, eval, no_grad) | **PASS** | mission bf16 bar < 1e-2; actually bitwise |
| 4 | DECISIONS `.choice` K{2,5}×2q×2states + `.score`/`.noul`/mixed | row sums max dev **1.923e-3** (JSON-of-record run; independent runs: 1.953e-3 targeted, 7.44e-4 debug); first real probs `[0.00014, 1, 0, 0, 0]` (K=2 padded to K_max=5 with exact zeros); predicted_values in schema; confidence ∈ [0.809, 1.000]; score ES 2.125 ∈ [0,4]; noul p=0.609 ∈ (0,1); mixed types `[choice, score, noul]`; probs dtype bf16 | **PASS** | sums atol **4e-3**: probabilities are STORED bf16 (eps at 1.0 = 7.8125e-3); measured max deviation ≤ 1.953e-3 across three independent runs → atol = 2× worst measurement, < 1 ulp; mission's 1e-5→1e-4 suggestion too tight for bf16-stored rows |
| 5 | SHARED-STATE forward-hook counter, m=4 | backbone forward count **exactly 1** | **PASS** | exact integer contract (Def. 11 / T7) |
| 6a | HYBRID cache: `use_cache` path + drift | `use_cache` NOT declared on `Qwen3_5Model.forward` (kwargs-only); post-fix: extraction output carries **no cache** (regression-guarded assert); raw default-mode calls build `DynamicCache`; two identical sequential forwards **bitwise equal (0.0)**; cache-vs-no-cache sequences **0.0** | **PASS** | exact bitwise equality for drift; cache absence is an exact `is None` assert (bug fix §4) |
| 6b | RIGHT-pad batch(2 diff lengths) vs each alone | JevX: row0 **0.125**, row1 **0.03125**; backbone-alone: row0 **0.0238**, row0-long **0.0100**; bf16 noise floor (2 ulp at batch max) **0.25** | **PASS** | JevX bar = not worse than backbone above the bf16 kernel-rounding noise floor: `jevx ≤ max(backbone, 2·ulp) + 1e-6`. JevX rows carry the bf16 *store* rounding of the pooled vector on top of the backbone's kernel-level difference (backbone reference pooled in fp32); all values ≤ 0.125 ≤ 0.25 floor |
| 6c | LEFT-pad vs RIGHT-pad same content | JevX **0.125**, backbone-alone **0.0465** (also left-vs-alone: 0.0625 / 0.0446) | **PASS** (measured, no equality asserted) | DecoderAdapter's unconditional left/right equivalence is GPT-2/DecoderAdapter-scoped and does NOT apply: this wrapper resolves MultimodalAdapter (no cumsum `position_ids` derivation; the backbone uses arange positions). Differences stay at bf16 noise-floor scale because GDN has no positional encoding and RoPE attention depends only on *relative* positions (uniform pad shift preserves all intra-row offsets). Asserted: not-worse-than-backbone; NOT unconditional equality |
| 7 | SAVE/LOAD round-trip | probs **max_abs_error 0.000e+00**; `jevx_metadata.json` written; NO `jevx_calibration.pt` (calibration_method `none`); saved `backbone_config.architectures=["Qwen3_5Model"]` == reloaded backbone class == original class; reload adapter `MultimodalAdapter`; save 42.8 s / reload 5.24 s | **PASS** | mission bf16 bar < 1e-2; actually bitwise; sidecar rules exact |
| 8a | SST-2 smoke, N=40 (UNTRAINED head — pipeline verification only, NOT a quality claim) | source **`stanfordnlp/sst2`** (both final runs; an earlier debug run hit SSL handshake timeouts and used the `nyu-mll/glue/sst2` fallback — identical rows, ladder recorded); acc **0.450**, macro-F1 **0.310**, NLL **7.350**, ECE(10-bin) **0.550**; row-sum dev 1.10e-3; forward wall **191.3 s** (length-sorted chunk=8; unsorted debug run: 423.7 s) | **PASS** (structural; numbers recorded as pipeline evidence) | only structural asserts (finite, ranges, sums ≤ 4e-3); NO quality threshold — untrained head ≈ chance is expected |
| 8b | OPTIONAL HeadTrainer choice-head training (featurize once, frozen backbone, 200 SST-2 rows, 1 epoch) | see §"8b protocol" below | see §"8b protocol" | protocol evidence only; no quality claim |
| 9 | FAILURE TRANSPARENCY (pixel_values / garbage inputs) | 7/7 raise, none silent, none terse: `pixel_values`-only → `ValueError: You must specify exactly one of input_ids or inputs_embeds`; `input_ids`+`pixel_values` → `AttributeError: 'NoneType' object has no attribute 'device'` (upstream, cryptic); int state / int list / blank text / int questions / noul+options → actionable JevX TypeErrors/ValueErrors naming argument + expected types (full text in JSON) | **PASS** | every probe must raise with >20-char message; JevX-owned paths must start TypeError/ValueError |

### 8b protocol (optional item)

- Featurize-once: 200 train rows (`nyu-mll/glue` sst2 validation rows 40..239, disjoint from eval) + eval rows through the real `adapter.extract_representation` (frozen backbone, `torch.no_grad`), chunked, `max_length=128`.
- Train: `HeadTrainer` (AdamW default lr 1e-3, grad_clip 1.0, seed 42), 1 epoch, batch 10, via `make_supervised_step_fn` default dispatch — JevX-side params only (backbone frozen, 0 backbone forwards during training; test-local `_FeaturizedChoiceModel` feeds pre-featurized rows through the real `state_proj`/`question_encoder`/`schema_encoder`/`fusion`/`heads['choice']`).
- Reported: epoch loss, pre/post eval accuracy, featurize seconds — recorded in
  the JSON under `battery.8b_optional_head_training`.

**8b executed result (JSON-of-record, full-suite run)**: featurize-once
**1071 s** (200 rows → `[200, 1024]` bf16 feats through the real adapter; 0
backbone forwards during training), epoch-1 loss **115.77** (finite),
in-sample accuracy **0.520 → 0.480** (pre → post, one epoch), dataset
`nyu-mll/glue/sst2 [offline-cache]`. Independent targeted run: featurize
881.1 s, loss 200.76, in-sample acc **0.475 → 0.525** (+5.0 points).
**Protocol evidence only — NOT a quality claim** (frozen random-init
features, single epoch; head weights are randomly initialized per wrap, so
single-run accuracy movement over 1 epoch is noise-level either direction).

## 3. Hybrid-specific findings

**Detector output vs `layer_types`** — `JevXCapabilityDetector.detect` reports (verbatim quotes in JSON `battery.1_load.capability.reasons`):

- `"modalities from forward signature: text=True, vision=True, audio=False"`
- `"directions: is_encoder_decoder=False, is_decoder flag=None, encoder submodule=False, decoder submodule=False, output exposes past_key_values=True, pooler field/module=False/False → encoder=False, decoder=True"`
- `"multimodal interleaved states pool lossily → experimental (pin a token-selection convention)"` → **`jevx_conversion="experimental"`**
- `"vision/audio inputs require a processor or feature extractor"`
- `representation_field="last_hidden_state"`, `output_class="Qwen3_5ModelOutputWithPast"`.

**Does it notice `layer_types`? NO** — `capability_mentions_layer_types=false`. The detector is signature/config-field introspection only; `layer_types` appears nowhere in its reasons. That is scope, not a defect: conversion is decided by output fields, and the hybrid's `layer_types` does not change the extraction route (field `last_hidden_state` + generic pooling). The hybrid keys DO reach JevX intact (verified in config: 24 `layer_types`, `full_attention_interval=4`, `decision_hidden_size=1024` from nested `text_config`).

**Adapter resolution** — expected `DecoderAdapter`, actually **`MultimodalAdapter`**: the AutoModel chain loads the full `Qwen3_5Model` wrapper (vision tower present → `config.vision_config` → `vision=True` in detect), and the built-in chain checks multimodal before decoder. On text-only states `MultimodalAdapter` behaves like the generic path (identical `prepare_inputs`/`extract_representation`), differing only in `default_pooling="masked_mean"` instead of `last_valid` — the pooling actually taken (recorded in JSON). Not a defect (clip/llava precedent), but documented: the DecoderAdapter's cumsum-`position_ids` left/right-pad derivation is NOT active here — see 6c.

**Cache behavior (bug found+fixed, §4)** — pre-fix, `use_cache=False` was injected only when the forward *declared* the parameter; `Qwen3_5Model.forward` declares it solely via `**kwargs: Unpack[TransformersKwargs]`, so the config default (`use_cache: true`) built a fresh `DynamicCache` on every extraction. Post-fix the extraction output carries no cache. Raw default-mode forwards still build a `DynamicCache`; two identical sequential forwards are bitwise equal (no drift), and cache-vs-no-cache sequences agree exactly (0.0) — so the pre-fix behavior was a resource waste, not a correctness issue.

**Padding/decay measurements** — right-pad batch vs alone: JevX 0.125 / 0.03125 vs backbone-alone 0.0238 / 0.0100; left-vs-right: JevX 0.125 vs backbone 0.0465 (all ≤ 2-ulp noise floor 0.25 at the observed pooled magnitude). Mechanism: `apply_mask_to_padding_states` zeroes padded hidden states before each GDN block and the conv has `bias=False`, so pad steps inject exact zeros; state decay through padded steps therefore multiplies zeros; the full-attention layers see masked pads, and RoPE scores depend only on relative positions, which a uniform pad shift preserves. No JevX-level padding bug; the residual differences are bf16 kernel/store rounding, and JevX is never worse than the backbone above the noise floor.

**Featurizer resolution despite `processor_required=True`** — the text path works: `Qwen3VLProcessor` carries a `Qwen2Tokenizer`, so raw-`str` states tokenize normally (JSON records tokenizer/processor classes).

## 4. Bugs found + fixed

1. **`jevx/adapters/base.py:250-278` (`BaseAdapter.extract_representation`) — `use_cache=False` not injected for kwargs-only forwards.**
   - Cause: the gate tested only for an *explicitly declared* `use_cache` parameter. `Qwen3_5Model.forward` (transformers 5.16.1 `modeling_qwen3_5.py:1509`) declares `use_cache` only through `**kwargs: Unpack[TransformersKwargs]`, so the shared path fell through, the config default (`use_cache: true`) applied, and every extraction call allocated a `DynamicCache` (18 linear-attn + 6 full-attn layers) — contradicting the documented §5/§8.7 contract ("pass `use_cache=False` (done in the shared extraction path)", `jevx/adapters/decoder.py:12-14`).
   - Fix (minimal): inject `use_cache=False` when the forward declares the parameter **or** has `**kwargs`; a caller-supplied `use_cache` in `inputs` is passed through unchanged (it was previously filtered out for kwargs-only signatures). Docstring in `jevx/adapters/generic.py` updated to match.
   - Regression guard: `test_06a_hybrid_cache_behavior` asserts the extraction output carries no cache. Full suite re-run after the fix (summary in §7).
2. Other observations, deliberately NOT code changes (documented findings, not defects): adapter resolves `MultimodalAdapter` (by design, §3); detector is `layer_types`-blind and reports `experimental` for the vision-carrying wrapper (scope, conservative); capability `architecture` field reads `config.architectures` (`Qwen3_5ForConditionalGeneration`) while the loaded class is `Qwen3_5Model` — `backbone_config.architectures` is correctly pinned to the actual class for rebuilds (verified in §7 round-trip); `input_ids`+`pixel_values` without grid metadata surfaces a cryptic upstream `AttributeError` (JevX forwards declared kwargs unmodified — no JevX-side context wrapper exists for backbone-internal failures).

## 5. Honest limits

- **bf16 tolerances**: probability rows are stored bf16 → sum atol 4e-3 (measured ≤ 1.95e-3); cross-path pooled comparisons bounded by 2 ulp of the observed magnitude; same-call comparisons (base_forward, save/load) demanded < 1e-2 and measured 0.0 (bitwise).
- **CPU-only, 2 torch threads**: this CPU has no AMX/AVX512-BF16, so bf16 GEMMs run on slow fallback kernels — measured ~0.5 s/token; a single 20-row forward ≈ 3-4 min (8a forward wall 423.7 s). Timings are machine-specific.
- **Network**: `stanfordnlp/sst2` primary load hit SSL handshake timeouts (`_ssl.c:1063 handshake operation timed out` on `sst2.py` HEAD) and was replaced by `nyu-mll/glue/sst2` (identical rows) via the offline-cache ladder; both dataset loads are env-skipped (repo convention) if all sources fail.
- **Untrained head**: 8a/8b numbers are pipeline evidence only — NOT quality claims (labels in JSON and log).
- **Run-to-run head init**: JevX-side weights are randomly initialized at wrap
  time (no pretrained JevX head exists), so raw decision numbers (first probs,
  confidence, ES, 8a/8b metrics) vary across runs; every run re-verified the
  invariants (sums ≤ atol, bounds, bitwise base/save equalities — all stable).
- Save/load briefly holds two 853M copies (peak RSS recorded); phase 0 (raw reference) frees its copy before the wrap.
- 6b/6c compare JevX's bf16-*stored* pooled vector against fp32-pooled backbone references, which adds store rounding to the JevX side (documented in the JSON `interpretation` field).

## 6. Reproduction

```
cd C:\Users\ysnrfd\Documents\Transformer\JevX
python -m pytest tests/integration/test_qwen35_hybrid.py -q     # this battery (~1 h on this CPU)
python -m pytest -q                                              # full suite
```

Artifacts: `benchmarks/results/qwen35_jevx_validation.json` (measured numbers),
this report. Checkpoint must exist at the snapshot path in §1 (local, no download).

## 7. Verbatim pytest summaries

Targeted battery — `python -m pytest tests/integration/test_qwen35_hybrid.py -q`:

```
...........                                                              [100%]
11 passed in 1325.33s (0:22:05)
```

Full suite — `python -m pytest -q`:

```
779 passed, 1 skipped, 2 warnings in 1754.98s (0:29:14)
```

Baseline context: pre-change full-suite runs in this repo reported
`768 passed, 1 skipped, 2 warnings` — the delta is exactly the +11 new
battery tests; the single skip and both warnings (torch.jit FutureWarning,
audio mel-filter UserWarning) are pre-existing and unchanged.

---

## Addendum — 2026-09-25 (final verification pass)

Append-only: §§1–7 above are the original run's record and are left untouched;
numbers below come from the full-suite battery re-run of 2026-09-25
(`python -m pytest -q` → `883 passed, 1 skipped, 2 warnings in 2393.61s`,
`tests/integration/test_qwen35_hybrid.py` 11/11 green inside it; artifact
`benchmarks/results/qwen35_jevx_validation.json`, rewritten incrementally by
that run).

### A1. `attention_layout` is now reported (Known #1 flipped to asserted)

The capability detector parses the hybrid's declared `config.layer_types`
(direct and nested `text_config.layer_types`) and reports:

* `CapabilityReport.attention_layout == "hybrid:18xlinear_attention+6xfull_attention"`;
* a `reasons` entry starting `hybrid attention layout … 18 linear_attention + 6 full_attention`;
* `capability_mentions_layer_types == True`.

`test_01_load_and_capability` **asserts all three** (post-fix flip of the
original finding: the pre-fix battery merely *recorded* the detector's silence —
`capability_mentions_layer_types=false` — see §3 above). Implementation:
`jevx/capabilities/detector.py` (`_attention_layout`, `CapabilityReport.attention_layout`);
unit coverage `tests/unit/test_layer_types_detection.py` (9 tests: hybrid, uniform
full, uniform linear, sliding-only, absent/None/empty/malformed configs). The
field is informational only — verdicts (`jevx_conversion`) are unchanged.

### A2. Item 6c re-measured after the multimodal text-path `position_ids` fix

`MultimodalAdapter` now derives padding-consistent
`position_ids = cumsum(attention_mask) - 1` (clamped ≥ 0) for text-only inputs
whenever the backbone's forward declares `position_ids` and the caller supplied
none (`jevx/adapters/multimodal.py`, text-path guard at the top of the
derivation; skipped when the caller passed `position_ids`, when the forward
does not declare it, or for multimodal/3-D-position forwards that compute their
own). Left-pad vs right-pad, same content (item 6c), re-measured 2026-09-25:

| Quantity | Original run (§6c above) | Re-measured 2026-09-25 |
|---|---|---|
| JevX left-pad vs right-pad (max abs) | **0.125** | **0.0625** |
| backbone-alone left vs right (max abs) | 0.0465 | 0.0465 |
| bf16 2·ulp noise floor at batch max | 0.25 | 0.25 |
| JevX right-pad batch vs alone (row0/row1) | 0.125 / 0.03125 | 0.125 / 0.03125 |

The JevX-side left/right difference halved (0.125 → 0.0625 — both far below the
0.25 bf16 floor) after the text-path `position_ids` derivation went in; the
backbone-alone reference is unchanged, as expected (the fix is JevX-side
adapter plumbing, not a backbone change). The asserted bar
(`jevx ≤ max(backbone, 2·ulp) + 1e-6`) held before and after; §6c's PASS
rationale and the "no unconditional equality is asserted for this hybrid"
caveat are unchanged. Original §6c numbers remain above as history.

### A3. Item 9 now records a JevX-wrapped error (Q-013 `call_backbone`)

The `input_ids` + `pixel_values` probe, recorded as a *cryptic upstream*
`AttributeError` in the original run (§4.2, "documented finding, not defect"),
now surfaces as a JevX-owned error through `call_backbone`
(`jevx/adapters/base.py`, audit Q-013):

```
ValueError: JevX could not call the backbone Qwen3_5Model with the prepared inputs
(keys: ['attention_mask', 'input_ids', 'pixel_values', 'use_cache']). Upstream
AttributeError: … This is an INPUT/MODALITY mismatch, not a model bug: …
```

The original upstream `AttributeError` (`'NoneType' object has no attribute
'device'`, `image_grid_thw` missing) is preserved as `__cause__`. The battery's
`quality_notes` annotation in the test source was corrected to match
(annotation only — no assertion changed; this run's JSON was flushed *before*
that source-side correction, so its `quality_notes` string still carries the
pre-fix wording while its measured `errors` entry — quoted above — is the
wrapped `ValueError`). `tests/unit/test_error_wrapping.py` owns the enforcing
tests, 9 of them, including the end-to-end `extract_representation` wrap.
Everything else in item 9 is unchanged: `pixel_values`-only still raises the
upstream `ValueError` naming `input_ids`/`inputs_embeds`, and the five
JevX-owned garbage-input probes still raise actionable `TypeError`/`ValueError`.

### A4. Re-run invariants (unchanged, re-asserted 2026-09-25)

| Battery item | Re-measured |
|---|---|
| 2 extraction vs independent fp32 reference | 5.059e-02 ≤ 2-ulp bound 2.5e-01 |
| 3 `base_forward` vs raw AutoModel | max_abs_error 0.0, bitwise equal |
| 4 row-sum max dev (bf16) | 1.953e-3 ≤ atol 4e-3 |
| 5 shared-state m=4 backbone forwards | exactly 1 |
| 7 save/load probs max_abs_error | 0.0 (save 58.3 s) |
| 6b right-pad batch vs alone | 0.125 / 0.03125 (bar held) |

8a/8b remain protocol evidence only (untrained/frozen-random heads; run-to-run
numbers vary with head init — §5). This run: 8a acc 0.450 / NLL 1.852
(40 rows, `stanfordnlp/sst2`, forward wall 402.2 s); 8b executed (featurize
1244.7 s, epoch loss 164.82, in-sample 0.520 → 0.470). No quality claim.
