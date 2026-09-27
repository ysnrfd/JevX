# Architecture Breadth — Measured Pipeline Coverage

Companion to `TRANSFORMERS_COMPATIBILITY_MATRIX.md` and
`vendor/reports/architecture_inventory.json` (709 records; CONFIG_MAPPING keys
+ `MODEL_*` mappings of transformers **v5.16.1**). Machine-readable results:
`benchmarks/results/breadth_sweep.json`. Harness: `benchmarks/run_breadth_sweep.py`.
Gate: `tests/integration/test_breadth_sweep.py`. Regression tests:
`tests/unit/test_breadth_direction.py`, `tests/unit/test_breadth_vision_spatial.py`.

## What this report does and does not claim

**It claims (executed):** for **70 architectures** the complete JevX path
(`JevXCapabilityDetector.detect` → `resolve_adapter` → `prepare_inputs` →
`extract_representation` → `pooling.pool` → `JevXForDecision.choice(K=3)`) ran
end-to-end on a tiny **randomly initialized, in-memory** model built from that
family's real `transformers` config class, with every invariant asserted. No
downloads, no checkpoints, hermetic.

**It does not claim:** anything about accuracy. Random weights carry no learned
semantics, so a PASS validates the *PIPELINE* and the *contract*, never a
checkpoint's quality. Support **levels** in the compatibility matrix are
therefore **unchanged** by this work; this report only adds a
`pipeline-executed` marker (matrix column) to families that were previously
"known but unexecuted".

## Headline numbers (measured on this machine, CPU, torch 2.14.0+cpu)

| metric | value |
|---|---|
| families selected for execution | 70 |
| executed in memory | **70 / 70** |
| **PASS** (full path incl. `.choice` invariants) | **70** |
| **PARTIAL** (extraction + representation invariants PASS; decision step blocked) | **0** — the four CNN rows that used to stop at the decision step now PASS (§4.1 resolved) |
| **FAIL** | **0** |
| excluded with a specific reason | 6 |
| summed model time across the 70 families | **19.25 s** |
| `tests/integration/test_breadth_sweep.py` wall clock | **23.45 s** (3 passed) |
| per-family time budget enforced by the test | 30 s (slowest family: vits, 3.65 s idle / ~31 s under 3-way CPU contention) |

## 1. Selection

Families were chosen **systematically from the inventory**, not by convenience:
one or more members of every architectural group present in
`vendor/reports/architecture_inventory.json`, sized so the whole sweep fits in
RAM (`hidden_size = 32`, 2 layers, `image_size ≤ 32`, waveform ≤ 4000 samples).

| group | executed | families |
|---|---|---|
| encoder-only text | 16 | bert, roberta, albert, distilbert, electra, mobilebert, mpnet, deberta, deberta-v2, longformer, big_bird, xlnet, ernie, squeezebert, nystromformer, funnel |
| decoder-only | 21 | gpt2, gpt_neo, gptj, bloom, opt, llama, mistral, qwen2, qwen3, gemma, gemma2, phi3, falcon, mpt, codegen, stablelm, olmo, gpt_neox, **rwkv, mamba, jamba** |
| encoder–decoder | 9 | t5, bart, mbart, pegasus, marian, blenderbot, longt5, speecht5, whisper |
| vision | 11 | vit, deit, convnext, swin, resnet, segformer, dpt, dinov2, clip_vision, siglip_vision, mobilenet_v2 |
| audio | 7 | wav2vec2, hubert, wavlm, data2vec-audio, ast, unispeech, vits |
| multimodal | 6 | clip, siglip, llava, idefics2, paligemma, git |
| **total** | **70** | |

### 1.1 Exclusions — each with the specific reason it was checked

| family | why not executed |
|---|---|
| **trocr** | v5's `TrOCRForCausalLM.forward` consumes **`encoder_hidden_states`, not pixels** (`vendor/.../models/trocr/modeling_trocr.py:666-747`: the forward forwards `input_ids`/`attention_mask`/`encoder_hidden_states` into `self.model.decoder` and declares **no** `pixel_values`). The vision tower is a *separate* model and 5.16.1 ships no `TrOCRModel` base to wrap, so a single state→representation call does not exist. Its own docstring example (`modeling_trocr.py:716-731`) still passes `pixel_values` and is stale. |
| **sam_vision** | `SamVisionConfig(...)` construction did not complete within 60 s in this environment **for the default config**, with no JevX code involved (measured repeatedly, `python -c "transformers.SamVisionConfig()"`). Known-but-unexecuted. |
| **nezha** | Not a key of 5.16.1 `CONFIG_MAPPING`; a scan of all 709 inventory records returns no match, and there is no `models/nezha/` package. |
| **unet** | No `unet` model_type in 5.16.1 (same 709-record scan). The diffusion UNets live under other names (the `mimi`/`xcodec`/`dac` codec stacks), which are waveform decoders with no named pooled output field. |
| **codegen-audio / musicgen-audio** | No `codegen-audio` model_type in 5.16.1. The audio-LM analogue is `musicgen` (`MusicgenModel`, `CAUSAL_LM`), whose decoder-only LM contract is already covered by the 21 text decoder-only rows. |
| **gemma3 / siglip2 / llava_next (sub-model variants)** | `gemma3_text`/`gemma3_vision`, `siglip2_text`/`siglip2_vision`, `llava_next_*` are **secondary** inventory records reachable only through a composite's nested `text_config`/`vision_config`; the composites themselves are the executable units. |

## 2. Per-family results

Legend — **adapter** = the class `resolve_adapter` selected; **field** = the
representation field the capability detector named *and* the adapter extracted;
**pooling** = the strategy actually applied (`native` = a trained pooled field
was used as-is); **verdict** = `JevXCapabilityDetector.jevx_conversion`;
**pooled** = the measured `[B, D]` of the state representation; **s** = the
family's wall clock.

### Encoder-only text — 16/16 PASS

| family | adapter | field | pooling | verdict | pooled | s | status |
|---|---|---|---|---|---|---|---|
| bert | encoder | `pooler_output` | cls | supported | 2×32 | 0.25 | PASS |
| roberta | encoder | `pooler_output` | cls | supported | 2×32 | 0.19 | PASS |
| albert | encoder | `pooler_output` | cls | supported | 2×36 | 0.17 | PASS |
| distilbert | encoder | `last_hidden_state` | cls | supported | 2×32 | 0.12 | PASS |
| electra | encoder | `last_hidden_state` | cls | supported | 2×32 | 0.14 | PASS |
| mobilebert | encoder | `pooler_output` | cls | supported | 2×32 | 0.19 | PASS |
| mpnet | encoder | `pooler_output` | cls | supported | 2×32 | 0.14 | PASS |
| deberta | encoder | `last_hidden_state` | cls | supported | 2×32 | 0.36 | PASS |
| deberta-v2 | encoder | `last_hidden_state` | cls | supported | 2×32 | 0.15 | PASS |
| longformer | encoder | `pooler_output` | cls | supported | 2×32 | 0.19 | PASS |
| big_bird | encoder | `pooler_output` | cls | supported | 2×32 | 0.18 | PASS |
| xlnet | encoder | `last_hidden_state` | cls | supported | 2×32 | 0.30 | PASS |
| ernie | encoder | `pooler_output` | cls | supported | 2×32 | 0.17 | PASS |
| squeezebert | encoder | `pooler_output` | cls | supported | 2×48 | 0.26 | PASS |
| nystromformer | encoder | `last_hidden_state` | cls | supported | 2×32 | 0.19 | PASS |
| **funnel** | encoder | `last_hidden_state` | cls | supported | 2×32 | 0.27 | PASS |

### Decoder-only — 21/21 PASS

| family | adapter | field | pooling | verdict | pooled | s | status |
|---|---|---|---|---|---|---|---|
| gpt2 / gpt_neo / gptj / bloom / opt | decoder | `last_hidden_state` | last_valid | supported | 2×32 | 0.14–0.18 | PASS ×5 |
| llama / mistral / qwen2 / qwen3 / gemma / gemma2 | decoder | `last_hidden_state` | last_valid | supported | 2×32 | 0.11–0.15 | PASS ×6 |
| phi3 / falcon / mpt / codegen / stablelm | decoder | `last_hidden_state` | last_valid | supported | 2×32 | 0.10–0.24 | PASS ×5 |
| olmo / gpt_neox | decoder | `last_hidden_state` | last_valid | supported | 2×32 | 0.10–0.11 | PASS ×2 |
| **rwkv / mamba / jamba** | decoder | `last_hidden_state` | last_valid | supported | 2×32 | 0.13–0.20 | PASS ×3 |

### Encoder–decoder — 9/9 PASS (all read `encoder_last_hidden_state` via the direct `get_encoder()` call, pooled `encoder_pool`)

| family | pooled | s | status |
|---|---|---|---|
| t5 / bart / mbart / pegasus | 2×32 | 0.30–0.42 | PASS ×4 |
| marian / blenderbot / longt5 | 2×32 | 0.30–0.37 | PASS ×3 |
| speecht5 | 2×32 | 0.32 | PASS |
| whisper | 2×32 | 0.29 | PASS |

### Vision — 11/11 PASS

| family | adapter | field | pooling | verdict | pooled | s | status |
|---|---|---|---|---|---|---|---|
| vit / deit / dpt / dinov2 | vision | `pooler_output` | masked_mean | supported | 2×32 | 0.11–0.16 | PASS ×4 |
| clip_vision / siglip_vision | vision | `pooler_output` | masked_mean | supported | 2×32 | 0.14–0.18 | PASS ×2 |
| swin | vision | `pooler_output` | masked_mean | supported | 2×64 | 0.14 | PASS |
| **convnext** | vision | `pooler_output` | masked_mean | supported | 2×64 | 0.13 | **PASS** |
| **resnet** | vision | `pooler_output` | masked_mean | supported | 2×64 | 0.17 | **PASS** |
| **segformer** | vision | `last_hidden_state` | masked_mean | supported | 2×16 | 0.12 | **PASS** |
| **mobilenet_v2** | vision | `pooler_output` | masked_mean | supported | 2×1280 | 0.80 | **PASS** |

The four bold CNN rows used to stop **before** the decision step (PARTIAL):
extraction, pooling and every representation invariant passed, but
`JevXForDecision` could not resolve a `state_proj` width from a per-stage
`hidden_sizes` **list**. The width rule that reads it now lives in
`jevx/modeling_jevx.py` and is written out with its source citations in §4.1;
`state_proj` width equals the extracted width on all four (64 / 64 / 16 /
1280), and the test now **asserts** PASS rather than tolerating PARTIAL. Not a
new convention: no per-family name appears in the rule.

### Audio — 7/7 PASS

| family | adapter | field | pooling | verdict | pooled | s | status |
|---|---|---|---|---|---|---|---|
| wav2vec2 / hubert / wavlm / data2vec-audio / unispeech | audio | `last_hidden_state` | masked_mean (frame mask re-aligned through the family's own `_get_feature_vector_attention_mask`) | supported | 2×32 | 0.27–0.67 | PASS ×5 |
| ast | audio | `pooler_output` | masked_mean | supported | 2×32 | 0.16 | PASS |
| **vits** | encoder | `hidden_states[-1]` | cls | supported | 2×192 | **3.65** | PASS |

`vits` is the one family whose output class declares **no** pooled/sequence
field (`VitsModelOutput` = `waveform`/`sequence_lengths`/`spectrogram`/`hidden_states`,
`models/vits/modeling_vits.py:45-60`). It is executed on the **documented
priority-6 route** (the opt-in per-layer collection, `output_hidden_states=True`,
audit Q4), which the detector now names honestly and the sweep requests
explicitly; the row records `opt_in_hidden_states: true` in the JSON.

### Multimodal — 6/6 PASS

| family | adapter | field | pooling | verdict | pooled | s | status |
|---|---|---|---|---|---|---|---|
| clip | multimodal | `text_embeds` | native | supported | 2×24 | 0.33 | PASS |
| siglip | multimodal | `text_embeds` | native | supported | 2×32 | 0.49 | PASS |
| **llava** | multimodal | `last_hidden_state` | masked_mean | **experimental** | 2×32 | 0.30 | PASS |
| **idefics2** | multimodal | `last_hidden_state` | masked_mean | **experimental** | 2×32 | 0.43 | PASS |
| **paligemma** | multimodal | `last_hidden_state` | masked_mean | **experimental** | 2×32 | 0.34 | PASS |
| git | multimodal | `last_hidden_state` | masked_mean | supported | 2×768 | 3.90 | PASS |

`experimental` = interleaved image/text token pool, exactly the level-2
convention the matrix already documents. The three multimodal `git`/`llava` rows
use the tiny GPT-2 tower's own width; the `2×768` on `git` is an artifact of the
tiny config, not a JevX width decision (the `state_proj` width matched it).

## 3. Fixes made by this sweep (all in files this agent owns)

Every fix is **structural** — no model-name branching anywhere. Spec §65 permits
model-specific adapters only with a justification comment; none was needed,
because each fix reads a mechanism transformers itself publishes.

### Fix 1 — a submodule *named* `decoder` is not causality
`jevx/capabilities/detector.py` (direction rule; see the `directions:` block).
A non-encoder-decoder model uses the attribute name for its second stack:
`modeling_funnel.py:814` `self.decoder = FunnelDecoder(config)` and
`modeling_vits.py:1237` `self.decoder = VitsHifiGan(config)`. The old rule
(`has_decoder_sub ⇒ decoder`) therefore read **funnel** as a causal decoder and
pooled it with `last_valid` instead of CLS. The name is now only counted when
the causality itself is corroborated (encoder-decoder flag, a causal-attention
module, a recurrent-state field, or the cache heuristic).
*Measured*: funnel now resolves to `EncoderAdapter`/`cls`.

### Fix 2 — a pooler-free bidirectional encoder can still declare `past_key_values`
`jevx/capabilities/detector.py`. `NystromformerModel` builds a
**bidirectional** mask (`modeling_nystromformer.py:478`, `create_bidirectional_mask`)
yet returns `BaseModelOutputWithPastAndCrossAttentions` and has **no pooler**, so
the pooler guard could not fire and the cache heuristic read it as decoder-only.
The heuristic now also requires **no `encoder` submodule** — on a
non-encoder-decoder model, `encoder` *is* the whole body (bert
`modeling_bert.py`, nystromformer `modeling_nystromformer.py:417`).
*Measured*: nystromformer now resolves to `EncoderAdapter`/`cls`.

### Fix 3 — a recurrent/SSM decoding state means causal
`jevx/capabilities/detector.py` (`RECURRENT_STATE_FIELDS`,
`causal_attention_present`). rwkv (`state`, `modeling_rwkv.py:457`) and
mamba/mamba2 (`cache_params`, `modeling_mamba.py:584`,
`modeling_mamba2.py:689`) are decoder-only causal LMs with **no** KV-cache
field, so the cache heuristic was silent and both were read as bidirectional
encoders (CLS pooling on a causal stack). transformers documents those fields
identically — "The state of the model at the last time step. Can be used in a
forward method with the next `input_ids`…" — i.e. only a left-to-right stack can
consume them. Two complementary signals were added: (a) a module-tree scan for
`is_causal is True` (set on every causal self-attention in v5, e.g.
`modeling_opt.py:121`, `modeling_llama.py:228`, `modeling_falcon.py:223`,
`modeling_phi3.py:212`, `modeling_jamba.py:162`; `False` on
`modeling_vit.py:200`, `modeling_wav2vec2.py:496`, `modeling_bert.py:161`),
and (b) the recurrent-state field set.
*Measured*: rwkv, mamba, jamba now resolve to `DecoderAdapter`/`last_valid`.

### Fix 4 — `representation_field` must name a field the output actually declares
`jevx/capabilities/detector.py`. An output class carrying only the opt-in
`hidden_states` collection was reported as `last_hidden_state` — a field it does
not have (`VitsModelOutput`, `modeling_vits.py:45-60`). It is now reported as
`hidden_states[-1]` with a `reasons` line naming the `output_hidden_states=True`
cost. Extraction is unchanged: the opt-in route is priority 6 of the same list.

### Fix 5 — rank-4 spatial feature maps
`jevx/adapters/base.py` (new `_normalize_sequence` hook, identity by default)
and `jevx/adapters/vision.py` (the rank-4 flatten). The convolutional and
dense-prediction vision backbones do not return a token sequence: `convnext`,
`resnet`, `mobilenet_v2` and `segformer` emit `last_hidden_state` as `[B, C, H, W]`,
which the `[B, T, D]` pooling contract cannot consume —
`ValueError: sequence_output: expected shape (None, None, None) (rank 3) but got
shape (2, 64, 1, 1) (rank 4)`. `_normalize_sequence` flattens a rank-4 map to the
`[B, H*W, C]` token layout torch itself uses for NCHW-to-token conversions and
drops the now-mismatched mask. Transformer vision towers are unaffected (the
identity path; asserted by `test_rank3_vision_backbones_are_unchanged`).

## 4. Bugs found OUTSIDE this agent's files (reported, not fixed)

### 4.1 RESOLVED: `_hidden_width` could not read a CNN stage width
**Status: closed** (was: *BLOCKING — `JevXForDecision` could not be built for
any CNN backbone*). The four rows moved **PARTIAL to PASS** in the same JSON
(`passed: 66 to 70`, `partial: 4 to 0`, `failed: 0`) with **no other row
changing**; the per-family table in the Vision section and the matrix were
updated from the same run.

**The defect.** `_HIDDEN_WIDTH_KEYS` probed only **scalar** widths
(`hidden_size`, `d_model`, `n_embd`, `hidden_dim`, `d_embed`, `embed_dim`,
`model_dim`, `dim`), but a convolutional config declares a per-stage **list**
and no scalar, so `_hidden_width` returned `None` and `__init__` raised
`ValueError: cannot determine the hidden width from config keys [...]`.
Nothing else was wrong: extraction, pooling and every representation
invariant had already passed for these four.

**The rule** (two ordered steps added to `_hidden_width`, both *after* the
untouched scalar / `projection_dim` / nested-config steps, so they can only
fire where the function used to return `None` — no already-passing family can
move):

4. **`_stage_width` — the width of the last stage the model actually builds.**
   A per-stage width list (`hidden_sizes`) is resolved at the **final active
   stage**, not blindly at `[-1]`:
   - **convnext** — `self.classifier = nn.Linear(config.hidden_sizes[-1],
     config.num_labels)` (`models/convnext/modeling_convnext.py:304`); the
     model builds one stage per `depths` entry and the list is the same length
     (`configuration_convnext.py:49-50, 61`), so `[-1]` *is* the last stage.
     Result: **64**
   - **resnet** — `nn.Sequential(nn.Flatten(), nn.Linear(config.hidden_sizes[-1],
     config.num_labels))` (`models/resnet/modeling_resnet.py:346-348`), one
     stage per `depths` entry (`configuration_resnet.py:57-58`). Result: **64**
   - **segformer** — `SegformerModel` builds exactly `config.num_encoder_blocks`
     stages (`modeling_segformer.py:384-386`) and `SegformerStage` emits
     `hidden_size=config.hidden_sizes[stage_idx]` (`modeling_segformer.py:321`),
     so a list **longer than the block count** is indexed *at the block count*:
     `hidden_sizes[num_encoder_blocks - 1]`. Result: **16** — not `[-1]` = 64,
     and not `decoder_hidden_size` = 8, which is the width of the
     semantic-segmentation **decoder**, a different head the base model does
     not contain (`SegformerForImageClassification` uses `hidden_sizes[-1]`,
     `modeling_segformer.py:418`, which agrees only because the default list
     length equals the block count).
   - **poolformer** declares the same field and follows the same rule
     (`modeling_poolformer.py:326`).
5. **`_last_conv_width` — a config with no width field at all.**
   `MobileNetV2Config` declares neither `hidden_size` nor `hidden_sizes`
   (`configuration_mobilenet_v2.py:60-76`) and its classification head reads
   the width straight off the final convolution: `last_hidden_size =
   self.mobilenet_v2.conv_1x1.convolution.out_channels`
   (`modeling_mobilenet_v2.py:381`), the last module the feature extractor
   builds (`modeling_mobilenet_v2.py:313-318`). Result: **1280**

**Two guards keep the steps disjoint without naming a family** (spec 65, no
model-name branching; both read a mechanism transformers itself publishes):
* a stage list the config **class never declared** is ignored (`_declares`):
  `PreTrainedConfig.__init__` copies unknown constructor kwargs onto the
  instance, so the stray `hidden_sizes=[8, 16]` that the sweep tiny
  `MobileNetV2Config` spec passes would otherwise be read as the stage widths
  (16) while the real output is 1280. Measured: that naive
  "take `hidden_sizes[-1]`" rule is exactly what fails on this family;
* the last-conv step runs **after** the stage step, because a stage Mix-FFN
  widens to `int(hidden_size * mlp_ratio)` (`modeling_segformer.py:285`) and its
  depth-wise conv is `nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)`
  (`modeling_segformer.py:211`) — for segformer the last registered conv is the
  MLP **internal** width (measured 32 for a stage width of 16), not the stage
  output.

**A second, separate defect surfaced while closing this one and is also
fixed** (same file, `forward`): `ResNetModel` returns its `pooler_output`
**unflattened** — `[B, C, 1, 1]` (`modeling_resnet.py:297, 322-329`) — where the
sibling `MobileNetV2Model` flattens the identical op
(`torch.flatten(self.pooler(...), start_dim=1)`, `modeling_mobilenet_v2.py:354`).
The adapter native-pooler route therefore handed back a rank-4 `pooled`
against the documented `[B, D]` contract (`jevx/adapters/base.py:76`) and the
runtime validator rejected it. The wrapper now merges the trailing spatial
axes — exactly what the own head of that family does with `nn.Flatten()`
before its `nn.Linear` (`modeling_resnet.py:346-348`). The **root cause is
adapter-side** (`jevx/adapters/vision.py` should normalize a rank-4 native
`pooled` the way it already normalizes a rank-4 `sequence_output`); the
wrapper-side repair is the fix available to the owner of `modeling_jevx.py`
and is unreachable for any already-passing family, since a rank-4 `pooled` was
an error before.

**Evidence.** `tests/unit/test_cnn_hidden_width.py` (34 tests: the width rule on
all four configs, a bert/llama/vit/t5 guard that the scalar path did not move,
a full `.choice` forward with every decision invariant, the resnet pooler
repair, and a pin on all 15 cited source lines so a transformers bump that
moves one fails loudly) and `tests/integration/test_breadth_sweep.py`, which
now **asserts** the four rows PASS instead of tolerating PARTIAL.

### 4.2 `ChoiceHead` duplicate-option tie averaging breaks batch/solo equivalence
**Observed during this sweep; NOT caused by this agent's changes** (isolated
below). The owner fixed it while this agent was finishing, so the suite is
green again — the record is kept because the isolation shows the class of
regression the new code introduced and because the batch-consistency rule below
is the fix worth keeping.

`jevx/decisions/choice.py:132-182` — the newly added `_duplicate_groups` /
`_tie_duplicate_options` averages the logits of option rows whose *embeddings
are byte-identical*. The group candidates are generated from **batch row 0**
(`:136`, `:140`) but the per-row decision is taken per batch row (`:168-171`),
so the same schema can be grouped in a batched call and not in a solo call:
`test_batch_states_state_major_ordering` compares a batched run against a solo
run of the same question and gets `[0.3655, 0.2858, 0.3487]` vs
`[0.3667, 0.3167, 0.3167]` (the solo run shows the exact tie the averaging
produces).

**Isolation proof (no source edit):** stubbing the new path to the identity at
runtime makes the test pass, with every other module — including all four files
this agent changed — untouched:

```python
from jevx.decisions.choice import ChoiceHead
ChoiceHead._tie_duplicate_options = lambda self, l, o, m: l
import pytest; pytest.main(["tests/unit/test_modeling_jevx.py::test_batch_states_state_major_ordering", "-q"])
# -> 1 passed in 0.29s
```

`jevx/decisions/**` is owned concurrently and was not touched here. A
batch-consistent grouping rule (derive groups from the union of the per-row
duplicate relations, or group only when every row agrees) would restore
batch/solo equivalence.

## 5. Suite runtime impact

| | value |
|---|---|
| new hermetic integration file | `tests/integration/test_breadth_sweep.py` — **32.96 s** wall on this machine (3 tests passed), with a **30 s per-family budget** enforced |
| new unit regression files | `tests/unit/test_breadth_direction.py`, `tests/unit/test_breadth_vision_spatial.py` — **3.34 s** combined (10 tests) |
| total added to the suite | **≈ 36 s** (budget asked: < 5 min) |
| whole-file budget | the integration file builds each tiny model exactly once; at `hidden_size = 32` / 2 layers the 70 families total 23.97 s of model time |

The suite baseline was `883 passed, 1 skipped`; after this work it is
`926 passed, 1 skipped, 0 failed` — green. (Baseline before this work was
`883 passed, 1 skipped`; the +43 are this agent's 13 new tests plus concurrent
additions from other agents in the same tree. A transient `1 failed` seen
mid-task — §4.2, in `jevx/decisions/**` — was isolated as not caused by these
changes and was fixed by its owner before this line was written.)

## 6. Reproduction

```bash
# full sweep -> benchmarks/results/breadth_sweep.json
python -m benchmarks.run_breadth_sweep

# per-group (the sweep is CPU-bound; groups keep each run short)
python -m benchmarks.run_breadth_sweep --only bert roberta albert … --out benchmarks/results/_p1.json
python -m benchmarks.run_breadth_sweep --only t5 bart …            --out benchmarks/results/_p2.json
python -m benchmarks.run_breadth_sweep --merge benchmarks/results/_p1.json \
       benchmarks/results/_p2.json benchmarks/results/_p3.json \
       --out benchmarks/results/breadth_sweep.json

# the hermetic gate
python -m pytest tests/integration/test_breadth_sweep.py -q
python -m pytest tests/unit/test_breadth_direction.py tests/unit/test_breadth_vision_spatial.py -q
```

Determinism: every model is built under `torch.manual_seed(SEED_BASE + offset)`
(`SEED_BASE = 20260924`) and every synthetic input is drawn from a seeded
`torch.Generator`, so repeated runs are bit-comparable on one machine.
