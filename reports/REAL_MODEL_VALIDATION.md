# Real Pretrained Model Validation Matrix (§92-94)

**Scope** — REAL executed validation of JevX across encoder-only, decoder-only,
encoder–decoder, semantic/embedding, vision, audio and multimodal families with
REAL small checkpoints on CPU. Companion to
`TRANSFORMERS_COMPATIBILITY_MATRIX.md` (support levels) and
`reports/TRANSFORMERS_SOURCE_AUDIT.md` (source evidence).

**Status: 18/18 checkpoints executed green.** Every row below ran the full
§92 protocol: load → real-modality input → decision invariants
(`.choice`/`.score`/`.noul`) → `save_pretrained → from_pretrained` decision
equality → `base_forward` vs a freshly loaded raw `AutoModel` → shared-state
single forward (m=4) → §71 extract-representation correctness vs an
**independent manual reference** (own forward + own pooling math, never
`jevx.pool`). Machine-readable results:
`vendor/reports/_jevx_matrix_results.json` (regenerated on every test run;
per-id resolved commit shas in `vendor/reports/_matrix_env_probe.json`).

**Honesty note**: except `sentence-transformers/all-MiniLM-L6-v2`, every
checkpoint carries **random weights** (`tiny-random-*`, `sshleifer/tiny-gpt2`)
— these rows prove the *pipeline* is correct end-to-end (functional
validation), **not** predictive quality. MiniLM is a real trained
sentence-embedding checkpoint and is additionally pinned to the
sentence-transformers `masked_mean` pooling convention, but no similarity-task
metrics were run here — quality numbers belong to the benchmark/dataset agent.

---

## §93 — Validation matrix (all values measured, this machine)

Accuracy / F1 / NLL / Brier / ECE / Latency / Memory: see
[Quality-metric pointers](#quality-metric-pointers) below — **not fabricated
here**; the dataset/benchmark agent owns quality numbers.

| Model | Architecture | Modality | Params | Input Type | Representation | JevX Mode | Load | Forward | Save/Load | Numerical Check | Real Data | Support Level | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `hf-internal-testing/tiny-random-bert` | bert | text | 362,906 | tokenized text | `pooler_output` | encoder / supported | 3.21 s | invariants ✓, m=4 fwd 0.026 s | **0.0** | base 0.0 · extract 0.0 · state_repr 0.0 | real English sentences | **3** (was 4 → quality gate) | also E2E in `tests/integration/test_jevx_e2e_tiny.py`; golden fixture |
| `hf-internal-testing/tiny-random-RobertaModel` | roberta | text | 359,706 | tokenized text | `pooler_output` | encoder / supported | 3.70 s | ✓, 0.033 s | **0.0** | 0.0 / 0.0 / 0.0 | real sentences | **3** (was 4 → quality gate) | |
| `hf-internal-testing/tiny-random-DistilBertModel` | distilbert | text | 361,338 | tokenized text | `last_hidden_state` (CLS) | encoder / supported | 5.00 s | ✓, 0.026 s | **0.0** | 0.0 / 0.0 / 0.0 | real sentences | **3** | no pooler → CLS convention |
| `hf-internal-testing/tiny-random-AlbertModel` | albert | text | 4,274,147 | tokenized text | `pooler_output` | encoder / supported | 4.03 s | ✓, 0.035 s | **0.0** | 0.0 / 0.0 / 0.0 | real sentences | **3** (was 4 → quality gate) | sentencepiece fast-tokenizer path OK without slow tokenizer |
| `hf-internal-testing/tiny-random-DebertaV2Model` | deberta-v2 | text | 4,421,914 | tokenized text | `last_hidden_state` (CLS) | encoder / supported | 7.20 s | ✓, 0.119 s | **0.0** | 0.0 / 0.0 / 0.0 | real sentences | **3** | base model has no pooler |
| `sentence-transformers/all-MiniLM-L6-v2` | bert (semantic) | text | 27,634,945 | tokenized text | `last_hidden_state` → `masked_mean` | encoder / supported | 3.86 s | ✓, 0.102 s | **0.0** | 0.0 / 0.0 / 0.0 | real sentences | **3** | **real trained embedding checkpoint**, ST pooling convention pinned via `pooling_method="masked_mean"`; FUNCTIONALLY validated, NOT quality-validated |
| `hf-internal-testing/tiny-random-t5` | t5 | text | 396,321 | tokenized text | `encoder_last_hidden_state` | encoder_decoder / supported | 4.69 s | ✓, 0.079 s | **0.0** | 0.0 / 0.0 / 0.0 | real sentences | **3** | also E2E in `test_jevx_e2e_tiny.py`; golden fixture |
| `hf-internal-testing/tiny-random-BartForConditionalGeneration` | bart | text | 161,569 | tokenized text | `encoder_last_hidden_state` | encoder_decoder / supported | 4.31 s | ✓, 0.019 s | **0.0** | 0.0 / 0.0 / 0.0 | real sentences | **3** | |
| `sshleifer/tiny-gpt2` | gpt2 | text | 119,181 | tokenized text | `last_hidden_state` (`last_valid`) | decoder / supported | 3.46 s | ✓, 0.043 s | **0.0** | 0.0 / 0.0 / 0.0 | real sentences | **2** (executed) | also E2E + golden fixture; lossy-pooling semantics keep it at 2 |
| `yujiepan/qwen2-tiny-random` | qwen2 | text | 1,284,177 | tokenized text | `last_hidden_state` (`last_valid`) | decoder / supported | 5.02 s | ✓, 0.045 s | **0.0** | 0.0 / 0.0 / 0.0 | real sentences | **2** (executed) | modern decoder LLM family (verified alternate; `hf-internal-testing/tiny-random-Qwen2ForCausalLM` 401s) |
| `hf-internal-testing/tiny-random-LlamaForCausalLM` | llama | text | 654,689 | tokenized text | `last_hidden_state` (`last_valid`) | decoder / supported | 4.26 s | ✓, 0.091 s | **0.0** | 0.0 / 0.0 / 0.0 | real sentences | **2** (executed) | |
| `hf-internal-testing/tiny-random-ViTModel` | vit | vision | 317,722 | 224×224 PIL image → `ViTImageProcessor` | `pooler_output` | vision / supported | 10.15 s | ✓, 0.020 s | **0.0** | 0.0 / 0.0 / 0.0 | deterministic PNG fixture through the image processor | **3** (was 4 → quality gate) | fixture `tests/golden/fixtures/input_image_224.png` |
| `hf-internal-testing/tiny-random-CLIPModel` | clip | text+vision | 703,508 | image + text → `CLIPProcessor` | `text_embeds` (post-projection) | multimodal / supported | 10.94 s | ✓, 0.032 s | **0.0** | 0.0 / 0.0 / 0.0 | real image + real text through the processor | **3** (was 4 → quality gate) | both towers in one forward; detector field alignment fixed (bug 1) |
| `hf-internal-testing/tiny-random-SiglipModel` | siglip | text+vision | 1,358,076 | image + text, `padding="max_length"` | `text_embeds` (post-projection) | multimodal / supported | 8.60 s | ✓, 0.029 s | **0.0** | 0.0 / 0.0 / 0.0 | real image + real text; max_length padding asserted | **3** (was 4 → quality gate) | needed `pip install sentencepiece` (0.2.2); pad mask built as `input_ids != pad` |
| `hf-internal-testing/tiny-random-Wav2Vec2Model` | wav2vec2 | audio | 161,153 | 1 s `.wav` file → soundfile → `Wav2Vec2Processor` → `input_values` | `last_hidden_state` (masked mean) | audio / supported | 11.41 s | ✓, 0.024 s | **0.0** | 0.0 / 0.0 / 0.0 | **real wav file on disk** (fixture, soundfile read) | **3** (input pipeline was 1) | reload crash fixed (bug 3), drift fixed (bug 4) |
| `hf-internal-testing/tiny-random-ASTModel` | audio-spectrogram-transformer | audio | 312,346 | 1 s `.wav` → numpy fbank (FE rebuilt from model config) → `input_values` | `pooler_output` | audio / supported | 13.48 s | ✓, 0.034 s | **0.0** | 0.0 / 0.0 / 0.0 | real wav file; kaldi-style fbank, **no torchaudio needed** | **3** (was 4 → quality gate) | repo `preprocessor_config.json` (1024×128) contradicts its `config.json` (24×16) — FE built from model config |
| `hf-internal-testing/tiny-random-WhisperForConditionalGeneration` | whisper | audio (enc-dec) | 953,281 | 1 s `.wav` → `WhisperProcessor` → mel, trimmed to 60 frames | `encoder_last_hidden_state` | encoder_decoder / supported | 11.30 s | ✓, 0.776 s | **0.0** | 0.0 / 0.0 / 0.0 | real wav file through the real processor | **3** (input pipeline was 1) | mel trim: repo processor emits 3000 frames, tiny model needs 60 (`max_source_positions=30`, `modeling_whisper.py:616`); decoder prompt fix (bug 5) |
| `katuni4ka/tiny-random-llava-next` | llava_next | text+vision (VLM) | 676,891 | image + `<image>` prompt → `LlavaNextProcessor` | `last_hidden_state` over interleaved tokens (masked mean) | multimodal / **experimental** | 10.23 s | ✓, 0.056 s | **0.0** | base **5.66e-7** · extract 0.0 · state_repr 0.0 | real image + real prompt through the processor (**with-image path**) | **2** (executed incl. image path) | repo `patch_size: null` filled from `vision_config.patch_size`; token/feature counts 1155==1155 probe-verified; `_hidden_width` fix (bug 2) |

Columns `Save/Load` and `Numerical Check` are `max_abs_error` values (contract
< 1e-5; **every cell 0.0 except llava-next base_forward 5.7e-7**).
`JevX Mode` = resolved adapter / `capability_report.jevx_conversion`.
`Forward` = invariant pass (choice sums/bounds/K-mask, score bounds, noul
range, confidence ∈ [0,1], B=2 batch) + wall time of the m=4 shared-state call.
`state_repr` = `JevXOutput.state_representation` vs `state_proj(manual pooled)`
(§71).

### Quality-metric pointers (not run here — do NOT read values off this table)

| metric group | bert / t5 / gpt2 (the 3 original E2E families) | all other rows |
|---|---|---|
| Accuracy / F1 / NLL / Brier / ECE | quality benchmarks DO exist, but only for `google/bert_uncased_L-2_H-128_A-2` on SST-2 (reports/DATASET_BENCHMARKS.md + `benchmarks/results/exp1…exp6*.json`) — **no per-family quality result for the checkpoints in this table**, so no level 4 is claimed | N/A here — functional validation only |
| Latency | partial: per-family load time and m=4 forward wall time are in this report + `_jevx_matrix_results.json`; full latency sweeps belong to the benchmark suite | same |
| Memory | N/A in this report (param counts given; RSS profiling belongs to the benchmark suite) | same |

---

## Golden reference fixtures (§94)

Six fixed models × fixed inputs × `torch.manual_seed(0)` × `model.eval()`,
captured by **running** `python vendor/reports/_gen_golden_fixtures.py`
(never hand-written) and re-derived by `tests/golden/test_golden_references.py`
with documented tolerance **1e-6**:

| fixture | model | guards |
|---|---|---|
| `tests/golden/fixtures/golden_tiny_random_bert.json` | `hf-internal-testing/tiny-random-bert` | probs (7 dp), predicted indices, `state_representation` sum/mean/std (7 dp), 14 config fields, adapter + representation_field identity |
| `tests/golden/fixtures/golden_tiny_random_t5.json` | `hf-internal-testing/tiny-random-t5` | same |
| `tests/golden/fixtures/golden_tiny_gpt2.json` | `sshleifer/tiny-gpt2` | same |
| `tests/golden/fixtures/golden_tiny_random_roberta.json` | `hf-internal-testing/tiny-random-RobertaModel` | same (added by quality-audit Q-016, 3 → 6 families) |
| `tests/golden/fixtures/golden_tiny_random_bart.json` | `hf-internal-testing/tiny-random-BartForConditionalGeneration` | same |
| `tests/golden/fixtures/golden_tiny_random_vit.json` | `hf-internal-testing/tiny-random-ViTModel` | same; `state_kind: "vision"` — the checked-in `input_image_224.png` through the model's own image processor |

The other 12 validated checkpoints stay matrix-gated (live reference equality on
every `test_matrix_*` run) — rationale in `docs/TESTING.md` §5 and the golden
module docstring.

Checked-in modality fixtures (deterministic bytes, regenerated only if
missing): `tests/golden/fixtures/input_image_224.png` (seeded 224×224 PIL
pattern), `tests/golden/fixtures/input_audio_1s.wav` (1 s / 16 kHz two-tone,
written with soundfile, read back with soundfile).

---

## Bugs found by real-model execution and fixed (never weakened a test)

1. `jevx/capabilities/detector.py` — `representation_field` picked
   `sorted(...)[0]` → reported `image_embeds` while extraction
   (`adapters/base.py` `SEMANTIC_FIELDS`) selects `text_embeds` for CLIP/SigLIP.
   **Fix**: same sentence→text→image priority as extraction.
2. `jevx/modeling_jevx.py` `_hidden_width` — composite configs
   (clip/siglip/llava_next) have no top-level hidden key →
   `ValueError: cannot determine hidden width of CLIPModel/...` at wrap time.
   **Fix**: fall back to `projection_dim`, then nested `text_config` /
   `vision_config` (pooled-vector width per family).
3. `jevx/modeling_jevx.py` `__init__` device/dtype alignment loop —
   `NotImplementedError: Cannot copy out of meta tensor` when reloading
   wav2vec2 (its backbone escapes transformers' meta construction context, so
   `reference` is CPU while JevX modules are meta; confidence-engine buffers
   have no parameters, so a params-only guard would not catch them either).
   **Fix**: skip alignment when reference/module holds any meta tensor
   (params **or** buffers) — transformers materializes them afterwards.
4. `jevx/modeling_jevx.py` `_resolve_featurizers` — wrap left
   `wrapper.tokenizer = None` for processor-based families while reload
   attached `AutoTokenizer(save_dir)`; hashed question encoding is
   tokenizer-sensitive → **save/load decision drift at identical weights**:
   CLIP 6.785e-02, wav2vec2 1.739e-03 (measured before the fix).
   **Fix**: expose the processor's embedded tokenizer at wrap time (symmetry
   with the reload path). Post-fix: both 0.0.
5. `jevx/adapters/encoder_decoder.py` `prepare_inputs` — decoder prompt was
   mirrored only from `input_ids` (t5/bart); Whisper's encoder input is a mel
   spectrogram → `WhisperDecoder.forward` raised `ValueError: You cannot
   specify both decoder_input_ids and decoder_inputs_embeds`.
   **Fix**: for encoder modalities without a shared vocabulary, prompt the
   decoder with `config.decoder_start_token_id` (what the
   `ForConditionalGeneration` heads build internally).

Checkpoint-repo defects handled **in the test harness** (documented, not JevX
bugs): AST FE geometry from model config; whisper mel trimmed to the
checkpoint's required length; SigLIP `attention_mask` built as
`input_ids != pad`; LLaVA `patch_size: null` filled from
`vision_config.patch_size` (strategy left at the repo's `None`, which the
probe showed is the count-consistent combination).

## Environment

- Python 3.14.3 · torch 2.14.0+cpu (NO GPU) · transformers 5.16.1 ·
  Windows 11 (Intel64 Family 6 Model 78) · ~8 GB RAM · datasets 4.4.1 ·
  PIL + torchvision present.
- Installed this phase (whitelisted, verified harmless afterwards):
  **soundfile 0.14.0**, **sentencepiece 0.2.2**
  (`python -c "import torch, transformers, soundfile, sentencepiece"` OK
  after install; SigLIP processor loads). `librosa`/`torchaudio` still absent
  and **not needed** (numpy fbank/mel paths).
- Environment-blocked items: **none remaining**. Initially
  `hf-internal-testing/tiny-random-SiglipModel` input prep was blocked with
  `ImportError: SiglipTokenizer requires the SentencePiece library but it was
  not found in your environment` — resolved by the whitelisted install.
  `librosa`-dependent paths were never required. The only suite skip is
  pre-existing `tests/unit/test_failure_modes.py::test_cross_device_input_raises_device_error`
  (CPU-only host: no second torch device).

## Appendix A — reproduce (exact commands, repo root)

```bash
# 1. environment + checkpoint existence/sha probe (configs/tokenizers only)
python vendor/reports/_probe_matrix_env.py        # -> vendor/reports/_matrix_env_probe.json

# 2. REAL validation matrix (network on first run; results JSON via atexit)
python -m pytest tests/integration/test_matrix_text.py -q -s          # 11 text families
python -m pytest tests/integration/test_matrix_vision_audio.py -q -s  # 7 vision/audio/multimodal families

# 3. golden reference fixtures (capture; run ONLY to deliberately regenerate)
python vendor/reports/_gen_golden_fixtures.py      # -> tests/golden/fixtures/golden_*.json
python -m pytest tests/golden -q                   # re-derive + compare (tol 1e-6)

# 4. full suite
python -m pytest -q                                # pytest.ini scopes to tests/

# 5. modality fixture assets (idempotent)
python vendor/reports/_gen_fixture_assets.py
```

Debug/patch scripts referenced above (all under `vendor/reports/`):
`_probe_matrix_env.py`, `_patch_modeling_fixes.py`, `_patch_import_and_diag.py`,
`_patch_meta_guard_buffers.py`, `_patch_featurizer_symmetry.py`,
`_patch_encdec_decoder_prompt.py`, `_patch_matrix_levels.py`,
`_probe_llava_tokens.py`, `_debug_roundtrip_diff.py`, `_debug_reload_meta.py`,
`_gen_golden_fixtures.py`, `_gen_fixture_assets.py`.

## Appendix B — checkpoint id → resolved commit sha (huggingface_hub revision)

| model id | sha |
|---|---|
| `hf-internal-testing/tiny-random-bert` | `f171d7baecaf37b5da5a3616d8833b9969753535` |
| `hf-internal-testing/tiny-random-RobertaModel` | `5f1d755e93e9fe46b9bb90404f1e810515b2410a` |
| `hf-internal-testing/tiny-random-DistilBertModel` | `cc4425ad0676f3ec00e8bffe485fe83cae61041a` |
| `hf-internal-testing/tiny-random-AlbertModel` | `34a63314686b64aaeb595ddb95006f1ff2ffda17` |
| `hf-internal-testing/tiny-random-DebertaV2Model` | `9dab549e7d9f719c1438cdd3928d59545c72b0d0` |
| `sentence-transformers/all-MiniLM-L6-v2` | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| `hf-internal-testing/tiny-random-t5` | `2f582cd79ed5795b71539951d237945bc1c5ac7e` |
| `hf-internal-testing/tiny-random-BartForConditionalGeneration` | `45bbb767e8601574fe69ce80e82ed295716c9076` |
| `sshleifer/tiny-gpt2` | `5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be` |
| `yujiepan/qwen2-tiny-random` | `b01b9c82aaf1efb4c26c94b6342d611b397245ff` |
| `hf-internal-testing/tiny-random-LlamaForCausalLM` | `9fb191250dd56d0ba7ec9785a025ed29c03d5998` |
| `hf-internal-testing/tiny-random-ViTModel` | `31817b7a64ebc3333fcd4801dfbb356ab07b13dd` |
| `hf-internal-testing/tiny-random-CLIPModel` | `703b075aa101387f504f075dca2e0c78c99ee420` |
| `hf-internal-testing/tiny-random-SiglipModel` | `5f4a673ccb0e7dc620d13a946dd99f95dba45653` |
| `hf-internal-testing/tiny-random-Wav2Vec2Model` | `7a998ee3ee0619a52828a79c3eed6872fd053f37` |
| `hf-internal-testing/tiny-random-ASTModel` | `75e68f956f6f2c0709b01e596e7a6aecb1b29dce` |
| `hf-internal-testing/tiny-random-WhisperForConditionalGeneration` | `598101b885b24508042d9292e54aa04bff96318e` |
| `katuni4ka/tiny-random-llava-next` | `7ca7d02e9a5da08ddd26b306b9482632578dc471` |

All 18 ids returned HTTP 200 from `GET https://huggingface.co/api/models/<id>`
before execution (`vendor/reports/_matrix_env_probe.json`).

## Appendix C — support-level changes applied to the matrix

| cell | old → new | justification |
|---|---|---|
| bert, roberta, albert, vit, ast(extraction), clip(extraction), siglip(extraction) | **4 → 3** | Executed fully (former extraction evidence + real input), but level 4 now requires a per-family real-dataset quality result on the cited checkpoint; the committed quality results cover only `google/bert_uncased_L-2_H-128_A-2` × SST-2 (reports/DATASET_BENCHMARKS.md), not these checkpoints. |
| wav2vec2 input pipeline, whisper input pipeline | **1 → 3** | Real `.wav` fixtures executed through the real processors. |
| distilbert, deberta-v2, t5, bart | **3 → 3** (evidence added) | Previously probe-verified; now executed with all errors 0.0 (cell notes updated). |
| llama, qwen2, gpt2 | **2 → 2** (executed, noted) | Execution does not remove the lossy-pooling semantics that define level 2. |
| llava_next | **2 → 2** (with-image path executed, noted) | Interleaved-token pooling remains a pinned-but-lossy convention (`jevx_conversion=experimental`). |
| mistral, gemma, phi3 | **2 → 2** (not re-executed, noted) | Phase-1 probe evidence stands; not part of the Phase-92 execution set — recorded for transparency. |
| level 1 / level 0 | — → none | All former input-pipeline-1 sub-items were executed. |
