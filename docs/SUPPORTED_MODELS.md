# Supported Models — the 18 executed checkpoints

Sources: [vendor/reports/_jevx_matrix_results.json](../vendor/reports/_jevx_matrix_results.json)
(machine-readable, format `jevx-matrix-results-v1`, regenerated on every test
run) and [reports/REAL_MODEL_VALIDATION.md](../reports/REAL_MODEL_VALIDATION.md)
(human-readable protocol + Appendix B revision SHAs). Support *levels* per
family come from the live source
[TRANSFORMERS_COMPATIBILITY_MATRIX.md](../TRANSFORMERS_COMPATIBILITY_MATRIX.md);
this page summarizes both and never overrides them.

## 1. Inventory ≠ support (§79)

The repository inventories **709 architecture records**
(`vendor/reports/architecture_inventory.json`) and **510 `model_type` entries**
in `vendor/reports/auto_model_mapping.json`. Those are *facts about the mapping
and the source tree* — they are **not** support claims. Support is
**per-family, per-level, evidence-based**: a level is only claimed with the
evidence the matrix defines for it (§7 of docs/LIMITATIONS.md; level table in
the matrix). The universality statement itself is scoped verbatim in
[docs/LIMITATIONS.md](LIMITATIONS.md) §2: *"HF Transformers-compatible models
with a meaningful differentiable representation."*

## 2. The 18 executed checkpoints (§92–§94 protocol)

Every row ran: load → real-modality input → decision invariants
(`.choice/`.score/`.noul`) → `save_pretrained → from_pretrained` decision
equality → `base_forward` vs a freshly loaded raw `AutoModel` → shared-state
single forward (m=4) → extraction correctness vs an independent manual
reference. Gate values: **save/load max_abs_error 0.0** and **base_forward
0.0** for all rows except llava-next base_forward **5.66e-07**;
**backbone_forward_calls = 1** at m=4 for every row (JSON). All values are
measured on this host (CPU-only; environment in
[docs/REPRODUCIBILITY.md](REPRODUCIBILITY.md)).

| # | model id | family | adapter | level | representation (pinned route) | notes |
|---|---|---|---|---|---|---|
| 1 | `hf-internal-testing/tiny-random-bert` | bert | encoder | **3** | `pooler_output` | native pooler; also E2E + golden fixture |
| 2 | `hf-internal-testing/tiny-random-RobertaModel` | roberta | encoder | **3** | `pooler_output` | native pooler |
| 3 | `hf-internal-testing/tiny-random-DistilBertModel` | distilbert | encoder | **3** | `last_hidden_state` (CLS) | no pooler → CLS convention |
| 4 | `hf-internal-testing/tiny-random-AlbertModel` | albert | encoder | **3** | `pooler_output` | sentencepiece fast-tokenizer path |
| 5 | `hf-internal-testing/tiny-random-DebertaV2Model` | deberta-v2 | encoder | **3** | `last_hidden_state` (CLS) | base model has no pooler |
| 6 | `sentence-transformers/all-MiniLM-L6-v2` | bert (semantic) | encoder | **3** | `last_hidden_state` pooled `masked_mean` (pinned) | **real trained embedding checkpoint**; ST pooling convention pinned; functionally validated, NOT quality-validated |
| 7 | `hf-internal-testing/tiny-random-t5` | t5 | encoder_decoder | **3** | `encoder_last_hidden_state` | also E2E + golden fixture |
| 8 | `hf-internal-testing/tiny-random-BartForConditionalGeneration` | bart | encoder_decoder | **3** | `encoder_last_hidden_state` | |
| 9 | `sshleifer/tiny-gpt2` | gpt2 | decoder | **2** | `last_hidden_state` (`last_valid`) | lossy-pooling semantics keep level 2; E2E + golden |
| 10 | `yujiepan/qwen2-tiny-random` | qwen2 | decoder | **2** | `last_hidden_state` (`last_valid`) | verified alternate (hf-internal-testing Qwen2 id 401s) |
| 11 | `hf-internal-testing/tiny-random-LlamaForCausalLM` | llama | decoder | **2** | `last_hidden_state` (`last_valid`) | lossy-pooling semantics |
| 12 | `hf-internal-testing/tiny-random-ViTModel` | vit | vision | **3** | `pooler_output` | 224×224 PNG fixture through `ViTImageProcessor` |
| 13 | `hf-internal-testing/tiny-random-CLIPModel` | clip | multimodal | **3** | `text_embeds` (post-projection) | both towers in one forward |
| 14 | `hf-internal-testing/tiny-random-SiglipModel` | siglip | multimodal | **3** | `text_embeds` (post-projection) | needs `sentencepiece` (installed, recorded) |
| 15 | `hf-internal-testing/tiny-random-Wav2Vec2Model` | wav2vec2 | audio | **3** | `last_hidden_state` (masked mean) | real 1 s `.wav` → `Wav2Vec2Processor` |
| 16 | `hf-internal-testing/tiny-random-ASTModel` | ast | audio | **3** | `pooler_output` | numpy fbank (no torchaudio); FE built from model config |
| 17 | `hf-internal-testing/tiny-random-WhisperForConditionalGeneration` | whisper | encoder_decoder | **3** | `encoder_last_hidden_state` | mel trimmed to the checkpoint's required length |
| 18 | `katuni4ka/tiny-random-llava-next` | llava_next | multimodal | **2** | `last_hidden_state` over interleaved tokens (masked mean) | with-image path executed; `jevx_conversion=experimental`; base_forward 5.66e-07 |

**Honesty note (quoted scope, reports/REAL_MODEL_VALIDATION.md):** except
`sentence-transformers/all-MiniLM-L6-v2`, every checkpoint carries **random
weights** (`tiny-random-*`, `sshleifer/tiny-gpt2`) — these rows establish that
the *pipeline* is correct end-to-end (functional validation), **not** predictive
quality. Quality numbers exist only for the SST-2 benchmark backbone
(`google/bert_uncased_L-2_H-128_A-2`, frozen Mode A) — see
[docs/BENCHMARKS.md](BENCHMARKS.md).

## 3. Level distribution (matrix summary)

* **Level 3 (13 families):** bert, roberta, albert, distilbert, deberta-v2,
  t5, bart, wav2vec2, whisper, ast, vit, clip, siglip — executed with every
  numerical gate at 0.0 (llava-next's 5.66e-07 aside, which is level 2 anyway).
* **Level 2 (7 families):** llama, qwen2, gpt2 (executed; lossy-pooling
  semantics unchanged by execution), mistral, gemma, phi3 (Phase-1 probe
  evidence only — not re-executed in Phase 92, noted in the matrix), llava_next
  (executed including the with-image path; interleaved pooling is
  `jevx_conversion=experimental`).
* **Levels 1/0:** none among audited families.
* Former level-4 cells were **demoted to 3** when executed: level 4 now
  additionally requires a real-dataset quality result (evidence overlay in the
  matrix header; reports/REAL_MODEL_VALIDATION.md Appendix C).

## 4. Where to look next

* Per-family evidence, checkpoint examples, output classes, adapter column:
  [TRANSFORMERS_COMPATIBILITY_MATRIX.md](../TRANSFORMERS_COMPATIBILITY_MATRIX.md).
* Protocol + per-id revision SHAs: reports/REAL_MODEL_VALIDATION.md
  (Appendix B; also in [docs/REPRODUCIBILITY.md](REPRODUCIBILITY.md) §2).
* Machine-readable gates: `vendor/reports/_jevx_matrix_results.json`
  (`save_load_max_abs_error`, `base_forward_max_abs_error`,
  `backbone_forward_calls_m4`, per-family `notes`).
* How a checkpoint becomes a decision model:
  [docs/UNIVERSAL_CONVERSION.md](UNIVERSAL_CONVERSION.md).
