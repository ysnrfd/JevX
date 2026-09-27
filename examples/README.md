# JevX examples — 7 executable scripts (§46)

Each file is a real, runnable script — **running one is testing it**: every
example asserts its key invariants (probability simplex, schema membership,
confidence range, batch layouts, backbone-forward counts, determinism) and
exits non-zero if any fails. All were executed on this host (2026-09-24,
CPU-only; see [../docs/REPRODUCIBILITY.md](../docs/REPRODUCIBILITY.md) for the
environment).

## How to run

From the repository root (no packaging file exists yet — each script inserts
the repo root into `sys.path` itself, so absolute paths work too):

```bash
python examples/example_01_encoder_bert.py
python examples/example_02_decoder_llm.py
# …
```

First run downloads the checkpoint (network required); afterwards loads are
cached. Measured wall time on this host: **~25–40 s per example** (e.g.
32.2 s for 01, 34.4 s for 03) — dominated by interpreter startup,
`torch/transformers` import and model load, not by the decision call.

## The examples

| File | Shows | Checkpoint | Measured status |
|---|---|---|---|
| `example_01_encoder_bert.py` | encoder family; `choice` (variable K), `score` (expected score), `noul` in separate typed calls over one state | `hf-internal-testing/tiny-random-bert` | ran green, all invariants held |
| `example_02_decoder_llm.py` | decoder family, `last_valid` pooling; mixed decision types in ONE `decide` call; asserts **backbone_forward_calls == 1 for m=4** via a forward hook | `sshleifer/tiny-gpt2` | ran green (after an example-side invariant fix: the noul row is a sigmoid, not a simplex row) |
| `example_03_encoder_decoder_t5.py` | encoder-decoder encoder-side extraction; B=2 states × m=2 questions → 4 decisions, verifies the state-major `b*m + j` layout against `schema_metadata` | `hf-internal-testing/tiny-random-t5` | ran green |
| `example_04_semantic_embedding.py` | the ONE real trained checkpoint; sentence-transformers `masked_mean` convention pinned in config; inspects `state_representation` [B, 384]; asserts within-process bitwise eval determinism | `sentence-transformers/all-MiniLM-L6-v2` | ran green (after an example-side fix: the detector's preferred native field for this id is `pooler_output`; the pinned strategy still extracts via `last_hidden_state` — both asserted) |
| `example_05_vision_vit.py` | vision family: raw PIL image payload → `ViTImageProcessor` (`processor_required`), deterministic self-contained 224×224 fixture | `hf-internal-testing/tiny-random-ViTModel` | ran green |
| `example_06_audio_wav2vec2.py` | audio family: synthesize 1 s/16 kHz two-tone WAV with numpy + soundfile, read the real file back, route through `Wav2Vec2Processor`; plus a `score` head bounded by [0, K−1] | `hf-internal-testing/tiny-random-Wav2Vec2Model` | ran green |
| `example_07_multimodal_clip.py` | multimodal: image + text in ONE joint forward, post-projection `text_embeds` convention, prints `capability_metadata` (incl. `backbone_forward_calls=1`), plus a `noul` check in (0,1) | `hf-internal-testing/tiny-random-CLIPModel` | ran green |

## Honesty note

Except `sentence-transformers/all-MiniLM-L6-v2` (example 04), every
checkpoint carries **random weights** — pipeline fixtures, not trained models
([../reports/REAL_MODEL_VALIDATION.md](../reports/REAL_MODEL_VALIDATION.md)).
The printed probabilities are real pipeline outputs with their invariants
asserted; they are **not** predictive quality. Quality measurements live in
[../reports/DATASET_BENCHMARKS.md](../reports/DATASET_BENCHMARKS.md).
Examples re-run in a fresh process re-randomize the JevX-side layers, so
*cross-process* outputs differ by design — the contract asserted in example 04
is *within-process* bitwise determinism (docs/LIMITATIONS.md §6,
reports/FAILURE_ANALYSIS.md §74).
