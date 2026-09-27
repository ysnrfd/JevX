# Copyright (c) 2026 ysnrfd
"""Update TRANSFORMERS_COMPATIBILITY_MATRIX.md support-level cells with the
Phase 92-94 execution evidence (reports/REAL_MODEL_VALIDATION.md).

Rules applied:
* level 4 requires a real-dataset quality result (benchmark suite owns it;
  none committed) -> former probe-only 4s demoted to 3 once EXECUTED;
* level 3 requires executed load+forward+save/load+invariants+real input;
* execution never RAISES a level whose semantics stay convention-dependent
  (decoder-only / llava stay 2), it is recorded in the cell note;
* input-pipeline sub-items previously at 1 are raised to executed.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "TRANSFORMERS_COMPATIBILITY_MATRIX.md"

REPLACEMENTS: list[tuple[str, str]] = [
    # --- overlay note after the level definitions table ---------------------
    (
        "**Checkpoint verification**: `✓` = `GET https://huggingface.co/api/models/<id>` returned 200 in this audit (53 IDs; raw map `vendor/reports/_hf_verified_ids.json`). `✗` = request returned HTTP 401 / not verified — do **not** assume existence.",
        "**Checkpoint verification**: `✓` = `GET https://huggingface.co/api/models/<id>` returned 200 in this audit (53 IDs; raw map `vendor/reports/_hf_verified_ids.json`). `✗` = request returned HTTP 401 / not verified — do **not** assume existence.\n\n**Evidence overlay (Phases 92-94)**: a family only holds **3 (Validated)** if load → forward → save/load equality → output invariants → **REAL-data input** were *executed on the actual checkpoint* (`reports/REAL_MODEL_VALIDATION.md`; machine-readable `vendor/reports/_jevx_matrix_results.json`; tests `tests/integration/test_matrix_*.py`, golden `tests/golden/`). **4 additionally requires a real-dataset quality result** (Accuracy/F1/NLL/Brier/ECE — owned by the benchmark suite; no results committed at time of writing), so cells previously at 4 on probe evidence alone were **demoted to 3 when executed**. Cells that could not be executed are noted in place (failure transparency §77).",
    ),
    # --- encoder-only rows --------------------------------------------------
    (
        "| pooler | native | **4** | Trained pooler ships with base model (`add_pooling_layer=True` default); probe-verified; all checkpoints 200. |",
        "| pooler | native | **3** | Trained pooler ships with base model (`add_pooling_layer=True` default); probe-verified; all checkpoints 200. **Executed Phase 92** on `hf-internal-testing/tiny-random-bert` (real sentences; save/load, base_forward, §71 extraction all 0.0) — demoted 4 → 3: level-4 quality gate pending the benchmark suite. |",
    ),
    (
        "| pooler | native | **4** | Identical contract to bert. |",
        "| pooler | native | **3** | Identical contract to bert. **Executed Phase 92** on `hf-internal-testing/tiny-random-RobertaModel` (all errors 0.0) — demoted 4 → 3 on the level-4 quality gate. |",
    ),
    (
        "| pooler | native | **4** | Trained pooler shipped (`add_pooling_layer` default True). |",
        "| pooler | native | **3** | Trained pooler shipped (`add_pooling_layer` default True). **Executed Phase 92** on `hf-internal-testing/tiny-random-AlbertModel` (all errors 0.0) — demoted 4 → 3 on the level-4 quality gate. |",
    ),
    (
        "| last_hidden_state → CLS/mean | native | **3** | Standard pooling over bidirectional states; note legacy-style manual `output_hidden_states` handling in this model's forward (not the hook machinery) — both verified working. |",
        "| last_hidden_state → CLS/mean | native | **3** | Standard pooling over bidirectional states; note legacy-style manual `output_hidden_states` handling in this model's forward (not the hook machinery) — both verified working. **Executed Phase 92** on `hf-internal-testing/tiny-random-DebertaV2Model` (all errors 0.0). |",
    ),
    (
        "| last_hidden_state → CLS/mean | native | **3** | CLS/mean over bidirectional states is the established convention; no ambiguity. |",
        "| last_hidden_state → CLS/mean | native | **3** | CLS/mean over bidirectional states is the established convention; no ambiguity. **Executed Phase 92** on `hf-internal-testing/tiny-random-DistilBertModel` (all errors 0.0). |",
    ),
    # --- semantic note ------------------------------------------------------
    (
        "**Semantic note**: for sentence-embedding semantics, `sentence-transformers/all-MiniLM-L6-v2` ✓ (model_type `bert`, architectures `[BertModel]` — both read from its config.json via HF API) is a bert-architecture checkpoint whose *fine-tuned* space is preferable to raw bert pooler; JevX can load it through the same `AutoModel`/`BertModel` route — no special casing.",
        "**Semantic note**: for sentence-embedding semantics, `sentence-transformers/all-MiniLM-L6-v2` ✓ (model_type `bert`, architectures `[BertModel]` — both read from its config.json via HF API) is a bert-architecture checkpoint whose *fine-tuned* space is preferable to raw bert pooler; JevX can load it through the same `AutoModel`/`BertModel` route — no special casing. **Executed Phase 92** with `pooling_method=\"masked_mean\"` (the sentence-transformers pooling convention): real sentences, save/load & base_forward & §71 errors 0.0 — **FUNCTIONALLY validated on a real trained embedding checkpoint; NOT quality-validated** (no similarity-task metrics run; benchmark suite owns those).",
    ),
    # --- decoder-only rows --------------------------------------------------
    (
        "| last_hidden_state → mean/last-token (mask-aware) | lossy | **2** | Extraction mechanics standard + probe-verified, but no trained sentence pooler exists; pooling choice (mean vs last non-pad token) and layer choice change semantics — JevX must pin the convention. `tie_word_embeddings` defaults **False** in v5 (`configuration_llama.py:79`). |",
        "| last_hidden_state → mean/last-token (mask-aware) | lossy | **2** | Extraction mechanics standard + probe-verified, but no trained sentence pooler exists; pooling choice (mean vs last non-pad token) and layer choice change semantics — JevX must pin the convention. `tie_word_embeddings` defaults **False** in v5 (`configuration_llama.py:79`). **Executed Phase 92** on `hf-internal-testing/tiny-random-LlamaForCausalLM` (all errors 0.0) — level stays 2: execution does not change the lossy-pooling semantics. |",
    ),
    (
        "| last_hidden_state → mean/last-token | lossy | **2** | Same causal-LM pooling caveat. |",
        "| last_hidden_state → mean/last-token | lossy | **2** | Same causal-LM pooling caveat. **Executed Phase 92** on `yujiepan/qwen2-tiny-random` (verified alternate; all errors 0.0) — level stays 2 (lossy pooling semantics). |",
    ),
    (
        "| last_hidden_state → mean/last-token | lossy | **2** | Same caveat; GPT2 uses `transformer` as `base_model_prefix` (`modeling_gpt2.py:415`). |",
        "| last_hidden_state → mean/last-token | lossy | **2** | Same caveat; GPT2 uses `transformer` as `base_model_prefix` (`modeling_gpt2.py:415`). **Executed Phase 92** on `sshleifer/tiny-gpt2` (all errors 0.0; also E2E in `tests/integration/test_jevx_e2e_tiny.py`) — level stays 2 (lossy pooling semantics). |",
    ),
    # --- encoder-decoder rows ----------------------------------------------
    (
        "| encoder | native | **3** | `encoder_last_hidden_state` is a named field; direct `.encoder` call verified equivalent in eval mode; masked-mean pooling over it is standard (sentence-T5 convention). Pooling still JevX-side → not level 4 (no trained pooler). |",
        "| encoder | native | **3** | `encoder_last_hidden_state` is a named field; direct `.encoder` call verified equivalent in eval mode; masked-mean pooling over it is standard (sentence-T5 convention). Pooling still JevX-side → not level 4 (no trained pooler). **Executed Phase 92** on `hf-internal-testing/tiny-random-t5` (all errors 0.0; also E2E in `tests/integration/test_jevx_e2e_tiny.py`). |",
    ),
    (
        "| encoder | native | **3** | Same as t5. Note tied `shared` embedding (`modeling_bart.py:683-687`). |",
        "| encoder | native | **3** | Same as t5. Note tied `shared` embedding (`modeling_bart.py:683-687`). **Executed Phase 92** on `hf-internal-testing/tiny-random-BartForConditionalGeneration` (all errors 0.0). |",
    ),
    # --- audio rows ---------------------------------------------------------
    (
        "| last_hidden_state → masked mean over time | native | **3** (extraction) / input pipeline **1** until array-input path executed with a real file | Forward contract fully verified; pooling convention documented in-family; audio I/O deps unverified here. `main_input_name=\"input_values\"` (`modeling_wav2vec2.py:901`). |",
        "| last_hidden_state → masked mean over time | native | **3** (extraction + input pipeline **EXECUTED Phase 92**: real 1 s `.wav` fixture read with soundfile → `Wav2Vec2Processor` → `input_values`) | Forward contract fully verified; pooling convention documented in-family; save/load & base_forward & §71 errors 0.0 on `hf-internal-testing/tiny-random-Wav2Vec2Model`. `main_input_name=\"input_values\"` (`modeling_wav2vec2.py:901`). |",
    ),
    (
        "| encoder → masked mean over time | native | **3** (extraction) / input pipeline **1** (same deps caveat) | Encoder extraction mirrors what `WhisperForConditionalGeneration` does internally (`modeling_whisper.py:961-963` wraps encoder outputs identically). `main_input_name=\"input_features\"` (`:514`). |",
        "| encoder → masked mean over time | native | **3** (extraction + input pipeline **EXECUTED Phase 92**: real `.wav` → `WhisperProcessor`, mel trimmed to the checkpoint's required length — repo processor emits 3000 frames but the tiny model's `max_source_positions=30` demands 60, `modeling_whisper.py:616`) | Encoder extraction mirrors what `WhisperForConditionalGeneration` does internally (`modeling_whisper.py:961-963` wraps encoder outputs identically). Save/load & base_forward errors 0.0 on `hf-internal-testing/tiny-random-WhisperForConditionalGeneration`; full-forward decoder prompt via `decoder_start_token_id` (JevX fix, Phase 92). `main_input_name=\"input_features\"` (`:514`). |",
    ),
    (
        "| pooler | native | **4** (extraction) / input pipeline **1** (deps caveat) | In-model pooling field; contract probe-verified. Same missing-audio-deps caveat for processor input. `main_input_name=\"input_values\"` (`:231`). |",
        "| pooler | native | **3** (extraction + input **EXECUTED Phase 92**: real `.wav` → numpy fbank, no torchaudio needed; FE geometry rebuilt from the model config because the repo's `preprocessor_config.json` (1024×128) contradicts its own `config.json` (24×16)) | In-model pooling field; contract probe-verified. Was 4 → demoted on the level-4 quality gate after execution (save/load & base_forward errors 0.0 on `hf-internal-testing/tiny-random-ASTModel`). `main_input_name=\"input_values\"` (`:231`). |",
    ),
    # --- vision row ---------------------------------------------------------
    (
        "| pooler | native | **4** | Trained/identity-init pooler shipped; note ViT uses `@capture_outputs(tie_last_hidden_states=False)` (`modeling_vit.py:354`) so `hidden_states[-1]` is the **pre-final-norm** state — JevX should use `last_hidden_state` for the pooled value. `main_input_name=\"pixel_values\"` (`:308`). |",
        "| pooler | native | **3** | Trained/identity-init pooler shipped. **Executed Phase 92** on `hf-internal-testing/tiny-random-ViTModel` with a real PIL fixture through `ViTImageProcessor` (save/load & base_forward & §71 errors 0.0) — demoted 4 → 3 on the level-4 quality gate. Note ViT uses `@capture_outputs(tie_last_hidden_states=False)` (`modeling_vit.py:354`) so `hidden_states[-1]` is the **pre-final-norm** state — JevX uses `last_hidden_state`/`pooler_output` for the pooled value. `main_input_name=\"pixel_values\"` (`:308`). |",
    ),
    # --- contrastive rows ---------------------------------------------------
    (
        "| pooler (+projection) | native | **4** for extraction (both fields probe-verified) — pre- vs post-projection choice documented but both are native fields, so semantics pinned by picking one (recommend post-projection `get_*_features` for cross-modal comparability) | Input prep contract read from `CLIPProcessor` source (`processing_clip.py:23`), not executed end-to-end (no checkpoint download attempted) → end-to-end input prep sub-item **1**. |",
        "| pooler (+projection) | native | **3** (**EXECUTED Phase 92** on `hf-internal-testing/tiny-random-CLIPModel`: both towers in one forward, multimodal adapter, post-projection `text_embeds` extracted, save/load & base_forward & §71 errors 0.0; was 4-extraction → demoted on the level-4 quality gate) | Input prep **executed Phase 92** (real 224×224 PIL fixture + real text through `CLIPProcessor`). Pre- vs post-projection semantics pinned by extracting `text_embeds` (post-projection, cross-modal comparability). |",
    ),
    (
        "| pooler (+projection) | native | **4** for extraction (probe-verified) — the `padding=\"max_length\"` requirement is a hard input-prep convention; violating it silently shifts the \"last token\" pooler | Same end-to-end input-prep caveat as CLIP (**1** for unexecuted processor path). SigLIP text is bidirectional (no causal mask, `:520-525`) unlike CLIP. |",
        "| pooler (+projection) | native | **3** (**EXECUTED Phase 92** on `hf-internal-testing/tiny-random-SiglipModel`: post-projection `text_embeds`, save/load & base_forward & §71 errors 0.0; was 4-extraction → demoted on the level-4 quality gate) — the `padding=\"max_length\"` requirement is a hard input-prep convention; violating it silently shifts the \"last token\" pooler (asserted in the Phase-92 harness) | Input prep **executed Phase 92** (required `pip install sentencepiece` 0.2.2 for `SiglipTokenizer`; repo processor output carries no `attention_mask`, built as `input_ids != pad`). SigLIP text is bidirectional (no causal mask, `:520-525`) unlike CLIP. |",
    ),
    # --- multimodal row -----------------------------------------------------
    (
        "| last_hidden_state over **text-token positions only** (must exclude scattered image spans) | lossy | **2** | Representation semantics require selecting text-token positions and pooling across a sequence whose length/layout depends on image resolution and `image_grid_pinpoints` — a per-checkpoint convention JevX must pin. Text-only contract verified; full image path source-verified but unexecuted. Composite config: nested `text_config`/`vision_config` (`configuration_llava_next.py:55-58`), children built via `AutoModel.from_config` (`modeling_llava_next.py:253,260`). |",
        "| all mask-valid positions incl. image spans (JevX pins masked-mean over the interleaved sequence; `jevx_conversion=experimental`) | lossy | **2** | Representation semantics require selecting token positions across a sequence whose length/layout depends on image resolution and `image_grid_pinpoints` — a per-checkpoint convention JevX pins (masked-mean over all mask-valid positions, `MultimodalAdapter`). **WITH-IMAGE path EXECUTED Phase 92** on `katuni4ka/tiny-random-llava-next`: real image + prompt through `LlavaNextProcessor` (repo `patch_size: null` filled from `vision_config.patch_size`; token/feature counts probe-verified 1155==1155), save/load 0.0, base_forward 5.7e-7, shared-state forward count 1 — level stays 2: execution does not make interleaved pooling canonical. Composite config: nested `text_config`/`vision_config` (`configuration_llava_next.py:55-58`), children built via `AutoModel.from_config` (`modeling_llava_next.py:253,260`). |",
    ),
    # --- cross-cutting summary ---------------------------------------------
    (
        "- **Env gaps on this machine**: `soundfile`, `librosa`, `torchaudio`, `sentencepiece` are **not installed** — audio file→array preprocessing and slow sentencepiece tokenizers were not exercised; `PIL` and `torchvision` **are** available (vision input prep unblocked).\n- **Level distribution**: 4 → bert, roberta, albert, vit, ast(extraction), clip(extraction), siglip(extraction); 3 → distilbert, deberta-v2, t5, bart, wav2vec2(extraction), whisper(extraction); 2 → llama, qwen2, mistral, gemma, gpt2, phi3, llava_next; 1 → audio/vision *input pipelines* sub-items only (deps/end-to-end caveats above); 0 → none.",
        "- **Env gaps at Phase-1 audit time**: `soundfile`, `librosa`, `torchaudio`, `sentencepiece` were not installed. **Phase 92 update**: `soundfile` 0.14.0 and `sentencepiece` 0.2.2 are now installed (whitelisted, torch/transformers verified intact); audio file→array and sentencepiece tokenizers are therefore exercised. `librosa`/`torchaudio` remain absent and were **not needed** (numpy fbank/mel paths used by AST/Whisper/Wav2Vec2). `PIL` and `torchvision` available.\n- **Level distribution (post Phase 92-94 execution; overlay: 3 = executed load+forward+save/load+invariants+real input on the cited checkpoint, 4 = 3 + real-dataset quality from the benchmark suite — none committed yet)**: **3** → bert, roberta, albert, distilbert, deberta-v2, t5, bart, wav2vec2, whisper, ast, vit, clip, siglip (all 13 executed; the former 4s demoted on the quality gate); **2** → llama, qwen2, gpt2 (executed, lossy-pooling semantics unchanged), mistral, gemma, phi3 (Phase-1 probe only — not re-executed Phase 92, noted for transparency), llava_next (executed incl. with-image path); **1/0** → none. Evidence: `reports/REAL_MODEL_VALIDATION.md` (18 checkpoints, 0 errors on every numerical gate), `vendor/reports/_jevx_matrix_results.json`, tests `tests/integration/test_matrix_{text,vision_audio}.py` + `tests/golden/` (3 golden fixtures).",
    ),
]


def main() -> int:
    raw = PATH.read_bytes()
    text = raw.decode("utf-8")
    crlf = "\r\n" in text
    failures = []
    applied = 0
    for old, new in REPLACEMENTS:
        o = old.replace("\n", "\r\n") if crlf else old
        n = new.replace("\n", "\r\n") if crlf else new
        if o in text:
            text = text.replace(o, n, 1)
            applied += 1
        else:
            failures.append(old[:90])
    PATH.write_bytes(text.encode("utf-8"))
    print(f"applied {applied}/{len(REPLACEMENTS)}")
    for f in failures:
        print(f"MISSING: {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
