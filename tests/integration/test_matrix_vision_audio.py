# Copyright (c) 2026 ysnrfd
"""REAL pretrained validation matrix — VISION / AUDIO / MULTIMODAL families
(Phases 20/46/47/48/70/92).

Real modality inputs, all from checked-in fixtures under
``tests/golden/fixtures/``:

* vision: ``input_image_224.png`` (deterministic 224×224 PIL pattern) through
  each model's own image processor;
* audio: ``input_audio_1s.wav`` (1 s / 16 kHz two-tone, written with
  ``soundfile``) READ BACK with ``soundfile`` → real ndarray → the family's
  processor/feature extractor (documented: real wav file, not a synthetic
  input_values tensor);
* multimodal: real image + real text through ``CLIPProcessor`` /
  ``LlavaNextProcessor``; SigLIP text padded to ``max_length`` per its model
  docstring (the adapter passes no processor kwargs, so tensors are prepared
  explicitly — see ``matrix_common.build_siglip_state``).

Families: vit (pooler), clip (both towers + multimodal adapter, post-
projection ``text_embeds``), siglip (padding=max_length convention),
wav2vec2 (masked mean over frames), ast (native pooler, numpy fbank — no
torchaudio), whisper (encoder–decoder over mel), llava-next (vision-language,
with-image path through the real processor).
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import pytest

from matrix_common import execute_matrix_case, load_wrapper

#: family -> (model_id, state kind, extra JevX.from_pretrained kwargs, notes)
VISION_AUDIO_FAMILIES: Dict[str, Tuple[str, str, Dict[str, Any], str]] = {
    "vit": (
        "hf-internal-testing/tiny-random-ViTModel",
        "vision",
        {},
        "vision; native ViT pooler; PIL fixture through ViTImageProcessor",
    ),
    "clip": (
        "hf-internal-testing/tiny-random-CLIPModel",
        "multimodal",
        {},
        "both towers in one forward; semantic field text_embeds (post-projection)",
    ),
    "siglip": (
        "hf-internal-testing/tiny-random-SiglipModel",
        "siglip",
        {},
        "text padded to max_length per SigLIP docstring; needs sentencepiece "
        "(installed this phase: 0.2.2) for SiglipTokenizer",
    ),
    "wav2vec2": (
        "hf-internal-testing/tiny-random-Wav2Vec2Model",
        "audio",
        {},
        "real wav fixture via soundfile → Wav2Vec2Processor → input_values",
    ),
    "ast": (
        "hf-internal-testing/tiny-random-ASTModel",
        "ast",
        {},
        "native AST pooler; numpy fbank (torchaudio NOT required); FE geometry "
        "rebuilt from model config (repo preprocessor_config inconsistent)",
    ),
    "whisper": (
        "hf-internal-testing/tiny-random-WhisperForConditionalGeneration",
        "whisper",
        {},
        "encoder-decoder over mel; WhisperProcessor on wav fixture, mel trimmed "
        "to the checkpoint's required length (modeling_whisper.py:616)",
    ),
    "llava-next": (
        "katuni4ka/tiny-random-llava-next",
        "vlm",
        {},
        "vision-language; with-image path via real LlavaNextProcessor batch",
    ),
}


@pytest.fixture(scope="module", params=sorted(VISION_AUDIO_FAMILIES))
def vision_audio_case(request) -> Tuple[str, Any, str, float]:
    """(family, wrapper, used_model_id, load_seconds) — one load per family."""
    family = request.param
    model_id, _kind, kwargs, _notes = VISION_AUDIO_FAMILIES[family]
    wrapper, load_seconds = load_wrapper(model_id, **kwargs)
    return family, wrapper, model_id, load_seconds


def test_matrix_vision_audio_family(vision_audio_case, tmp_path) -> None:
    family, wrapper, model_id, load_seconds = vision_audio_case
    kind = VISION_AUDIO_FAMILIES[family][1]
    notes = VISION_AUDIO_FAMILIES[family][3]
    execute_matrix_case(
        family,
        model_id,
        kind,
        wrapper,
        load_seconds,
        tmp_path,
        test_file="tests/integration/test_matrix_vision_audio.py",
        notes=notes,
    )
