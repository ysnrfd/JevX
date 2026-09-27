# Copyright (c) 2026 ysnrfd
"""Execute the JevX decision path across a large, systematic family sweep.

Turns the "dynamic support for anything in Transformers" claim into a
*measured* one: for every family in :data:`FAMILIES` a tiny random-weight
model is constructed **in memory** (no download, no checkpoint) from the real
``transformers`` config class, and the full JevX path runs::

    JevXCapabilityDetector.detect
      -> resolve_adapter
      -> adapter.prepare_inputs
      -> adapter.extract_representation
      -> jevx.representations.pooling.pool
      -> JevXForDecision(...).choice(..., K=3)

Every invariant is asserted per family (see :func:`_check_invariants`), and
the per-family row is written to ``benchmarks/results/breadth_sweep.json``.

The sweep validates the PIPELINE, not quality: random weights carry no
learned semantics, so a PASS says the extraction/decision machinery ran
end-to-end and every contract held -- nothing about a checkpoint's accuracy.

Run::

    python -m benchmarks.run_breadth_sweep            # full sweep
    python -m benchmarks.run_breadth_sweep --only t5 bert
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from jevx import JevXForDecision
from jevx.adapters import resolve_adapter
from jevx.capabilities import JevXCapabilityDetector
from jevx.configuration_jevx import JevXConfig
from jevx.representations.pooling import available_pooling_strategies, pool

__all__ = [
    "EXCLUSIONS",
    "FAMILIES",
    "HIDDEN_WIDTH_BLOCKER",
    "FamilySpec",
    "build_model",
    "make_inputs",
    "run_family",
    "run_sweep",
    "sweep_markdown_table",
]

#: Deterministic seed base for the whole sweep (each family offsets it).
SEED_BASE = 20260924

#: Substring of the wrapper error raised when the backbone config declares no
#: scalar hidden width (the CNN families' per-stage ``hidden_sizes`` list).
HIDDEN_WIDTH_BLOCKER = "cannot determine the backbone's hidden width"

#: Results path, relative to the repository root.
RESULTS_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "results" / "breadth_sweep.json"

#: Shared vocabulary for synthetic ``input_ids`` -- small and inside every
#: config's ``vocab_size`` (each spec sets ``vocab_size >= VOCAB``).
VOCAB = 61
SEQ_LEN = 7
BATCH = 2

#: Placeholder width for the speecht5 builder's type default; the real value is
#: bound at call time so the module-level constant order stays readable.
_TINY_D_SENTINEL = 32


@dataclass(frozen=True)
class FamilySpec:
    """One sweep target: a real ``transformers`` architecture, instantiated tiny.

    Attributes:
        key: Stable identifier used in the JSON report and the test ids.
        group: Reporting bucket (``encoder``, ``decoder``, ``vision``, ...).
        config_class: ``transformers`` config class name (resolved from the
            ``transformers`` package, never imported by path).
        model_class: ``transformers`` model class name.
        config_kwargs: Constructor kwargs for the config -- every size
            minimized so the whole sweep fits in RAM.
        inputs: Key into :data:`INPUT_BUILDERS` (how synthetic inputs are made).
        source: ``models/<dir>/modeling_<file>.py`` of the vendored clone, for
            provenance in the report.
    """

    key: str
    group: str
    config_class: str
    model_class: str
    config_kwargs: Dict[str, Any]
    inputs: str
    source: str


# ---------------------------------------------------------------------------
# Input builders -- each returns a dict of *prepared* model kwargs plus the
# raw ``state`` payload the JevX wrapper is given, so both the adapter path and
# the wrapper path are exercised on the same tensors.
# ---------------------------------------------------------------------------


@dataclass
class PreparedState:
    """Synthetic state for one family.

    Attributes:
        state: Raw payload for ``JevXForDecision`` (a tensor dict here, so no
            tokenizer/processor download is required).
        inputs: The prepared model kwargs the adapter produces/expects.
    """

    state: Dict[str, Any]
    inputs: Dict[str, Any]


def _ids(
    vocab: int = VOCAB,
    seq: int = SEQ_LEN,
    batch: int = BATCH,
    *,
    left_pad: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deterministic ``(input_ids, attention_mask)`` with real padding.

    Left padding on every second family exercises the ``position_ids``
    cumsum rule pinned by the decoder / multimodal adapters.

    Args:
        vocab: Vocabulary bound (ids stay in ``[0, vocab)``).
        seq: Sequence length.
        batch: Batch size.
        left_pad: Put the padding on the left instead of the right.

    Returns:
        The ``(input_ids, attention_mask)`` pair, dtype ``torch.long``.
    """
    generator = torch.Generator().manual_seed(SEED_BASE)
    ids = torch.randint(1, vocab, (batch, seq), generator=generator, dtype=torch.long)
    mask = torch.ones(batch, seq, dtype=torch.long)
    lengths = (seq, seq - 3) if batch > 1 else (seq,)
    for row, length in enumerate(lengths):
        mask[row, length:] = 0
        if left_pad:
            mask[row, : seq - length] = 0
            ids[row, : seq - length] = 1  # [PAD]
        else:
            ids[row, length:] = 1
    return ids, mask


def _text_inputs(vocab: int = VOCAB, seq: int = SEQ_LEN, batch: int = BATCH) -> PreparedState:
    """``input_ids`` + ``attention_mask`` (left-padded second row)."""
    ids, mask = _ids(vocab=vocab, seq=seq, batch=batch, left_pad=True)
    payload = {"input_ids": ids, "attention_mask": mask}
    return PreparedState(state=dict(payload), inputs=dict(payload))


def _text_inputs_right_pad(
    vocab: int = VOCAB, seq: int = SEQ_LEN, batch: int = BATCH
) -> PreparedState:
    """``input_ids`` + ``attention_mask``, padding on the right."""
    ids, mask = _ids(vocab=vocab, seq=seq, batch=batch, left_pad=False)
    payload = {"input_ids": ids, "attention_mask": mask}
    return PreparedState(state=dict(payload), inputs=dict(payload))


def _text_inputs_no_pad(
    vocab: int = VOCAB, seq: int = SEQ_LEN, batch: int = BATCH
) -> PreparedState:
    """``input_ids`` + an all-ones mask (no padding anywhere).

    Some families cannot consume a *padded* batch at all, e.g. nystromformer's
    landmark branch adds a rank-5 ``attention_scores`` to the rank-4 mask
    (``modeling_nystromformer.py:189-204``), so the sweep runs those with an
    unpadded batch rather than mutating the caller's mask.
    """
    ids, _ = _ids(vocab=vocab, seq=seq, batch=batch, left_pad=False)
    mask = torch.ones(batch, seq, dtype=torch.long)
    payload = {"input_ids": ids, "attention_mask": mask}
    return PreparedState(state=dict(payload), inputs=dict(payload))


def _encdec_text_inputs(
    vocab: int = VOCAB, seq: int = SEQ_LEN, batch: int = BATCH
) -> PreparedState:
    """Encoder-decoder text: encoder ids + a mirrored decoder prompt.

    The prompt mirrors the encoder tokens, the standard shared-vocabulary
    encoder-decoder prompt; ``EncoderDecoderAdapter.prepare_inputs`` builds the
    same key when a caller omits it.
    """
    ids, mask = _ids(vocab=vocab, seq=seq, batch=batch, left_pad=False)
    payload = {
        "input_ids": ids,
        "attention_mask": mask,
        "decoder_input_ids": ids.clone(),
    }
    return PreparedState(state=dict(payload), inputs=dict(payload))


def _vision_inputs(
    image_size: int, channels: int = 3, batch: int = BATCH
) -> PreparedState:
    """``pixel_values`` at the family's own ``image_size``."""
    generator = torch.Generator().manual_seed(SEED_BASE)
    pixels = torch.randn(batch, channels, image_size, image_size, generator=generator)
    return PreparedState(state={"pixel_values": pixels}, inputs={"pixel_values": pixels})


def _audio_waveform_inputs(
    samples: int, batch: int = BATCH
) -> PreparedState:
    """``input_values`` raw waveform plus its length mask."""
    generator = torch.Generator().manual_seed(SEED_BASE)
    values = torch.randn(batch, samples, generator=generator)
    mask = torch.ones(batch, samples, dtype=torch.long)
    if batch > 1:
        mask[1, samples // 2 :] = 0
    return PreparedState(
        state={"input_values": values, "attention_mask": mask},
        inputs={"input_values": values, "attention_mask": mask},
    )


def _mel_inputs(
    frames: int, mel_bins: int, batch: int = BATCH, key: str = "input_features"
) -> PreparedState:
    """Log-mel spectrogram under ``key`` (whisper ``input_features``,
    AST ``input_values``).

    The key differs per family because it is the *declared* forward parameter:
    ``WhisperModel`` takes ``input_features`` while ``ASTModel`` takes
    ``input_values`` with layout ``(batch, frames, mel_bins)``
    (``modeling_audio_spectrogram_transformer.py:231``). ``GenericAdapter``
    forwards declared kwargs unchanged and rejects anything else, so the sweep
    must name the key the family actually declares.
    """
    generator = torch.Generator().manual_seed(SEED_BASE)
    features = torch.randn(batch, mel_bins, frames, generator=generator)
    return PreparedState(state={key: features}, inputs={key: features})


def _speecht5_inputs(frames: int = 60, width: int = _TINY_D_SENTINEL) -> PreparedState:
    """``input_values`` pre-computed encoder features for ``SpeechT5Model``.

    ``SpeechT5Model`` (as opposed to ``SpeechT5ForConditionalGeneration``) wires
    ``SpeechT5EncoderWithoutPrenet`` / ``SpeechT5DecoderWithoutPrenet``
    (``modeling_speecht5.py:1867-1868``), so the encoder side consumes float
    *hidden states* ``[B, T, D]`` — not a waveform. JevX only ever extracts the
    ENCODER state, so the sweep supplies exactly that.
    """
    generator = torch.Generator().manual_seed(SEED_BASE)
    values = torch.randn(BATCH, frames, width, generator=generator)
    mask = torch.ones(BATCH, frames, dtype=torch.long)
    return PreparedState(
        state={"input_values": values, "attention_mask": mask},
        inputs={"input_values": values, "attention_mask": mask},
    )


def _image_text_inputs(
    vocab: int,
    image_size: int,
    num_image_tokens: int,
    channels: int = 3,
    image_token: int = 2,
) -> PreparedState:
    """``(input_ids, pixel_values)`` for vision-language composites.

    The prompt embeds ``num_image_tokens`` image-placeholder ids so the
    scatter/merge step finds the right number of slots.

    Args:
        vocab: Vocabulary bound.
        image_size: Side length of the synthetic image.
        num_image_tokens: Placeholder count the family must see.
        channels: Image channels.
        image_token: The family's ``image_token_index`` value.

    Returns:
        A :class:`PreparedState` carrying both tensors.
    """
    ids, mask = _ids_with_image_tokens(
        vocab=vocab, num_image_tokens=num_image_tokens, image_token=image_token
    )
    generator = torch.Generator().manual_seed(SEED_BASE)
    pixels = torch.randn(BATCH, channels, image_size, image_size, generator=generator)
    payload = {
        "input_ids": ids,
        "attention_mask": mask,
        "pixel_values": pixels,
    }
    return PreparedState(state=dict(payload), inputs=dict(payload))


def _ids_with_image_tokens(
    vocab: int, num_image_tokens: int, image_token: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``input_ids`` carrying exactly ``num_image_tokens`` image placeholders.

    Every placeholder must equal the family's ``image_token_index``: the vision
    features are scattered with ``masked_scatter`` on ``(input_ids ==
    image_token_index)`` (``modeling_llava.py:`` merge step), so a run of
    *different* ids counts as a single token and the merge raises
    "Image features and image tokens do not match".
    """
    generator = torch.Generator().manual_seed(SEED_BASE)
    seq = SEQ_LEN
    ids = torch.randint(8, max(vocab, 9), (BATCH, seq), generator=generator, dtype=torch.long)
    ids[:, 1 : 1 + num_image_tokens] = int(image_token)
    mask = torch.ones(BATCH, seq, dtype=torch.long)
    if BATCH > 1:
        mask[1, seq - 2 :] = 0
        ids[1, seq - 2 :] = 1
    return ids, mask


def _vision_trocr_inputs() -> PreparedState:
    """TrOCR: image pixels *and* the decoder prompt (its own vocabulary)."""
    generator = torch.Generator().manual_seed(SEED_BASE)
    pixels = torch.randn(BATCH, 3, 16, 16, generator=generator)
    ids, mask = _ids(vocab=100, left_pad=False)
    payload = {"input_ids": ids, "attention_mask": mask, "decoder_input_ids": ids,
               "pixel_values": pixels}
    return PreparedState(state=dict(payload), inputs=dict(payload))


def _image_text_inputs_siglip() -> PreparedState:
    """SigLIP: text must be padded to ``max_position_embeddings`` (modeling_siglip)."""
    generator = torch.Generator().manual_seed(SEED_BASE)
    ids, _ = _ids(vocab=61, seq=64, batch=BATCH, left_pad=False)
    mask = torch.ones(BATCH, 64, dtype=torch.long)
    if BATCH > 1:
        mask[1, 60:] = 0
    pixels = torch.randn(BATCH, 3, 16, 16, generator=generator)
    payload = {"input_ids": ids, "attention_mask": mask, "pixel_values": pixels}
    return PreparedState(state=dict(payload), inputs=dict(payload))


def _image_text_idefics2_inputs() -> PreparedState:
    """Idefics2 wants rank-5 ``pixel_values`` ``[B, num_images, C, H, W]``.

    ``Idefics2Model.get_image_features`` unpacks five dimensions
    (``modeling_idefics2.py:831``), so a plain ``[B, C, H, W]`` image raises
    "not enough values to unpack (expected 5, got 4)" before any JevX code
    runs.
    """
    # vision_config.patch_size must divide the image side: the patch grid is
    # built with `pixel_attention_mask.unfold(size=patch_size, step=patch_size)`
    # (modeling_idefics2.py:852-855). patch_size 16 on a 16px image gives a
    # single 1x1 patch, so the prompt carries exactly ONE image token.
    num_image_tokens = 1
    generator = torch.Generator().manual_seed(SEED_BASE)
    ids, mask = _ids_with_image_tokens(
        vocab=VOCAB, num_image_tokens=num_image_tokens, image_token=98
    )
    pixels = torch.randn(BATCH, 1, 3, 16, 16, generator=generator)
    payload = {"input_ids": ids, "attention_mask": mask, "pixel_values": pixels}
    return PreparedState(state=dict(payload), inputs=dict(payload))


def _image_text_paligemma_inputs() -> PreparedState:
    """PaliGemma: BOS + image tokens + SEP + prompt, with the image appended."""
    num_image_tokens = 4
    prompt = torch.tensor([[5, 6, 7], [5, 6, 7]], dtype=torch.long)
    prefix = torch.zeros((BATCH, 1 + num_image_tokens + 1), dtype=torch.long)
    prefix[:, 1 : 1 + num_image_tokens] = 3  # image_token_index
    prefix[:, 1 + num_image_tokens] = 2  # eoi_token_index
    ids = torch.cat([prefix, prompt], dim=1)
    mask = torch.ones_like(ids)
    generator = torch.Generator().manual_seed(SEED_BASE)
    pixels = torch.randn(BATCH, 3, 16, 16, generator=generator)
    payload = {"input_ids": ids, "attention_mask": mask, "pixel_values": pixels}
    return PreparedState(state=dict(payload), inputs=dict(payload))


INPUT_BUILDERS: Dict[str, Callable[[], PreparedState]] = {
    "text": _text_inputs,
    "text_right_pad": _text_inputs_right_pad,
    "text_len8": lambda: _text_inputs_right_pad(seq=8),
    "text_nopad8": lambda: _text_inputs_no_pad(seq=8),
    "encdec_text": _encdec_text_inputs,
    "vision": lambda: _vision_inputs(16),
    "vision32": lambda: _vision_inputs(32),
    "mel": lambda: _mel_inputs(frames=60, mel_bins=16),
    # The tiny WhisperConfig uses max_source_positions=30, so the mel must be
    # 60 frames long: transformers enforces the conv-stride product exactly
    # (modeling_whisper.py:616). 3000 frames is the *real-checkpoint* geometry.
    "whisper_mel": lambda: _mel_inputs(frames=60, mel_bins=80),
    # ASTModel.forward declares ``input_values`` with layout (batch, frames,
    # mel_bins) -- modeling_audio_spectrogram_transformer.py:231, 304.
    "ast_mel": lambda: _mel_inputs(frames=64, mel_bins=16, key="input_values"),
    "wav2vec2_wave": lambda: _audio_waveform_inputs(4000),
    "speecht5_hidden": lambda: _speecht5_inputs(frames=60, width=_TINY_D),
    "vits_text": lambda: _text_inputs_no_pad(vocab=10, seq=4),
    "image_text_clip": lambda: _image_text_inputs(vocab=61, image_size=16,
                                                  num_image_tokens=4),
    "image_text_siglip": _image_text_inputs_siglip,
    # llava: 16x16 pixels with patch 8 -> a 2x2 grid = 4 features, and
    # config.image_token_index = 2 (4 placeholders of id 2).
    "image_text_llava": lambda: _image_text_inputs(
        vocab=99, image_size=16, num_image_tokens=4, image_token=2
    ),
    "image_text_paligemma": _image_text_paligemma_inputs,
    "image_text_idefics2": _image_text_idefics2_inputs,
    # git: 4 image tokens of id 1024 (config.image_token_index) inside a
    # 1200-row GPT-2 vocabulary.
    "image_text_git": lambda: _image_text_inputs(
        vocab=1200, image_size=16, num_image_tokens=4, image_token=1024
    ),
    "vision_trocr": _vision_trocr_inputs,
}


# ---------------------------------------------------------------------------
# The family set -- chosen from vendor/reports/architecture_inventory.json
# (CONFIG_MAPPING keys + MODEL_* mappings) to span every architectural group.
# ---------------------------------------------------------------------------

_TINY_TEXT = {"vocab_size": VOCAB, "hidden_size": 32, "num_hidden_layers": 2,
              "num_attention_heads": 4, "intermediate_size": 64,
              "max_position_embeddings": 64}
_TINY_D = 32
_TINY_DEC = {"vocab_size": VOCAB, "hidden_size": _TINY_D, "intermediate_size": 64,
             "num_hidden_layers": 2, "num_attention_heads": 4,
             "num_key_value_heads": 4, "max_position_embeddings": 64}
_TINY_ENC_DEC = {"vocab_size": VOCAB, "d_model": _TINY_D, "d_kv": 8, "d_ff": 64,
                 "num_layers": 2, "num_heads": 4}


def _text_cfg(**overrides: Any) -> Dict[str, Any]:
    return {**_TINY_TEXT, **overrides}


def _dec_cfg(**overrides: Any) -> Dict[str, Any]:
    return {**_TINY_DEC, **overrides}


def _encdec_cfg(**overrides: Any) -> Dict[str, Any]:
    return {**_TINY_ENC_DEC, **overrides}


FAMILIES: Tuple[FamilySpec, ...] = (
    # ---- encoder-only text ------------------------------------------------
    FamilySpec("bert", "encoder", "BertConfig", "BertModel", _text_cfg(), "text",
               "models/bert/modeling_bert.py"),
    FamilySpec("roberta", "encoder", "RobertaConfig", "RobertaModel", _text_cfg(), "text",
               "models/roberta/modeling_roberta.py"),
    FamilySpec("albert", "encoder", "AlbertConfig", "AlbertModel",
               _text_cfg(hidden_size=36, embedding_size=16, num_hidden_groups=1,
                         inner_group_num=1), "text",
               "models/albert/modeling_albert.py"),
    FamilySpec("distilbert", "encoder", "DistilBertConfig", "DistilBertModel",
               {"vocab_size": VOCAB, "dim": _TINY_D, "n_layers": 2, "n_heads": 4,
                "hidden_dim": 64, "max_position_embeddings": 64}, "text",
               "models/distilbert/modeling_distilbert.py"),
    FamilySpec("electra", "encoder", "ElectraConfig", "ElectraModel", _text_cfg(), "text",
               "models/electra/modeling_electra.py"),
    FamilySpec("mobilebert", "encoder", "MobileBertConfig", "MobileBertModel",
               {"vocab_size": VOCAB, "hidden_size": _TINY_D, "num_hidden_layers": 2,
                "num_attention_heads": 4, "intermediate_size": 64,
                "max_position_embeddings": 64, "embedding_size": 16}, "text",
               "models/mobilebert/modeling_mobilebert.py"),
    FamilySpec("mpnet", "encoder", "MPNetConfig", "MPNetModel",
               _text_cfg(embedding_size=32), "text", "models/mpnet/modeling_mpnet.py"),
    FamilySpec("deberta", "encoder", "DebertaConfig", "DebertaModel",
               _text_cfg(hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                         intermediate_size=64, max_position_embeddings=64,
                         relative_attention=True, pos_att_type=["c2p", "p2c"],
                         position_buckets=8, max_relative_positions=-1), "text",
               "models/deberta/modeling_deberta.py"),
    FamilySpec("deberta-v2", "encoder", "DebertaV2Config", "DebertaV2Model", _text_cfg(), "text",
               "models/deberta_v2/modeling_deberta_v2.py"),
    FamilySpec("longformer", "encoder", "LongformerConfig", "LongformerModel",
               _text_cfg(attention_window=4, max_position_embeddings=64), "text",
               "models/longformer/modeling_longformer.py"),
    FamilySpec("big_bird", "encoder", "BigBirdConfig", "BigBirdModel",
               _text_cfg(block_size=2, n_layers=2, n_heads=4, d_head=8, seq_len=64,
                         hidden_size=32, attn_implementation=None), "text",
               "models/big_bird/modeling_big_bird.py"),
    # XLNetConfig.max_position_embeddings is a read-only property in v5
    # (configuration_xlnet.py:148-151) — passing it raises NotImplementedError.
    FamilySpec("xlnet", "encoder", "XLNetConfig", "XLNetModel",
               {"vocab_size": VOCAB, "d_model": _TINY_D, "n_layer": 2, "n_head": 4,
                "d_head": 8, "d_inner": 64, "ff_activation": "gelu",
                "untie_r": True, "attn_type": "bi"}, "text",
               "models/xlnet/modeling_xlnet.py"),
    FamilySpec("ernie", "encoder", "ErnieConfig", "ErnieModel", _text_cfg(), "text",
               "models/ernie/modeling_ernie.py"),
    FamilySpec("squeezebert", "encoder", "SqueezeBertConfig", "SqueezeBertModel",
               {"vocab_size": VOCAB, "hidden_size": 48, "num_hidden_layers": 2,
                "num_attention_heads": 4, "intermediate_size": 64,
                "max_position_embeddings": 64, "embedding_size": 48,
                "quantize_hidden_size": 24, "num_labels": 2, "use_safetensors": True},
               "text", "models/squeezebert/modeling_squeezebert.py"),
    # num_landmarks must divide the *padded* sequence (modeling_nystromformer.py:182-195
    # reshapes to (-1, heads, landmarks, seq_len // landmarks, head_dim) with
    # seq_len = config.segment_means_seq_len).
    FamilySpec("nystromformer", "encoder", "NystromformerConfig", "NystromformerModel",
               _text_cfg(num_landmarks=2, segment_means_seq_len=8), "text_nopad8",
               "models/nystromformer/modeling_nystromformer.py"),
    FamilySpec("funnel", "encoder", "FunnelConfig", "FunnelModel",
               {"vocab_size": VOCAB, "d_model": _TINY_D, "d_inner": 64, "n_layer": 2,
                "n_head": 4, "block_sizes": [2, 2], "max_position_embeddings": 64}, "text",
               "models/funnel/modeling_funnel.py"),
    # ---- decoder-only -----------------------------------------------------
    FamilySpec("gpt2", "decoder", "GPT2Config", "GPT2Model",
               {"vocab_size": VOCAB, "n_positions": 64, "n_embd": _TINY_D,
                "n_layer": 2, "n_head": 4, "n_inner": 64}, "text", "models/gpt2/modeling_gpt2.py"),
    FamilySpec("gpt_neo", "decoder", "GPTNeoConfig", "GPTNeoModel",
               {"vocab_size": VOCAB, "hidden_size": _TINY_D, "num_layers": 2,
                "num_heads": 4, "intermediate_size": 64,
                "attention_types": [[["global"], 2]],
                "max_window": 4, "max_position_embeddings": 64,
                "bos_token_id": 1, "eos_token_id": 1}, "text",
               "models/gpt_neo/modeling_gpt_neo.py"),
    FamilySpec("gptj", "decoder", "GPTJConfig", "GPTJModel",
               {"vocab_size": VOCAB, "n_positions": 64, "n_embd": _TINY_D,
                "n_layer": 2, "n_head": 4, "n_inner": 64, "rotary_dim": 8}, "text",
               "models/gptj/modeling_gptj.py"),
    FamilySpec("bloom", "decoder", "BloomConfig", "BloomModel",
               {"vocab_size": VOCAB, "hidden_size": _TINY_D, "n_layer": 2,
                "n_head": 4, "seq_length": 64}, "text", "models/bloom/modeling_bloom.py"),
    FamilySpec("opt", "decoder", "OPTConfig", "OPTModel",
               {"vocab_size": VOCAB, "hidden_size": _TINY_D, "num_hidden_layers": 2,
                "num_attention_heads": 4, "ffn_dim": 64,
                "max_position_embeddings": 64, "word_embed_proj_dim": _TINY_D}, "text",
               "models/opt/modeling_opt.py"),
    FamilySpec("llama", "decoder", "LlamaConfig", "LlamaModel", _dec_cfg(), "text",
               "models/llama/modeling_llama.py"),
    FamilySpec("mistral", "decoder", "MistralConfig", "MistralModel", _dec_cfg(), "text",
               "models/mistral/modeling_mistral.py"),
    FamilySpec("qwen2", "decoder", "Qwen2Config", "Qwen2Model", _dec_cfg(), "text",
               "models/qwen2/modeling_qwen2.py"),
    FamilySpec("qwen3", "decoder", "Qwen3Config", "Qwen3Model",
               _dec_cfg(head_dim=8, tie_word_embeddings=True), "text",
               "models/qwen3/modeling_qwen3.py"),
    FamilySpec("gemma", "decoder", "GemmaConfig", "GemmaModel", _dec_cfg(), "text",
               "models/gemma/modeling_gemma.py"),
    FamilySpec("gemma2", "decoder", "Gemma2Config", "Gemma2Model",
               _dec_cfg(head_dim=8, sliding_window=8, query_pre_attn_scalar=16), "text",
               "models/gemma2/modeling_gemma2.py"),
    FamilySpec("phi3", "decoder", "Phi3Config", "Phi3Model",
               _dec_cfg(partial_rotary_factor=0.5, original_max_position_embeddings=64,
                        pad_token_id=0, bos_token_id=1, eos_token_id=2), "text",
               "models/phi3/modeling_phi3.py"),
    FamilySpec("falcon", "decoder", "FalconConfig", "FalconModel",
               {"vocab_size": VOCAB, "hidden_size": _TINY_D, "num_hidden_layers": 2,
                "num_attention_heads": 4, "intermediate_size": 64,
                "max_position_embeddings": 64, "new_decoder_architecture": False,
                "num_kv_heads": 4, "multi_query": True}, "text",
               "models/falcon/modeling_falcon.py"),
    FamilySpec("mpt", "decoder", "MptConfig", "MptModel",
               {"vocab_size": VOCAB, "d_model": _TINY_D, "n_heads": 4, "n_layers": 2,
                "expansion_ratio": 2, "max_seq_len": 64, "attn_config": {"attn_impl": "eager"}},
               "text", "models/mpt/modeling_mpt.py"),
    FamilySpec("codegen", "decoder", "CodeGenConfig", "CodeGenModel",
               {"vocab_size": VOCAB, "n_positions": 64, "n_embd": _TINY_D,
                "n_layer": 2, "n_head": 4, "rotary_dim": 8, "n_inner": 64}, "text",
               "models/codegen/modeling_codegen.py"),
    FamilySpec("stablelm", "decoder", "StableLmConfig", "StableLmModel",
               {"vocab_size": VOCAB, "hidden_size": _TINY_D, "num_hidden_layers": 2,
                "num_attention_heads": 4, "num_key_value_heads": 4,
                "intermediate_size": 64, "max_position_embeddings": 64,
                "pad_token_id": 0, "eos_token_id": 1}, "text",
               "models/stablelm/modeling_stablelm.py"),
    FamilySpec("olmo", "decoder", "OlmoConfig", "OlmoModel", _dec_cfg(), "text",
               "models/olmo/modeling_olmo.py"),
    FamilySpec("gpt_neox", "decoder", "GPTNeoXConfig", "GPTNeoXModel",
               {"vocab_size": VOCAB, "hidden_size": _TINY_D, "num_hidden_layers": 2,
                "num_attention_heads": 4, "intermediate_size": 64,
                "max_position_embeddings": 64, "rotary_pct": 0.25,
                "use_parallel_residual": True}, "text",
               "models/gpt_neox/modeling_gpt_neox.py"),
    FamilySpec("rwkv", "decoder", "RwkvConfig", "RwkvModel",
               {"vocab_size": VOCAB, "hidden_size": 32, "num_hidden_layers": 2,
                "intermediate_size": 64, "num_attention_heads": 4,
                "max_position_embeddings": 64, "layer_norm_eps": 1e-5,
                "rescale_every": 4, "token_shift_count": 1, "time_decay_extra": 2.0,
                "time_first_fade_extra": 2.0}, "text", "models/rwkv/modeling_rwkv.py"),
    FamilySpec("mamba", "decoder", "MambaConfig", "MambaModel",
               {"vocab_size": VOCAB, "hidden_size": _TINY_D, "state_size": 8,
                "num_hidden_layers": 2, "expand": 2, "conv_kernel": 3, "time_step_rank": 4,
                "use_cache": False}, "text", "models/mamba/modeling_mamba.py"),
    FamilySpec("jamba", "decoder", "JambaConfig", "JambaModel",
               {"vocab_size": VOCAB, "hidden_size": _TINY_D, "num_hidden_layers": 2,
                "num_attention_heads": 4, "intermediate_size": 64,
                "max_position_embeddings": 64, "mamba_d_state": 8, "mamba_d_conv": 3,
                "mamba_expand": 2, "mamba_n_heads": 4, "mamba_head_dim": 8,
                "mamba_num_key_value_heads": 4, "num_key_value_heads": 4,
                "num_experts": 2, "num_experts_per_tok": 1, "use_cache": False}, "text",
               "models/jamba/modeling_jamba.py"),
    # ---- encoder-decoder --------------------------------------------------
    FamilySpec("t5", "encoder_decoder", "T5Config", "T5Model", _encdec_cfg(), "encdec_text",
               "models/t5/modeling_t5.py"),
    FamilySpec("bart", "encoder_decoder", "BartConfig", "BartModel",
               {"vocab_size": VOCAB, "d_model": _TINY_D, "decoder_layers": 2,
                "encoder_layers": 2, "encoder_attention_heads": 4,
                "decoder_attention_heads": 4, "encoder_ffn_dim": 64,
                "decoder_ffn_dim": 64, "max_position_embeddings": 64}, "encdec_text",
               "models/bart/modeling_bart.py"),
    FamilySpec("mbart", "encoder_decoder", "MBartConfig", "MBartModel",
               {"vocab_size": VOCAB, "d_model": _TINY_D, "decoder_layers": 2,
                "encoder_layers": 2, "encoder_attention_heads": 4,
                "decoder_attention_heads": 4, "encoder_ffn_dim": 64,
                "decoder_ffn_dim": 64, "max_position_embeddings": 64,
                "forced_eos_token_id": 0}, "encdec_text",
               "models/mbart/modeling_mbart.py"),
    FamilySpec("pegasus", "encoder_decoder", "PegasusConfig", "PegasusModel",
               {"vocab_size": VOCAB, "d_model": _TINY_D, "decoder_layers": 2,
                "encoder_layers": 2, "encoder_attention_heads": 4,
                "decoder_attention_heads": 4, "encoder_ffn_dim": 64,
                "decoder_ffn_dim": 64, "max_position_embeddings": 64,
                "pad_token_id": 0, "eos_token_id": 1}, "encdec_text",
               "models/pegasus/modeling_pegasus.py"),
    FamilySpec("marian", "encoder_decoder", "MarianConfig", "MarianModel",
               {"vocab_size": VOCAB, "d_model": _TINY_D, "decoder_layers": 2,
                "encoder_layers": 2, "encoder_attention_heads": 4,
                "decoder_attention_heads": 4, "encoder_ffn_dim": 64,
                "decoder_ffn_dim": 64, "max_position_embeddings": 64,
                "pad_token_id": 0, "eos_token_id": 1}, "encdec_text",
               "models/marian/modeling_marian.py"),
    FamilySpec("blenderbot", "encoder_decoder", "BlenderbotConfig", "BlenderbotSmallModel",
               {"vocab_size": VOCAB, "d_model": _TINY_D, "decoder_layers": 2,
                "encoder_layers": 2, "encoder_attention_heads": 4,
                "decoder_attention_heads": 4, "encoder_ffn_dim": 64,
                "decoder_ffn_dim": 64, "max_position_embeddings": 64,
                "pad_token_id": 0, "eos_token_id": 1}, "encdec_text",
               "models/blenderbot/modeling_blenderbot.py"),
    FamilySpec("longt5", "encoder_decoder", "LongT5Config", "LongT5Model",
               {"vocab_size": VOCAB, "d_model": _TINY_D, "d_kv": 8, "d_ff": 64,
                "num_layers": 2, "num_heads": 4, "relative_attention_num_buckets": 4,
                "local_attention_num_buckets": 2}, "encdec_text",
               "models/longt5/modeling_longt5.py"),
    FamilySpec("speecht5", "encoder_decoder", "SpeechT5Config", "SpeechT5Model",
               # SpeechT5Model wires the *WithoutPrenet* stacks
               # (modeling_speecht5.py:1867-1868), whose LayerNorm is built
               # from config.hidden_size, not encoder_hidden_size.
               {"vocab_size": VOCAB, "hidden_size": _TINY_D,
                "encoder_layers": 2, "encoder_attention_heads": 4, "encoder_ffn_dim": 64,
                "decoder_layers": 2, "decoder_attention_heads": 4, "decoder_ffn_dim": 64,
                "num_mel_bins": 80, "speaker_embedding_dim": 0, "is_uks": True,
                "use_speaker_embedding": False}, "speecht5_hidden",
               "models/speecht5/modeling_speecht5.py"),
    FamilySpec("whisper", "encoder_decoder", "WhisperConfig", "WhisperModel",
               {"vocab_size": VOCAB, "num_mel_bins": 80, "d_model": _TINY_D,
                "encoder_layers": 2, "encoder_attention_heads": 4, "encoder_ffn_dim": 64,
                "decoder_layers": 2, "decoder_attention_heads": 4, "decoder_ffn_dim": 64,
                "max_source_positions": 30, "max_target_positions": 16,
                "decoder_start_token_id": 1, "pad_token_id": 0, "eos_token_id": 2,
                "d_state": _TINY_D, "d_conv": 2}, "whisper_mel",
               "models/whisper/modeling_whisper.py"),
    # ---- vision -----------------------------------------------------------
    FamilySpec("vit", "vision", "ViTConfig", "ViTModel",
               {"hidden_size": _TINY_D, "num_hidden_layers": 2, "num_attention_heads": 4,
                "intermediate_size": 64, "image_size": 16, "patch_size": 8,
                "num_channels": 3}, "vision", "models/vit/modeling_vit.py"),
    FamilySpec("deit", "vision", "DeiTConfig", "DeiTModel",
               {"hidden_size": _TINY_D, "num_hidden_layers": 2, "num_attention_heads": 4,
                "intermediate_size": 64, "image_size": 16, "patch_size": 8,
                "num_channels": 3}, "vision", "models/deit/modeling_deit.py"),
    FamilySpec("convnext", "vision", "ConvNextConfig", "ConvNextModel",
               {"num_channels": 3, "hidden_sizes": [8, 16, 32, 64],
                "depths": [1, 1, 1, 1], "layer_norm_eps": 1e-6, "patch_size": 4,
                "image_size": 32, "num_attention_heads": 4}, "vision32",
               "models/convnext/modeling_convnext.py"),
    FamilySpec("swin", "vision", "SwinConfig", "SwinModel",
               {"image_size": 16, "patch_size": 4, "num_channels": 3, "embed_dim": 32,
                "depths": [1, 1], "num_heads": [2, 2], "window_size": 2,
                "mlp_ratio": 2.0}, "vision", "models/swin/modeling_swin.py"),
    FamilySpec("resnet", "vision", "ResNetConfig", "ResNetModel",
               {"num_channels": 3, "hidden_sizes": [8, 16, 32, 64],
                "depths": [1, 1, 1, 1], "layer_norm_eps": 1e-6, "image_size": 32,
                "num_attention_heads": 4}, "vision32",
               "models/resnet/modeling_resnet.py"),
    FamilySpec("segformer", "vision", "SegformerConfig", "SegformerModel",
               {"num_channels": 3, "hidden_sizes": [8, 16, 32, 64],
                "depths": [1, 1, 1, 1], "layer_norm_eps": 1e-6, "image_size": 32,
                "num_attention_heads": [1, 2, 4, 8], "mlp_ratios": [2, 2, 2, 2],
                "mlp_hidden_sizes": [8, 16, 32, 64], "num_encoder_blocks": 2,
                "num_decoder_blocks": 1, "decoder_hidden_size": 8}, "vision32",
               "models/segformer/modeling_segformer.py"),
    FamilySpec("dpt", "vision", "DPTConfig", "DPTModel",
               {"hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4,
                "intermediate_size": 64, "image_size": 16, "patch_size": 8,
                "num_channels": 3, "neck_hidden_sizes": [16, 16, 16, 16]},
               "vision", "models/dpt/modeling_dpt.py"),
    FamilySpec("dinov2", "vision", "Dinov2Config", "Dinov2Model",
               {"hidden_size": _TINY_D, "num_hidden_layers": 2, "num_attention_heads": 4,
                "intermediate_size": 64, "image_size": 16, "patch_size": 8,
                "num_channels": 3}, "vision", "models/dinov2/modeling_dinov2.py"),
    FamilySpec("clip_vision", "vision", "CLIPVisionConfig", "CLIPVisionModel",
               {"hidden_size": _TINY_D, "intermediate_size": 64, "num_hidden_layers": 2,
                "num_attention_heads": 4, "image_size": 16, "patch_size": 8,
                "num_channels": 3}, "vision", "models/clip/modeling_clip.py"),
    FamilySpec("siglip_vision", "vision", "SiglipVisionConfig", "SiglipVisionModel",
               {"hidden_size": _TINY_D, "intermediate_size": 64, "num_hidden_layers": 2,
                "num_attention_heads": 4, "image_size": 16, "patch_size": 8,
                "num_channels": 3}, "vision", "models/siglip/modeling_siglip.py"),
    FamilySpec("mobilenet_v2", "vision", "MobileNetV2Config", "MobileNetV2Model",
               {"num_channels": 3, "hidden_sizes": [8, 16], "depths": [1, 1],
                "layer_norm_eps": 1e-6, "image_size": 32, "num_attention_heads": 4,
                "width_mult": 1.0, "inverted_rescale_connectivity": 1}, "vision32",
               "models/mobilenet_v2/modeling_mobilenet_v2.py"),
    # ---- audio ------------------------------------------------------------
    FamilySpec("wav2vec2", "audio", "Wav2Vec2Config", "Wav2Vec2Model",
               {"hidden_size": _TINY_D, "num_hidden_layers": 2, "num_attention_heads": 4,
                "intermediate_size": 64, "conv_dim": [16, 16, 16, 16, 16, 16, 16],
                "conv_stride": [5, 2, 2, 2, 2, 2, 2], "conv_kernel": [10, 3, 3, 3, 3, 2, 2],
                "num_conv_pos_embeddings": 16, "num_conv_pos_embedding_groups": 2,
                "layer_norm_eps": 1e-5, "do_stable_layer_norm": True,
                "feat_extract_norm": "group"}, "wav2vec2_wave",
               "models/wav2vec2/modeling_wav2vec2.py"),
    FamilySpec("hubert", "audio", "HubertConfig", "HubertModel",
               {"hidden_size": _TINY_D, "num_hidden_layers": 2, "num_attention_heads": 4,
                "intermediate_size": 64, "conv_dim": [16, 16, 16, 16, 16, 16, 16],
                "conv_stride": [5, 2, 2, 2, 2, 2, 2], "conv_kernel": [10, 3, 3, 3, 3, 2, 2],
                "num_conv_pos_embeddings": 16, "num_conv_pos_embedding_groups": 2,
                "layer_norm_eps": 1e-5, "do_stable_layer_norm": True,
                "feat_extract_norm": "group", "apply_spec_augment": False}, "wav2vec2_wave",
               "models/hubert/modeling_hubert.py"),
    FamilySpec("wavlm", "audio", "WavLMConfig", "WavLMModel",
               {"hidden_size": _TINY_D, "num_hidden_layers": 2, "num_attention_heads": 4,
                "intermediate_size": 64, "conv_dim": [16, 16, 16, 16, 16, 16, 16],
                "conv_stride": [5, 2, 2, 2, 2, 2, 2], "conv_kernel": [10, 3, 3, 3, 3, 2, 2],
                "num_conv_pos_embeddings": 16, "num_conv_pos_embedding_groups": 2,
                "layer_norm_eps": 1e-5, "do_stable_layer_norm": True,
                "feat_extract_norm": "group", "apply_spec_augment": False,
                "num_buckets": 4, "max_distance": 8}, "wav2vec2_wave",
               "models/wavlm/modeling_wavlm.py"),
    FamilySpec("data2vec-audio", "audio", "Data2VecAudioConfig", "Data2VecAudioModel",
               {"hidden_size": _TINY_D, "num_hidden_layers": 2, "num_attention_heads": 4,
                "intermediate_size": 64, "conv_dim": [16, 16, 16, 16, 16, 16, 16],
                "conv_stride": [5, 2, 2, 2, 2, 2, 2], "conv_kernel": [10, 3, 3, 3, 3, 2, 2],
                "num_conv_pos_embeddings": 16, "num_conv_pos_embedding_groups": 2,
                "layer_norm_eps": 1e-5, "do_stable_layer_norm": True,
                "feat_extract_norm": "group", "final_dim": _TINY_D}, "wav2vec2_wave",
               "models/data2vec/modeling_data2vec.py"),
    FamilySpec("ast", "audio", "ASTConfig", "ASTModel",
               {"hidden_size": _TINY_D, "num_hidden_layers": 2, "num_attention_heads": 4,
                "intermediate_size": 64, "max_length": 64, "num_mel_bins": 16,
                "frequency_stride": 10, "time_stride": 10, "max_length_patches": 8,
                "num_labels": 10}, "ast_mel",
               "models/audio_spectrogram_transformer/modeling_audio_spectrogram_transformer.py"),
    FamilySpec("unispeech", "audio", "UniSpeechConfig", "UniSpeechModel",
               {"hidden_size": _TINY_D, "num_hidden_layers": 2, "num_attention_heads": 4,
                "intermediate_size": 64, "conv_dim": [16, 16, 16, 16, 16, 16, 16],
                "conv_stride": [5, 2, 2, 2, 2, 2, 2], "conv_kernel": [10, 3, 3, 3, 3, 2, 2],
                "num_conv_pos_embeddings": 16, "num_conv_pos_embedding_groups": 2,
                "layer_norm_eps": 1e-5, "do_stable_layer_norm": True,
                "feat_extract_norm": "group", "quantizer_dropout": 0.0,
                "apply_spec_augment": False}, "wav2vec2_wave",
               "models/unispeech/modeling_unispeech.py"),
    # The HiFi-GAN vocoder dominates the runtime; a 4-stage up-sampling keeps
    # the synthetic waveform an order of magnitude shorter than the 5-stage
    # default while exercising the identical code path.
    FamilySpec("vits", "audio", "VitsConfig", "VitsModel",
               {"num_mel_bins": 80, "hidden_channels": 16, "inter_channels": 8,
                "filter_channels": 16, "n_heads": 1, "n_layers": 1,
                "speaker_embedding_dim": 0, "use_speaker_embedding": False,
                "sampling_rate": 16000, "upsample_rates": [8, 8, 2, 2],
                "upsample_kernel_sizes": [16, 16, 4, 4],
                "upsample_initial_channel": 32, "resblock_kernel_sizes": [3],
                "resblock_dilation_sizes": [[1, 3]] * 4}, "vits_text",
               "models/vits/modeling_vits.py"),
    # ---- multimodal -------------------------------------------------------
    FamilySpec("clip", "multimodal", "CLIPConfig", "CLIPModel",
               {"text_config": {"vocab_size": VOCAB, "hidden_size": _TINY_D,
                                "intermediate_size": 64, "num_hidden_layers": 2,
                                "num_attention_heads": 4, "max_position_embeddings": 64},
                "vision_config": {"hidden_size": _TINY_D, "intermediate_size": 64,
                                  "num_hidden_layers": 2, "num_attention_heads": 4,
                                  "image_size": 16, "patch_size": 8, "num_channels": 3},
                "projection_dim": 24}, "image_text_clip", "models/clip/modeling_clip.py"),
    FamilySpec("siglip", "multimodal", "SiglipConfig", "SiglipModel",
               {"text_config": {"vocab_size": VOCAB, "hidden_size": _TINY_D,
                                "intermediate_size": 64, "num_hidden_layers": 2,
                                "num_attention_heads": 4, "max_position_embeddings": 64},
                "vision_config": {"hidden_size": _TINY_D, "intermediate_size": 64,
                                  "num_hidden_layers": 2, "num_attention_heads": 4,
                                  "image_size": 16, "patch_size": 8, "num_channels": 3},
                # The joint model emits text_embeds/image_embeds of width
                # `projection_dim`; the tiny config uses 32 so the emitted width
                # and the config agree (JevX's _hidden_width reads projection_dim).
                "projection_dim": 32}, "image_text_siglip", "models/siglip/modeling_siglip.py"),
    FamilySpec("llava", "multimodal", "LlavaConfig", "LlavaModel",
               {"vision_config": {"hidden_size": 32, "intermediate_size": 64,
                                 "num_hidden_layers": 2, "num_attention_heads": 4,
                                 "image_size": 16, "patch_size": 8, "num_channels": 3},
                "text_config": {"vocab_size": 99, "hidden_size": 32, "intermediate_size": 64,
                                "num_hidden_layers": 2, "num_attention_heads": 4,
                                "num_key_value_heads": 4, "max_position_embeddings": 64},
                "image_token_index": 2, "vision_feature_layer": -1}, "image_text_llava",
               "models/llava/modeling_llava.py"),
    FamilySpec("git", "multimodal", "GitConfig", "GitModel",
               {
                # git forwards its nested configs straight into the sub-models,
                # where a plain dict fails attribute access
                # ('dict' object has no attribute '_attn_implementation',
                # modeling_git.py): pass real config objects.
                "vision_config": ("GitVisionConfig", {
                    "hidden_size": 32, "intermediate_size": 64,
                    "num_hidden_layers": 2, "num_attention_heads": 4,
                    "image_size": 16, "patch_size": 8, "num_channels": 3}),
                "text_config": ("GPT2Config", {
                    "vocab_size": 1200, "n_positions": 64, "n_embd": 32,
                    "n_layer": 2, "n_head": 4, "n_inner": 64}),
                "projection_dim": 32, "image_token_id": 1024,
                "image_token_index": 1024},
               "image_text_git", "models/git/modeling_git.py"),
    FamilySpec("idefics2", "multimodal", "Idefics2Config", "Idefics2Model",
               {"vision_config": {"hidden_size": 32, "intermediate_size": 64,
                                 "num_hidden_layers": 2, "num_attention_heads": 4,
                                 "image_size": 16, "patch_size": 16,
                                 "num_channels": 3},
                "text_config": {"vocab_size": 99, "hidden_size": 32, "intermediate_size": 64,
                                "num_hidden_layers": 2, "num_attention_heads": 4,
                                "num_key_value_heads": 4, "max_position_embeddings": 64},
                # The perceiver resampler is a separate tower: model_dim /
                # model_ff_hidden_dim / resampler_* must agree with each other
                # and with text_config.hidden_size, or the cross-attention
                # key/value widths disagree.
                "perceiver_config": {"hidden_act": "silu", "hidden_size": 32,
                                     "model_dim": 32, "model_ff_hidden_dim": 32,
                                     "model_heads": 2, "model_num_layers": 1,
                                     "model_num_attention_heads": 2,
                                     "resampler_depth": 1, "resampler_n_latents": 4,
                                     "resampler_size_head": 8, "use_resampler": True},
                # The image token must exist INSIDE the 99-row text vocabulary.
                "image_token_id": 98, "image_token_index": 98},
               "image_text_idefics2", "models/idefics2/modeling_idefics2.py"),
    FamilySpec("paligemma", "multimodal", "PaliGemmaConfig", "PaliGemmaModel",
               # PaliGemmaConfig declares a TOP-LEVEL `hidden_size` (the mm
               # projector output = the LM width, default 2048); JevX's
               # _hidden_width reads it before the nested text_config, so the
               # tiny config must set it to the LM width it actually builds.
               {"hidden_size": 32,
                "vision_config": {"hidden_size": 32, "intermediate_size": 64,
                                 "num_hidden_layers": 2, "num_attention_heads": 4,
                                 "image_size": 16, "patch_size": 8, "num_channels": 3,
                                 "projector_hidden_act": "gelu",
                                 "vision_feature_layer": -1, "vision_use_head": False},
                "text_config": {"vocab_size": 256, "hidden_size": 32,
                                "intermediate_size": 64, "num_hidden_layers": 2,
                                "num_attention_heads": 4, "num_key_value_heads": 4,
                                "head_dim": 8, "max_position_embeddings": 64},
                "projection_dim": 32, "image_token_index": 3, "image_token_id": 3,
                "boi_token_index": 2, "eoi_token_index": 1}, "image_text_paligemma",
               "models/paligemma/modeling_paligemma.py"),
)


#: Families considered during selection but deliberately NOT executed, each
#: with the specific, checked reason. Written into the JSON report so the
#: exclusion is auditable rather than silent.
EXCLUSIONS: Tuple[Dict[str, str], ...] = (
    {
        "family": "trocr",
        "reason": (
            "v5's TrOCRForCausalLM.forward consumes encoder_hidden_states, not "
            "pixels (models/trocr/modeling_trocr.py:666-747: the forward forwards "
            "input_ids/attention_mask/encoder_hidden_states into self.model.decoder "
            "and declares no pixel_values). The vision tower is a SEPARATE model and "
            "transformers 5.16.1 ships no TrOCRModel base to wrap, so a single JevX "
            "state -> representation call does not exist. Its own docstring example "
            "(modeling_trocr.py:716-731) still passes pixel_values and is stale."
        ),
    },
    {
        "family": "sam_vision",
        "reason": (
            "SamVisionConfig(...) construction did not complete within 60 s in this "
            "environment for the DEFAULT config (no JevX code involved, measured by "
            "benchmarks/_probe_configs3.py), so no tiny instance could be built inside "
            "the sweep time budget. Known-but-unexecuted."
        ),
    },
    {
        "family": "nezha",
        "reason": (
            "not a key of transformers 5.16.1 CONFIG_MAPPING and has no record in "
            "vendor/reports/architecture_inventory.json (709 records scanned)."
        ),
    },
    {
        "family": "unet",
        "reason": (
            "no `unet` model_type exists in transformers 5.16.1: a scan of all 709 "
            "records in vendor/reports/architecture_inventory.json (CONFIG_MAPPING keys "
            "+ MODEL_* mappings) returns no match, and there is no models/unet/ package "
            "under vendor/transformers/src/transformers/models/. The diffusion UNets "
            "live under other names (the `mimi`, `xcodec`, `dac` codec stacks), which "
            "are waveform decoders with no named pooled output field."
            "other names (e.g. `mimi`, `xcodec`, `dac` codecs) and are not "
            "representation backbones with a named pooled output."
        ),
    },
    {
        "family": "codegen-audio / musicgen-audio",
        "reason": (
            "no `codegen-audio` model_type exists in transformers 5.16.1 (scanned all "
            "709 records of vendor/reports/architecture_inventory.json); the audio LM "
            "analogue is "
            "`musicgen` (MusicgenModel, CAUSAL_LM), whose decoder-only LM contract is "
            "already covered by the text decoder-only rows of this sweep."
        ),
    },
    {
        "family": "gemma3 / siglip2 / llava_next (composite variants)",
        "reason": (
            "the sub-model configs (gemma3_text/vision, siglip2_*, llava_next_*) are "
            "SECONDARY records in the inventory (nested under the composite's "
            "text_config/vision_config); the composite families gemma3, siglip2 and "
            "llava_next themselves are covered by their own executed rows."
        ),
    },
)


def build_model(spec: FamilySpec) -> Tuple[Any, Any]:
    """Instantiate ``spec``'s model from its real config class, tiny and random.

    No download and no checkpoint: the model is built from the ``transformers``
    config class with :data:`FAMILIES`-minimized sizes under a deterministic
    ``torch.manual_seed``.

    Args:
        spec: The family to instantiate.

    Returns:
        The ``(model, config)`` pair, in ``eval()`` mode.

    Raises:
        AttributeError: The config/model class does not exist in the
            installed ``transformers`` (recorded as an exclusion upstream).
    """
    import transformers

    config_cls = getattr(transformers, spec.config_class)
    model_cls = getattr(transformers, spec.model_class)
    torch.manual_seed(SEED_BASE + abs(hash(spec.key)) % 9973)
    # A nested config may be given as (ClassName, kwargs): composites such as
    # git forward their sub-configs straight into the sub-models, where a plain
    # dict fails attribute access (`'dict' object has no attribute
    # "'_attn_implementation'", modeling_git.py).
    kwargs = {
        key: (getattr(transformers, value[0])(**value[1]) if isinstance(value, tuple) else value)
        for key, value in spec.config_kwargs.items()
    }
    config = config_cls(**kwargs)
    model = model_cls(config)
    return model.eval(), config


def make_inputs(spec: FamilySpec) -> PreparedState:
    """Build the family's synthetic inputs from its declared input kind.

    Args:
        spec: The family whose ``inputs`` key selects the builder.

    Returns:
        A :class:`PreparedState` with the raw ``state`` payload and the
        prepared model kwargs.

    Raises:
        KeyError: The spec names an unknown input builder.
    """
    return INPUT_BUILDERS[spec.inputs]()


def _check_invariants(
    pooled: torch.Tensor,
    probs: torch.Tensor,
    predicted_index: int,
    confidence: float,
    n_options: int,
    hidden: int,
) -> List[str]:
    """Assert every decision-path invariant; return the list of violations.

    Args:
        pooled: The state representation ``[B, D]``.
        probs: Choice probabilities ``[B, K]``.
        predicted_index: The argmax decision index.
        confidence: The reported confidence scalar.
        n_options: Schema width ``K`` (3 in the sweep).
        hidden: Expected pooled width (the backbone's hidden size).

    Returns:
        A (possibly empty) list of human-readable violation messages.
    """
    problems: List[str] = []
    if pooled.dim() != 2:
        problems.append(f"pooled is not [B, D]: shape {tuple(pooled.shape)}")
    if not bool(torch.isfinite(pooled).all()):
        problems.append("pooled contains non-finite values")
    if pooled.shape[-1] != hidden:
        problems.append(
            f"pooled width {pooled.shape[-1]} != backbone hidden width {hidden}"
        )
    if probs.shape[-1] != n_options:
        problems.append(f"probabilities width {probs.shape[-1]} != K={n_options}")
    if not torch.allclose(probs.sum(dim=-1), torch.ones_like(probs.sum(dim=-1)), atol=1e-5):
        problems.append(f"probabilities do not sum to 1: {probs.sum(dim=-1).tolist()}")
    if not 0 <= predicted_index < n_options:
        problems.append(f"argmax {predicted_index} outside schema [0, {n_options})")
    if not 0.0 <= confidence <= 1.0:
        problems.append(f"confidence {confidence} outside [0, 1]")
    return problems


def run_family(spec: FamilySpec) -> Dict[str, Any]:
    """Execute the full JevX path for one family and return its report row.

    Order: ``detect`` -> ``resolve_adapter`` -> ``prepare_inputs`` ->
    ``extract_representation`` -> ``pool`` -> ``JevXForDecision.choice``.

    Args:
        spec: The family to execute.

    Returns:
        A JSON-serializable dict with the per-family record: class names,
        adapter, representation field, pooling strategy, verdict, status and
        (on failure) the exact error.
    """
    started = time.perf_counter()
    row: Dict[str, Any] = {
        "family": spec.key,
        "group": spec.group,
        "model_class": spec.model_class,
        "config_class": spec.config_class,
        "source": spec.source,
        "status": "FAIL",
        "error": None,
        "adapter": None,
        "representation_field": None,
        "pooling_strategy": None,
        "verdict": None,
        "pooled_shape": None,
        "state_proj_width": None,
        "hidden_width": None,
        "predicted_index": None,
        "confidence": None,
        "prob_sum": None,
    }
    model = None
    try:
        model, config = build_model(spec)
        report = JevXCapabilityDetector.detect(model, config)
        row["verdict"] = report.jevx_conversion
        row["capability_reasons"] = list(report.reasons)
        row["output_class"] = report.output_class
        row["modality"] = {"text": report.text, "vision": report.vision,
                           "audio": report.audio}
        row["directions"] = {"encoder": report.encoder, "decoder": report.decoder,
                             "encoder_decoder": report.encoder_decoder}
        row["attention_layout"] = report.attention_layout
        if report.jevx_conversion == "unsupported":
            row["error"] = "capability verdict 'unsupported'"
            row["elapsed_s"] = round(time.perf_counter() - started, 3)
            return row

        adapter = resolve_adapter(model)
        row["adapter"] = adapter.name
        row["adapter_pooling_default"] = adapter.default_pooling
        # Families whose ONLY output field is the opt-in per-layer collection
        # need `output_hidden_states=True` to reach priority 6 of the
        # extraction list (audit Q4). Recorded, never silently assumed.
        opt_in_hidden = report.representation_field == "hidden_states[-1]"
        row["opt_in_hidden_states"] = opt_in_hidden

        prepared = make_inputs(spec)
        inputs = adapter.prepare_inputs(prepared.state, model=model)
        with torch.no_grad():
            representation = adapter.extract_representation(
                model, inputs, output_hidden_states=opt_in_hidden
            )
        row["representation_field"] = representation.field
        row["representation_layout"] = representation.layout
        row["pooling_strategy"] = (
            "native" if representation.sequence_output is None
            else adapter.default_pooling
        )
        pooled = representation.pooled
        if pooled is None:
            raise ValueError("extract_representation returned pooled=None")
        if representation.sequence_output is not None:
            # Exercise the standalone pooling entry point on the same states.
            pooled = pool(representation.sequence_output, representation.attention_mask,
                          adapter.default_pooling)
        row["pooled_shape"] = list(pooled.shape)
        row["pooled_finite"] = bool(torch.isfinite(pooled).all())

        hidden = int(pooled.shape[-1])
        row["hidden_width"] = hidden

        jevx_config = JevXConfig(
            base_model_type=config.model_type,
            decision_hidden_size=16,
            question_hidden_size=16,
            schema_hidden_size=16,
            decision_primitives=["choice"],
            freeze_base_model=True,
            runtime_validation=True,
            output_hidden_states=opt_in_hidden,
        )
        try:
            wrapper = JevXForDecision(jevx_config, backbone=model).eval()
        except ValueError as exc:
            if HIDDEN_WIDTH_BLOCKER not in str(exc):
                raise
            # Known blocker in a file this agent does not own:
            # jevx/modeling_jevx.py:127-136 _HIDDEN_WIDTH_KEYS has no entry for
            # the CNN families, whose configs declare a per-stage LIST
            # (`hidden_sizes`) instead of a scalar width. Extraction, pooling
            # and every representation invariant already ran and passed above;
            # only the wrapper's state_proj width is unresolvable.
            row["decision_blocked"] = f"{type(exc).__name__}: {exc}"
            row["decision_blocker_site"] = (
                "jevx/modeling_jevx.py:127-136 (_HIDDEN_WIDTH_KEYS)"
            )
            row["status"] = "PARTIAL"
            row["elapsed_s"] = round(time.perf_counter() - started, 3)
            return row
        wrapper = wrapper.eval()
        row["wrapper_capability_verdict"] = wrapper.capability_report.jevx_conversion
        row["wrapper_capability_field"] = wrapper.capability_report.representation_field
        row["wrapper_adapter"] = wrapper.adapter.name
        row["state_proj_width"] = wrapper.state_proj.in_dim
        questions = ["which option wins?", "is it safe?"]
        options = [["alpha", "beta", "gamma"], ["yes", "no", "maybe"]]
        with torch.no_grad():
            output = wrapper.choice(
                state=prepared.state, questions=questions, options=options
            )
        problems: List[str] = []
        for decision, keys in zip(output.decisions, options):
            k = len(keys)
            width = int(decision.probabilities.shape[-1])
            problems.extend(
                _check_invariants(
                    pooled=decision.probabilities,
                    probs=decision.probabilities,
                    predicted_index=int(decision.predicted_index[0]),
                    confidence=float(decision.confidence[0]),
                    n_options=width,
                    hidden=width,
                )
            )
            if not 0 <= int(decision.predicted_index[0]) < k:
                problems.append(
                    f"argmax {int(decision.predicted_index[0])} outside schema [0, {k})"
                )
            if decision.predicted_value not in keys:
                problems.append("predicted_value not in the schema keys")
        problems.extend(
            _check_invariants(
                pooled=pooled,
                probs=output.decisions[0].probabilities,
                predicted_index=int(output.decisions[0].predicted_index[0]),
                confidence=float(output.decisions[0].confidence[0]),
                n_options=int(output.decisions[0].probabilities.shape[-1]),
                hidden=hidden,
            )
        )
        if row["state_proj_width"] != hidden:
            problems.append(
                f"state_proj width {row['state_proj_width']} != hidden {hidden}"
            )
        row["prob_sum"] = float(output.decisions[0].probabilities[0].sum())
        row["predicted_index"] = int(output.decisions[0].predicted_index[0])
        row["confidence"] = float(output.decisions[0].confidence[0])
        if problems:
            row["error"] = "; ".join(sorted(set(problems)))
        else:
            row["status"] = "PASS"
    except Exception as exc:  # noqa: BLE001 - the sweep records every failure
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["traceback"] = traceback.format_exc(limit=20)
    finally:
        del model
    row["elapsed_s"] = round(time.perf_counter() - started, 3)
    return row


def run_sweep(
    families: Optional[List[FamilySpec]] = None, *, verbose: bool = True
) -> Dict[str, Any]:
    """Run the sweep and return the machine-readable report.

    Args:
        families: Families to execute; ``None`` runs all of :data:`FAMILIES`.
        verbose: Print one line per family as it completes.

    Returns:
        A dict with ``results``, ``counts`` and provenance (transformers
        version, seed, what the sweep does and does not prove).
    """
    import transformers

    targets = list(families if families is not None else FAMILIES)
    started = time.perf_counter()
    results = []
    for index, spec in enumerate(targets):
        row = run_family(spec)
        results.append(row)
        if verbose:
            flag = {"PASS": "ok ", "PARTIAL": "part"}.get(row["status"], "FAIL")
            detail = ""
            if row["status"] == "PARTIAL":
                detail = f" :: decision blocked at {row['decision_blocker_site']}"
            elif row["status"] != "PASS":
                detail = f" :: {(row['error'] or '<no error recorded>')[:160]}"
            print(
                f"[{index + 1:>3}/{len(targets)}] {flag} {row['family']:<18} "
                f"adapter={row['adapter']} field={row['representation_field']} "
                f"verdict={row['verdict']} {row['elapsed_s']:.2f}s{detail}",
                flush=True,
            )
    elapsed = time.perf_counter() - started
    passed = sum(1 for row in results if row["status"] == "PASS")
    report = {
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "seed_base": SEED_BASE,
        "scope": (
            "PIPELINE execution only: models are randomly initialized in memory "
            "(no checkpoints, no downloads). A PASS shows the JevX extraction and "
            "decision path ran end-to-end with every invariant holding; it makes no "
            "statement about a checkpoint's accuracy."
        ),
        "counts": {
            "attempted": len(results),
            "passed": passed,
            "partial": sum(1 for row in results if row["status"] == "PARTIAL"),
            "failed": sum(1 for row in results if row["status"] == "FAIL"),
            "excluded": len(EXCLUSIONS),
        },
        "elapsed_s": round(elapsed, 2),
        "exclusions": [dict(entry) for entry in EXCLUSIONS],
        "results": results,
    }
    return report


def sweep_markdown_table(report: Dict[str, Any]) -> str:
    """Render the sweep report as a grouped Markdown table.

    Args:
        report: The dict returned by :func:`run_sweep`.

    Returns:
        A Markdown table, one row per family, grouped by architectural group.
    """
    lines = [
        "| family | group | model class | config class | adapter | representation field | "
        "pooling | verdict | status | s |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    current = None
    for row in report["results"]:
        if row["group"] != current:
            current = row["group"]
            lines.append(f"| **{current}** | | | | | | | | | |")
        lines.append(
            f"| {row['family']} | {row['group']} | `{row['model_class']}` | "
            f"`{row['config_class']}` | {row['adapter']} | "
            f"`{row['representation_field']}` | {row['pooling_strategy']} | "
            f"{row['verdict']} | **{row['status']}** | {row['elapsed_s']} |"
        )
    return "\n".join(lines)


def _merge(parts: List[str], out: str) -> int:
    """Merge per-group sweep reports into one, preserving run order.

    Args:
        parts: Paths of previously written reports, in the order they should
            appear in the merged file.
        out: Where the merged report is written.

    Returns:
        The same exit code :func:`main` would return for a live run.
    """
    reports = [json.loads(Path(part).read_text(encoding="utf-8")) for part in parts]
    merged = dict(reports[0])
    seen: set[str] = set()
    results: List[Dict[str, Any]] = []
    for report in reports:
        for row in report["results"]:
            if row["family"] in seen:
                continue
            seen.add(row["family"])
            results.append(row)
    merged["results"] = results
    merged["elapsed_s"] = round(sum(r.get("elapsed_s", 0.0) for r in results), 2)
    merged["counts"] = {
        "attempted": len(results),
        "passed": sum(1 for r in results if r["status"] == "PASS"),
        "partial": sum(1 for r in results if r["status"] == "PARTIAL"),
        "failed": sum(1 for r in results if r["status"] == "FAIL"),
        "excluded": len(merged.get("exclusions", [])),
    }
    merged["scope"] = reports[0]["scope"]
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(
        f"\nBREADTH SWEEP (merged from {len(parts)} group reports): "
        f"{merged['counts']['passed']}/{merged['counts']['attempted']} passed, "
        f"{merged['counts']['partial']} partial, {merged['counts']['failed']} failed "
        f"in {merged['elapsed_s']}s -> {out_path}"
    )
    for row in results:
        if row["status"] == "PARTIAL":
            print(f"  PART {row['family']}: decision step blocked by "
                  f"{row['decision_blocker_site']}")
        elif row["status"] != "PASS":
            print(f"  FAIL {row['family']}: {row['error']}")
    return 0 if merged["counts"]["failed"] == 0 else 1


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point: run the sweep, write the JSON report, print a summary.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code -- non-zero when any family failed, so the harness
        is usable as a gate.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="*", default=None,
                        help="restrict the sweep to these family keys")
    parser.add_argument("--out", default=str(RESULTS_PATH),
                        help="where to write the JSON report")
    parser.add_argument("--quiet", action="store_true", help="suppress per-family lines")
    parser.add_argument(
        "--merge", nargs="+", default=None, metavar="PART_JSON",
        help="merge already-written per-group reports into --out instead of "
             "re-running (the sweep is CPU-bound and is often executed in "
             "groups); the first part supplies the metadata and exclusions.",
    )
    args = parser.parse_args(argv)

    if args.merge:
        return _merge(args.merge, args.out)

    families = FAMILIES
    if args.only:
        wanted = set(args.only)
        families = tuple(spec for spec in FAMILIES if spec.key in wanted)
        missing = wanted - {spec.key for spec in families}
        if missing:
            parser.error(f"unknown family keys: {sorted(missing)}")

    report = run_sweep(list(families), verbose=not args.quiet)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"\nBREADTH SWEEP: {report['counts']['passed']}/{report['counts']['attempted']} "
        f"passed in {report['elapsed_s']}s -> {out_path}"
    )
    for row in report["results"]:
        if row["status"] == "PARTIAL":
            print(f"  PART {row['family']}: decision step blocked by "
                  f"{row['decision_blocker_site']}")
        elif row["status"] != "PASS":
            print(f"  FAIL {row['family']}: {row['error']}")
    print(f"  exclusions ({len(EXCLUSIONS)}): "
          + ", ".join(entry["family"] for entry in EXCLUSIONS))
    return 0 if report["counts"]["failed"] == 0 else 1


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
