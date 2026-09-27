# Copyright (c) 2026 ysnrfd
"""Fix: EncoderDecoderAdapter decoder-prompt mirror for non-token encoder
modalities (whisper: input_features).

The mirror only handled `input_ids` (shared-vocab t5/bart). Whisper's full
forward (used by base_forward and the raw-model comparison) needs
decoder_input_ids too, but its encoder input is a mel spectrogram — pass the
config's decoder_start_token_id as a BOS-style prompt, exactly what
WhisperForConditionalGeneration constructs internally (real-model matrix,
Phase 92: WhisperModel.forward raised "You cannot specify both
decoder_input_ids and decoder_inputs_embeds" because the decoder received
neither).
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "jevx" / "adapters" / "encoder_decoder.py"

OLD = '''        inputs = super().prepare_inputs(state, tokenizer=tokenizer, processor=processor, **kwargs)
        model = kwargs.get("model")
        if model is not None and "input_ids" in inputs and "decoder_input_ids" not in inputs:
            try:
                params = set(inspect.signature(model.forward).parameters)
            except (TypeError, ValueError):  # pragma: no cover - uninspectable
                params = set()
            if "decoder_input_ids" in params:
                inputs["decoder_input_ids"] = inputs["input_ids"]
        return inputs'''

NEW = '''        inputs = super().prepare_inputs(state, tokenizer=tokenizer, processor=processor, **kwargs)
        model = kwargs.get("model")
        if model is not None and "decoder_input_ids" not in inputs:
            try:
                params = set(inspect.signature(model.forward).parameters)
            except (TypeError, ValueError):  # pragma: no cover - uninspectable
                params = set()
            if "decoder_input_ids" in params:
                if "input_ids" in inputs:
                    # Shared vocabulary (t5/bart): mirror the encoder tokens,
                    # the standard encoder-decoder prompt.
                    inputs["decoder_input_ids"] = inputs["input_ids"]
                else:
                    # Encoder-side modality WITHOUT a shared vocabulary
                    # (whisper: input_features is a mel spectrogram): prompt
                    # the decoder with the config's decoder_start_token_id —
                    # what the ForConditionalGeneration heads construct
                    # internally before calling the base stack.
                    config = getattr(model, "config", None)
                    start = getattr(config, "decoder_start_token_id", None)
                    vocab = getattr(config, "vocab_size", None)
                    encoder_input = next(
                        (
                            inputs[key]
                            for key in ("input_features", "pixel_values")
                            if isinstance(inputs.get(key), torch.Tensor)
                            and inputs[key].dim() >= 1
                        ),
                        None,
                    )
                    if (
                        encoder_input is not None
                        and isinstance(start, int)
                        and start >= 0
                        and (not isinstance(vocab, int) or start < vocab)
                    ):
                        batch = int(encoder_input.shape[0])
                        inputs["decoder_input_ids"] = torch.full(
                            (batch, 1), int(start), dtype=torch.long
                        )
        return inputs'''


def main() -> int:
    src = PATH.read_text(encoding="utf-8")
    if OLD not in src:
        print("pattern not found")
        return 1
    src = src.replace(OLD, NEW, 1)
    PATH.write_text(src, encoding="utf-8")
    print("encoder-decoder decoder-prompt fix applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
