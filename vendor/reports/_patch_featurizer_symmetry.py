# Copyright (c) 2026 ysnrfd
"""Fix A: featurizer symmetry between wrap and reload paths.

The wrap path left ``wrapper.tokenizer = None`` for processor-based families
(CLIP/SigLIP/Whisper/LLaVA/wav2vec2) even though their processor embeds a
tokenizer, while the reload path attached ``AutoTokenizer(save_dir)`` whenever
tokenizer files existed (they do — ``processor.save_pretrained`` writes them).
Question encoding uses the tokenizer when present (projection/hashed path is
tokenizer-sensitive), so save→reload decisions drifted (measured: CLIP
6.785e-02, wav2vec2 1.739e-03) with identical weights. Expose the
processor's embedded tokenizer at wrap time so both paths encode questions
with the same tokenizer (real-model matrix, Phase 92).
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "jevx" / "modeling_jevx.py"

OLD = '''        if tokenizer is None and processor is None:
            try:
                processor = AutoFeatureExtractor.from_pretrained(
                    model_id, trust_remote_code=trust_remote_code
                )
            except Exception:  # noqa: BLE001 - last resort: no featurizer
                processor = None
        return tokenizer, processor'''

NEW = '''        if tokenizer is None and processor is None:
            try:
                processor = AutoFeatureExtractor.from_pretrained(
                    model_id, trust_remote_code=trust_remote_code
                )
            except Exception:  # noqa: BLE001 - last resort: no featurizer
                processor = None
        if tokenizer is None and processor is not None:
            # Symmetry with the reload path (_attach_reload_assets attaches
            # AutoTokenizer whenever tokenizer files were saved — and they
            # ARE saved: processor.save_pretrained writes them). Question and
            # schema encoding are tokenizer-sensitive, so a processor-only
            # wrap followed by a reload with an attached tokenizer produced
            # different decisions at identical weights (real-model matrix:
            # CLIP 6.8e-02, wav2vec2 1.7e-03 save/load drift).
            embedded = getattr(processor, "tokenizer", None)
            if embedded is not None:
                tokenizer = embedded
        return tokenizer, processor'''


def main() -> int:
    src = PATH.read_text(encoding="utf-8")
    if OLD not in src:
        print("pattern not found")
        return 1
    src = src.replace(OLD, NEW, 1)
    PATH.write_text(src, encoding="utf-8")
    print("featurizer symmetry fix applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
