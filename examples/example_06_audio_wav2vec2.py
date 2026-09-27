# Copyright (c) 2026 ysnrfd
"""Example 06 — audio (wav2vec2): decisions from a 1 s waveform.

What this shows
---------------
Wrapping an audio encoder (``hf-internal-testing/tiny-random-Wav2Vec2Model``,
audio family). The pipeline exercised here is the one validated in
reports/REAL_MODEL_VALIDATION.md: synthesize a 1 s / 16 kHz two-tone WAV with
numpy + soundfile -> READ THE FILE BACK with soundfile (a real .wav on disk,
not a synthetic tensor) -> pass it as a raw modality payload ``{"audio": ...}``
-> JevX routes it through the model's own ``Wav2Vec2Processor`` to
``input_values`` -> mask-aware ``masked_mean`` pooling over frames (the
family convention: wav2vec2's own sequence-classification head mean-pools
over time).

No torchaudio/librosa needed (they are absent on this host; numpy+soundfile
suffice — reports/REAL_MODEL_VALIDATION.md environment section).

Honesty note: ``tiny-random-Wav2Vec2Model`` carries RANDOM weights (pipeline
fixture). Printed numbers are real outputs with asserted invariants, not
predictive quality.

Run from the repo root:  python examples/example_06_audio_wav2vec2.py
Expected runtime: ~5-15 s (first run downloads the checkpoint; measured load
11.41 s — reports/REAL_MODEL_VALIDATION.md).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf
import torch

from jevx import JevX

MODEL_ID = "hf-internal-testing/tiny-random-Wav2Vec2Model"
SAMPLE_RATE = 16000


def write_tone_wav(path: Path) -> None:
    """Deterministic 1 s two-tone WAV (the recipe of the matrix fixture)."""
    t = np.arange(SAMPLE_RATE, dtype=np.float64) / SAMPLE_RATE  # exactly 1.0 s
    envelope = np.minimum(1.0, t / 0.05) * np.minimum(1.0, (1.0 - t) / 0.05)
    wave = 0.6 * np.sin(2 * np.pi * 440.0 * t) + 0.3 * np.sin(2 * np.pi * 880.0 * t)
    sf.write(str(path), (wave * envelope).astype(np.float32), SAMPLE_RATE, subtype="PCM_16")


def main() -> None:
    model = JevX.from_pretrained(MODEL_ID)
    model.eval()
    print(f"model: {MODEL_ID}")
    print(f"adapter: {model.adapter.name} | conversion: {model.capability_report.jevx_conversion}"
          f" | representation_field: {model.capability_report.representation_field}"
          f" | processor_required: {model.capability_report.processor_required}")

    # Real file on disk -> read back -> raw modality payload for the processor.
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "tone_1s.wav"
        write_tone_wav(wav_path)
        waveform, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        print(f"\nsynthesized wav: {wav_path.name} | sr={sr} | samples={waveform.shape[0]}")
    assert sr == SAMPLE_RATE and waveform.shape[0] == SAMPLE_RATE

    state = {"audio": waveform}
    questions = ["What does this audio contain?", "Is the signal clean?"]
    options = [["a 440 Hz tone with an 880 Hz overtone", "speech", "noise"],
               ["clean", "noisy"]]

    with torch.no_grad():
        out = model.choice(state, questions, options=options)

    print("\n[choice] decisions over one 1 s waveform")
    for q, opts, dec in zip(questions, options, out.decisions):
        probs = [round(float(p), 4) for p in dec.probabilities[0, : len(opts)]]
        print(f"  q={q!r}")
        print(f"    probs={dict(zip(opts, probs))}")
        print(f"    predicted={dec.predicted_value!r} confidence={float(dec.confidence[0]):.4f}")

    # --- invariants: running the example IS testing -------------------------
    assert model.capability_report.audio and not model.capability_report.text
    assert model.processor is not None, "audio wrap must resolve a processor/FE"
    assert len(out.decisions) == len(questions)
    for opts, dec in zip(options, out.decisions):
        probs = dec.probabilities[0]
        k = len(opts)
        assert torch.all(probs >= 0) and torch.all(probs <= 1)
        assert abs(float(probs[:k].sum()) - 1.0) < 1e-6
        assert 0 <= int(dec.predicted_index[0]) < k
        assert dec.predicted_value in opts
        assert 0.0 <= float(dec.confidence[0]) <= 1.0
    # score head over the same audio: expected score stays bounded by [0, K-1].
    with torch.no_grad():
        score_out = model.score(state, ["Rate the loudness."], options=[["1", "2", "3"]])
    es = float(score_out.decisions[0].expected_score[0])
    assert 0.0 <= es <= 2.0 + 1e-6, f"expected_score {es} outside [0, K-1]"
    print(f"  [score] Rate the loudness. -> expected_score={es:.4f} "
          f"confidence={float(score_out.decisions[0].confidence[0]):.4f}")
    print("\nAll invariants held (audio capability, processor resolved, probs sum to 1, "
          "schema membership, score expectation bounded).")


if __name__ == "__main__":
    main()
