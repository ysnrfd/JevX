# Copyright (c) 2026 ysnrfd
"""Modality quality smoke: synthetic image + waveform through JevX end-to-end.

Network-free (tiny in-memory ViT / Wav2Vec2 fixtures from tests/conftest.py +
in-memory transformers processors), seconds-fast, 12 samples per modality.

Scope: PIPELINE INVARIANTS ONLY - choice probabilities sum to 1 / stay in
[0,1] / pick a schema member, featurize-once agrees with model.choice, and
HeadTrainer + choice_loss runs on both modalities. No predictive-quality
claim of any kind: the fixtures carry random weights and the inputs are
synthetic. Quality numbers live in reports/MODALITY_BENCHMARKS.md.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from PIL import Image
from transformers import ViTImageProcessor, Wav2Vec2FeatureExtractor

from benchmarks.run_modality import (
    InvariantLog,
    check_decision_invariants,
    collect_logits_checked,
    featurize_payloads,
)
from jevx import JevX, JevXConfig
from benchmarks import bench_common as bc

N = 12
K = 3
BATCH = 4
QUESTION = "Which of the three synthetic categories applies?"
OPTIONS = ["alpha", "beta", "gamma"]


def _synthetic_images() -> list:
    rng = np.random.RandomState(0)
    return [
        Image.fromarray(rng.randint(0, 255, (32, 32, 3), dtype=np.uint8), "RGB")
        for _ in range(N)
    ]


def _synthetic_waves() -> list:
    rng = np.random.RandomState(1)
    t = np.arange(8000, dtype=np.float32) / 16000.0
    return [
        (0.3 * np.sin(2 * np.pi * (200.0 + 30.0 * i) * t)
         + 0.05 * rng.randn(len(t))).astype(np.float32)
        for i in range(N)
    ]


def _labels() -> list:
    return [i % K for i in range(N)]


def _exercise(model: JevX, payloads: list, make_state, feat_batch: int) -> None:
    """choice invariants -> featurize-once -> head training -> equivalence."""
    labels = _labels()

    # 1) full-path choice invariants (probs sum/bounds/schema membership)
    with torch.no_grad():
        out = model.choice(
            state=make_state(payloads),
            questions=[QUESTION],
            options=[OPTIONS],
        )
    assert len(out.decisions) == N
    check_decision_invariants(out, OPTIONS, tag="e2e")

    # 2) featurize-once (benchmark path) - shape sanity
    feats = featurize_payloads(model, payloads, make_state, batch_size=feat_batch)
    assert feats.shape[0] == N and feats.ndim == 2
    assert np.isfinite(feats).all()

    # 3) cached head vs full model: same modules, float tolerance only
    head = bc.FeaturizedJevX(model, QUESTION, OPTIONS)
    model.eval()
    with torch.no_grad():
        full = torch.cat(
            [d.logits for d in out.decisions], dim=0)
        cached = head({"state": [feats[i] for i in range(N)]})
    diff = float((full - cached).abs().max())
    assert diff < 1e-3, f"full vs featurized mismatch: {diff}"

    # 4) HeadTrainer + choice_loss on the modality features (Mode A: the
    #    backbone is frozen so only JevX heads are optimized)
    for p in model.model.parameters():
        p.requires_grad_(False)
    records = bc.build_records(feats, labels, QUESTION, OPTIONS)
    trainer, history = bc.train_heads(
        head, records, seed=0, lr=1e-3, epochs=1, batch_size=BATCH,
    )
    assert history and all(np.isfinite(h["loss"]) for h in history)

    # 5) every eval batch passes the invariants contract
    inv = InvariantLog()
    logits, got_labels = collect_logits_checked(
        head, records, batch_size=BATCH, num_classes=K, inv=inv, tag="eval")
    assert inv.n_batches == math.ceil(N / BATCH)
    assert inv.max_sum_dev <= 1e-4
    assert 0.0 <= inv.min_prob and inv.max_prob <= 1.0
    metrics = bc.metrics_from_logits(logits, got_labels)
    assert all(np.isfinite(v) for v in metrics.values())


def test_modality_choice_smoke_vision(tiny_vit) -> None:
    torch.manual_seed(0)
    processor = ViTImageProcessor(size={"height": 32, "width": 32})
    model = JevX(
        JevXConfig(decision_hidden_size=32, question_hidden_size=32,
                   schema_hidden_size=32),
        backbone=tiny_vit,
        processor=processor,
    )
    model.eval()
    images = _synthetic_images()
    _exercise(model, images, lambda chunk: {"images": chunk}, feat_batch=BATCH)


def test_modality_choice_smoke_audio(tiny_wav2vec2) -> None:
    torch.manual_seed(1)
    processor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16000, padding_value=0.0, return_tensors="pt",
    )
    model = JevX(
        JevXConfig(decision_hidden_size=32, question_hidden_size=32,
                   schema_hidden_size=32),
        backbone=tiny_wav2vec2,
        processor=processor,
    )
    model.eval()
    waves = _synthetic_waves()
    _exercise(model, waves, lambda chunk: {"audio": chunk}, feat_batch=BATCH)
