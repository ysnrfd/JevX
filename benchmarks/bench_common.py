# Copyright (c) 2026 ysnrfd
"""Shared machinery for the JevX real-dataset benchmarks (EXP-1..EXP-6).

This module contains the *reproducible core* exercised by
benchmarks/run_quality.py and by tests/unit/test_benchmark_smoke.py:

* deterministic stratified splitting (train / calibration / test, seed-fixed);
* one-pass featurization of raw text through a frozen HF backbone via the
  JevX adapter (native pooler or any registered pooling strategy);
* FeaturizedJevX - the JevX head stack (steps 2-8 of the forward path:
  state projection, question encoding, schema encoding, fusion, choice
  head) executed on PRECOMPUTED pooled features, so only JevX heads train
  while the backbone runs exactly once per split (Mode A, frozen);
* metric helpers (accuracy, macro-F1, NLL, Brier, ECE/MCE, reliability
  bins, confidence-ECE, AURC, coverage-at-risk) built on the same
  jevx.utils.numerical functions the library itself uses;
* latency and generative-baseline (B3) helpers.

Every number produced here comes from executed code; nothing is
hard-coded. Network downloads happen only inside load_* functions.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from jevx import (
    CalibrationEngine,
    ConfidenceEngine,
    JevX,
    coverage_at_risk,
    risk_coverage_curve,
)
from jevx.questions import Option
from jevx.representations.pooling import pool
from jevx.training import (
    HeadTrainer,
    accuracy,
    decision_collate_fn,
    macro_f1,
)
from jevx.utils.numerical import (
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    negative_log_likelihood,
    stable_softmax,
)

# --------------------------------------------------------------------------
# Provenance / environment
# --------------------------------------------------------------------------

DEFAULT_BACKBONE = "google/bert_uncased_L-2_H-128_A-2"
DEFAULT_DATASET = "stanfordnlp/sst2"
DEFAULT_DATASET_REVISION = "8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb"
DEFAULT_QUESTION = "Is the sentiment of this review positive or negative?"
#: Label order IS the option order: index 0 = negative (sst2 label 0),
#: index 1 = positive (sst2 label 1).
DEFAULT_OPTIONS = ["negative", "positive"]
GENERIC_BASELINE = "distilgpt2"  # B3 constrained-decoding backbone


def environment_info() -> Dict[str, Any]:
    """Machine + library facts every result file is stamped with (§87)."""
    import datasets
    import transformers

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "cpu_count": __import__("os").cpu_count(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "numpy": np.__version__,
        "torch_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "dtype": "float32",
    }


def provenance(
    backbone_id: str,
    dataset_id: str,
    dataset_revision: str,
    seeds: Sequence[int],
    hyperparams: Dict[str, Any],
) -> Dict[str, Any]:
    """Exact ids/revisions/seeds/hyperparams block required by §87."""
    return {
        "backbone_id": backbone_id,
        "backbone_revision": None,  # filled by caller after first load (config._commit_hash)
        "dataset_id": dataset_id,
        "dataset_revision": dataset_revision,
        "question": DEFAULT_QUESTION,
        "options": list(DEFAULT_OPTIONS),
        "seeds": list(seeds),
        "hyperparams": hyperparams,
        "environment": environment_info(),
    }


def write_json(path: Path | str, payload: Dict[str, Any]) -> None:
    """Write JSON via a .tmp sibling then replace, so a crash mid-write
    never leaves a truncated result file behind."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    tmp.replace(path)


# --------------------------------------------------------------------------
# Dataset + deterministic stratified splits (§52)
# --------------------------------------------------------------------------

def load_sst2(
    dataset_id: str = DEFAULT_DATASET, revision: str = DEFAULT_DATASET_REVISION
):
    """Load the sentiment dataset pinned to an exact revision.

    Returns (train_texts, train_labels, val_texts, val_labels): the official
    train split (pool for train/cal carving) and the official validation
    split (used as the never-tuned-on test set). Labels: 0 = negative,
    1 = positive.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_id, revision=revision)
    train = ds["train"]
    val = ds["validation"]
    return (
        list(train["sentence"]),
        [int(x) for x in train["label"]],
        list(val["sentence"]),
        [int(x) for x in val["label"]],
    )


def stratified_indices(labels: Sequence[int], n: int, seed: int) -> np.ndarray:
    """Deterministic stratified sample of n indices preserving class ratios.

    Allocation uses the largest-remainder method over the requested n, so
    per-class counts match the population proportions as closely as integer
    arithmetic allows. The RNG is a seeded numpy RandomState drawn class by
    class in sorted class order, so results are stable across platforms for
    a fixed seed.
    """
    labels = np.asarray(labels)
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if n > len(labels):
        raise ValueError(f"cannot sample {n} rows from {len(labels)}")
    classes, counts = np.unique(labels, return_counts=True)
    if n < len(classes):
        raise ValueError(f"n={n} smaller than the number of classes {len(classes)}")
    quotas_raw = n * counts / counts.sum()
    quotas = np.floor(quotas_raw).astype(int)
    order = np.argsort(-(quotas_raw - quotas))
    shortfall = int(n - quotas.sum())
    for i in range(shortfall):
        quotas[order[i % len(order)]] += 1
    rng = np.random.RandomState(seed)
    chosen: List[int] = []
    for cls, quota in zip(classes.tolist(), quotas.tolist()):
        pool_idx = np.flatnonzero(labels == cls)
        perm = rng.permutation(pool_idx)
        chosen.extend(perm[:quota].tolist())
    chosen_arr = np.asarray(chosen, dtype=np.int64)
    return chosen_arr[rng.permutation(len(chosen_arr))]


def make_splits(
    train_labels: Sequence[int],
    n_train: int,
    n_cal: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Disjoint (train, cal) index arrays carved from one pool, both
    stratified; cal is drawn from the rows left after train, so the two
    never overlap and both depend on the seed."""
    train_idx = stratified_indices(train_labels, n_train, seed)
    remaining_mask = np.ones(len(train_labels), dtype=bool)
    remaining_mask[train_idx] = False
    remaining_labels = np.asarray(train_labels)[remaining_mask]
    remaining_positions = np.flatnonzero(remaining_mask)
    cal_local = stratified_indices(remaining_labels, n_cal, seed + 1)
    cal_idx = remaining_positions[cal_local]
    assert len(np.intersect1d(train_idx, cal_idx)) == 0, "train/cal overlap"
    return train_idx, cal_idx


# --------------------------------------------------------------------------
# Featurization (backbone runs exactly once per split)
# --------------------------------------------------------------------------

def featurize(
    model: JevX,
    texts: Sequence[str],
    *,
    batch_size: int = 64,
    max_length: int = 128,
    strategies: Sequence[Optional[str]] = (None,),
) -> Dict[str, np.ndarray]:
    """Encode texts through the frozen backbone ONCE per batch.

    Returns a dict strategy-name -> float32 array [N, d_model]. One
    backbone call per batch yields the native pooled vector AND the
    sequence output; every other requested pooling is derived from that
    same output with jevx.representations.pooling.pool - the exact function
    the adapter uses - so all pooling ablations share a single forward and
    stay bit-compatible with model.choice() (checked by
    equivalence_check()).

    Strategy keys: "native" (pooler / adapter default) or any registered
    pooling name ("cls", "masked_mean", "last_valid", ...).
    """
    strategies = list(strategies)
    keys = ["native" if s is None else s for s in strategies]
    out: Dict[str, List[np.ndarray]] = {k: [] for k in keys}
    adapter = model.adapter
    backbone = model.model
    tok = model.tokenizer
    was_training = backbone.training
    backbone.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                chunk = list(texts[start : start + batch_size])
                enc = tok(
                    chunk,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                )
                inputs = adapter.prepare_inputs(dict(enc), tokenizer=tok, model=backbone)
                rep = adapter.extract_representation(backbone, inputs, strategy=None)
                seq, mask = rep.sequence_output, rep.attention_mask
                for strat, key in zip(strategies, keys):
                    if key == "native":
                        pooled = rep.pooled
                        if pooled is None:
                            pooled = pool(seq, mask, adapter.default_pooling)
                    else:
                        pooled = pool(seq, mask, strat)
                    out[key].append(pooled.detach().cpu().numpy().astype(np.float32))
    finally:
        if was_training:
            backbone.train()
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


def equivalence_check(
    model: JevX,
    head_model: "FeaturizedJevX",
    features: np.ndarray,
    texts: Sequence[str],
    *,
    n: int = 16,
) -> Dict[str, Any]:
    """Max-abs difference between full model.choice() logits and the
    featurized head-stack logits on the same n examples.

    Both paths share every trainable module; a nonzero difference can only
    come from float re-association between independently batched backbone
    passes. Recorded as an honesty artifact of the featurize-once design.
    """
    model.eval()
    n = min(n, len(texts))
    with torch.no_grad():
        out = model.choice(
            state=list(texts[:n]),
            questions=[head_model.question],
            options=[list(head_model.options)],
        )
        full = torch.cat([d.logits for d in out.decisions], dim=0)
        cached = head_model({"state": [features[i] for i in range(n)]})
    diff = (full - cached).abs().max().item()
    return {
        "n": n,
        "max_abs_diff": diff,
        "note": (
            "identical modules; nonzero diff only from float re-association "
            "between independent batched backbone passes"
        ),
    }


# --------------------------------------------------------------------------
# Featurized JevX head stack (Mode A: train heads only)
# --------------------------------------------------------------------------

class FeaturizedJevX(nn.Module):
    """JevX steps 2-8 on precomputed pooled features (single-question task).

    forward(batch) accepts a collated DecisionDataset batch (state rows are
    pooled feature vectors; questions/options/labels as usual) and returns
    raw choice logits [B, K]. Question and schema texts are FIXED for the
    benchmark (one question, one schema), so they are encoded through the
    model's real E_Q / E_C exactly as model.forward does (trainable hashed
    encoders re-run every step so their gradients flow), then expanded to
    the batch - mathematically identical to per-row encoding because both
    encoders are row-independent.

    The wrapped JevX model is registered as a submodule: HeadTrainer's
    default optimizer picks up every requires_grad parameter (the frozen
    backbone contributes none), so training this wrapper trains exactly
    model.trainable_parameters().
    """

    def __init__(
        self,
        jevx_model: JevX,
        question: str = DEFAULT_QUESTION,
        options: Sequence[str] = DEFAULT_OPTIONS,
        *,
        question_mode: str = "text",
    ) -> None:
        super().__init__()
        if question_mode not in ("text", "zero"):
            raise ValueError(
                f"question_mode: expected 'text' or 'zero', got {question_mode!r}"
            )
        self.jevx = jevx_model
        self.question = question
        self.options = list(options)
        self.question_mode = question_mode

    def _compute_question_rows(self, batch_size: int) -> torch.Tensor:
        m = self.jevx
        rows = m.question_encoder.encode_texts([self.question], tokenizer=m.tokenizer)
        rows = m.question_encoder(rows)
        rows = m.question_proj(rows)  # [1, D]
        if self.question_mode == "zero":
            rows = torch.zeros_like(rows)
        return rows.expand(batch_size, -1).contiguous()

    def _compute_schema(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        m = self.jevx
        schema_row = [Option(key=o, label=o) for o in self.options]
        feats, mask = m.schema_encoder.encode([schema_row], tokenizer=m.tokenizer)
        feats = m.schema_proj(feats)  # [1, K, D]
        feats = feats.expand(batch_size, -1, -1).contiguous()
        mask_b = mask.expand(batch_size, -1).contiguous() if mask is not None else None
        option_mask = (
            mask_b if (mask_b is not None and not bool(mask_b.all())) else None
        )
        return feats, option_mask

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        m = self.jevx
        state = batch["state"]
        if isinstance(state, np.ndarray):
            feats = torch.from_numpy(state)
        else:
            feats = torch.as_tensor(np.stack([np.asarray(r) for r in state]))
        feats = feats.to(dtype=torch.float32)
        batch_size = feats.shape[0]
        state_repr = m.state_proj(feats)  # [B, D]
        question_rows = self._compute_question_rows(batch_size)
        option_features, option_mask = self._compute_schema(batch_size)
        fused = m.fusion(state_repr, question_rows, option_features, option_mask)
        out = m.heads["choice"](
            fused.fused, fused.option_features, option_mask=option_mask
        )
        return out.logits  # [B, K]


# --------------------------------------------------------------------------
# Baselines B1 / B2 (same features, same protocol)
# --------------------------------------------------------------------------

class LinearProbe(nn.Module):
    """B1: single Linear(d, K) over the frozen pooled features."""

    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.head = nn.Linear(in_dim, num_classes)

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        state = batch["state"]
        if isinstance(state, np.ndarray):
            x = torch.from_numpy(state)
        else:
            x = torch.as_tensor(np.stack([np.asarray(r) for r in state]))
        return self.head(x.to(dtype=torch.float32))


class MLPProbe(nn.Module):
    """B2: Linear(d, d) -> ReLU -> Linear(d, K) over the same features."""

    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, num_classes)
        )

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        state = batch["state"]
        if isinstance(state, np.ndarray):
            x = torch.from_numpy(state)
        else:
            x = torch.as_tensor(np.stack([np.asarray(r) for r in state]))
        return self.trunk(x.to(dtype=torch.float32))


# --------------------------------------------------------------------------
# Records / training / evaluation plumbing
# --------------------------------------------------------------------------

def build_records(
    features: np.ndarray,
    labels: Sequence[int],
    question: str = DEFAULT_QUESTION,
    options: Sequence[str] = DEFAULT_OPTIONS,
) -> List[Dict[str, Any]]:
    """DecisionDataset records with state = pooled feature row."""
    labels = list(labels)
    if len(features) != len(labels):
        raise ValueError(
            f"features/labels length mismatch: {len(features)} vs {len(labels)}"
        )
    return [
        {
            "state": np.ascontiguousarray(features[i], dtype=np.float32),
            "questions": [question],
            "decision_type": "choice",
            "options": [list(options)],
            "labels": [int(labels[i])],
        }
        for i in range(len(labels))
    ]


def make_loader(
    records: List[Dict[str, Any]],
    batch_size: int,
    *,
    shuffle: bool,
    generator: Optional[torch.Generator] = None,
) -> DataLoader:
    return DataLoader(
        records,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=decision_collate_fn,
        generator=generator,
    )


def train_heads(
    module: nn.Module,
    train_records: List[Dict[str, Any]],
    *,
    seed: int,
    lr: float,
    epochs: int,
    batch_size: int,
    eval_records: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[HeadTrainer, List[Dict[str, Any]]]:
    """HeadTrainer + choice_loss protocol shared by JevX, B1 and B2."""
    torch.manual_seed(seed)
    trainer = HeadTrainer(module, lr=lr, seed=seed)
    train_loader = make_loader(
        train_records, batch_size, shuffle=True, generator=trainer.generator
    )
    eval_loader = None
    if eval_records:
        eval_loader = make_loader(eval_records, batch_size, shuffle=False)
    history = trainer.fit(train_loader, epochs=epochs, eval_loader=eval_loader)
    return trainer, list(history.epochs)


def collect_logits(
    module: nn.Module,
    records: List[Dict[str, Any]],
    *,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Forward a loader once; return (logits [N,K] float32, labels [N] int64)."""
    module.eval()
    loader = make_loader(records, batch_size, shuffle=False)
    logits_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            logits = module(batch)
            logits_all.append(logits.detach().cpu().numpy().astype(np.float32))
            labels_all.append(np.asarray(batch["labels"]).reshape(-1).astype(np.int64))
    return np.concatenate(logits_all, axis=0), np.concatenate(labels_all, axis=0)


# --------------------------------------------------------------------------
# Metrics (all from executed jevx code paths)
# --------------------------------------------------------------------------

def metrics_from_logits(
    logits: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    *,
    n_bins: int = 15,
) -> Dict[str, float]:
    """accuracy / macro_f1 / nll / brier / ece / mce for raw (or already
    calibrated) logits. ECE and MCE use n_bins equal-width confidence bins
    (benchmark default 15; the library default of 10 is overridden here).
    """
    logits_t = torch.as_tensor(
        logits if isinstance(logits, torch.Tensor) else np.asarray(logits)
    )
    labels_t = torch.as_tensor(
        labels if isinstance(labels, torch.Tensor) else np.asarray(labels)
    )
    probs = stable_softmax(logits_t, dim=-1)
    return {
        "accuracy": accuracy(logits_t, labels_t),
        "macro_f1": macro_f1(logits_t, labels_t),
        "nll": negative_log_likelihood(probs, labels_t),
        "brier": brier_score(probs, labels_t),
        "ece": expected_calibration_error(probs, labels_t, n_bins=n_bins),
        "mce": maximum_calibration_error(probs, labels_t, n_bins=n_bins),
    }


def calibrated_metrics(
    test_logits_raw: np.ndarray,
    test_labels: np.ndarray,
    method: str,
    *,
    n_bins: int = 15,
    fit_logits: Optional[np.ndarray] = None,
    fit_labels: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Fit calibration METHOD on the CALIBRATION split, apply to the TEST
    logits, score against the TEST labels - the same six-metric dict as
    metrics_from_logits.

    For method != "none", fit_logits/fit_labels are mandatory: fitting on
    test would leak (§53) and is rejected.
    """
    engine = CalibrationEngine(method)
    raw = torch.as_tensor(np.asarray(test_logits_raw, dtype=np.float32))
    test_lab = torch.as_tensor(np.asarray(test_labels))
    if method != "none":
        if fit_logits is None or fit_labels is None:
            raise ValueError(
                "calibrated_metrics: method != 'none' requires fit_logits/fit_labels "
                "(the calibration split - never the test split)"
            )
        engine.fit(
            torch.as_tensor(np.asarray(fit_logits, dtype=np.float32)),
            torch.as_tensor(np.asarray(fit_labels)),
        )
    cal = engine.calibrate(raw, "logits")
    return metrics_from_logits(cal, test_lab, n_bins=n_bins)


def reliability_bins(
    logits: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    *,
    n_bins: int = 15,
) -> List[Dict[str, Any]]:
    """Reliability diagram table: equal-width confidence bins with counts,
    mean confidence, accuracy and the signed gap (acc - conf). Empty bins
    are kept (count 0, null statistics) so the table is complete.
    """
    logits_t = torch.as_tensor(
        logits if isinstance(logits, torch.Tensor) else np.asarray(logits)
    )
    labels_t = torch.as_tensor(
        labels if isinstance(labels, torch.Tensor) else np.asarray(labels)
    )
    probs = stable_softmax(logits_t, dim=-1)
    conf, pred = probs.max(dim=-1)
    correct = (pred == labels_t.long()).double()
    bins = (conf * n_bins).long().clamp(max=n_bins - 1)
    table: List[Dict[str, Any]] = []
    for b in range(n_bins):
        sel = bins == b
        count = int(sel.sum())
        lower, upper = b / n_bins, (b + 1) / n_bins
        if count == 0:
            table.append(
                {"bin": b, "lower": lower, "upper": upper, "count": 0,
                 "avg_confidence": None, "accuracy": None, "gap": None}
            )
            continue
        avg_conf = float(conf[sel].double().mean())
        acc = float(correct[sel].mean())
        table.append(
            {"bin": b, "lower": lower, "upper": upper, "count": count,
             "avg_confidence": avg_conf, "accuracy": acc, "gap": acc - avg_conf}
        )
    return table


def ece_of_confidence(
    confidence: np.ndarray | torch.Tensor,
    correct: np.ndarray | torch.Tensor,
    *,
    n_bins: int = 15,
) -> float:
    """ECE of an arbitrary per-example confidence: equal-width bins over
    [0,1], sum_b (n_b/N) * |acc_b - conf_b|. For confidence = max prob and
    correct = 1[argmax == y] this equals expected_calibration_error on the
    probabilities (asserted in the smoke test).
    """
    conf = torch.as_tensor(
        confidence if isinstance(confidence, torch.Tensor) else np.asarray(confidence)
    ).double().reshape(-1)
    corr = torch.as_tensor(
        correct if isinstance(correct, torch.Tensor) else np.asarray(correct)
    ).double().reshape(-1)
    if conf.shape[0] != corr.shape[0]:
        raise ValueError(
            f"ece_of_confidence: length mismatch {conf.shape[0]} vs {corr.shape[0]}"
        )
    if conf.numel() == 0:
        raise ValueError("ece_of_confidence: empty inputs")
    bins = (conf * n_bins).long().clamp(max=n_bins - 1)
    total = conf.shape[0]
    ece = 0.0
    for b in range(n_bins):
        sel = bins == b
        count = int(sel.sum())
        if count == 0:
            continue
        gap = abs(float(corr[sel].mean()) - float(conf[sel].mean()))
        ece += (count / total) * gap
    return float(ece)


def aurc(confidence: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve: mean of the empirical risk over
    the n coverage points 1/n .. 1 (equivalently the standard discrete
    AURC (1/n) * sum_i risk_i).
    """
    coverage, risk = risk_coverage_curve(confidence, correct)
    if risk.numel() == 0:
        raise ValueError("aurc: empty inputs")
    return float(risk.mean())


def coverage_at(confidence: np.ndarray, correct: np.ndarray, max_risk: float) -> float:
    """Wrapper over jevx.coverage_at_risk (max coverage at a risk budget)."""
    return float(coverage_at_risk(confidence, correct, max_risk))


def confidence_vectors(
    logits: np.ndarray, method: str, *, cal_logits: Optional[np.ndarray] = None
) -> np.ndarray:
    """Per-example confidence from a ConfidenceEngine method.

    max_probability / margin / entropy need only the probabilities; energy
    is fitted on the CALIBRATION logits (never test) then applied to the
    test logits.
    """
    logits_t = torch.as_tensor(np.asarray(logits, dtype=np.float32))
    probs = stable_softmax(logits_t, dim=-1)
    engine = ConfidenceEngine(method)
    if method == "energy":
        if cal_logits is None:
            raise ValueError(
                "confidence_vectors: method='energy' requires cal_logits (fit split)"
            )
        engine.fit(torch.as_tensor(np.asarray(cal_logits, dtype=np.float32)))
    conf = engine(probs, logits=logits_t)
    return conf.detach().cpu().numpy().astype(np.float64)


# --------------------------------------------------------------------------
# Latency (batch 1, warmup then timed iterations)
# --------------------------------------------------------------------------

def measure_latency(
    fn: Callable[[], Any],
    *,
    warmup: int = 10,
    iters: int = 100,
) -> Dict[str, float]:
    """Mean/std/min/max latency in milliseconds for one batch-1 call,
    after warmup untimed calls. Concurrent-agent load makes wall-clock
    noisy, so std is reported alongside the mean.
    """
    for _ in range(warmup):
        fn()
    samples: List[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(samples)
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "warmup": warmup,
        "iters": iters,
    }


# --------------------------------------------------------------------------
# B3: constrained decoding with a real pretrained generative LM
# --------------------------------------------------------------------------

def load_generative(model_id: str = GENERIC_BASELINE):
    """Load a real pretrained causal LM + tokenizer for B3."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()
    return model, tok


def generative_choice_logits(
    model: nn.Module,
    tok: Any,
    texts: Sequence[str],
    *,
    options: Sequence[str] = DEFAULT_OPTIONS,
    batch_size: int = 8,
    max_length: int = 128,
) -> np.ndarray:
    """Constrained-decoding logits [N, K] for a fixed template.

    Template: "Review: {text}\\nSentiment:" and the next token is forced
    into one of the option continuations (" negative" / " positive" for the
    default options - leading space included so each continuation is a
    single GPT-2 BPE token). We read the model's next-token logits at the
    last real position and KEEP ONLY the K option-token logits (raw, not
    normalized), so the shared metric/calibration pipeline applies
    unchanged.

    Raises ValueError if any option continuation is not exactly one token -
    multi-token continuations need a different scoring rule and are
    deliberately not silently approximated.
    """
    continuations = [" " + o for o in options]
    token_ids: List[int] = []
    for cont in continuations:
        ids = tok.encode(cont, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                f"generative_choice_logits: option continuation {cont!r} tokenizes to "
                f"{len(ids)} tokens; B3 supports single-token continuations only"
            )
        token_ids.append(ids[0])
    rows: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            chunk = [
                f"Review: {t}\nSentiment:" for t in texts[start : start + batch_size]
            ]
            enc = tok(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            out = model(**enc)
            # right-padded -> last real position per row
            last_idx = enc["attention_mask"].sum(dim=1) - 1
            gather = torch.arange(len(chunk))
            last_logits = out.logits[gather, last_idx, :]  # [B, V]
            rows.append(
                last_logits[:, token_ids].detach().cpu().numpy().astype(np.float32)
            )
    return np.concatenate(rows, axis=0)


# --------------------------------------------------------------------------
# Small aggregation helpers
# --------------------------------------------------------------------------

def mean_std_over_seeds(
    per_seed: List[Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """{metric: {"mean": ..., "std": ...}} over a list of metric dicts."""
    if not per_seed:
        raise ValueError("mean_std_over_seeds: empty input")
    keys = per_seed[0].keys()
    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        vals = np.asarray([row[k] for row in per_seed], dtype=np.float64)
        out[k] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        }
    return out


def count_trainable(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


__all__ = [
    "DEFAULT_BACKBONE",
    "DEFAULT_DATASET",
    "DEFAULT_DATASET_REVISION",
    "DEFAULT_OPTIONS",
    "DEFAULT_QUESTION",
    "GENERIC_BASELINE",
    "FeaturizedJevX",
    "LinearProbe",
    "MLPProbe",
    "aurc",
    "build_records",
    "calibrated_metrics",
    "collect_logits",
    "confidence_vectors",
    "count_trainable",
    "coverage_at",
    "ece_of_confidence",
    "environment_info",
    "equivalence_check",
    "featurize",
    "generative_choice_logits",
    "load_generative",
    "load_sst2",
    "make_loader",
    "make_splits",
    "measure_latency",
    "mean_std_over_seeds",
    "metrics_from_logits",
    "provenance",
    "reliability_bins",
    "stratified_indices",
    "train_heads",
    "write_json",
]
