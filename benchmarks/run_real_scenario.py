# Copyright (c) 2026 ysnrfd
"""REAL SCENARIO VALIDATION: one coherent, user-level run of JevX on a real
5-class ORDINAL rating dataset, with three decision types answered in ONE
`decide()` call over ONE state batch and ONE backbone forward.

The experiment is deliberately a *scenario*, not a sweep: the script follows
the path a real user follows (load a pinned dataset -> stratified splits ->
featurize once through a frozen backbone -> train three heads on those
features -> calibrate on a held-out split -> ONE multi-question decide()
-> correctness battery -> quality vs baselines -> risk-coverage -> print real
decisions), and records every measured number into
`benchmarks/results/real_scenario.json` incrementally, phase by phase.

* dataset   : `Yelp/yelp_review_full` (Hub API sha c1f9ee939b7d05667af864ee1cb066393154bf85,
              parquet) - ClassLabel names ARE the five star levels, so every
              question's ground truth is a REAL dataset label.
              Q1 choice (K=5)  <- label y in 0..4  (= 1..5 stars)
              Q2 score  (0..4) <- label y in 0..4  (same stars, ordinal)
              Q3 noul         <- label 1[y == 0] (1-star detractor)
* backbone  : `google/bert_uncased_L-2_H-128_A-2` (sha
              30b0a37ccaaa32f332884b96992754e246e48c5f), frozen (Mode A),
              pooling=cls, fusion=concat, featurized ONCE per split.
* splits    : train + calibration are a disjoint stratified carve from the
              official train split (`bc.make_splits`, split seed 0); test is
              the first `--test-size` rows of the OFFICIAL test split and is
              never used for fitting, calibration, or model selection.
* calibration is fit on the CALIBRATION split only; the test split is scored.

Nothing in the output is hard-coded: every number is produced by executed code
and is written to the JSON as it is measured. If a phase raises, the failure is
recorded in the JSON and the run continues with the remaining phases.

Usage (the normal, single-command repro):

    python benchmarks/run_real_scenario.py

Useful flags: `--phases 1,2` (run a subset), `--seeds 0` (single seed),
`--quick` (tiny sizes for a smoke run), `--out` (result path).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import bench_common as bc  # noqa: E402
from jevx import CalibrationEngine, JevX, coverage_at_risk, risk_coverage_curve  # noqa: E402
from jevx.questions import Option  # noqa: E402
from jevx.training import (  # noqa: E402
    HeadTrainer,
    accuracy,
    decision_collate_fn,
    make_supervised_step_fn,
    macro_f1,
)
from jevx.utils.numerical import (  # noqa: E402
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    negative_log_likelihood,
    stable_softmax,
)

# --------------------------------------------------------------------------
# Provenance constants (ids verified with a Hub API GET before download)
# --------------------------------------------------------------------------

DATASET_ID = "Yelp/yelp_review_full"
#: Hub API `sha` of Yelp/yelp_review_full (GET /api/datasets/Yelp/yelp_review_full,
#: tag "format:parquet" -> loadable by datasets 4.4.1 without a loading script).
DATASET_REVISION = "c1f9ee939b7d05667af864ee1cb066393154bf85"
#: Recorded from the Hub API tags verbatim, not inferred.
DATASET_LICENSE = "other (HF API tag license:other)"
BACKBONE_ID = "google/bert_uncased_L-2_H-128_A-2"
BACKBONE_SHA = "30b0a37ccaaa32f332884b96992754e246e48c5f"

TRAIN_SIZE = 4000
CAL_SIZE = 1200
TEST_SIZE = 1500
SPLIT_SEED = 0
MAX_LENGTH = 128
BATCH_SIZE = 64
EPOCHS = 4
LR = 1e-3
N_BINS = 15
THRESHOLD = 0.9
#: How many test states the batched-vs-solo equality check walks one at a time.
SOLO_STATES = 128
#: How many test states the save/load and runtime-validation checks replay.
SMALL_STATES = 64
#: How many real decisions phase 8 prints.
N_PRINTED = 10
#: Text truncation for the printed decisions.
PRINT_CHARS = 120

TYPES: Tuple[str, str, str] = ("choice", "score", "noul")

QUESTIONS: Dict[str, str] = {
    "choice": "What star rating will this reviewer give?",
    "score": "On a 0-to-4 scale, how many stars does this review earn?",
    "noul": "Is this a 1-star (detractor) review?",
}
#: Q1 options ARE the dataset's own ClassLabel names, in ClassLabel order:
#: index 0 = "1 star" ... index 4 = "5 stars" -> the class index IS the label.
CHOICE_OPTIONS: List[str] = ["1 star", "2 star", "3 stars", "4 stars", "5 stars"]
#: Q2 levels 0..4 are the same five stars re-indexed to a 0-based scale.
SCORE_LEVELS: List[str] = ["0", "1", "2", "3", "4"]
OPTION_SETS: Dict[str, Optional[List[str]]] = {
    "choice": CHOICE_OPTIONS,
    "score": SCORE_LEVELS,
    "noul": None,
}


#: The run log is a plain console transcript written OUTSIDE the repository
#: (OS temp dir) so the only files this script creates in the repo are its
#: result documents.
LOG_PATH = Path(os.environ.get("JEVX_REAL_SCENARIO_LOG", "")) if os.environ.get(
    "JEVX_REAL_SCENARIO_LOG"
) else Path(tempfile.gettempdir()) / "jevx_real_scenario.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')} pid={os.getpid()}] {msg}"
    print(line, flush=True)
    try:
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError:
        pass


def log_block(text: str) -> None:
    """Print a multi-line block (the phase-8 decision dump) and tee it."""
    print(text, flush=True)
    try:
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")
    except OSError:
        pass



# --------------------------------------------------------------------------
# CLI + incremental result file
# --------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default=str(REPO_ROOT / "benchmarks" / "results" / "real_scenario.json"))
    p.add_argument("--phases", default="1,2,3,4,5,6,7,8",
                   help="comma separated subset of phase ids to run")
    p.add_argument("--seeds", default="0,1,2", help="training seeds")
    p.add_argument("--train-size", type=int, default=TRAIN_SIZE)
    p.add_argument("--cal-size", type=int, default=CAL_SIZE)
    p.add_argument("--test-size", type=int, default=TEST_SIZE)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--max-length", type=int, default=MAX_LENGTH)
    p.add_argument("--pooling", default="cls")
    p.add_argument("--fusion", default="concat")
    p.add_argument("--threshold", type=float, default=THRESHOLD)
    p.add_argument("--quick", action="store_true", help="tiny smoke sizes")
    p.add_argument("--fresh-log", action="store_true",
                   help=f"truncate the run log at {LOG_PATH} before starting")
    p.add_argument("--keep-going", action="store_true", default=True,
                   help="record a phase failure and continue (default)")
    return p.parse_args(argv)


class Result:
    """The single result document, rewritten atomically after every phase so a
    crash or a kill never leaves a truncated file behind."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if self.path.exists():
            try:
                self.data: Dict[str, Any] = json.loads(self.path.read_text("utf-8"))
            except json.JSONDecodeError:
                self.data = {}
        else:
            self.data = {}
        self.data.setdefault("experiment", "real_scenario")
        self.data.setdefault(
            "description",
            "one coherent real scenario: 3 decision types (choice/score/noul) over "
            "ONE state batch and ONE backbone forward on Yelp/yelp_review_full",
        )
        self.data.setdefault("phases", {})
        self.data.setdefault("failures", {})
        self.data.setdefault("timings_s", {})

    def set(self, phase: str, payload: Dict[str, Any]) -> None:
        self.data["phases"][phase] = payload
        self.flush()

    def time(self, key: str, seconds: float) -> None:
        self.data["timings_s"][key] = round(float(seconds), 2)
        self.flush()

    def fail(self, phase: str, exc: BaseException) -> None:
        self.data["failures"][phase] = f"{type(exc).__name__}: {exc}"
        self.flush()

    def flush(self) -> None:
        bc.write_json(self.path, self.data)


# --------------------------------------------------------------------------
# Metrics used by the scenario (rank statistics implemented here directly;
# no sklearn anywhere in the pipeline)
# --------------------------------------------------------------------------

def auc_rank_score(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC as the Mann-Whitney U statistic: (sum of positive ranks - n_pos*(n_pos+1)/2)
    / (n_pos * n_neg), with mid-ranks for ties. Returns nan when a class is absent.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    if scores.shape[0] != labels.shape[0]:
        raise ValueError(f"auc_rank_score: length mismatch {scores.shape[0]} vs {labels.shape[0]}")
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    y_sorted = labels[order]
    # mid-ranks, ASCENDING (rank 1 == smallest score), ties averaged
    r = np.arange(1, s_sorted.shape[0] + 1, dtype=np.float64)
    i = 0
    while i < s_sorted.shape[0]:
        j = i
        while j + 1 < s_sorted.shape[0] and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            r[i : j + 1] = r[i : j + 1].mean()
        i = j + 1
    # r and y_sorted share the SORTED index space — pairing them is the point.
    rank_sum_pos = float(r[y_sorted == 1].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def auc_rank_self_check() -> Dict[str, Any]:
    """Closed-form cases for auc_rank_score, recorded in the JSON so the
    rank statistic can be checked without rerunning the pipeline."""
    cases = []
    perfect = auc_rank_score(np.array([0.1, 0.2, 0.3, 0.8, 0.9]), np.array([0, 0, 0, 1, 1]))
    cases.append({"case": "perfectly separated", "got": perfect, "expected": 1.0})
    reversed_ = auc_rank_score(
        np.array([-0.1, -0.2, -0.3, -0.8, -0.9]), np.array([0, 0, 0, 1, 1])
    )
    cases.append({"case": "fully reversed", "got": reversed_, "expected": 0.0})
    tied = auc_rank_score(np.ones(10), np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]))
    cases.append({"case": "all scores tied", "got": tied, "expected": 0.5})
    # 6 positives, 6 negatives, 21 of 36 positive-over-negative pairs ordered
    s6 = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
    y6 = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    cases.append({"case": "disjoint blocks (36/36)", "got": auc_rank_score(s6, y6),
                  "expected": 1.0})
    inter = np.array([0.0, 1.0, 2.0, 3.0])
    inter_y = np.array([0, 0, 1, 1])
    cases.append({"case": "y=[0,0,1,1] on ascending scores (4/4 pairs)",
                  "got": auc_rank_score(inter, inter_y), "expected": 1.0})
    return {"cases": cases, "all_passed": all(c["got"] == c["expected"] for c in cases)}


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom == 0.0:
        return float("nan")
    return float((a * b).sum() / denom)


def score_metrics(expected: np.ndarray, truth: np.ndarray) -> Dict[str, Any]:
    """MAE / RMSE / Pearson r of an expected ordinal score, plus the
    mean |E[s] - s| broken down by TRUE level (the ordinal-error table)."""
    e = np.asarray(expected, dtype=np.float64).reshape(-1)
    y = np.asarray(truth, dtype=np.float64).reshape(-1)
    err = e - y
    by_level = []
    for level in sorted(set(y.astype(int).tolist())):
        sel = y.astype(int) == level
        by_level.append(
            {
                "true_level": int(level),
                "n": int(sel.sum()),
                "mean_abs_error": float(np.abs(err[sel]).mean()),
                "rmse": float(np.sqrt((err[sel] ** 2).mean())),
                "mean_expected": float(e[sel].mean()),
            }
        )
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "pearson_r": pearson_r(e, y),
        "by_true_level": by_level,
    }


def binary_metrics(logits: np.ndarray, labels: np.ndarray, n_bins: int = N_BINS) -> Dict[str, Any]:
    """acc / auc / nll / ece / mce for a [n] or [n,1] logit vector (P(yes) = sigmoid)."""
    lg = torch.as_tensor(np.asarray(logits, dtype=np.float32)).reshape(-1)
    yb = torch.as_tensor(np.asarray(labels)).reshape(-1).long()
    p = torch.sigmoid(lg)
    probs2 = torch.stack([1.0 - p, p], dim=1)
    return {
        "accuracy": accuracy(probs2, yb),
        "auc": auc_rank_score(p.detach().cpu().numpy(), yb.numpy()),
        "nll": negative_log_likelihood(probs2, yb),
        "ece": expected_calibration_error(probs2, yb, n_bins=n_bins),
        "mce": maximum_calibration_error(probs2, yb, n_bins=n_bins),
        "positive_rate": float(p.mean()),
    }


def trunc(text: str, limit: int = PRINT_CHARS) -> str:
    t = " ".join(str(text).split())
    return t if len(t) <= limit else t[: limit - 1] + "…"



# --------------------------------------------------------------------------
# Featurized multi-question head stack (Mode A: the backbone runs once)
# --------------------------------------------------------------------------

class FeaturizedJevXMulti(nn.Module):
    r"""The JevX head stack (forward steps 2-8) for ALL m questions at once, on
    PRECOMPUTED pooled features.

    One \`JevXForDecision\` holds one \`state_proj\`, one \`question_encoder\` /
    \`question_proj\`, one \`schema_encoder\` / \`schema_proj\`, one \`fusion\` and
    the three registered heads; this wrapper routes a collated batch to the
    (question, decision type) pair it carries, using exactly the same modules
    and the same call order as \`JevXForDecision.forward\`. The question and
    schema texts are FIXED for the scenario, and both encoders are
    row-independent, so encoding one row and expanding it over the batch is
    mathematically identical to encoding every row — the same argument
    \`bc.FeaturizedJevX\` makes for the single-question case.

    \`forward(batch)\` reads \`batch["decision_type"]\` (a batch must be uniform;
    \`make_supervised_step_fn\` enforces that) and returns the raw \`[B, K]\`
    logits of that question's head, so the stock loss dispatch applies.
    """

    def __init__(
        self,
        jevx_model: JevX,
        types: Sequence[str] = TYPES,
        questions: Optional[Dict[str, str]] = None,
        option_sets: Optional[Dict[str, Optional[List[str]]]] = None,
    ) -> None:
        super().__init__()
        self.jevx = jevx_model
        self.types = list(types)
        self.questions = dict(questions or QUESTIONS)
        self.option_sets = dict(option_sets or OPTION_SETS)
        self.question_index = {t: i for i, t in enumerate(self.types)}

    # -- per-question encodings (recomputed every step so gradients flow) ----
    def _question_rows(self, qtype: str, batch_size: int) -> torch.Tensor:
        m = self.jevx
        rows = m.question_encoder.encode_texts(
            [self.questions[qtype]], tokenizer=m.tokenizer
        )
        rows = m.question_encoder(rows)
        rows = m.question_proj(rows)  # [1, D]
        return rows.expand(batch_size, -1).contiguous()

    def _schema(
        self, qtype: str, batch_size: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        m = self.jevx
        options = self.option_sets.get(qtype)
        if not options:
            return None, None
        schema_row = [[Option(key=o, label=o) for o in options]]
        feats, mask = m.schema_encoder.encode(schema_row, tokenizer=m.tokenizer)
        feats = m.schema_proj(feats)  # [1, K, D]
        mask_b = mask.expand(batch_size, -1).contiguous() if mask is not None else None
        option_mask = mask_b if (mask_b is not None and not bool(mask_b.all())) else None
        return feats.expand(batch_size, -1, -1).contiguous(), option_mask

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        m = self.jevx
        state = batch["state"]
        if isinstance(state, np.ndarray):
            feats = torch.from_numpy(state)
        else:
            feats = torch.as_tensor(np.stack([np.asarray(r) for r in state]))
        feats = feats.to(dtype=torch.float32)
        batch_size = feats.shape[0]
        qtype = batch["decision_type"]
        if isinstance(qtype, (list, tuple)):
            qtype = qtype[0]
        state_repr = m.state_proj(feats)  # [B, D]
        question_rows = self._question_rows(qtype, batch_size)
        option_features, option_mask = self._schema(qtype, batch_size)
        fused = m.fusion(state_repr, question_rows, option_features, option_mask)
        out = m.heads[qtype](
            fused.fused, fused.option_features, option_mask=option_mask
        )
        return out.logits


class JointBatchSampler(Sampler):
    """Yields uniform-decision-type batches for all three questions inside ONE
    epoch, so every shared parameter (state_proj, E_Q, E_C, fusion) is updated
    by all three losses — joint multi-task training, one optimizer, one epoch
    loop. Batches are drawn per type, concatenated, then permuted together;
    the permutation is seeded, so the order repeats exactly for a given seed.
    """

    def __init__(
        self,
        spans: Dict[str, Tuple[int, int]],
        batch_size: int,
        seed: int,
    ) -> None:
        self.spans = dict(spans)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self._batches = self._build(0)

    def _build(self, epoch: int) -> List[List[int]]:
        rng = np.random.RandomState(self.seed + epoch)
        batches: List[List[int]] = []
        for start, end in self.spans.values():
            perm = rng.permutation(np.arange(start, end, dtype=np.int64))
            n_batches = len(perm) // self.batch_size
            for i in range(n_batches):
                batches.append(perm[i * self.batch_size : (i + 1) * self.batch_size].tolist())
        order = rng.permutation(len(batches))
        return [batches[k] for k in order]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._batches = self._build(self.epoch)

    def __iter__(self) -> Iterator[List[int]]:
        return iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)


def build_multi_records(
    features: np.ndarray, labels: Sequence[int]
) -> Tuple[List[Dict[str, Any]], Dict[str, Tuple[int, int]]]:
    """One record per (example, question); records are grouped into contiguous
    per-type spans so the joint sampler can draw uniform-type batches."""
    records: List[Dict[str, Any]] = []
    spans: Dict[str, Tuple[int, int]] = {}
    for qtype in TYPES:
        start = len(records)
        for i, y in enumerate(labels):
            y = int(y)
            records.append(
                {
                    "state": np.ascontiguousarray(features[i], dtype=np.float32),
                    "questions": [QUESTIONS[qtype]],
                    "decision_type": qtype,
                    "options": [list(OPTION_SETS[qtype])] if OPTION_SETS[qtype] else None,
                    "labels": [1 if (y == 0) else 0] if qtype == "noul" else [y],
                }
            )
        spans[qtype] = (start, len(records))
    return records, spans


def train_multi_head(
    module: FeaturizedJevXMulti,
    records: List[Dict[str, Any]],
    spans: Dict[str, Tuple[int, int]],
    *,
    seed: int,
    lr: float,
    epochs: int,
    batch_size: int,
) -> Tuple[HeadTrainer, List[Dict[str, Any]]]:
    r"""HeadTrainer over the interleaved three-question loader.

    \`make_supervised_step_fn\` does the loss dispatch (choice -> CE, score ->
    CE, noul -> BCE) exactly as the library does; the wrapper only tallies the
    per-type losses so the epoch record can carry one loss curve per question.
    """
    torch.manual_seed(seed)
    sampler = JointBatchSampler(spans, batch_size, seed)
    loader = DataLoader(
        records,
        batch_sampler=sampler,
        collate_fn=decision_collate_fn,
    )
    trainer = HeadTrainer(module, lr=lr, seed=seed)
    base_step = make_supervised_step_fn(module)
    tally: Dict[str, List[float]] = {}

    def step(batch: Dict[str, Any]):
        loss, components = base_step(batch)
        t = components["decision_type"]
        total, count = tally.get(t, (0.0, 0))
        tally[t] = (total + float(components["loss"]), count + 1)
        return loss, components

    history: List[Dict[str, Any]] = []

    def on_epoch(record: Dict[str, Any]) -> None:
        sampler.set_epoch(int(record["epoch"]))
        row = dict(record)
        row["loss_by_type"] = {
            t: round(total / count, 6) for t, (total, count) in tally.items()
        }
        row["steps_by_type"] = {t: count for t, (total, count) in tally.items()}
        history.append(row)
        tally.clear()
        log(
            f"    epoch {row['epoch']}: loss={row['loss']:.4f} "
            + " ".join(f"{t}={v:.4f}" for t, v in sorted(row["loss_by_type"].items()))
        )

    trainer.fit(loader, epochs=epochs, loss_fn=step, callback=on_epoch)
    return trainer, history


# --------------------------------------------------------------------------
# Phase 1: load + deterministic splits + featurize ONCE per split
# --------------------------------------------------------------------------

def load_yelp():
    """Load the pinned revision.

    Returns the Dataset objects plus the label columns and the ClassLabel
    names. The TEXT column is deliberately NOT materialized here: the official
    train split has 650,000 reviews and turning it into a python list costs
    hundreds of megabytes for rows this scenario never reads. phase1 selects
    only the rows the splits actually use.
    """
    from datasets import ClassLabel, load_dataset

    ds = load_dataset(DATASET_ID, revision=DATASET_REVISION)
    train, test = ds["train"], ds["test"]
    names = None
    feats = train.features.get("label")
    if isinstance(feats, ClassLabel):
        names = list(feats.names)
    return train, test, list(names) if names else None


def select_texts(dataset, positions: Sequence[int]) -> List[str]:
    """Texts at the given row positions, in that order, without reading the
    whole text column."""
    return list(dataset.select(list(int(p) for p in positions))["text"])


def _counts(labels: Sequence[int]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for y in labels:
        key = str(int(y))
        out[key] = out.get(key, 0) + 1
    return out


class Context:
    """Dataset, splits, frozen backbone, featurize-once, trained heads."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        self.timings: Dict[str, float] = {}
        self.log_lines: List[str] = []

    def new_model(self) -> JevX:
        model = JevX.from_pretrained(
            BACKBONE_ID,
            freeze_base_model=True,
            pooling_method=self.args.pooling,
            fusion_type=self.args.fusion,
        )
        model.eval()
        return model


def phase1(ctx: Context, res: Result) -> Context:
    args = ctx.args
    log("phase 1: loading dataset")
    t0 = time.perf_counter()
    train_ds, test_ds, class_names = load_yelp()
    tr_labels = [int(x) for x in train_ds["label"]]
    te_labels = [int(x) for x in test_ds["label"]]
    ctx.timings["load_dataset_s"] = time.perf_counter() - t0
    log(f"  dataset loaded in {ctx.timings['load_dataset_s']:.1f}s; "
        f"official train={len(tr_labels)} test={len(te_labels)}; class names={class_names}")

    # splits: train + cal carved disjointly from the official train split.
    train_idx, cal_idx = bc.make_splits(tr_labels, args.train_size, args.cal_size, seed=SPLIT_SEED)
    assert len(np.intersect1d(train_idx, cal_idx)) == 0, "train/cal overlap"
    wanted = list(train_idx) + list(cal_idx)
    t0 = time.perf_counter()
    texts = select_texts(train_ds, wanted)
    ctx.timings["select_texts_s"] = time.perf_counter() - t0
    ctx.train_texts = texts[: len(train_idx)]
    ctx.cal_texts = texts[len(train_idx) :]
    ctx.train_labels = [tr_labels[i] for i in train_idx]
    ctx.cal_labels = [tr_labels[i] for i in cal_idx]
    del texts, tr_labels
    # test: first N rows of the OFFICIAL test split (deterministic; the same
    # [:test_size] cap semantics run_quality.py / run_exp7.py use).
    n_test = min(args.test_size, len(te_labels))
    t0 = time.perf_counter()
    ctx.test_texts = select_texts(test_ds, range(n_test))
    ctx.test_labels = te_labels[:n_test]
    ctx.timings["select_test_texts_s"] = time.perf_counter() - t0
    ctx.test_cap_applied = n_test < len(te_labels)
    ctx.class_names = class_names
    log(f"  splits: train={len(ctx.train_texts)} cal={len(ctx.cal_texts)} "
        f"test={len(ctx.test_texts)} (test cap applied={ctx.test_cap_applied})")

    log("  loading frozen backbone")
    t0 = time.perf_counter()
    ctx.model = ctx.new_model()
    ctx.timings["load_backbone_s"] = time.perf_counter() - t0
    cfg = ctx.model.config.backbone_config
    ctx.backbone_revision = (
        cfg.get("_commit_hash") if isinstance(cfg, dict) else None
    ) or getattr(getattr(ctx.model.model, "config", None), "_commit_hash", None)
    log(f"  backbone loaded in {ctx.timings['load_backbone_s']:.1f}s; "
        f"revision={ctx.backbone_revision}")

    log("  featurizing (backbone forward count == 1 per batch per split)")
    t0 = time.perf_counter()
    ctx.feats: Dict[str, np.ndarray] = {}
    for name in ("train", "cal", "test"):
        texts = getattr(ctx, f"{name}_texts")
        t1 = time.perf_counter()
        ctx.feats[name] = bc.featurize(
            ctx.model, texts, batch_size=args.batch_size,
            max_length=args.max_length, strategies=[args.pooling],
        )[args.pooling]
        ctx.timings[f"featurize_{name}_s"] = time.perf_counter() - t1
        log(f"    {name}: {ctx.feats[name].shape} in {ctx.timings[f'featurize_{name}_s']:.1f}s")
    ctx.timings["featurize_splits_s"] = time.perf_counter() - t0

    # truncation bookkeeping: featurize truncates to max_length; the live
    # decide() path never truncates (JevX raises instead).
    ctx.token_len_stats = {}
    for name in ("train", "cal", "test"):
        lens = [
            len(ids)
            for ids in ctx.model.tokenizer(
                list(getattr(ctx, f"{name}_texts")), truncation=False
            )["input_ids"]
        ]
        ctx.token_len_stats[name] = {
            "n": len(lens),
            "over_max_length": int(sum(1 for l in lens if l > args.max_length)),
            "max_len_observed": int(max(lens)) if lens else 0,
        }
        log(f"    {name} token lengths: over {args.max_length} = "
            f"{ctx.token_len_stats[name]['over_max_length']}/{len(lens)}, "
            f"max={ctx.token_len_stats[name]['max_len_observed']}")

    res.set(
        "1_load_splits_featurize",
        {
            "dataset": {
                "id": DATASET_ID,
                "revision": DATASET_REVISION,
                "license": DATASET_LICENSE,
                "official_train_n": int(train_ds.num_rows),
                "official_test_n": int(test_ds.num_rows),
                "class_label_names": class_names,
                "label_semantics": (
                    "class index y in 0..4 == (y+1) stars; Q1 choice label = y, "
                    "Q2 score label = y (levels 0..4), Q3 noul label = 1[y == 0]"
                ),
            },
            "backbone": {
                "id": BACKBONE_ID,
                "sha_expected": BACKBONE_SHA,
                "commit_hash_observed": ctx.backbone_revision,
                "frozen": True,
                "mode": "A (featurize-once, heads only)",
                "pooling_method": args.pooling,
                "fusion_type": args.fusion,
            },
            "splits": {
                "split_seed": SPLIT_SEED,
                "train_n": len(ctx.train_texts),
                "cal_n": len(ctx.cal_texts),
                "test_n": len(ctx.test_texts),
                "test_source": (
                    f"official test split, first {len(ctx.test_texts)} rows "
                    f"(cap applied={ctx.test_cap_applied})"
                ),
                "train_cal_source": "disjoint stratified carve from the official train split (bc.make_splits)",
                "train_label_counts": _counts(ctx.train_labels),
                "cal_label_counts": _counts(ctx.cal_labels),
                "test_label_counts": _counts(ctx.test_labels),
                "train_noul_positive_rate": float(np.mean([1.0 if y == 0 else 0.0 for y in ctx.train_labels])),
                "test_noul_positive_rate": float(np.mean([1.0 if y == 0 else 0.0 for y in ctx.test_labels])),
            },
            "features": {
                "max_length": args.max_length,
                "shapes": {k: list(v.shape) for k, v in ctx.feats.items()},
                "dtypes": {k: str(v.dtype) for k, v in ctx.feats.items()},
                "token_length_stats": ctx.token_len_stats,
            },
            "questions": {t: QUESTIONS[t] for t in TYPES},
            "option_sets": {t: OPTION_SETS[t] for t in TYPES},
            "wall_time_s": {
                "load_dataset": round(ctx.timings["load_dataset_s"], 2),
                "load_backbone": round(ctx.timings["load_backbone_s"], 2),
                "featurize": round(ctx.timings["featurize_splits_s"], 2),
            },
        },
    )
    for k, v in ctx.timings.items():
        res.time(k, v)
    return ctx


# --------------------------------------------------------------------------
# Phase 2: train all three heads on the SAME features
# --------------------------------------------------------------------------

def phase2(ctx: Context, res: Result) -> Context:
    args = ctx.args
    train_records, train_spans = build_multi_records(ctx.feats["train"], ctx.train_labels)
    cal_records, cal_spans = build_multi_records(ctx.feats["cal"], ctx.cal_labels)
    test_records, test_spans = build_multi_records(ctx.feats["test"], ctx.test_labels)
    ctx.records = {"train": train_records, "cal": cal_records, "test": test_records}
    ctx.spans = {"train": train_spans, "cal": cal_spans, "test": test_spans}
    log(f"phase 2: {len(train_records)} training records "
        f"({len(ctx.train_texts)} examples x 3 questions), spans={train_spans}")

    per_seed: List[Dict[str, Any]] = []
    ctx.trained: Dict[int, JevX] = {}
    ctx.head_modules: Dict[int, FeaturizedJevXMulti] = {}
    ctx.history: Dict[int, List[Dict[str, Any]]] = {}
    total0 = time.perf_counter()
    for seed in ctx.seeds:
        # A fresh JevX per seed so head init is drawn from torch.manual_seed(seed),
        # exactly like run_quality.py's per-seed reload.
        model = ctx.new_model()
        head = FeaturizedJevXMulti(model)
        log(f"  seed {seed}: trainable params={bc.count_trainable(head)}")
        t0 = time.perf_counter()
        trainer, history = train_multi_head(
            head, train_records, train_spans,
            seed=seed, lr=args.lr, epochs=args.epochs, batch_size=args.batch_size,
        )
        wall = time.perf_counter() - t0
        ctx.trained[seed] = model
        ctx.history[seed] = history
        ctx.head_modules[seed] = head
        per_seed.append(
            {
                "seed": seed,
                "trainable_params": bc.count_trainable(head),
                "steps_per_epoch": sum(history[0]["steps_by_type"].values()),
                "loss_first": history[0]["loss"],
                "loss_last": history[-1]["loss"],
                "loss_curves": history,
                "wall_time_s": round(wall, 2),
            }
        )
        log(f"  seed {seed}: {wall:.1f}s, loss {history[0]['loss']:.4f} -> {history[-1]['loss']:.4f}")
    ctx.timings["train_all_seeds_s"] = time.perf_counter() - total0

    # The seed-0 model is the model the live scenario (phases 4-8) runs.
    ctx.model = ctx.trained[ctx.seeds[0]]
    ctx.head_module = ctx.head_modules[ctx.seeds[0]]
    ctx.model.eval()
    res.set(
        "2_train_three_heads",
        {
            "protocol": {
                "trainer": "jevx.training.HeadTrainer (AdamW, grad_clip=1.0)",
                "loss_dispatch": "jevx.training.make_supervised_step_fn (choice=CE, score=CE, noul=BCE)",
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "joint": (
                    "one loader interleaves all three decision types inside every "
                    "epoch, so state_proj / E_Q / E_C / fusion are updated by all "
                    "three losses; each type's head sees only its own loss"
                ),
                "shared_features": "the same precomputed [N, 128] matrices for all three questions",
            },
            "seeds": list(ctx.seeds),
            "per_seed": per_seed,
            "wall_time_s": round(ctx.timings["train_all_seeds_s"], 2),
        },
    )
    res.time("train_all_seeds_s", ctx.timings["train_all_seeds_s"])
    return ctx


# --------------------------------------------------------------------------
# Phase 3: calibrate on the CALIBRATION split only
# --------------------------------------------------------------------------

def collect_head_logits(
    module: FeaturizedJevXMulti, records: List[Dict[str, Any]], batch_size: int
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Raw per-type logits [n, K] on one split, via the SAME featurized modules
    the live decide() path uses."""
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    module.eval()
    for qtype in TYPES:
        subset = [r for r in records if r["decision_type"] == qtype]
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                            collate_fn=decision_collate_fn)
        logits_all, labels_all = [], []
        with torch.no_grad():
            for batch in loader:
                logits_all.append(module(batch).detach().cpu().numpy().astype(np.float32))
                labels_all.append(
                    np.asarray(batch["labels"], dtype=np.float32).reshape(-1)
                )
        out[qtype] = (
            np.concatenate(logits_all, axis=0),
            np.concatenate(labels_all, axis=0),
        )
    return out


def phase3(ctx: Context, res: Result) -> None:
    args = ctx.args
    module = ctx.head_module  # seed-0 modules (the live model)
    t0 = time.perf_counter()
    split_logits = {
        name: collect_head_logits(module, ctx.records[name], args.batch_size)
        for name in ("cal", "test")
    }
    ctx.split_logits = split_logits
    fitted: Dict[str, Any] = {}
    for qtype in TYPES:
        cal_logits, cal_labels = split_logits["cal"][qtype]
        row: Dict[str, Any] = {
            "n_cal": int(cal_logits.shape[0]),
            "logit_shape": list(cal_logits.shape),
            "cal_label_mean": float(cal_labels.mean()),
        }
        for method in ("temperature", "isotonic"):
            engine = CalibrationEngine(method)
            engine.fit(
                torch.as_tensor(cal_logits),
                torch.as_tensor(cal_labels.astype(np.int64)),
            )
            if method == "temperature":
                row["temperature_T"] = float(engine.calibrator.temperature)
            else:
                row["isotonic_state_keys"] = sorted(engine.state_dict().keys())
            ctx.engines = getattr(ctx, "engines", {})
            ctx.engines.setdefault(qtype, {})[method] = engine
        fitted[qtype] = row
        log(f"  {qtype}: cal n={row['n_cal']} logits{row['logit_shape']} "
            f"T={row.get('temperature_T')}")
    ctx.timings["calibrate_s"] = time.perf_counter() - t0
    res.set(
        "3_calibration",
        {
            "fit_split": "calibration split ONLY (never train, never test)",
            "engines": "jevx.CalibrationEngine (temperature, isotonic) per question type",
            "note": (
                "JevXForDecision holds ONE calibration engine per model, so a mixed "
                "decide() call applies a single calibrator to all three types. The "
                "scenario therefore fits one engine PER QUESTION TYPE and reports "
                "the per-type temperatures; the live decide() call in phase 4 runs "
                "the raw (uncalibrated) model."
            ),
            "per_type": fitted,
            "wall_time_s": round(ctx.timings["calibrate_s"], 2),
        },
    )
    res.time("calibrate_s", ctx.timings["calibrate_s"])


# --------------------------------------------------------------------------
# Live multi-question decide() helpers
# --------------------------------------------------------------------------

def question_list() -> List[str]:
    return [QUESTIONS[t] for t in TYPES]


def option_list() -> List[Optional[List[str]]]:
    return [list(OPTION_SETS[t]) if OPTION_SETS[t] else None for t in TYPES]


def encode_states(
    model: JevX, texts: Sequence[str], max_length: int, *, pad_to_max: bool = False
) -> Dict[str, torch.Tensor]:
    """Tokenize a state batch EXACTLY as bc.featurize does (truncation=True,
    max_length=max_length, dynamic padding).

    The live decide() path never truncates a state — it raises when the token
    length exceeds the backbone's learned position table (bert L-2 H-128 has
    max_position_embeddings=512, and Yelp reviews are far longer). forward()
    documents `input_ids=`/`attention_mask=` as the pre-tokenized spelling,
    so tokenizing here with the SAME max_length the featurize-once pass used
    makes the live call and the training features see identical tokens, and
    the raw review text is still the state being decided about.
    """
    enc = model.tokenizer(
        list(texts),
        return_tensors="pt",
        padding="max_length" if pad_to_max else True,
        truncation=True,
        max_length=max_length,
    )
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}


def do_decide(
    model: JevX,
    texts: Sequence[str],
    max_length: int,
    threshold: Optional[float] = None,
    questions: Optional[Sequence[str]] = None,
    decision_type: Optional[Sequence[str]] = None,
    options: Optional[Sequence[Optional[List[str]]]] = None,
    pad_to_max: bool = False,
):
    """THE scenario call: one state batch, m questions of different decision
    types, ONE backbone forward. `questions`/`decision_type`/`options`
    default to the scenario's three questions."""
    enc = encode_states(model, texts, max_length, pad_to_max=pad_to_max)
    return model.decide(
        state=None,
        questions=list(questions) if questions is not None else question_list(),
        decision_type=list(decision_type) if decision_type is not None else list(TYPES),
        options=list(options) if options is not None else option_list(),
        confidence_threshold=threshold,
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
    )


def decide_all(model: JevX, texts: Sequence[str], max_length: int,
               threshold: Optional[float] = None):
    """Convenience wrapper for the standard three-question scenario call."""
    return do_decide(model, texts, max_length, threshold=threshold)


def rows_by_type(output, batch_size: int) -> Dict[str, List[Any]]:
    """Split the flat B*m state-major decision list back into per-type rows."""
    out: Dict[str, List[Any]] = {t: [] for t in TYPES}
    for b in range(batch_size):
        for j, t in enumerate(TYPES):
            out[t].append(output.decisions[b * len(TYPES) + j])
    return out


def probs_of(decision) -> np.ndarray:
    return decision.probabilities.detach().cpu().numpy().astype(np.float64).reshape(-1)


def conf_of(decision) -> float:
    return float(decision.confidence.detach().cpu().numpy().reshape(-1)[0])


# --------------------------------------------------------------------------
# Phase 4: ONE decide() over the test batch, invariants asserted live
# --------------------------------------------------------------------------

def phase4(ctx: Context, res: Result) -> None:
    args = ctx.args
    model = ctx.model
    model.eval()
    texts = ctx.test_texts
    b = len(texts)
    log(f"phase 4: one decide() over the {b}-state test batch, 3 questions each")

    calls = {"n": 0}
    handle = model.model.register_forward_hook(
        lambda mod, inp, out: calls.__setitem__("n", calls["n"] + 1)
    )
    t0 = time.perf_counter()
    try:
        output = decide_all(model, texts, args.max_length)
    finally:
        handle.remove()
    ctx.decision_output = output
    ctx.call_seconds = time.perf_counter() - t0
    n_expected = b * len(TYPES)
    rows = rows_by_type(output, b)
    log(f"  {len(output.decisions)} decisions (expected {n_expected}); "
        f"backbone forwards={calls['n']}; wall={ctx.call_seconds:.1f}s")

    failures: List[str] = []

    def check(name: str, ok: bool, detail: Any = None) -> None:
        if not ok:
            failures.append(f"{name}: {detail}")
        log(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" ({detail})" if detail is not None else ""))

    check("backbone_forward_hook_count == 1", calls["n"] == 1, f"observed {calls['n']}")
    check("len(decisions) == B*m", len(output.decisions) == n_expected,
          f"{len(output.decisions)} vs {n_expected}")
    check("state-major ordering b*m+j",
          [d.schema_metadata["decision_type"] for d in output.decisions[: len(TYPES)]]
          == list(TYPES)
          and [d.schema_metadata["decision_type"] for d in output.decisions[-len(TYPES):]]
          == list(TYPES),
          "first and last m entries must be the 3 types in question order")

    choice_p = np.stack([probs_of(d) for d in rows["choice"]])
    score_p = np.stack([probs_of(d) for d in rows["score"]])
    noul_p = np.concatenate([probs_of(d) for d in rows["noul"]])
    choice_conf = np.asarray([conf_of(d) for d in rows["choice"]])
    score_conf = np.asarray([conf_of(d) for d in rows["score"]])
    noul_conf = np.asarray([conf_of(d) for d in rows["noul"]])
    choice_val = [d.predicted_value for d in rows["choice"]]
    score_val = [d.predicted_value for d in rows["score"]]
    score_es = np.asarray(
        [float(d.expected_score.detach().cpu().numpy().reshape(-1)[0]) for d in rows["score"]]
    )

    simplex_err = float(np.abs(choice_p.sum(axis=1) - 1.0).max())
    score_simplex_err = float(np.abs(score_p.sum(axis=1) - 1.0).max())
    check("choice rows are a simplex (atol 1e-5)", simplex_err <= 1e-5, f"max |sum-1|={simplex_err:.3e}")
    check("score rows are a simplex (atol 1e-5)", score_simplex_err <= 1e-5, f"max |sum-1|={score_simplex_err:.3e}")
    check("choice argmax in options",
          all(v in CHOICE_OPTIONS for v in choice_val),
          f"distinct predicted values={sorted(set(choice_val))}")
    check("score argmax in levels",
          all(str(v) in SCORE_LEVELS for v in score_val),
          f"distinct predicted values={sorted(set(map(str, score_val)))}")
    es_lo, es_hi = float(score_es.min()), float(score_es.max())
    check("expected_score in [0, 4]", es_lo >= 0.0 and es_hi <= 4.0, f"observed [{es_lo:.4f}, {es_hi:.4f}]")
    noul_lo, noul_hi = float(noul_p.min()), float(noul_p.max())
    check("noul p in (0, 1) open interior", noul_lo > 0.0 and noul_hi < 1.0,
          f"observed [{noul_lo:.6f}, {noul_hi:.6f}]")
    check("noul probabilities are 1 per row",
          all(probs_of(d).shape == (1,) for d in rows["noul"]))
    conf_lo = float(min(choice_conf.min(), score_conf.min(), noul_conf.min()))
    conf_hi = float(max(choice_conf.max(), score_conf.max(), noul_conf.max()))
    check("confidence in [0, 1]", 0.0 <= conf_lo and conf_hi <= 1.0,
          f"observed [{conf_lo:.4f}, {conf_hi:.4f}]")
    check("no cross-type contamination: distinct probability widths",
          choice_p.shape[1] == 5 and score_p.shape[1] == 5 and noul_p.shape[0] == b,
          f"choice{choice_p.shape} score{score_p.shape} noul{noul_p.shape}")
    check("every decision's decision_type matches its question slot",
          all(d.schema_metadata["decision_type"] == t
              for d, t in zip(output.decisions, (list(TYPES) * b))),
          "flat index b*m+j carries its own question's type")
    check("abstained is None without a threshold",
          all(d.abstained is None for d in output.decisions))
    check("capability_metadata reports one backbone call",
          output.capability_metadata.get("backbone_forward_calls") == 1)

    # featurize-once claim, checked live: the head stack run on the CACHED
    # features must reproduce the live call's logits on the same states.
    module = ctx.head_module
    module.eval()
    eq_n = min(64, b)
    eq_slice = slice(0, eq_n)
    with torch.no_grad():
        cached_choice = module({
            "state": [ctx.feats["test"][i] for i in range(eq_n)],
            "decision_type": "choice",
        })
    live_choice_logits = torch.stack(
        [output.decisions[i * len(TYPES)].logits.detach().reshape(-1) for i in range(eq_n)]
    )
    featurized_vs_live = {
        "n": eq_n,
        "max_abs_logit_delta": float((live_choice_logits - cached_choice).abs().max()),
        "note": (
            "identical modules; the residual is float re-association between two "
            "independently batched runs of the same arithmetic, plus the padding-width "
            "difference between the two batches"
        ),
    }
    log(f"  featurized-vs-live Q1 max |logit delta| = "
        f"{featurized_vs_live['max_abs_logit_delta']:.3e} over {eq_n} states")

    ctx.live = {
        "choice_probs": choice_p, "score_probs": score_p, "noul_probs": noul_p,
        "choice_conf": choice_conf, "score_conf": score_conf, "noul_conf": noul_conf,
        "choice_val": choice_val, "score_val": score_val, "score_expected": score_es,
    }
    res.set(
        "4_live_decide_invariants",
        {
            "call": (
                "JevXForDecision.decide(input_ids=<tokenized test batch, truncation=True, "
                f"max_length={args.max_length}>, questions=3, decision_type=['choice','score',"
                "'noul'], options=[K=5, levels 0..4, None])"
            ),
            "why_pre_tokenized": (
                "the live path never truncates a state and raises above the backbone's "
                f"learned position table; the same tokenizer call bc.featurize() makes "
                f"(max_length={args.max_length}) is used so the live call and the "
                "featurize-once training features see identical tokens"
            ),
            "featurized_vs_live": featurized_vs_live,
            "batch_size": b,
            "questions": list(TYPES),
            "decisions_returned": len(output.decisions),
            "backbone_forward_hook_count": calls["n"],
            "wall_time_s": round(ctx.call_seconds, 3),
            "invariants": {
                "choice_simplex_max_abs_error": simplex_err,
                "score_simplex_max_abs_error": score_simplex_err,
                "expected_score_range": [es_lo, es_hi],
                "noul_p_range": [noul_lo, noul_hi],
                "confidence_range": [conf_lo, conf_hi],
                "choice_predicted_values": sorted(set(choice_val)),
                "score_predicted_values": sorted(set(map(str, score_val))),
            },
            "checks_total": 16,
            "checks_failed": failures,
            "all_passed": not failures,
        },
    )


# --------------------------------------------------------------------------
# Phase 5: correctness battery
# --------------------------------------------------------------------------

def _flat_probs(output) -> np.ndarray:
    return np.concatenate(
        [d.probabilities.detach().cpu().numpy().astype(np.float64).reshape(-1)
         for d in output.decisions]
    )


def phase5(ctx: Context, res: Result) -> None:
    args = ctx.args
    model = ctx.model
    model.eval()
    small = ctx.test_texts[:SMALL_STATES]
    solo_n = min(SOLO_STATES, len(ctx.test_texts))
    base_full = ctx.decision_output
    log(f"phase 5: correctness battery (solo walk over {solo_n} states, "
        f"small replays over {len(small)} states)")
    battery: List[Dict[str, Any]] = []

    def record(name: str, detail: Dict[str, Any], ok: bool) -> None:
        battery.append({"check": name, "passed": bool(ok), **detail})
        log(f"  [{'PASS' if ok else 'FAIL'}] {name}: "
            + ", ".join(f"{k}={v}" for k, v in detail.items()))

    # (a) question independence: perturbing Q2's options must not move Q1 or Q3.
    perturbed = [list(v) if v else None for v in option_list()]
    perturbed[1] = ["worst", "bad", "so-so", "good", "best"]
    out_pert = do_decide(model, ctx.test_texts, args.max_length, options=perturbed)
    b = len(ctx.test_texts)
    rows_p = rows_by_type(out_pert, b)
    d_choice = float(np.abs(np.stack([probs_of(d) for d in rows_p["choice"]])
                            - ctx.live["choice_probs"]).max())
    d_noul = float(np.abs(np.concatenate([probs_of(d) for d in rows_p["noul"]])
                          - ctx.live["noul_probs"]).max())
    d_score = float(np.abs(np.stack([probs_of(d) for d in rows_p["score"]])
                           - ctx.live["score_probs"]).max())
    record(
        "a_question_independence",
        {"q2_options_before": SCORE_LEVELS, "q2_options_after": perturbed[1],
         "q1_max_abs_prob_delta": d_choice, "q3_max_abs_prob_delta": d_noul,
         "q2_max_abs_prob_delta (sanity: must be > 0)": d_score},
        d_choice <= 1e-6 and d_noul <= 1e-6 and d_score > 0.0,
    )

    # (b) batched == three solo calls (one state at a time).
    max_err = {t: 0.0 for t in TYPES}
    base_rows = rows_by_type(base_full, b)
    for i in range(solo_n):
        solo = do_decide(model, [ctx.test_texts[i]], args.max_length)
        for j, t in enumerate(TYPES):
            solo_p = probs_of(solo.decisions[j])
            batch_p = probs_of(base_rows[t][i])
            max_err[t] = max(max_err[t], float(np.abs(solo_p - batch_p).max()))
    worst = max(max_err.values())

    # Same walk, but every 1-row call is padded to max_length so the backbone
    # sees the SAME tensor width in both calls. Whatever residual survives this
    # is arithmetic re-association; whatever the dynamic-padding walk adds on
    # top is the padding-width effect. Both numbers are reported; neither is
    # hidden by picking the flattering one.
    max_err_fixed = {t: 0.0 for t in TYPES}
    for i in range(solo_n):
        solo = do_decide(
            model, [ctx.test_texts[i]], args.max_length, pad_to_max=True
        )
        for j, t in enumerate(TYPES):
            max_err_fixed[t] = max(
                max_err_fixed[t],
                float(np.abs(probs_of(solo.decisions[j])
                             - probs_of(base_rows[t][i])).max()),
            )
    worst_fixed = max(max_err_fixed.values())
    record(
        "b_batched_equals_solo_calls",
        {"states_walked": solo_n,
         "max_abs_prob_delta_choice": max_err["choice"],
         "max_abs_prob_delta_score": max_err["score"],
         "max_abs_prob_delta_noul": max_err["noul"],
         "worst": worst, "tolerance": 1e-5, "within_tolerance": worst <= 1e-5,
         "matched_padding_walk": {
             "scheme": "every solo call padded to max_length so both calls share one tensor width",
             "max_abs_prob_delta_choice": max_err_fixed["choice"],
             "max_abs_prob_delta_score": max_err_fixed["score"],
             "max_abs_prob_delta_noul": max_err_fixed["noul"],
             "worst": worst_fixed,
             "within_tolerance": worst_fixed <= 1e-5,
         },
         "note": "the batched call pads to the longest state in the batch, a 1-row call "
                 "pads to that row's own length, so the two passes reduce the same "
                 "arithmetic over different tensor shapes; the comparison is a numerical "
                 "bound, not a bitwise identity",
         },
        worst <= 1e-5,
    )

    # (c) runtime_validation on/off -> bitwise identical.
    keep = model.config.runtime_validation
    model.config.runtime_validation = False
    try:
        out_off = decide_all(model, small, args.max_length)
    finally:
        model.config.runtime_validation = keep
    out_on = decide_all(model, small, args.max_length)
    off_p, on_p = _flat_probs(out_off), _flat_probs(out_on)
    bitwise = bool(np.array_equal(off_p, on_p))
    record(
        "c_runtime_validation_on_off_bitwise_identical",
        {"states": len(small), "bitwise_equal_probabilities": bitwise,
         "max_abs_prob_delta": float(np.abs(off_p - on_p).max())},
        bitwise,
    )

    # (d) save -> load -> same call.
    tmpdir = Path(tempfile.mkdtemp(prefix="jevx_real_scenario_"))
    model.save_pretrained(tmpdir)
    reloaded = JevX.from_pretrained(tmpdir)
    reloaded.eval()
    out_loaded = decide_all(reloaded, small, args.max_length)
    out_ref = decide_all(model, small, args.max_length)
    lp, rp = _flat_probs(out_loaded), _flat_probs(out_ref)
    record(
        "d_save_load_same_call",
        {"states": len(small), "max_abs_prob_delta": float(np.abs(lp - rp).max()),
         "bitwise_equal": bool(np.array_equal(lp, rp)),
         "saved_to": str(tmpdir)},
        float(np.abs(lp - rp).max()) <= 1e-6,
    )

    # (e) the same call twice is bitwise identical.
    out_again = decide_all(model, small, args.max_length)
    ap, bp = _flat_probs(out_again), rp
    record(
        "e_repeat_call_bitwise_identical",
        {"states": len(small), "bitwise_equal": bool(np.array_equal(ap, bp)),
         "max_abs_prob_delta": float(np.abs(ap - bp).max())},
        bool(np.array_equal(ap, bp)),
    )

    # (f) confidence_threshold on Q1.
    th = args.threshold
    out_th = decide_all(model, ctx.test_texts, args.max_length, threshold=th)
    rows_th = rows_by_type(out_th, b)
    abst = {
        t: int(sum(1 for d in rows_th[t] if bool(d.abstained.reshape(-1)[0])))
        for t in TYPES
    }
    q1_abst = abst["choice"]
    cleared_ok = all(
        d.predicted_index is None and d.predicted_value is None
        for d in rows_th["choice"] if bool(d.abstained.reshape(-1)[0])
    )
    probs_untouched = float(np.abs(
        np.stack([probs_of(d) for d in rows_th["choice"]]) - ctx.live["choice_probs"]
    ).max())
    conf_below = int(sum(1 for d in rows_th["choice"]
                         if conf_of(d) < th))
    record(
        "f_confidence_threshold_abstention",
        {"threshold": th, "states": b,
         "abstained_choice": q1_abst, "abstained_score": abst["score"],
         "abstained_noul": abst["noul"],
         "abstained_count_equals_conf_below_threshold": q1_abst == conf_below,
         "predicted_fields_cleared_on_abstained": cleared_ok,
         "probabilities_max_abs_delta_vs_unthresholded": probs_untouched},
        q1_abst > 0 and cleared_ok and probs_untouched == 0.0,
    )
    ctx.threshold_output = out_th
    ctx.threshold = th

    res.set(
        "5_correctness_battery",
        {
            "batch_size_for_full_call": b,
            "checks": battery,
            "passed": sum(1 for r in battery if r["passed"]),
            "total": len(battery),
            "all_passed": all(r["passed"] for r in battery),
        },
    )


# --------------------------------------------------------------------------
# Phase 6: quality on TEST, uncalibrated vs temperature
# --------------------------------------------------------------------------

def expected_from_probs(probs: np.ndarray, k: int = 5) -> np.ndarray:
    levels = np.arange(k, dtype=np.float64)
    return (np.asarray(probs, dtype=np.float64) * levels).sum(axis=1)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    t = torch.as_tensor(np.asarray(logits, dtype=np.float32))
    return stable_softmax(t, dim=-1).numpy().astype(np.float64)


def phase6(ctx: Context, res: Result) -> None:
    args = ctx.args
    b = len(ctx.test_texts)
    y = np.asarray(ctx.test_labels, dtype=np.int64)
    noul_y = (y == 0).astype(np.int64)
    train_counts = _counts(ctx.train_labels)
    majority_class = int(max(train_counts, key=lambda k: int(train_counts[k])))
    majority_acc = float((y == majority_class).mean())
    test_counts = _counts(ctx.test_labels)
    test_majority_class = int(max(test_counts, key=lambda k: int(test_counts[k])))
    log(f"phase 6: quality on the {b}-row test split; train-majority class={majority_class} "
        f"(acc on test {majority_acc:.4f}); seeds={ctx.seeds}")

    per_seed: List[Dict[str, Any]] = []
    agg: Dict[str, List[Dict[str, float]]] = {}
    for seed in ctx.seeds:
        log(f"  seed {seed}")
        module = FeaturizedJevXMulti(ctx.trained[seed])
        split_logits = {
            name: collect_head_logits(module, ctx.records[name], args.batch_size)
            for name in ("cal", "test")
        }
        row: Dict[str, Any] = {"seed": seed}

        # ---- B1 linear probe on the IDENTICAL features -----------------------
        in_dim = ctx.feats["train"].shape[1]
        torch.manual_seed(seed)
        b1 = bc.LinearProbe(in_dim, 5)
        b1_train = bc.build_records(ctx.feats["train"], ctx.train_labels)
        bc.train_heads(b1, b1_train, seed=seed, lr=args.lr,
                       epochs=args.epochs, batch_size=args.batch_size)
        b1_cal, b1_cal_y = bc.collect_logits(
            b1, bc.build_records(ctx.feats["cal"], ctx.cal_labels), batch_size=args.batch_size)
        b1_test, b1_test_y = bc.collect_logits(
            b1, bc.build_records(ctx.feats["test"], ctx.test_labels), batch_size=args.batch_size)
        assert np.array_equal(np.asarray(b1_test_y).reshape(-1), y)
        row["b1_linear"] = bc.metrics_from_logits(b1_test, y, n_bins=N_BINS)
        row["b1_temperature"] = bc.calibrated_metrics(
            b1_test, y, "temperature", fit_logits=b1_cal,
            fit_labels=np.asarray(b1_cal_y).reshape(-1), n_bins=N_BINS)

        for qtype in TYPES:
            cal_logits, cal_labels = split_logits["cal"][qtype]
            test_logits, _ = split_logits["test"][qtype]
            t_engine = CalibrationEngine("temperature").fit(
                torch.as_tensor(cal_logits), torch.as_tensor(cal_labels.astype(np.int64))
            )
            t_value = float(t_engine.calibrator.temperature)
            iso = CalibrationEngine("isotonic")
            iso.fit(torch.as_tensor(cal_logits), torch.as_tensor(cal_labels.astype(np.int64)))
            if qtype == "choice":
                row["q1_choice"] = {
                    "uncalibrated": bc.metrics_from_logits(test_logits, y, n_bins=N_BINS),
                    "temperature": bc.calibrated_metrics(
                        test_logits, y, "temperature", fit_logits=cal_logits,
                        fit_labels=cal_labels.astype(np.int64), n_bins=N_BINS),
                    "isotonic": bc.calibrated_metrics(
                        test_logits, y, "isotonic", fit_logits=cal_logits,
                        fit_labels=cal_labels.astype(np.int64), n_bins=N_BINS),
                    "fitted_T": t_value,
                    "accuracy_majority_baseline": majority_acc,
                    "accuracy_uniform_baseline": 1.0 / 5.0,
                }
            elif qtype == "score":
                p_uncal = softmax_np(test_logits)
                p_temp = softmax_np(t_engine.calibrate(torch.as_tensor(test_logits), "logits").numpy())
                p_iso = softmax_np(iso.calibrate(torch.as_tensor(test_logits), "logits").numpy())
                row["q2_score"] = {
                    "uncalibrated": score_metrics(expected_from_probs(p_uncal), y),
                    "temperature": score_metrics(expected_from_probs(p_temp), y),
                    "isotonic": score_metrics(expected_from_probs(p_iso), y),
                    "fitted_T": t_value,
                    "argmax_accuracy": bc.metrics_from_logits(test_logits, y)["accuracy"],
                }
            else:
                cl = cal_logits.reshape(-1, 1)
                tl = test_logits.reshape(-1, 1)
                row["q3_noul"] = {
                    "uncalibrated": binary_metrics(tl, noul_y),
                    "temperature": binary_metrics(
                        t_engine.calibrate(torch.as_tensor(tl), "logits").numpy(), noul_y),
                    "isotonic": binary_metrics(
                        iso.calibrate(torch.as_tensor(tl), "logits").numpy(), noul_y),
                    "fitted_T": t_value,
                    "always_yes_accuracy": float(noul_y.mean()),
                    "always_no_accuracy": float(1.0 - noul_y.mean()),
                }
        # One pass, one append per (metric, seed): duplicates would silently
        # change the across-seed std.
        for key in ("q1_choice", "q2_score", "q3_noul"):
            for variant in ("uncalibrated", "temperature", "isotonic"):
                for mname, mval in row[key][variant].items():
                    if isinstance(mval, (int, float)) and not isinstance(mval, bool):
                        agg.setdefault(f"{key}.{variant}.{mname}", []).append(float(mval))
            if key == "q1_choice":
                row["q1_choice"]["accuracy_majority_baseline"] = majority_acc
                row["q1_choice"]["accuracy_uniform_baseline"] = 1.0 / 5.0
                agg.setdefault("q1_choice.accuracy_majority_baseline", []).append(majority_acc)
                agg.setdefault("q1_choice.accuracy_uniform_baseline", []).append(1.0 / 5.0)
        for variant in ("b1_linear", "b1_temperature"):
            for mname, mval in row[variant].items():
                if isinstance(mval, (int, float)) and not isinstance(mval, bool):
                    agg.setdefault(f"B1.{variant}.{mname}", []).append(float(mval))
        per_seed.append(row)
        log(f"    Q1 acc={row['q1_choice']['uncalibrated']['accuracy']:.4f} "
            f"f1={row['q1_choice']['uncalibrated']['macro_f1']:.4f} "
            f"ece={row['q1_choice']['uncalibrated']['ece']:.4f}->"
            f"{row['q1_choice']['temperature']['ece']:.4f} | "
            f"Q2 mae={row['q2_score']['uncalibrated']['mae']:.4f} "
            f"r={row['q2_score']['uncalibrated']['pearson_r']:.4f} | "
            f"Q3 acc={row['q3_noul']['uncalibrated']['accuracy']:.4f} "
            f"auc={row['q3_noul']['uncalibrated']['auc']:.4f}")
        del module, b1

    aggregate = {
        k: {
            "mean": float(np.mean(v)),
            "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
            "n_seeds": len(v),
        }
        for k, v in agg.items()
        if "by_true_level" not in k
    }
    res.set(
        "6_quality_test",
        {
            "split": f"test (first {b} rows of the official test split)",
            "n_test": b,
            "seeds": list(ctx.seeds),
            "n_bins_for_ece_mce": N_BINS,
            "baselines": {
                "majority_class_from_train": majority_class,
                "majority_accuracy_on_test": majority_acc,
                "test_most_frequent_class": test_majority_class,
                "test_most_frequent_class_rate": float(
                    max(int(v) for v in test_counts.values()) / b),
                "uniform_accuracy": 1.0 / 5.0,
                "noul_always_yes_accuracy": float(noul_y.mean()),
                "b1_linear_probe": "single Linear(128 -> 5) on the identical frozen features, same splits/seeds/hyperparams",
            },
            "auc_rank_self_check": auc_rank_self_check(),
            "per_seed": per_seed,
            "aggregate_mean_std": aggregate,
            "q2_ordinal_error_by_true_level_seed0": per_seed[0]["q2_score"]["uncalibrated"]["by_true_level"],
        },
    )


# --------------------------------------------------------------------------
# Phase 7: risk-coverage on Q1 with temperature-calibrated confidences
# --------------------------------------------------------------------------

def phase7(ctx: Context, res: Result) -> None:
    module = FeaturizedJevXMulti(ctx.trained[ctx.seeds[0]])
    cal_logits, cal_labels = collect_head_logits(
        module, ctx.records["cal"], ctx.args.batch_size)["choice"]
    test_logits, _ = collect_head_logits(
        module, ctx.records["test"], ctx.args.batch_size)["choice"]
    y = np.asarray(ctx.test_labels, dtype=np.int64)
    engine = CalibrationEngine("temperature").fit(
        torch.as_tensor(cal_logits), torch.as_tensor(cal_labels.astype(np.int64))
    )
    T = float(engine.calibrator.temperature)
    probs = softmax_np(engine.calibrate(torch.as_tensor(test_logits), "logits").numpy())
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(np.float64)
    coverage, risk = risk_coverage_curve(conf, correct)
    aurc = float(risk.mean())
    cov_at = {str(b): float(coverage_at_risk(conf, correct, b)) for b in (0.1, 0.2)}
    uncal_conf = softmax_np(test_logits).max(axis=1)
    uncal_cov, uncal_risk = risk_coverage_curve(uncal_conf, correct)
    grid = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
    table = []
    for g in grid:
        idx = min(int(round(g * (len(coverage) - 1))), len(coverage) - 1)
        table.append({
            "coverage": g,
            "risk_at_coverage": float(risk[idx]),
            "n_covered": int(round(g * len(correct))),
        })
    # Selective prediction with thresholds taken from the CALIBRATION split's
    # confidence quantiles -- nothing is chosen using the test labels.
    cal_probs = softmax_np(engine.calibrate(torch.as_tensor(cal_logits), "logits").numpy())
    cal_conf = cal_probs.max(axis=1)
    cal_correct = (cal_probs.argmax(axis=1) == cal_labels.astype(np.int64)).astype(np.float64)
    selective = []
    for q in (0.1, 0.25, 0.5, 0.75):
        tau = float(np.quantile(cal_conf, q))
        keep = conf >= tau
        selective.append({
            "cal_conf_quantile": q,
            "threshold_tau": tau,
            "test_coverage": float(keep.mean()),
            "test_accuracy_on_answered": float(correct[keep].mean()) if bool(keep.any()) else None,
            "test_n_answered": int(keep.sum()),
            "cal_accuracy_on_answered": float(cal_correct[cal_conf >= tau].mean()),
        })
    log(f"phase 7: risk-coverage (T={T:.4f}) AURC={aurc:.4f}; "
        f"coverage@risk0.1={cov_at['0.1']:.4f}; @0.2={cov_at['0.2']:.4f}; "
        f"selective acc at cal-q50 tau={selective[2]['threshold_tau']:.4f} -> "
        f"{selective[2]['test_accuracy_on_answered']:.4f} on "
        f"{selective[2]['test_coverage']:.0%} answered")
    res.set(
        "7_risk_coverage",
        {
            "question": "Q1 choice (K=5)",
            "confidence": "max probability AFTER temperature scaling fit on the cal split",
            "fitted_T": T,
            "aurc": aurc,
            "coverage_at_risk_0.1": cov_at["0.1"],
            "coverage_at_risk_0.2": cov_at["0.2"],
            "risk_at_coverage_grid": table,
            "uncalibrated_reference": {
                "aurc": float(uncal_risk.mean()),
                "coverage_at_risk_0.1": float(coverage_at_risk(uncal_conf, correct, 0.1)),
                "coverage_at_risk_0.2": float(coverage_at_risk(uncal_conf, correct, 0.2)),
            },
            "overall_accuracy": float(correct.mean()),
            "selective_prediction": {
                "rule": "thresholds = quantiles of the CAL-split confidence; the test "
                        "split is only scored at those thresholds",
                "rows": selective,
            },
            "note": "AURC is the mean of the empirical risk over the n coverage points 1/n..1 (bench_common.aurc).",
        },
    )


# --------------------------------------------------------------------------
# Phase 8: print REAL decisions
# --------------------------------------------------------------------------

def phase8(ctx: Context, res: Result) -> None:
    b = len(ctx.test_texts)
    y = np.asarray(ctx.test_labels, dtype=np.int64)
    rows = rows_by_type(ctx.decision_output, b)
    rows_th = rows_by_type(ctx.threshold_output, b)
    choice_p = ctx.live["choice_probs"]
    correct = np.asarray(
        [CHOICE_OPTIONS.index(rows["choice"][i].predicted_value) for i in range(b)]
    ) == y
    conf = ctx.live["choice_conf"]

    hi_conf = np.argsort(-conf)
    right = [i for i in hi_conf if correct[i]]
    wrong = [i for i in hi_conf if not correct[i]]
    order = np.argsort(conf)
    abstained = [i for i in order if bool(rows_th["choice"][i].abstained.reshape(-1)[0])]
    mid = [i for i in order if bool(rows_th["choice"][i].abstained.reshape(-1)[0]) is False]

    # Deterministic, documented selection: the three highest-confidence
    # correct rows, the three highest-confidence WRONG rows, the two rows
    # closest to the median confidence, and the two lowest-confidence rows.
    n_abstain_at_thr = len(abstained)
    n_mid = max(1, b // 2)
    mid_idx = int(np.argsort(np.abs(np.arange(b) - n_mid))[0])
    picks: List[Tuple[str, int]] = []
    for i in right[:3]:
        picks.append(("highest-confidence CORRECT", int(i)))
    for i in wrong[:3]:
        picks.append(("highest-confidence WRONG", int(i)))
    for i in order[n_mid : n_mid + 2]:
        picks.append(("median-confidence", int(i)))
    for i in order[:2]:
        picks.append(("lowest-confidence (would ABSTAIN at the threshold)", int(i)))
    picks = picks[:N_PRINTED]

    printed: List[Dict[str, Any]] = []
    lines: List[str] = []
    lines.append("")
    lines.append("=" * 100)
    lines.append(f"REAL DECISIONS — live JevX.decide() over the {b}-row Yelp test batch")
    lines.append(f"  Q1 choice: {QUESTIONS['choice']}  options={CHOICE_OPTIONS}")
    lines.append(f"  Q2 score : {QUESTIONS['score']}  levels={SCORE_LEVELS}")
    lines.append(f"  Q3 noul  : {QUESTIONS['noul']}  (P(yes) = 1-star detractor)")
    lines.append("=" * 100)
    for kind, i in picks:
        d1, d2, d3 = rows["choice"][i], rows["score"][i], rows["noul"][i]
        a1 = rows_th["choice"][i]
        abst_flag = bool(a1.abstained.reshape(-1)[0]) if a1.abstained is not None else False
        top = np.argsort(-choice_p[i])[:2]
        p1 = choice_p[i]
        rec = {
            "bucket": kind,
            "test_index": i,
            "text": trunc(ctx.test_texts[i], PRINT_CHARS),
            "true_stars": int(y[i]) + 1,
            "true_label_index": int(y[i]),
            "true_noul_1star": bool(y[i] == 0),
            "q1_predicted_value": None if abst_flag else d1.predicted_value,
            "q1_predicted_index": None if d1.predicted_index is None else int(d1.predicted_index.reshape(-1)[0]),
            "q1_probabilities": [round(float(v), 4) for v in p1],
            "q1_top2": [
                {"option": CHOICE_OPTIONS[int(k)], "p": round(float(p1[int(k)]), 4)} for k in top
            ],
            "q1_confidence": round(conf_of(d1), 4),
            "q1_abstained_at_threshold": abst_flag,
            "q1_correct": bool(correct[i]),
            "q2_expected_score": round(float(ctx.live["score_expected"][i]), 4),
            "q2_predicted_level": d2.predicted_value,
            "q2_confidence": round(conf_of(d2), 4),
            "q3_p_yes": round(float(ctx.live["noul_probs"][i]), 4),
            "q3_confidence": round(conf_of(d3), 4),
            "q3_predicted_index": int(d3.predicted_index.reshape(-1)[0]),
        }
        printed.append(rec)
        lines.append("")
        lines.append(
            f"[{kind}] test row {i}  true={int(y[i]) + 1} star"
            f"{'' if int(y[i]) == 0 else 's'}  abstained_at_{ctx.threshold}={abst_flag}"
        )
        lines.append(f"  text           : {rec['text']}")
        lines.append(
            f"  Q1 choice      : pred={rec['q1_predicted_value']}"
            + ("  (ABSTAINED)" if abst_flag else "")
            + f"  conf={rec['q1_confidence']:.4f}"
            + f"  {'CORRECT' if rec['q1_correct'] else 'WRONG'}"
            + f"  argmax_p={CHOICE_OPTIONS[int(np.argmax(p1))]}"
        )
        lines.append("  Q1 top probs   : " + ", ".join(
            f"{t['option']}={t['p']:.4f}" for t in rec["q1_top2"])
            + "   full=" + str([CHOICE_OPTIONS[k] for k in range(5)]))
        lines.append("  Q1 probs       : " + str(rec["q1_probabilities"]))
        lines.append(
            f"  Q2 score       : E[score]={rec['q2_expected_score']:.4f} (argmax level {rec['q2_predicted_level']})"
            f"  conf={rec['q2_confidence']:.4f}"
        )
        lines.append(
            f"  Q3 noul        : P(1-star yes)={rec['q3_p_yes']:.4f}"
            f"  conf={rec['q3_confidence']:.4f}"
        )
        n_stars = rec["true_stars"]
        lines.append(
            f"  TRUE           : {n_stars} star" + ("" if n_stars == 1 else "s")
            + f" (class index {rec['true_label_index']})"
            + f"  | 1-star detractor={rec['true_noul_1star']}"
        )
    lines.append("")
    lines.append("=" * 100)
    log_lines = "\n".join(lines)
    log_block(log_lines)
    res.set(
        "8_printed_decisions",
        {
            "source": "live JevXForDecision.decide() over the test batch (phase 4 output), "
                      "plus the same call with confidence_threshold="
                      f"{ctx.threshold}",
            "probabilities_shown": "raw (uncalibrated) model probabilities; the per-type "
                                   "temperatures are reported in phase 3 / phase 6",
            "selection_rule": (
                "3 highest-confidence correct + 3 highest-confidence wrong + 2 rows at the "
                "median confidence + the 2 lowest-confidence rows, all taken from the live "
                f"decide() output; at threshold {ctx.threshold} "
                f"{n_abstain_at_thr}/{b} Q1 rows abstained, so the abstained flag is reported "
                "per row rather than used to select rows"
            ),
            "n_printed": len(printed),
            "buckets": sorted(set(r["bucket"] for r in printed)),
            "decisions": printed,
            "verbatim_console_block": log_lines,
        },
    )


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

PHASE_FUNCS = {
    "1": phase1,
    "2": phase2,
    "3": phase3,
    "4": phase4,
    "5": phase5,
    "6": phase6,
    "7": phase7,
    "8": phase8,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.quick:
        args.train_size, args.cal_size, args.test_size = 300, 150, 200
        args.epochs = 2
        args.seeds = "0"
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    if args.fresh_log and LOG_PATH.exists():
        try:
            LOG_PATH.unlink()
        except OSError:
            pass
    res = Result(Path(args.out))
    ctx = Context(args)
    t_start = time.perf_counter()
    log(f"real scenario start: phases={phases} seeds={ctx.seeds} "
        f"train={args.train_size} cal={args.cal_size} test={args.test_size} "
        f"epochs={args.epochs} lr={args.lr} batch={args.batch_size} "
        f"max_len={args.max_length}")
    res.data["protocol"] = {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "dataset_license": DATASET_LICENSE,
        "backbone_id": BACKBONE_ID,
        "backbone_sha": BACKBONE_SHA,
        "split_seed": SPLIT_SEED,
        "seeds": ctx.seeds,
        "train_size": args.train_size,
        "cal_size": args.cal_size,
        "test_size": args.test_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "pooling": args.pooling,
        "fusion": args.fusion,
        "ece_bins": N_BINS,
        "confidence_threshold": args.threshold,
        "questions": dict(QUESTIONS),
        "option_sets": {t: OPTION_SETS[t] for t in TYPES},
        "label_derivation": (
            "dataset ClassLabel y in 0..4 (== y+1 stars); Q1 label = y, "
            "Q2 level = y, Q3 label = 1[y == 0]. All three questions are scored "
            "against REAL dataset labels on the same state."
        ),
    }
    res.data["environment"] = bc.environment_info()
    res.flush()

    for phase in phases:
        fn = PHASE_FUNCS.get(phase)
        if fn is None:
            log(f"unknown phase {phase!r}; known: {sorted(PHASE_FUNCS)}")
            continue
        t0 = time.perf_counter()
        try:
            ctx = fn(ctx, res) or ctx
            res.time(f"phase_{phase}_s", time.perf_counter() - t0)
        except Exception as exc:  # noqa: BLE001 - recorded, never silent
            import traceback

            res.fail(f"phase_{phase}", exc)
            log(f"[FAILED] phase {phase}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            if not args.keep_going:
                return 1
    res.data["total_wall_time_s"] = round(time.perf_counter() - t_start, 2)
    res.data["phases_run"] = phases
    res.flush()
    log(f"real scenario done in {res.data['total_wall_time_s']}s -> {res.path} "
        f"(failures: {len(res.data['failures'])})")
    return 1 if res.data["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
