# Copyright (c) 2026 ysnrfd
"""EXP-6b: the `learned` confidence method MEASURED on the SST-2 setup.

Closes the "learned was recorded not-run" gap of reports/ABLATIONS.md EXP-6.
The run mirrors `run_quality.run_exp6` exactly (same sst2 revision, same
seed-0 stratified train/cal carve, same test = full official validation split,
same featurize-once features, same seed-0 HeadTrainer head training), and all
FIVE confidence methods are evaluated on the SAME test logits in this single
run, so the comparison against the four already-reported EXP-6 rows is
**paired**. The four baseline rows are re-checked field-by-field against
`benchmarks/results/exp6_confidence.json`.

Learned-confidence training protocol (documented exactly, fit on CAL only):

* inputs   : x = [fused ; max_prob ; 1 - H/log K ; margin] on the 1000-row
             CALIBRATION split, where `fused` is the 128-d concat-fusion
             output H of the trained JevX head stack (computed by the same
             modules model.forward uses);
* targets  : y = 1[ argmax(cal_logits) == cal_label ] (cal-split correctness);
* module   : ConfidenceEngine("learned", fused_dim=128) -> MLP 131 -> 32
             (tanh) -> 1 -> sigmoid (the library class, unmodified);
* loss     : BCE on the sigmoid output; optimizer Adam, lr=1e-3;
* schedule : 100 epochs, batch 64, shuffle by torch.Generator(seed=0);
             torch.manual_seed(0) before constructing the module so the MLP
             weight init is seeded;
* test     : used ONLY for the final reported metrics (never for fitting,
             tuning or early stopping - there is no early stopping).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import bench_common as bc  # noqa: E402

EXP6_JSON = REPO_ROOT / "benchmarks" / "results" / "exp6_confidence.json"
CONFIDENCE_FIELDS = (
    "ece_of_confidence",
    "aurc",
    "coverage_at_risk_0.1",
    "coverage_at_risk_0.2",
    "conf_mean",
    "conf_std",
)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--backbone", default=bc.DEFAULT_BACKBONE)
    p.add_argument("--dataset", default=bc.DEFAULT_DATASET)
    p.add_argument("--dataset-revision", default=bc.DEFAULT_DATASET_REVISION)
    p.add_argument("--train-size", type=int, default=3000)
    p.add_argument("--cal-size", type=int, default=1000)
    p.add_argument("--test-size", type=int, default=1500)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--fusion", default="concat")
    p.add_argument("--pooling", default="cls")
    p.add_argument("--ablation-seed", type=int, default=0)
    p.add_argument("--learned-epochs", type=int, default=100)
    p.add_argument("--learned-lr", type=float, default=1e-3)
    p.add_argument("--learned-batch-size", type=int, default=64)
    p.add_argument("--learned-hidden", type=int, default=32)
    p.add_argument("--learned-seed", type=int, default=0)
    p.add_argument("--output-dir", default=str(REPO_ROOT / "benchmarks" / "results"))
    return p.parse_args(argv)


# --------------------------------------------------------------------------
# FeaturizedJevX variant that also exposes the fused state H
# --------------------------------------------------------------------------

class FusedFeaturizedJevX(bc.FeaturizedJevX):
    """Exactly FeaturizedJevX's math, with the fused state exposed.

    `forward()` is bit-for-bit the parent's computation (state_proj ->
    question -> schema -> fusion -> choice head); `forward_parts()` returns
    the intermediate (fused, option_features, option_mask) so EXP-6b can feed
    the fused rows to ConfidenceEngine("learned"). `equivalence_check` in the
    run below re-verifies this subclass against the full model.choice().
    """

    def forward_parts(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
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
        return fused.fused, fused.option_features, option_mask

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        fused, option_features, option_mask = self.forward_parts(batch)
        out = self.jevx.heads["choice"](
            fused, option_features, option_mask=option_mask
        )
        return out.logits


def collect_logits_and_fused(
    head: FusedFeaturizedJevX,
    records: List[Dict[str, Any]],
    *,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One forward per batch; returns (logits [N,K], labels [N], fused [N,D])."""
    head.eval()
    loader = bc.make_loader(records, batch_size, shuffle=False)
    logits_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    fused_all: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            fused, option_features, option_mask = head.forward_parts(batch)
            logits = head.jevx.heads["choice"](
                fused, option_features, option_mask=option_mask
            ).logits
            logits_all.append(logits.detach().cpu().numpy().astype(np.float32))
            fused_all.append(fused.detach().cpu().numpy().astype(np.float32))
            labels_all.append(
                np.asarray(batch["labels"]).reshape(-1).astype(np.int64)
            )
    return (
        np.concatenate(logits_all, axis=0),
        np.concatenate(labels_all, axis=0),
        np.concatenate(fused_all, axis=0),
    )


# --------------------------------------------------------------------------
# Learned-confidence training (CAL split only)
# --------------------------------------------------------------------------

def train_learned_confidence(
    *,
    cal_probs: np.ndarray,
    cal_logits: np.ndarray,
    cal_fused: np.ndarray,
    cal_labels: np.ndarray,
    fused_dim: int,
    hidden: int,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
) -> Tuple[bc.ConfidenceEngine, List[float]]:
    """BCE-trained LearnedConfidence MLP on calibration-split correctness.

    Returns (engine, per-epoch mean training BCE). Seeded: module init under
    torch.manual_seed(seed), shuffling via a dedicated Generator(seed).
    """
    torch.manual_seed(seed)
    engine = bc.ConfidenceEngine("learned", fused_dim=fused_dim, hidden_size=hidden)
    optimizer = torch.optim.Adam(engine.parameters(), lr=lr)
    criterion = nn.BCELoss()

    probs_t = torch.as_tensor(cal_probs, dtype=torch.float32)
    logits_t = torch.as_tensor(cal_logits, dtype=torch.float32)
    fused_t = torch.as_tensor(cal_fused, dtype=torch.float32)
    pred = cal_logits.argmax(axis=1)
    target = torch.as_tensor(
        (pred == np.asarray(cal_labels)).astype(np.float32), dtype=torch.float32
    )
    n = probs_t.shape[0]
    generator = torch.Generator().manual_seed(seed)

    history: List[float] = []
    engine.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=generator)
        total, seen = 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            conf = engine(
                probs_t[idx], logits=logits_t[idx], fused=fused_t[idx]
            )
            loss = criterion(conf, target[idx])
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"learned-confidence training produced a non-finite loss "
                    f"(epoch {_ + 1}, start {start})"
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(idx)
            seen += len(idx)
        history.append(total / max(seen, 1))
    engine.eval()
    return engine, history


def evaluate_confidence(
    engine: bc.ConfidenceEngine,
    *,
    probs: np.ndarray,
    logits: np.ndarray,
    fused: np.ndarray,
    correct: np.ndarray,
) -> Tuple[Dict[str, float], np.ndarray]:
    """EXP-6 metric set for one confidence vector (15 bins, 0-1 loss)."""
    with torch.no_grad():
        conf = engine(
            torch.as_tensor(probs, dtype=torch.float32),
            logits=torch.as_tensor(logits, dtype=torch.float32),
            fused=torch.as_tensor(fused, dtype=torch.float32),
        )
    conf_np = conf.detach().cpu().numpy().astype(np.float64)
    return {
        "ece_of_confidence": bc.ece_of_confidence(conf_np, correct, n_bins=15),
        "aurc": bc.aurc(conf_np, correct),
        "coverage_at_risk_0.1": bc.coverage_at(conf_np, correct, 0.1),
        "coverage_at_risk_0.2": bc.coverage_at(conf_np, correct, 0.2),
        "conf_mean": float(np.mean(conf_np)),
        "conf_std": float(np.std(conf_np, ddof=1)),
    }, conf_np


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def _counts(labels: Sequence[int]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for y in labels:
        out[str(int(y))] = out.get(str(int(y)), 0) + 1
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    seed = args.ablation_seed
    t_start = time.perf_counter()
    timings: Dict[str, float] = {}

    t0 = time.perf_counter()
    train_texts_all, train_labels_all, val_texts, val_labels = bc.load_sst2(
        args.dataset, args.dataset_revision
    )
    timings["load_dataset_s"] = round(time.perf_counter() - t0, 2)

    if args.test_size and args.test_size < len(val_texts):
        test_texts, test_labels = val_texts[: args.test_size], val_labels[: args.test_size]
    else:
        test_texts, test_labels = val_texts, val_labels

    train_idx, cal_idx = bc.make_splits(
        train_labels_all, args.train_size, args.cal_size, seed=seed
    )
    train_texts = [train_texts_all[i] for i in train_idx]
    train_labels = [train_labels_all[i] for i in train_idx]
    cal_texts = [train_texts_all[i] for i in cal_idx]
    cal_labels = [train_labels_all[i] for i in cal_idx]

    t0 = time.perf_counter()
    model = bc.JevX.from_pretrained(
        args.backbone,
        freeze_base_model=True,
        pooling_method=args.pooling,
        fusion_type=args.fusion,
    )
    model.eval()
    timings["load_backbone_s"] = round(time.perf_counter() - t0, 2)
    backbone_cfg = model.config.backbone_config
    backbone_revision = (
        backbone_cfg.get("_commit_hash") if isinstance(backbone_cfg, dict) else None
    ) or getattr(getattr(model.model, "config", None), "_commit_hash", None)

    t0 = time.perf_counter()
    feats: Dict[str, Dict[str, np.ndarray]] = {}
    for name, texts in (
        ("train", train_texts),
        ("cal", cal_texts),
        ("test", test_texts),
    ):
        feats[name] = bc.featurize(
            model, texts,
            batch_size=args.batch_size, max_length=args.max_length,
            strategies=[None, args.pooling],
        )
    timings["featurize_all_splits_s"] = round(time.perf_counter() - t0, 2)

    train_records = bc.build_records(feats["train"][args.pooling], train_labels)
    cal_records = bc.build_records(feats["cal"][args.pooling], cal_labels)
    test_records = bc.build_records(feats["test"][args.pooling], test_labels)

    # ---- seed-0 head training, identical recipe to run_quality.run_exp6 ----
    # The head's JevX wrapper must be built AFTER torch.manual_seed(seed):
    # run_quality builds a FRESH from_pretrained wrapper inside run_exp6 right
    # after seeding, and the wrapper's non-backbone modules (state_proj,
    # question/schema encoders, fusion, choice head) draw their init from that
    # seeded RNG. Building it earlier (e.g. the featurization model) draws from
    # an unseeded stream and yields a different trained model - this was caught
    # by the pairing check below on the first attempt and fixed here.
    t0 = time.perf_counter()
    torch.manual_seed(seed)
    train_model = bc.JevX.from_pretrained(
        args.backbone,
        freeze_base_model=True,
        pooling_method=args.pooling,
        fusion_type=args.fusion,
    )
    train_model.eval()
    head = FusedFeaturizedJevX(train_model, bc.DEFAULT_QUESTION, bc.DEFAULT_OPTIONS)
    trainer, history = bc.train_heads(
        head, train_records,
        seed=seed, lr=args.lr, epochs=args.epochs, batch_size=args.batch_size,
    )
    cal_logits, cal_lab, cal_fused = collect_logits_and_fused(
        head, cal_records, batch_size=args.batch_size
    )
    test_logits, test_lab, test_fused = collect_logits_and_fused(
        head, test_records, batch_size=args.batch_size
    )
    timings["train_and_collect_s"] = round(time.perf_counter() - t0, 2)

    eq = bc.equivalence_check(
        train_model, head, feats["test"][args.pooling], test_texts, n=16
    )
    test_acc = float((test_logits.argmax(axis=1) == np.asarray(test_lab)).mean())

    # ---- the four already-reported methods, recomputed on THIS run's logits
    # (paired check against exp6_confidence.json) ----
    cal_probs = _softmax(cal_logits)
    test_probs = _softmax(test_logits)
    correct_test = (test_logits.argmax(axis=1) == np.asarray(test_lab)).astype(
        np.float64
    )
    baseline_methods = ["max_probability", "margin", "entropy", "energy"]
    results: Dict[str, Dict[str, float]] = {}
    for method in baseline_methods:
        conf = bc.confidence_vectors(test_logits, method, cal_logits=cal_logits)
        results[method] = {
            "ece_of_confidence": bc.ece_of_confidence(conf, correct_test, n_bins=15),
            "aurc": bc.aurc(conf, correct_test),
            "coverage_at_risk_0.1": bc.coverage_at(conf, correct_test, 0.1),
            "coverage_at_risk_0.2": bc.coverage_at(conf, correct_test, 0.2),
            "conf_mean": float(np.mean(conf)),
            "conf_std": float(np.std(conf, ddof=1)),
        }

    # ---- train + evaluate `learned` (fit on CAL only) ----
    t0 = time.perf_counter()
    fused_dim = int(cal_fused.shape[1])
    engine, loss_history = train_learned_confidence(
        cal_probs=cal_probs,
        cal_logits=cal_logits,
        cal_fused=cal_fused,
        cal_labels=np.asarray(cal_labels),
        fused_dim=fused_dim,
        hidden=args.learned_hidden,
        epochs=args.learned_epochs,
        lr=args.learned_lr,
        batch_size=args.learned_batch_size,
        seed=args.learned_seed,
    )
    timings["train_learned_s"] = round(time.perf_counter() - t0, 2)

    test_metrics, test_conf = evaluate_confidence(
        engine,
        probs=test_probs, logits=test_logits, fused=test_fused,
        correct=correct_test,
    )
    correct_cal = (
        cal_logits.argmax(axis=1) == np.asarray(cal_labels)
    ).astype(np.float64)
    cal_metrics, _ = evaluate_confidence(
        engine,
        probs=cal_probs, logits=cal_logits, fused=cal_fused,
        correct=correct_cal,
    )
    results["learned"] = test_metrics

    # ---- pairing check vs the stored EXP-6 rows ----
    pairing: Dict[str, Any] = {"source_json": str(EXP6_JSON.name)}
    if EXP6_JSON.exists():
        stored = json.loads(EXP6_JSON.read_text(encoding="utf-8"))["results"]
        per_method: Dict[str, float] = {}
        for method in baseline_methods:
            diffs = [
                abs(float(results[method][k]) - float(stored[method][k]))
                for k in CONFIDENCE_FIELDS
            ]
            per_method[method] = float(max(diffs))
        pairing.update(
            {
                "status": "run",
                "max_abs_diff_by_method": per_method,
                "all_baselines_reproduced_exactly": all(
                    v == 0.0 for v in per_method.values()
                ),
                "note": (
                    "each method's 6 fields compared against "
                    "exp6_confidence.json; 0.0 means this re-run reproduced the "
                    "published EXP-6 row bit-for-bit (same seeds, same recipe)"
                ),
            }
        )
    else:
        pairing["status"] = "missing_source_json"

    ok = results
    rank_ece = sorted(ok, key=lambda k: ok[k]["ece_of_confidence"])
    rank_aurc = sorted(ok, key=lambda k: ok[k]["aurc"])
    rank_cov10 = sorted(ok, key=lambda k: -ok[k]["coverage_at_risk_0.1"])

    payload = {
        "experiment": "exp6b",
        "description": (
            "learned confidence (MEASURED) on the SST-2 EXP-6 setup; all five "
            "methods computed on the same test logits (SINGLE SEED)"
        ),
        "n_seeds": 1,
        "seed": seed,
        "protocol": {
            "head_training": (
                "identical to run_quality.run_exp6: HeadTrainer + choice_loss, "
                f"{args.epochs} epochs, lr {args.lr}, batch {args.batch_size}, "
                f"seed {seed}, featurize-once features"
            ),
            "learned_fit_split": "CALIBRATION split only (n=1000)",
            "learned_targets": "1[argmax(cal_logits) == cal_label] (cal correctness)",
            "learned_inputs": "[fused(128) ; max_prob ; 1-H/logK ; margin]",
            "learned_module": (
                f'ConfidenceEngine("learned", fused_dim={fused_dim}, '
                f"hidden_size={args.learned_hidden}) -> MLP "
                f"{fused_dim + 3}->{args.learned_hidden}(tanh)->1->sigmoid"
            ),
            "learned_loss": "BCE(sigmoid_output, correctness)",
            "learned_optimizer": f"Adam lr={args.learned_lr}",
            "learned_epochs": args.learned_epochs,
            "learned_batch_size": args.learned_batch_size,
            "learned_seed": args.learned_seed,
            "learned_shuffle": "torch.Generator().manual_seed(seed) per epoch permutation",
            "test_usage": "final metrics only; never fit, never tuned, no early stopping",
        },
        "learned_training": {
            "bce_first_epoch": loss_history[0],
            "bce_last_epoch": loss_history[-1],
            "bce_by_epoch": loss_history,
            "learned_params": int(
                sum(p.numel() for p in engine.parameters() if p.requires_grad)
            ),
            "fused_dim": fused_dim,
        },
        "head_training": {
            "train_loss_first": history[0]["loss"],
            "train_loss_last": history[-1]["loss"],
            "test_accuracy": test_acc,
            "equivalence_check": eq,
        },
        "cal_split_diagnostic": {
            "note": (
                "in-sample for `learned` (it was fit on cal); recorded for "
                "transparency only - the reported comparison is test-side"
            ),
            "learned": cal_metrics,
        },
        "results": results,
        "rank_by_ece_of_confidence": rank_ece,
        "rank_by_aurc": rank_aurc,
        "rank_by_coverage_at_risk_0.1": rank_cov10,
        "pairing_check_vs_exp6": pairing,
        "provenance": {
            "backbone_id": args.backbone,
            "backbone_revision": backbone_revision,
            "dataset_id": args.dataset,
            "dataset_revision": args.dataset_revision,
            "question": bc.DEFAULT_QUESTION,
            "options": list(bc.DEFAULT_OPTIONS),
            "seeds": [seed],
            "hyperparams": {
                "train_size": args.train_size,
                "cal_size": args.cal_size,
                "test_size": args.test_size,
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "max_length": args.max_length,
                "fusion": args.fusion,
                "pooling": args.pooling,
                "ablation_seed": seed,
            },
            "environment": bc.environment_info(),
            "splits": {
                "train_n": len(train_texts),
                "cal_n": len(cal_texts),
                "test_n": len(test_texts),
                "train_label_counts": _counts(train_labels),
                "cal_label_counts": _counts(cal_labels),
                "test_label_counts": _counts(test_labels),
                "test_source": "official validation split (capped)",
                "train_cal_source": "stratified carve from official train split",
                "split_seed": seed,
            },
            "context_timings_s": timings,
            "platform": platform.platform(),
        },
        "wall_time_s": round(time.perf_counter() - t_start, 2),
    }
    out = Path(args.output_dir) / "exp6b_learned_confidence.json"
    bc.write_json(out, payload)
    print(f"[wrote] {out}", flush=True)
    print(
        f"[done] learned test ece={test_metrics['ece_of_confidence']:.4f} "
        f"aurc={test_metrics['aurc']:.4f} "
        f"cov@0.1={test_metrics['coverage_at_risk_0.1']:.4f} "
        f"cov@0.2={test_metrics['coverage_at_risk_0.2']:.4f}; "
        f"pairing exact={pairing.get('all_baselines_reproduced_exactly')}",
        flush=True,
    )
    return 0


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Probability matrix via the library's stable_softmax (same function the
    metric paths use), so confidence inputs match metrics_from_logits exactly."""
    with torch.no_grad():
        probs = bc.stable_softmax(
            torch.as_tensor(np.asarray(logits, dtype=np.float32)), dim=-1
        )
    return probs.numpy().astype(np.float32)


if __name__ == "__main__":
    raise SystemExit(main())
