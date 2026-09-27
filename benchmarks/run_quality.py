# Copyright (c) 2026 ysnrfd
"""JevX real-dataset quality benchmarks: EXP-1..EXP-6 (Phases 21-23, 49, 51-54, 57-61).

Reproducible entry point. Every experiment writes its own JSON under
benchmarks/results/ AS IT COMPLETES, so a crash mid-run never loses the
finished parts.

Protocol (train / calibration / test are disjoint - §53):

* data      : stanfordnlp/sst2 pinned by revision; binary sentiment framed
              as a Choice question {negative, positive};
* splits    : train + calibration are stratified samples carved from the
              OFFICIAL train split (seeded, disjoint); test = the full
              OFFICIAL validation split, which is never used for fitting
              heads or calibrators;
* backbone  : a real pretrained small BERT (google/bert_uncased_L-2_H-128_A-2,
              4.4M params) frozen (Mode A) - featurized ONCE per split;
* heads     : HeadTrainer + choice_loss, shared lr/epochs/batch protocol
              across JevX and every baseline;
* calibrate : CalibrationEngine fit on the CALIBRATION split only.

Experiments:
  exp1  JevX heads, 3 training seeds, uncal / temperature / isotonic
  exp2  baselines B1 linear probe, B2 MLP, B4 = B1 + temperature,
        B3 constrained-decoding (distilgpt2), plus batch-1 latency
  exp3  pooling ablation {native, cls, masked_mean, last_valid}
  exp4  fusion ablation {concat, additive, gated, bilinear} + zero-question control
  exp5  calibration ablation {none, temperature, vector, isotonic} + reliability bins
  exp6  confidence ablation {max_probability, margin, entropy, energy}

Example:
    python benchmarks/run_quality.py --train-size 3000 --cal-size 1000 \
        --seeds 0,1,2 --epochs 4 --experiments all
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

# Allow running as a script from the repo root or anywhere.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import bench_common as bc  # noqa: E402


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--backbone", default=bc.DEFAULT_BACKBONE)
    p.add_argument("--dataset", default=bc.DEFAULT_DATASET)
    p.add_argument("--dataset-revision", default=bc.DEFAULT_DATASET_REVISION)
    p.add_argument("--seeds", default="0,1,2",
                   help="comma-separated training seeds for EXP-1/EXP-2 (default 0,1,2)")
    p.add_argument("--ablation-seed", type=int, default=0,
                   help="single seed for EXP-3..EXP-6 (labeled single-seed)")
    p.add_argument("--train-size", type=int, default=3000)
    p.add_argument("--cal-size", type=int, default=1000)
    p.add_argument("--test-size", type=int, default=1500,
                   help="cap on the official validation split used as test")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3,
                   help="HeadTrainer AdamW lr (library default 1e-3; chosen a priori, "
                        "no hyperparameter search on any split)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--fusion", default="concat", choices=[
        "concat", "additive", "gated", "bilinear", "dot", "mlp"])
    p.add_argument("--pooling", default="cls", choices=[
        "cls", "masked_mean", "last_valid", "mean", "max"])
    p.add_argument("--experiments", default="all",
                   help="comma list: exp1,exp2,exp3,exp4,exp5,exp6 or 'all'")
    p.add_argument("--skip-b3", action="store_true",
                   help="skip the distilgpt2 constrained-decoding baseline")
    p.add_argument("--skip-latency", action="store_true")
    p.add_argument("--latency-iters", type=int, default=100)
    p.add_argument("--output-dir", default=str(REPO_ROOT / "benchmarks" / "results"))
    p.add_argument("--run-label", default=None,
                   help="suffix for run_summary_<label>.json so concurrent/repeat "
                        "runs never clobber each other's summary")
    return p.parse_args(argv)


def selected_experiments(spec: str) -> List[str]:
    all_exp = ["exp1", "exp2", "exp3", "exp4", "exp5", "exp6"]
    if spec.strip() == "all":
        return all_exp
    chosen = [s.strip().lower() for s in spec.split(",") if s.strip()]
    bad = [c for c in chosen if c not in all_exp]
    if bad:
        raise ValueError(f"unknown experiment(s) {bad}; choose from {all_exp} or 'all'")
    return chosen


# --------------------------------------------------------------------------
# Shared context: one model load, one featurization pass, reused everywhere
# --------------------------------------------------------------------------

class BenchContext:
    """Loads data + backbone once and exposes the shared splits/features."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        self.hyperparams = {
            "train_size": args.train_size,
            "cal_size": args.cal_size,
            "test_size": args.test_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "fusion": args.fusion,
            "pooling": args.pooling,
            "ablation_seed": args.ablation_seed,
        }
        self.out_dir = Path(args.output_dir)
        self.timings: Dict[str, float] = {}

        t0 = time.perf_counter()
        (self.train_texts_all, self.train_labels_all,
         self.val_texts, self.val_labels) = bc.load_sst2(
            args.dataset, args.dataset_revision
        )
        self.timings["load_dataset_s"] = round(time.perf_counter() - t0, 2)

        # Test = official validation split, capped (§: never tune on it).
        if args.test_size and args.test_size < len(self.val_texts):
            self.test_texts = self.val_texts[: args.test_size]
            self.test_labels = self.val_labels[: args.test_size]
        else:
            self.test_texts = self.val_texts
            self.test_labels = self.val_labels

        # Disjoint stratified train/cal from the official train split.
        self.train_idx, self.cal_idx = bc.make_splits(
            self.train_labels_all, args.train_size, args.cal_size, seed=args.ablation_seed
        )
        self.train_texts = [self.train_texts_all[i] for i in self.train_idx]
        self.train_labels = [self.train_labels_all[i] for i in self.train_idx]
        self.cal_texts = [self.train_texts_all[i] for i in self.cal_idx]
        self.cal_labels = [self.train_labels_all[i] for i in self.cal_idx]

        t0 = time.perf_counter()
        self.model = bc.JevX.from_pretrained(
            args.backbone,
            freeze_base_model=True,
            pooling_method=args.pooling,
            fusion_type=args.fusion,
        )
        self.model.eval()
        self.timings["load_backbone_s"] = round(time.perf_counter() - t0, 2)
        backbone_cfg = self.model.config.backbone_config
        self.backbone_revision = (
            backbone_cfg.get("_commit_hash")
            if isinstance(backbone_cfg, dict)
            else None
        ) or getattr(getattr(self.model.model, "config", None), "_commit_hash", None)

        # Featurize all three splits ONCE. The configured --pooling strategy
        # is the PRIMARY feature set (it matches what model.choice() computes,
        # so the equivalence check is apples-to-apples); when exp3 is selected
        # every ablation pooling is derived from the SAME backbone pass (pool()
        # on one sequence output), so extra strategies are nearly free.
        t0 = time.perf_counter()
        strategies: List[Optional[str]] = [None, args.pooling]
        if "exp3" in selected_experiments(args.experiments):
            for extra in ("cls", "masked_mean", "last_valid"):
                if extra not in [s for s in strategies if s is not None]:
                    strategies.append(extra)
        self.primary_key = args.pooling
        self.feats: Dict[str, Dict[str, np.ndarray]] = {}
        for name, texts in (
            ("train", self.train_texts),
            ("cal", self.cal_texts),
            ("test", self.test_texts),
        ):
            self.feats[name] = bc.featurize(
                self.model,
                texts,
                batch_size=args.batch_size,
                max_length=args.max_length,
                strategies=strategies,
            )
        self.timings["featurize_all_splits_s"] = round(time.perf_counter() - t0, 2)

    # -- records -----------------------------------------------------------
    def records(self, split: str, strategy_key: Optional[str] = None):
        key = strategy_key or self.primary_key
        return bc.build_records(
            self.feats[split][key],
            getattr(self, f"{split}_labels"),
        )

    def provenance(self) -> Dict[str, Any]:
        prov = bc.provenance(
            self.args.backbone,
            self.args.dataset,
            self.args.dataset_revision,
            self.seeds,
            self.hyperparams,
        )
        prov["backbone_revision"] = self.backbone_revision
        prov["splits"] = {
            "train_n": len(self.train_texts),
            "cal_n": len(self.cal_texts),
            "test_n": len(self.test_texts),
            "train_label_counts": _counts(self.train_labels),
            "cal_label_counts": _counts(self.cal_labels),
            "test_label_counts": _counts(self.test_labels),
            "test_source": "official validation split (capped)",
            "train_cal_source": "stratified carve from official train split",
            "split_seed": self.args.ablation_seed,
        }
        return prov

    def save(self, name: str, payload: Dict[str, Any]) -> None:
        payload = {"provenance": self.provenance(), **payload}
        bc.write_json(self.out_dir / name, payload)
        print(f"[wrote] {self.out_dir / name}", flush=True)


def _counts(labels: Sequence[int]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for y in labels:
        out[str(int(y))] = out.get(str(int(y)), 0) + 1
    return out


# --------------------------------------------------------------------------
# EXP-1: JevX heads, 3 seeds, uncal / temperature / isotonic
# --------------------------------------------------------------------------

def run_exp1(ctx: BenchContext) -> None:
    t0 = time.perf_counter()
    train_records = ctx.records("train")
    cal_records = ctx.records("cal")
    test_records = ctx.records("test")
    per_seed: List[Dict[str, Any]] = []
    per_seed_metrics: Dict[str, List[Dict[str, float]]] = {
        "uncalibrated": [], "temperature": [], "isotonic": [],
    }
    equivalence: List[Dict[str, Any]] = []
    for seed in ctx.seeds:
        print(f"[exp1] training JevX heads seed={seed}", flush=True)
        torch.manual_seed(seed)
        # Re-init heads deterministically per seed: reload resets RNG-drawn
        # head weights (verified: same seed -> identical init).
        model = bc.JevX.from_pretrained(
            ctx.args.backbone,
            freeze_base_model=True,
            pooling_method=ctx.args.pooling,
            fusion_type=ctx.args.fusion,
        )
        model.eval()
        head = bc.FeaturizedJevX(
            model, bc.DEFAULT_QUESTION, bc.DEFAULT_OPTIONS
        )
        trainer, history = bc.train_heads(
            head,
            train_records,
            seed=seed,
            lr=ctx.args.lr,
            epochs=ctx.args.epochs,
            batch_size=ctx.args.batch_size,
        )
        cal_logits, cal_labels = bc.collect_logits(
            head, cal_records, batch_size=ctx.args.batch_size
        )
        test_logits, test_labels = bc.collect_logits(
            head, test_records, batch_size=ctx.args.batch_size
        )
        row: Dict[str, Any] = {
            "seed": seed,
            "train_loss_first": history[0]["loss"],
            "train_loss_last": history[-1]["loss"],
            "trainable_params": bc.count_trainable(head),
        }
        for method, key in (
            ("none", "uncalibrated"),
            ("temperature", "temperature"),
            ("isotonic", "isotonic"),
        ):
            m = bc.calibrated_metrics(
                test_logits, test_labels, method,
                fit_logits=cal_logits, fit_labels=cal_labels,
            )
            per_seed_metrics[key].append(m)
            row[key] = m
        per_seed.append(row)
        equivalence.append(
            bc.equivalence_check(
                model, head, ctx.feats["test"][ctx.primary_key],
                ctx.test_texts, n=16,
            )
        )
        del model, head, trainer
    payload = {
        "experiment": "exp1",
        "description": "JevX heads over frozen bert backbone (Mode A), 3 training seeds",
        "n_seeds": len(ctx.seeds),
        "per_seed": per_seed,
        "aggregate": {
            key: bc.mean_std_over_seeds(rows) for key, rows in per_seed_metrics.items()
        },
        "featurized_vs_full_model": equivalence,
        "wall_time_s": round(time.perf_counter() - t0, 2),
    }
    ctx.save("exp1_jevx_quality.json", payload)


# --------------------------------------------------------------------------
# EXP-2: baselines B1/B2/B4 (+B3) and latency
# --------------------------------------------------------------------------

def run_exp2(ctx: BenchContext, *, include_b3: bool = True) -> None:
    t0 = time.perf_counter()
    train_records = ctx.records("train")
    cal_records = ctx.records("cal")
    test_records = ctx.records("test")
    in_dim = ctx.feats["train"][ctx.primary_key].shape[1]
    n_classes = len(bc.DEFAULT_OPTIONS)

    per_seed: List[Dict[str, Any]] = []
    metric_sets: Dict[str, List[Dict[str, float]]] = {
        "b1_linear": [], "b1_temperature": [], "b2_mlp": [],
    }
    b1_head_for_latency: Optional[torch.nn.Module] = None
    for seed in ctx.seeds:
        print(f"[exp2] training B1/B2 probes seed={seed}", flush=True)
        torch.manual_seed(seed)
        b1 = bc.LinearProbe(in_dim, n_classes)
        bc.train_heads(
            b1, train_records, seed=seed, lr=ctx.args.lr,
            epochs=ctx.args.epochs, batch_size=ctx.args.batch_size,
        )
        b1_cal, cal_labels = bc.collect_logits(b1, cal_records, batch_size=ctx.args.batch_size)
        b1_test, test_labels = bc.collect_logits(b1, test_records, batch_size=ctx.args.batch_size)

        torch.manual_seed(seed + 10_000)
        b2 = bc.MLPProbe(in_dim, n_classes)
        bc.train_heads(
            b2, train_records, seed=seed + 10_000, lr=ctx.args.lr,
            epochs=ctx.args.epochs, batch_size=ctx.args.batch_size,
        )
        b2_test, _ = bc.collect_logits(b2, test_records, batch_size=ctx.args.batch_size)

        row: Dict[str, Any] = {"seed": seed}
        row["b1_linear"] = bc.metrics_from_logits(b1_test, test_labels)
        row["b1_temperature"] = bc.calibrated_metrics(
            b1_test, test_labels, "temperature",
            fit_logits=b1_cal, fit_labels=cal_labels,
        )
        row["b2_mlp"] = bc.metrics_from_logits(b2_test, test_labels)
        for key in metric_sets:
            metric_sets[key].append(row[key])
        per_seed.append(row)
        b1_head_for_latency = b1
        del b2

    # ---- B3: constrained decoding with distilgpt2 (optional) ----
    b3_status: Dict[str, Any]
    if include_b3 and not ctx.args.skip_b3:
        try:
            t_b3 = time.perf_counter()
            gen, tok = bc.load_generative(bc.GENERIC_BASELINE)
            # Fit temperature on the CALIBRATION split, score TEST.
            b3_cal = bc.generative_choice_logits(
                gen, tok, ctx.cal_texts, batch_size=ctx.args.batch_size,
                max_length=ctx.args.max_length,
            )
            b3_test = bc.generative_choice_logits(
                gen, tok, ctx.test_texts, batch_size=ctx.args.batch_size,
                max_length=ctx.args.max_length,
            )
            zero_shot = bc.metrics_from_logits(
                b3_test, np.asarray(ctx.test_labels)
            )
            calibrated = bc.calibrated_metrics(
                b3_test, np.asarray(ctx.test_labels), "temperature",
                fit_logits=b3_cal, fit_labels=np.asarray(ctx.cal_labels),
            )
            b3_status = {
                "status": "run",
                "model_id": bc.GENERIC_BASELINE,
                "template": "Review: {text}\\nSentiment: + next-token logit of ' negative'/' positive'",
                "zero_shot": zero_shot,
                "temperature_calibrated": calibrated,
                "wall_time_s": round(time.perf_counter() - t_b3, 2),
                "note": (
                    "B3 is a DIFFERENT (generative, 82M) backbone used zero-shot - "
                    "reported for reference, not a same-backbone comparison"
                ),
            }
            del gen
        except Exception as exc:  # noqa: BLE001 - recorded, never silent
            b3_status = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    else:
        b3_status = {
            "status": "skipped",
            "reason": "--skip-b3" if ctx.args.skip_b3 else "not requested",
        }

    # ---- latency: JevX full path vs B1 head ----
    latency: Dict[str, Any] = {"status": "skipped"}
    if not ctx.args.skip_latency:
        try:
            torch.manual_seed(ctx.seeds[0])
            lat_model = bc.JevX.from_pretrained(
                ctx.args.backbone, freeze_base_model=True,
                pooling_method=ctx.args.pooling, fusion_type=ctx.args.fusion,
            )
            lat_model.eval()
            sample_texts = ctx.test_texts[: ctx.args.latency_iters]
            q, opts = bc.DEFAULT_QUESTION, bc.DEFAULT_OPTIONS

            jevx_times = bc.measure_latency(
                lambda: lat_model.choice(
                    state=[sample_texts[0]], questions=[q], options=[opts]
                ),
                warmup=10, iters=ctx.args.latency_iters,
            )
            # per-example mean over the sample (first call excluded as warmup)
            per_text = []
            with torch.no_grad():
                for text in sample_texts:
                    t_begin = time.perf_counter()
                    lat_model.choice(state=[text], questions=[q], options=[opts])
                    per_text.append((time.perf_counter() - t_begin) * 1000.0)
            jevx_over_texts = {
                "mean_ms": float(np.mean(per_text)),
                "std_ms": float(np.std(per_text, ddof=1)) if len(per_text) > 1 else 0.0,
                "n_texts": len(per_text),
            }

            assert b1_head_for_latency is not None
            b1_sample = [ctx.feats["test"][ctx.primary_key][i] for i in range(32)]
            b1_times = bc.measure_latency(
                lambda: b1_head_for_latency({"state": b1_sample[:1]}),
                warmup=10, iters=ctx.args.latency_iters,
            )
            latency = {
                "status": "run",
                "protocol": "batch=1, warmup 10, mean over timed iters; noisy host (concurrent agents) -> std reported",
                "jevx_choice_single_text_repeated": jevx_times,
                "jevx_choice_over_test_texts": jevx_over_texts,
                "b1_linear_head_only": b1_times,
                "note": (
                    "JevX number includes tokenization + backbone + heads; B1 head-only "
                    "number is heads alone on cached features (the backbone cost is "
                    "shared and already paid in featurize)"
                ),
            }
            del lat_model
        except Exception as exc:  # noqa: BLE001 - recorded, never silent
            latency = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}

    payload = {
        "experiment": "exp2",
        "description": "baselines: B1 linear probe, B2 MLP, B4=B1+temperature, B3 generative; latency",
        "n_seeds": len(ctx.seeds),
        "per_seed": per_seed,
        "aggregate": {k: bc.mean_std_over_seeds(v) for k, v in metric_sets.items()},
        "b3_generative": b3_status,
        "latency": latency,
        "wall_time_s": round(time.perf_counter() - t0, 2),
    }
    ctx.save("exp2_baselines.json", payload)


# --------------------------------------------------------------------------
# EXP-3: pooling ablation (single seed, labeled)
# --------------------------------------------------------------------------

def run_exp3(ctx: BenchContext) -> None:
    t0 = time.perf_counter()
    seed = ctx.args.ablation_seed
    n_classes = len(bc.DEFAULT_OPTIONS)
    results: Dict[str, Any] = {}
    # All four poolings were featurized from one backbone pass per batch.
    for strategy_key in ("native", "cls", "masked_mean", "last_valid"):
        if strategy_key not in ctx.feats["train"]:
            continue
        print(f"[exp3] pooling={strategy_key} seed={seed}", flush=True)
        # Records are strategy-specific only in their feature rows.
        train_records = bc.build_records(
            ctx.feats["train"][strategy_key], ctx.train_labels
        )
        cal_records = bc.build_records(ctx.feats["cal"][strategy_key], ctx.cal_labels)
        test_records = bc.build_records(ctx.feats["test"][strategy_key], ctx.test_labels)
        torch.manual_seed(seed)
        model = bc.JevX.from_pretrained(
            ctx.args.backbone, freeze_base_model=True,
            pooling_method=(None if strategy_key == "native" else strategy_key),
            fusion_type=ctx.args.fusion,
        )
        model.eval()
        head = bc.FeaturizedJevX(model, bc.DEFAULT_QUESTION, bc.DEFAULT_OPTIONS)
        bc.train_heads(
            head, train_records, seed=seed, lr=ctx.args.lr,
            epochs=ctx.args.epochs, batch_size=ctx.args.batch_size,
        )
        cal_logits, cal_labels = bc.collect_logits(
            head, cal_records, batch_size=ctx.args.batch_size
        )
        test_logits, test_labels = bc.collect_logits(
            head, test_records, batch_size=ctx.args.batch_size
        )
        results[strategy_key] = {
            "pooling_config": strategy_key,
            "uncalibrated": bc.metrics_from_logits(test_logits, test_labels),
            "temperature": bc.calibrated_metrics(
                test_logits, test_labels, "temperature",
                fit_logits=cal_logits, fit_labels=cal_labels,
            ),
        }
        del model, head
    payload = {
        "experiment": "exp3",
        "description": "pooling ablation on the same protocol (SINGLE SEED)",
        "n_seeds": 1,
        "seed": seed,
        "results": results,
        "wall_time_s": round(time.perf_counter() - t0, 2),
    }
    ctx.save("exp3_pooling.json", payload)


# --------------------------------------------------------------------------
# EXP-4: fusion ablation + question-conditioning control (single seed)
# --------------------------------------------------------------------------

def run_exp4(ctx: BenchContext) -> None:
    t0 = time.perf_counter()
    seed = ctx.args.ablation_seed
    fusions = ["concat", "additive", "gated", "bilinear"]
    train_records = ctx.records("train")
    cal_records = ctx.records("cal")
    test_records = ctx.records("test")
    results: Dict[str, Any] = {}
    for fusion in fusions:
        print(f"[exp4] fusion={fusion} seed={seed}", flush=True)
        torch.manual_seed(seed)
        model = bc.JevX.from_pretrained(
            ctx.args.backbone, freeze_base_model=True,
            pooling_method=ctx.args.pooling, fusion_type=fusion,
        )
        model.eval()
        head = bc.FeaturizedJevX(model, bc.DEFAULT_QUESTION, bc.DEFAULT_OPTIONS)
        bc.train_heads(
            head, train_records, seed=seed, lr=ctx.args.lr,
            epochs=ctx.args.epochs, batch_size=ctx.args.batch_size,
        )
        cal_logits, cal_labels = bc.collect_logits(
            head, cal_records, batch_size=ctx.args.batch_size
        )
        test_logits, test_labels = bc.collect_logits(
            head, test_records, batch_size=ctx.args.batch_size
        )
        fusion_params = sum(p.numel() for p in model.fusion.parameters())
        results[fusion] = {
            "uncalibrated": bc.metrics_from_logits(test_logits, test_labels),
            "temperature": bc.calibrated_metrics(
                test_logits, test_labels, "temperature",
                fit_logits=cal_logits, fit_labels=cal_labels,
            ),
            "head_stack_trainable_params": bc.count_trainable(head),
            "fusion_module_params": fusion_params,
        }
        del model, head

    # Question-conditioning control (§59): same protocol, question replaced
    # by a zero embedding -> measures what the question pathway contributes.
    print(f"[exp4] question-mode=zero (control) seed={seed}", flush=True)
    torch.manual_seed(seed)
    model = bc.JevX.from_pretrained(
        ctx.args.backbone, freeze_base_model=True,
        pooling_method=ctx.args.pooling, fusion_type=ctx.args.fusion,
    )
    model.eval()
    head = bc.FeaturizedJevX(
        model, bc.DEFAULT_QUESTION, bc.DEFAULT_OPTIONS, question_mode="zero"
    )
    bc.train_heads(
        head, train_records, seed=seed, lr=ctx.args.lr,
        epochs=ctx.args.epochs, batch_size=ctx.args.batch_size,
    )
    z_cal, z_cal_labels = bc.collect_logits(
        head, cal_records, batch_size=ctx.args.batch_size
    )
    z_test, z_test_labels = bc.collect_logits(
        head, test_records, batch_size=ctx.args.batch_size
    )
    results["zero_question_control"] = {
        "question_mode": "zero",
        "fusion": ctx.args.fusion,
        "uncalibrated": bc.metrics_from_logits(z_test, z_test_labels),
        "temperature": bc.calibrated_metrics(
            z_test, z_test_labels, "temperature",
            fit_logits=z_cal, fit_labels=z_cal_labels,
        ),
    }
    del model, head

    payload = {
        "experiment": "exp4",
        "description": "fusion ablation + zero-question control (SINGLE SEED)",
        "n_seeds": 1,
        "seed": seed,
        "results": results,
        "wall_time_s": round(time.perf_counter() - t0, 2),
    }
    ctx.save("exp4_fusion.json", payload)


# --------------------------------------------------------------------------
# EXP-5: calibration ablation + reliability bins (single seed)
# --------------------------------------------------------------------------

def run_exp5(ctx: BenchContext) -> None:
    t0 = time.perf_counter()
    seed = ctx.args.ablation_seed
    train_records = ctx.records("train")
    cal_records = ctx.records("cal")
    test_records = ctx.records("test")
    torch.manual_seed(seed)
    model = bc.JevX.from_pretrained(
        ctx.args.backbone, freeze_base_model=True,
        pooling_method=ctx.args.pooling, fusion_type=ctx.args.fusion,
    )
    model.eval()
    head = bc.FeaturizedJevX(model, bc.DEFAULT_QUESTION, bc.DEFAULT_OPTIONS)
    bc.train_heads(
        head, train_records, seed=seed, lr=ctx.args.lr,
        epochs=ctx.args.epochs, batch_size=ctx.args.batch_size,
    )
    cal_logits, cal_labels = bc.collect_logits(
        head, cal_records, batch_size=ctx.args.batch_size
    )
    test_logits, test_labels = bc.collect_logits(
        head, test_records, batch_size=ctx.args.batch_size
    )
    methods = ["none", "temperature", "vector", "isotonic"]
    results: Dict[str, Any] = {}
    for method in methods:
        metrics = bc.calibrated_metrics(
            test_logits, test_labels, method,
            fit_logits=cal_logits, fit_labels=cal_labels,
        )
        engine = bc.CalibrationEngine(method)
        raw_test = torch.as_tensor(test_logits, dtype=torch.float32)
        if method != "none":
            engine.fit(
                torch.as_tensor(cal_logits, dtype=torch.float32),
                torch.as_tensor(np.asarray(cal_labels)),
            )
            cal_test = engine.calibrate(raw_test, "logits")
        else:
            cal_test = raw_test
        results[method] = {
            "metrics": metrics,
            "reliability_bins": bc.reliability_bins(
                cal_test, np.asarray(test_labels), n_bins=15
            ),
        }
    payload = {
        "experiment": "exp5",
        "description": "calibration families fit on CALIBRATION split (SINGLE SEED)",
        "n_seeds": 1,
        "seed": seed,
        "results": results,
        "wall_time_s": round(time.perf_counter() - t0, 2),
    }
    ctx.save("exp5_calibration.json", payload)


# --------------------------------------------------------------------------
# EXP-6: confidence ablation (single seed)
# --------------------------------------------------------------------------

def run_exp6(ctx: BenchContext) -> None:
    t0 = time.perf_counter()
    seed = ctx.args.ablation_seed
    train_records = ctx.records("train")
    cal_records = ctx.records("cal")
    test_records = ctx.records("test")
    torch.manual_seed(seed)
    model = bc.JevX.from_pretrained(
        ctx.args.backbone, freeze_base_model=True,
        pooling_method=ctx.args.pooling, fusion_type=ctx.args.fusion,
    )
    model.eval()
    head = bc.FeaturizedJevX(model, bc.DEFAULT_QUESTION, bc.DEFAULT_OPTIONS)
    bc.train_heads(
        head, train_records, seed=seed, lr=ctx.args.lr,
        epochs=ctx.args.epochs, batch_size=ctx.args.batch_size,
    )
    cal_logits, _ = bc.collect_logits(head, cal_records, batch_size=ctx.args.batch_size)
    test_logits, test_labels = bc.collect_logits(
        head, test_records, batch_size=ctx.args.batch_size
    )
    pred = test_logits.argmax(axis=1)
    correct = (pred == np.asarray(test_labels)).astype(np.float64)

    methods = ["max_probability", "margin", "entropy", "energy"]
    results: Dict[str, Any] = {}
    for method in methods:
        try:
            conf = bc.confidence_vectors(test_logits, method, cal_logits=cal_logits)
            results[method] = {
                "ece_of_confidence": bc.ece_of_confidence(conf, correct, n_bins=15),
                "aurc": bc.aurc(conf, correct),
                "coverage_at_risk_0.1": bc.coverage_at(conf, correct, 0.1),
                "coverage_at_risk_0.2": bc.coverage_at(conf, correct, 0.2),
                "conf_mean": float(np.mean(conf)),
                "conf_std": float(np.std(conf, ddof=1)),
            }
        except Exception as exc:  # noqa: BLE001 - recorded, never silent
            results[method] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}

    # Ranking BY MEASUREMENT: lower ECE-conf and AURC are better; higher
    # coverage-at-risk is better. All three orderings are reported.
    ok = {k: v for k, v in results.items() if "status" not in v}
    rank_ece = sorted(ok, key=lambda k: ok[k]["ece_of_confidence"])
    rank_aurc = sorted(ok, key=lambda k: ok[k]["aurc"])
    rank_cov10 = sorted(ok, key=lambda k: -ok[k]["coverage_at_risk_0.1"])
    payload = {
        "experiment": "exp6",
        "description": "confidence methods on test logits (SINGLE SEED)",
        "n_seeds": 1,
        "seed": seed,
        "results": results,
        "rank_by_ece_of_confidence": rank_ece,
        "rank_by_aurc": rank_aurc,
        "rank_by_coverage_at_risk_0.1": rank_cov10,
        "wall_time_s": round(time.perf_counter() - t0, 2),
    }
    ctx.save("exp6_confidence.json", payload)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    experiments = selected_experiments(args.experiments)
    print(f"[plan] experiments={experiments} seeds={args.seeds} "
          f"train={args.train_size} cal={args.cal_size} test={args.test_size} "
          f"epochs={args.epochs} lr={args.lr} batch={args.batch_size}", flush=True)
    t0 = time.perf_counter()
    ctx = BenchContext(args)
    print(f"[ctx] dataset {ctx.timings['load_dataset_s']}s, "
          f"backbone {ctx.timings['load_backbone_s']}s, "
          f"featurize {ctx.timings['featurize_all_splits_s']}s; "
          f"splits train={len(ctx.train_texts)} cal={len(ctx.cal_texts)} "
          f"test={len(ctx.test_texts)}", flush=True)

    runners = {
        "exp1": lambda: run_exp1(ctx),
        "exp2": lambda: run_exp2(ctx),
        "exp3": lambda: run_exp3(ctx),
        "exp4": lambda: run_exp4(ctx),
        "exp5": lambda: run_exp5(ctx),
        "exp6": lambda: run_exp6(ctx),
    }
    failures: Dict[str, str] = {}
    for exp in experiments:
        try:
            runners[exp]()
        except Exception as exc:  # noqa: BLE001 - record, continue, never silent
            import traceback

            failures[exp] = f"{type(exc).__name__}: {exc}"
            print(f"[FAILED] {exp}: {failures[exp]}", flush=True)
            traceback.print_exc()

    total = round(time.perf_counter() - t0, 2)
    summary = {
        "experiments_run": experiments,
        "failures": failures,
        "context_timings_s": ctx.timings,
        "total_wall_time_s": total,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "dtype": "float32",
    }
    summary_name = (
        f"run_summary_{args.run_label}.json" if args.run_label else "run_summary.json"
    )
    bc.write_json(Path(args.output_dir) / summary_name, summary)
    print(f"[done] total wall time {total}s; failures={len(failures)}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
