# Copyright (c) 2026 ysnrfd
"""EXP-7: second real text dataset (ag_news) under the EXP-1/EXP-2 protocol.

Mirrors `benchmarks/run_quality.py` exactly, with only the dataset, question
and option list swapped:

* dataset   : fancyzhx/ag_news pinned by revision (id verified via the Hub
              API before download; ClassLabel order = option order, so no
              label remapping exists anywhere in the pipeline);
* splits    : train + calibration are stratified samples carved from the
              OFFICIAL train split (seed 0, disjoint; `make_splits`); test =
              the FIRST 1500 rows of the official test split (7600 rows ->
              the same deterministic `[:test_size]` cap semantics
              `run_quality.py` applies to SST-2's validation split);
* backbone  : google/bert_uncased_L-2_H-128_A-2 frozen (Mode A), featurized
              ONCE per split (featurize-once), pooling=cls, fusion=concat;
* exp7a     : JevX heads, seeds {0,1,2}, uncal / temperature / isotonic
              (calibrators fit on the CALIBRATION split only);
* exp7b     : B1 linear probe, B2 MLP probe, B4 = B1 + temperature, same
              features / splits / hyperparams / seeds as exp7a.

Every metric comes from executed code; results are written as
`benchmarks/results/exp7_agnews_*.json`.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import bench_common as bc  # noqa: E402

# --------------------------------------------------------------------------
# Provenance constants (all verified before use — see reports/DATASET_BENCHMARKS.md)
# --------------------------------------------------------------------------

DATASET_ID = "fancyzhx/ag_news"
#: Hub API `sha` of fancyzhx/ag_news (GET 200, fetched 2026-09-24); the bare
#: id `ag_news` redirects to this repo via the Hub API.
DATASET_REVISION = "eb185aade064a813bc0b7f42de02595523103ca4"
#: Recorded from the Hub API tags verbatim (`license:unknown`), not inferred.
DATASET_LICENSE = "unknown (HF API tag license:unknown, fetched 2026-09-24)"
QUESTION = "What is the topic of this news article?"
#: Order IS the dataset's ClassLabel order: 0 World, 1 Sports, 2 Business,
#: 3 Sci/Tech -> no label remapping anywhere in the pipeline.
OPTIONS = ["World", "Sports", "Business", "Sci/Tech"]
TEXT_FIELD = "text"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--backbone", default=bc.DEFAULT_BACKBONE)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--train-size", type=int, default=3000)
    p.add_argument("--cal-size", type=int, default=1000)
    p.add_argument("--test-size", type=int, default=1500,
                   help="cap on the official test split rows (first N, mirroring "
                        "run_quality's [:test_size] cap)")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--pooling", default="cls")
    p.add_argument("--fusion", default="concat")
    p.add_argument("--output-dir", default=str(REPO_ROOT / "benchmarks" / "results"))
    return p.parse_args(argv)


# --------------------------------------------------------------------------
# Data + shared context (one model load, one featurization pass)
# --------------------------------------------------------------------------

def load_ag_news() -> Tuple[List[str], List[int], List[str], List[int], int]:
    """Load ag_news pinned to the exact revision.

    Returns (train_texts, train_labels, test_texts_capped, test_labels_capped,
    official_test_n). The test cap (first N rows) is applied by the caller so
    the recorded provenance states it explicitly.
    """
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, revision=DATASET_REVISION)
    train, test = ds["train"], ds["test"]
    return (
        list(train[TEXT_FIELD]),
        [int(x) for x in train["label"]],
        list(test[TEXT_FIELD]),
        [int(x) for x in test["label"]],
        len(test),
    )


def _counts(labels: Sequence[int]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for y in labels:
        out[str(int(y))] = out.get(str(int(y)), 0) + 1
    return out


def _token_lengths(tokenizer: Any, texts: Sequence[str]) -> List[int]:
    """Untruncated token counts (special tokens included) for bookkeeping."""
    enc = tokenizer(list(texts), truncation=False)
    return [len(ids) for ids in enc["input_ids"]]


class BenchContext:
    """Dataset + splits + frozen backbone + featurize-once, EXP-7 flavor."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        self.out_dir = Path(args.output_dir)
        self.timings: Dict[str, float] = {}

        t0 = time.perf_counter()
        (self.train_texts_all, self.train_labels_all,
         self.test_texts_all, self.test_labels_all, official_test_n) = load_ag_news()
        self.timings["load_dataset_s"] = round(time.perf_counter() - t0, 2)
        self.official_test_n = official_test_n

        # Test = first N rows of the official test split (cap never binds on
        # the 7600-row official test; 1500 requested like SST-2's 1500 cap).
        if args.test_size and args.test_size < len(self.test_texts_all):
            self.test_texts = self.test_texts_all[: args.test_size]
            self.test_labels = self.test_labels_all[: args.test_size]
            self.test_cap_applied = True
        else:
            self.test_texts = self.test_texts_all
            self.test_labels = self.test_labels_all
            self.test_cap_applied = False

        self.train_idx, self.cal_idx = bc.make_splits(
            self.train_labels_all, args.train_size, args.cal_size, seed=0
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

        t0 = time.perf_counter()
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
                strategies=[None, args.pooling],
            )
        self.timings["featurize_all_splits_s"] = round(time.perf_counter() - t0, 2)

        # Truncation bookkeeping: featurize applies max_length (128); the full
        # model.choice() path never truncates (JevX raises rather than silently
        # truncating). On SST-2 every sentence fit in 128 tokens, so the two
        # paths agreed to 1e-6; ag_news has longer articles, so the equivalence
        # subset is restricted to rows whose UNTRUNCATED token length is
        # <= max_length - the only rows where the two paths are comparable.
        # Truncation counts per split are recorded as provenance.
        t0 = time.perf_counter()
        self.token_len_stats: Dict[str, Dict[str, int]] = {}
        for name, texts in (
            ("train", self.train_texts),
            ("cal", self.cal_texts),
            ("test", self.test_texts),
        ):
            lens = _token_lengths(self.model.tokenizer, texts)
            self.token_len_stats[name] = {
                "n": len(lens),
                "over_max_length": int(sum(1 for l in lens if l > args.max_length)),
                "max_len_observed": int(max(lens)) if lens else 0,
            }
        test_lens = _token_lengths(self.model.tokenizer, self.test_texts)
        eq_idx = [i for i, l in enumerate(test_lens) if l <= args.max_length][:16]
        self.eq_texts = [self.test_texts[i] for i in eq_idx]
        self.eq_feats = self.feats["test"][args.pooling][
            np.asarray(eq_idx, dtype=np.int64)
        ]
        self.timings["token_len_stats_s"] = round(time.perf_counter() - t0, 2)

    def records(self, split: str) -> List[Dict[str, Any]]:
        return bc.build_records(
            self.feats[split][self.primary_key],
            getattr(self, f"{split}_labels"),
            question=QUESTION,
            options=OPTIONS,
        )

    def new_head_model(self):
        """Fresh JevX reload so per-seed head init is RNG-drawn identically to
        run_quality's exp1 (torch.manual_seed(seed) immediately before)."""
        model = bc.JevX.from_pretrained(
            self.args.backbone,
            freeze_base_model=True,
            pooling_method=self.args.pooling,
            fusion_type=self.args.fusion,
        )
        model.eval()
        return model

    def provenance(self) -> Dict[str, Any]:
        prov = bc.provenance(
            self.args.backbone,
            DATASET_ID,
            DATASET_REVISION,
            self.seeds,
            {
                "train_size": self.args.train_size,
                "cal_size": self.args.cal_size,
                "test_size": self.args.test_size,
                "epochs": self.args.epochs,
                "lr": self.args.lr,
                "batch_size": self.args.batch_size,
                "max_length": self.args.max_length,
                "fusion": self.args.fusion,
                "pooling": self.args.pooling,
                "split_seed": 0,
            },
        )
        prov["backbone_revision"] = self.backbone_revision
        prov["question"] = QUESTION
        prov["options"] = list(OPTIONS)
        prov["dataset_license"] = DATASET_LICENSE
        prov["splits"] = {
            "train_n": len(self.train_texts),
            "cal_n": len(self.cal_texts),
            "test_n": len(self.test_texts),
            "train_label_counts": _counts(self.train_labels),
            "cal_label_counts": _counts(self.cal_labels),
            "test_label_counts": _counts(self.test_labels),
            "official_train_n": len(self.train_texts_all),
            "official_test_n": self.official_test_n,
            "test_source": (
                f"official test split, first {len(self.test_texts)} rows "
                f"(cap applied={self.test_cap_applied}; deterministic "
                "[:test_size] slice, mirroring run_quality.py)"
            ),
            "train_cal_source": "stratified carve from official train split",
            "split_seed": 0,
            "disjointness": "asserted at construction by bc.make_splits",
            "truncation": {
                "max_length": self.args.max_length,
                "note": (
                    "featurize() truncates to max_length; model.choice() never "
                    "truncates (JevX raises instead). Rows over max_length are "
                    "excluded from the equivalence subset only - the benchmark "
                    "metrics always use the truncated (featurize-once) features."
                ),
                "per_split_token_len": self.token_len_stats,
            },
        }
        return prov

    def save(self, name: str, payload: Dict[str, Any]) -> None:
        payload = {"provenance": self.provenance(), **payload}
        bc.write_json(self.out_dir / name, payload)
        print(f"[wrote] {self.out_dir / name}", flush=True)


# --------------------------------------------------------------------------
# exp7a: JevX heads, 3 seeds, uncal / temperature / isotonic
# --------------------------------------------------------------------------

def run_exp7a(ctx: BenchContext) -> None:
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
        print(f"[exp7a] training JevX heads seed={seed}", flush=True)
        torch.manual_seed(seed)
        model = ctx.new_head_model()
        head = bc.FeaturizedJevX(model, QUESTION, OPTIONS)
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
                model, head, ctx.eq_feats, ctx.eq_texts, n=len(ctx.eq_texts),
            )
        )
        del model, head, trainer
    payload = {
        "experiment": "exp7",
        "sub_experiment": "exp7a",
        "description": "JevX heads over frozen bert backbone (Mode A) on ag_news, 3 training seeds",
        "dataset": DATASET_ID,
        "n_seeds": len(ctx.seeds),
        "per_seed": per_seed,
        "aggregate": {
            key: bc.mean_std_over_seeds(rows) for key, rows in per_seed_metrics.items()
        },
        "featurized_vs_full_model": equivalence,
        "equivalence_subset": {
            "n": len(ctx.eq_texts),
            "selection": (
                "first test rows whose UNTRUNCATED token length <= max_length "
                f"({ctx.args.max_length}); rows over the cap are excluded because "
                "the full model.choice() path does not truncate while featurize() "
                "does - comparing them would measure truncation, not path drift"
            ),
            "token_lens": _token_lengths(ctx.model.tokenizer, ctx.eq_texts),
        },
        "wall_time_s": round(time.perf_counter() - t0, 2),
    }
    ctx.save("exp7_agnews_jevx.json", payload)


# --------------------------------------------------------------------------
# exp7b: baselines B1 / B2 / B4 on the identical features
# --------------------------------------------------------------------------

def run_exp7b(ctx: BenchContext) -> None:
    t0 = time.perf_counter()
    train_records = ctx.records("train")
    cal_records = ctx.records("cal")
    test_records = ctx.records("test")
    in_dim = ctx.feats["train"][ctx.primary_key].shape[1]
    n_classes = len(OPTIONS)

    per_seed: List[Dict[str, Any]] = []
    metric_sets: Dict[str, List[Dict[str, float]]] = {
        "b1_linear": [], "b1_temperature": [], "b2_mlp": [],
    }
    for seed in ctx.seeds:
        print(f"[exp7b] training B1/B2 probes seed={seed}", flush=True)
        torch.manual_seed(seed)
        b1 = bc.LinearProbe(in_dim, n_classes)
        bc.train_heads(
            b1, train_records, seed=seed, lr=ctx.args.lr,
            epochs=ctx.args.epochs, batch_size=ctx.args.batch_size,
        )
        b1_cal, cal_labels = bc.collect_logits(
            b1, cal_records, batch_size=ctx.args.batch_size
        )
        b1_test, test_labels = bc.collect_logits(
            b1, test_records, batch_size=ctx.args.batch_size
        )

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
        del b1, b2

    payload = {
        "experiment": "exp7",
        "sub_experiment": "exp7b",
        "description": "baselines on ag_news: B1 linear probe, B2 MLP, B4=B1+temperature (same features/splits/seeds)",
        "dataset": DATASET_ID,
        "n_seeds": len(ctx.seeds),
        "per_seed": per_seed,
        "aggregate": {k: bc.mean_std_over_seeds(v) for k, v in metric_sets.items()},
        "note": (
            "B3 (distilgpt2 constrained-decoding) is NOT run here: its template is "
            "sentiment-specific ('Review: ... Sentiment:'), and the EXP-7 brief "
            "specifies B1/B2/B4 only. Latency is not re-measured either - the "
            "SST-2 latency rows already cover this host."
        ),
        "wall_time_s": round(time.perf_counter() - t0, 2),
    }
    ctx.save("exp7_agnews_baselines.json", payload)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    print(f"[plan] exp7 ag_news seeds={args.seeds} train={args.train_size} "
          f"cal={args.cal_size} test={args.test_size} epochs={args.epochs} "
          f"lr={args.lr} batch={args.batch_size}", flush=True)
    t0 = time.perf_counter()
    ctx = BenchContext(args)
    print(f"[ctx] dataset {ctx.timings['load_dataset_s']}s, backbone "
          f"{ctx.timings['load_backbone_s']}s, featurize "
          f"{ctx.timings['featurize_all_splits_s']}s; splits train="
          f"{len(ctx.train_texts)} cal={len(ctx.cal_texts)} "
          f"test={len(ctx.test_texts)}; backbone_rev={ctx.backbone_revision}",
          flush=True)

    failures: Dict[str, str] = {}
    for name, fn in (("exp7a", lambda: run_exp7a(ctx)),
                     ("exp7b", lambda: run_exp7b(ctx))):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - recorded, never silent
            import traceback

            failures[name] = f"{type(exc).__name__}: {exc}"
            print(f"[FAILED] {name}: {failures[name]}", flush=True)
            traceback.print_exc()

    total = round(time.perf_counter() - t0, 2)
    summary = {
        "experiment": "exp7",
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "failures": failures,
        "context_timings_s": ctx.timings,
        "total_wall_time_s": total,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "dtype": "float32",
    }
    bc.write_json(ctx.out_dir / "run_summary_exp7.json", summary)
    print(f"[done] total wall time {total}s; failures={len(failures)}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
