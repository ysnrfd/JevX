# Copyright (c) 2026 ysnrfd
"""EXP-6c: the `learned` confidence method MEASURED on ag_news (SECOND DATASET).

EXP-6b (benchmarks/run_exp6b.py) scored all five confidence methods on SST-2
and measured `learned` SECOND on ECE-of-confidence and FOURTH on AURC - in a
SINGLE draw, with a SINGLE learned-head seed (0). That is one observation.
EXP-6c asks the actual question: is the ordering a property of the method, or
a single-draw artifact? The EXP-6b protocol is therefore re-run on ag_news -
a SECOND dataset, and a K = 4 MULTI-CLASS one, where the K=2 three-way monotone
tie between max_probability / margin / entropy does NOT apply, so the
risk-coverage rankings are genuinely separated instead of forced to tie - and
the learned head is trained under FIVE seeds (0, 1, 2, 3, 4) so it is reported
as mean +/- std. Seed 0 is included precisely so the single-draw number the
SST-2 row is built on is reproduced here too.

REFERENCE-IMPLEMENTATION PARITY IS BY CONSTRUCTION, NOT BY TRANSCRIPTION.
The featurized head-stack subclass, the logits+fused collector, the
learned-confidence trainer and the metric evaluator are IMPORTED from
`benchmarks.run_exp6b`, so the objects executed below are literally the
reference implementation's - not a re-typing of them that could drift. The
ag_news dataset id/revision/question/options come from `benchmarks.run_exp7`
and the feature cache from `benchmarks.run_head_quality`, both of which are
the producers of the published ag_news protocol constants.

Learned-confidence training protocol - IDENTICAL to EXP-6b, fit on CAL only:

* inputs   : x = [fused ; max_prob ; 1 - H/log K ; margin] on the 1000-row
             CALIBRATION split, where `fused` is the 128-d concat-fusion
             output H of the trained JevX head stack (computed by the same
             modules model.forward uses);
* targets  : y = 1[ argmax(cal_logits) == cal_label ] (cal-split correctness);
* module   : ConfidenceEngine("learned", fused_dim=128, hidden_size=32) ->
             MLP 131 -> 32 (tanh) -> 1 -> sigmoid (the library class,
             unmodified, 4,257 trainable parameters);
* loss     : BCE on the sigmoid output; optimizer Adam, lr=1e-3;
* schedule : 100 epochs, batch 64, shuffle by torch.Generator(seed=s);
             torch.manual_seed(s) before constructing the module;
* test     : used ONLY for the final reported metrics (never for fitting,
             tuning or early stopping - there is no early stopping).

The head stack itself is trained ONCE, at seed 0, under the EXP-7/EXP-6 recipe
(HeadTrainer + choice_loss, 4 epochs, lr 1e-3, batch 64, model constructed
immediately after torch.manual_seed(0)), so all five learned seeds and all four
reference methods are scored on the SAME test logits - the comparison is
paired, and the seed axis is the learned head alone.

DEVIATIONS FROM EXP-6b, stated in full (all recorded in the result JSON too):

  D1  Features are READ from the shared featurize-once cache
      `benchmarks/cache/head_quality/ag_news_features.npz` instead of being
      re-featurized in this run (it is present; a fresh pass is only taken if
      it is missing, and then re-frozen into the same cache). The backbone is
      FROZEN, so pooled features cannot depend on the head stack's RNG stream;
      to keep that from being an assertion, this run re-featurizes a
      `--cache-check-n` row sample of the test split and reports the measured
      max abs difference against the cache (`cache_spot_check`).
  D2  The cache's train/cal/test splits are CHECKED against a freshly computed
      `bc.make_splits(..., seed=0)` on the pinned dataset (`split_check`), so
      the cache is known to BE the EXP-7 protocol split rather than assumed to
      be.
  D3  The built-in equivalence check is restricted to test rows whose
      UNTRUNCATED token length is <= max_length. ag_news articles are longer
      than SST-2 sentences, and `featurize` truncates at max_length while
      `model.choice()` never truncates, so on truncated rows the two paths are
      not comparable - the same restriction run_exp7.py applies and records.
  D4  `learned` is trained under 5 seeds (0,1,2,3,4) instead of 1, and is
      reported as mean +/- std. Every other protocol element is unchanged.

A dataset property, not a deviation: ag_news is K = 4, so max_probability,
margin and entropy are NOT monotone transforms of one another there (that
identity holds only for K = 2) and their AURC/coverage rows are expected to
separate. Whether they do is a measured outcome, not an assumption.

Usage (from the repo root):
    python benchmarks/run_exp6c.py
    python benchmarks/run_exp6c.py --learned-seeds 0 --recompute-features
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import bench_common as bc  # noqa: E402
from benchmarks import run_exp6b as exp6b  # noqa: E402
from benchmarks import run_exp7 as exp7  # noqa: E402
from benchmarks import run_head_quality as hq  # noqa: E402

# --------------------------------------------------------------------------
# Reference-implementation parity: IMPORTED, not re-typed.
# --------------------------------------------------------------------------

FusedFeaturizedJevX = exp6b.FusedFeaturizedJevX
collect_logits_and_fused = exp6b.collect_logits_and_fused
train_learned_confidence = exp6b.train_learned_confidence
evaluate_confidence = exp6b.evaluate_confidence
_softmax = exp6b._softmax

DATASET_ID = exp7.DATASET_ID
DATASET_REVISION = exp7.DATASET_REVISION
QUESTION = exp7.QUESTION
OPTIONS = list(exp7.OPTIONS)
#: The ag_news protocol constants, from the producer of the published rows.
AGNEWS_CFG: Dict[str, Any] = dict(hq.DATASETS["ag_news"])

BASELINE_METHODS: Tuple[str, ...] = ("max_probability", "margin", "entropy", "energy")
CONFIDENCE_FIELDS: Tuple[str, ...] = (
    "ece_of_confidence",
    "aurc",
    "coverage_at_risk_0.1",
    "coverage_at_risk_0.2",
    "conf_mean",
    "conf_std",
)
EXP6B_JSON = REPO_ROOT / "benchmarks" / "results" / "exp6b_learned_confidence.json"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--backbone", default=AGNEWS_CFG["backbone"])
    p.add_argument("--train-size", type=int, default=AGNEWS_CFG["train_size"])
    p.add_argument("--cal-size", type=int, default=AGNEWS_CFG["cal_size"])
    p.add_argument("--test-size", type=int, default=AGNEWS_CFG["test_size"])
    p.add_argument("--epochs", type=int, default=AGNEWS_CFG["epochs"])
    p.add_argument("--lr", type=float, default=AGNEWS_CFG["lr"])
    p.add_argument("--batch-size", type=int, default=AGNEWS_CFG["batch_size"])
    p.add_argument("--max-length", type=int, default=AGNEWS_CFG["max_length"])
    p.add_argument("--fusion", default=AGNEWS_CFG["fusion"])
    p.add_argument("--pooling", default=AGNEWS_CFG["pooling"])
    p.add_argument("--ablation-seed", type=int, default=0,
                   help="seed of the HEAD STACK (EXP-6b used 0; unchanged)")
    p.add_argument("--learned-seeds", default="0,1,2,3,4",
                   help="comma-separated seeds for the learned-confidence head "
                        "(EXP-6b used 1: seed 0 only)")
    p.add_argument("--learned-epochs", type=int, default=100)
    p.add_argument("--learned-lr", type=float, default=1e-3)
    p.add_argument("--learned-batch-size", type=int, default=64)
    p.add_argument("--learned-hidden", type=int, default=32)
    p.add_argument("--cache-check-n", type=int, default=64,
                   help="test rows re-featurized and compared against the "
                        "cache to make the cache provenance measured (D1)")
    p.add_argument("--recompute-features", action="store_true",
                   help="ignore the feature cache and re-featurize (only "
                        "needed if the cache is missing)")
    p.add_argument("--output-dir", default=str(REPO_ROOT / "benchmarks" / "results"))
    return p.parse_args(argv)


def _counts(labels: Sequence[int]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for y in labels:
        out[str(int(y))] = out.get(str(int(y)), 0) + 1
    return out


def _token_lengths(tokenizer: Any, texts: Sequence[str]) -> List[int]:
    """Untruncated token counts (special tokens included) - same helper as
    benchmarks/run_exp7.py, used for the truncation-aware equivalence subset."""
    enc = tokenizer(list(texts), truncation=False)
    return [len(ids) for ids in enc["input_ids"]]


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    seed = args.ablation_seed
    learned_seeds = [int(s) for s in args.learned_seeds.split(",") if s.strip()]
    if not learned_seeds:
        raise ValueError("run_exp6c: --learned-seeds must list at least one seed")
    t_start = time.perf_counter()
    timings: Dict[str, float] = {}

    # ---- raw dataset (canonical loader) + the split check (D2) ----
    # Everything downstream of the FEATURE CACHE is derived from the pinned
    # dataset; loading it here lets the cache's splits be checked rather than
    # trusted. If the dataset cannot be reached, the run continues on the cache
    # and records the reason - the numbers are still cache-sourced and stated
    # as such.
    t0 = time.perf_counter()
    raw: Optional[Dict[str, Any]] = None
    try:
        (train_texts_all, train_labels_all,
         test_texts_all, test_labels_all, official_test_n) = exp7.load_ag_news()
        test_texts = list(test_texts_all[: args.test_size])
        test_labels = [int(v) for v in test_labels_all[: args.test_size]]
        train_idx, cal_idx = bc.make_splits(
            train_labels_all, args.train_size, args.cal_size, seed=seed
        )
        raw = {
            "train_idx": train_idx,
            "cal_idx": cal_idx,
            "test_texts": test_texts,
            "test_labels": test_labels,
            "official_test_n": official_test_n,
        }
        timings["load_dataset_s"] = round(time.perf_counter() - t0, 2)
    except Exception as exc:  # noqa: BLE001 - recorded, never silent
        raw = None
        timings["load_dataset_s"] = round(time.perf_counter() - t0, 2)
        print("[warn] dataset unavailable, running on the feature cache alone: "
              + type(exc).__name__ + ": " + str(exc)[:200], flush=True)

    # ---- features: featurize-once cache, exactly as EXP-7 produced them ----
    t0 = time.perf_counter()
    feats, labels, cache_meta, feature_source = hq.build_features(
        "ag_news", AGNEWS_CFG, recompute=args.recompute_features
    )
    timings["features_s"] = round(time.perf_counter() - t0, 2)

    split_check: Dict[str, Any] = {"status": "skipped", "reason": "dataset not loaded"}
    if raw is not None:
        exp_train = [int(train_labels_all[i]) for i in raw["train_idx"]]
        exp_cal = [int(train_labels_all[i]) for i in raw["cal_idx"]]
        split_check = {
            "status": "run",
            "note": (
                "the cached split labels are compared against a freshly computed "
                "bc.make_splits(train_labels, %d, %d, seed=%d) on the pinned "
                "dataset, so the cache is known to BE the EXP-7 protocol split"
                % (args.train_size, args.cal_size, seed)
            ),
            "train_labels_match": exp_train == [int(v) for v in labels["train"]],
            "cal_labels_match": exp_cal == [int(v) for v in labels["cal"]],
            "test_labels_match": raw["test_labels"] == [int(v) for v in labels["test"]],
            "train_n": int(len(labels["train"])),
            "cal_n": int(len(labels["cal"])),
            "test_n": int(len(labels["test"])),
            "official_test_n": raw["official_test_n"],
            "train_cal_overlap": int(
                np.intersect1d(raw["train_idx"], raw["cal_idx"]).size
            ),
        }
        split_check["all_match"] = bool(
            split_check["train_labels_match"]
            and split_check["cal_labels_match"]
            and split_check["test_labels_match"]
        )

    train_records = bc.build_records(feats["train"], labels["train"], QUESTION, OPTIONS)
    cal_records = bc.build_records(feats["cal"], labels["cal"], QUESTION, OPTIONS)
    test_records = bc.build_records(feats["test"], labels["test"], QUESTION, OPTIONS)

    # ---- seed-0 head training, identical recipe to run_exp7.exp7a ----
    # The head's JevX wrapper must be built AFTER torch.manual_seed(seed): its
    # non-backbone modules (state_proj, question/schema encoders, fusion,
    # choice head) draw their init from that seeded RNG. Building it earlier
    # draws from an unseeded stream and yields a different trained model -
    # this is exactly the constraint run_exp6b documents and honours.
    t0 = time.perf_counter()
    torch.manual_seed(seed)
    train_model = bc.JevX.from_pretrained(
        args.backbone,
        freeze_base_model=True,
        pooling_method=args.pooling,
        fusion_type=args.fusion,
    )
    train_model.eval()
    backbone_cfg = train_model.config.backbone_config
    backbone_revision = (
        backbone_cfg.get("_commit_hash") if isinstance(backbone_cfg, dict) else None
    ) or getattr(getattr(train_model.model, "config", None), "_commit_hash", None)

    head = FusedFeaturizedJevX(train_model, QUESTION, OPTIONS)
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
    test_acc = float((test_logits.argmax(axis=1) == np.asarray(test_lab)).mean())

    # ---- equivalence check on the UNTRUNCATED subset (D3) ----
    if raw is not None:
        lengths = _token_lengths(train_model.tokenizer, raw["test_texts"])
        keep = [i for i, L in enumerate(lengths) if L <= args.max_length]
        n_eq = min(16, len(keep))
        eq = bc.equivalence_check(
            train_model, head,
            feats["test"][keep[:n_eq]], [raw["test_texts"][i] for i in keep[:n_eq]],
            n=n_eq,
        )
        eq["note"] = (
            "D3: restricted to test rows whose UNTRUNCATED token length is <= "
            "max_length (%d); featurize truncates at max_length while "
            "model.choice() never truncates, so truncated rows are not "
            "comparable (same restriction run_exp7.py applies)." % args.max_length
        )
        eq["subset_rule"] = "untruncated_len <= %d" % args.max_length
        eq["n_eligible_rows"] = len(keep)
        eq["n_truncated_rows"] = int(len(lengths) - len(keep))
    else:
        eq = {"status": "skipped", "reason": "raw test texts unavailable"}

    # ---- cache provenance, MEASURED rather than asserted (D1) ----
    cache_spot_check: Dict[str, Any] = {"status": "skipped",
                                        "reason": "raw test texts unavailable"}
    if raw is not None and args.cache_check_n > 0:
        t0 = time.perf_counter()
        n_chk = min(int(args.cache_check_n), len(raw["test_texts"]))
        fchk = bc.JevX.from_pretrained(
            args.backbone, freeze_base_model=True,
            pooling_method=args.pooling, fusion_type=args.fusion,
        )
        fchk.eval()
        try:
            got = bc.featurize(
                fchk, raw["test_texts"][:n_chk],
                batch_size=args.batch_size, max_length=args.max_length,
                strategies=[args.pooling],
            )[args.pooling]
            cache_spot_check = {
                "status": "run",
                "n_rows": int(n_chk),
                "max_abs_diff_vs_cache": float(
                    np.abs(got - feats["test"][:n_chk]).max()
                ),
                "note": (
                    "the first %d test rows were re-featurized in this run and "
                    "compared against the cache, so the cache-sourced features "
                    "are verified rather than assumed" % n_chk
                ),
            }
        finally:
            del fchk
        timings["cache_spot_check_s"] = round(time.perf_counter() - t0, 2)

    # ---- the four already-published methods, on THIS run's test logits ----
    cal_probs = _softmax(cal_logits)
    test_probs = _softmax(test_logits)
    correct_test = (test_logits.argmax(axis=1) == np.asarray(test_lab)).astype(
        np.float64
    )
    correct_cal = (cal_logits.argmax(axis=1) == np.asarray(cal_lab)).astype(np.float64)
    results: Dict[str, Dict[str, Any]] = {}
    for method in BASELINE_METHODS:
        conf = bc.confidence_vectors(test_logits, method, cal_logits=cal_logits)
        results[method] = {
            "ece_of_confidence": bc.ece_of_confidence(conf, correct_test, n_bins=15),
            "aurc": bc.aurc(conf, correct_test),
            "coverage_at_risk_0.1": bc.coverage_at(conf, correct_test, 0.1),
            "coverage_at_risk_0.2": bc.coverage_at(conf, correct_test, 0.2),
            "conf_mean": float(np.mean(conf)),
            "conf_std": float(np.std(conf, ddof=1)),
            "n_seeds": 1,
            "note": "deterministic function of the test logits; one row",
        }

    # ---- train + evaluate `learned` (fit on CAL only), once per seed ----
    t0 = time.perf_counter()
    fused_dim = int(cal_fused.shape[1])
    learned_per_seed: List[Dict[str, Any]] = []
    learned_training: List[Dict[str, Any]] = []
    cal_diag: List[Dict[str, Any]] = []
    for ls in learned_seeds:
        engine, loss_history = train_learned_confidence(
            cal_probs=cal_probs,
            cal_logits=cal_logits,
            cal_fused=cal_fused,
            cal_labels=np.asarray(cal_lab),
            fused_dim=fused_dim,
            hidden=args.learned_hidden,
            epochs=args.learned_epochs,
            lr=args.learned_lr,
            batch_size=args.learned_batch_size,
            seed=ls,
        )
        test_metrics, _ = evaluate_confidence(
            engine, probs=test_probs, logits=test_logits, fused=test_fused,
            correct=correct_test,
        )
        cal_metrics, _ = evaluate_confidence(
            engine, probs=cal_probs, logits=cal_logits, fused=cal_fused,
            correct=correct_cal,
        )
        learned_per_seed.append(dict(test_metrics, seed=ls))
        cal_diag.append(dict(cal_metrics, seed=ls))
        learned_training.append({
            "seed": ls,
            "bce_first_epoch": loss_history[0],
            "bce_last_epoch": loss_history[-1],
            "bce_by_epoch": loss_history,
            "learned_params": int(
                sum(p.numel() for p in engine.parameters() if p.requires_grad)
            ),
        })
        print("  [learned] seed %d  ece=%.4f  aurc=%.4f  cov@0.1=%.4f  "
              "cov@0.2=%.4f" % (ls, test_metrics["ece_of_confidence"],
                                test_metrics["aurc"],
                                test_metrics["coverage_at_risk_0.1"],
                                test_metrics["coverage_at_risk_0.2"]),
              flush=True)
        del engine
    timings["train_learned_s"] = round(time.perf_counter() - t0, 2)

    learned_agg = bc.mean_std_over_seeds(
        [{k: row[k] for k in CONFIDENCE_FIELDS} for row in learned_per_seed]
    )
    learned_agg["n_seeds"] = len(learned_seeds)
    learned_agg["seeds"] = list(learned_seeds)
    results["learned"] = {
        "n_seeds": len(learned_seeds),
        "seeds": list(learned_seeds),
        "aggregate": learned_agg,
        "per_seed": learned_per_seed,
    }

    # ---- rankings, computed on the learned MEAN ----
    def _key(method: str, field: str) -> float:
        if method == "learned":
            return float(learned_agg[field]["mean"])
        return float(results[method][field])

    methods = list(BASELINE_METHODS) + ["learned"]
    rank_ece = sorted(methods, key=lambda m: _key(m, "ece_of_confidence"))
    rank_aurc = sorted(methods, key=lambda m: _key(m, "aurc"))
    rank_cov10 = sorted(methods, key=lambda m: -_key(m, "coverage_at_risk_0.1"))
    rank_cov20 = sorted(methods, key=lambda m: -_key(m, "coverage_at_risk_0.2"))
    # Per-seed rank of `learned`, so the reader sees whether the ordering is
    # stable across draws or only holds on the mean.
    learned_rank_by_seed: Dict[str, List[str]] = {}
    for field, name in (("ece_of_confidence", "rank_by_ece_of_confidence"),
                        ("aurc", "rank_by_aurc")):
        per_seed_ranks: List[List[str]] = []
        for row in learned_per_seed:
            order = sorted(
                methods,
                key=lambda m: (row[field] if m == "learned" else results[m][field]),
            )
            per_seed_ranks.append(order)
        learned_rank_by_seed[name] = per_seed_ranks

    # ---- the SST-2 single-draw reference, quoted verbatim for comparison ----
    sst2_reference: Dict[str, Any] = {"status": "missing_source_json"}
    if EXP6B_JSON.exists():
        b = json.loads(EXP6B_JSON.read_text(encoding="utf-8"))
        sst2_reference = {
            "status": "run",
            "source_json": str(EXP6B_JSON.name),
            "n_seeds": b["n_seeds"],
            "learned_seed": b["protocol"]["learned_seed"],
            "n_test": b["provenance"]["splits"]["test_n"],
            "num_classes": 2,
            "learned": {
                k: b["results"]["learned"][k] for k in CONFIDENCE_FIELDS
            },
            "baselines": {
                m: {k: b["results"][m][k] for k in CONFIDENCE_FIELDS}
                for m in BASELINE_METHODS
            },
            "rank_by_ece_of_confidence": b["rank_by_ece_of_confidence"],
            "rank_by_aurc": b["rank_by_aurc"],
            "note": (
                "SST-2 / K=2 / single learned seed 0, quoted for comparison only; "
                "not recomputed here"
            ),
        }

    payload = {
        "experiment": "exp6c",
        "description": (
            "learned confidence (MEASURED) on ag_news - the SECOND dataset, K=4 - "
            "under the EXP-6b protocol, with FIVE learned-head seeds; all five "
            "methods computed on the same test logits"
        ),
        "dataset_name": "ag_news",
        "num_classes": len(OPTIONS),
        "n_seeds": len(learned_seeds),
        "learned_seeds": list(learned_seeds),
        "seed": seed,
        "protocol": {
            "head_training": (
                "identical to run_exp7.exp7a and run_exp6b: HeadTrainer + "
                f"choice_loss, {args.epochs} epochs, lr {args.lr}, batch "
                f"{args.batch_size}, seed {seed}, featurize-once features, model "
                "constructed immediately after torch.manual_seed(seed)"
            ),
            "head_seeds": [seed],
            "learned_fit_split": f"CALIBRATION split only (n={len(cal_lab)})",
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
            "learned_seeds": list(learned_seeds),
            "learned_shuffle": (
                "torch.Generator().manual_seed(seed) per epoch permutation, "
                "re-created per seed"
            ),
            "test_usage": "final metrics only; never fit, never tuned, no early stopping",
            "pairing": (
                "the head stack is trained ONCE at seed 0; all four reference "
                "methods and all %d learned seeds are scored on the SAME test "
                "logits, so every row is paired" % len(learned_seeds)
            ),
            "parity_mechanism": (
                "FusedFeaturizedJevX / collect_logits_and_fused / "
                "train_learned_confidence / evaluate_confidence are IMPORTED "
                "from benchmarks.run_exp6b (not re-typed), so the code objects "
                "executed here are the reference implementation's"
            ),
        },
        "deviations_from_exp6b": {
            "D1_features_from_cache": (
                "features read from benchmarks/cache/head_quality/"
                "ag_news_features.npz instead of re-featurized in-run; verified "
                "against a %d-row in-run re-featurization (cache_spot_check) and "
                "against a fresh make_splits (split_check)" % int(args.cache_check_n)
            ),
            "D2_split_check": "cache split labels checked against a fresh make_splits",
            "D3_equivalence_subset": (
                "equivalence check restricted to rows whose untruncated token "
                "length <= max_length (ag_news articles are longer than SST-2 "
                "sentences); same restriction run_exp7.py applies"
            ),
            "D4_five_learned_seeds": (
                "learned head trained under seeds %s instead of {0}; every other "
                "protocol element unchanged" % list(learned_seeds)
            ),
            "dataset_property_not_a_deviation": (
                "K=4, so max_probability / margin / entropy are NOT monotone "
                "transforms of one another (that identity is K=2-only) and their "
                "risk-coverage rows are free to separate"
            ),
        },
        "learned_training": learned_training,
        "head_training": {
            "train_loss_first": history[0]["loss"],
            "train_loss_last": history[-1]["loss"],
            "test_accuracy": test_acc,
            "cal_accuracy": float(
                (cal_logits.argmax(axis=1) == np.asarray(cal_lab)).mean()
            ),
            "train_accuracy": None,
            "equivalence_check": eq,
        },
        "features": dict(cache_meta, source=feature_source),
        "split_check": split_check,
        "cache_spot_check": cache_spot_check,
        "cal_split_diagnostic": {
            "note": (
                "in-sample for `learned` (it was fit on cal); recorded for "
                "transparency only - the reported comparison is test-side"
            ),
            "learned_per_seed": cal_diag,
        },
        "results": results,
        "rank_by_ece_of_confidence": rank_ece,
        "rank_by_aurc": rank_aurc,
        "rank_by_coverage_at_risk_0.1": rank_cov10,
        "rank_by_coverage_at_risk_0.2": rank_cov20,
        "learned_rank_by_seed": learned_rank_by_seed,
        "sst2_exp6b_reference": sst2_reference,
        "provenance": {
            "backbone_id": args.backbone,
            "backbone_revision": backbone_revision,
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "question": QUESTION,
            "options": list(OPTIONS),
            "seeds": [seed],
            "learned_seeds": list(learned_seeds),
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
                "learned_epochs": args.learned_epochs,
                "learned_lr": args.learned_lr,
                "learned_batch_size": args.learned_batch_size,
                "learned_hidden": args.learned_hidden,
            },
            "environment": bc.environment_info(),
            "splits": {
                "train_n": len(labels["train"]),
                "cal_n": len(labels["cal"]),
                "test_n": len(labels["test"]),
                "train_label_counts": _counts(labels["train"]),
                "cal_label_counts": _counts(labels["cal"]),
                "test_label_counts": _counts(labels["test"]),
                "test_source": (
                    "official test split, first %d rows (same [:test_size] cap "
                    "semantics run_exp7.py applies to the 7600-row official test)"
                    % args.test_size
                ),
                "train_cal_source": (
                    "stratified carve from official train split "
                    "(bc.make_splits, seed %d)" % seed
                ),
                "split_seed": seed,
            },
            "context_timings_s": timings,
            "platform": platform.platform(),
        },
        "wall_time_s": round(time.perf_counter() - t_start, 2),
    }
    out = Path(args.output_dir) / "exp6c_learned_agnews.json"
    bc.write_json(out, payload)
    print("[wrote] " + str(out), flush=True)
    print(
        "[done] learned (mean+-std over %d seeds) ece=%.4f+-%.4f aurc=%.4f+-%.4f "
        "cov@0.1=%.4f+-%.4f cov@0.2=%.4f+-%.4f; split_check=%s; cache_spot_check=%.3g"
        % (
            len(learned_seeds),
            learned_agg["ece_of_confidence"]["mean"],
            learned_agg["ece_of_confidence"]["std"],
            learned_agg["aurc"]["mean"], learned_agg["aurc"]["std"],
            learned_agg["coverage_at_risk_0.1"]["mean"],
            learned_agg["coverage_at_risk_0.1"]["std"],
            learned_agg["coverage_at_risk_0.2"]["mean"],
            learned_agg["coverage_at_risk_0.2"]["std"],
            split_check.get("all_match"),
            float(cache_spot_check.get("max_abs_diff_vs_cache", float("nan"))),
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
