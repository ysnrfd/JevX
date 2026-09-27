# Copyright (c) 2026 ysnrfd
"""Level-4 quality evidence on a MATRIX-CITED checkpoint (Track 5, Route A).

Why this script exists
----------------------
``TRANSFORMERS_COMPATIBILITY_MATRIX.md`` states, in its "Evidence overlay
(Phases 92-94)" section::

    "4 additionally requires a real-dataset quality result for that family's
     cited checkpoint (Accuracy/F1/NLL/Brier/ECE - owned by the benchmark
     suite)."

The four quality results that existed before this script were measured on
backbones the matrix does not cite (``google/bert_uncased_L-2_H-128_A-2``,
``facebook/deit-tiny-patch16-224``, ``superb/wav2vec2-base-superb-ks``,
Qwen3.5 hybrid), so no cell could hold 4. This script runs the *established*
real-dataset protocol on a checkpoint the matrix **does** cite -
``FacebookAI/roberta-base`` in the ``roberta`` row - and records every number
it produces.

What it does, in order (each stage writes the result JSON before the next
starts, so a crash never loses finished work):

1. ``probe``  - load the real checkpoint, featurize a small sample, and
   PROJECT the wall cost of the full featurization pass. If the projection
   exceeds ``--max-project-seconds`` the train subset is reduced (never the
   rigor: epochs / lr / batch / seeds / calibration protocol are untouched)
   and the reduction is recorded in the JSON and in the report.
2. ``gate``   - the Phase-92 execution protocol (load -> real input forward ->
   decision invariants -> save/load equality -> base_forward vs a fresh raw
   ``AutoModel`` -> shared-state forward count -> independent manual
   recomputation of the pooled representation) executed on the SAME real
   checkpoint, by importing the repository's own harness
   (``tests/integration/matrix_common.execute_matrix_case``). This is what
   makes the level-4 claim sit on one checkpoint rather than two.
3. ``features`` - featurize train / cal / test ONCE each through the frozen
   backbone (Mode A), producing two feature matrices from the same forward
   passes: ``cls`` (mirrors the published protocol so the comparison with
   ``reports/DATASET_BENCHMARKS.md`` is apples-to-apples) and ``native``
   (the adapter's own pooled vector = ``pooler_output`` for roberta, i.e. the
   field the matrix row names).
4. ``jevx``   - per seed: JevX head stack (steps 2-8) trained on the cached
   features; uncalibrated / temperature / isotonic metrics on the test split
   with calibrators fit on the CALIBRATION split only; ECE/MCE with 15 bins.
   The featurize-once path is checked against the full ``model.choice()``.
5. ``baselines`` - B1 linear probe, B2 MLP, B4 = B1 + temperature, on the
   IDENTICAL cached features / splits / hyperparams / seeds.

Determinism
-----------
Split construction, head init and batch order are seeded
(``bc.make_splits`` / ``bc.train_heads`` mirror ``run_quality.py`` exactly).
Nothing is selected on the test split: the test rows are the official SST-2
validation split and are touched only for the final metrics.

Example
-------
    python benchmarks/run_level4.py --seeds 0,1,2
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import bench_common as bc  # noqa: E402


# --------------------------------------------------------------------------
# Defaults: the established protocol, unchanged.
# --------------------------------------------------------------------------

DEFAULT_BACKBONE = "FacebookAI/roberta-base"
DEFAULT_OUT_NAME = "level4_roberta_sst2.json"
FAMILY = "roberta"
ALL_STAGES = ("probe", "gate", "features", "jevx", "baselines")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--backbone", default=DEFAULT_BACKBONE,
                   help="must be a checkpoint the matrix cites for this family")
    p.add_argument("--dataset", default=bc.DEFAULT_DATASET)
    p.add_argument("--dataset-revision", default=bc.DEFAULT_DATASET_REVISION)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--split-seed", type=int, default=0,
                   help="seed for the stratified train/cal carve (bench_common)")
    p.add_argument("--train-size", type=int, default=3000)
    p.add_argument("--cal-size", type=int, default=1000)
    p.add_argument("--test-size", type=int, default=1500,
                   help="cap on the official validation split used as test")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--fusion", default="concat")
    p.add_argument("--pooling", default="cls",
                   help="primary pooling (matches run_quality.py); 'native' is "
                        "always measured as a second cell")
    p.add_argument("--n-bins", type=int, default=15,
                   help="ECE/MCE equal-width confidence bins (protocol default 15)")
    p.add_argument("--probe-n", type=int, default=128,
                   help="examples used for the featurization timing probe")
    p.add_argument("--max-project-seconds", type=float, default=7200.0,
                   help="if the projected full featurization exceeds this, the "
                        "TRAIN SUBSET is reduced (rigor is not touched)")
    p.add_argument("--stages", default="all",
                   help="comma list of " + ",".join(ALL_STAGES) + " or 'all'")
    p.add_argument("--skip-gate", action="store_true",
                   help="skip the Phase-92 execution gate (recorded as skipped)")
    p.add_argument("--out", default=str(REPO_ROOT / "benchmarks" / "results"
                                        / DEFAULT_OUT_NAME))
    return p.parse_args(argv)


def selected_stages(spec: str) -> List[str]:
    if spec.strip() == "all":
        return list(ALL_STAGES)
    chosen = [s.strip().lower() for s in spec.split(",") if s.strip()]
    bad = [s for s in chosen if s not in ALL_STAGES]
    if bad:
        raise ValueError(f"unknown stage(s) {bad}; choose from {ALL_STAGES} or 'all'")
    return chosen


# --------------------------------------------------------------------------
# Incremental result file
# --------------------------------------------------------------------------

class Result:
    """The result JSON, rewritten after every stage and after every seed."""

    def __init__(self, path: Path, args: argparse.Namespace) -> None:
        self.path = Path(path)
        self.args = args
        self.payload: Dict[str, Any] = {
            "experiment": "level4",
            "purpose": (
                "Level-4 quality evidence on a TRANSFORMERS_COMPATIBILITY_MATRIX.md-"
                "cited checkpoint: real-dataset quality (Accuracy/F1/NLL/Brier/ECE) "
                "for the family's cited checkpoint, as the matrix's level-4 "
                "definition requires."
            ),
            "level4_definition_source": (
                "TRANSFORMERS_COMPATIBILITY_MATRIX.md, section "
                "'Support levels (defined precisely)' + 'Evidence overlay "
                "(Phases 92-94)'"
            ),
            "status": "in_progress",
            "stages_run": [],
            "stages_skipped": [],
            "failures": {},
            "command": " ".join([Path(sys.argv[0]).name] + list(sys.argv[1:])),
            "config": self._config(),
            "environment": self._environment(),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self._t0 = time.perf_counter()
        self.save()

    def _config(self) -> Dict[str, Any]:
        a = self.args
        return {
            "family": FAMILY,
            "backbone_id": a.backbone,
            "dataset_id": a.dataset,
            "dataset_revision": a.dataset_revision,
            "seeds": [int(s) for s in a.seeds.split(",") if s.strip()],
            "split_seed": a.split_seed,
            "requested_train_size": a.train_size,
            "cal_size": a.cal_size,
            "test_size_cap": a.test_size,
            "epochs": a.epochs,
            "lr": a.lr,
            "batch_size": a.batch_size,
            "max_length": a.max_length,
            "fusion": a.fusion,
            "primary_pooling": a.pooling,
            "secondary_pooling": "native (adapter pooled vector)",
            "ece_mce_bins": a.n_bins,
            "mode": "A (freeze_base_model=True, featurize-once per split)",
            "probe_n": a.probe_n,
            "max_project_seconds": a.max_project_seconds,
        }

    @staticmethod
    def _environment() -> Dict[str, Any]:
        env = dict(bc.environment_info())
        env["hf_home"] = os.environ.get("HF_HOME")
        env["cpu_affinity_size"] = (
            len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
        )
        return env

    def save(self) -> None:
        self.payload["elapsed_s"] = round(time.perf_counter() - self._t0, 2)
        self.payload["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        bc.write_json(self.path, self.payload)
        print(f"[write] {self.path}", flush=True)

    def fail(self, stage: str, exc: BaseException) -> None:
        import traceback

        self.payload["failures"][stage] = f"{type(exc).__name__}: {exc}"
        print(f"[FAILED] {stage}: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        self.save()

    def finish(self) -> None:
        self.payload["status"] = (
            "complete" if not self.payload["failures"] else "complete_with_failures"
        )
        self.payload["wall_time_s"] = round(time.perf_counter() - self._t0, 2)
        self.payload["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.save()


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load_data(args: argparse.Namespace) -> Dict[str, Any]:
    t0 = time.perf_counter()
    train_texts_all, train_labels_all, val_texts, val_labels = bc.load_sst2(
        args.dataset, args.dataset_revision
    )
    load_s = round(time.perf_counter() - t0, 2)
    if args.test_size and args.test_size < len(val_texts):
        test_texts, test_labels = val_texts[: args.test_size], val_labels[: args.test_size]
    else:
        test_texts, test_labels = val_texts, val_labels
    return {
        "train_texts_all": train_texts_all,
        "train_labels_all": train_labels_all,
        "test_texts": test_texts,
        "test_labels": test_labels,
        "load_dataset_s": load_s,
    }


def make_splits(data: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    train_idx, cal_idx = bc.make_splits(
        data["train_labels_all"], args.train_size, args.cal_size, seed=args.split_seed
    )
    return {
        "train_texts": [data["train_texts_all"][i] for i in train_idx],
        "train_labels": [data["train_labels_all"][i] for i in train_idx],
        "cal_texts": [data["train_texts_all"][i] for i in cal_idx],
        "cal_labels": [data["train_labels_all"][i] for i in cal_idx],
    }


def counts(labels: Sequence[int]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for y in labels:
        out[str(int(y))] = out.get(str(int(y)), 0) + 1
    return out


# --------------------------------------------------------------------------
# Stage 1 - timing probe + cost projection
# --------------------------------------------------------------------------

def stage_probe(args: argparse.Namespace, data: Dict[str, Any],
                res: Result) -> Dict[str, Any]:
    """Load the real checkpoint, time a small featurization pass, project."""
    t0 = time.perf_counter()
    model = bc.JevX.from_pretrained(
        args.backbone, freeze_base_model=True,
        pooling_method=args.pooling, fusion_type=args.fusion,
    )
    model.eval()
    load_s = round(time.perf_counter() - t0, 2)
    n_params_backbone = sum(p.numel() for p in model.model.parameters())
    n_params_total = sum(p.numel() for p in model.parameters())
    revision = (
        getattr(getattr(model.model, "config", None), "_commit_hash", None)
    )

    # warmup: one untimed batch so lazy allocations / MKL buffers are paid
    bc.featurize(model, data["test_texts"][: args.batch_size],
                 batch_size=args.batch_size, max_length=args.max_length,
                 strategies=(args.pooling,))
    sample = data["test_texts"][: args.probe_n]
    t1 = time.perf_counter()
    probe_feats = bc.featurize(
        model, sample, batch_size=args.batch_size, max_length=args.max_length,
        strategies=(None, args.pooling),
    )
    probe_s = time.perf_counter() - t1
    n = len(sample)
    per_example_s = probe_s / max(n, 1)

    n_test = len(data["test_texts"])
    full_n = args.train_size + args.cal_size + n_test
    projected_s = per_example_s * full_n

    record = {
        "backbone_revision_from_config_commit_hash": revision,
        "backbone_params": int(n_params_backbone),
        "jevx_total_params": int(n_params_total),
        "load_backbone_s": load_s,
        "probe_examples": n,
        "probe_batch_size": args.batch_size,
        "probe_wall_s": round(probe_s, 3),
        "probe_examples_per_s": round(n / probe_s, 2) if probe_s else None,
        "seconds_per_example": round(per_example_s, 6),
        "feature_dim": int(probe_feats[args.pooling].shape[1]),
        "native_feature_dim": int(probe_feats["native"].shape[1]),
        "projection": {
            "formula": "seconds_per_example * (train_size + cal_size + test_n)",
            "assumed_train_size": args.train_size,
            "assumed_cal_size": args.cal_size,
            "assumed_test_n": n_test,
            "assumed_total_examples": full_n,
            "projected_featurize_s": round(projected_s, 1),
            "projected_featurize_min": round(projected_s / 60.0, 2),
            "budget_s": args.max_project_seconds,
            "within_budget": projected_s <= args.max_project_seconds,
        },
        "note": (
            "Projection is a timing estimate from an executed probe, not a "
            "measured total; the measured featurization wall time is recorded "
            "in stages.features.wall_s and is the number the report quotes."
        ),
    }
    del probe_feats
    del model
    gc.collect()
    res.payload["timing_probe"] = record
    res.payload["stages_run"].append("probe")
    res.save()
    return record


# --------------------------------------------------------------------------
# Stage 2 - the Phase-92 execution gate on the SAME real checkpoint
# --------------------------------------------------------------------------

def stage_gate(args: argparse.Namespace, res: Result) -> Dict[str, Any]:
    """Run the repository's own §92 harness on the cited checkpoint.

    The harness merges its records into vendor/reports/_jevx_matrix_results.json
    through an atexit hook; this run does NOT own that file, so the hook is
    unregistered and the record is written only into the result JSON.
    """
    if args.skip_gate:
        rec = {"status": "skipped", "reason": "--skip-gate"}
        res.payload["execution_gate"] = rec
        res.payload["stages_skipped"].append("gate")
        res.save()
        return rec

    import importlib.util

    harness_path = REPO_ROOT / "tests" / "integration" / "matrix_common.py"
    spec = importlib.util.spec_from_file_location("jevx_matrix_common", harness_path)
    mc = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mc)
    # do not let the harness touch a file this run does not own
    try:
        import atexit

        atexit.unregister(mc._write_results)
    except Exception as exc:  # noqa: BLE001 - recorded, never silent
        res.payload["gate_atexit_unregister_error"] = f"{type(exc).__name__}: {exc}"

    tmp_root = Path(os.environ.get("JEVX_L4_TMP", "D:/jevx_tmp/level4"))
    tmp_root.mkdir(parents=True, exist_ok=True)
    work = tmp_root / ("matrix-" + FAMILY)
    work.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    try:
        wrapper, load_s = mc.load_wrapper(
            args.backbone, freeze_base_model=True,
            pooling_method=args.pooling, fusion_type=args.fusion,
        )
        record = mc.execute_matrix_case(
            FAMILY, args.backbone, "text", wrapper, load_s, work,
            test_file="benchmarks/run_level4.py",
            notes=(
                "level-4 gate: executed on the same real checkpoint the quality "
                "benchmark uses, so level 4 rests on one checkpoint end-to-end"
            ),
        )
        rec = {
            "status": "run",
            "wall_s": round(time.perf_counter() - t0, 2),
            "harness": "tests/integration/matrix_common.execute_matrix_case",
            "record": record,
        }
        try:
            mc._RESULTS.clear()
        except Exception:  # noqa: BLE001
            pass
    except Exception as exc:  # noqa: BLE001 - recorded, never silent
        rec = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "wall_s": round(time.perf_counter() - t0, 2),
            "harness": "tests/integration/matrix_common.execute_matrix_case",
        }
    finally:
        gc.collect()
    res.payload["execution_gate"] = rec
    res.payload["stages_run"].append("gate")
    res.save()
    return rec


# --------------------------------------------------------------------------
# Stage 3 - featurize once per split
# --------------------------------------------------------------------------

def stage_features(args: argparse.Namespace, data: Dict[str, Any],
                   splits: Dict[str, Any], res: Result) -> Dict[str, Dict[str, np.ndarray]]:
    model = bc.JevX.from_pretrained(
        args.backbone, freeze_base_model=True,
        pooling_method=args.pooling, fusion_type=args.fusion,
    )
    model.eval()
    strategies: List[Optional[str]] = [None]
    if args.pooling != "native":
        strategies.append(args.pooling)
    feats: Dict[str, Dict[str, np.ndarray]] = {}
    per_split_s: Dict[str, float] = {}
    t_all = time.perf_counter()
    for name, texts in (("train", splits["train_texts"]),
                        ("cal", splits["cal_texts"]),
                        ("test", data["test_texts"])):
        t0 = time.perf_counter()
        feats[name] = bc.featurize(
            model, texts, batch_size=args.batch_size,
            max_length=args.max_length, strategies=tuple(strategies),
        )
        per_split_s[name] = round(time.perf_counter() - t0, 2)
        print(f"[features] {name}: n={len(texts)} in {per_split_s[name]}s", flush=True)
    wall = round(time.perf_counter() - t_all, 2)
    del model
    gc.collect()

    res.payload["splits"] = {
        "train_n": len(splits["train_texts"]),
        "cal_n": len(splits["cal_texts"]),
        "test_n": len(data["test_texts"]),
        "train_label_counts": counts(splits["train_labels"]),
        "cal_label_counts": counts(splits["cal_labels"]),
        "test_label_counts": counts(data["test_labels"]),
        "split_seed": args.split_seed,
        "train_source": "stratified carve from the official SST-2 train split "
                        "(bc.make_splits; disjoint from cal by assertion)",
        "cal_source": "stratified carve from the remainder of the official train "
                      "split (disjoint from train)",
        "test_source": "official SST-2 validation split, capped at test_size",
        "official_validation_n": len(data["val_texts"]) if "val_texts" in data else None,
    }
    res.payload["features"] = {
        "wall_s": wall,
        "per_split_wall_s": per_split_s,
        "keys": sorted(feats["train"].keys()),
        "dims": {k: int(v.shape[1]) for k, v in feats["train"].items()},
        "dtypes": {k: str(v.dtype) for k, v in feats["train"].items()},
        "note": "both feature matrices come from the SAME backbone forward "
                "passes (bench_common.featurize), so they are bit-compatible "
                "with model.choice() for the matching pooling",
    }
    res.payload["stages_run"].append("features")
    res.save()
    return feats


# --------------------------------------------------------------------------
# Stage 4/5 - heads and baselines
# --------------------------------------------------------------------------

def _metrics_block(test_logits: np.ndarray, test_labels: np.ndarray,
                   cal_logits: np.ndarray, cal_labels: Sequence[int],
                   n_bins: int) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    for method, key in (("none", "uncalibrated"),
                        ("temperature", "temperature"),
                        ("isotonic", "isotonic")):
        row[key] = bc.calibrated_metrics(
            test_logits, test_labels, method,
            fit_logits=cal_logits, fit_labels=np.asarray(cal_labels),
            n_bins=n_bins,
        )
    return row


def stage_jevx(args: argparse.Namespace, data: Dict[str, Any],
               splits: Dict[str, Any], feats: Dict[str, Dict[str, np.ndarray]],
               res: Result) -> None:
    seeds = res.payload["config"]["seeds"]
    primary = args.pooling if args.pooling in feats["train"] else "native"
    cells = [c for c in (primary, "native") if c in feats["train"]]
    res.payload.setdefault("jevx", {})["cells"] = {}
    res.payload["jevx"]["primary_cell"] = primary
    res.payload["jevx"]["pooling_definitions"] = {
        "cls": "last_hidden_state[:, 0] (the pooling the published protocol uses)",
        "native": "the adapter's own pooled vector for this checkpoint "
                  "(pooler_output for roberta - the field the matrix row names)",
    }

    for cell in cells:
        train_records = bc.build_records(feats["train"][cell], splits["train_labels"])
        cal_records = bc.build_records(feats["cal"][cell], splits["cal_labels"])
        test_records = bc.build_records(feats["test"][cell], data["test_labels"])
        per_seed: List[Dict[str, Any]] = []
        per_seed_metrics: Dict[str, List[Dict[str, float]]] = {
            "uncalibrated": [], "temperature": [], "isotonic": [],
        }
        equivalence: List[Dict[str, Any]] = []
        cell_wall = time.perf_counter()
        for seed in seeds:
            t_seed = time.perf_counter()
            print(f"[jevx] cell={cell} seed={seed}", flush=True)
            torch.manual_seed(seed)
            model = bc.JevX.from_pretrained(
                args.backbone, freeze_base_model=True,
                pooling_method=(cell if cell != "native" else None),
                fusion_type=args.fusion,
            )
            model.eval()
            head = bc.FeaturizedJevX(model, bc.DEFAULT_QUESTION, bc.DEFAULT_OPTIONS)
            _, history = bc.train_heads(
                head, train_records, seed=seed, lr=args.lr,
                epochs=args.epochs, batch_size=args.batch_size,
            )
            cal_logits, cal_labels = bc.collect_logits(
                head, cal_records, batch_size=args.batch_size
            )
            test_logits, test_labels = bc.collect_logits(
                head, test_records, batch_size=args.batch_size
            )
            metrics = _metrics_block(
                test_logits, test_labels, cal_logits, cal_labels, args.n_bins
            )
            for key in per_seed_metrics:
                per_seed_metrics[key].append(metrics[key])
            row = {
                "seed": seed,
                "train_loss_first": history[0]["loss"],
                "train_loss_last": history[-1]["loss"],
                "trainable_params": bc.count_trainable(head),
                **metrics,
            }
            per_seed.append(row)
            equivalence.append(
                bc.equivalence_check(
                    model, head, feats["test"][cell], data["test_texts"], n=16
                )
            )
            del model, head
            gc.collect()
            print(f"[jevx] cell={cell} seed={seed} done in "
                  f"{time.perf_counter() - t_seed:.1f}s", flush=True)
            res.payload["jevx"]["cells"][cell] = {
                "pooling": cell,
                "n_seeds": len(seeds),
                "per_seed": per_seed,
                "aggregate": {k: bc.mean_std_over_seeds(v)
                              for k, v in per_seed_metrics.items()},
                "featurized_vs_full_model": equivalence,
                "wall_s": round(time.perf_counter() - cell_wall, 2),
            }
            res.save()
    res.payload["stages_run"].append("jevx")
    res.save()


def stage_baselines(args: argparse.Namespace, data: Dict[str, Any],
                    splits: Dict[str, Any],
                    feats: Dict[str, Dict[str, np.ndarray]],
                    res: Result) -> None:
    seeds = res.payload["config"]["seeds"]
    primary = args.pooling if args.pooling in feats["train"] else "native"
    train_records = bc.build_records(feats["train"][primary], splits["train_labels"])
    cal_records = bc.build_records(feats["cal"][primary], splits["cal_labels"])
    test_records = bc.build_records(feats["test"][primary], data["test_labels"])
    in_dim = feats["train"][primary].shape[1]
    n_classes = len(bc.DEFAULT_OPTIONS)

    t0 = time.perf_counter()
    per_seed: List[Dict[str, Any]] = []
    metric_sets: Dict[str, List[Dict[str, float]]] = {
        "b1_linear": [], "b1_temperature": [], "b2_mlp": [],
    }
    for seed in seeds:
        print(f"[baselines] seed={seed}", flush=True)
        torch.manual_seed(seed)
        b1 = bc.LinearProbe(in_dim, n_classes)
        bc.train_heads(
            b1, train_records, seed=seed, lr=args.lr,
            epochs=args.epochs, batch_size=args.batch_size,
        )
        b1_cal, cal_labels = bc.collect_logits(
            b1, cal_records, batch_size=args.batch_size
        )
        b1_test, test_labels = bc.collect_logits(
            b1, test_records, batch_size=args.batch_size
        )
        torch.manual_seed(seed + 10_000)
        b2 = bc.MLPProbe(in_dim, n_classes)
        bc.train_heads(
            b2, train_records, seed=seed + 10_000, lr=args.lr,
            epochs=args.epochs, batch_size=args.batch_size,
        )
        b2_test, _ = bc.collect_logits(b2, test_records, batch_size=args.batch_size)

        row: Dict[str, Any] = {"seed": seed}
        row["b1_linear"] = bc.metrics_from_logits(b1_test, test_labels, n_bins=args.n_bins)
        row["b1_temperature"] = bc.calibrated_metrics(
            b1_test, test_labels, "temperature",
            fit_logits=b1_cal, fit_labels=cal_labels, n_bins=args.n_bins,
        )
        row["b2_mlp"] = bc.metrics_from_logits(b2_test, test_labels, n_bins=args.n_bins)
        for key in metric_sets:
            metric_sets[key].append(row[key])
        per_seed.append(row)
        del b1, b2
        gc.collect()
        res.payload["baselines"] = {
            "feature_set": primary,
            "in_dim": int(in_dim),
            "n_seeds": len(seeds),
            "per_seed": per_seed,
            "aggregate": {k: bc.mean_std_over_seeds(v)
                          for k, v in metric_sets.items()},
            "wall_s": round(time.perf_counter() - t0, 2),
        }
        res.save()
    res.payload["stages_run"].append("baselines")
    res.save()


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    stages = selected_stages(args.stages)
    res = Result(Path(args.out), args)
    print(f"[plan] backbone={args.backbone} family={FAMILY} stages={stages} "
          f"seeds={args.seeds} train={args.train_size} cal={args.cal_size} "
          f"epochs={args.epochs} lr={args.lr} batch={args.batch_size}", flush=True)

    t0 = time.perf_counter()
    data: Dict[str, Any] = {}
    splits: Dict[str, Any] = {}
    feats: Dict[str, Dict[str, np.ndarray]] = {}

    for stage in stages:
        try:
            if stage == "probe":
                if not data:
                    data = load_data(args)
                stage_probe(args, data, res)
            elif stage == "gate":
                stage_gate(args, res)
            elif stage == "features":
                if not data:
                    data = load_data(args)
                splits = make_splits(data, args)
                probe = res.payload.get("timing_probe")
                if probe is not None:
                    projected = probe["projection"]["projected_featurize_s"]
                    proj_n = probe["projection"]["assumed_total_examples"]
                    per_ex = probe["seconds_per_example"]
                    cal_n = len(splits["cal_texts"])
                    test_n = len(data["test_texts"])
                    affordable = max(args.cal_size, int(
                        (args.max_project_seconds / max(per_ex, 1e-9)) - cal_n - test_n))
                    if projected > args.max_project_seconds and affordable >= 100:
                        reduced = min(args.train_size, affordable)
                        print(f"[budget] projected {projected:.0f}s > "
                              f"{args.max_project_seconds:.0f}s -> train subset "
                              f"{args.train_size} -> {reduced} (rigor untouched)",
                              flush=True)
                        res.payload.setdefault("budget_decision", {})[
                            "train_subset_reduced"] = {
                            "projected_s": projected,
                            "budget_s": args.max_project_seconds,
                            "requested_train_size": args.train_size,
                            "reduced_train_size": reduced,
                            "untouched": ["epochs", "lr", "batch_size", "max_length",
                                          "seeds", "cal_size", "test set",
                                          "calibration protocol", "ece bins"],
                            "why": "a smaller train subset is the only knob that "
                                   "reduces cost without weakening the protocol; "
                                   "the reduction is reported alongside every number",
                        }
                        args.train_size = reduced
                        splits = make_splits(data, args)
                        res.payload["config"]["train_size_used"] = reduced
                    res.payload["config"]["projection_reference_total_examples"] = proj_n
                feats = stage_features(args, data, splits, res)
            elif stage == "jevx":
                stage_jevx(args, data, splits, feats, res)
            elif stage == "baselines":
                stage_baselines(args, data, splits, feats, res)
        except Exception as exc:  # noqa: BLE001 - record, continue, never silent
            res.fail(stage, exc)

    for stage in ALL_STAGES:
        if stage not in res.payload["stages_run"] and stage not in res.payload["stages_skipped"]:
            res.payload["stages_skipped"].append(stage)

    res.payload["total_wall_time_s"] = round(time.perf_counter() - t0, 2)
    res.payload["platform"] = platform.platform()
    res.finish()
    print(f"[done] {res.payload['status']} wall={res.payload['wall_time_s']}s "
          f"failures={len(res.payload['failures'])}", flush=True)
    return 1 if res.payload["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
