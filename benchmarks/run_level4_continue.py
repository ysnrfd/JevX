# Copyright (c) 2026 ysnrfd
"""WAVE-4 level-4 continuation: structural gates + a quality run.

Why a second file
-----------------
benchmarks/run_level4.py (ADOPTED, written by another session) measured a
quality result on FacebookAI/roberta-base, but two of its five stages are not
useable as level-4 evidence and this script supplies what is missing:

1. its stage "gate" FAILED on the real checkpoint --
   "base_forward diverged from the raw checkpoint by 1.077e+00";
2. its "native" (pooler_output) cell is at chance, with featurize-vs-full-model
   diffs of 5.8 / 8.7 / 23.6 where float re-association is ~1e-6.

Both symptoms have one cause, verified here: FacebookAI/roberta-base ships NO
pooler weights, so RobertaModel(add_pooling_layer=True) random-initialises
pooler.dense on every load. This file makes that reproducible and measurable.

What it does
-------------
--stage gates    Phase-92 execution protocol (load -> REAL input forward ->
                 decision invariants -> save/load equality -> base_forward vs a
                 fresh raw AutoModel -> shared-state single forward ->
                 independent manual recomputation) for a list of (family,
                 backbone, modality, pooling) targets, PLUS a
                 trained_pooled_field probe that loads the checkpoint TWICE and
                 reports whether the named pooled field is bit-stable. Level 4
                 requires a TRAINED/pooled named field, so a field that is
                 randomly re-initialised per load cannot support it.
--stage quality  The established SST-2 protocol on one matrix-cited checkpoint,
                 delegating every numeric step to the functions in
                 benchmarks/run_level4.py so the protocol is identical by
                 construction (same splits, featurize-once, 4 epochs, lr 1e-3,
                 batch 64, max_len 128, ECE(15), calibrators fit on the
                 calibration split only).

Writes only benchmarks/results/level4_*.json. Never touches vendor/, and
unregisters the harness atexit hook so vendor/reports/_jevx_matrix_results.json
(owned by another run) is not modified.
"""
from __future__ import annotations

import argparse
import atexit
import gc
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from transformers import AutoModel

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import bench_common as bc  # noqa: E402
import benchmarks.run_level4 as r4  # noqa: E402  (ADOPTED protocol functions)


def _load_harness():
    spec = importlib.util.spec_from_file_location(
        "jevx_matrix_common", REPO_ROOT / "tests" / "integration" / "matrix_common.py")
    mc = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mc)
    # this run does not own vendor/reports/_jevx_matrix_results.json
    try:
        atexit.unregister(mc._write_results)
    except Exception:  # noqa: BLE001
        pass
    return mc


#: (family, backbone, modality kind, pooling strategy, matrix-row status note)
GATE_TARGETS = [
    ("bert", "google/bert_uncased_L-2_H-128_A-2", "text", "cls",
     "bert row exists"),
    ("bert", "google-bert/bert-base-uncased", "text", "cls", "bert row exists"),
    ("roberta", "FacebookAI/roberta-base", "text", "cls", "roberta row exists"),
    ("deit", "facebook/deit-tiny-patch16-224", "vision", "masked_mean",
     "deit has NO matrix row (breadth-sweep marker only)"),
    ("wav2vec2", "superb/wav2vec2-base-superb-ks", "audio", "masked_mean",
     "wav2vec2 row exists but cites different checkpoints"),
]

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    p.add_argument("--stage", required=True, choices=["gates", "quality"])
    p.add_argument("--backbone", default="google-bert/bert-base-uncased")
    p.add_argument("--dataset", default=bc.DEFAULT_DATASET)
    p.add_argument("--dataset-revision", default=bc.DEFAULT_DATASET_REVISION)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--train-size", type=int, default=3000)
    p.add_argument("--cal-size", type=int, default=1000)
    p.add_argument("--test-size", type=int, default=1500)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--fusion", default="concat")
    p.add_argument("--pooling", default="cls")
    p.add_argument("--n-bins", type=int, default=15)
    p.add_argument("--probe-n", type=int, default=128)
    p.add_argument("--max-project-seconds", type=float, default=7200.0)
    p.add_argument("--out", default=None)
    return p.parse_args(argv)


def _env() -> Dict[str, Any]:
    env = dict(bc.environment_info())
    env["hf_home"] = os.environ.get("HF_HOME")
    return env


def trained_pooled_field_probe(backbone: str, pooling: str) -> Dict[str, Any]:
    """Is the named pooled field TRAINED, or re-initialised on every load?

    Loads the SAME checkpoint twice, independently, and compares (a) every
    parameter and (b) the outputs on a real input. A checkpoint that ships no
    pooler weights shows non-zero differences between the two loads, which is
    the condition that makes the field unusable as a stable representation and
    therefore ineligible for a level-4 "trained/pooled named field" claim.
    """
    from transformers import AutoTokenizer

    out: Dict[str, Any] = {"backbone": backbone, "pooling": pooling}
    models: List[Any] = []
    try:
        for _ in range(2):
            m = AutoModel.from_pretrained(backbone)
            m.eval()
            models.append(m)
        a, b = models[0], models[1]
        out["params_total"] = int(sum(p.numel() for p in a.parameters()))
        out["model_class"] = type(a).__name__
        out["config_add_pooling_layer"] = getattr(a.config, "add_pooling_layer", None)
        sa, sb = a.state_dict(), b.state_dict()
        diffs: Dict[str, float] = {}
        for k in sorted(set(sa) & set(sb)):
            x, y = sa[k].float(), sb[k].float()
            if x.shape == y.shape:
                d = float(np.abs(x - y).max())
                if d > 0:
                    diffs[k] = d
        out["params_differing_between_two_loads"] = diffs
        out["n_params_differing"] = len(diffs)
        out["all_params_bitwise_stable"] = len(diffs) == 0
        tok = AutoTokenizer.from_pretrained(backbone)
        batch = tok(["The maintenance crew inspected the bridge deck."],
                    return_tensors="pt", truncation=True, max_length=32)
        keep = ("input_ids", "attention_mask", "token_type_ids")
        inputs = {k: v for k, v in batch.items() if k in keep}
        with torch.no_grad():
            oa, ob = a(**inputs), b(**inputs)
        po = getattr(oa, "pooler_output", None)
        out["pooler_output_max_abs_diff_two_loads"] = (
            None if po is None
            else float(np.abs(po.float() - ob.pooler_output.float()).max()))
        out["last_hidden_state_max_abs_diff_two_loads"] = float(np.abs(
            oa.last_hidden_state.float() - ob.last_hidden_state.float()).max())
        if out["all_params_bitwise_stable"]:
            out["verdict"] = ("TRAINED_STABLE: every parameter is bit-stable"
                              " across two independent loads")
        else:
            out["verdict"] = (
                "NOT_TRAINED: " + str(len(diffs)) + " parameters differ between"
                " two independent loads of the same checkpoint")
    except Exception as exc:  # noqa: BLE001 - recorded, never silent
        out["error"] = type(exc).__name__ + ": " + str(exc)
    finally:
        del models
        gc.collect()
    return out

def stage_gates(args: argparse.Namespace) -> Dict[str, Any]:
    mc = _load_harness()
    payload: Dict[str, Any] = {
        "experiment": "level4_structural_gates",
        "purpose": ("Phase-92 execution protocol on the checkpoints that carry"
                    " the level-4 quality evidence, plus a trained-vs-random"
                    " probe of each named pooled field."),
        "command": " ".join([Path(sys.argv[0]).name] + list(sys.argv[1:])),
        "environment": _env(),
        "platform": platform.platform(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol_source": "tests/integration/matrix_common.execute_matrix_case",
        "note": ("vendor/reports/_jevx_matrix_results.json is NOT written by this"
                 " run: the harness atexit hook is unregistered so another run's"
                 " artifact is left untouched."),
        "results": {},
    }
    tmp_root = Path(os.environ.get("JEVX_L4_TMP", "D:/jevx_tmp/level4"))
    tmp_root.mkdir(parents=True, exist_ok=True)
    for family, backbone, kind, pooling, row_note in GATE_TARGETS:
        key = family + " | " + backbone
        rec: Dict[str, Any] = {"family": family, "backbone_id": backbone,
                               "modality_kind": kind, "pooling": pooling,
                               "matrix_row_status": row_note}
        t0 = time.perf_counter()
        rec["trained_pooled_field"] = trained_pooled_field_probe(backbone, pooling)
        work = tmp_root / ("matrix-" + family + "-" + backbone.replace("/", "_"))
        work.mkdir(parents=True, exist_ok=True)
        try:
            wrapper, load_s = mc.load_wrapper(backbone, freeze_base_model=True,
                                             pooling_method=pooling,
                                             fusion_type="concat")
            record = mc.execute_matrix_case(
                family, backbone, kind, wrapper, load_s, work,
                test_file="benchmarks/run_level4_continue.py",
                notes="WAVE-4 level-4 gate on the checkpoint that carries the"
                      " quality evidence",
            )
            rec["gate"] = {"status": "PASS", "record": record}
        except Exception as exc:  # noqa: BLE001
            rec["gate"] = {"status": "FAIL",
                           "error": type(exc).__name__ + ": " + str(exc)}
        rec["wall_s"] = round(time.perf_counter() - t0, 2)
        payload["results"][key] = rec
        try:
            mc._RESULTS.clear()
        except Exception:  # noqa: BLE001
            pass
        gc.collect()
        print("[gates] " + key + " -> " + rec["gate"]["status"]
              + " (" + str(rec["wall_s"]) + "s)", flush=True)
    payload["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return payload

def stage_quality(args: argparse.Namespace) -> Dict[str, Any]:
    """Delegate to the ADOPTED protocol functions, so the numbers are comparable."""
    r4_args = argparse.Namespace(
        backbone=args.backbone, dataset=args.dataset,
        dataset_revision=args.dataset_revision, seeds=args.seeds,
        split_seed=args.split_seed, train_size=args.train_size,
        cal_size=args.cal_size, test_size=args.test_size, epochs=args.epochs,
        lr=args.lr, batch_size=args.batch_size, max_length=args.max_length,
        fusion=args.fusion, pooling=args.pooling, n_bins=args.n_bins,
        probe_n=args.probe_n, max_project_seconds=args.max_project_seconds,
        out=args.out,
    )
    r4.FAMILY = args.backbone.split("/")[-1].split("-")[0]
    res = r4.Result(Path(args.out), r4_args)
    res.payload["experiment"] = "level4_wave4"
    res.payload["runner"] = ("benchmarks/run_level4_continue.py --stage quality;"
                            " every numeric step delegated to the functions in"
                            " benchmarks/run_level4.py")
    data = r4.load_data(r4_args)
    splits = r4.make_splits(data, r4_args)
    feats = r4.stage_features(r4_args, data, splits, res)
    r4.stage_jevx(r4_args, data, splits, feats, res)
    r4.stage_baselines(r4_args, data, splits, feats, res)
    for stage in r4.ALL_STAGES:
        if stage not in res.payload["stages_run"]:
            res.payload["stages_skipped"].append(stage)
    res.finish()
    return res.payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.out is None:
        if args.stage == "gates":
            args.out = str(REPO_ROOT / "benchmarks" / "results"
                           / "level4_structural_gates.json")
        else:
            tag = args.backbone.split("/")[-1]
            args.out = str(REPO_ROOT / "benchmarks" / "results"
                           / ("level4_" + tag + "_sst2.json"))
    if args.stage == "gates":
        payload = stage_gates(args)
        bc.write_json(Path(args.out), payload)
        print("[write] " + args.out, flush=True)
        return 0
    payload = stage_quality(args)
    print("[write] " + args.out, flush=True)
    return 1 if payload.get("failures") else 0


if __name__ == "__main__":
    raise SystemExit(main())
