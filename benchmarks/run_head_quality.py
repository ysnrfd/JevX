# Copyright (c) 2026 ysnrfd
"""EXP-H: controlled diagnosis of the JevX choice head on FROZEN features.

The measured problem this script isolates: on IDENTICAL pre-computed pooled
features, the JevX choice head LOSES to a plain linear / MLP probe
(reports/DATASET_BENCHMARKS.md, reports/MODALITY_BENCHMARKS.md,
reports/REAL_SCENARIO_VALIDATION.md). A decision layer that cannot match a
1-layer MLP on fixed features is a real quality gap.

Every cell runs under ONE identical protocol - same frozen feature matrix,
same labels, same splits, same seeds {0,1,2}, same epochs, same optimizer/lr,
same batch size, same HeadTrainer + choice_loss - so the ONLY thing that
varies is the scoring function under test:

  P1  p1_linear          Linear[D -> K] on the pooled state (== baseline B1).
  P2  p2_mlp             Linear[D -> H] -> ReLU -> Linear[H -> K] (== B2).
  P3  p3_jevx_dot        JevX choice head, interaction="dot".
  P3b p3b_jevx_bilinear  JevX choice head, interaction="bilinear" (SHIPPED).
  P3c p3c_jevx_mlp       JevX choice head, interaction="mlp".
  P4  p4_state_only      JevX choice head with the question/schema pathway
                         bypassed: fused = projected pooled state, option
                         rows = a FIXED state-independent basis. Isolates the
                         option-interaction parameterization from the fusion.
  P5  p5_probe_init      JevX choice head whose STATE PATHWAY is initialized
                         from a closed-form least-squares linear probe of the
                         frozen features. Isolates initialization scale.
  P6  p6_jevx_v2         The FIXED head: additive state-only linear term +
                         option-interaction residual (zero-initialized).
  P7..P11                   EXP-H follow-up: the SAME v2 head and the SAME full
                         question+schema stack, with the OPTION/SCHEMA pathway
                         (the fusion option formula only) patched per candidate.
                         P7 = candidate (a) zero-init neutral option pathway,
                         P8/P9 = candidate (b) state-independent LayerNorm of
                         the option features (branch / fused row),
                         P10 = candidate (c) option-pathway dropout,
                         P11 = (a)+(b2) the combination.

The backbone is frozen and featurized ONCE per split, so no backbone is in
the loop: this is exactly head-vs-probe on the same numbers.

Usage (from the repo root):
    python benchmarks/run_head_quality.py --datasets synthetic sst2 ag_news yelp
    python benchmarks/run_head_quality.py --datasets ravdess
    python benchmarks/run_head_quality.py --datasets all --smoke

Result files : benchmarks/results/head_quality_<dataset>.json
Cached features: benchmarks/cache/head_quality/<dataset>_features.npz
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import bench_common as bc  # noqa: E402

CACHE_DIR = REPO_ROOT / "benchmarks" / "cache" / "head_quality"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"

# --------------------------------------------------------------------------
# Dataset registry. Every protocol constant is copied VERBATIM from the
# benchmark that produced the published number, so the v1 baseline reproduced
# here is directly comparable with the existing report.
# --------------------------------------------------------------------------

TEXT_BACKBONE = bc.DEFAULT_BACKBONE

DATASETS: Dict[str, Dict[str, Any]] = {
    "synthetic": {
        "modality": "synthetic",
        "backbone": None,
        "pooling": None,
        "fusion": "concat",
        "max_length": 0,
        "question": "Which Gaussian component generated this vector?",
        "options": ["c0", "c1", "c2", "c3"],
        "train_size": 3000,
        "cal_size": 1000,
        "test_size": 1500,
        "epochs": 4,
        "lr": 1e-3,
        "batch_size": 64,
        "source": "synthetic hard Gaussian-mixture control (no backbone)",
    },
    "sst2": {
        "modality": "text",
        "backbone": TEXT_BACKBONE,
        "pooling": "cls",
        "fusion": "concat",
        "max_length": 128,
        "question": bc.DEFAULT_QUESTION,
        "options": list(bc.DEFAULT_OPTIONS),
        "train_size": 3000,
        "cal_size": 1000,
        "test_size": 1500,
        "epochs": 4,
        "lr": 1e-3,
        "batch_size": 64,
        "source": "benchmarks/run_quality.py (EXP-1 / EXP-2)",
    },
    "ag_news": {
        "modality": "text",
        "backbone": TEXT_BACKBONE,
        "pooling": "cls",
        "fusion": "concat",
        "max_length": 128,
        "question": "What is the topic of this news article?",
        "options": ["World", "Sports", "Business", "Sci/Tech"],
        "train_size": 3000,
        "cal_size": 1000,
        "test_size": 1500,
        "epochs": 4,
        "lr": 1e-3,
        "batch_size": 64,
        "source": "benchmarks/run_exp7.py (EXP-7)",
    },
    "yelp": {
        "modality": "text",
        "backbone": TEXT_BACKBONE,
        "pooling": "cls",
        "fusion": "concat",
        "max_length": 128,
        "question": "What star rating will this reviewer give?",
        "options": ["1 star", "2 star", "3 stars", "4 stars", "5 stars"],
        "train_size": 4000,
        "cal_size": 1200,
        "test_size": 1500,
        "epochs": 4,
        "lr": 1e-3,
        "batch_size": 64,
        "source": "benchmarks/run_real_scenario.py (Yelp phase1 / phase2)",
    },
    "ravdess": {
        "modality": "audio",
        "backbone": "superb/wav2vec2-base-superb-ks",
        "pooling": None,
        "fusion": "concat",
        "max_length": 0,
        "question": "Which emotion is spoken in this audio?",
        "options": [
            "neutral", "calm", "happy", "sad", "angry", "fearful",
            "disgusted", "surprised",
        ],
        "train_size": 700,
        "cal_size": 240,
        "test_size": 500,
        "epochs": 4,
        "lr": 1e-3,
        "batch_size": 64,
        "source": "benchmarks/run_modality.py --modality audio (EXP-A)",
    },
}

ALL_CELLS: Tuple[str, ...] = (
    "p1_linear",
    "p2_mlp",
    "p3_jevx_dot",
    "p3b_jevx_bilinear",
    "p3c_jevx_mlp",
    "p4_state_only",
    "p5_probe_init",
    "p6_jevx_v2",
    "p7_option_neutral",
    "p8_option_ln_branch",
    "p9_option_ln_row",
    "p10_option_dropout",
    "p11_option_neutral_ln",
    "p12_neutral_residual",
    "p13_unit_option",
    "p14_stateless_option",
    "p15_option_unnormalized",
)

CELL_LABELS: Dict[str, str] = {
    "p1_linear": "P1 Linear probe [D->K] on pooled state (= baseline B1)",
    "p2_mlp": "P2 MLP probe [D->H->K] on pooled state (= baseline B2)",
    "p3_jevx_dot": "P3 JevX choice head v1, interaction=dot",
    "p3b_jevx_bilinear": "P3b JevX choice head v1 (HEAD V1 as shipped), interaction=bilinear",
    "p3c_jevx_mlp": "P3c JevX choice head v1, interaction=mlp",
    "p4_state_only": "P4 JevX head v1, question/schema pathway bypassed (state-only)",
    "p5_probe_init": "P5 JevX head v1 with a probe-initialized state pathway (state_proj)",
    "p6_jevx_v2": "P6 JevX head v2 (additive state-only linear term + zero-init residual)",
    "p7_option_neutral": "P7 v2 head + candidate (a): option pathway zero-init (neutral at step 0)",
    "p8_option_ln_branch": "P8 v2 head + candidate (b1): LayerNorm on the projected option rows",
    "p9_option_ln_row": "P9 v2 head + candidate (b2): LayerNorm on the fused option row e_k",
    "p10_option_dropout": "P10 v2 head + candidate (c): dropout(0.5) on the option branch (train only)",
    "p11_option_neutral_ln": "P11 v2 head + (a) and (b2) together",
    "p12_neutral_residual": "P12 v2 head + candidate (a) at the RESIDUAL: learnable zero-init gain on r_k",
    "p13_unit_option": "P13 v2 head + candidate (b3): option rows L2-normalized to a unit direction",
    "p14_stateless_option": "P14 v2 head + e_k = W_o o_k (option features exactly state-independent)",
    "p15_option_unnormalized": "P15 v2 head, option_normalize=False (the PRE-FIX formula, shipped flag off)",
}

# The three interaction modes the shipped ChoiceHead offers.
INTERACTION_OF = {
    "p3_jevx_dot": "dot",
    "p3b_jevx_bilinear": "bilinear",
    "p3c_jevx_mlp": "mlp",
}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    p.add_argument("--datasets", nargs="+",
                   default=["synthetic", "sst2", "ag_news", "yelp"],
                   help="subset of " + str(sorted(DATASETS)) + ", or all")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--cells", nargs="+", default=list(ALL_CELLS),
                   help="subset of " + str(list(ALL_CELLS)))
    p.add_argument("--smoke", action="store_true",
                   help="tiny sizes for pipeline validation (never reportable)")
    p.add_argument("--output-dir", default=str(RESULTS_DIR))
    p.add_argument("--recompute-features", action="store_true",
                   help="ignore the feature cache and re-featurize")
    return p.parse_args(argv)

# --------------------------------------------------------------------------
# Feature production: frozen backbone, featurize ONCE per split, cached
# --------------------------------------------------------------------------

def _cache_path(name: str) -> Path:
    return CACHE_DIR / (name + "_features.npz")


def _save_cache(name: str, splits: Dict[str, np.ndarray],
                labels: Dict[str, List[int]], meta: Dict[str, Any]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {}
    for split, arr in splits.items():
        payload["x_" + split] = np.ascontiguousarray(arr, dtype=np.float32)
        payload["y_" + split] = np.asarray(labels[split], dtype=np.int64)
    payload["meta_json"] = np.frombuffer(
        json.dumps(meta).encode("utf-8"), dtype=np.uint8
    )
    path = _cache_path(name)
    # np.savez appends ".npz" unless the name already ends with it, so write to
    # a ".tmp" FILE HANDLE and rename that - the same crash-safe pattern
    # bench_common.write_json uses.
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, **payload)
    tmp.replace(path)
    print("[cache] wrote " + str(path), flush=True)


def _load_cache(name: str):
    path = _cache_path(name)
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=False)
    splits = {k[2:]: z[k] for k in z.files if k.startswith("x_")}
    labels = {k[2:]: z[k].tolist() for k in z.files if k.startswith("y_")}
    meta = json.loads(bytes(z["meta_json"]).decode("utf-8"))
    return splits, labels, meta


def synthetic_task(cfg: Dict[str, Any]):
    """A HARD but well-conditioned control: D = 128, K = 4 Gaussian mixture
    with Kingman-style equiangular class means scaled to pairwise cosine 0.55.
    A linear probe is near-optimal but the task is NOT trivially separable, and
    there is no backbone and no text at all - so any head-vs-probe gap here is
    purely the scoring function. Feature dim 128 matches the tiny BERT
    backbone width so the SAME JevX stack can consume it unchanged."""
    d, k = 128, 4
    rng = np.random.RandomState(1234)
    a = rng.randn(k, d)
    a -= a.mean(axis=0, keepdims=True)
    q, _ = np.linalg.qr(a.T)
    # Signal scale: means of norm sqrt(d)*SNR_MARGIN vs unit-variance noise.
    # SNR_MARGIN = 0.28 puts a linear probe around 60-75% - hard for a head to
    # reach by accident, well above the 25% chance floor but far from saturation.
    means = q.T * (np.sqrt(d) * 0.28)
    sizes = {"train": cfg["train_size"], "cal": cfg["cal_size"],
             "test": cfg["test_size"]}
    xs: Dict[str, np.ndarray] = {}
    ys: Dict[str, List[int]] = {}
    for i, (split, m) in enumerate(sizes.items()):
        r = np.random.RandomState(9000 + i)
        y = r.randint(0, k, size=m)
        x = means[y] + r.randn(m, d)
        x = (x - x.mean(0, keepdims=True)) / (x.std(0, keepdims=True) + 1e-6)
        xs[split] = x.astype(np.float32)
        ys[split] = [int(v) for v in y]
    meta = {
        "kind": "synthetic_gaussian_mixture",
        "dim": d,
        "classes": k,
        "snr_margin": 0.28,
        "note": ("Kingman-style equiangular Gaussian mixture, unit-variance "
                 "features, D=128=K-free control matching the tiny BERT "
                 "backbone width. No backbone / no text: any head-vs-probe "
                 "gap is purely the scoring function."),
    }
    return xs, ys, meta


def text_task(name: str, cfg: Dict[str, Any]):
    """ag_news / sst2 / yelp under the EXACT protocol of the source benchmark."""
    if name == "ag_news":
        from datasets import load_dataset

        ds = load_dataset("fancyzhx/ag_news",
                          revision="eb185aade064a813bc0b7f42de02595523103ca4")
        pool = list(ds["train"]["text"])
        pool_y = [int(v) for v in ds["train"]["label"]]
        test = list(ds["test"]["text"])[: cfg["test_size"]]
        test_y = [int(v) for v in ds["test"]["label"]][: cfg["test_size"]]
        ds_meta = {"dataset_id": "fancyzhx/ag_news",
                   "dataset_revision": "eb185aade064a813bc0b7f42de02595523103ca4"}
    elif name == "sst2":
        tr, tr_y, te, te_y = bc.load_sst2()
        pool, pool_y = list(tr), [int(v) for v in tr_y]
        test = te[: cfg["test_size"]]
        test_y = [int(v) for v in te_y][: cfg["test_size"]]
        ds_meta = {"dataset_id": bc.DEFAULT_DATASET,
                   "dataset_revision": bc.DEFAULT_DATASET_REVISION}
    elif name == "yelp":
        from datasets import load_dataset

        ds = load_dataset("Yelp/yelp_review_full",
                          revision="c1f9ee939b7d05667af864ee1cb066393154bf85")
        pool = list(ds["train"]["text"])
        pool_y = [int(v) for v in ds["train"]["label"]]
        test = list(ds["test"]["text"])[: cfg["test_size"]]
        test_y = [int(v) for v in ds["test"]["label"]][: cfg["test_size"]]
        ds_meta = {"dataset_id": "Yelp/yelp_review_full",
                   "dataset_revision": "c1f9ee939b7d05667af864ee1cb066393154bf85"}
    else:  # pragma: no cover - guarded by the caller
        raise ValueError("unknown text dataset " + repr(name))

    train_idx, cal_idx = bc.make_splits(
        pool_y, cfg["train_size"], cfg["cal_size"], seed=0
    )
    texts = {"train": [pool[i] for i in train_idx],
             "cal": [pool[i] for i in cal_idx],
             "test": test}
    labels = {"train": [pool_y[i] for i in train_idx],
              "cal": [pool_y[i] for i in cal_idx],
              "test": test_y}
    model = bc.JevX.from_pretrained(
        cfg["backbone"], freeze_base_model=True,
        pooling_method=cfg["pooling"], fusion_type=cfg["fusion"],
    )
    model.eval()
    feats: Dict[str, np.ndarray] = {}
    for split in ("train", "cal", "test"):
        feats[split] = bc.featurize(
            model, texts[split], batch_size=cfg["batch_size"],
            max_length=cfg["max_length"], strategies=[cfg["pooling"]],
        )[cfg["pooling"]]
    meta = {
        "kind": "text",
        **ds_meta,
        "backbone_id": cfg["backbone"],
        "pooling": cfg["pooling"],
        "fusion": cfg["fusion"],
        "max_length": cfg["max_length"],
        "feature_dim": int(feats["train"].shape[1]),
        "source_protocol": cfg["source"],
        "split_seed": 0,
    }
    del model
    return feats, labels, meta


def audio_task(cfg: Dict[str, Any]):
    """RAVDESS under the EXACT protocol of benchmarks/run_modality.py."""
    from benchmarks import run_modality as rm

    paths, labels, data_prov = rm.load_ravdess()
    train_idx, cal_idx = bc.make_splits(
        labels, cfg["train_size"], cfg["cal_size"], seed=0
    )
    keep = np.ones(len(labels), dtype=bool)
    keep[train_idx] = False
    keep[cal_idx] = False
    test_idx = np.flatnonzero(keep)
    if len(test_idx) > cfg["test_size"]:
        rem = np.asarray(labels)[test_idx]
        local = bc.stratified_indices(rem, cfg["test_size"], seed=2)
        test_idx = test_idx[local]
    split_paths = {"train": [paths[i] for i in train_idx],
                   "cal": [paths[i] for i in cal_idx],
                   "test": [paths[i] for i in test_idx]}
    split_labels = {"train": [labels[i] for i in train_idx],
                    "cal": [labels[i] for i in cal_idx],
                    "test": [labels[i] for i in test_idx]}
    model = bc.JevX.from_pretrained(cfg["backbone"], freeze_base_model=True,
                                    pooling_method=None)
    model.eval()

    def make_state(chunk: Sequence[Any]) -> Dict[str, Any]:
        return {"audio": [rm.load_waveform(Path(q)) for q in chunk]}

    feats: Dict[str, np.ndarray] = {}
    for split in ("train", "cal", "test"):
        print("[ravdess] featurize " + split + " (n=" +
              str(len(split_paths[split])) + ")", flush=True)
        feats[split] = rm.featurize_payloads(
            model, split_paths[split], make_state, batch_size=8
        )
    meta = {
        "kind": "audio",
        "dataset": data_prov,
        "backbone_id": cfg["backbone"],
        "pooling": "native (wav2vec2 has no pooler; adapter masked_mean)",
        "fusion": cfg["fusion"],
        "audio_max_seconds": rm.AUDIO_MAX_SECONDS,
        "feature_dim": int(feats["train"].shape[1]),
        "source_protocol": cfg["source"],
        "split_seed": 0,
    }
    del model
    return feats, split_labels, meta


def build_features(name: str, cfg: Dict[str, Any], *, recompute: bool):
    if not recompute:
        cached = _load_cache(name)
        if cached is not None:
            print("[cache] hit " + name, flush=True)
            return cached[0], cached[1], cached[2], "cache"
    if cfg["modality"] == "synthetic":
        feats, labels, meta = synthetic_task(cfg)
    elif cfg["modality"] == "text":
        feats, labels, meta = text_task(name, cfg)
    else:
        feats, labels, meta = audio_task(cfg)
    _save_cache(name, feats, labels, meta)
    return feats, labels, meta, "computed"

# --------------------------------------------------------------------------
# The scoring functions under test
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# EXP-H follow-up (2026-09-25): OPTION/SCHEMA-pathway candidates
# --------------------------------------------------------------------------
#
# EXP-H measured that on AUDIO the damage comes from the option/schema pathway
# (P4 bypass = +10.5 pt over the shipped v2 head on RAVDESS), NOT from the head.
# So every candidate here patches ONLY the fusion option formula: the question
# encoder, the schema encoder, the fusion CONTEXT c and the head's scoring
# function are all left exactly as shipped, and a cell that patches nothing is
# bit-identical to p6_jevx_v2.
#
#   "none"         - no patch. Identical to p6_jevx_v2 (the control cell).
#   "neutral_init" - candidate (a): the option projection (W_o, b_o) is
#                    ZERO-initialized, so e_k = (1 - g) (*) c is IDENTICAL for
#                    every k at step 0: the option pathway starts NEUTRAL and
#                    only acquires what reduces the loss. Same zero-init
#                    residual convention as LinearProbePathway.
#   "ln_branch"    - candidate (b1): a state-independent LayerNorm on the
#                    PROJECTED option rows W_o o_k, so the option branch
#                    carries unit scale whatever norm o_k has.
#   "ln_row"       - candidate (b2): a LayerNorm on the FUSED option row e_k,
#                    so no option row can carry a row-specific scale that
#                    swamps the fused state s.
#   "dropout"      - candidate (c): nn.Dropout(p) on the projected option rows.
#                    Training only by construction (nn.Dropout is identity in
#                    eval) - the head is prevented from depending on the
#                    option pathway at every step.
#   "neutral_ln"   - (a) + (b2), measured only if both win alone.

OPTION_VARIANT_OF: Dict[str, str] = {
    "p7_option_neutral": "neutral_init",
    "p8_option_ln_branch": "ln_branch",
    "p9_option_ln_row": "ln_row",
    "p10_option_dropout": "dropout",
    "p11_option_neutral_ln": "neutral_ln",
    "p12_neutral_residual": "neutral_residual",
    "p14_stateless_option": "stateless_option",
    "p15_option_unnormalized": "unnormalized",
}

#: P13 is the WINNER of the EXP-H follow-up and is now the SHIPPED option
#: formula (``Fusion.option_normalize=True``, jevx/representations/fusion.py),
#: so its cell runs the LIBRARY path with no patch and asserts the landed flag:
#: the cell IS the shipped configuration, not a monkeypatch. The pre-fix arm of
#: the same ablation is "unnormalized" (option_normalize=False).
VARIANT_LANDED: Dict[str, str] = {
    "p13_unit_option": "none",
}
#: Dropout rate of the (c) candidate (training only).
OPTION_DROPOUT_P: float = 0.5


def _option_features_for_variant(
    variant: str, norm: Optional[nn.Module], drop: Optional[nn.Module]
):
    """Replacement for ``Fusion._option_features`` carrying one candidate.

    Identical to the shipped formula
    ``e_k = g (*) (W_o o_k) + (1 - g) (*) c`` except for the one candidate
    term, and with the masked-row ``masked_fill`` still LAST so an invalid
    option row is exactly 0 whatever the candidate does.
    """

    def _option_features(self, context, options, option_mask):
        gate = torch.sigmoid(self.option_gate(context)).unsqueeze(1)  # [B, 1, D]
        if variant == "unit_option":
            # (b3) The option branch carries option IDENTITY, not magnitude:
            # o_k is scaled to a unit DIRECTION first, so the branch
            # magnitude is fixed by W_o init and cannot be set by the
            # (learnable) schema rows. State-independent and scale-free.
            options = torch.nn.functional.normalize(options, dim=-1, eps=1e-12)
        projected = self.option_proj(options)                          # [B, K, D]
        if drop is not None:
            projected = drop(projected)
        if norm is not None and variant in ("ln_branch", "neutral_ln"):
            projected = norm(projected)
        if variant == "stateless_option":
            # The option row carries NO state: no gate, no context fallback.
            # The head reads the state through the fused state only.
            features = projected
        else:
            features = gate * projected + (1.0 - gate) * context.unsqueeze(1)
        if norm is not None and variant == "ln_row":
            features = norm(features)
        if option_mask is not None:
            # masked_fill LAST: an invalid option row is exactly 0 even when
            # the candidate's LayerNorm/dropout saw a padded input.
            features = features.masked_fill(~option_mask.unsqueeze(-1), 0.0)
        return features

    return _option_features


#: The three kinds of learned interaction have very different parameter
#: counts, so the (a)-at-the-residual candidate is expressed as ONE learnable
#: gain α initialized to zero: r_k = α f(s, e_k).
def _neutral_residual_head(head: Any) -> Any:
    """Give αf(·) to a ChoiceHead with α = 0 at step 0.

    Zero-initializing the OPTION FEATURES instead (candidate P7) is
    degenerate and measured so: with W_o = 0 every e_k collapses to
    (1 - g) (*) c, the duplicate-option tie in ChoiceHead then averages all
    K columns, and the gradient to EVERY parameter is exactly zero - the
    model can never leave the uniform distribution. Neutrality is therefore
    only meaningful at the residual, where the option features stay
    distinct so the tie never fires.
    """
    if getattr(head, "neutral_residual", None) is not None:
        return head
    head.neutral_residual = nn.Parameter(torch.zeros(()))
    original = head.interaction_logits

    def interaction_logits(self, fused, option_features):
        return self.neutral_residual * original(fused, option_features)

    head.interaction_logits = types.MethodType(interaction_logits, head)
    return head

def _iter_fusions(module: nn.Module) -> List[Any]:
    """Every Fusion instance in the module tree (duck-typed, no import cycle)."""
    return [
        sub
        for sub in module.modules()
        if hasattr(sub, "option_proj") and hasattr(sub, "_option_features")
    ]


def hq_fusions(module: nn.Module) -> List[Any]:
    """Every Fusion in ``module`` (public alias; used by the landed cell)."""
    return _iter_fusions(module)

def apply_option_variant(module: nn.Module, variant: str) -> int:
    """Patch the option formula of every Fusion in ``module``; return the count.

    ``"none"`` patches nothing and returns 0, so the control cell runs the
    shipped code path untouched.
    """
    if variant == "none":
        return 0
    patched = 0
    if variant == "unnormalized":
        # The pre-EXP-H option formula, reproduced from the shipped flag
        # rather than by a formula re-implementation.
        for fusion in _iter_fusions(module):
            fusion.option_normalize = False
            patched += 1
        return patched
    if variant == "neutral_residual":
        for sub in module.modules():
            if hasattr(sub, "interaction_logits") and hasattr(sub, "state_pathway"):
                _neutral_residual_head(sub)
                return 1
        raise ValueError(
            "option variant 'neutral_residual': no ChoiceHead found to patch."
        )
    for fusion in _iter_fusions(module):
        d = int(fusion.hidden_size)
        norm = nn.LayerNorm(d) if variant in ("ln_branch", "ln_row", "neutral_ln") else None
        drop = nn.Dropout(OPTION_DROPOUT_P) if variant == "dropout" else None
        if variant in ("neutral_init", "neutral_ln"):
            # Candidate (a): the option pathway is NEUTRAL at step 0.
            with torch.no_grad():
                fusion.option_proj.weight.zero_()
                fusion.option_proj.bias.zero_()
        # Registered as submodules so their parameters/buffers are visible to
        # the trainer's optimizer and move with .to(device)/.train()/.eval().
        fusion._option_features = types.MethodType(  # type: ignore[method-assign]
            _option_features_for_variant(variant, norm, drop), fusion
        )
        patched += 1
    return patched


def _state_tensor(batch: Dict[str, Any]) -> torch.Tensor:
    """Pull the pooled feature matrix out of a collated batch as [B, D]."""
    state = batch["state"]
    if isinstance(state, np.ndarray):
        x = torch.from_numpy(state)
    else:
        x = torch.as_tensor(np.stack([np.asarray(r) for r in state]))
    return x.to(dtype=torch.float32)


class JevXStack(nn.Module):
    """The shipped FeaturizedJevX path (steps 2-8) with a swappable choice
    head, so P3/P5/P6 differ ONLY in the head wiring, never in the noise."""

    def __init__(self, model: Any, question: str, options: Sequence[str],
                 *, interaction: Optional[str] = None,
                 state_only: bool = False,
                 state_pathway: bool = True,
                 probe_init: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> None:
        super().__init__()
        self.jevx = model
        self.question = question
        self.options = list(options)
        self.state_only = state_only
        d = int(model.config.decision_hidden_size)
        if state_only:
            # P4: a FIXED, state-independent option basis. Rows are unit-norm
            # so the option-interaction PARAMETERIZATION - not the option
            # scale - is what is under test.
            gen = torch.Generator().manual_seed(0)
            basis = torch.randn(len(self.options), d, generator=gen)
            self.register_buffer("option_basis", basis / float(math.sqrt(d)))
        else:
            self.inner = bc.FeaturizedJevX(model, question, options)
        if interaction is not None:
            from jevx.decisions import build_decision_head

            # state_pathway=False restores the HEAD v1 scoring function (the
            # interaction residual alone), so P3/P3b/P3c measure the shipped
            # v1 head and P6 measures v2 under one identical protocol.
            self.jevx.heads["choice"] = build_decision_head(
                "choice", hidden_size=d, interaction=interaction,
                state_pathway=state_pathway,
            )
        if probe_init is not None:
            self._init_state_pathway(*probe_init)

    def _init_state_pathway(self, w: np.ndarray, b: np.ndarray) -> None:
        """P5: write a closed-form least-squares probe of the FROZEN features
        into the head state pathway (state_proj). The first K output
        coordinates carry the probe scores [D_feat -> K]; the remaining
        coordinates stay zero, so the map keeps the probe signal. Everything
        downstream (fusion, head) is untouched - P5 therefore differs from
        P3b ONLY in the state pathway initialization."""
        w_t = torch.as_tensor(np.asarray(w), dtype=torch.float32)  # [D_feat, K]
        b_t = torch.as_tensor(np.asarray(b), dtype=torch.float32)  # [K]
        # state_proj.proj.weight is [D_decision, D_feat]: the probe's K output
        # coordinates go into the first K rows, so transpose the probe weight.
        wt_t = w_t.t().contiguous()  # [K, D_feat]
        k = int(wt_t.shape[0])
        sp = self.jevx.state_proj
        with torch.no_grad():
            r = min(k, int(sp.proj.weight.shape[0]))
            c = min(int(wt_t.shape[1]), int(sp.proj.weight.shape[1]))
            sp.proj.weight.zero_()
            sp.proj.weight[:r, :c] = wt_t[:r, :c]
            sp.proj.bias.zero_()
            sp.proj.bias[:r] = b_t[:r]
            # A zeroed LayerNorm gain would erase the probe signal.
            sp.norm.weight.fill_(1.0)
            sp.norm.bias.zero_()

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        if not self.state_only:
            return self.inner(batch)
        m = self.jevx
        x = _state_tensor(batch)
        b = x.shape[0]
        state_repr = m.state_proj(x)  # [B, D] - the question is BYPASSED
        opts = self.option_basis.unsqueeze(0).expand(b, -1, -1)
        out = m.heads["choice"](state_repr, opts)
        return out.logits


def ridge_probe(x: np.ndarray, y: np.ndarray, num_classes: int,
                lam: float = 1e-2) -> Tuple[np.ndarray, np.ndarray]:
    """Closed-form least-squares linear probe [D -> K] on frozen features
    (ridge-regularized normal equations). Returns (W [D, K], b [K])."""
    d = x.shape[1]
    onehot = np.eye(num_classes, dtype=np.float64)[np.asarray(y)]
    xb = np.concatenate([x.astype(np.float64), np.ones((len(x), 1))], axis=1)
    a = xb.T @ xb + lam * np.eye(d + 1)
    w = np.linalg.solve(a, xb.T @ onehot)
    return w[:d], w[d]


def _new_jevx(cfg: Dict[str, Any], in_dim: int | None = None):
    """A fresh JevX per cell/seed, built from the frozen backbone.

    The head stack consumes a POOLED feature of width d_model, so when the
    cached feature width differs from the backbone's d_model (audio: wav2vec2
    emits 768, vision: deit 192) the decision/question/schema widths are pinned
    to the feature width so state_proj accepts the features unchanged.
    """
    kwargs: Dict[str, Any] = {}
    if in_dim is not None and in_dim != 128:
        kwargs.update(
            decision_hidden_size=in_dim,
            question_hidden_size=in_dim,
            schema_hidden_size=in_dim,
        )
    # The synthetic control carries no backbone of its own; its features
    # are D = 128, the width of the tiny BERT stack every other cell uses, so
    # it falls back to that backbone rather than asking from_pretrained for a
    # checkpoint named "None" (which raises).
    backbone = cfg["backbone"] or TEXT_BACKBONE
    model = bc.JevX.from_pretrained(
        backbone, freeze_base_model=True,
        pooling_method=cfg["pooling"], fusion_type=cfg["fusion"], **kwargs,
    )
    model.eval()
    return model


def build_cell(cell: str, *, cfg: Dict[str, Any], feats: Dict[str, np.ndarray],
               labels: Dict[str, List[int]], num_classes: int,
               question: str, options: Sequence[str], seed: int,
               base_model: Any = None) -> nn.Module:
    """Instantiate ONE cell for ONE seed. Every cell draws torch.manual_seed
    first, so the init noise stream is identical across cells."""
    torch.manual_seed(seed)
    in_dim = int(feats["train"].shape[1])
    if cell == "p1_linear":
        return bc.LinearProbe(in_dim, num_classes)
    if cell == "p2_mlp":
        torch.manual_seed(seed + 10_000)  # mirrors the B2 seeding of run_exp7
        return bc.MLPProbe(in_dim, num_classes)
    if cell in INTERACTION_OF:
        return JevXStack(copy.deepcopy(base_model), question, options,
                         interaction=INTERACTION_OF[cell], state_pathway=False)
    if cell == "p4_state_only":
        return JevXStack(copy.deepcopy(base_model), question, options, state_only=True)
    if cell == "p5_probe_init":
        w, b = ridge_probe(feats["train"], labels["train"], num_classes)
        return JevXStack(copy.deepcopy(base_model), question, options,
                         probe_init=(w, b))
    if cell == "p6_jevx_v2":
        return JevXStack(copy.deepcopy(base_model), question, options,
                         interaction="bilinear", state_pathway=True)
    if cell in VARIANT_LANDED:
        # The shipped option formula: no patch, and the landed flag is
        # asserted so a silent regression cannot hide in this cell.
        stack = JevXStack(copy.deepcopy(base_model), question, options,
                          interaction="bilinear", state_pathway=True)
        fusions = hq_fusions(stack)
        if not fusions or not all(f.option_normalize for f in fusions):
            raise ValueError(
                f"cell {cell!r} expects the LANDED option formula "
                "(option_normalize=True on every fusion); it is off."
            )
        return stack
    if cell in OPTION_VARIANT_OF:
        # Identical to p6_jevx_v2 except for the option-pathway patch, so the
        # candidate table is a ONE-VARIABLE comparison.
        stack = JevXStack(copy.deepcopy(base_model), question, options,
                          interaction="bilinear", state_pathway=True)
        if apply_option_variant(stack, OPTION_VARIANT_OF[cell]) == 0:
            raise ValueError(
                f"cell {cell!r}: no fusion module found to patch - the option "
                "pathway variant could not be applied."
            )
        return stack
    raise ValueError("unknown cell " + repr(cell))


# --------------------------------------------------------------------------
# Training / evaluation driver (ONE shared protocol)
# --------------------------------------------------------------------------

def run_cell(cell: str, *, cfg: Dict[str, Any], feats: Dict[str, np.ndarray],
             labels: Dict[str, List[int]], num_classes: int, question: str,
             options: Sequence[str], seed: int) -> Dict[str, Any]:
    """Train + score ONE cell for ONE seed under the shared protocol."""
    train_records = bc.build_records(feats["train"], labels["train"], question, options)
    test_records = bc.build_records(feats["test"], labels["test"], question, options)
    # Build (or reuse) the base JevX stack. Constructing it costs a full
    # backbone load, so it is created ONCE per seed and deep-copied per cell:
    # every cell then starts from byte-identical parameters, and the ONLY
    # difference between cells is the scoring function under test.
    torch.manual_seed(seed)
    base = _new_jevx(cfg, in_dim=int(feats["train"].shape[1]))
    module = build_cell(cell, cfg=cfg, feats=feats, labels=labels,
                        num_classes=num_classes, question=question,
                        options=options, seed=seed, base_model=base)
    _, history = bc.train_heads(
        module, train_records, seed=seed, lr=cfg["lr"],
        epochs=cfg["epochs"], batch_size=cfg["batch_size"],
    )
    test_logits, test_labels = bc.collect_logits(
        module, test_records, batch_size=cfg["batch_size"]
    )
    m = bc.metrics_from_logits(test_logits, test_labels)
    row = {
        "seed": seed,
        "accuracy": m["accuracy"],
        "macro_f1": m["macro_f1"],
        "nll": m["nll"],
        "ece": m["ece"],
        "brier": m["brier"],
        "train_loss_first": history[0]["loss"],
        "train_loss_last": history[-1]["loss"],
        "trainable_params": bc.count_trainable(module),
    }
    del module, base
    return row

# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def run_dataset(name: str, args: argparse.Namespace) -> Dict[str, Any]:
    cfg = dict(DATASETS[name])
    if args.smoke:
        cfg["train_size"] = min(cfg["train_size"], 48)
        cfg["cal_size"] = min(cfg["cal_size"], 24)
        cfg["test_size"] = min(cfg["test_size"], 48)
        cfg["epochs"] = 1
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    t0 = time.perf_counter()
    print("", flush=True)
    print("[=== " + name + " ===] protocol from " + cfg["source"], flush=True)

    if cfg["modality"] == "synthetic":
        feats, labels, meta = synthetic_task(cfg)
        source = "computed"
    else:
        feats, labels, meta, source = build_features(
            name, cfg, recompute=args.recompute_features
        )
    num_classes = len(cfg["options"])

    rows_by_cell: Dict[str, List[Dict[str, Any]]] = {
        c: [] for c in list(args.cells)
    }
    failures: Dict[str, str] = {}
    t_cells = time.perf_counter()
    for cell in list(args.cells):
        rows_by_cell[cell] = []
    for seed in seeds:
        print("  -- seed " + str(seed), flush=True)
        for cell in list(args.cells):
            ct = time.perf_counter()
            try:
                rows_by_cell[cell].append(run_cell(
                    cell, feats=feats, labels=labels, cfg=cfg,
                    num_classes=num_classes, question=cfg["question"],
                    options=cfg["options"], seed=seed,
                ))
            except Exception as exc:  # noqa: BLE001 - recorded, never silent
                import traceback

                failures[cell] = type(exc).__name__ + ": " + str(exc)
                print("[FAILED] " + name + "/" + cell + "/" + str(seed) +
                      ": " + failures[cell], flush=True)
                traceback.print_exc()
                continue
            row = rows_by_cell[cell][-1]
            print("     " + cell.ljust(22) + " acc=" +
                  ("%.4f" % row["accuracy"]) +
                  "  f1=" + ("%.4f" % row["macro_f1"]) +
                  "  nll=" + ("%.4f" % row["nll"]) +
                  "  (" + str(round(time.perf_counter() - ct, 1)) + "s)", flush=True)
    results: Dict[str, Any] = {}
    for cell in list(args.cells):
        rows = rows_by_cell[cell]
        if not rows:
            continue
        results[cell] = {
            "label": CELL_LABELS[cell],
            "per_seed": rows,
            "aggregate": bc.mean_std_over_seeds(rows),
            "wall_s": round(time.perf_counter() - t_cells, 2),
        }
        agg = results[cell]["aggregate"]
        print("  " + cell.ljust(22) + " acc=" +
              ("%.4f+-%.4f" % (agg["accuracy"]["mean"], agg["accuracy"]["std"])) +
              "  f1=" + ("%.4f" % agg["macro_f1"]["mean"]) +
              "  nll=" + ("%.4f" % agg["nll"]["mean"]) +
              "  (" + str(results[cell]["wall_s"]) + "s)", flush=True)

    ranking = sorted(
        ((c, results[c]["aggregate"]["accuracy"]["mean"]) for c in results),
        key=lambda kv: -kv[1],
    )
    payload = {
        "experiment": "exp_h_head_quality",
        "dataset": name,
        "smoke": bool(args.smoke),
        "head_version_under_test": (
            "v2 (additive state-only linear term + option-interaction residual)"
            if "p6_jevx_v2" in results else "v1 (as shipped)"
        ),
        "protocol": {
            "seeds": seeds,
            "epochs": cfg["epochs"],
            "lr": cfg["lr"],
            "batch_size": cfg["batch_size"],
            "train_n": int(feats["train"].shape[0]),
            "test_n": int(feats["test"].shape[0]),
            "feature_dim": int(feats["train"].shape[1]),
            "num_classes": num_classes,
            "question": cfg["question"],
            "options": list(cfg["options"]),
            "optimizer": "torch.optim.AdamW (HeadTrainer default), grad_clip=1.0",
            "loss": "jevx.training.choice_loss (masked cross-entropy)",
            "note": ("IDENTICAL protocol for every cell: same frozen feature "
                     "matrix, same splits, same seeds, same optimizer / lr / "
                     "epochs / batch, same choice_loss. The frozen backbone is "
                     "not in the loop (featurize-once)."),
        },
        "features": dict(meta, source=source),
        "cells": results,
        "ranking_by_accuracy": [{"cell": c, "accuracy": a} for c, a in ranking],
        "failures": failures,
        "environment": bc.environment_info(),
        "wall_s": round(time.perf_counter() - t0, 2),
    }
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    names = sorted(DATASETS) if args.datasets == ["all"] else list(args.datasets)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        if name not in DATASETS:
            print("[skip] unknown dataset " + repr(name), flush=True)
            continue
        payload = run_dataset(name, args)
        suffix = "_smoke" if args.smoke else ""
        path = out_dir / ("head_quality_" + name + suffix + ".json")
        bc.write_json(path, payload)
        print("[wrote] " + str(path) + " (" + str(payload["wall_s"]) +
              "s, failures=" + str(len(payload["failures"])) + ")", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
