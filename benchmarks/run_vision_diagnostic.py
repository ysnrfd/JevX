# Copyright (c) 2026 ysnrfd
"""WAVE-3b vision diagnostic: the V0-V6 grid on frozen DeiT-tiny x Imagenette.

Why this file exists
--------------------
reports/MODALITY_BENCHMARKS.md 4.1/4.3 records a measured gap: on the same
cached 192-d CLS features and the same 800/300/500 splits the JevX stack
scored 0.9087 +/- 0.0257 accuracy while the plain MLP probe scored
0.9673 +/- 0.0042, with NON-OVERLAPPING seed ranges. That report says WHAT the
gap is; this file isolates WHY, with a grid that changes exactly one thing per
cell so the paired deltas are attributable.

Protocol (mirrors MODALITY_BENCHMARKS.md 3/4 verbatim - no silent changes)
------------------------------------------------------------------------
* featurize ONCE through the frozen ``facebook/deit-tiny-patch16-224``
  backbone, ``pooling_method="cls"`` (the checkpoint ships no pooler), split
  seed 0, stratified train 800 / calibration 300 / test 500 (test = stratified
  subsample of the OFFICIAL validation split, never fitted on);
* heads fit ONLY on train, temperature fitted ONLY on calibration, the test
  split only ever scored;
* optimizer / lr / epochs / batch / loss are the HeadTrainer + choice_loss
  defaults already published for this task (AdamW lr 1e-3, 4 epochs, head batch
  64), held FIXED across every cell - V4 is the one deliberate exception
  (20 epochs) and is labelled as such;
* 5 training seeds (0-4) per cell; the split is IDENTICAL for every seed, so
  all cross-cell deltas are PAIRED;
* the reference probes (B1 linear, B2 MLP) are re-run here on the identical
  cached features / splits / 5 seeds so the gap can be quoted paired instead of
  against a 3-seed number from a different run.

Nothing under ``jevx/`` is modified: the V1-V6 variants are assembled from the
shipped public surface (``JevXConfig`` fields, ``Fusion.forward``, the choice
head's ``conditioning_enabled`` / ``duplicate_tie`` per-call switches) by the
wrappers defined in this file.

Usage
-----
    python benchmarks/run_vision_diagnostic.py --phase featurize
    python benchmarks/run_vision_diagnostic.py --phase grid
    python benchmarks/run_vision_diagnostic.py --phase all

Result files (written as each phase completes):
    benchmarks/cache/vision_diagnostic/deit_tiny_cls_features.npz
    benchmarks/cache/vision_diagnostic/featurize_provenance.json
    benchmarks/results/vision_diagnostic_features.json
    benchmarks/results/vision_diagnostic_grid.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import bench_common as bc  # noqa: E402
from benchmarks import run_modality as rm  # noqa: E402
from jevx.questions import Option  # noqa: E402
from jevx.training import make_supervised_step_fn  # noqa: E402

# --------------------------------------------------------------------------
# Ids / protocol constants - copied from benchmarks/run_modality.py, which is
# the published implementation of MODALITY_BENCHMARKS.md 3/4. Re-declared here
# (not imported) so this file records the pinned ids in its own provenance.
# --------------------------------------------------------------------------

VISION_BACKBONE = rm.VISION_BACKBONE
VISION_BACKBONE_SHA = rm.VISION_BACKBONE_SHA
VISION_DATASET_ID = rm.VISION_DATASET_ID
VISION_DATASET_SHA = rm.VISION_DATASET_SHA
IMAGENETTE_CONFIG = rm.IMAGENETTE_CONFIG
VISION_QUESTION = rm.VISION_QUESTION
CLASSES: List[str] = list(rm.IMAGENETTE_CLASSES)
NUM_CLASSES = len(CLASSES)

FEATURE_DIR = REPO_ROOT / "benchmarks" / "cache" / "vision_diagnostic"
FEATURE_NPZ = FEATURE_DIR / "deit_tiny_cls_features.npz"
FEATURE_PROV = FEATURE_DIR / "featurize_provenance.json"
RESULT_DIR = REPO_ROOT / "benchmarks" / "results"
FEATURE_RESULT = RESULT_DIR / "vision_diagnostic_features.json"
GRID_RESULT = RESULT_DIR / "vision_diagnostic_grid.json"

#: Featurize batch 32 - the published EXP-V value (MODALITY_BENCHMARKS.md 1).
FEAT_BATCH = 32
#: Head batch 64 / lr 1e-3 / 4 epochs - the published EXP-V values.
HEAD_BATCH = 64
LR = 1e-3
EPOCHS = 4
SPLIT_SEED = 0
N_TRAIN, N_CAL, N_TEST = 800, 300, 500
SEEDS = [0, 1, 2, 3, 4]

#: Published reference numbers (MODALITY_BENCHMARKS.md 4.1, 3 seeds 0-2).
PUBLISHED = {
    "jevx_accuracy_mean": 0.9087,
    "jevx_accuracy_std": 0.0257,
    "jevx_ece_temperature": 0.0207,
    "b1_linear_accuracy_mean": 0.9467,
    "b1_linear_accuracy_std": 0.0046,
    "b2_mlp_accuracy_mean": 0.9673,
    "b2_mlp_accuracy_std": 0.0042,
    "seeds": [0, 1, 2],
    "source": "reports/MODALITY_BENCHMARKS.md 4.1/4.2 (run 2026-09-25, 3 seeds)",
}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--phase", default="all",
                   choices=["featurize", "grid", "summarize", "all"])
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--head-batch-size", type=int, default=HEAD_BATCH)
    p.add_argument("--feat-batch", type=int, default=FEAT_BATCH)
    p.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    p.add_argument("--cells", default="all",
                   help="comma-separated subset of cell ids (default: all)")
    p.add_argument("--output-dir", default=str(RESULT_DIR))
    p.add_argument("--force-refeaturize", action="store_true",
                   help="ignore the cached npz and re-run the backbone")
    p.add_argument("--merge", action="store_true",
                   help="grid phase: carry the cells already in the result JSON "
                        "forward instead of replacing them (use with --cells)")
    return p.parse_args(argv)


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def environment_info() -> Dict[str, Any]:
    info = bc.environment_info()
    info["script"] = "benchmarks/run_vision_diagnostic.py"
    info["platform_detail"] = platform.platform()
    return info


def class_balance(labels: Sequence[int], num_classes: int = NUM_CLASSES) -> Dict[str, Any]:
    """Exact imbalance quantification of one label vector."""
    arr = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(arr, minlength=num_classes).astype(np.int64)
    n = int(counts.sum())
    p = counts / n
    nonzero = p[p > 0]
    entropy = float(-(nonzero * np.log(nonzero)).sum())
    max_entropy = math.log(num_classes)
    chi2 = float((((counts - n / num_classes) ** 2) / (n / num_classes)).sum())
    return {
        "n": n,
        "counts": {str(i): int(c) for i, c in enumerate(counts)},
        "min": int(counts.min()),
        "max": int(counts.max()),
        "max_over_min": float(counts.max() / counts.min()) if counts.min() else None,
        "max_dev_from_uniform_frac": float(np.abs(counts - n / num_classes).max() / (n / num_classes)),
        "entropy_nats": entropy,
        "max_entropy_nats": max_entropy,
        "entropy_deficit_nats": float(max_entropy - entropy),
        "normalized_entropy_deficit": float((max_entropy - entropy) / max_entropy),
        "chi2_vs_uniform": chi2,
        "chi2_df": num_classes - 1,
    }


def build_splits() -> Dict[str, Any]:
    """Exactly the MODALITY_BENCHMARKS.md 3 split recipe (split seed 0)."""
    train_all, val_all, data_prov = rm.load_imagenette()
    pool_paths = [r[0] for r in train_all]
    pool_labels = [r[1] for r in train_all]
    val_paths = [r[0] for r in val_all]
    val_labels = [r[1] for r in val_all]
    train_idx, cal_idx = bc.make_splits(pool_labels, N_TRAIN, N_CAL, seed=SPLIT_SEED)
    n_test = min(N_TEST, len(val_labels))
    test_idx = bc.stratified_indices(val_labels, n_test, seed=SPLIT_SEED)
    out: Dict[str, Any] = {
        "train": ([pool_paths[i] for i in train_idx], [pool_labels[i] for i in train_idx]),
        "cal": ([pool_paths[i] for i in cal_idx], [pool_labels[i] for i in cal_idx]),
        "test": ([val_paths[i] for i in test_idx], [val_labels[i] for i in test_idx]),
    }
    prov = {
        "data_provenance": data_prov,
        "official_train_pool_n": len(pool_labels),
        "official_val_pool_n": len(val_labels),
        "official_train_pool_balance": class_balance(pool_labels),
        "official_val_pool_balance": class_balance(val_labels),
        "train_source": (f"stratified carve from official train split "
                         f"(disjoint, seed={SPLIT_SEED})"),
        "cal_source": (f"stratified carve from the remainder of official train "
                       f"(disjoint, seed={SPLIT_SEED + 1})"),
        "test_source": (f"official validation split, stratified subsample "
                        f"(n={n_test}, seed={SPLIT_SEED}) - never fitted on"),
    }
    return out, prov


def _as_float_array(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


# --------------------------------------------------------------------------
# PHASE 0 - featurize once, cache
# --------------------------------------------------------------------------

def run_featurize(args: argparse.Namespace) -> Dict[str, Any]:
    t_start = time.perf_counter()
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    print("[featurize] loading imagenette + building splits", flush=True)
    t0 = time.perf_counter()
    splits, prov = build_splits()
    load_splits_s = round(time.perf_counter() - t0, 2)
    paths: List[str] = []
    labels: List[int] = []
    split_of: List[str] = []
    for name in ("train", "cal", "test"):
        sp, sl = splits[name]
        paths.extend(sp)
        labels.extend(sl)
        split_of.extend([name] * len(sp))
        prov[f"{name}_n"] = len(sp)
        prov[f"{name}_balance"] = class_balance(sl)
        print(f"[featurize]   {name}: n={len(sp)}", flush=True)

    assert len(set(paths)) == len(paths), "split membership is not disjoint"
    assert prov["train_n"] == N_TRAIN and prov["cal_n"] == N_CAL, "split sizes drifted"
    assert prov["test_n"] == N_TEST, "test size drifted from the published protocol"

    print("[featurize] extracting selected images", flush=True)
    t0 = time.perf_counter()
    needed = list(zip(paths, labels))
    extracted = rm.extract_imagenette_rows(needed)
    extract_s = round(time.perf_counter() - t0, 2)
    disk_paths = [str(extracted[k]) for k in paths]

    print(f"[featurize] loading frozen backbone {VISION_BACKBONE}", flush=True)
    torch.manual_seed(0)
    t0 = time.perf_counter()
    model = bc.JevX.from_pretrained(
        VISION_BACKBONE, freeze_base_model=True, pooling_method="cls"
    )
    model.eval()
    load_s = round(time.perf_counter() - t0, 2)
    cfg = model.config.backbone_config
    loaded_sha = (
        cfg.get("_commit_hash") if isinstance(cfg, dict) else None
    ) or getattr(getattr(model.model, "config", None), "_commit_hash", None)
    backbone_params = int(sum(p.numel() for p in model.model.parameters()))
    feature_dim = int(model.config.decision_hidden_size)

    print(f"[featurize] backbone forward over n={len(disk_paths)} images "
          f"(batch {args.feat_batch})", flush=True)
    t0 = time.perf_counter()
    feats = rm.featurize_payloads(
        model, disk_paths,
        lambda chunk: {"images": [rm.pil_loader(Path(p)) for p in chunk]},
        batch_size=args.feat_batch,
    )
    featurize_s = round(time.perf_counter() - t0, 2)
    if feats.shape != (len(disk_paths), feature_dim):
        raise RuntimeError(f"unexpected feature shape {feats.shape}")

    payload = {
        "features": feats,
        "labels": np.asarray(labels, dtype=np.int64),
        "split_of": np.asarray(split_of),
        "paths": np.asarray(paths),
    }
    np.savez_compressed(FEATURE_NPZ, **payload)

    out = {
        "phase": "featurize",
        "provenance": {
            "backbone_id": VISION_BACKBONE,
            "backbone_revision_api": VISION_BACKBONE_SHA,
            "backbone_revision_loaded": loaded_sha,
            "backbone_params": backbone_params,
            "pooling_method": "cls",
            "pooling_rationale": (
                "deit-tiny ships no pooler (load report: pooler.dense.{weight,bias} "
                "MISSING) so pooler_output would be a random projection; 'cls' is "
                "the token the checkpoint's own classifier consumes (MODALITY_"
                "BENCHMARKS.md 4)"),
            "feature_dim": feature_dim,
            "feat_batch": args.feat_batch,
            "head_batch": HEAD_BATCH,
            "lr": LR,
            "epochs": EPOCHS,
            "split_seed": SPLIT_SEED,
            "split_sizes": {"train": N_TRAIN, "cal": N_CAL, "test": N_TEST},
            "question": VISION_QUESTION,
            "options": CLASSES,
            "dataset": prov,
            "environment": environment_info(),
        },
        "features": {
            "n": int(feats.shape[0]),
            "dim": int(feats.shape[1]),
            "dtype": str(feats.dtype),
            "abs_mean": float(np.abs(feats).mean()),
            "finite": bool(np.isfinite(feats).all()),
            "n_rows_all_finite": bool(np.isfinite(feats).all(axis=1).all()),
        },
        "cache": {
            "path": str(FEATURE_NPZ.relative_to(REPO_ROOT)),
            "bytes": FEATURE_NPZ.stat().st_size,
            "sha256": sha256_file(FEATURE_NPZ),
        },
        "wall_s": {
            "load_and_splits_s": load_splits_s,
            "extract_selected_images_s": extract_s,
            "load_backbone_s": load_s,
            "featurize_forward_s": featurize_s,
            "total_s": round(time.perf_counter() - t_start, 2),
        },
        "published_reference": PUBLISHED,
    }
    bc.write_json(FEATURE_PROV, out)
    bc.write_json(FEATURE_RESULT, out)
    print(f"[wrote] {FEATURE_PROV}", flush=True)
    print(f"[wrote] {FEATURE_RESULT}", flush=True)
    print(f"[featurize] done in {out['wall_s']['total_s']}s "
          f"(forward {featurize_s}s)", flush=True)
    return out


def load_features() -> Dict[str, Any]:
    if not FEATURE_NPZ.exists():
        raise SystemExit(
            f"{FEATURE_NPZ} not found - run `--phase featurize` first "
            "(the deit features are not cached anywhere in the repo)."
        )
    z = np.load(FEATURE_NPZ, allow_pickle=False)
    feats = _as_float_array(z["features"])
    labels = np.asarray(z["labels"], dtype=np.int64)
    split_of = [str(s) for s in z["split_of"]]
    sel = {name: np.asarray([i for i, s in enumerate(split_of) if s == name],
                             dtype=np.int64)
           for name in ("train", "cal", "test")}
    return {
        "features": feats,
        "labels": labels,
        "sel": sel,
        "n": int(feats.shape[0]),
        "dim": int(feats.shape[1]),
    }


# --------------------------------------------------------------------------
# Head wrappers (the only "new" code; everything else is the shipped surface)
# --------------------------------------------------------------------------

class VisionHead(nn.Module):
    """Steps 2-8 of JevX over cached pooled features, in three switchable forms.

    ``conditioning=True`` reproduces :class:`benchmarks.bench_common.
    FeaturizedJevX` byte-for-byte (asserted at run time by
    ``assert_matches_shipped_wrapper``): state_proj -> concat fusion with the
    fixed question row and the fixed 10-option schema -> choice head with the
    option interaction at full strength.

    ``conditioning=False`` is the shipped ``forward(condition_options=False)``
    path, replicated exactly as ``JevXForDecision.forward`` spells it: the
    schema rows are removed BEFORE fusion (``options=None``), the head receives
    an all-zeros ``[B, K, D]`` shape placeholder, and ``conditioning_enabled=
    False`` makes ``_combine`` return the state scores with the residual exactly
    zero (not computed, so no gradient reaches the option block). The duplicate-
    option tie is switched off with it, exactly as the shipped call does.

    ``use_question=False`` additionally removes the question pathway: the
    scoring input is the projected state alone, so the choice head reduces to
    an exact ``nn.Linear(D, K)`` (``state_pathway="linear"``) or an exact
    ``nn.Sequential(Linear(D, H), act, Linear(H, K))`` (``state_pathway="mlp"``)
    - the P1 / P2 probe forms, with no fusion layer anywhere in the graph.
    """

    def __init__(self, model: Any, *, conditioning: bool, use_question: bool,
                 num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.jevx = model
        self.conditioning = conditioning
        self.use_question = use_question
        self.num_classes = num_classes
        self.question = VISION_QUESTION
        self.options = list(CLASSES)

    # -- the shipped encoders -------------------------------------------------

    def _question_rows(self, batch_size: int) -> torch.Tensor:
        m = self.jevx
        rows = m.question_encoder.encode_texts([self.question], tokenizer=m.tokenizer)
        rows = m.question_encoder(rows)
        rows = m.question_proj(rows)  # [1, D]
        return rows.expand(batch_size, -1).contiguous()

    def _schema(self, batch_size: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        m = self.jevx
        schema_row = [Option(key=o, label=o) for o in self.options]
        feats, mask = m.schema_encoder.encode([schema_row], tokenizer=m.tokenizer)
        feats = m.schema_proj(feats)  # [1, K, D]
        feats = feats.expand(batch_size, -1, -1).contiguous()
        mask_b = mask.expand(batch_size, -1).contiguous() if mask is not None else None
        option_mask = mask_b if (mask_b is not None and not bool(mask_b.all())) else None
        return feats, option_mask

    # -- forward --------------------------------------------------------------

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        m = self.jevx
        state = batch["state"]
        feats = torch.from_numpy(state) if isinstance(state, np.ndarray) else \
            torch.as_tensor(np.stack([np.asarray(r) for r in state]))
        feats = feats.to(dtype=torch.float32)
        b = feats.shape[0]
        head = m.heads["choice"]

        if not self.use_question:
            # State path only: no question, no schema, no fusion. The head still
            # needs a [B, K, D] option block for its shape contract; it is an
            # exact-zeros placeholder and the residual is disabled, so the
            # logits are EXACTLY the state pathway applied to the projected
            # state.
            state_repr = m.state_proj(feats)
            placeholder = torch.zeros(
                b, self.num_classes, state_repr.shape[-1], dtype=state_repr.dtype
            )
            out = head(state_repr, placeholder, option_mask=None,
                       conditioning_enabled=False, duplicate_tie=False)
            return out.logits

        state_repr = m.state_proj(feats)
        question_rows = self._question_rows(b)
        option_features, option_mask = self._schema(b)
        if self.conditioning:
            fused = m.fusion(state_repr, question_rows, option_features, option_mask)
            head_options = fused.option_features
            cond = True
        else:
            fused = m.fusion(state_repr, question_rows, None, None)
            head_options = fused.option_features
            if head_options is None and option_features is not None:
                head_options = torch.zeros_like(option_features)
            cond = False
        out = head(fused.fused, head_options, option_mask=option_mask,
                   conditioning_enabled=cond, duplicate_tie=cond)
        return out.logits


def assert_matches_shipped_wrapper(model: Any, feats: np.ndarray) -> float:
    """VisionHead(conditioning=True) vs bench_common.FeaturizedJevX, max |d|."""
    mine = VisionHead(model, conditioning=True, use_question=True)
    ref = bc.FeaturizedJevX(model, VISION_QUESTION, CLASSES)
    probe = {"state": feats[:8]}
    mine.eval()
    ref.eval()
    with torch.no_grad():
        a = mine(probe)
        b = ref(probe)
    return float((a - b).abs().max().item())


# --------------------------------------------------------------------------
# The V0-V6 grid
# --------------------------------------------------------------------------

def _cell_specs() -> List[Dict[str, Any]]:
    """One dict per cell. ``kind`` selects the head form, the rest the config.

    V5 is stated in the brief as "V2 with state_hidden_size doubled then
    halved", but V2 is the LINEAR state path, which has no hidden width. The
    sweep is therefore run on the mlp state path (the only place
    ``state_hidden_size`` exists): V5a = H = D/2 = 96, V2m = H = D = 192 (the
    shipped centre), V5b = H = 2D = 384. Documented as a deviation.
    """
    d = 192  # decision_hidden_size for deit-tiny CLS features (measured)
    return [
        dict(cell="V0", kind="full", state_pathway="mlp", conditioning=True,
             use_question=True, epochs=EPOCHS, init="default", loss="ce",
             desc="reference: the shipped stack (interaction ON, state_pathway=mlp)"),
        dict(cell="V0L", kind="full", state_pathway="linear", conditioning=True,
             use_question=True, epochs=EPOCHS, init="default", loss="ce",
             desc="supplementary: the shipped stack with the CURRENT config default "
                  "state_pathway='linear' (the MODALITY_BENCHMARKS.md 4.1 stack)"),
        dict(cell="V1", kind="full", state_pathway="mlp", conditioning=False,
             use_question=True, epochs=EPOCHS, init="default", loss="ce",
             desc="conditioning_mode=off equivalent: state-only head, schema removed "
                  "before fusion, residual exactly zero (the shipped "
                  "forward(condition_options=False) path)"),
        dict(cell="V2", kind="probe", state_pathway="linear", conditioning=False,
             use_question=False, epochs=EPOCHS, init="zero", loss="ce",
             desc="state path ONLY as an exact nn.Linear(D,K) probe (P1): "
                  "state_proj -> Linear(192,10), no question/schema/fusion"),
        dict(cell="V3", kind="probe", state_pathway="linear", conditioning=False,
             use_question=False, epochs=EPOCHS, init="probe", loss="ce",
             desc="V2 + probe-style initialization of the state path (nn.Linear's own "
                  "kaiming-uniform weight / uniform bias instead of the shipped "
                  "zero-init residual convention)"),
        dict(cell="V4", kind="probe", state_pathway="linear", conditioning=False,
             use_question=False, epochs=20, init="zero", loss="ce",
             desc="V2 + 20 epochs (budget sensitivity)"),
        dict(cell="V2m", kind="probe", state_pathway="mlp", conditioning=False,
             use_question=False, epochs=EPOCHS, init="zero", loss="ce",
             desc="V2 with the mlp state path, H = D = 192 (sweep centre for V5)"),
        dict(cell="V5a", kind="probe", state_pathway="mlp", conditioning=False,
             use_question=False, epochs=EPOCHS, init="zero", loss="ce",
             state_hidden_size=d // 2, desc="V2m with state_hidden_size halved: H = 96"),
        dict(cell="V5b", kind="probe", state_pathway="mlp", conditioning=False,
             use_question=False, epochs=EPOCHS, init="zero", loss="ce",
             state_hidden_size=2 * d, desc="V2m with state_hidden_size doubled: H = 384"),
        dict(cell="V6", kind="probe", state_pathway="linear", conditioning=False,
             use_question=False, epochs=EPOCHS, init="zero", loss="balanced",
             desc="V2 with a class-balanced (inverse-frequency) re-weighting of the "
                  "training cross-entropy"),
        dict(cell="V2I", kind="probe", state_pathway="linear", conditioning=False,
             use_question=True, epochs=EPOCHS, init="zero", loss="ce",
             fusion_type="identity",
             desc="V2 expressed ONLY through shipped config fields: "
                  "fusion_type='identity' (the Fusion.context of "
                  "jevx/representations/fusion.py:IdentityFusion returns the state "
                  "unchanged) + conditioning_mode='off' + state_pathway='linear'. "
                  "Confirms the V2 logits are reachable without any code change."),
    ]


def _reference_specs() -> List[Dict[str, Any]]:
    return [
        dict(cell="R1", kind="b1_linear",
             desc="B1 linear probe, nn.Linear(192,10) on the pooled feature "
                  "(== P1), same features / splits / 5 seeds"),
        dict(cell="R2", kind="b2_mlp",
             desc="B2 MLP probe, Linear(192,192)-ReLU-Linear(192,10) (== P2), same "
                  "features / splits / 5 seeds"),
    ]


def _apply_probe_init(head: nn.Module) -> None:
    """Replace the zero-init state path with nn.Linear's own init scheme.

    Exactly what ``nn.Linear(H, K).reset_parameters()`` does (kaiming_uniform_
    with a=sqrt(5) on the [K, H] weight, uniform bias in the same bound), so
    the state path starts from the SAME distribution as the published P1/P2
    probes instead of the shipped zero-init residual convention
    (jevx/decisions/generic.py:294 and :423).
    """
    pathway = head.heads["choice"].state_pathway
    bound_w = 1.0 / math.sqrt(pathway.hidden_size)
    bound_b = 1.0 / math.sqrt(pathway.hidden_size)
    with torch.no_grad():
        # nn.Linear computes bound = 1/sqrt(fan_in) with fan_in = H for the
        # weight and the SAME bound for the bias; fan_in of the state-scoring
        # layer is the pathway width.
        fan_in = getattr(pathway, "mlp_hidden_size", pathway.hidden_size)
        bound_w = 1.0 / math.sqrt(fan_in)
        bound_b = 1.0 / math.sqrt(fan_in)
        if hasattr(pathway, "w2"):  # MLPStatePathway
            pathway.w2.uniform_(-bound_w, bound_w)
            pathway.b2.uniform_(-bound_b, bound_b)
        else:  # LinearProbePathway
            pathway.weight.uniform_(-bound_w, bound_w)
            pathway.bias.uniform_(-bound_b, bound_b)


def _balanced_loss_builder(train_labels: Sequence[int], num_classes: int):
    """Class-balanced cross-entropy: identical to choice_loss up to the weight.

    ``choice_loss`` is ``F.cross_entropy(masked_logits, target)``; with
    ``option_mask=None`` (every column valid, which is the case here) the two
    are the same call. V6 replaces the reduction with the inverse-frequency
    weighting ``w_k = n / (K * n_k)``, PyTorch's standard ``reduction="mean"``
    weighted cross-entropy (the denominator is the batch's summed weights, so
    the loss scale is comparable to the unweighted arm).
    """
    counts = np.bincount(np.asarray(train_labels, dtype=np.int64),
                         minlength=num_classes).astype(np.float64)
    w = counts.sum() / (num_classes * np.maximum(counts, 1.0))
    weight = torch.tensor(w, dtype=torch.float32)

    def loss_fn(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, labels, weight=weight, reduction="mean")

    return loss_fn, {"class_counts": counts.tolist(), "weights": w.tolist(),
                     "formula": "w_k = n / (K * n_k)"}


def _build_jevx(spec: Dict[str, Any], seed: int) -> Any:
    kwargs: Dict[str, Any] = {
        "freeze_base_model": True,
        "pooling_method": "cls",
        "state_pathway": spec["state_pathway"],
        "conditioning_mode": "full" if spec["conditioning"] else "off",
    }
    if spec.get("state_hidden_size") is not None:
        kwargs["state_hidden_size"] = int(spec["state_hidden_size"])
    if spec.get("fusion_type") is not None:
        kwargs["fusion_type"] = str(spec["fusion_type"])
    torch.manual_seed(seed)
    model = bc.JevX.from_pretrained(VISION_BACKBONE, **kwargs)
    model.eval()
    return model


def _params_with_gradient(module: nn.Module, train_records: List[Dict[str, Any]],
                          batch_size: int, seed: int) -> int:
    """Count parameters that actually receive a gradient in the training graph.

    ``bc.count_trainable`` counts every ``requires_grad`` parameter, which
    includes the question / schema encoders of the V2-V6 arms even though those
    cells never route them through the forward. This measures the difference
    instead of asserting it: one forward+backward on a real training batch,
    counted from ``p.grad is not None``. The optimizer is NOT stepped, and the
    module is left in eval mode with zeroed grads.
    """
    module.eval()
    for p in module.parameters():
        p.grad = None
    loader = bc.make_loader(train_records, batch_size, shuffle=False)
    with torch.enable_grad():
        for batch in loader:
            loss, _ = make_supervised_step_fn(module)(batch)
            loss.backward()
            break
    n = sum(1 for p in module.parameters() if p.requires_grad and p.grad is not None)
    for p in module.parameters():
        p.grad = None
    return int(n)


def _run_one(seed: int, spec: Dict[str, Any], data: Dict[str, Any],
             records: Dict[str, List[Dict[str, Any]]],
             balanced: Optional[Tuple[Any, Dict[str, Any]]],
             args: argparse.Namespace, inv: rm.InvariantLog) -> Dict[str, Any]:
    kind = spec["kind"]
    epochs = int(spec.get("epochs", args.epochs))
    t0 = time.perf_counter()

    if kind in ("b1_linear", "b2_mlp"):
        torch.manual_seed(seed)
        module: nn.Module = (bc.LinearProbe(data["dim"], NUM_CLASSES) if kind == "b1_linear"
                             else bc.MLPProbe(data["dim"], NUM_CLASSES))
    else:
        model = _build_jevx(spec, seed)
        # The cell must be the CONFIG-driven state-only path, not a hand-set
        # flag: the shipped resolver must answer exactly what the cell claims.
        resolved = bool(model._resolve_condition(None))
        if resolved != bool(spec["conditioning"]):
            raise RuntimeError(
                f"{spec['cell']}: _resolve_condition(None)={resolved} but the cell "
                f"spec says conditioning={spec['conditioning']} "
                f"(config.conditioning_mode={model.config.conditioning_mode!r})"
            )
        if spec.get("init") == "probe":
            _apply_probe_init(model)
        module = VisionHead(
            model,
            conditioning=bool(spec["conditioning"]),
            use_question=bool(spec["use_question"]),
        )

    loss_fn = None
    loss_meta = {"kind": "choice_loss (shipped default)"}
    if spec.get("loss") == "balanced":
        assert balanced is not None
        loss_fn, loss_meta = balanced[0], dict(balanced[1], kind="class-balanced CE")

    torch.manual_seed(seed)
    trainer = bc.HeadTrainer(module, lr=args.lr, seed=seed)
    train_loader = bc.make_loader(records["train"], args.head_batch_size,
                                 shuffle=True, generator=trainer.generator)
    step = None
    if loss_fn is not None:
        step = make_supervised_step_fn(module, loss_fn=loss_fn)
    history = trainer.fit(train_loader, epochs=epochs, loss_fn=step)
    hist = list(history.epochs)

    cal_logits, cal_labels = rm.collect_logits_checked(
        module, records["cal"], batch_size=args.head_batch_size,
        num_classes=NUM_CLASSES, inv=inv, tag=f"{spec['cell']}/cal/seed{seed}")
    test_logits, test_labels = rm.collect_logits_checked(
        module, records["test"], batch_size=args.head_batch_size,
        num_classes=NUM_CLASSES, inv=inv, tag=f"{spec['cell']}/test/seed{seed}")

    row: Dict[str, Any] = {
        "seed": seed,
        "epochs": epochs,
        "lr": args.lr,
        "head_batch_size": args.head_batch_size,
        "train_loss_first": float(hist[0]["loss"]),
        "train_loss_last": float(hist[-1]["loss"]),
        "train_loss_curve": [float(r["loss"]) for r in hist],
        "trainable_params": int(bc.count_trainable(module)),
        "params_with_gradient": _params_with_gradient(module, records["train"],
                                                      args.head_batch_size, seed),
        "loss": loss_meta,
        "uncalibrated": bc.metrics_from_logits(test_logits, test_labels),
        "temperature": bc.calibrated_metrics(
            test_logits, test_labels, "temperature",
            fit_logits=cal_logits, fit_labels=cal_labels),
        "wall_s": round(time.perf_counter() - t0, 2),
    }
    if kind not in ("b1_linear", "b2_mlp"):
        row["config"] = {
            "state_pathway": spec["state_pathway"],
            "state_hidden_size": spec.get("state_hidden_size"),
            "conditioning_mode": module.jevx.config.conditioning_mode,
            "conditioning_enabled": bool(module.jevx.config.conditioning_enabled),
            "interaction_scale": float(module.jevx.heads["choice"].get_interaction_scale),
            "decision_hidden_size": int(module.jevx.config.decision_hidden_size),
        }
    return row


def run_grid(args: argparse.Namespace) -> Dict[str, Any]:
    t_start = time.perf_counter()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    data = load_features()
    feats, labels, sel = data["features"], data["labels"], data["sel"]
    records = {
        name: bc.build_records(feats[sel[name]], labels[sel[name]],
                               VISION_QUESTION, CLASSES)
        for name in ("train", "cal", "test")
    }
    train_labels = [int(y) for y in labels[sel["train"]]]
    balanced = _balanced_loss_builder(train_labels, NUM_CLASSES)

    wanted = None if args.cells == "all" else {c.strip() for c in args.cells.split(",")}
    specs = [s for s in _cell_specs() + _reference_specs()
             if wanted is None or s["cell"] in wanted]
    if not specs:
        raise SystemExit(f"no cells matched --cells {args.cells!r}")

    inv = rm.InvariantLog()
    # The head wrapper must reproduce the shipped FeaturizedJevX exactly for
    # the conditioned arm; assert it on a CANONICAL model (the shipped default
    # fusion), independently of whichever cells this invocation happens to run.
    probe_model = _build_jevx({"state_pathway": "mlp", "conditioning": True}, 0)
    wrapper_diff = assert_matches_shipped_wrapper(probe_model, feats)
    del probe_model
    print(f"[grid] VisionHead(conditioning=True) vs FeaturizedJevX "
          f"(canonical concat-fusion model): max_abs_diff={wrapper_diff}", flush=True)

    cells: Dict[str, Any] = {}
    failures: Dict[str, str] = {}
    if getattr(args, "merge", False) and GRID_RESULT.exists():
        prev = json.loads(GRID_RESULT.read_text(encoding="utf-8"))
        cells = dict(prev.get("cells", {}))
        failures = dict(prev.get("failures", {}))
        print(f"[grid] merging into the existing result: {len(cells)} cells carried over",
              flush=True)
    for spec in specs:
        cell = spec["cell"]
        t0 = time.perf_counter()
        per_seed: List[Dict[str, Any]] = []
        cell_fail: Dict[str, str] = {}
        for seed in seeds:
            try:
                per_seed.append(_run_one(seed, spec, data, records, balanced,
                                         args, inv))
            except Exception as exc:  # reported verbatim, never dropped
                cell_fail[str(seed)] = f"{type(exc).__name__}: {exc}"
                print(f"[grid] {cell} seed={seed} FAILED: {type(exc).__name__}: {exc}",
                      flush=True)
        if not per_seed:
            failures[cell] = json.dumps(cell_fail)[:2000]
            print(f"[grid] {cell}: ALL SEEDS FAILED ({len(cell_fail)}); recorded, "
                  f"not omitted", flush=True)
            continue
        agg: Dict[str, Any] = {}
        for cal_key in ("uncalibrated", "temperature"):
            agg[cal_key] = bc.mean_std_over_seeds([r[cal_key] for r in per_seed])
        cells[cell] = {
            "cell": cell,
            "description": spec["desc"],
            "state_pathway": spec.get("state_pathway"),
            "state_hidden_size": spec.get("state_hidden_size"),
            "conditioning": spec.get("conditioning"),
            "use_question": spec.get("use_question"),
            "init": spec.get("init"),
            "loss": per_seed[0]["loss"],
            "per_seed": per_seed,
            "aggregate": agg,
            "trainable_params": per_seed[0]["trainable_params"],
            "params_with_gradient": per_seed[0]["params_with_gradient"],
            "wall_s": round(time.perf_counter() - t0, 2),
            "per_seed_wall_s": [r["wall_s"] for r in per_seed],
        }
        acc = agg["uncalibrated"]["accuracy"]
        acc_t = agg["temperature"]["accuracy"]
        print(f"[grid] {cell}: acc={acc['mean']:.4f}+/-{acc['std']:.4f} "
              f"(cal {acc_t['mean']:.4f}) params={per_seed[0]['trainable_params']} "
              f"(grad {per_seed[0]['params_with_gradient']}) "
              f"wall={cells[cell]['wall_s']}s", flush=True)
        # Resumable: persist after every cell.
        bc.write_json(GRID_RESULT, _grid_envelope(args, data, cells, failures,
                                                  inv, wrapper_diff, t_start,
                                                  balanced))

    envelope = _grid_envelope(args, data, cells, failures, inv, wrapper_diff,
                              t_start, balanced)
    bc.write_json(GRID_RESULT, envelope)
    print(f"[wrote] {GRID_RESULT}", flush=True)
    return envelope


def _grid_envelope(args: argparse.Namespace, data: Dict[str, Any],
                   cells: Dict[str, Any], failures: Dict[str, str],
                   inv: rm.InvariantLog, wrapper_diff: float, t_start: float,
                   balanced: Tuple[Any, Dict[str, Any]]) -> Dict[str, Any]:
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    return {
        "phase": "grid",
        "provenance": {
            "feature_cache": str(FEATURE_NPZ.relative_to(REPO_ROOT)),
            "feature_cache_sha256": sha256_file(FEATURE_NPZ) if FEATURE_NPZ.exists() else None,
            "n_features": data["n"],
            "feature_dim": data["dim"],
            "backbone_id": VISION_BACKBONE,
            "backbone_revision_api": VISION_BACKBONE_SHA,
            "dataset_id": VISION_DATASET_ID,
            "dataset_revision": VISION_DATASET_SHA,
            "question": VISION_QUESTION,
            "options": CLASSES,
            "seeds": seeds,
            "split_seed": SPLIT_SEED,
            "split_sizes": {"train": N_TRAIN, "cal": N_CAL, "test": N_TEST},
            "hyperparams": {"lr": args.lr, "epochs_default": args.epochs,
                            "head_batch_size": args.head_batch_size,
                            "optimizer": "HeadTrainer AdamW (lr a priori, no search)"},
            "cell_definitions": {s["cell"]: s["desc"]
                                 for s in _cell_specs() + _reference_specs()},
            "class_balance": {
                "train": class_balance(data["labels"][data["sel"]["train"]]),
                "cal": class_balance(data["labels"][data["sel"]["cal"]]),
                "test": class_balance(data["labels"][data["sel"]["test"]]),
            },
            "balanced_loss_weights": balanced[1],
            "environment": environment_info(),
            "published_reference": PUBLISHED,
        },
        "wrapper_equivalence": {
            "check": ("VisionHead(conditioning=True, use_question=True) vs "
                      "benchmarks.bench_common.FeaturizedJevX on 8 cached rows, "
                      "canonical model (shipped default fusion_type='concat', "
                      "state_pathway='mlp')"),
            "max_abs_diff": wrapper_diff,
        },
        "cells": cells,
        "failures": failures,
        "invariants": inv.summary(),
        "wall_s": round(time.perf_counter() - t_start, 2),
    }


# --------------------------------------------------------------------------
# PHASE summarize - render every reported number from the result JSON
# --------------------------------------------------------------------------

_CELL_ORDER = ["V0", "V0L", "V1", "V2", "V3", "V4", "V2m", "V5a", "V5b", "V6",
               "V2I", "R1", "R2"]


def _fmt(v: Optional[float], nd: int = 4) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{nd}f}"


def summarize(args: argparse.Namespace) -> str:
    if not GRID_RESULT.exists():
        raise SystemExit(f"{GRID_RESULT} not found - run `--phase grid` first")
    j = json.loads(GRID_RESULT.read_text(encoding="utf-8"))
    cells = j["cells"]
    out: List[str] = []
    seeds = j["provenance"]["seeds"]

    def acc(cell: str, key: str = "uncalibrated") -> List[float]:
        return [r[key]["accuracy"] for r in cells[cell]["per_seed"]]

    def agg(cell: str, metric: str, key: str = "uncalibrated") -> Tuple[float, float]:
        a = cells[cell]["aggregate"][key][metric]
        return a["mean"], a["std"]

    out.append("## 1. Headline grid (test split, 5 seeds 0-4, mean +/- sample std)\n")
    out.append("| cell | definition | acc | macro-F1 | NLL | ECE(15) | Brier(15) | "
               "ECE(15) temp | trainable params | params w/ grad (tensors) | "
               "train loss 1st->last | wall s |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in _CELL_ORDER:
        if c not in cells:
            continue
        cc = cells[c]
        a = cc["aggregate"]["uncalibrated"]
        t = cc["aggregate"]["temperature"]
        loss1 = sum(r["train_loss_first"] for r in cc["per_seed"]) / len(cc["per_seed"])
        lossL = sum(r["train_loss_last"] for r in cc["per_seed"]) / len(cc["per_seed"])
        out.append(
            f"| {c} | {cc['description'][:88]} "
            f"| {a['accuracy']['mean']:.4f} +/- {a['accuracy']['std']:.4f} "
            f"| {a['macro_f1']['mean']:.4f} | {a['nll']['mean']:.4f} "
            f"| {a['ece']['mean']:.4f} | {a['brier']['mean']:.4f} "
            f"| {t['ece']['mean']:.4f} | {cc['trainable_params']:,} "
            f"| {cc['params_with_gradient']:,} "
            f"| {loss1:.4f} -> {lossL:.4f} | {cc['wall_s']:.1f} |")
    out.append("")

    out.append("## 2. Per-seed accuracy (paired; the split is identical for all cells)\n")
    out.append("| cell | " + " | ".join(f"seed {s}" for s in seeds) +
               " | mean | std | min | max |")
    out.append("|---|" + "---|" * (len(seeds) + 4))
    for c in _CELL_ORDER:
        if c not in cells:
            continue
        v = acc(c)
        m, s_ = agg(c, "accuracy")
        out.append(f"| {c} | " + " | ".join(f"{x:.4f}" for x in v) +
                   f" | {m:.4f} | {s_:.4f} | {min(v):.4f} | {max(v):.4f} |")
    out.append("")

    out.append("## 3. Paired per-seed deltas (same seeds, same features, same splits)\n")
    for ref in ("V0", "V2"):
        if ref not in cells:
            continue
        out.append(f"### vs {ref} (accuracy, percentage points, positive = better)\n")
        out.append("| cell | " + " | ".join(f"seed {s}" for s in seeds) +
                   " | mean | std | seeds better | seed ranges overlap |")
        out.append("|---|" + "---|" * (len(seeds) + 4))
        base = acc(ref)
        for c in _CELL_ORDER:
            if c not in cells or c == ref:
                continue
            v = acc(c)
            d = [100.0 * (a - b) for a, b in zip(v, base)]
            better = sum(1 for x in d if x > 0)
            lo_a, hi_a = min(v), max(v)
            lo_b, hi_b = min(base), max(base)
            overlap = not (hi_a < lo_b or hi_b < lo_a)
            dm = sum(d) / len(d)
            ds = (sum((x - dm) ** 2 for x in d) / (len(d) - 1)) ** 0.5
            out.append(f"| {c} | " + " | ".join(f"{x:+.2f}" for x in d) +
                       f" | {dm:+.2f} | {ds:.2f} | {better}/{len(d)} | "
                       f"{'YES' if overlap else 'NO'} |")
        out.append("")

    # ---- Task 3: the MLP gap ------------------------------------------------
    out.append("## 4. MLP-gap quantification (paired, same 5 seeds)\n")
    probes = [c for c in ("R2", "R1") if c in cells]
    jevx_cells = [c for c in _CELL_ORDER
                  if c in cells and c not in ("R1", "R2")]
    best = max(jevx_cells, key=lambda c: agg(c, "accuracy")[0])
    out.append(f"Best JevX-family cell by mean accuracy: **{best}** "
               f"({agg(best, 'accuracy')[0]:.4f} +/- {agg(best, 'accuracy')[1]:.4f}).\n")
    out.append("| comparison | paired mean delta (acc, pp) | paired std | "
               "seeds better | JevX seed range | probe seed range | overlap |")
    out.append("|---|---|---|---|---|---|---|")
    for p in probes:
        for c in jevx_cells:
            v, b = acc(c), acc(p)
            d = [100.0 * (a - x) for a, x in zip(v, b)]
            dm = sum(d) / len(d)
            ds = (sum((x - dm) ** 2 for x in d) / (len(d) - 1)) ** 0.5 if len(d) > 1 else 0.0
            better = sum(1 for x in d if x > 0)
            overlap = not (max(v) < min(b) or max(b) < min(v))
            out.append(f"| {c} - {p} | {dm:+.2f} | {ds:.2f} | {better}/{len(d)} | "
                       f"[{min(v):.4f}, {max(v):.4f}] | [{min(b):.4f}, {max(b):.4f}] | "
                       f"{'YES' if overlap else 'NO'} |")
    out.append("")
    out.append("| probe | acc mean | acc std | NLL | ECE(15) | Brier(15) | params |")
    out.append("|---|---|---|---|---|---|---|")
    for p in probes:
        a = cells[p]["aggregate"]["uncalibrated"]
        out.append(f"| {p} | {a['accuracy']['mean']:.4f} | {a['accuracy']['std']:.4f} "
                   f"| {a['nll']['mean']:.4f} | {a['ece']['mean']:.4f} "
                   f"| {a['brier']['mean']:.4f} | {cells[p]['trainable_params']:,} |")
    out.append("")

    out.append("## 5. NLL / ECE / Brier, per cell (uncalibrated and +temperature)\n")
    out.append("| cell | NLL | NLL (temp) | ECE(15) | ECE(15) temp | Brier(15) | "
               "Brier(15) temp | macro-F1 | acc (temp) |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for c in _CELL_ORDER:
        if c not in cells:
            continue
        u, t = cells[c]["aggregate"]["uncalibrated"], cells[c]["aggregate"]["temperature"]
        out.append(f"| {c} | {u['nll']['mean']:.4f} | {t['nll']['mean']:.4f} "
                   f"| {u['ece']['mean']:.4f} | {t['ece']['mean']:.4f} "
                   f"| {u['brier']['mean']:.4f} | {t['brier']['mean']:.4f} "
                   f"| {u['macro_f1']['mean']:.4f} | {t['accuracy']['mean']:.4f} |")
    out.append("")

    out.append("## 6. Per-seed train loss first -> last, and per-seed wall time\n")
    out.append("| cell | " + " | ".join(f"seed {s}" for s in seeds) + " | params |")
    out.append("|---|" + "---|" * (len(seeds) + 1))
    for c in _CELL_ORDER:
        if c not in cells:
            continue
        parts = " | ".join(
            f"{r['train_loss_first']:.3f}->{r['train_loss_last']:.3f}"
            for r in cells[c]["per_seed"])
        out.append(f"| {c} | {parts} | {cells[c]['trainable_params']:,} |")
    out.append("")
    out.append("| cell | " + " | ".join(f"seed {s} wall s" for s in seeds) + " |")
    out.append("|---|" + "---|" * len(seeds))
    for c in _CELL_ORDER:
        if c not in cells:
            continue
        out.append(f"| {c} | " + " | ".join(
            f"{w:.1f}" for w in cells[c]["per_seed_wall_s"]) + " |")
    out.append("")

    out.append("## 7. Failures / invariants / equivalence\n")
    out.append("```json")
    out.append(json.dumps({"failures": j["failures"],
                           "invariants": j["invariants"],
                           "wrapper_equivalence": j["wrapper_equivalence"]}, indent=1))
    out.append("```")
    out.append("")
    text = "\n".join(out)
    (RESULT_DIR / "vision_diagnostic_summary.md").write_text(text, encoding="utf-8")
    print(text, flush=True)
    print(f"[wrote] {RESULT_DIR / 'vision_diagnostic_summary.md'}", flush=True)
    return text


# --------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[env] python={sys.version.split()[0]} torch={torch.__version__} "
          f"threads={torch.get_num_threads()}", flush=True)
    if args.phase in ("featurize", "all"):
        if FEATURE_NPZ.exists() and not args.force_refeaturize:
            print(f"[featurize] cache present: {FEATURE_NPZ} - skipping "
                  f"(pass --force-refeaturize to redo)", flush=True)
        else:
            run_featurize(args)
    if args.phase in ("grid", "all"):
        run_grid(args)
    if args.phase == "summarize":
        summarize(args)


if __name__ == "__main__":
    main()
