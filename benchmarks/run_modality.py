# Copyright (c) 2026 ysnrfd
"""Real-dataset modality quality benchmarks: EXP-V (vision) + EXP-A (audio).

Companion to benchmarks/run_quality.py (the SST-2 text protocol reported in
reports/DATASET_BENCHMARKS.md). The protocol is MIRRORED exactly:

* splits    : stratified, disjoint train / calibration / test (bench_common
              make_splits / stratified_indices, seeded);
* fitting   : heads fit ONLY on train, calibrators (temperature / isotonic)
              fit ONLY on calibration, test used ONLY for reported metrics -
              no hyperparameter was selected on any split (lr / epochs /
              batch fixed a priori at the HeadTrainer defaults);
* backbone  : frozen (Mode A, freeze_base_model=True), featurized ONCE per
              split through the JevX adapter's extract_representation with
              the MODEL-NATIVE pooling (strategy=None -> ViT pooler_output;
              wav2vec2 masked_mean family default - recorded in provenance);
* heads     : HeadTrainer + choice_loss (shared protocol across JevX, B1,
              B2);
* metrics   : Accuracy / Macro-F1 / NLL / Brier / ECE(15) / MCE(15);
* baselines : B1 linear probe, B2 MLP probe, B4 = B1 logits + temperature,
              all on the IDENTICAL cached features and splits;
* latency   : batch-1, warmup + timed iters, JevX .choice end-to-end and
              the linear-probe head only;
* seeds     : 3 training seeds (0, 1, 2) unless --seeds overrides.

Every model / dataset id below was verified via the Hugging Face API BEFORE
the run; the exact id + revision sha (and, for imagenette, the script-pinned
data archive sha256) are recorded in the result JSONs.

Result files (written as each section completes):
    benchmarks/results/{vision,audio}_jevx_quality.json
    benchmarks/results/{vision,audio}_baselines.json
    benchmarks/results/{vision,audio}_run_summary.json

Example:
    python benchmarks/run_modality.py --modality vision
    python benchmarks/run_modality.py --modality audio
    python benchmarks/run_modality.py --modality all --smoke
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

# Allow running as a script from the repo root or anywhere.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import bench_common as bc  # noqa: E402
from jevx.utils.numerical import stable_softmax  # noqa: E402

# --------------------------------------------------------------------------
# Verified ids (HF API checked before the run - reports/MODALITY_BENCHMARKS.md)
# --------------------------------------------------------------------------

VISION_BACKBONE = "facebook/deit-tiny-patch16-224"
VISION_BACKBONE_SHA = "b3428f18dcc7b543470d07f14b4a4157815d1880"
VISION_DATASET_ID = "frgfm/imagenette"
VISION_DATASET_SHA = "a8a496123b40fa739da2acf9b0dae339d30c7bae"
IMAGENETTE_CONFIG = "imagenette2-160"
IMAGENETTE_TGZ_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
IMAGENETTE_META_URL = (
    "https://huggingface.co/datasets/frgfm/imagenette/raw/"
    "{revision}/metadata/imagenette2-160/{split}.txt"
)

#: WNID order of frgfm/imagenette's _LABEL_MAP (label index = position).
IMAGENETTE_WNIDS = [
    "n01440764", "n02102040", "n02979186", "n03000684", "n03028079",
    "n03394916", "n03417042", "n03425413", "n03445777", "n03888257",
]
#: ClassLabel(names=...) order from the same script (index-aligned with WNIDs).
IMAGENETTE_CLASSES = [
    "tench", "English springer", "cassette player", "chain saw", "church",
    "French horn", "garbage truck", "gas pump", "golf ball", "parachute",
]
VISION_QUESTION = "Which category does this image belong to?"

AUDIO_BACKBONE = "superb/wav2vec2-base-superb-ks"
AUDIO_BACKBONE_SHA = "372e0486cd83e6f0c05c20a27262e9ca09450d24"
AUDIO_DATASET_ID = "Codec-SUPERB/RAVDESS"
AUDIO_DATASET_SHA = "46aa36c5fdeb1b6d63b746b74fde528a540044b2"
#: Filename-prefix label order (RAVDESS emotion codes 01-08).
AUDIO_CLASSES = [
    "neutral", "calm", "happy", "sad", "angry", "fearful", "disgusted",
    "surprised",
]
AUDIO_QUESTION = "Which emotion is spoken in this audio?"
AUDIO_SAMPLE_RATE = 16000
AUDIO_MAX_SECONDS = 3.0
AUDIO_MAX_SAMPLES = int(AUDIO_SAMPLE_RATE * AUDIO_MAX_SECONDS)

CACHE_ROOT = REPO_ROOT / "benchmarks" / ".cache"

# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--modality", default="all", choices=["vision", "audio", "all"])
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3,
                   help="HeadTrainer AdamW lr (library default 1e-3; a priori, "
                        "no hyperparameter search on any split)")
    p.add_argument("--head-batch-size", type=int, default=64)
    p.add_argument("--vision-feat-batch", type=int, default=32)
    p.add_argument("--audio-feat-batch", type=int, default=8)
    # split sizes (caps from the benchmark contract)
    p.add_argument("--vision-train-size", type=int, default=800)
    p.add_argument("--vision-cal-size", type=int, default=300)
    p.add_argument("--vision-test-size", type=int, default=500)
    p.add_argument("--audio-train-size", type=int, default=700)
    p.add_argument("--audio-cal-size", type=int, default=240)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--latency-warmup", type=int, default=10)
    p.add_argument("--latency-iters", type=int, default=100)
    p.add_argument("--skip-latency", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="tiny sizes for pipeline validation (smoke numbers are "
                        "never reportable quality results)")
    p.add_argument("--output-dir", default=str(REPO_ROOT / "benchmarks" / "results"))
    return p.parse_args(argv)


# --------------------------------------------------------------------------
# Download helper (the only network touch point)
# --------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def http_get(url: str, dest: Path, *, timeout: float = 300.0) -> Dict[str, Any]:
    """Stream url to dest (atomic .part rename); record size/sha256/etag."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return {"url": url, "cached": True, "bytes": dest.stat().st_size,
                "sha256": sha256_file(dest), "etag": None}
    tmp = dest.with_name(dest.name + ".part")
    sha = hashlib.sha256()
    t0 = time.perf_counter()
    req = urllib.request.Request(url, headers={"User-Agent": "jevx-benchmark/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        etag = resp.headers.get("ETag")
        with open(tmp, "wb") as fh:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                sha.update(chunk)
                fh.write(chunk)
    tmp.replace(dest)
    return {"url": url, "cached": False, "bytes": dest.stat().st_size,
            "sha256": sha.hexdigest(), "etag": etag,
            "wall_s": round(time.perf_counter() - t0, 2)}


# --------------------------------------------------------------------------
# Invariants (asserted on EVERY eval batch - sums / bounds / K)
# --------------------------------------------------------------------------

class InvariantLog:
    """Tracks per-batch choice-probability invariants across a run."""

    def __init__(self) -> None:
        self.n_batches = 0
        self.max_sum_dev = 0.0
        self.min_prob = 1.0
        self.max_prob = 0.0

    def check(self, logits: Any, num_classes: int, tag: str = "") -> None:
        arr = torch.as_tensor(logits)
        if not bool(torch.isfinite(arr).all()):
            raise AssertionError(f"{tag}: non-finite logits")
        probs = stable_softmax(arr, dim=-1)
        dev = float((probs.sum(dim=-1) - 1.0).abs().max())
        lo, hi = float(probs.min()), float(probs.max())
        if dev > 1e-4:
            raise AssertionError(f"{tag}: prob sum deviation {dev}")
        if lo < -1e-7 or hi > 1.0 + 1e-7:
            raise AssertionError(f"{tag}: probs outside [0,1]: [{lo}, {hi}]")
        if int(probs.shape[-1]) != num_classes:
            raise AssertionError(
                f"{tag}: expected K={num_classes}, got {int(probs.shape[-1])}")
        self.n_batches += 1
        self.max_sum_dev = max(self.max_sum_dev, dev)
        self.min_prob = min(self.min_prob, lo)
        self.max_prob = max(self.max_prob, hi)

    def summary(self) -> Dict[str, Any]:
        return {
            "n_batches_checked": self.n_batches,
            "max_prob_sum_abs_dev": self.max_sum_dev,
            "min_prob": self.min_prob,
            "max_prob": self.max_prob,
            "contract": "probs finite, in [0,1], sum=1 (tol 1e-4), K = schema size",
        }


def check_decision_invariants(out: Any, options: Sequence[str], tag: str = "") -> None:
    """Full-path (.choice) decision invariants - same contract as examples."""
    for dec in out.decisions:
        k = len(options)
        probs = dec.probabilities[0, :k]
        assert torch.all(probs >= 0) and torch.all(probs <= 1), f"{tag}: bounds"
        assert abs(float(probs.sum()) - 1.0) < 1e-6, f"{tag}: sum"
        assert 0 <= int(dec.predicted_index[0]) < k, f"{tag}: index range"
        assert dec.predicted_value in options, f"{tag}: schema membership"
        assert 0.0 <= float(dec.confidence[0]) <= 1.0, f"{tag}: confidence"


# --------------------------------------------------------------------------
# Featurize-once (mirror of bench_common.featurize for modality payloads)
# --------------------------------------------------------------------------

def featurize_payloads(
    model: Any,
    payloads: Sequence[Any],
    make_state: Callable[[Sequence[Any]], Dict[str, Any]],
    *,
    batch_size: int,
) -> np.ndarray:
    """Encode modality payloads through the frozen backbone ONCE per batch.

    Uses the JevX adapter's own prepare_inputs + extract_representation with
    strategy=None (the MODEL-NATIVE pooling - the same call model.forward
    makes for pooling_method=None), so the cached features are exactly what
    model.choice() computes (verified by equivalence_check_modality).
    """
    adapter = model.adapter
    backbone = model.model
    was_training = backbone.training
    backbone.eval()
    rows: List[np.ndarray] = []
    try:
        with torch.no_grad():
            for start in range(0, len(payloads), batch_size):
                chunk = list(payloads[start : start + batch_size])
                inputs = adapter.prepare_inputs(
                    make_state(chunk), tokenizer=None,
                    processor=model.processor, model=backbone,
                )
                # strategy identical to model.forward (strategy=config.pooling_method)
                strategy = getattr(model.config, "pooling_method", None)
                rep = adapter.extract_representation(backbone, inputs,
                                                     strategy=strategy)
                pooled = rep.pooled
                if pooled is None:
                    pooled = bc.pool(rep.sequence_output, rep.attention_mask,
                                     adapter.default_pooling)
                rows.append(pooled.detach().cpu().numpy().astype(np.float32))
    finally:
        if was_training:
            backbone.train()
    return np.concatenate(rows, axis=0)


def equivalence_check_modality(
    model: Any,
    head_model: Any,
    features: np.ndarray,
    payloads: Sequence[Any],
    make_state: Callable[[Sequence[Any]], Dict[str, Any]],
    *,
    n: int = 16,
) -> Dict[str, Any]:
    """Max-abs logit diff: full model.choice() vs cached head stack."""
    model.eval()
    n = min(n, len(payloads))
    with torch.no_grad():
        out = model.choice(
            state=make_state(list(payloads[:n])),
            questions=[head_model.question],
            options=[list(head_model.options)],
        )
        full = torch.cat([d.logits for d in out.decisions], dim=0)
        cached = head_model({"state": [features[i] for i in range(n)]})
    diff = (full - cached).abs().max().item()
    return {
        "n": n,
        "max_abs_diff": diff,
        "note": "identical modules; nonzero diff only from float re-association "
                "between independently batched backbone passes",
    }


def collect_logits_checked(
    module: torch.nn.Module,
    records: List[Dict[str, Any]],
    *,
    batch_size: int,
    num_classes: int,
    inv: InvariantLog,
    tag: str = "",
) -> Tuple[np.ndarray, np.ndarray]:
    """Forward a loader once; assert choice invariants on every batch."""
    module.eval()
    loader = bc.make_loader(records, batch_size, shuffle=False)
    logits_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            logits = module(batch)
            inv.check(logits, num_classes, tag=tag)
            labels = np.asarray(batch["labels"]).reshape(-1).astype(np.int64)
            if labels.min() < 0 or labels.max() >= num_classes:
                raise AssertionError(f"{tag}: labels outside [0, {num_classes})")
            logits_all.append(logits.detach().cpu().numpy().astype(np.float32))
            labels_all.append(labels)
    return np.concatenate(logits_all), np.concatenate(labels_all)


# --------------------------------------------------------------------------
# VISION data: frgfm/imagenette (imagenette2-160 archive - the exact S3
# tarball the pinned HF loader script downloads). Split membership follows
# the repo's metadata/train.txt + val.txt at the pinned revision; labels
# come from the parent-directory WNID via the script's _LABEL_MAP.
# --------------------------------------------------------------------------

def load_imagenette(revision: str = VISION_DATASET_SHA) -> Tuple[
        List[Tuple[str, int]], List[Tuple[str, int]], Dict[str, Any]]:
    cache = CACHE_ROOT / "imagenette2-160"
    prov: Dict[str, Any] = {"dataset_id": VISION_DATASET_ID,
                            "dataset_revision": revision,
                            "config": IMAGENETTE_CONFIG}
    prov["archive"] = http_get(IMAGENETTE_TGZ_URL, cache / "imagenette2-160.tgz")
    def strip_root(p: str) -> str:
        # metadata lines are "imagenette2-160/train/<wnid>/<file>.JPEG";
        # normalize both sides to "train/<wnid>/<file>.JPEG".
        return p.split("/", 1)[1] if p.startswith("imagenette2-160/") else p

    metas: Dict[str, set] = {}
    for split in ("train", "val"):
        url = IMAGENETTE_META_URL.format(revision=revision, split=split)
        m = http_get(url, cache / f"{split}.txt")
        metas[split] = {strip_root(line) for line in
                        (cache / f"{split}.txt").read_text(
                            encoding="utf-8").splitlines() if line.strip()}
        prov[f"metadata_{split}"] = {"url": m["url"], "bytes": m["bytes"],
                                     "sha256": m["sha256"]}

    def rel_name(name: str) -> Optional[Tuple[str, str]]:
        """(split, normalized-key) matching a metadata line, if any."""
        parts = name.split("/")
        candidates: List[str] = [strip_root(name)]
        if len(parts) > 1:
            candidates.append("/".join(parts[1:]))
        for i, p in enumerate(parts):
            if p in ("train", "val", "validation"):
                candidates.append("/".join(parts[i:]))
                break
        for c in candidates:
            if c in metas["train"]:
                return ("train", c)
            if c in metas["val"]:
                return ("val", c)
        return None

    rows: Dict[str, List[Tuple[str, int]]] = {"train": [], "val": []}
    unmatched = 0
    with tarfile.open(cache / "imagenette2-160.tgz", "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = member.name
            if not name.lower().endswith((".jpeg", ".jpg", ".png")):
                continue
            hit = rel_name(name)
            if hit is None:
                unmatched += 1
                continue
            split, key = hit
            wnid = Path(key).parent.name
            if wnid not in IMAGENETTE_WNIDS or not key.startswith(("train/", "val/")):
                unmatched += 1
                continue
            rows["train" if key.startswith("train/") else "val"].append(
                (key, IMAGENETTE_WNIDS.index(wnid)))
    if not rows["train"] or not rows["val"]:
        raise RuntimeError(
            f"imagenette indexing failed: train={len(rows['train'])} "
            f"val={len(rows['val'])} unmatched={unmatched} - tarball/metadata "
            "path convention changed; inspect the archive layout"
        )
    prov["archive_members_indexed"] = {
        "train": len(rows["train"]), "val": len(rows["val"]),
        "unmatched": unmatched,
    }
    return rows["train"], rows["val"], prov


def extract_imagenette_rows(rows: Sequence[Tuple[str, int]]) -> Dict[str, Path]:
    """Extract only the selected members (one sequential tar pass) to disk."""
    wanted = [k for k, _ in rows]
    out: Dict[str, Path] = {}
    base = CACHE_ROOT / "imagenette2-160" / "extracted"
    with tarfile.open(CACHE_ROOT / "imagenette2-160" / "imagenette2-160.tgz",
                      "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = member.name
            key = next((k for k in wanted if name.endswith(k)), None)
            if key is None or key in out:
                continue
            dest = base / key
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                fh = tar.extractfile(member)
                if fh is None:
                    continue
                dest.write_bytes(fh.read())
            out[key] = dest
    missing = [k for k in wanted if k not in out]
    if missing:
        raise RuntimeError(f"imagenette extraction missing {len(missing)} rows "
                           f"(e.g. {missing[:3]})")
    return out


def pil_loader(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


# --------------------------------------------------------------------------
# AUDIO data: Codec-SUPERB/RAVDESS (pinned revision), labels from filenames
# --------------------------------------------------------------------------

def load_ravdess(revision: str = AUDIO_DATASET_SHA) -> Tuple[
        List[str], List[int], Dict[str, Any]]:
    from huggingface_hub import snapshot_download

    t0 = time.perf_counter()
    root = snapshot_download(
        repo_id=AUDIO_DATASET_ID,
        repo_type="dataset",
        revision=revision,
        allow_patterns=["ravdess/*.wav", "ravdess.txt"],
    )
    dl_s = round(time.perf_counter() - t0, 2)
    paths = sorted(str(p) for p in Path(root).glob("ravdess/*.wav"))
    if not paths:
        raise RuntimeError(f"Codec-SUPERB/RAVDESS: no wavs under {root}")
    labels: List[int] = []
    for p in paths:
        prefix = Path(p).name.split("-")[0]
        if prefix not in AUDIO_CLASSES:
            raise RuntimeError(f"unknown RAVDESS label prefix {prefix!r} in {Path(p).name}")
        labels.append(AUDIO_CLASSES.index(prefix))
    prov = {
        "dataset_id": AUDIO_DATASET_ID,
        "dataset_revision": revision,
        "snapshot_component": Path(root).name,
        "n_clips": len(paths),
        "label_counts": {AUDIO_CLASSES[c]: int(sum(1 for y in labels if y == c))
                         for c in sorted(set(labels))},
        "download_wall_s": dl_s,
    }
    return paths, labels, prov


def load_waveform(path: Path) -> np.ndarray:
    import soundfile as sf

    wave, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if sr != AUDIO_SAMPLE_RATE:
        raise RuntimeError(f"{path}: sample rate {sr} != {AUDIO_SAMPLE_RATE}")
    if wave.ndim != 1:
        wave = wave[:, 0]
    return wave[:AUDIO_MAX_SAMPLES]  # clip contract: <= 3.0 s


# --------------------------------------------------------------------------
# Per-modality task
# --------------------------------------------------------------------------

def _counts(labels: Sequence[int]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for y in labels:
        out[str(int(y))] = out.get(str(int(y)), 0) + 1
    return out


class ModalityTask:
    """One modality experiment: data -> splits -> featurize -> train -> eval."""

    def __init__(self, args: argparse.Namespace, modality: str) -> None:
        assert modality in ("vision", "audio")
        self.args = args
        self.modality = modality
        self.seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        self.out_dir = Path(args.output_dir)
        self.inv = InvariantLog()
        self.timings: Dict[str, float] = {}
        self.failures: Dict[str, str] = {}
        self.seed_models: Dict[int, Any] = {}
        self.questions = VISION_QUESTION if modality == "vision" else AUDIO_QUESTION
        self.classes = IMAGENETTE_CLASSES if modality == "vision" else AUDIO_CLASSES
        self.num_classes = len(self.classes)
        self.backbone = VISION_BACKBONE if modality == "vision" else AUDIO_BACKBONE
        self.backbone_sha = (VISION_BACKBONE_SHA if modality == "vision"
                             else AUDIO_BACKBONE_SHA)
        if args.smoke:
            if modality == "vision":
                args.vision_train_size = 40
                args.vision_cal_size = 20
                args.vision_test_size = 20
            else:
                args.audio_train_size = 40
                args.audio_cal_size = 20

    # -- data + splits ------------------------------------------------------

    def load_data(self) -> None:
        t0 = time.perf_counter()
        if self.modality == "vision":
            train_all, val_all, self.data_prov = load_imagenette()
            self.pool_paths = [r[0] for r in train_all]
            self.pool_labels = [r[1] for r in train_all]
            self.val_paths = [r[0] for r in val_all]
            self.val_labels = [r[1] for r in val_all]
        else:
            paths, labels, self.data_prov = load_ravdess()
            self.pool_paths = list(paths)
            self.pool_labels = labels
        self.timings["load_dataset_s"] = round(time.perf_counter() - t0, 2)

        if self.modality == "vision":
            train_idx, cal_idx = bc.make_splits(
                self.pool_labels, self.args.vision_train_size,
                self.args.vision_cal_size, seed=self.args.split_seed,
            )
            # test = stratified subsample of the OFFICIAL validation split
            n_test = min(self.args.vision_test_size, len(self.val_labels))
            test_idx = bc.stratified_indices(
                self.val_labels, n_test, seed=self.args.split_seed,
            )
            self.train_paths = [self.pool_paths[i] for i in train_idx]
            self.train_labels = [self.pool_labels[i] for i in train_idx]
            self.cal_paths = [self.pool_paths[i] for i in cal_idx]
            self.cal_labels = [self.pool_labels[i] for i in cal_idx]
            self.test_paths = [self.val_paths[i] for i in test_idx]
            self.test_labels = [self.val_labels[i] for i in test_idx]
            self.train_source = ("stratified carve from official train split "
                                 f"(disjoint, seed={self.args.split_seed})")
            self.test_source = ("official validation split, stratified subsample "
                                f"(n={n_test}, seed={self.args.split_seed})")
            t0 = time.perf_counter()
            needed: List[Tuple[str, int]] = []
            for name in ("train", "cal", "test"):
                needed.extend(zip(getattr(self, f"{name}_paths"),
                                  getattr(self, f"{name}_labels")))
            self.extracted = extract_imagenette_rows(needed)
            # remap metadata keys -> on-disk extracted paths
            for name in ("train", "cal", "test"):
                setattr(self, f"{name}_paths",
                        [str(self.extracted[k])
                         for k in getattr(self, f"{name}_paths")])
            self.timings["extract_selected_images_s"] = round(
                time.perf_counter() - t0, 2)
        else:
            # RAVDESS has no official split: train/cal carved from the whole
            # corpus pool; test = the untouched disjoint remainder (capped 500).
            train_idx, cal_idx = bc.make_splits(
                self.pool_labels, self.args.audio_train_size,
                self.args.audio_cal_size, seed=self.args.split_seed,
            )
            keep = np.ones(len(self.pool_labels), dtype=bool)
            keep[train_idx] = False
            keep[cal_idx] = False
            test_idx = np.flatnonzero(keep)
            if len(test_idx) > 500:
                rem_labels = np.asarray(self.pool_labels)[test_idx]
                local = bc.stratified_indices(
                    rem_labels, 500, seed=self.args.split_seed + 2)
                test_idx = test_idx[local]
            self.train_paths = [self.pool_paths[i] for i in train_idx]
            self.train_labels = [self.pool_labels[i] for i in train_idx]
            self.cal_paths = [self.pool_paths[i] for i in cal_idx]
            self.cal_labels = [self.pool_labels[i] for i in cal_idx]
            self.test_paths = [self.pool_paths[i] for i in test_idx]
            self.test_labels = [self.pool_labels[i] for i in test_idx]
            self.train_source = ("stratified carve from the corpus pool "
                                 f"(disjoint, seed={self.args.split_seed})")
            self.test_source = ("disjoint remainder of the pool after train/cal "
                                f"(n={len(test_idx)}, seed={self.args.split_seed})")

        for name in ("train", "cal", "test"):
            if len(getattr(self, f"{name}_labels")) == 0:
                raise RuntimeError(f"{name} split is empty")
        tset, cset, xset = (set(self.train_paths), set(self.cal_paths),
                            set(self.test_paths))
        assert not (tset & cset), "train/cal overlap"
        assert not (tset & xset), "train/test overlap"
        assert not (cset & xset), "cal/test overlap"

    # -- model + featurize --------------------------------------------------

    def load_model(self, seed: Optional[int] = None) -> Any:
        if seed is not None:
            torch.manual_seed(seed)
        t0 = time.perf_counter()
        # Vision: the deit-tiny checkpoint ships no pooler weights (transformers
        # load report marks pooler.dense.{weight,bias} MISSING -> randomly
        # initialized), so the model-native pooler_output is an UNTRAINED
        # projection and would also differ across per-seed reloads. Primary
        # pooling is therefore "cls" - the token the checkpoint's own
        # classifier consumes. Audio keeps the native default (wav2vec2 has no
        # pooler; the adapter default masked_mean applies, stateless).
        pooling = "cls" if self.modality == "vision" else None
        model = bc.JevX.from_pretrained(self.backbone, freeze_base_model=True,
                                        pooling_method=pooling)
        model.eval()
        if seed is None:
            self.timings["load_backbone_s"] = round(time.perf_counter() - t0, 2)
            cfg = model.config.backbone_config
            self.backbone_revision = (
                cfg.get("_commit_hash") if isinstance(cfg, dict) else None
            ) or getattr(getattr(model.model, "config", None), "_commit_hash", None)
            self.backbone_params = sum(p.numel() for p in model.model.parameters())
            self.decision_hidden = int(model.config.decision_hidden_size)
            self.pooling_used = (
                f"pooling_method={model.config.pooling_method!r} (adapter "
                f"default = {model.adapter.default_pooling}); "
                + ("vision: 'cls' because the deit checkpoint has no trained "
                   "pooler (load report: pooler.dense.* MISSING)"
                   if self.modality == "vision" else
                   "audio: native default -> wav2vec2 has no pooler, adapter "
                   "masked_mean over frames")
            )
        return model

    def make_state(self, chunk: Sequence[Any]) -> Dict[str, Any]:
        if self.modality == "vision":
            return {"images": [
                pil_loader(Path(p)) if isinstance(p, str) else p for p in chunk]}
        return {"audio": [load_waveform(Path(p)) for p in chunk]}

    def featurize_all(self, model: Any) -> None:
        t0 = time.perf_counter()
        batch = (self.args.vision_feat_batch if self.modality == "vision"
                 else self.args.audio_feat_batch)
        self.feats: Dict[str, np.ndarray] = {}
        for name in ("train", "cal", "test"):
            print(f"[{self.modality}] featurize {name} "
                  f"(n={len(getattr(self, name + '_paths'))})", flush=True)
            self.feats[name] = featurize_payloads(
                model, list(getattr(self, f"{name}_paths")),
                self.make_state, batch_size=batch,
            )
        self.timings["featurize_all_splits_s"] = round(time.perf_counter() - t0, 2)

    # -- records ------------------------------------------------------------

    def records(self, split: str) -> List[Dict[str, Any]]:
        return bc.build_records(self.feats[split],
                                getattr(self, f"{split}_labels"),
                                self.questions, self.classes)

    # -- provenance / save --------------------------------------------------

    def provenance(self) -> Dict[str, Any]:
        return {
            "modality": self.modality,
            "backbone_id": self.backbone,
            "backbone_revision_api": self.backbone_sha,
            "backbone_revision_loaded": getattr(self, "backbone_revision", None),
            "backbone_params": getattr(self, "backbone_params", None),
            "dataset": dict(self.data_prov),
            "question": self.questions,
            "options": list(self.classes),
            "pooling": getattr(self, "pooling_used", None),
            "decision_hidden_size": getattr(self, "decision_hidden", None),
            "seeds": self.seeds,
            "hyperparams": {
                "epochs": self.args.epochs,
                "lr": self.args.lr,
                "head_batch_size": self.args.head_batch_size,
                "vision_feat_batch": self.args.vision_feat_batch,
                "audio_feat_batch": self.args.audio_feat_batch,
                "split_seed": self.args.split_seed,
                "smoke": bool(self.args.smoke),
                "audio_max_seconds":
                    AUDIO_MAX_SECONDS if self.modality == "audio" else None,
            },
            "splits": {
                "train_n": len(self.train_labels),
                "cal_n": len(self.cal_labels),
                "test_n": len(self.test_labels),
                "train_label_counts": _counts(self.train_labels),
                "cal_label_counts": _counts(self.cal_labels),
                "test_label_counts": _counts(self.test_labels),
                "train_source": self.train_source,
                "test_source": self.test_source,
            },
            "environment": bc.environment_info(),
        }

    def save(self, name: str, payload: Dict[str, Any]) -> None:
        bc.write_json(self.out_dir / name,
                      {"provenance": self.provenance(), **payload})
        print(f"[wrote] {self.out_dir / name}", flush=True)


# --------------------------------------------------------------------------
# JevX experiment (per seed): train heads, calibrate on cal, score test
# --------------------------------------------------------------------------

def run_jevx(task: ModalityTask) -> None:
    t0 = time.perf_counter()
    train_records = task.records("train")
    cal_records = task.records("cal")
    test_records = task.records("test")
    per_seed: List[Dict[str, Any]] = []
    metrics_by_method: Dict[str, List[Dict[str, float]]] = {
        "uncalibrated": [], "temperature": [], "isotonic": [],
    }
    equivalence: List[Dict[str, Any]] = []
    for seed in task.seeds:
        print(f"[{task.modality}] training JevX heads seed={seed}", flush=True)
        model = task.load_model(seed=seed)
        head = bc.FeaturizedJevX(model, task.questions, task.classes)
        trainer, history = bc.train_heads(
            head, train_records, seed=seed, lr=task.args.lr,
            epochs=task.args.epochs, batch_size=task.args.head_batch_size,
        )
        cal_logits, cal_labels = collect_logits_checked(
            head, cal_records, batch_size=task.args.head_batch_size,
            num_classes=task.num_classes, inv=task.inv, tag=f"cal/seed{seed}")
        test_logits, test_labels = collect_logits_checked(
            head, test_records, batch_size=task.args.head_batch_size,
            num_classes=task.num_classes, inv=task.inv, tag=f"test/seed{seed}")
        row: Dict[str, Any] = {
            "seed": seed,
            "train_loss_first": history[0]["loss"],
            "train_loss_last": history[-1]["loss"],
            "trainable_params": bc.count_trainable(head),
        }
        for method, key in (("none", "uncalibrated"),
                            ("temperature", "temperature"),
                            ("isotonic", "isotonic")):
            m = bc.calibrated_metrics(
                test_logits, test_labels, method,
                fit_logits=cal_logits, fit_labels=cal_labels,
            )
            metrics_by_method[key].append(m)
            row[key] = m
        equivalence.append({
            "seed": seed,
            **equivalence_check_modality(
                model, head, task.feats["test"], list(task.test_paths),
                task.make_state, n=16,
            ),
        })
        with torch.no_grad():
            check_decision_invariants(
                model.choice(
                    state=task.make_state(list(task.test_paths[:4])),
                    questions=[task.questions], options=[list(task.classes)],
                ), task.classes, tag=f"e2e/seed{seed}")
        per_seed.append(row)
        if seed == task.seeds[0]:
            task.seed_models[seed] = model  # kept for latency measurement
        else:
            del head, trainer, model
    task.save(f"{task.modality}_jevx_quality.json", {
        "per_seed": per_seed,
        "aggregate": {k: bc.mean_std_over_seeds(v)
                      for k, v in metrics_by_method.items()},
        "equivalence": equivalence,
        "invariants": task.inv.summary(),
        "wall_s": round(time.perf_counter() - t0, 2),
    })


# --------------------------------------------------------------------------
# Baselines: B1 linear probe, B2 MLP probe, B4 = B1 + temperature
# --------------------------------------------------------------------------

def run_baselines(task: ModalityTask) -> None:
    t0 = time.perf_counter()
    train_records = task.records("train")
    cal_records = task.records("cal")
    test_records = task.records("test")
    in_dim = int(task.feats["train"].shape[1])
    per_seed: List[Dict[str, Any]] = []
    metrics_by_model: Dict[str, List[Dict[str, float]]] = {
        "b1_linear": [], "b2_mlp": [], "b4_linear_temperature": [],
    }
    for seed in task.seeds:
        print(f"[{task.modality}] training baselines seed={seed}", flush=True)
        results: Dict[str, Any] = {"seed": seed}
        # --- B1 -------------------------------------------------------------
        b1 = bc.LinearProbe(in_dim, task.num_classes)
        _, hist_b1 = bc.train_heads(
            b1, train_records, seed=seed, lr=task.args.lr,
            epochs=task.args.epochs, batch_size=task.args.head_batch_size)
        b1_cal, b1_cal_lab = collect_logits_checked(
            b1, cal_records, batch_size=task.args.head_batch_size,
            num_classes=task.num_classes, inv=task.inv, tag=f"b1/cal/seed{seed}")
        b1_test, b1_test_lab = collect_logits_checked(
            b1, test_records, batch_size=task.args.head_batch_size,
            num_classes=task.num_classes, inv=task.inv, tag=f"b1/test/seed{seed}")
        results["b1_linear"] = {
            "train_loss_first": hist_b1[0]["loss"],
            "train_loss_last": hist_b1[-1]["loss"],
            "trainable_params": bc.count_trainable(b1),
            "test": bc.metrics_from_logits(b1_test, b1_test_lab),
        }
        metrics_by_model["b1_linear"].append(results["b1_linear"]["test"])
        # --- B4 = B1 logits + temperature fitted on the SAME cal split -----
        results["b4_linear_temperature"] = bc.calibrated_metrics(
            b1_test, b1_test_lab, "temperature",
            fit_logits=b1_cal, fit_labels=b1_cal_lab,
        )
        metrics_by_model["b4_linear_temperature"].append(
            results["b4_linear_temperature"])
        # --- B2 -------------------------------------------------------------
        b2 = bc.MLPProbe(in_dim, task.num_classes)
        _, hist_b2 = bc.train_heads(
            b2, train_records, seed=seed, lr=task.args.lr,
            epochs=task.args.epochs, batch_size=task.args.head_batch_size)
        _, b2_cal_lab = collect_logits_checked(
            b2, cal_records, batch_size=task.args.head_batch_size,
            num_classes=task.num_classes, inv=task.inv, tag=f"b2/cal/seed{seed}")
        b2_test, b2_test_lab = collect_logits_checked(
            b2, test_records, batch_size=task.args.head_batch_size,
            num_classes=task.num_classes, inv=task.inv, tag=f"b2/test/seed{seed}")
        results["b2_mlp"] = {
            "train_loss_first": hist_b2[0]["loss"],
            "train_loss_last": hist_b2[-1]["loss"],
            "trainable_params": bc.count_trainable(b2),
            "test": bc.metrics_from_logits(b2_test, b2_test_lab),
        }
        metrics_by_model["b2_mlp"].append(results["b2_mlp"]["test"])
        per_seed.append(results)
    task.save(f"{task.modality}_baselines.json", {
        "per_seed": per_seed,
        "aggregate": {k: bc.mean_std_over_seeds(v)
                      for k, v in metrics_by_model.items()},
        "invariants": task.inv.summary(),
        "wall_s": round(time.perf_counter() - t0, 2),
    })


# --------------------------------------------------------------------------
# Latency: batch-1 JevX .choice end-to-end + linear-probe head only
# --------------------------------------------------------------------------

def run_latency(task: ModalityTask) -> None:
    if task.args.skip_latency:
        return
    t0 = time.perf_counter()
    model = task.seed_models.get(task.seeds[0])
    if model is None:
        model = task.load_model()
        task.seed_models[0] = model
    model.eval()
    payload = task.test_paths[0]
    state = task.make_state([payload])
    with torch.no_grad():
        check_decision_invariants(
            model.choice(state=state, questions=[task.questions],
                         options=[list(task.classes)]),
            task.classes, tag="latency-probe")

    def e2e() -> Any:
        with torch.no_grad():
            return model.choice(state=state, questions=[task.questions],
                                options=[list(task.classes)])

    jevx_lat = bc.measure_latency(
        e2e, warmup=task.args.latency_warmup, iters=task.args.latency_iters)

    in_dim = int(task.feats["train"].shape[1])
    probe = bc.LinearProbe(in_dim, task.num_classes).eval()
    row = torch.from_numpy(task.feats["test"][0]).unsqueeze(0)

    def head_only() -> Any:
        with torch.no_grad():
            return probe({"state": row})

    head_lat = bc.measure_latency(
        head_only, warmup=task.args.latency_warmup, iters=task.args.latency_iters)
    task.save(f"{task.modality}_baselines_latency.json", {
        "latency": {
            "jevx_choice_e2e_batch1": jevx_lat,
            "linear_probe_head_only_batch1": head_lat,
            "warmup": task.args.latency_warmup,
            "iters": task.args.latency_iters,
            "note": ("ms per single-example call on a shared host; e2e = "
                     "processor + frozen backbone + JevX heads through "
                     "model.choice (torch.no_grad); head-only = cached "
                     "feature -> Linear"),
        },
        "wall_s": round(time.perf_counter() - t0, 2),
    })


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def run_modality(args: argparse.Namespace, modality: str) -> None:
    t_start = time.perf_counter()
    task = ModalityTask(args, modality)
    # setup failures are fatal (nothing to evaluate without data/backbone);
    # they are re-raised for the top-level fatal report in main()
    try:
        task.load_data()
        model = task.load_model()
        task.featurize_all(model)
    except Exception as exc:  # noqa: BLE001 - recorded, never silent
        task.failures["setup"] = f"{type(exc).__name__}: {exc}"
        raise

    for name, fn in (
        ("exp_jevx", lambda: run_jevx(task)),
        ("exp_baselines", lambda: run_baselines(task)),
        ("latency", lambda: run_latency(task)),
    ):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            task.failures[name] = f"{type(exc).__name__}: {exc}"
            print(f"[FAILED] {modality}/{name}: {task.failures[name]}", flush=True)

    total = round(time.perf_counter() - t_start, 2)
    bc.write_json(Path(args.output_dir) / f"{modality}_run_summary.json", {
        "modality": modality,
        "failures": task.failures,
        "context_timings_s": task.timings,
        "total_wall_s": total,
        "invariants": task.inv.summary(),
        "smoke": bool(args.smoke),
    })
    print(f"[{modality}] done in {total}s failures={task.failures}", flush=True)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    modalities = (["vision", "audio"] if args.modality == "all"
                  else [args.modality])
    failures: Dict[str, str] = {}
    for modality in modalities:
        try:
            run_modality(args, modality)
        except Exception as exc:  # noqa: BLE001
            failures[modality] = f"{type(exc).__name__}: {exc}"
            print(f"[FATAL] {modality}: {failures[modality]}", flush=True)
    if failures:
        print(f"completed with fatal failures: {failures}", flush=True)


if __name__ == "__main__":
    main()
