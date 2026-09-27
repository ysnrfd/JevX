# Copyright (c) 2026 ysnrfd
"""WAVE-3a AUDIO diagnostic: the A0-A6 grid on frozen RAVDESS features.

The measured audio gap (G4) is that the SHIPPED stack scores 0.1340-0.2220
accuracy on RAVDESS over frozen ``superb/wav2vec2-base-superb-ks`` features
while a plain linear probe scores 0.3467 and an MLP probe 0.3547 on the SAME
frozen features, splits and seeds. This script asks WHICH of the remaining
degrees of freedom explains the gap, by holding the data, the splits, the
seeds, the optimizer, the lr, the epochs, the batch size and the loss fixed
and varying one thing per cell:

  ====  ==================================================================
  Cell  Definition
  ====  ==================================================================
  A0    reference: the SHIPPED stack (JevX head, interaction ON,
        state_pathway="mlp") - the current default, cell C of the A-F matrix
  A1    ``conditioning_mode="off"`` + state-only head (the P4 bypass
        equivalent): the question/schema pathway is bypassed and the
        option-interaction residual is exactly zero (lambda = 0,
        ``conditioning_enabled=False``, BOTH asserted at run time)
  A2    state path ONLY, as an exact ``nn.Linear(D, K)`` map: the same
        stack with ``state_pathway="linear"`` and the interaction OFF, so
        the head computes EXACTLY ``z_k = s^T W[:, k] + b[k]`` on the fused
        state and the state pathway is the shipped ZERO-initialized
        ``LinearProbePathway``
  A3    A2 with the SAME optimizer/lr/epochs/batch/loss as A1 (isolates the
        optimizer/protocol axis from the architecture axis)
  A4    A2 + probe-style initialization of the state path: the same
        ``nn.Linear(D, K)`` map re-initialized with the scheme the
        linear-probe reference (``bench_common.LinearProbe``) uses, i.e.
        ``nn.Linear``'s own ``reset_parameters`` (kaiming-uniform
        ``a=sqrt(5)`` weight, uniform bias) instead of the shipped zeros
  A5    A2 + 20 epochs (budget sensitivity; wall time reported)
  A6    A2 with ``decision_hidden_size`` HALVED (768 -> 384) (capacity test)
  ====  ==================================================================

Three reference cells run alongside the grid, on the identical features /
splits / seeds, because the A0-A6 grid is meaningless without them: ``R_p1``
(the bare ``nn.Linear(D, K)`` probe on the POOLED features, i.e. the matrix's
P1 - "P1 exact"), ``R_p2`` (the bare MLP probe, matrix P2) and ``R_p4`` (the
matrix's P4 bypass cell, interaction still ON) so the state-only arm is
anchored on both its conditioned and its unconditioned form.

Nothing here is shipped code and nothing under ``jevx/`` is touched: the file
imports the read-only harness helpers from ``benchmarks/run_head_ablation.py``
(never edits them) so the protocol is bit-for-bit the A-F matrix's.

Usage (from the repo root)::

    python benchmarks/run_audio_diagnostic.py --smoke
    python benchmarks/run_audio_diagnostic.py --seeds 0,1,2,3,4

Result file: ``benchmarks/results/audio_diagnostic_ravdess.json``
Cached features: ``benchmarks/cache/head_quality/ravdess_features.npz``
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import bench_common as bc  # noqa: E402
from benchmarks import run_head_quality as hq  # noqa: E402
from benchmarks import run_head_ablation as ha  # noqa: E402

from jevx.decisions import build_decision_head  # noqa: E402
from jevx.decisions.generic import (  # noqa: E402
    LinearProbePathway,
    MLPStatePathway,
)

CACHE_DIR = REPO_ROOT / "benchmarks" / "cache" / "head_quality"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"

#: The SHIPPED option-interaction mode (unchanged from the A-F matrix).
INTERACTION_MODE = ha.INTERACTION_MODE
LAMBDA_ON = ha.LAMBDA_ON
LAMBDA_OFF = ha.LAMBDA_OFF

#: A5's epoch budget: 5x the A-F matrix budget of 4.
BUDGET_CELLS_EPOCHS = 20

#: A6 halves the decision width the A-F matrix pins to the feature width.
CAPACITY_DIVISOR = 2


# --------------------------------------------------------------------------
# The grid
# --------------------------------------------------------------------------


CELLS: Dict[str, Dict[str, Any]] = {
    # -- the diagnostic grid -------------------------------------------------
    "A0_shipped_stack": {
        "kind": "jevx",
        "state_pathway": "mlp",
        "interaction": True,
        "epochs": None,          # the A-F matrix budget (cfg["epochs"])
        "hidden_divisor": 1,
        "state_path_init": "shipped",
        "label": "A0 reference: the SHIPPED stack (JevX head, interaction ON, "
                 "state_pathway=mlp)",
    },
    "A1_state_only_off": {
        "kind": "state_only",
        "state_pathway": "linear",
        "interaction": False,
        "epochs": None,
        "hidden_divisor": 1,
        "state_path_init": "shipped",
        "label": 'A1 conditioning_mode="off" + state-only head (P4 bypass '
                 "equivalent): question/schema pathway bypassed, residual "
                 "exactly 0",
    },
    "A2_state_path_only": {
        "kind": "jevx",
        "state_pathway": "linear",
        "interaction": False,
        "epochs": None,
        "hidden_divisor": 1,
        "state_path_init": "shipped",
        "label": "A2 state path ONLY, an exact nn.Linear(D,K): z_k = "
                 "s^T W[:,k] + b[k] on the fused state, interaction OFF, "
                 "shipped zero init",
    },
    "A3_A2_A1_protocol": {
        "kind": "jevx",
        "state_pathway": "linear",
        "interaction": False,
        "epochs": "a1",           # explicitly A1's optimizer/lr/epochs
        "hidden_divisor": 1,
        "state_path_init": "shipped",
        "label": "A3 = A2 architecture with A1's optimizer / lr / epochs / "
                 "batch / loss (isolates the protocol axis)",
    },
    "A4_probe_init": {
        "kind": "jevx",
        "state_pathway": "linear",
        "interaction": False,
        "epochs": None,
        "hidden_divisor": 1,
        "state_path_init": "probe",   # nn.Linear's own reset_parameters
        "label": "A4 = A2 + probe-style initialization of the state path "
                 "(nn.Linear kaiming-uniform a=sqrt(5) + uniform bias, the "
                 "init bench_common.LinearProbe uses)",
    },
    "A5_budget_20ep": {
        "kind": "jevx",
        "state_pathway": "linear",
        "interaction": False,
        "epochs": BUDGET_CELLS_EPOCHS,
        "hidden_divisor": 1,
        "state_path_init": "shipped",
        "label": "A5 = A2 + 20 epochs (budget sensitivity)",
    },
    "A6_half_width": {
        "kind": "jevx",
        "state_pathway": "linear",
        "interaction": False,
        "epochs": None,
        "hidden_divisor": CAPACITY_DIVISOR,
        "state_path_init": "shipped",
        "label": "A6 = A2 with decision_hidden_size halved (768 -> 384) "
                 "(capacity test)",
    },
    # -- references on the identical features / splits / seeds ----------------
    "R_p1_linear_probe": {
        "kind": "reference",
        "reference": "p1_linear",
        "epochs": None,
        "label": "R_p1 = the matrix's P1: a bare nn.Linear(D,K) on the "
                 "POOLED features ('P1 exact')",
    },
    "R_p2_mlp_probe": {
        "kind": "reference",
        "reference": "p2_mlp",
        "epochs": None,
        "label": "R_p2 = the matrix's P2: a bare MLP(D->D)->ReLU->(D->K) "
                 "probe on the POOLED features",
    },
    "R_p4_bypass_on": {
        "kind": "reference",
        "reference": "p4_state_only_bypass",
        "epochs": None,
        "label": "R_p4 = the matrix's P4: the state-only bypass with the "
                 "option interaction still ON (A1's conditioned counterpart)",
    },
}

#: Canonical execution order: references first, then the grid A0..A6.
ALL_CELLS: Tuple[str, ...] = (
    "R_p1_linear_probe",
    "R_p2_mlp_probe",
    "R_p4_bypass_on",
    "A0_shipped_stack",
    "A1_state_only_off",
    "A2_state_path_only",
    "A3_A2_A1_protocol",
    "A4_probe_init",
    "A5_budget_20ep",
    "A6_half_width",
)

#: Cells that ARE the A0-A6 grid proper (the references are context).
GRID_CELLS: Tuple[str, ...] = tuple(c for c in ALL_CELLS if c.startswith("A"))


# --------------------------------------------------------------------------
# Probe-style initialization (A4)
# --------------------------------------------------------------------------


@torch.no_grad()
def probe_style_init_(pathway: nn.Module) -> Dict[str, Any]:
    """Re-initialize a state pathway with ``nn.Linear``'s own scheme.

    That is the scheme the linear-probe reference
    (:class:`benchmarks.bench_common.LinearProbe`) gets for free from
    ``nn.Linear(in_dim, num_classes)``: kaiming-uniform with ``a=sqrt(5)`` on
    the weight and a uniform bias in ``(-1/sqrt(fan_in), 1/sqrt(fan_in))``.
    The shipped state pathway is ZERO-initialized instead (the zero-init
    residual convention, ``LinearProbePathway.reset_parameters``), so A4
    differs from A2 in EXACTLY the initialization of ``W``/``b`` and in
    nothing else.

    Returns a small record of what was written, so the wiring is auditable.
    """
    if isinstance(pathway, LinearProbePathway):
        weight, bias = pathway.weight, pathway.bias
    elif isinstance(pathway, MLPStatePathway):
        raise ValueError(
            "probe_style_init_: the mlp state pathway is not the A2/A4 map; "
            "this initializer is defined for the exact nn.Linear(D,K) pathway."
        )
    else:
        raise ValueError(
            "probe_style_init_: expected a LinearProbePathway, got "
            f"{type(pathway).__name__}."
        )
    fan_in = int(weight.shape[1])
    bound_w = math.sqrt(3.0) / math.sqrt(fan_in)          # kaiming-uniform a=sqrt(5)
    bound_b = 1.0 / math.sqrt(fan_in)
    before = {
        "weight_absmax": float(weight.detach().abs().max()),
        "bias_absmax": float(bias.detach().abs().max()),
    }
    weight.uniform_(-bound_w, bound_w)
    bias.uniform_(-bound_b, bound_b)
    return {
        "scheme": "nn.Linear.reset_parameters (kaiming_uniform_(a=sqrt(5)) + "
                  "uniform bias)",
        "fan_in": fan_in,
        "weight_bound": bound_w,
        "bias_bound": bound_b,
        "weight_before": before,
        "weight_after_absmax": float(weight.detach().abs().max()),
        "weight_after_std": float(weight.detach().std()),
    }


def assert_state_path_is_zero_init(pathway: nn.Module) -> None:
    """Assert the shipped state pathway really is the zero-initialized map.

    A2's whole claim ("the exact nn.Linear(D,K), shipped zero init") is only
    worth anything if the zeros are actually there, so this is checked rather
    than assumed.
    """
    if isinstance(pathway, LinearProbePathway):
        tensors = {"weight": pathway.weight, "bias": pathway.bias}
    elif isinstance(pathway, MLPStatePathway):
        tensors = {"w2": pathway.w2, "b2": pathway.b2}
    else:
        raise ValueError("unknown state pathway " + repr(type(pathway).__name__))
    for name, tensor in tensors.items():
        if float(tensor.detach().abs().max()) != 0.0:
            raise AssertionError(
                f"state pathway {name} is not zero-initialized "
                f"(absmax={float(tensor.abs().max())}) - the shipped zero-init "
                "residual convention changed; A2/A4 would no longer differ only "
                "in the initialization."
            )


# --------------------------------------------------------------------------
# Model construction
# --------------------------------------------------------------------------


def new_base(cfg: Dict[str, Any], in_dim: int, hidden_divisor: int = 1):
    """A fresh JevX per (width, seed) from the frozen backbone.

    ``ha._new_jevx`` pins decision/question/schema widths to the cached
    feature width; A6 needs the same thing with the width DIVIDED, so the
    three widths are passed explicitly here instead.
    """
    # The seed is the CALLER's contract (``torch.manual_seed(seed)`` immediately
    # before this call), exactly as ``run_head_ablation`` builds its bases, so
    # cells sharing a configuration start from byte-identical weights.
    width = in_dim // int(hidden_divisor)
    model = bc.JevX.from_pretrained(
        cfg["backbone"] or hq.TEXT_BACKBONE,
        freeze_base_model=True,
        pooling_method=cfg["pooling"],
        fusion_type=cfg["fusion"],
        decision_hidden_size=width,
        question_hidden_size=width,
        schema_hidden_size=width,
    )
    model.eval()
    return model


def build_cell(
    cell: str,
    cfg: Dict[str, Any],
    in_dim: int,
    bases: Mapping[str, Any],
    question: str,
    options: Sequence[str],
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Instantiate one grid cell from the shared per-(pathway, width) base.

    Returns ``(module, wiring)``. The wiring records the state-pathway module
    class, its initialization, the interaction mode / scale, the config flag
    AND the head flag, and the decision width, and every A1/A2 "off" claim is
    ASSERTED rather than inherited from a config default.
    """
    spec = CELLS[cell]
    if spec["kind"] == "state_only":
        # A1: the P4 bypass (question/schema pathway bypassed, a fixed
        # state-independent option basis) with the residual EXACTLY zero.
        model = copy.deepcopy(bases[("linear", 1)])
        model.config.state_pathway = "linear"
        model.config.conditioning_enabled = False
        module = hq.JevXStack(
            model, question, options, state_only=True
        )
        module.jevx.heads["choice"].conditioning_enabled = False
        module.jevx.heads["choice"].set_interaction_scale(LAMBDA_OFF)
        wiring: Dict[str, Any] = {
            "kind": "state_only_bypass",
            "decision_hidden_size": int(model.config.decision_hidden_size),
            "state_pathway_built": str(
                getattr(module.jevx.heads["choice"], "state_pathway_name", "?")
            ),
            "interaction_mode": str(
                getattr(module.jevx.heads["choice"], "interaction", "?")
            ),
            "interaction_scale": float(
                module.jevx.heads["choice"].get_interaction_scale
            ),
            "conditioning_enabled_config": bool(
                model.config.conditioning_enabled
            ),
            "conditioning_enabled_head": bool(
                module.jevx.heads["choice"].conditioning_enabled
            ),
            "state_path_init": "shipped",
        }
        assert wiring["interaction_scale"] == LAMBDA_OFF, wiring
        assert wiring["conditioning_enabled_config"] is False, wiring
        assert wiring["conditioning_enabled_head"] is False, wiring
        return module, wiring

    # kind == "jevx": build the choice head through the shipped registry.
    width = in_dim // int(spec["hidden_divisor"])
    model = copy.deepcopy(bases[(spec["state_pathway"], int(spec["hidden_divisor"]))])
    model.heads["choice"] = build_decision_head(
        "choice",
        hidden_size=width,
        interaction=INTERACTION_MODE,
        state_pathway=spec["state_pathway"],
        conditioning_scale_init=LAMBDA_ON,
        conditioning_enabled=True,
    )
    head = model.heads["choice"]
    model.config.state_pathway = head.state_pathway_name
    model.config.state_hidden_size = head.state_hidden_size
    model.config.state_activation = head.state_activation
    model.config.conditioning_enabled = True
    init_record: Dict[str, Any] = {}
    if spec["state_path_init"] == "probe":
        init_record = probe_style_init_(head.state_pathway)
    else:
        assert_state_path_is_zero_init(head.state_pathway)
    if not spec["interaction"]:
        model.config.conditioning_enabled = False
        head.conditioning_enabled = False
        head.set_interaction_scale(LAMBDA_OFF)
    else:
        head.set_interaction_scale(LAMBDA_ON)
    wiring = {
        "kind": "jevx",
        "decision_hidden_size": int(model.config.decision_hidden_size),
        "state_pathway_configured": str(spec["state_pathway"]),
        "state_pathway_built": str(getattr(head, "state_pathway_name", "?")),
        "state_pathway_module": type(head.state_pathway).__name__,
        "state_hidden_size": int(head.state_hidden_size),
        "state_activation": str(head.state_activation),
        "state_path_init": str(spec["state_path_init"]),
        "state_path_init_record": init_record,
        "interaction_mode": INTERACTION_MODE,
        "interaction_scale": float(head.get_interaction_scale),
        "conditioning_enabled_config": bool(model.config.conditioning_enabled),
        "conditioning_enabled_head": bool(head.conditioning_enabled),
    }
    assert wiring["state_pathway_built"] == str(spec["state_pathway"]), wiring
    if not spec["interaction"]:
        assert wiring["interaction_scale"] == LAMBDA_OFF, wiring
        assert wiring["conditioning_enabled_config"] is False, wiring
    feature_model = ha.FeatureJevX(model, question, options)
    return feature_model, wiring


# --------------------------------------------------------------------------
# One cell, one seed
# --------------------------------------------------------------------------


def run_cell(
    cell: str,
    *,
    cfg: Dict[str, Any],
    feats: Mapping[str, np.ndarray],
    labels: Mapping[str, List[int]],
    num_classes: int,
    question: str,
    options: Sequence[str],
    seed: int,
    bases: Mapping[str, Any],
) -> Dict[str, Any]:
    """Train + score ONE cell for ONE seed under the shared protocol."""
    spec = CELLS[cell]
    in_dim = int(feats["train"].shape[1])
    train_records = bc.build_records(feats["train"], labels["train"], question, options)
    cal_records = bc.build_records(feats["cal"], labels["cal"], question, options)
    test_records = bc.build_records(feats["test"], labels["test"], question, options)

    # A3's protocol is A1's. Under the A-F matrix protocol A1 runs at
    # ``cfg["epochs"]`` with the HeadTrainer AdamW default, so "A1's
    # optimizer/lr/epochs" resolves to the same numbers A2 already runs under;
    # it is executed anyway so the identity is measured, not asserted.
    if spec["epochs"] is None:
        epochs = int(cfg["epochs"])
        epoch_source = "a_f_matrix_budget"
    elif spec["epochs"] == "a1":
        epochs = int(cfg["epochs"])
        epoch_source = "a1_optimizer_lr_epochs"
    else:
        epochs = int(spec["epochs"])
        epoch_source = "cell_override"
    lr = cfg["lr"]
    batch_size = cfg["batch_size"]

    torch.manual_seed(seed)
    wiring: Dict[str, Any] = {}
    diagnostics: Dict[str, Any] = {}
    if spec["kind"] == "reference":
        module = ha.build_reference_cell(
            str(spec["reference"]), cfg, feats, num_classes, question, options,
            seed, bases[("linear", 1)],
        )
        view: nn.Module = module
        _, curve, _ = ha.train_flat(
            module, train_records, cal_records, seed=seed, lr=lr,
            epochs=epochs, batch_size=batch_size,
        )
        feature_model = None
    else:
        feature_model, wiring = build_cell(
            cell, cfg, in_dim, bases, question, options
        )
        view = ha.LogitView(feature_model)
        module = None
        _, curve, _ = ha.train_flat(
            view, train_records, cal_records, seed=seed, lr=lr,
            epochs=epochs, batch_size=batch_size,
        )
        if isinstance(feature_model, ha.FeatureJevX):
            # The state-only probe speaks the ``condition_options`` calling
            # convention; only the FeatureJevX wrapper does.
            diagnostics = ha.state_only_diagnostic(
                feature_model, test_records, min(batch_size, 64)
            )
    test_logits, test_labels = bc.collect_logits(
        view, test_records, batch_size=batch_size
    )
    m = bc.metrics_from_logits(test_logits, test_labels)
    params = bc.count_trainable(
        feature_model if feature_model is not None else module
    )
    row: Dict[str, Any] = {
        "seed": seed,
        "cell": cell,
        "accuracy": m["accuracy"],
        "macro_f1": m["macro_f1"],
        "nll": m["nll"],
        "ece": m["ece"],
        "brier": m["brier"],
        "mce": m["mce"],
        "train_loss_first": float(curve[0].get("train_loss", curve[0].get("loss"))),
        "train_loss_last": float(curve[-1].get("train_loss", curve[-1].get("loss"))),
        "val_loss_last": (
            float(curve[-1]["val_loss"]) if "val_loss" in curve[-1] else None
        ),
        "val_accuracy_last": (
            float(curve[-1]["val_accuracy"]) if "val_accuracy" in curve[-1] else None
        ),
        "trainable_params": int(params),
        "protocol": {
            "optimizer": "torch.optim.AdamW (HeadTrainer default)",
            "grad_clip": 1.0,
            "lr": lr,
            "epochs": epochs,
            "epochs_source": epoch_source,
            "batch_size": batch_size,
            "loss": "jevx.training.choice_loss (masked cross-entropy)",
            "splits": "identical frozen features + identical stratified "
                      "splits for every cell",
        },
        "parameter_groups": ha.group_parameter_counts(
            feature_model if feature_model is not None else module
        ),
        "convergence_curve": [dict(h) for h in curve],
        "wiring": wiring,
        "state_only_diagnostic": diagnostics,
    }
    del view, feature_model, module
    return row


# --------------------------------------------------------------------------
# Paired deltas
# --------------------------------------------------------------------------


#: Cells the deltas are reported against.
DELTA_REFERENCES: Tuple[str, ...] = ("A0_shipped_stack", "A2_state_path_only")

DELTA_KEYS: Tuple[str, ...] = (
    "accuracy", "macro_f1", "nll", "ece", "brier", "train_loss_last",
)


def paired_deltas(
    rows_by_cell: Mapping[str, List[Dict[str, Any]]], seeds: Sequence[int],
) -> Dict[str, Any]:
    """Per-seed paired deltas of every cell against A0 and against A2.

    A "paired" delta is cell(seed s) - reference(seed s) on the IDENTICAL test
    split, so the two arms differ only in the thing under test; the aggregate
    reports the mean of the per-seed deltas plus the number of seeds on which
    the delta is positive.
    """
    index: Dict[str, Dict[int, Dict[str, Any]]] = {
        cell: {int(row["seed"]): row for row in rows}
        for cell, rows in rows_by_cell.items()
    }
    out: Dict[str, Any] = {}
    for ref in DELTA_REFERENCES:
        if ref not in index:
            continue
        per_cell: Dict[str, Any] = {}
        for cell, by_seed in index.items():
            if cell == ref:
                continue
            common = [s for s in seeds if s in by_seed and s in index[ref]]
            if not common:
                continue
            entry: Dict[str, Any] = {"paired_seeds": len(common)}
            for key in DELTA_KEYS:
                deltas = [
                    float(by_seed[s][key]) - float(index[ref][s][key])
                    for s in common
                ]
                array = np.asarray(deltas, dtype=np.float64)
                entry[key] = {
                    "mean": float(array.mean()),
                    "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
                    "per_seed": [float(d) for d in deltas],
                    "n_positive": int((array > 0).sum()),
                }
            per_cell[cell] = entry
        out["vs_" + ref] = per_cell
    return out


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    p.add_argument("--dataset", default="ravdess", choices=sorted(hq.DATASETS))
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--cells", nargs="+", default=list(ALL_CELLS))
    p.add_argument("--smoke", action="store_true",
                   help="tiny sizes for pipeline validation (never reportable)")
    p.add_argument("--output-dir", default=str(RESULTS_DIR))
    p.add_argument("--recompute-features", action="store_true")
    return p.parse_args(argv)


def _cache_sha256(name: str) -> Optional[str]:
    path = CACHE_DIR / (name + "_features.npz")
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> Dict[str, Any]:
    name = args.dataset
    cfg = dict(hq.DATASETS[name])
    if args.smoke:
        cfg["train_size"] = min(cfg["train_size"], 64)
        cfg["cal_size"] = min(cfg["cal_size"], 32)
        cfg["test_size"] = min(cfg["test_size"], 48)
        cfg["epochs"] = 2
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    cells = [c for c in ALL_CELLS if c in set(args.cells)]
    t0 = time.perf_counter()
    print("", flush=True)
    print("[=== audio diagnostic: " + name + " ===] protocol from " +
          cfg["source"], flush=True)

    feats, labels, meta, source = hq.build_features(
        name, cfg, recompute=args.recompute_features
    )
    num_classes = len(cfg["options"])
    in_dim = int(feats["train"].shape[1])
    question = cfg["question"]
    options = cfg["options"]

    rows_by_cell: Dict[str, List[Dict[str, Any]]] = {c: [] for c in cells}
    failures: Dict[str, str] = {}
    timings: Dict[str, float] = {}
    for seed in seeds:
        print("  -- seed " + str(seed), flush=True)
        torch.manual_seed(seed)
        # One base per (state pathway, width) per seed, deep-copied per cell:
        # cells sharing a configuration start from byte-identical weights.
        bases: Dict[Tuple[str, int], Any] = {}
        wanted = {(CELLS[c].get("state_pathway"), int(CELLS[c]["hidden_divisor"]))
                  for c in cells if CELLS[c]["kind"] != "reference"}
        wanted.add(("linear", 1))  # the reference cells' P4 base
        for pathway, divisor in sorted(
            wanted, key=lambda kv: (str(kv[0]), kv[1])
        ):
            torch.manual_seed(seed)
            bases[(pathway, divisor)] = new_base(cfg, in_dim, divisor)
        for cell in cells:
            ct = time.perf_counter()
            try:
                row = run_cell(
                    cell, cfg=cfg, feats=feats, labels=labels,
                    num_classes=num_classes, question=question, options=options,
                    seed=seed, bases=bases,
                )
            except Exception as exc:  # noqa: BLE001 - recorded, never silent
                failures[cell + "/seed" + str(seed)] = (
                    type(exc).__name__ + ": " + str(exc)
                )
                print("[FAILED] " + cell + "/" + str(seed) + ": " +
                      failures[cell + "/seed" + str(seed)], flush=True)
                traceback.print_exc()
                continue
            row["wall_s"] = round(time.perf_counter() - ct, 2)
            rows_by_cell[cell].append(row)
            timings.setdefault(cell, 0.0)
            timings[cell] += float(row["wall_s"])
            print("     " + cell.ljust(22) + " acc=" +
                  ("%.4f" % row["accuracy"]) +
                  "  f1=" + ("%.4f" % row["macro_f1"]) +
                  "  nll=" + ("%.4f" % row["nll"]) +
                  "  ep=" + str(row["protocol"]["epochs"]) +
                  "  (" + str(row["wall_s"]) + "s)", flush=True)
        del bases

    results: Dict[str, Any] = {}
    for cell in cells:
        rows = rows_by_cell[cell]
        if not rows:
            continue
        results[cell] = {
            "label": CELLS[cell]["label"],
            "spec": {k: v for k, v in CELLS[cell].items() if k != "label"},
            "per_seed": rows,
            "aggregate": ha.numeric_aggregate(rows),
            "seed_values": {
                key: [float(r[key]) for r in rows]
                for key in ha.METRIC_KEYS if all(key in r for r in rows)
            },
            "parameter_groups": rows[0]["parameter_groups"],
            "state_only_diagnostic": rows[0].get("state_only_diagnostic", {}),
            "wiring": rows[0]["wiring"],
            "wall_s_total": round(timings.get(cell, 0.0), 2),
            "wall_s_per_seed_mean": round(timings.get(cell, 0.0) / len(rows), 2),
        }
        agg = results[cell]["aggregate"]
        print("  " + cell.ljust(22) + " acc=" +
              ("%.4f+-%.4f" % (agg["accuracy"]["mean"], agg["accuracy"]["std"])) +
              "  nll=" + ("%.4f" % agg["nll"]["mean"]) +
              "  ece=" + ("%.4f" % agg["ece"]["mean"]) +
              "  params=" + str(rows[0]["trainable_params"]) +
              "  (" + str(results[cell]["wall_s_per_seed_mean"]) + "s/seed)",
              flush=True)

    ranking = sorted(
        ((c, results[c]["aggregate"]["accuracy"]["mean"]) for c in results),
        key=lambda kv: -kv[1],
    )
    test_n = int(feats["test"].shape[0])
    test_labels_arr = np.asarray(labels["test"], dtype=np.int64)
    majority = float(
        np.bincount(test_labels_arr).max() / max(1, len(test_labels_arr))
    )
    chance = 1.0 / num_classes
    payload: Dict[str, Any] = {
        "experiment": "wave3a_audio_diagnostic_A0_A6",
        "dataset": name,
        "smoke": bool(args.smoke),
        "protocol": {
            "seeds": seeds,
            "n_seeds": len(seeds),
            "cells": cells,
            "grid_cells": [c for c in cells if c in GRID_CELLS],
            "reference_cells": [c for c in cells if c not in GRID_CELLS],
            "epochs": cfg["epochs"],
            "epochs_override": {
                c: CELLS[c]["epochs"] for c in cells
                if CELLS[c]["epochs"] is not None
            },
            "lr": cfg["lr"],
            "batch_size": cfg["batch_size"],
            "optimizer": "torch.optim.AdamW (HeadTrainer default), grad_clip=1.0",
            "loss": "jevx.training.choice_loss (masked cross-entropy)",
            "train_n": int(feats["train"].shape[0]),
            "cal_n": int(feats["cal"].shape[0]),
            "test_n": test_n,
            "feature_dim": in_dim,
            "num_classes": num_classes,
            "question": question,
            "options": list(options),
            "interaction_mode": INTERACTION_MODE,
            "lambda_on": LAMBDA_ON,
            "lambda_off": LAMBDA_OFF,
            "test_majority_baseline": majority,
            "chance": chance,
            "note": (
                "IDENTICAL frozen features, splits, seeds, optimizer, lr, "
                "epochs, batch and loss for every cell; the backbone is frozen "
                "and featurized once (cache reused from the EXP-H runs), so no "
                "backbone is in the loop. Exactly one thing varies per cell. "
                "'conditioning OFF' and 'state-only' are enforced EXPLICITLY "
                "(config.conditioning_enabled=False AND head."
                "conditioning_enabled=False AND set_interaction_scale(0.0), "
                "all asserted at run time), never by relying on a config "
                "default. Nothing under jevx/ is modified by this harness."
            ),
            "cell_definition_note": (
                "A2 is read as the exact nn.Linear(D,K) STATE PATH inside the "
                "shipped stack (state_pathway='linear', interaction OFF, the "
                "rest of the stack trainable), because under the alternative "
                "reading (a bare probe on the pooled features) A2 is R_p1 and "
                "both the A2-vs-A1 and the A4-vs-A2 decision branches are "
                "structurally unreachable. R_p1 IS the bare 'P1 exact' probe "
                "and is reported in the same run so both readings are covered "
                "by executed numbers. A3's optimizer/lr/epochs are A1's, which "
                "under the A-F matrix protocol are the same values A2 already "
                "runs under; A3 is executed anyway so the identity is measured, "
                "not asserted."
            ),
        },
        "features": dict(meta, source=source,
                         cache_sha256=_cache_sha256(name)),
        "cells": results,
        "ranking_by_accuracy": [{"cell": c, "accuracy": a} for c, a in ranking],
        "paired_deltas": paired_deltas(rows_by_cell, seeds),
        "failures": failures,
        "environment": bc.environment_info(),
        "wall_s": round(time.perf_counter() - t0, 2),
    }
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    payload = run(args)
    out_dir = Path(args.output_dir)
    suffix = "_smoke" if args.smoke else ""
    path = out_dir / f"audio_diagnostic_{args.dataset}{suffix}.json"
    bc.write_json(path, payload)
    print("[wrote] " + str(path), flush=True)
    if payload["failures"]:
        print("[FAILURES] " + json.dumps(payload["failures"], indent=2), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
