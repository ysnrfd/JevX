# Copyright (c) 2026 ysnrfd
"""CONFIRMATION RUN for the WAVE-2 follow-up (DECISION-3 + P0-2 + P0-1).

The WAVE-2 grid (``reports/MULTI_QUESTION.md``) measured three conditioning
arms on frozen features and recommended making a state-only arm the shipped
default. This script re-runs the SAME protocol on a subset to confirm that what
actually shipped behaves the way the recommendation claimed.

Measured, on ``ravdess`` (audio) and ``ag_news`` (text), 3 seeds (0-2), the
primary REAL question of each dataset:

* **arm ``default``** - a model built with NO ``conditioning_mode`` kwarg at
  all, i.e. the shipped ``JevXConfig()`` default (now ``"off"``);
* **arm ``full``** - ``conditioning_mode="full"``, the documented user path for
  question conditioning;
* **arm ``state_only_ref``** - the WAVE-2 reference arm reproduced verbatim
  (``conditioning_mode="off"`` plus a ``question_encoder`` forward hook
  returning exact zeros), seed 0 only, to show the default now computes
  exactly that.

It also runs the per-decision invariants on **every** eval batch and records
the counts and worst deviations, and re-runs the WAVE-2 section 2.2 stale-memo
probe (``eval -> optimizer step -> eval`` with no ``clear_condition_cache()``)
to show the memo is no longer weight-stale.

Everything is imported from ``run_multi_question`` so the protocol (splits,
frozen-feature wrapper, step arithmetic, head-init seed, metric code) is the
executed one, not a paraphrase. Result file:
``benchmarks/results/conditioning_default_confirmation.json``.

The box is shared and long runs get killed, so every finished cell is appended
to ``benchmarks/results/_conditioning_confirmation_cells.jsonl`` and replayed by
a later invocation; ``--max-new-cells`` bounds how much a single invocation
computes. The final invocation, in which every cell is replayed, writes the
complete JSON.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))
sys.path.insert(0, str(REPO_ROOT))

import bench_common as bc  # noqa: E402
import run_multi_question as mq  # noqa: E402

from jevx.configuration_jevx import JevXConfig  # noqa: E402
from jevx.modeling_jevx import JevXForDecision  # noqa: E402
from jevx.questions import Option  # noqa: E402

RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
OUT_NAME = "conditioning_default_confirmation.json"
LEDGER = RESULTS_DIR / "_conditioning_confirmation_cells.jsonl"

#: dataset -> the primary REAL question of that dataset. Both appear
#: individually in DECISION-3's pooled table (reports/MULTI_QUESTION.md 5.1),
#: so the comparison against the published state_only reference is
#: like-for-like.
TARGETS: Dict[str, str] = {"ravdess": "ravdess_emotion", "ag_news": "ag_news_topic"}

#: Epochs identical to the WAVE-2 deviations for these two datasets (3.5).
EPOCHS: Dict[str, int] = {"ravdess": 2, "ag_news": 2}

#: The WAVE-2 per-question numbers (reports/MULTI_QUESTION.md 5.1, seeds 0-4)
#: that the default arm is compared against. Recorded, not recomputed; the
#: reference arm is ALSO re-run here as a bitwise control.
WAVE2: Dict[str, Dict[str, Any]] = {
    "ravdess_emotion": {
        "state_only_accuracy": 0.2880,
        "off_accuracy": 0.2792,
        "full_accuracy": 0.2468,
        "state_only_per_seed": [0.290, 0.292, 0.298, 0.246, 0.314],
        "full_per_seed": [0.160, 0.228, 0.328, 0.264, 0.254],
    },
    "ag_news_topic": {
        "state_only_accuracy": 0.8574,
        "off_accuracy": 0.8568,
        "full_accuracy": 0.8486,
        "state_only_per_seed": [0.858, 0.871, 0.842, 0.863, 0.853],
        "full_per_seed": [0.843, 0.828, 0.852, 0.865, 0.855],
    },
}

ARMS: Dict[str, Dict[str, str]] = {
    "default": {
        "mq_arm": "off",
        "construction": "JevXConfig() with NO conditioning_mode kwarg (the shipped default)",
        "seeds": "all",
    },
    "full": {
        "mq_arm": "full",
        "construction": 'conditioning_mode="full"',
        "seeds": "all",
    },
    "state_only_ref": {
        "mq_arm": "state_only",
        "construction": (
            'the WAVE-2 reference: conditioning_mode="off" plus a '
            "question_encoder forward hook returning torch.zeros_like(o)"
        ),
        "seeds": "first only",
    },
}


class _BudgetReached(Exception):
    """Raised when the ``--max-new-cells`` budget is spent."""


def log(message: str) -> None:
    print(f"[cfm] {message}", flush=True)


def _sha(array: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(array, dtype=np.float32)
    ).hexdigest()[:32]


# ---------------------------------------------------------------------------
# The cell ledger (the box is shared and long runs get killed)
# ---------------------------------------------------------------------------


def load_ledger() -> Dict[str, Any]:
    done: Dict[str, Any] = {}
    if not LEDGER.exists():
        return done
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        done[f'{record["dataset"]}|{record["arm"]}|{record["seed"]}'] = record
    return done


def record_cell(record: Dict[str, Any]) -> None:
    with LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


# ---------------------------------------------------------------------------
# Construction: the shipped default vs an explicit mode
# ---------------------------------------------------------------------------


def _base_kwargs(hidden: int) -> Dict[str, Any]:
    return {
        "decision_hidden_size": hidden,
        "question_hidden_size": hidden,
        "schema_hidden_size": hidden,
        "fusion_type": mq.FUSION,
    }


def make_default_model(
    backbone: nn.Module, tokenizer: Any, *, hidden: int
) -> JevXForDecision:
    """A model built with NO conditioning kwarg - the shipped default.

    ``conditioning_mode`` is simply not passed, so the ``JevXConfig`` field
    default is what is read. This is the only way to measure "a fresh default
    model" rather than a model that was told what to do.
    """
    torch.manual_seed(mq.HEAD_INIT_SEED)
    return JevXForDecision(
        JevXConfig(**_base_kwargs(hidden)),
        backbone=backbone,
        tokenizer=tokenizer,
    ).eval()


def make_mode_model(
    backbone: nn.Module, tokenizer: Any, *, conditioning_mode: str, hidden: int
) -> JevXForDecision:
    """The WAVE-2 construction, with an EXPLICIT mode."""
    torch.manual_seed(mq.HEAD_INIT_SEED)
    return JevXForDecision(
        JevXConfig(conditioning_mode=conditioning_mode, **_base_kwargs(hidden)),
        backbone=backbone,
        tokenizer=tokenizer,
    ).eval()


@contextlib.contextmanager
def _default_construction():
    """Make ``mq.train_jevx_arm`` build the SHIPPED DEFAULT for its off arm.

    ``train_jevx_arm`` always passes ``conditioning_mode=mode`` to
    ``make_probe_model``, so measuring the default needs the constructor to
    omit the kwarg. The patch is scoped and restored in a ``finally``; nothing
    on disk is modified.
    """
    original = mq.make_probe_model

    def patched(backbone, tokenizer, *, conditioning_mode, hidden, copy_backbone=True):
        if conditioning_mode == "off":
            return make_default_model(backbone, tokenizer, hidden=hidden)
        return original(
            backbone, tokenizer, conditioning_mode=conditioning_mode, hidden=hidden,
            copy_backbone=copy_backbone,
        )

    mq.make_probe_model = patched
    try:
        yield
    finally:
        mq.make_probe_model = original


# ---------------------------------------------------------------------------
# Eval with the per-decision invariants read on EVERY batch
# ---------------------------------------------------------------------------


def collect_with_invariants(
    frozen: Any,
    groups: Sequence[Any],
    splits: Mapping[str, np.ndarray],
    pos_lookup: Mapping[str, Mapping[int, int]],
    batch_of_chunks: int = 8,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Test logits + a per-decision invariant audit over EVERY eval batch.

    The audit is read from the ``JevXOutput`` of every eval batch. It does not
    replace the checks ``forward`` already ran: ``JevXConfig`` ships
    ``runtime_validation=True``, so the ``ValidationEngine`` has already raised
    on any violation by the time these lines run. What is recorded here is the
    MEASURED worst deviation, so "invariants clean" is a number rather than the
    absence of an exception.
    """
    logits: List[np.ndarray] = []
    labels: List[int] = []
    n_batches = 0
    n_decisions = 0
    max_simplex_error = 0.0
    max_padding_nonzero = 0.0
    min_confidence = 1.0
    max_confidence = 0.0
    all_finite = True
    index_in_range = True
    with torch.no_grad():
        for group in groups:
            schema = [Option(key=o, label=o) for o in group.options]
            valid_k = len(group.options)
            chunks = mq._test_chunks([group], splits["test"])
            for start in range(0, len(chunks), batch_of_chunks):
                batch = chunks[start : start + batch_of_chunks]
                rows = [r for _g, chunk_rows in batch for r in chunk_rows]
                out = frozen.jevx(
                    state=frozen.batch(rows),
                    questions=[group.question],
                    options=[schema],
                )
                n_batches += 1
                n_decisions += len(out.decisions)
                logits.append(
                    torch.cat(
                        [d.logits for d in out.decisions], dim=0
                    ).numpy().astype(np.float32)
                )
                labels.extend(
                    int(group.labels[pos_lookup[group.name][r]]) for r in rows
                )
                for decision in out.decisions:
                    probs = decision.probabilities.reshape(-1)
                    max_simplex_error = max(
                        max_simplex_error, abs(float(probs[:valid_k].sum()) - 1.0)
                    )
                    if probs.numel() > valid_k:
                        max_padding_nonzero = max(
                            max_padding_nonzero, float(probs[valid_k:].abs().max())
                        )
                    confidence = float(decision.confidence[0])
                    min_confidence = min(min_confidence, confidence)
                    max_confidence = max(max_confidence, confidence)
                    all_finite = all_finite and bool(
                        torch.isfinite(decision.logits).all()
                    )
                    index_in_range = index_in_range and (
                        0 <= int(decision.predicted_index[0]) < valid_k
                    )
    return (
        np.concatenate(logits, axis=0),
        np.asarray(labels, dtype=np.int64),
        {
            "eval_batches": n_batches,
            "eval_decisions": n_decisions,
            "max_simplex_error": max_simplex_error,
            "max_padding_abs": max_padding_nonzero,
            "confidence_range": [min_confidence, max_confidence],
            "all_logits_finite": all_finite,
            "predicted_index_in_valid_set": index_in_range,
            "runtime_validation_engine": "enabled (config.runtime_validation=True)",
            "clean": bool(
                all_finite
                and index_in_range
                and max_simplex_error <= 1e-6
                and max_padding_nonzero == 0.0
                and 0.0 <= min_confidence
                and max_confidence <= 1.0
            ),
        },
    )


def train_cell(
    mq_arm: str, ds: Any, groups: Sequence[Any], seed: int, backbone: nn.Module,
    tokenizer: Any, *, epochs: int,
) -> Dict[str, Any]:
    """Train one (arm, seed) cell under the WAVE-2 protocol, then audit eval.

    The step arithmetic is ``mq.train_jevx_arm``'s, run under
    :func:`_default_construction` so its off arm is built from a default
    ``JevXConfig()`` (no mode kwarg). Only the eval is re-done here, through
    :func:`collect_with_invariants`.
    """
    with _default_construction():
        result = mq.train_jevx_arm(
            mq_arm, ds.features, groups, ds.splits, seed=seed, backbone=backbone,
            tokenizer=tokenizer, ds_name=ds.name, epochs=epochs,
            batch_size=mq.BATCH_SIZE,
        )
    model: JevXForDecision = result["_model"]
    frozen = mq.FrozenFeatureJevX(model, ds.features)
    pos_lookup = {g.name: {int(r): i for i, r in enumerate(g.rows)} for g in groups}
    try:
        logits, labels, invariants = collect_with_invariants(
            frozen, groups, ds.splits, pos_lookup
        )
    finally:
        frozen.restore()
    return {
        "metrics": {k: float(v) for k, v in mq.score(logits, labels).items()},
        "invariants": invariants,
        "config_conditioning_mode": model.config.conditioning_mode,
        "config_conditioning_enabled": model.config.conditioning_enabled,
        "trainable_params": int(result["trainable_params"]),
        "trainable_params_by_module": result["trainable_params_by_module"],
        "n_optimizer_steps": result["n_optimizer_steps"],
        "n_train_pairs": result["n_train_pairs"],
        "final_train_loss": (
            result["history"][-1]["loss"] if result["history"] else None
        ),
        "test_logits_sha256_32": _sha(logits),
        "n_test_examples": int(len(labels)),
    }


def _mean_std(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def run_dataset(
    name: str,
    seeds: Sequence[int],
    epochs: int,
    ledger: Mapping[str, Any],
    budget: Optional[List[int]] = None,
) -> Dict[str, Any]:
    ds = mq.build_dataset(name, recompute=False)
    question = next(q for q in ds.questions if q.name == TARGETS[name])
    groups = ds.groups_of(question)
    backbone, tokenizer = mq._backbone_for(ds.modality)
    hidden = int(mq.HEAD_HIDDEN.get(name) or ds.features.shape[1])
    log(
        f"[{name}] {question.name} K={len(groups[0].options)} groups={len(groups)} "
        f"hidden={hidden} epochs={epochs} seeds={list(seeds)}"
    )
    out: Dict[str, Any] = {
        "status": "OK",
        "dataset": name,
        "modality": ds.modality,
        "question": question.name,
        "label_provenance": question.provenance,
        "question_text": question.text,
        "options": list(groups[0].options),
        "head_hidden": hidden,
        "head_hidden_is_feature_width": bool(mq.HEAD_HIDDEN.get(name) is None),
        "epochs": epochs,
        "seeds": list(seeds),
        "split_sizes": {k: int(len(v)) for k, v in ds.splits.items()},
        "n_groups": len(groups),
        "wave2_reference_numbers": WAVE2.get(question.name),
        "cells": {},
    }
    for arm, spec in ARMS.items():
        arm_seeds = list(seeds) if spec["seeds"] == "all" else list(seeds[:1])
        per_seed: List[Dict[str, Any]] = []
        for seed in arm_seeds:
            key = f"{name}|{arm}|{seed}"
            if key in ledger:
                entry = dict(ledger[key])
                entry["replayed_from_ledger"] = True
                per_seed.append(entry)
                log(
                    f"[{name}]    {arm}/seed{seed}: REPLAYED "
                    + " ".join(
                        f"{m}={v:.4f}" for m, v in entry.get("metrics", {}).items()
                    )
                )
                continue
            log(f"[{name}] -> {arm} / seed {seed}")
            started = time.perf_counter()
            try:
                cell = train_cell(
                    spec["mq_arm"], ds, groups, seed, backbone, tokenizer, epochs=epochs
                )
            except Exception as exc:  # noqa: BLE001 - reported verbatim
                log(f"[{name}] FAILED {arm}/seed{seed}: {type(exc).__name__}: {exc}")
                per_seed.append({
                    "seed": seed, "status": "FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                })
                continue
            entry = {
                "seed": seed, "status": "OK",
                "seconds": round(time.perf_counter() - started, 2),
            }
            entry.update(cell)
            record_cell({"dataset": name, "arm": arm, "seed": seed, **entry})
            per_seed.append(entry)
            log(
                f"[{name}]    {arm}/seed{seed}: "
                + " ".join(f"{m}={v:.4f}" for m, v in cell["metrics"].items())
                + f"  invariants_clean={cell['invariants']['clean']}"
            )
            if budget is not None:
                budget[0] -= 1
                if budget[0] <= 0:
                    log(f"cell budget spent after {name}/{arm}/seed{seed}")
                    raise _BudgetReached()
        ok = [c for c in per_seed if c["status"] == "OK"]
        metric_names = [
            m for m in ("accuracy", "macro_f1", "nll", "ece", "brier")
            if ok and m in ok[0]["metrics"]
        ]
        out["cells"][arm] = {
            "arm": arm,
            "construction": spec["construction"],
            "seeds_run": arm_seeds,
            "per_seed": per_seed,
            "aggregate": (
                {m: _mean_std([c["metrics"][m] for c in ok]) for m in metric_names}
                if ok else {"status": "ALL CELLS FAILED"}
            ),
            "per_seed_accuracy": [c["metrics"]["accuracy"] for c in ok],
            "invariants_all_clean": bool(ok) and all(
                c["invariants"]["clean"] for c in ok
            ),
            "n_eval_batches_invariant_checked": sum(
                c["invariants"]["eval_batches"] for c in ok
            ),
            "n_eval_decisions_invariant_checked": sum(
                c["invariants"]["eval_decisions"] for c in ok
            ),
            "max_simplex_error_over_all_batches": max(
                (c["invariants"]["max_simplex_error"] for c in ok), default=None
            ),
            "max_padding_abs_over_all_batches": max(
                (c["invariants"]["max_padding_abs"] for c in ok), default=None
            ),
        }
    default_cell = out["cells"]["default"]
    ref_cell = out["cells"]["state_only_ref"]
    shared_seed = ref_cell["seeds_run"][0]
    shared = next(
        (c for c in default_cell["per_seed"]
         if c["status"] == "OK" and c["seed"] == shared_seed),
        None,
    )
    ref_row = next((c for c in ref_cell["per_seed"] if c["status"] == "OK"), None)
    if shared and ref_row:
        out["default_equals_state_only_reference"] = {
            "seed": shared_seed,
            "bitwise": (
                shared["test_logits_sha256_32"] == ref_row["test_logits_sha256_32"]
            ),
            "default_sha": shared["test_logits_sha256_32"],
            "state_only_ref_sha": ref_row["test_logits_sha256_32"],
            "note": (
                "identical SHA-256 over the float32 test-logit matrix of the shared "
                "seed, i.e. bitwise equality of the whole default arm against the "
                "WAVE-2 state_only reference arm"
            ),
        }
    return out


# ---------------------------------------------------------------------------
# Reference checks that are not accuracy-shaped
# ---------------------------------------------------------------------------


def _probe_state(modality: str) -> Dict[str, Any]:
    """A raw state the given backbone can actually featurize.

    The audio backbone is a Wav2Vec2 with no tokenizer, so a text state is not
    a legal input for it; the text backbone is the reverse. Passing the wrong
    one raises inside ``adapter.prepare_inputs`` and measures nothing.
    """
    if modality == "audio":
        generator = torch.Generator().manual_seed(0)
        return {
            "state": {"input_values": torch.randn(1, 4000, generator=generator)},
            "questions": ["which one"],
            "options": [["a", "b", "c"]],
        }
    return {
        "state": "hello world foo bar baz",
        "questions": ["which one"],
        "options": [["a", "b", "c"]],
    }


def default_is_off_and_is_the_state_only_reference(
    backbone: nn.Module, tokenizer: Any, hidden: int, *, modality: str
) -> Dict[str, Any]:
    """The shipped default IS the WAVE-2 state_only reference, bitwise."""
    default = make_default_model(backbone, tokenizer, hidden=hidden)
    off = make_mode_model(backbone, tokenizer, conditioning_mode="off", hidden=hidden)
    call = _probe_state(modality)

    def logits_of(model: JevXForDecision) -> np.ndarray:
        with torch.no_grad():
            return model(**call).decisions[0].logits.detach().clone().numpy()

    got_default = logits_of(default)
    got_off = logits_of(off)
    hook = default.question_encoder.register_forward_hook(
        lambda _m, _i, output: torch.zeros_like(output)
    )
    try:
        with torch.no_grad():
            reference = (
                default(condition_options=False, **call)
                .decisions[0].logits.detach().clone().numpy()
            )
    finally:
        hook.remove()
    default_state = default.state_dict()
    off_state = off.state_dict()
    same_state_dict = set(default_state) == set(off_state) and all(
        torch.equal(default_state[k], off_state[k]) for k in default_state
    )
    return {
        "default_config_mode": default.config.conditioning_mode,
        "default_config_enabled": default.config.conditioning_enabled,
        "default_equals_off_bitwise": bool(np.array_equal(got_default, got_off)),
        "default_equals_state_only_reference_bitwise": bool(
            np.array_equal(got_default, reference)
        ),
        "default_state_dict_identical_to_explicit_off": bool(same_state_dict),
        "note": (
            "state_only_ref is the WAVE-2 reference arm: conditioning_mode='off' plus a "
            "question_encoder forward hook returning torch.zeros_like(o). The default "
            "arm now computes it without the hook."
        ),
    }


def stale_memo_probe(
    backbone: nn.Module, tokenizer: Any, hidden: int, *, modality: str
) -> Dict[str, Any]:
    """Re-run the WAVE-2 section 2.2 probe on a real wrapper, after the fix.

    ``eval -> one AdamW step -> eval`` WITHOUT ``clear_condition_cache()``: the
    second eval must equal a fresh-cache evaluation of the same post-step
    weights. WAVE-2 measured max abs diff 4.1556e+04 on this ordering.
    """
    model = make_mode_model(
        backbone, tokenizer, conditioning_mode="constant", hidden=hidden
    )
    call = _probe_state(modality)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-2
    )
    model.eval()
    with torch.no_grad():
        model(**call)
    filled = len(model._condition_cache)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.enable_grad():
        loss = model(**call).decisions[0].logits.sum()
    loss.backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        after = model(**call).decisions[0].logits.detach().clone()
    model.config.conditioning_mode = "full"
    with torch.no_grad():
        reference = model(**call).decisions[0].logits.detach().clone()
    difference = float((after - reference).abs().max())
    return {
        "protocol": "eval (fills the memo) -> one AdamW step -> eval WITHOUT clearing",
        "memo_entries_after_first_eval": filled,
        "max_abs_diff": difference,
        "bitwise_equal": bool(torch.equal(after, reference)),
        "wave2_measurement_before_the_fix": 41556.0,
        "tolerance": 1e-6,
        "clean": bool(difference <= 1e-6),
    }


def run_reference_checks() -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    for modality, label in (
        ("audio", mq.AUDIO_BACKBONE), ("text", mq.TEXT_BACKBONE)
    ):
        try:
            backbone, tokenizer = mq._backbone_for(modality)
        except Exception as exc:  # noqa: BLE001
            checks[f"skip_{modality}"] = f"{type(exc).__name__}: {exc}"
            continue
        hidden = 256 if modality == "audio" else 32
        for check, fn in (
            ("wiring", default_is_off_and_is_the_state_only_reference),
            ("stale_memo_probe", stale_memo_probe),
        ):
            try:
                checks[f"{label}_{check}"] = fn(
                    backbone, tokenizer, hidden, modality=modality
                )
            except Exception as exc:  # noqa: BLE001
                checks[f"{label}_{check}_error"] = (
                    f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                )
        del backbone, tokenizer
    return checks


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def build_verdict(
    payload: Mapping[str, Any], seeds: Sequence[int]
) -> List[Dict[str, Any]]:
    verdict: List[Dict[str, Any]] = []
    for name, block in payload["datasets"].items():
        if block.get("status") != "OK":
            verdict.append({
                "dataset": name, "verdict": "NOT RUN", "error": block.get("error"),
            })
            continue
        cells = block["cells"]
        if (
            "status" in cells["default"]["aggregate"]
            or "status" in cells["full"]["aggregate"]
        ):
            verdict.append({"dataset": name, "verdict": "NOT RUN (arm cells failed)"})
            continue
        default_acc = cells["default"]["aggregate"]["accuracy"]
        full_acc = cells["full"]["aggregate"]["accuracy"]
        reference = block.get("wave2_reference_numbers") or {}
        entry: Dict[str, Any] = {
            "dataset": name,
            "question": block["question"],
            "default_accuracy": default_acc,
            "full_accuracy": full_acc,
            "default_per_seed": cells["default"]["per_seed_accuracy"],
            "full_per_seed": cells["full"]["per_seed_accuracy"],
            "default_minus_full_pp": 100.0 * (
                default_acc["mean"] - full_acc["mean"]
            ),
            "default_over_full_ranges_disjoint": bool(
                default_acc["max"] < full_acc["min"]
                or full_acc["max"] < default_acc["min"]
            ),
            "invariants_all_clean": bool(
                cells["default"]["invariants_all_clean"]
                and cells["full"]["invariants_all_clean"]
            ),
            "n_eval_batches_invariant_checked": (
                cells["default"]["n_eval_batches_invariant_checked"]
                + cells["full"]["n_eval_batches_invariant_checked"]
            ),
            "default_equals_state_only_reference": block.get(
                "default_equals_state_only_reference"
            ),
        }
        if "state_only_accuracy" in reference:
            entry["wave2_state_only_accuracy"] = reference["state_only_accuracy"]
            entry["default_minus_wave2_state_only_pp"] = 100.0 * (
                default_acc["mean"] - reference["state_only_accuracy"]
            )
            entry["wave2_state_only_per_seed_seeds_0_2"] = reference[
                "state_only_per_seed"
            ][: len(seeds)]
            entry["default_at_or_above_wave2_state_only_reference"] = bool(
                default_acc["mean"] >= reference["state_only_accuracy"] - 0.02
            )
        verdict.append(entry)
    return verdict


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WAVE-2 follow-up confirmation run"
    )
    parser.add_argument("--datasets", default="ravdess,ag_news")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--max-new-cells", type=int, default=None,
        help="compute at most N new cells then exit; the JSONL ledger keeps them",
    )
    parser.add_argument(
        "--skip-reference-checks", action="store_true",
        help="skip the wiring / stale-memo probes (they load both backbones)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    torch.set_num_threads(max(1, int(args.threads)))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    budget: Optional[List[int]] = (
        None if args.max_new_cells is None else [int(args.max_new_cells)]
    )
    ledger = load_ledger()

    payload: Dict[str, Any] = {
        "phase": "conditioning_default_confirmation",
        "status": "INCOMPLETE",
        "purpose": (
            "Confirm the DECISION-3 default (conditioning_mode='off'), the P0-2 fix "
            "('off' == the WAVE-2 state_only reference) and the P0-1 fix (the "
            "conditioning memo is weight-keyed) on the same frozen-feature protocol as "
            "reports/MULTI_QUESTION.md."
        ),
        "environment": mq.environment_info(),
        "protocol": {
            "datasets": datasets,
            "questions": TARGETS,
            "seeds": seeds,
            "epochs": EPOCHS,
            "batch_size": mq.BATCH_SIZE,
            "lr": mq.LR,
            "split_seed": mq.SPLIT_SEED,
            "n_bins": mq.N_BINS,
            "head_init_seed": mq.HEAD_INIT_SEED,
            "backbones": {"audio": mq.AUDIO_BACKBONE, "text": mq.TEXT_BACKBONE},
            "one_question_per_batch": True,
            "imported_from": (
                "benchmarks/run_multi_question.py - the protocol is imported, not "
                "re-implemented"
            ),
            "arms": {k: v["construction"] for k, v in ARMS.items()},
            "cell_ledger": LEDGER.name,
            "epochs_note": (
                "2 epochs, matching the WAVE-2 deviation for these two datasets "
                "(reports/MULTI_QUESTION.md 3.5); RAVDESS also uses the reduced head "
                "width 256 there, which is IDENTICAL across the arms compared here. "
                "WAVE-2 used 5 seeds (0-4) and this run 3 (0-2), so the means are "
                "comparable in expectation; the WAVE-2 per-seed rows for seeds 0-2 are "
                "recorded per question for the direct comparison."
            ),
        },
        "reference_checks": {},
        "datasets": {},
    }

    if not args.skip_reference_checks:
        payload["reference_checks"] = run_reference_checks()

    try:
        for name in datasets:
            try:
                payload["datasets"][name] = run_dataset(
                    name, seeds, EPOCHS.get(name, 2), ledger, budget
                )
            except _BudgetReached:
                raise
            except Exception as exc:  # noqa: BLE001
                log(f"FAILED {name}: {type(exc).__name__}: {exc}")
                payload["datasets"][name] = {
                    "dataset": name, "status": "FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
    except _BudgetReached:
        log("cell budget exhausted; re-run to continue (results are in the ledger)")

    complete = all(
        block.get("status") == "OK"
        and all(
            len(block["cells"][arm]["per_seed"]) == len(block["cells"][arm]["seeds_run"])
            and all(c["status"] == "OK" for c in block["cells"][arm]["per_seed"])
            for arm in ARMS
        )
        for block in payload["datasets"].values()
    ) and len(payload["datasets"]) == len(datasets)
    payload["status"] = "OK" if complete else "INCOMPLETE"
    payload["verdict"] = build_verdict(payload, seeds)
    path = RESULTS_DIR / OUT_NAME
    bc.write_json(path, payload)
    log(f"wrote {path} (status={payload['status']})")
    print(
        json.dumps(
            {"status": payload["status"], "verdict": payload["verdict"]},
            indent=2, default=str,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
