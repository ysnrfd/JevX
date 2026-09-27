# Copyright (c) 2026 ysnrfd
"""WAVE-2 MULTI-QUESTION benchmark + conditioning-cache cost measurement.

This is the FIRST multi-question evaluation in the repository. Every other
benchmark under ``benchmarks/`` asks ONE question per state, which is the
regime in which the question/schema encoders were already measured at
50.6-55.5 % of the trainable parameters and 18.3-59.3x P2's latency for no
measured benefit. The encoders exist for the MULTI-question regime
(``JevXForDecision.forward`` answers m questions about B states in one call)
and that regime had never been run. This script runs it, and it also measures
what ``JevXConfig.conditioning_mode="constant"`` costs.

Phases, each independently runnable::

    python benchmarks/run_multi_question.py --phase selfcheck
    python benchmarks/run_multi_question.py --phase cache-cost
    python benchmarks/run_multi_question.py --phase shared-state
    python benchmarks/run_multi_question.py --phase grid --datasets absa
    python benchmarks/run_multi_question.py --phase all

WHAT THIS SCRIPT MEASURES, and how:

* PART 1 ``cache-cost`` - bitwise equality of ``"constant"`` vs ``"full"``
  (cold call, warm repeated call, after a content change) plus batch-1
  latency of four arms on FROZEN features: state-only (``"off"``),
  ``"constant"`` cache HIT, ``"constant"`` cache MISS, ``"full"``.
  Program success criterion: ``constant``-HIT <= 1.15x the state-only
  forward. The measured ratio is reported either way.
* PART 2 ``grid`` - the multi-question suite. Four datasets, stratified
  train/cal/test splits (seed 0), the repository's established
  FROZEN-PROTOCOL backbone per modality, featurize ONCE, then a 5-seed grid
  over ``full`` / ``constant`` / ``off`` / ``state_only`` plus the P1/P2 probe
  references on the same frozen features.
* PART 3 ``shared-state`` - with 4+ questions on ONE state, is the
  shared-state property still exactly ONE backbone forward? Counted with a
  real ``register_forward_hook`` on the backbone.

LABEL PROVENANCE is recorded per question and is never blurred:

  ``REAL``     the label is provided by the dataset - an annotation or a field
               the dataset ships. ABSA aspect polarity, the RAVDESS filename
               fields, the ag_news topic and the Yelp star rating are REAL.
  ``DERIVED``  the label is a DETERMINISTIC RULE this script applies to the
               data (e.g. "is this a 1-star detractor" = ``stars == 1``). A
               derived label is never presented as an annotation.

PROTOCOL NOTE (one question per call). ``JevXForDecision.forward`` shares one
question across the whole state batch, so a batch here is a set of states that
all ask the SAME question (the schema is constant within a batch and every
option count is uniform, so no padded option column is ever produced). For
ABSA, whose question text is templated on the aspect term, states are
therefore bucketed by their question text. ``--phase shared-state`` shows the
genuine m-questions-in-one-call forward produces per-question logits that are
BITWISE EQUAL to the per-question calls, so this protocol is the same
computation, not an approximation.

Nothing under ``jevx/`` is touched. The conditioning stack is driven through
the real ``JevXForDecision.forward``; only the backbone's
``extract_representation`` is replaced by a feature-table lookup (the
"featurize once" half of the established Mode-A protocol), and that
replacement is asserted against a real backbone pass in ``--phase selfcheck``.

Result files::

    benchmarks/results/multi_question_selfcheck.json
    benchmarks/results/multi_question_cache_cost.json
    benchmarks/results/multi_question_<dataset>.json
    benchmarks/results/multi_question_shared_state.json
    benchmarks/results/multi_question_summary.json

Feature cache::

    benchmarks/cache/multi_question/<dataset>_features.npz
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import sys
import time
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import bench_common as bc  # noqa: E402
from benchmarks import run_modality as rm  # noqa: E402

from jevx.adapters.base import Representation  # noqa: E402
from jevx.configuration_jevx import JevXConfig  # noqa: E402
from jevx.modeling_jevx import JevXForDecision  # noqa: E402
from jevx.questions import Option  # noqa: E402
from jevx.training import HeadTrainer  # noqa: E402
from jevx.training.losses import choice_loss  # noqa: E402

RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
CACHE_DIR = REPO_ROOT / "benchmarks" / "cache" / "multi_question"

# --------------------------------------------------------------------------
# Protocol constants (the established frozen-protocol benchmarks' values)
# --------------------------------------------------------------------------

TEXT_BACKBONE = "google/bert_uncased_L-2_H-128_A-2"
AUDIO_BACKBONE = "superb/wav2vec2-base-superb-ks"
POOLING = "cls"
FUSION = "concat"
MAX_LENGTH = 128
EPOCHS = 4
LR = 1e-3
BATCH_SIZE = 64
SPLIT_SEED = 0
N_BINS = 15
HEAD_INIT_SEED = 1234

#: Split caps from the benchmark contract (train, cal, test); ``None`` = all.
#: The contract states these as UPPER BOUNDS, so ag_news and Yelp run at
#: 1500/500/1000 rather than 3000/1000/1500: at 3000 the topic question alone
#: costs ~170 s per (arm, seed) on a 4-core box shared with other agents, and
#: 4 questions x 6 arms x 5 seeds did not fit this wave's wall-clock budget.
#: ABSA and RAVDESS keep the established sizes (RAVDESS 700/240/500).
SPLIT_CAPS: Dict[str, Tuple[Optional[int], Optional[int], Optional[int]]] = {
    "absa": (None, None, None),
    "ravdess": (700, 240, 500),
    "ag_news": (1500, 500, 1000),
    "yelp": (1500, 500, 1000),
}

#: POOL caps: the split is carved from a deterministic stratified subsample of
#: this size, chosen BEFORE featurization so the frozen backbone only ever sees
#: the states the benchmark can actually use. ag_news and Yelp ship 120k and
#: 650k training rows; featurizing all of them would take hours and change no
#: number, because the split can use at most train+cal+test rows. The subsample
#: is ``bc.stratified_indices(labels, cap, seed=SPLIT_SEED)`` - a pure function
#: of (labels, cap, seed) - so the featurize path and the label path always pick
#: the SAME states.
POOL_CAPS: Dict[str, Optional[int]] = {
    "absa": None,
    "ravdess": None,
    "ag_news": 6000,
    "yelp": 6000,
}

#: Width of the JevX decision stack, per dataset. ``None`` = the frozen
#: feature width, which is the established protocol for both modalities (the
#: text backbone is 128-wide, so every text dataset already runs at its
#: feature width).
#:
#: DEVIATION - RAVDESS runs the head at 256, not at the 768 feature width.
#: Reason, measured: one ``(arm, seed)`` cell at width 768 costs ~270 s on a
#: 4-core box shared with other agents, so the full RAVDESS grid
#: (4 questions x 4 JevX arms x 5 seeds) did not fit this wave's wall-clock
#: budget. The width is IDENTICAL across all four JevX arms on RAVDESS, so the
#: within-dataset full-vs-off-vs-state_only comparison is unaffected. What this
#: does NOT establish is the audio head at the full 768 width; see the report.
HEAD_HIDDEN: Dict[str, Optional[int]] = {
    "absa": None,
    "ravdess": 256,
    "ag_news": None,
    "yelp": None,
}

#: ``full``/``constant``/``off`` are the three shipped
#: ``JevXConfig.conditioning_mode`` values. ``state_only`` is an ADDITIONAL
#: reference this script adds, because ``"off"`` zeroes the OPTION-interaction
#: residual but still feeds the QUESTION vector into ``ConcatFusion``'s
#: ``W_cat [s ; q]`` (``modeling_jevx.py`` always passes
#: ``question_rows[indices]`` to the fusion; only ``option_features`` is gated
#: by ``condition``). ``state_only`` therefore also replaces the question rows
#: with exact zeros, which is the true "the state alone decides" arm. It is
#: NOT a shipped mode and is labelled as a reference everywhere.
ARMS: Tuple[str, ...] = ("full", "constant", "off", "state_only", "p1_linear", "p2_mlp")
JEVX_ARMS: Tuple[str, ...] = ("full", "constant", "off", "state_only")
PROBE_ARMS: Tuple[str, ...] = ("p1_linear", "p2_mlp")
CACHE_HIT_TOLERANCE = 1.15


def log(msg: str) -> None:
    print(f"[mq] {msg}", flush=True)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    bc.write_json(path, dict(payload))


# --------------------------------------------------------------------------
# Frozen-feature wrapper
# --------------------------------------------------------------------------

class FrozenFeatureJevX(nn.Module):
    """A ``JevXForDecision`` whose backbone reads a precomputed feature table.

    The repository's Mode-A protocol is "featurize once, then run the head
    stack on precomputed pooled features". ``JevXForDecision.forward`` always
    calls the backbone, so this swaps ONLY the adapter's
    ``extract_representation`` for a lookup. The question encoder, the schema
    encoder, ``_resolve_condition``, the ``"constant"`` memo, the fusion, the
    option-interaction residual, the choice head and the duplicate tie are all
    the shipped code, unmodified.

    The row index travels in ``input_ids[:, 0]`` (offset by ``INDEX_OFFSET``),
    so the lookup is correct under any batch composition; the backbone is
    never called, so those token ids only have to be valid integers.
    """

    INDEX_OFFSET = 100_000

    def __init__(self, model: JevXForDecision, features: np.ndarray) -> None:
        super().__init__()
        self.jevx = model
        self.features = np.ascontiguousarray(np.asarray(features, dtype=np.float32))
        self._table = torch.from_numpy(self.features)
        self.backbone_forward_calls = 0
        adapter = model.adapter
        self._real_extract = adapter.extract_representation
        table = self._table

        def frozen_extract(
            _model: Any,
            inputs: Mapping[str, Any],
            *,
            strategy: Optional[str] = None,
            output_hidden_states: bool = False,
            mode: Optional[str] = None,
        ) -> Representation:
            self.backbone_forward_calls += 1
            marker = next(v for v in inputs.values() if isinstance(v, torch.Tensor))
            rows = marker[:, 0].to(torch.long) - self.INDEX_OFFSET
            if bool((rows < 0).any()) or bool((rows >= table.shape[0]).any()):
                raise IndexError(
                    "FrozenFeatureJevX: the state tensor must carry row indices in "
                    f"[{self.INDEX_OFFSET}, {self.INDEX_OFFSET + table.shape[0]}), got "
                    f"min={int(rows.min())} max={int(rows.max())}."
                )
            return Representation(
                sequence_output=None,
                pooled=table.index_select(0, rows),
                attention_mask=None,
                raw_output=None,
                field="frozen_feature_table",
                layout="bd",
            )

        adapter.extract_representation = frozen_extract  # type: ignore[method-assign]
        #: The adapter validates the state dict against the BACKBONE's declared
        #: inputs, so the row-index marker has to travel under the key the
        #: backbone actually declares (``input_ids`` for text,
        #: ``input_values`` for wav2vec2).
        self.main_input_name = str(
            getattr(model.model, "main_input_name", None)
            or getattr(getattr(model.model, "config", None), "main_input_name", "")
            or "input_ids"
        )

    def restore(self) -> None:
        self.jevx.adapter.extract_representation = self._real_extract

    def batch(self, rows: Sequence[int]) -> Dict[str, torch.Tensor]:
        ids = torch.tensor(
            [[self.INDEX_OFFSET + int(i)] for i in rows], dtype=torch.long
        )
        if self.main_input_name == "input_ids":
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        return {self.main_input_name: ids}


# --------------------------------------------------------------------------
# PART 1 - the ``constant`` conditioning-cache cost
# --------------------------------------------------------------------------

#: ``tests/conftest.py``'s ``TINY_VOCAB``, reproduced verbatim (the fixture is
#: pytest-scoped and this file is not a pytest module).
TINY_VOCAB: Dict[str, int] = {
    "[UNK]": 0, "[PAD]": 1, "hello": 2, "world": 3, "foo": 4, "bar": 5,
    "baz": 6, "the": 7, "cat": 8, "sat": 9, "a": 10, "b": 11, "c": 12,
    "question": 13, "option": 14,
}

CACHE_QUESTIONS_REAL = (
    "what is the sentiment of this review",
    "which aspect carries the negative polarity",
    "what is the emotion of this recording",
    "is this a one star detractor review",
)
CACHE_OPTIONS_REAL = ("negative", "neutral", "positive")
CACHE_QUESTIONS_TINY = ("a question", "b question", "the cat", "the bar")
CACHE_OPTIONS_TINY = ("foo", "bar", "baz")


def build_tiny_backbone_and_tokenizer() -> Tuple[nn.Module, Any]:
    """``tests/conftest.py``'s ``tiny_bert`` + ``tiny_tokenizer``, rebuilt."""
    from tokenizers import Tokenizer, models as tk_models, pre_tokenizers
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    backend = Tokenizer(tk_models.WordLevel(dict(TINY_VOCAB), unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]"
    )
    config = BertConfig(
        vocab_size=99, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=64, max_position_embeddings=64,
    )
    torch.manual_seed(0)
    return BertModel(config).eval(), tokenizer


def hidden_of(backbone: nn.Module) -> int:
    return int(getattr(backbone.config, "hidden_size", 128))


def make_probe_model(
    backbone: nn.Module, tokenizer: Any, *, conditioning_mode: str, hidden: int,
    copy_backbone: bool = True,
) -> JevXForDecision:
    """A small JevX wrapper over the backbone.

    ``HEAD_INIT_SEED`` is set first, so every arm of the grid starts from
    BITWISE-IDENTICAL head-stack weights: the arms are paired by construction.

    ``copy_backbone=False`` shares ONE frozen backbone instance across arms.
    The grid runs under the frozen-feature protocol, where the backbone is
    never executed (its ``extract_representation`` is replaced by a table
    lookup), so a per-arm copy of a 94M-parameter wav2vec2 would cost minutes
    of deepcopy per seed and change no number. It is only used for the grid;
    the cache-cost and self-check phases copy, because those call the backbone.
    """
    torch.manual_seed(HEAD_INIT_SEED)
    return JevXForDecision(
        JevXConfig(
            decision_hidden_size=hidden,
            question_hidden_size=hidden,
            schema_hidden_size=hidden,
            conditioning_mode=conditioning_mode,
            fusion_type=FUSION,
        ),
        backbone=copy.deepcopy(backbone) if copy_backbone else backbone,
        tokenizer=tokenizer,
    ).eval()


def _bitwise_checks(
    model: JevXForDecision, questions: Sequence[str], options: Sequence[str]
) -> Dict[str, Any]:
    """cold / warm / after-content-change equality of ``constant`` vs ``full``."""
    call = {"state": "hello world", "questions": [questions[0]],
            "options": [list(options)]}
    changed_opts = list(options[:-1]) + [
        "cat" if "cat" not in options else "sat"
    ]
    changed = {"state": "hello world", "questions": [questions[0]],
               "options": [changed_opts]}
    with torch.no_grad():
        model.config.conditioning_mode = "full"
        full_cold = model(**call)
        full_changed = model(**changed)
        model.clear_condition_cache()
        model.config.conditioning_mode = "constant"
        cold = model(**call)
        warm = model(**call)
        after_change = model(**changed)
        model.clear_condition_cache()
        model.config.conditioning_mode = "full"
    a, b = cold.decisions[0].logits, full_cold.decisions[0].logits
    c, d = after_change.decisions[0].logits, full_changed.decisions[0].logits
    return {
        "cold_equals_full": bool(torch.equal(a, b)),
        "warm_equals_full": bool(torch.equal(warm.decisions[0].logits, b)),
        "after_content_change_equals_full": bool(torch.equal(c, d)),
        "content_change_actually_moved_the_logits": bool(not torch.equal(c, a)),
        "cold_max_abs_diff": float((a - b).abs().max()),
        "warm_max_abs_diff": float((warm.decisions[0].logits - b).abs().max()),
        "after_content_change_max_abs_diff": float((c - d).abs().max()),
    }


def measure_cache_cost(
    backbone: nn.Module,
    tokenizer: Any,
    *,
    tag: str,
    questions: Sequence[str],
    options: Sequence[str],
    warmup: int = 20,
    iters: int = 120,
) -> Dict[str, Any]:
    """Batch-1 latency of the four conditioning arms on FROZEN features.

    The arms are timed ROUND-ROBIN inside one loop (``off``, ``hit``, ``full``,
    ``miss``, ``off``, ...) instead of one block each: the machine is shared
    with other agents, so a single block per arm would let a load spike land on
    one arm only. Round-robin makes the contention noise common-mode, and the
    MEDIAN (not the mean) is what is reported.
    """
    width = hidden_of(backbone)
    m_questions = len(questions)
    rs = np.random.RandomState(20260924)
    features = rs.randn(4, width).astype(np.float32)

    off_model = make_probe_model(backbone, tokenizer, conditioning_mode="off", hidden=width)
    const_model = make_probe_model(backbone, tokenizer, conditioning_mode="constant", hidden=width)
    full_model = make_probe_model(backbone, tokenizer, conditioning_mode="full", hidden=width)
    frozen_off = FrozenFeatureJevX(off_model, features)
    frozen_const = FrozenFeatureJevX(const_model, features)
    frozen_full = FrozenFeatureJevX(full_model, features)

    schema = [[Option(key=o, label=o) for o in options] for _ in range(m_questions)]
    q_same = list(questions)
    # A MISS on every iteration: a bank of distinct question sets, cycled.
    # With _CONDITION_CACHE_LIMIT = 8 and FIFO eviction this is the worst case
    # a real question sweep can produce.
    n_banks = max(4, iters // max(1, m_questions) + 2)
    miss_banks = [[f"{q} pass {k}" for q in questions] for k in range(n_banks)]
    ids1 = torch.tensor([[FrozenFeatureJevX.INDEX_OFFSET]], dtype=torch.long)
    am1 = torch.ones_like(ids1)

    def call_off() -> None:
        with torch.no_grad():
            off_model(input_ids=ids1, attention_mask=am1,
                      questions=q_same, options=schema)

    def call_hit() -> None:
        with torch.no_grad():
            const_model(input_ids=ids1, attention_mask=am1,
                        questions=q_same, options=schema)

    def call_miss(bank: Sequence[str]) -> None:
        with torch.no_grad():
            const_model(input_ids=ids1, attention_mask=am1,
                        questions=list(bank), options=schema)

    def call_full() -> None:
        with torch.no_grad():
            full_model(input_ids=ids1, attention_mask=am1,
                       questions=q_same, options=schema)

    call_hit()   # prime the memo so the first timed HIT really is a hit
    call_full()
    call_off()
    for _ in range(warmup):
        call_off()
        call_hit()
        call_full()
        for bank in miss_banks[: max(1, m_questions)]:
            call_miss(bank)

    names = ("off", "constant_hit", "constant_miss", "full")
    samples: Dict[str, List[float]] = {k: [] for k in names}
    for i in range(iters):
        for name, fn in (("off", call_off), ("constant_hit", call_hit), ("full", call_full)):
            t0 = time.perf_counter()
            fn()
            samples[name].append((time.perf_counter() - t0) * 1000.0)
        t0 = time.perf_counter()
        call_miss(miss_banks[i % len(miss_banks)])
        samples["constant_miss"].append((time.perf_counter() - t0) * 1000.0)

    def stats(key: str) -> Dict[str, float]:
        arr = np.asarray(samples[key], dtype=np.float64)
        return {
            "median": float(np.median(arr)),
            "p10": float(np.percentile(arr, 10)),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    st = {k: stats(k) for k in names}
    ratio = st["constant_hit"]["median"] / st["off"]["median"]
    return {
        "tag": tag,
        "batch_size": 1,
        "m_questions": m_questions,
        "num_options": len(options),
        "warmup": warmup,
        "iters": iters,
        "per_arm": st,
        "ratio_constant_hit_over_off": ratio,
        "ratio_constant_hit_over_off_p10": st["constant_hit"]["p10"] / st["off"]["p10"],
        "ratio_constant_miss_over_off": st["constant_miss"]["median"] / st["off"]["median"],
        "ratio_full_over_off": st["full"]["median"] / st["off"]["median"],
        "tolerance": CACHE_HIT_TOLERANCE,
        "criterion_met": bool(ratio <= CACHE_HIT_TOLERANCE),
        "criterion": "median(constant-HIT) <= 1.15 x median(state-only 'off')",
        "state_only_arm_definition": (
            "conditioning_mode='off' - the option-interaction residual is exactly "
            "zero, but the question vector still reaches ConcatFusion. The extra "
            "'state_only' reference in the grid zeroes the question as well."
        ),
        "interleaving": "round-robin off/hit/full/miss inside one loop (common-mode contention)",
        "unused_frozen_views": [type(frozen_off).__name__, type(frozen_const).__name__,
                                type(frozen_full).__name__],
    }


def phase_cache_cost(args: argparse.Namespace) -> Dict[str, Any]:
    results: Dict[str, Any] = {
        "phase": "cache-cost",
        "environment": environment_info(),
        "arms": ["off", "constant_hit", "constant_miss", "full"],
    }
    measurements: List[Dict[str, Any]] = []

    tiny_bert, tiny_tok = build_tiny_backbone_and_tokenizer()
    log("cache-cost: bitwise equality (tiny in-memory BERT + tests/conftest tokenizer)")
    results["bitwise_tiny_bert"] = _bitwise_checks(
        make_probe_model(tiny_bert, tiny_tok, conditioning_mode="constant", hidden=32),
        CACHE_QUESTIONS_TINY, CACHE_OPTIONS_TINY,
    )
    log(f"cache-cost: tiny bitwise -> {json.dumps(results['bitwise_tiny_bert'])}")
    log("cache-cost: timing (tiny in-memory BERT)")
    measurements.append(
        measure_cache_cost(
            tiny_bert, tiny_tok,
            tag="tiny_in_memory_bert_conftest_tokenizer",
            questions=CACHE_QUESTIONS_TINY, options=CACHE_OPTIONS_TINY,
            warmup=args.warmup, iters=args.latency_iters,
        )
    )

    log(f"cache-cost: repeating on the real backbone {TEXT_BACKBONE}")
    try:
        full = bc.JevX.from_pretrained(TEXT_BACKBONE, freeze_base_model=True,
                                      pooling_method=POOLING, fusion_type=FUSION)
        real_bb, real_tok = full.model.eval(), full.tokenizer
        results["bitwise_real_backbone"] = _bitwise_checks(
            make_probe_model(real_bb, real_tok, conditioning_mode="constant", hidden=128),
            CACHE_QUESTIONS_REAL, CACHE_OPTIONS_REAL,
        )
        log(f"cache-cost: real bitwise -> {json.dumps(results['bitwise_real_backbone'])}")
        measurements.append(
            measure_cache_cost(
                real_bb, real_tok,
                tag=f"real_backbone_{TEXT_BACKBONE}",
                questions=CACHE_QUESTIONS_REAL, options=CACHE_OPTIONS_REAL,
                warmup=args.warmup, iters=args.latency_iters,
            )
        )
    except Exception as exc:  # noqa: BLE001 - reported verbatim, never hidden
        results["bitwise_real_backbone"] = {"error": f"{type(exc).__name__}: {exc}"}
        results["real_backbone_traceback"] = traceback.format_exc()
        log(f"cache-cost: FAILED real-backbone cell: {type(exc).__name__}: {exc}")

    results["measurements"] = measurements
    results["headline"] = [
        {
            "tag": m["tag"],
            "median_ms": {k: round(v["median"], 4) for k, v in m["per_arm"].items()},
            "hit_over_off": round(m["ratio_constant_hit_over_off"], 4),
            "miss_over_off": round(m["ratio_constant_miss_over_off"], 4),
            "full_over_off": round(m["ratio_full_over_off"], 4),
            "criterion_met": m["criterion_met"],
        }
        for m in measurements
    ]
    path = RESULTS_DIR / "multi_question_cache_cost.json"
    write_json(path, results)
    log(f"cache-cost: wrote {path}")
    return results


# --------------------------------------------------------------------------
# Questions, question groups, datasets
# --------------------------------------------------------------------------


class Question:
    """One REPORTING question asked about every state that carries it.

    A reporting question can expand into several ``Group``s: ABSA's question
    text is templated on the aspect term, so "aspect polarity" is many
    distinct question strings.
    """

    def __init__(self, name: str, text: str, options: Sequence[str],
                 provenance: str, why: str) -> None:
        if provenance not in ("REAL", "DERIVED"):
            raise ValueError(f"provenance must be REAL or DERIVED, got {provenance!r}")
        self.name = name
        self.text = text
        self.options = list(options)
        self.provenance = provenance
        self.why = why

    @property
    def k(self) -> int:
        return len(self.options)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "text": self.text,
            "options": self.options,
            "num_options": self.k,
            "label_provenance": self.provenance,
            "provenance_reason": self.why,
        }


class Group:
    """One distinct (question text, schema) actually handed to the model."""

    def __init__(self, name: str, question: str, options: Sequence[str],
                 rows: np.ndarray, labels: np.ndarray) -> None:
        self.name = name
        self.question = question
        self.options = list(options)
        self.rows = np.asarray(rows, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        if len(self.rows) != len(self.labels):
            raise ValueError(f"group {name}: {len(self.rows)} rows vs {len(self.labels)} labels")
        if len(self.rows) and int(self.labels.max()) >= len(self.options):
            raise ValueError(f"group {name}: label {int(self.labels.max())} >= K {len(self.options)}")


class MultiQuestionDataset:
    def __init__(self, name: str, modality: str, questions: List[Question],
                 features: np.ndarray, groups: List[Group],
                 dataset_ids: Mapping[str, Any], splits: Mapping[str, np.ndarray],
                 stratify_labels: np.ndarray, extra: Optional[Mapping[str, Any]] = None,
                 texts: Optional[List[str]] = None) -> None:
        self.name = name
        self.modality = modality
        self.questions = questions
        self.features = np.asarray(features, dtype=np.float32)
        self.groups = groups
        self.dataset_ids = dict(dataset_ids)
        self.splits = {k: np.asarray(v, dtype=np.int64) for k, v in splits.items()}
        self.stratify_labels = np.asarray(stratify_labels, dtype=np.int64)
        self.extra = dict(extra or {})
        self.texts = texts

    @property
    def n_states(self) -> int:
        return int(self.features.shape[0])

    def counts(self) -> Dict[str, Any]:
        per_state: List[int] = []
        for q in self.questions:
            asked = {int(r) for g in self.groups if g.question in self._texts_of(q) for r in g.rows}
            per_state.append(len(asked))
        return {
            "n_states": self.n_states,
            "feature_dim": int(self.features.shape[1]),
            "modality": self.modality,
            "n_question_groups": len(self.groups),
            "questions": [
                {
                    **q.as_dict(),
                    "n_distinct_question_strings": sum(
                        1 for g in self.groups if g.question.startswith(q.text.split("{")[0][:18])
                    ) if "{" in q.text else 1,
                    "n_labeled_pairs": {
                        split: int(sum(
                            len(g.rows) for g in self.groups
                            if self._group_belongs_to(q, g) and
                            bool(np.isin(g.rows, self.splits[split]).all())
                        ))
                        for split in ("train", "cal", "test")
                    },
                }
                for q in self.questions
            ],
            "questions_per_state_min_max": [
                int(min(per_state)) if per_state else 0,
                int(max(per_state)) if per_state else 0,
            ],
        }

    def _texts_of(self, q: Question) -> set:
        return {q.text} if "{" not in q.text else {
            g.question for g in self.groups if g.name.startswith(q.name + "::")
        }

    def _group_belongs_to(self, q: Question, g: Group) -> bool:
        if "{" in q.text:
            return g.name.startswith(q.name + "::")
        return g.question == q.text

    def groups_of(self, q: Question) -> List[Group]:
        return [g for g in self.groups if self._group_belongs_to(q, g)]


# -- ABSA: SemEval-2014 Task 4 (restaurant + laptop) -----------------------

ABSA_REPOS: Dict[str, Tuple[str, str]] = {
    "restaurant": ("xinyixiuxiu/semeval2014_restaurant",
                   "709b40b7f0a9ed142df433c0d84aaa14d0d2e8c2"),
    "laptop": ("xinyixiuxiu/semeval2014_laptop",
               "fed294ac48b1b9f903afb833baeeb45754fe7e2c"),
}
ABSA_POLARITY: Dict[int, str] = {0: "negative", 1: "neutral", 2: "positive"}
ABSA_TEMPLATE = "What is the sentiment toward the {aspect} of this review?"
ABSA_MIN_ASPECTS = 4

_ABSA_MEMO: Optional[Dict[str, Any]] = None


def load_absa(min_aspects: int = ABSA_MIN_ASPECTS) -> Dict[str, Any]:
    """SemEval-2014 Task 4 aspect-term + polarity; states with >= min_aspects.

    REAL labels: ``label`` is the dataset's own per-aspect polarity field.
    Nothing is annotated or derived here. A state is a sentence carrying at
    least ``min_aspects`` DISTINCT aspect terms, so every state supports at
    least ``min_aspects`` genuinely different questions.
    """
    global _ABSA_MEMO
    if _ABSA_MEMO is not None:
        return _ABSA_MEMO
    import collections

    import pandas as pd

    sentences: "OrderedDict[str, List[Tuple[str, int]]]" = OrderedDict()
    origin: Dict[str, str] = {}
    provenance: Dict[str, Any] = {}
    for domain, (repo, sha) in ABSA_REPOS.items():
        t0 = time.perf_counter()
        files = _hub_parquet_files(repo, sha)
        frames = [pd.read_parquet(io.BytesIO(_hub_resolve(repo, sha, fn))) for fn in files]
        df = pd.concat(frames, ignore_index=True)
        seen_labels = {int(v) for v in df["label"]}
        for row in df.to_dict("records"):
            text, aspect, label = str(row["text"]).strip(), str(row["aspect"]).strip(), int(row["label"])
            bucket = sentences.setdefault(text, [])
            if all(a != aspect for a, _ in bucket):
                bucket.append((aspect, label))
            origin.setdefault(text, domain)
        provenance[domain] = {
            "dataset_id": repo,
            "dataset_revision": sha,
            "resolved_sha_verified": True,
            "files": files,
            "n_rows": int(len(df)),
            "columns": sorted(map(str, df.columns)),
            "label_value_counts": {
                ABSA_POLARITY[k]: int((df["label"] == k).sum()) for k in sorted(seen_labels)
            },
            "fetch_wall_s": round(time.perf_counter() - t0, 2),
        }
    kept = sorted(
        ((t, v) for t, v in sentences.items() if len(v) >= min_aspects),
        key=lambda kv: (-len(kv[1]), kv[0]),
    )
    texts = [t for t, _ in kept]
    aspects = [[a for a, _ in v] for _, v in kept]
    labels = [[lab for _, lab in v] for _, v in kept]
    _ABSA_MEMO = {
        "texts": texts,
        "aspects": aspects,
        "labels": labels,
        "origin": [origin[t] for t in texts],
        "provenance": provenance,
        "n_sentences_total": len(sentences),
        "n_sentences_kept": len(texts),
        "min_aspects": min_aspects,
        "domain_counts": {
            d: {
                "all_sentences": int(sum(1 for o in origin.values() if o == d)),
                "kept_sentences": int(sum(1 for o in [origin[t] for t in texts] if o == d)),
            }
            for d in sorted(set(origin.values()))
        },
    }
    return _ABSA_MEMO


def _hub_parquet_files(repo: str, revision: str) -> List[str]:
    import requests

    r = requests.get(f"https://huggingface.co/api/datasets/{repo}", timeout=60)
    r.raise_for_status()
    files = sorted(
        f["rfilename"] for f in r.json().get("siblings", []) if f["rfilename"].endswith(".parquet")
    )
    if not files:
        raise RuntimeError(f"{repo}@{revision}: no parquet files")
    return files


def _hub_resolve(repo: str, revision: str, filename: str) -> bytes:
    import requests

    url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}"
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    return r.content


# -- RAVDESS: four filename-encoded labels ---------------------------------

RAVDESS_INTENSITY: Dict[str, str] = {"01": "medium", "02": "high", "03": "low"}


def load_ravdess_multi() -> Dict[str, Any]:
    """RAVDESS with FOUR filename-encoded labels per clip.

    All four are REAL: they are fields the dataset ships in the filename
    (``<emotion>-spk_NN-intensity_NN-statement_NN-repetition_NN.wav``), not
    anything a model produced and not a rule invented here. Gender follows the
    documented RAVDESS actor convention (odd actors 01-20 male, even female).
    """
    paths, emotion, prov = rm.load_ravdess()
    genders: List[int] = []
    intensities: List[int] = []
    statements: List[int] = []
    for p in paths:
        parts = Path(p).stem.split("-")
        if len(parts) != 5:
            raise RuntimeError(f"unexpected RAVDESS filename: {Path(p).name}")
        spk = int(parts[1].split("_")[1])
        intensity = parts[2].split("_")[1]
        statement = parts[3].split("_")[1]
        if intensity not in RAVDESS_INTENSITY or statement not in ("01", "02"):
            raise RuntimeError(f"unexpected RAVDESS filename: {Path(p).name}")
        genders.append(0 if spk % 2 == 1 else 1)
        intensities.append(["medium", "high", "low"].index(RAVDESS_INTENSITY[intensity]))
        statements.append(0 if statement == "01" else 1)
    counts = lambda a: {str(c): int((np.asarray(a) == c).sum()) for c in sorted(set(a))}  # noqa: E731
    prov = dict(prov)
    prov["filename_fields"] = {
        "emotion": "field 0 (prefix): the 8 RAVDESS emotion codes",
        "gender": "field 1 (spk_NN): odd actor = male, even actor = female "
                  "(the RAVDESS actor convention)",
        "intensity": "field 2 (intensity_NN): 01=medium, 02=high, 03=low",
        "statement_vs_question": "field 3 (statement_NN): 01=statement, 02=question",
    }
    prov["label_counts"] = {
        "emotion": counts(emotion), "gender": counts(genders),
        "intensity": counts(intensities), "statement_vs_question": counts(statements),
    }
    return {"paths": paths, "labels": {"emotion": emotion, "gender": genders,
                                       "intensity": intensities, "statement": statements},
            "provenance": prov}


# -- ag_news / Yelp ---------------------------------------------------------

AGNEWS_ID = ("fancyzhx/ag_news", "eb185aade064a813bc0b7f42de02595523103ca4")
AGNEWS_TOPICS = ["World", "Sports", "Business", "Sci/Tech"]
YELP_ID = ("Yelp/yelp_review_full", "c1f9ee939b7d05667af864ee1cb066393154bf85")


def load_ag_news() -> Dict[str, Any]:
    from datasets import load_dataset

    ds = load_dataset(AGNEWS_ID[0], revision=AGNEWS_ID[1])
    labels = [int(v) for v in ds["train"]["label"]]
    return {
        "texts": list(ds["train"]["text"]), "labels": labels,
        "provenance": {
            "dataset_id": AGNEWS_ID[0], "dataset_revision": AGNEWS_ID[1],
            "resolved_sha_verified": True, "n_rows": len(labels),
            "label_counts": {AGNEWS_TOPICS[c]: int(sum(1 for y in labels if y == c))
                             for c in range(4)},
        },
    }


def load_yelp() -> Dict[str, Any]:
    from datasets import load_dataset

    ds = load_dataset(YELP_ID[0], revision=YELP_ID[1])
    labels = [int(v) for v in ds["train"]["label"]]
    return {
        "texts": list(ds["train"]["text"]), "labels": labels,
        "provenance": {
            "dataset_id": YELP_ID[0], "dataset_revision": YELP_ID[1],
            "resolved_sha_verified": True, "n_rows": len(labels),
            "label_counts": {f"{c + 1} star": int(sum(1 for y in labels if y == c))
                             for c in range(5)},
        },
    }


# --------------------------------------------------------------------------
# Featurization + splits
# --------------------------------------------------------------------------


def _featurize_text(texts: Sequence[str]) -> Tuple[np.ndarray, Dict[str, Any]]:
    model = bc.JevX.from_pretrained(
        TEXT_BACKBONE, freeze_base_model=True, pooling_method=POOLING, fusion_type=FUSION
    ).eval()
    t0 = time.perf_counter()
    feats = bc.featurize(model, list(texts), batch_size=64, max_length=MAX_LENGTH,
                         strategies=[POOLING])[POOLING]
    meta = {
        "backbone_id": TEXT_BACKBONE,
        "backbone_revision": getattr(model.config, "_commit_hash", None),
        "pooling": POOLING, "fusion": FUSION, "max_length": MAX_LENGTH,
        "feature_dim": int(feats.shape[1]),
        "featurize_wall_s": round(time.perf_counter() - t0, 2),
        "featurize_once": True,
    }
    del model
    return feats, meta


def _featurize_audio(paths: Sequence[str]) -> Tuple[np.ndarray, Dict[str, Any]]:
    model = bc.JevX.from_pretrained(AUDIO_BACKBONE, freeze_base_model=True,
                                    pooling_method=None).eval()

    def make_state(chunk: Sequence[Any]) -> Dict[str, Any]:
        return {"audio": [rm.load_waveform(Path(q)) for q in chunk]}

    t0 = time.perf_counter()
    feats = rm.featurize_payloads(model, list(paths), make_state, batch_size=8)
    meta = {
        "backbone_id": AUDIO_BACKBONE,
        "backbone_revision": getattr(model.config, "_commit_hash", None),
        "pooling": "native (wav2vec2 has no pooler; adapter masked_mean)",
        "feature_dim": int(feats.shape[1]),
        "audio_max_seconds": rm.AUDIO_MAX_SECONDS,
        "featurize_wall_s": round(time.perf_counter() - t0, 2),
        "featurize_once": True,
    }
    del model
    return feats, meta


def pool_indices(name: str, stratify_labels: Sequence[int]) -> np.ndarray:
    """The deterministic stratified pool subsample a split is carved from."""
    labels = np.asarray(stratify_labels)
    cap = POOL_CAPS[name]
    if cap is None or cap >= len(labels):
        return np.arange(len(labels), dtype=np.int64)
    return np.sort(bc.stratified_indices(labels, int(cap), seed=SPLIT_SEED))


def split_states(stratify_labels: Sequence[int], name: str, *,
                 seed: int = SPLIT_SEED) -> Dict[str, np.ndarray]:
    """Deterministic, disjoint, stratified train/cal/test over STATES.

    The split is over STATES, never over (state, question) pairs: a state that
    can never contribute to two splits, or the multi-question regime would leak
    its own answers.
    """
    labels = np.asarray(stratify_labels)
    n = len(labels)
    if name == "ravdess":
        # The established run_modality / run_head_quality RAVDESS protocol:
        # 700 train + 240 cal carved from the pool, 500 test from the rest.
        train_idx, cal_idx = bc.make_splits(labels, 700, 240, seed=seed)
        rest = np.ones(n, dtype=bool)
        rest[train_idx] = False
        rest[cal_idx] = False
        test_idx = np.flatnonzero(rest)
        if len(test_idx) > 500:
            test_idx = test_idx[bc.stratified_indices(labels[test_idx], 500, seed=2)]
        return {"train": train_idx, "cal": cal_idx, "test": test_idx}
    cap_tr, cap_cal, cap_te = SPLIT_CAPS[name]
    if cap_tr is None and cap_cal is None and cap_te is None:
        n_train = int(round(0.70 * n))
        n_cal = int(round(0.15 * n))
    else:
        n_train = int(cap_tr or round(0.70 * n))
        n_cal = int(cap_cal or round(0.15 * n))
    n_train = max(2, min(n_train, n - 2))
    n_cal = max(1, min(n_cal, n - n_train - 1))
    train_idx, cal_idx = bc.make_splits(labels, n_train, n_cal, seed=seed)
    rest = np.ones(n, dtype=bool)
    rest[train_idx] = False
    rest[cal_idx] = False
    remaining = np.flatnonzero(rest)
    n_test = int(cap_te) if cap_te else len(remaining)
    n_test = max(1, min(n_test, len(remaining)))
    test_idx = remaining[bc.stratified_indices(labels[remaining], n_test, seed=seed + 2)]
    return {"train": train_idx, "cal": cal_idx, "test": test_idx}


# --------------------------------------------------------------------------
# Dataset construction
# --------------------------------------------------------------------------


def _questions_absa() -> Question:
    return Question(
        "absa_aspect_polarity", ABSA_TEMPLATE, ["negative", "neutral", "positive"],
        "REAL",
        "SemEval-2014 Task 4 ships a polarity label for every (sentence, aspect "
        "term) pair; this is the dataset's own annotation, not a rule applied "
        "here. The question TEXT is templated on the aspect term, so each state "
        "carries one genuinely different question per aspect it contains (>= 4).",
    )


def _questions_ravdess() -> List[Question]:
    return [
        Question("ravdess_emotion", "Which emotion is spoken in this recording?",
                 ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgusted",
                  "surprised"], "REAL",
                 "the emotion code is field 0 of the RAVDESS filename; the dataset ships it."),
        Question("ravdess_gender", "Is the speaker a man or a woman?", ["man", "woman"],
                 "REAL",
                 "the actor id is field 1 of the RAVDESS filename (spk_NN); RAVDESS "
                 "documents odd-numbered actors 01-20 as male and even-numbered as female."),
        Question("ravdess_intensity", "How intense is the emotional delivery?",
                 ["medium", "high", "low"], "REAL",
                 "field 2 of the RAVDESS filename (intensity_NN): 01=medium, 02=high, 03=low."),
        Question("ravdess_statement", "Is the utterance a statement or a question?",
                 ["statement", "question"], "REAL",
                 "field 3 of the RAVDESS filename (statement_NN): 01=statement, 02=question."),
    ]


def _questions_ag_news() -> List[Question]:
    return [
        Question("ag_news_topic", "What is the topic of this news article?",
                 AGNEWS_TOPICS, "REAL", "the 4-way topic label is ag_news's own annotation."),
        Question("ag_news_is_sports", "Is this news article about sports?", ["no", "yes"],
                 "DERIVED",
                 "deterministic rule: label == 1 (Sports). A threshold on the dataset's "
                 "own label, not a new annotation."),
        Question("ag_news_is_scitech", "Is this news article about science or technology?",
                 ["no", "yes"], "DERIVED", "deterministic rule: label == 3 (Sci/Tech)."),
        Question("ag_news_is_business", "Is this news article about business?",
                 ["no", "yes"], "DERIVED", "deterministic rule: label == 2 (Business)."),
    ]


def _questions_yelp() -> List[Question]:
    return [
        Question("yelp_stars", "How many stars did this reviewer give?",
                 ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"], "REAL",
                 "the 5-way star rating is Yelp's own label field."),
        Question("yelp_detractor", "Is this a 1-star detractor review?", ["no", "yes"],
                 "DERIVED",
                 "deterministic rule: stars == 1. The brief calls this real; it is a "
                 "threshold on a REAL dataset label, so it is recorded here as "
                 "DERIVED-from-a-real-label, which is the distinction that matters."),
    ]


def build_dataset(name: str, *, recompute: bool) -> MultiQuestionDataset:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{name}_features.npz"
    fmeta: Optional[Dict[str, Any]] = None
    if cache.exists() and not recompute:
        blob = np.load(cache, allow_pickle=False)
        features = np.asarray(blob["x"], dtype=np.float32)
        fmeta = json.loads(blob["meta_json"].tobytes().decode("utf-8"))
        log(f"[{name}] feature cache hit {cache} (shape {features.shape})")
    else:
        pool_meta: Dict[str, Any] = {}
        if name == "absa":
            raw = load_absa()
            idx = pool_indices(name, [min(len(a), 6) for a in raw["aspects"]])
            features, fmeta = _featurize_text([raw["texts"][i] for i in idx])
            pool_meta = {"pool_size": int(len(idx)), "n_rows_loaded": len(raw["texts"])}
        elif name == "ag_news":
            raw = load_ag_news()
            idx = pool_indices(name, raw["labels"])
            features, fmeta = _featurize_text([raw["texts"][i] for i in idx])
            pool_meta = {"pool_size": int(len(idx)), "n_rows_loaded": len(raw["texts"])}
        elif name == "yelp":
            raw = load_yelp()
            idx = pool_indices(name, raw["labels"])
            features, fmeta = _featurize_text([raw["texts"][i] for i in idx])
            pool_meta = {"pool_size": int(len(idx)), "n_rows_loaded": len(raw["texts"])}
        elif name == "ravdess":
            features, fmeta = _ravdess_features(recompute)
            pool_meta = {"pool_size": int(features.shape[0]), "n_rows_loaded": int(features.shape[0])}
        else:
            raise ValueError(f"unknown dataset {name!r}")
        fmeta = {**fmeta, "pool": pool_meta, "pool_cap": POOL_CAPS[name],
                 "pool_rule": ("the whole pool" if POOL_CAPS[name] is None else
                               f"bc.stratified_indices(labels, {POOL_CAPS[name]}, seed={SPLIT_SEED})")}
        meta_blob = json.dumps({"dataset": name, "featurize": fmeta}).encode("utf-8")
        np.savez_compressed(cache, x=features,
                            meta_json=np.frombuffer(meta_blob, dtype=np.uint8))
        log(f"[{name}] cached features -> {cache} (shape {features.shape})")

    ds = _assemble(name, features, fmeta or {})
    for a, b, split in (("train", "cal", "test"),):
        if bool(np.intersect1d(ds.splits[a], ds.splits[b]).any()):
            raise RuntimeError(f"{name}: {a}/{b} split overlap")
        if bool(np.intersect1d(ds.splits[a], ds.splits[split]).any()):
            raise RuntimeError(f"{name}: {a}/{split} split overlap")
    return ds


def _ravdess_features(recompute: bool) -> Tuple[np.ndarray, Dict[str, Any]]:
    raw = load_ravdess_multi()
    paths = raw["paths"]
    src = REPO_ROOT / "benchmarks" / "cache" / "head_quality" / "ravdess_features.npz"
    if src.exists() and not recompute:
        reuse = np.load(src, allow_pickle=False)
        splits = split_states(raw["labels"]["emotion"], "ravdess")
        order = np.concatenate([splits[k] for k in ("train", "cal", "test")])
        cached = np.concatenate([reuse["x_train"], reuse["x_cal"], reuse["x_test"]], axis=0)
        if len(order) != len(cached):
            raise RuntimeError(
                f"ravdess cache reuse refused: {len(order)} states vs {len(cached)} rows")
        for split in ("train", "cal", "test"):
            mine = {str(c): int((np.asarray(raw["labels"]["emotion"])[splits[split]] == c).sum())
                    for c in sorted(set(int(v) for v in np.asarray(raw["labels"]["emotion"])[splits[split]]))}
            theirs = {str(c): int((reuse[f"y_{split}"] == c).sum())
                      for c in sorted(set(int(v) for v in reuse[f"y_{split}"]))}
            if mine != theirs:
                raise RuntimeError(
                    f"ravdess cache reuse refused: {split} emotion counts {mine} vs {theirs}")
        inverse = np.empty(len(order), dtype=np.int64)
        inverse[order] = np.arange(len(order))
        feats = cached[inverse]
        log(f"[ravdess] reused {src.name}; emotion-label counts verified per split")
        return feats, {
            "backbone_id": AUDIO_BACKBONE,
            "pooling": "native (wav2vec2 has no pooler; adapter masked_mean)",
            "feature_dim": int(feats.shape[1]),
            "audio_max_seconds": rm.AUDIO_MAX_SECONDS,
            "featurize_once": True,
            "feature_cache_reused_from": str(src.relative_to(REPO_ROOT)),
            "feature_cache_reuse_verified": {
                "rows": int(feats.shape[0]),
                "per_split_emotion_label_counts": "identical to the recomputed split",
                "why_safe": (
                    "run_head_quality.audio_task carves train/cal/test from the SAME "
                    "sorted path list with the SAME bc.make_splits(seed=0), so the "
                    "cached rows are the same states; the reuse is only accepted "
                    "after the per-split emotion-label counts agree."),
            },
        }
    return _featurize_audio(paths)


def _assemble(name: str, features: np.ndarray, fmeta: Mapping[str, Any]) -> MultiQuestionDataset:
    if name == "absa":
        raw = load_absa()
        full_labels = [min(len(a), 6) for a in raw["aspects"]]
        keep = pool_indices(name, full_labels)
        texts = [raw["texts"][i] for i in keep]
        aspects = [raw["aspects"][i] for i in keep]
        labels_ = [raw["labels"][i] for i in keep]
        origin = [raw["origin"][i] for i in keep]
        if len(texts) != features.shape[0]:
            raise RuntimeError(
                f"absa cache refused: {features.shape[0]} rows vs {len(texts)} pooled texts")
        q = _questions_absa()
        buckets: Dict[str, Dict[int, int]] = {}
        for i, (asps, labs) in enumerate(zip(aspects, labels_)):
            for a, lab in zip(asps, labs):
                buckets.setdefault(a, {})[i] = int(lab)
        groups: List[Group] = []
        for a in sorted(buckets):
            table = buckets[a]
            rows = np.asarray(sorted(table), dtype=np.int64)
            groups.append(Group(
                name=f"{q.name}::{a}", question=ABSA_TEMPLATE.format(aspect=a),
                options=q.options, rows=rows,
                labels=np.asarray([table[int(r)] for r in rows], dtype=np.int64)))
        ids = {
            "per_domain": raw["provenance"],
            "n_sentences_total": raw["n_sentences_total"],
            "n_sentences_kept": raw["n_sentences_kept"],
            "min_aspects_per_state": raw["min_aspects"],
            "domain_counts": raw["domain_counts"],
            "aspects_per_state_max": max(len(a) for a in aspects),
            "aspects_per_state_min": min(len(a) for a in aspects),
            "aspects_per_state_mean": float(np.mean([len(a) for a in aspects])),
        }
        extra = {"aspects_per_state": [len(a) for a in aspects], "state_domain": origin}
        stratify = np.array([min(len(a), 6) for a in aspects], dtype=np.int64)
        ds = MultiQuestionDataset(name, "text", [q], features, groups, ids,
                                  split_states(stratify, name), stratify, extra,
                                  texts=texts)
    elif name == "ravdess":
        raw = load_ravdess_multi()
        if len(raw["paths"]) != features.shape[0]:
            raise RuntimeError(
                f"ravdess cache refused: {features.shape[0]} rows vs {len(raw['paths'])} clips")
        questions = _questions_ravdess()
        groups = []
        for i, q in enumerate(questions):
            lab = np.asarray(raw["labels"][
                ("emotion", "gender", "intensity", "statement")[i]], dtype=np.int64)
            groups.append(Group(q.name, q.text, q.options,
                                np.arange(len(lab), dtype=np.int64), lab))
        stratify = np.asarray(raw["labels"]["emotion"], dtype=np.int64)
        ds = MultiQuestionDataset(name, "audio", questions, features, groups,
                                  dict(raw["provenance"]),
                                  split_states(stratify, name), stratify,
                                  {"clip_paths_sample": raw["paths"][:3]})
    elif name == "ag_news":
        raw = load_ag_news()
        keep = pool_indices(name, raw["labels"])
        texts = [raw["texts"][i] for i in keep]
        topic = np.asarray([raw["labels"][i] for i in keep], dtype=np.int64)
        if len(texts) != features.shape[0]:
            raise RuntimeError(
                f"ag_news cache refused: {features.shape[0]} rows vs {len(texts)} pooled texts")
        questions = _questions_ag_news()
        rows = np.arange(len(topic), dtype=np.int64)
        groups = [Group(questions[0].name, questions[0].text, questions[0].options, rows, topic)]
        for q, rule in zip(questions[1:], (topic == 1, topic == 3, topic == 2)):
            groups.append(Group(q.name, q.text, q.options, rows, rule.astype(np.int64)))
        prov = dict(raw["provenance"])
        prov["pool"] = {"n_rows_loaded": len(raw["texts"]), "n_rows_used": int(len(keep)),
                        "rule": f"bc.stratified_indices(labels, {POOL_CAPS[name]}, "
                                f"seed={SPLIT_SEED})"}
        prov["label_counts"] = {AGNEWS_TOPICS[c]: int((topic == c).sum()) for c in range(4)}
        ds = MultiQuestionDataset(name, "text", questions, features, groups, prov,
                                  split_states(topic, name), topic, {}, texts=texts)
    elif name == "yelp":
        raw = load_yelp()
        keep = pool_indices(name, raw["labels"])
        texts = [raw["texts"][i] for i in keep]
        stars = np.asarray([raw["labels"][i] for i in keep], dtype=np.int64)
        if len(texts) != features.shape[0]:
            raise RuntimeError(
                f"yelp cache refused: {features.shape[0]} rows vs {len(texts)} pooled texts")
        questions = _questions_yelp()
        rows = np.arange(len(stars), dtype=np.int64)
        groups = [Group(questions[0].name, questions[0].text, questions[0].options, rows, stars),
                  Group(questions[1].name, questions[1].text, questions[1].options, rows,
                        (stars == 0).astype(np.int64))]
        prov = dict(raw["provenance"])
        prov["pool"] = {"n_rows_loaded": len(raw["texts"]), "n_rows_used": int(len(keep)),
                        "rule": f"bc.stratified_indices(labels, {POOL_CAPS[name]}, "
                                f"seed={SPLIT_SEED})"}
        prov["label_counts"] = {f"{c + 1} star": int((stars == c).sum()) for c in range(5)}
        ds = MultiQuestionDataset(
            name, "text", questions, features, groups, prov,
            split_states(stars, name), stars,
            {"omitted_questions": (
                "No aspect-level questions on Yelp. Yelp/yelp_review_full ships exactly "
                "one label field (the star rating), carries no aspect annotation, and no "
                "aspect-labelled source is reachable from it. Hand-writing aspect labels "
                "would be annotating a benchmark by hand and calling it real, so they are "
                "OMITTED. The only further obtainable 'questions' are restatements of the "
                "same star label, which add no information and would inflate the "
                "multi-question appearance without content.")},
            texts=raw["texts"])
    else:
        raise ValueError(f"unknown dataset {name!r}")
    ds.extra["featurize"] = dict(fmeta)
    return ds


# --------------------------------------------------------------------------
# The grid arms
# --------------------------------------------------------------------------


class _GradProbe:
    """Per-module gradient L2 norms at two chosen OPTIMIZER STEPS.

    The step index is advanced explicitly (begin_step) at the top of each
    optimizer step, not by a forward hook: one optimizer step here may issue
    SEVERAL forward calls (one per question-homogeneous chunk), so a forward
    count is not a step count. The norms are read by per-parameter
    register_post_accumulate_grad_hook callbacks, which fire DURING
    loss.backward() - before clip_grad_norm_(1.0) - so the numbers are the RAW
    gradients.

    Modules are the INNER JevXForDecision's own children (question_encoder,
    schema_encoder, fusion, heads, ...), not the outer wrapper's, so a module
    that received no gradient is visibly ABSENT from a record. That absence is
    the measurement.
    """

    def __init__(self, module: nn.Module, inner: nn.Module,
                 targets: Sequence[int], total_steps: int) -> None:
        self.module = module
        self.inner = inner
        self.targets = {int(t) for t in targets}
        self.total_steps = int(total_steps)
        self.step = 0
        self.captured = {}
        self._acc = {}
        self._handles = []
        for name, params in self._groups(inner).items():
            for prm in params:
                if prm.requires_grad:
                    self._handles.append(
                        prm.register_post_accumulate_grad_hook(self._hook(name)))

    @staticmethod
    def _groups(module):
        return {name: [prm for prm in child.parameters() if prm.requires_grad]
                for name, child in module.named_children()}

    def begin_step(self):
        self.step += 1
        if self.step in self.targets:
            self._acc = {}

    def _hook(self, group):
        def hook(param):
            if self.step not in self.targets or param.grad is None:
                return
            self._acc[group] = self._acc.get(group, 0.0) + float(
                param.grad.detach().double().pow(2).sum())
        return hook

    def record(self):
        if self.step in self.targets:
            self.captured[self.step] = {
                k: float(np.sqrt(v)) for k, v in sorted(self._acc.items())}

    def result(self):
        seen = set(m for rec in self.captured.values() for m in rec)
        return {
            "targets": sorted(self.targets),
            "observed_optimizer_steps": self.step,
            "expected_optimizer_steps": self.total_steps,
            "at_step": {str(k): v for k, v in sorted(self.captured.items())},
            "modules_with_no_gradient": sorted(set(self._groups(self.inner)) - seen),
            "note": (
                "per-module L2 norm of the accumulated gradient, read after "
                "loss.backward() and before clip_grad_norm_(1.0). A module listed "
                "in modules_with_no_gradient received NO gradient at all on the "
                "recorded steps."
            ),
        }

    def close(self):
        for h in self._handles:
            h.remove()


class _JevXArm(nn.Module):
    """One ``conditioning_mode`` arm: the real forward on frozen features.

    A forward call takes a list of CHUNKS, each ``(group, rows)``: states that
    ask the SAME question. Datasets with one question string per reporting
    question produce one chunk per step; ABSA, whose question text is
    templated on the aspect term, packs several question strings into one
    optimizer step so the grid does not pay one Python-level forward per state.
    Chunks are row-independent, so packing them changes the arithmetic
    re-association of a step, nothing else.

    ``zero_question=True`` (``state_only``) installs a forward hook on
    ``question_encoder`` that replaces its output with exact zeros, so no
    gradient can reach E_Q either, and passes ``condition_options=False`` so
    the option residual is exactly zero and the duplicate tie does not fire.
    """

    def __init__(self, model: JevXForDecision, features: np.ndarray, group: Group,
                 *, zero_question: bool = False) -> None:
        super().__init__()
        self.frozen = FrozenFeatureJevX(model, features)
        self.group = group
        self.schema = [Option(key=o, label=o) for o in group.options]
        self.zero_question = zero_question
        self._zero_hook = None
        if zero_question:
            self._zero_hook = model.question_encoder.register_forward_hook(
                lambda _m, _i, o: torch.zeros_like(o))

    @property
    def jevx(self) -> JevXForDecision:
        return self.frozen.jevx

    def forward(self, chunks: Sequence[Tuple[Group, Sequence[int]]]) -> torch.Tensor:
        if self.zero_question:
            self._zero_hook = self.jevx.question_encoder.register_forward_hook(
                lambda _m, _i, o: torch.zeros_like(o))
        try:
            rows: List[torch.Tensor] = []
            for group, chunk_rows in chunks:
                batch = self.frozen.batch(chunk_rows)
                out = self.jevx(
                    state=batch,
                    questions=[group.question],
                    options=[[Option(key=o, label=o) for o in group.options]],
                    condition_options=False if self.zero_question else None,
                )
                rows.append(torch.cat([d.logits for d in out.decisions], dim=0))
            return torch.cat(rows, dim=0)
        finally:
            if self.zero_question and self._zero_hook is not None:
                self._zero_hook.remove()
                self._zero_hook = None

    def close(self) -> None:
        if self._zero_hook is not None:
            self._zero_hook.remove()
            self._zero_hook = None
        self.frozen.restore()


def build_epoch_steps(
    groups: Sequence[Group], split: np.ndarray, *, seed: int, epochs: int,
    batch_size: int, max_chunks_per_step: int = 64,
) -> List[List[List[Tuple[Group, List[int]]]]]:
    """Per-epoch, seed-shuffled optimizer steps of question-homogeneous chunks.

    A chunk is up to ``batch_size`` states asking the SAME question, so a
    dataset whose states all ask one question costs ``ceil(n / batch_size)``
    forward calls per epoch rather than one per state. ABSA's groups are single
    aspect terms with few states each, so its chunks are small and several are
    packed into one optimizer step; chunks are packed until the step holds
    ``batch_size`` (state, question) PAIRS, so the effective batch size stays
    the protocol's 64 whichever dataset is running.
    """
    wanted = {int(r) for r in split}
    by_group: List[Tuple[Group, List[int]]] = []
    for g in groups:
        rows = [int(r) for r in g.rows if int(r) in wanted]
        if rows:
            by_group.append((g, rows))
    rng = np.random.RandomState(seed)
    out: List[List[List[Tuple[Group, List[int]]]]] = []
    for _ in range(epochs):
        order = rng.permutation(len(by_group))
        pool: List[Tuple[Group, List[int]]] = []
        for i in order:
            g, rows = by_group[int(i)]
            shuffled = [rows[j] for j in rng.permutation(len(rows))]
            pool.extend(
                (g, shuffled[k : k + batch_size])
                for k in range(0, len(shuffled), batch_size)
            )
        steps: List[List[Tuple[Group, List[int]]]] = []
        current: List[Tuple[Group, List[int]]] = []
        current_pairs = 0
        for chunk in pool:
            current.append(chunk)
            current_pairs += len(chunk[1])
            if current_pairs >= batch_size or len(current) >= max_chunks_per_step:
                steps.append(current)
                current, current_pairs = [], 0
        if current:
            steps.append(current)
        out.append(steps)
    return out


def train_jevx_arm(
    arm: str, features: np.ndarray, groups: Sequence[Group], splits: Mapping[str, np.ndarray],
    *, seed: int, backbone: nn.Module, tokenizer: Any, ds_name: str,
    epochs: int = EPOCHS, batch_size: int = BATCH_SIZE,
) -> Dict[str, Any]:
    hidden = int(HEAD_HIDDEN.get(ds_name) or features.shape[1])
    mode = "off" if arm == "state_only" else arm
    model = make_probe_model(backbone, tokenizer, conditioning_mode=mode, hidden=hidden,
                             copy_backbone=False)
    module = _JevXArm(model, features, groups[0], zero_question=(arm == "state_only"))
    train_steps = build_epoch_steps(
        groups, splits["train"], seed=seed, epochs=epochs, batch_size=batch_size)
    test_chunks = _test_chunks(groups, splits["test"])
    total_steps = sum(len(s) for s in train_steps)
    probe = _GradProbe(module, model, targets=(1, total_steps), total_steps=total_steps)
    trainer = HeadTrainer(module, lr=LR, seed=seed)
    pos_lookup = {g.name: {int(r): i for i, r in enumerate(g.rows)} for g in groups}

    history: List[Dict[str, Any]] = []
    try:
        # The step arithmetic is ``HeadTrainer.fit``'s, spelled out so the
        # gradient-norm probe can be read between ``backward()`` (raw) and
        # ``clip_grad_norm_`` (scaled). HeadTrainer still owns the optimizer,
        # the seed and the clip value, so the protocol is unchanged.
        optimizer = trainer.optimizer
        for steps in train_steps:
            payload = [
                (chunks, _labels_for_chunks(chunks, pos_lookup)) for chunks in steps
            ]
            loader = DataLoader(payload, batch_size=None, collate_fn=_passthrough, shuffle=False)
            total_loss = 0.0
            n = 0
            for chunks, labels in loader:
                probe.begin_step()
                optimizer.zero_grad(set_to_none=True)
                loss = choice_loss(module(chunks), labels)
                loss.backward()
                probe.record()
                if trainer.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        (p for p in module.parameters() if p.requires_grad),
                        trainer.grad_clip,
                    )
                optimizer.step()
                total_loss += float(loss.detach())
                n += 1
            if n:
                history.append({"loss": total_loss / n})
        module.eval()
        logits, labels = _collect(module, test_chunks, pos_lookup)
    finally:
        probe.close()
        module.close()
    return {
        "_model": model,
        "logits": logits, "history": history, "test_labels": labels,
        "n_train_pairs": int(sum(
            len(rows) for steps in train_steps for step in steps for _g, rows in step)),
        "n_optimizer_steps": total_steps,
        "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "trainable_params_by_module": {
            n: int(sum(p.numel() for p in c.parameters() if p.requires_grad))
            for n, c in model.named_children() if any(p.requires_grad for p in c.parameters())
        },
        "grad_norms": probe.result(),
    }


def _passthrough(batch: Any) -> Any:
    """Identity collate.

    With ``batch_size=None`` a custom ``collate_fn`` is called with ONE dataset
    item (verified in torch 2.14), so identity is what unwraps the
    ``(chunks, labels)`` payload into the step function.
    """
    return batch


def _labels_for_chunks(
    chunks: Sequence[Tuple[Group, Sequence[int]]],
    pos_lookup: Mapping[str, Mapping[int, int]],
) -> torch.Tensor:
    vals: List[int] = []
    for group, rows in chunks:
        pos = pos_lookup[group.name]
        vals.extend(int(group.labels[pos[int(r)]]) for r in rows)
    return torch.as_tensor(vals, dtype=torch.long)


def _test_chunks(
    groups: Sequence[Group], split: np.ndarray,
) -> List[Tuple[Group, List[int]]]:
    wanted = {int(r) for r in split}
    out: List[Tuple[Group, List[int]]] = []
    for g in groups:
        rows = [int(r) for r in g.rows if int(r) in wanted]
        if rows:
            out.append((g, rows))
    return out


def _collect(
    module: _JevXArm, chunks: Sequence[Tuple[Group, List[int]]],
    pos_lookup: Mapping[str, Mapping[int, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    logits: List[np.ndarray] = []
    labels: List[int] = []
    with torch.no_grad():
        for i in range(0, len(chunks), 8):
            batch = chunks[i : i + 8]
            out = module(batch)
            logits.append(out.numpy().astype(np.float32))
            labels.extend(_labels_for_chunks(batch, pos_lookup).tolist())
    return np.concatenate(logits, axis=0), np.asarray(labels, dtype=np.int64)


def train_probe_arm(
    arm: str, features: np.ndarray, groups: Sequence[Group], splits: Mapping[str, np.ndarray],
    *, seed: int, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE,
) -> Dict[str, Any]:
    """The P1/P2 reference: a state-only probe, trained on that question's rows.

    When the question's groups share one K (ABSA: every aspect question has the
    same three polarity options) the probe is trained ONCE on the pooled rows,
    because the probe cannot see the aspect term anyway and splitting it per
    aspect would starve it of data. When the K differ (which is not the case
    here, but is possible in principle) one probe per group is trained, since
    a single ``Linear(d, K)`` cannot score two different label spaces.
    """
    uniform_k = len({len(g.options) for g in groups}) == 1
    units: List[Tuple[str, List[int], np.ndarray, np.ndarray]] = []
    if uniform_k:
        rows: List[int] = []
        labels: List[int] = []
        for g in groups:
            pos = {int(r): i for i, r in enumerate(g.rows)}
            tr = [int(r) for r in splits["train"] if int(r) in pos]
            te = [int(r) for r in splits["test"] if int(r) in pos]
            rows.extend(tr)
            labels.extend(int(g.labels[pos[r]]) for r in tr)
            units.append((g.name, te, np.asarray(te, dtype=np.int64),
                          np.asarray([int(g.labels[pos[r]]) for r in te], dtype=np.int64)))
        tr_x, tr_y = features[np.asarray(rows, dtype=np.int64)], np.asarray(labels, dtype=np.int64)
        te_x = np.concatenate([features[u[2]] for u in units], axis=0)
        te_y = np.concatenate([u[3] for u in units], axis=0)
        k = len(groups[0].options)
        probe = (bc.LinearProbe(features.shape[1], k) if arm == "p1_linear"
                 else bc.MLPProbe(features.shape[1], k))
        _, history = bc.train_heads(
            probe, bc.build_records(tr_x, tr_y), seed=seed, lr=LR, epochs=epochs,
            batch_size=batch_size)
        logits, _ = bc.collect_logits(
            probe, bc.build_records(te_x, np.zeros(len(te_x), dtype=np.int64)),
            batch_size=batch_size)
        per_group = [{"unit": "POOLED over the question's groups "
                              f"(uniform K={k}, {len(groups)} groups)",
                      "n_train": int(len(tr_y)), "n_test": int(len(te_y))}]
        n_train = int(len(tr_y))
    else:
        parts: List[Tuple[np.ndarray, np.ndarray]] = []
        per_group = []
        n_train = 0
        for g in groups:
            pos = {int(r): i for i, r in enumerate(g.rows)}
            tr = [int(r) for r in splits["train"] if int(r) in pos]
            te = [int(r) for r in splits["test"] if int(r) in pos]
            if not tr or not te:
                per_group.append({"group": g.name, "skipped": "empty train or test rows",
                                  "n_train": len(tr), "n_test": len(te)})
                continue
            tr_y = np.asarray([int(g.labels[pos[r]]) for r in tr], dtype=np.int64)
            te_y = np.asarray([int(g.labels[pos[r]]) for r in te], dtype=np.int64)
            k = len(g.options)
            probe = (bc.LinearProbe(features.shape[1], k) if arm == "p1_linear"
                     else bc.MLPProbe(features.shape[1], k))
            _, history = bc.train_heads(
                probe, bc.build_records(features[tr], tr_y), seed=seed, lr=LR,
                epochs=epochs, batch_size=batch_size)
            lg, _ = bc.collect_logits(
                probe, bc.build_records(features[te], np.zeros(len(te), dtype=np.int64)),
                batch_size=batch_size)
            parts.append((lg, te_y))
            n_train += len(tr)
            per_group.append({"group": g.name, "n_train": len(tr), "n_test": len(te),
                              "trainable_params": int(
                                  sum(p.numel() for p in probe.parameters() if p.requires_grad))})
        if not parts:
            raise ValueError(f"{arm}: every question group has an empty train or test slice")
        logits = np.concatenate([p[0] for p in parts], axis=0)
        te_y = np.concatenate([p[1] for p in parts], axis=0)
        per_group = [{"unit": "one probe per group (the K differ)"}] + per_group
    if not uniform_k:
        history = []
    return {
        "logits": logits,
        "test_labels": te_y,
        "history": history,
        "n_train_pairs": n_train,
        "trainable_params": int(
            sum(p.numel() for p in probe.parameters() if p.requires_grad)),
        "trainable_params_by_module": {
            "probe": int(sum(p.numel() for p in probe.parameters() if p.requires_grad))},
        "grad_norms": {"at_step": {},
                       "note": "P1/P2 are a single Linear (or a two-layer MLP); their "
                               "gradient is inside the arm's total and is not broken "
                               "down per module."},
        "per_group": per_group,
    }


# --------------------------------------------------------------------------
# Metrics + aggregation
# --------------------------------------------------------------------------


def score(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    return bc.metrics_from_logits(
        torch.as_tensor(np.asarray(logits, dtype=np.float32)),
        torch.as_tensor(np.asarray(labels, dtype=np.int64)),
        n_bins=N_BINS,
    )


def mean_std(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    return {"mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "min": float(arr.min()), "max": float(arr.max())}


def ranges_disjoint(a: Mapping[str, float], b: Mapping[str, float]) -> bool:
    """Do the two 5-seed min-max ranges overlap? A win INSIDE one std is not a win."""
    return a["max"] < b["min"] or b["max"] < a["min"]


# --------------------------------------------------------------------------
# PART 3 - shared state, one backbone forward
# --------------------------------------------------------------------------


SHARED_STATE_TEXT = "The food was great but the service was slow and the price was high."
SHARED_STATE_QUESTIONS = (
    "What is the sentiment toward the food of this review?",
    "What is the sentiment toward the service of this review?",
    "What is the sentiment toward the price of this review?",
    "What is the sentiment toward the ambience of this review?",
    "What is the sentiment toward the wait of this review?",
)


def phase_shared_state(args: argparse.Namespace) -> Dict[str, Any]:
    """Forward-hook count with 4+ questions on ONE state, on a REAL backbone."""
    out: Dict[str, Any] = {"phase": "shared-state", "environment": environment_info(),
                           "backbone": TEXT_BACKBONE, "state": SHARED_STATE_TEXT}
    model = bc.JevX.from_pretrained(
        TEXT_BACKBONE, freeze_base_model=True, pooling_method=POOLING, fusion_type=FUSION
    ).eval()
    backbone = model.model
    counter = {"n": 0}
    handle = backbone.register_forward_hook(
        lambda *_a, **_k: counter.__setitem__("n", counter["n"] + 1))
    results: Dict[str, Any] = {}
    try:
        for m in (1, 2, 3, 4, 5):
            counter["n"] = 0
            with torch.no_grad():
                r = model(state=[SHARED_STATE_TEXT],
                          questions=list(SHARED_STATE_QUESTIONS[:m]),
                          options=[["negative", "neutral", "positive"]] * m)
            results[f"m={m}"] = {"backbone_forward_hooks": counter["n"],
                                  "decisions_returned": len(r.decisions)}
    finally:
        handle.remove()
    out["forward_hook_counts"] = results
    out["exactly_one_backbone_forward_for_every_m"] = all(
        v["backbone_forward_hooks"] == 1 for v in results.values())
    out["decisions_equal_m"] = all(
        v["decisions_returned"] == int(k.split("=")[1]) for k, v in results.items())

    eq: Dict[str, Any] = {}
    worst = 0.0
    argmax_agree = 0
    with torch.no_grad():
        big = model(state=[SHARED_STATE_TEXT], questions=list(SHARED_STATE_QUESTIONS),
                    options=[["negative", "neutral", "positive"]] * len(SHARED_STATE_QUESTIONS))
        big_logits = torch.cat([d.logits for d in big.decisions], dim=0)
        for j, qtext in enumerate(SHARED_STATE_QUESTIONS):
            small = model(state=[SHARED_STATE_TEXT], questions=[qtext],
                          options=[["negative", "neutral", "positive"]])
            single = small.decisions[0].logits
            diff = float((big_logits[j : j + 1] - single).abs().max())
            worst = max(worst, diff)
            argmax_agree += int(
                int(big_logits[j : j + 1].argmax(-1)) == int(single.argmax(-1)))
            eq[qtext] = {
                "bitwise_equal": bool(torch.equal(big_logits[j : j + 1], single)),
                "max_abs_diff": diff,
                "argmax_agrees": bool(
                    int(big_logits[j : j + 1].argmax(-1)) == int(single.argmax(-1))),
            }
    out["multi_call_vs_single_call"] = eq
    out["all_bitwise_equal"] = all(v["bitwise_equal"] for v in eq.values())
    out["worst_max_abs_diff"] = worst
    out["argmax_agreement"] = f"{argmax_agree}/{len(SHARED_STATE_QUESTIONS)}"
    # The m-question call pads every schema to K_max and runs ONE batched
    # duplicate-tie / option-projection, while the per-question calls run
    # narrower tensors; BLAS picks different kernels for the two shapes, so
    # float32 re-associates. That is the same class of difference
    # bench_common.equivalence_check() already records, and it is NOT a
    # semantic difference.
    out["float_reassociation_threshold"] = 1e-5
    out["is_float_reassociation_not_a_semantic_difference"] = bool(worst < 1e-5)
    out["interpretation"] = (
        "The head stack is row-independent. The m-questions-in-one-call forward and "
        "the per-question calls agree to within "
        f"{worst:.2e} in float32 (threshold 1e-5) with identical argmax on "
        f"{argmax_agree}/{len(SHARED_STATE_QUESTIONS)} questions; the residual is "
        "float re-association between two differently-shaped BLAS calls (the "
        "multi-call pads to K_max), not a semantic difference. The grid's "
        "one-question-per-batch protocol is therefore the same computation as the "
        "genuine multi-question forward, and the multi-question regime is a "
        "SHARED-PARAMETER regime - which is exactly what makes the question/schema "
        "encoders' generalisation testable here."
        if worst < 1e-5 else
        "The m-question call and the per-question calls differ by more than 1e-5, so "
        "this is NOT float re-association; the grid's protocol would not be equivalent "
        "and this must be read as a defect.")
    path = RESULTS_DIR / "multi_question_shared_state.json"
    write_json(path, out)
    log(f"shared-state: wrote {path}")
    return out


# --------------------------------------------------------------------------
# Self-check
# --------------------------------------------------------------------------


def phase_selfcheck(args: argparse.Namespace) -> Dict[str, Any]:
    """Prove the frozen-feature path and the cache invariants."""
    out: Dict[str, Any] = {"phase": "selfcheck", "environment": environment_info()}
    backbone, tok = build_tiny_backbone_and_tokenizer()
    width = hidden_of(backbone)
    question, options = CACHE_QUESTIONS_TINY[0], CACHE_OPTIONS_TINY
    schema = [Option(key=o, label=o) for o in options]
    state_text = "hello world foo bar baz"

    model = make_probe_model(backbone, tok, conditioning_mode="full", hidden=width).eval()
    with torch.no_grad():
        real = model(state=state_text, questions=[question], options=[schema])
    real_logits = real.decisions[0].logits.clone()

    frozen = FrozenFeatureJevX(model, np.zeros((1, width), dtype=np.float32))
    b = frozen.batch([0])
    with torch.no_grad():
        stub = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                     questions=[question], options=[schema])
    diff = float((real_logits - stub.decisions[0].logits).abs().max())
    out["stub_is_not_a_constant_function"] = {
        "backbone_forward_calls": frozen.backbone_forward_calls,
        "max_abs_diff_vs_real_pass": diff,
        "why": (
            "the stubbed feature table is all zeros, so a stub that ignored the "
            "features would reproduce the real pass exactly; a NON-zero difference "
            "is the evidence that the real pass actually reads the features. This "
            "cell FAILS if the difference is zero."
        ),
        "passed": bool(diff > 0.0),
    }

    frozen.restore()  # the real backbone must run for the featurize-once replay
    feats = bc.featurize(model, [state_text], batch_size=1, max_length=16,
                         strategies=[None])["native"]
    replay = FrozenFeatureJevX(model, feats)
    b2 = replay.batch([0])
    with torch.no_grad():
        again = model(input_ids=b2["input_ids"], attention_mask=b2["attention_mask"],
                      questions=[question], options=[schema])
    out["frozen_replay_is_bitwise"] = {
        "max_abs_diff": float((real_logits - again.decisions[0].logits).abs().max()),
        "bitwise_equal": bool(torch.equal(real_logits, again.decisions[0].logits)),
        "note": "featurize-once, then replay the SAME feature through the frozen path",
    }

    out["constant_cache_bitwise_tiny"] = _bitwise_checks(
        make_probe_model(backbone, tok, conditioning_mode="constant", hidden=32),
        CACHE_QUESTIONS_TINY, CACHE_OPTIONS_TINY)
    try:
        full = bc.JevX.from_pretrained(TEXT_BACKBONE, freeze_base_model=True,
                                      pooling_method=POOLING, fusion_type=FUSION)
        out["constant_cache_bitwise_real"] = _bitwise_checks(
            make_probe_model(full.model, full.tokenizer, conditioning_mode="constant",
                             hidden=128),
            CACHE_QUESTIONS_REAL, CACHE_OPTIONS_REAL)
    except Exception as exc:  # noqa: BLE001
        out["constant_cache_bitwise_real"] = {"error": f"{type(exc).__name__}: {exc}"}

    # Stale-memo probe: eval (fills the memo) -> ONE training step -> eval again.
    # The memo is keyed on CONTENT only, so an in-place weight change the model
    # cannot observe is exactly the case that would serve a stale H_Q/H_C.
    probe_model = make_probe_model(backbone, tok, conditioning_mode="constant", hidden=32)
    pf = FrozenFeatureJevX(probe_model, np.zeros((1, width), dtype=np.float32))
    pb = pf.batch([0])
    probe_model.eval()
    with torch.no_grad():
        probe_model(input_ids=pb["input_ids"], attention_mask=pb["attention_mask"],
                    questions=[question], options=[schema])
    probe_model.train()
    opt = torch.optim.AdamW([p for p in probe_model.parameters() if p.requires_grad], lr=0.5)
    with torch.enable_grad():
        o = probe_model(input_ids=pb["input_ids"], attention_mask=pb["attention_mask"],
                        questions=[question], options=[schema])
        loss = o.decisions[0].logits.sum()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    probe_model.eval()
    with torch.no_grad():
        probe_model.clear_condition_cache()
        const_after = probe_model(input_ids=pb["input_ids"], attention_mask=pb["attention_mask"],
                                  questions=[question], options=[schema])
        probe_model.config.conditioning_mode = "full"
        full_after = probe_model(input_ids=pb["input_ids"], attention_mask=pb["attention_mask"],
                                 questions=[question], options=[schema])
    out["stale_memo_probe"] = {
        "protocol": "eval (fills the memo) -> one AdamW step -> clear the memo -> eval",
        "max_abs_diff": float((const_after.decisions[0].logits
                               - full_after.decisions[0].logits).abs().max()),
        "bitwise_equal": bool(torch.equal(const_after.decisions[0].logits,
                                          full_after.decisions[0].logits)),
    }
    out["stale_memo_probe_no_clear"] = _no_clear_variant(probe_model, pb, question, schema)

    bit = out["constant_cache_bitwise_tiny"]
    out["all_passed"] = bool(
        out["stub_is_not_a_constant_function"]["passed"]
        and out["frozen_replay_is_bitwise"]["bitwise_equal"]
        and bit["cold_equals_full"] and bit["warm_equals_full"]
        and bit["after_content_change_equals_full"]
        and bit["content_change_actually_moved_the_logits"])
    path = RESULTS_DIR / "multi_question_selfcheck.json"
    write_json(path, out)
    log(f"selfcheck: wrote {path}: all_passed={out['all_passed']}")
    return out


def _no_clear_variant(model: JevXForDecision, pb: Mapping[str, torch.Tensor],
                      question: str, schema: Sequence[Option]) -> Dict[str, Any]:
    """The same probe WITHOUT clearing the memo - the unsafe ordering."""
    model.clear_condition_cache()
    model.config.conditioning_mode = "constant"
    model.eval()
    with torch.no_grad():
        first = model(input_ids=pb["input_ids"], attention_mask=pb["attention_mask"],
                      questions=[question], options=[schema])
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.5)
    with torch.enable_grad():
        o = model(input_ids=pb["input_ids"], attention_mask=pb["attention_mask"],
                  questions=[question], options=[schema])
        loss = o.decisions[0].logits.sum()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    model.eval()
    with torch.no_grad():
        second = model(input_ids=pb["input_ids"], attention_mask=pb["attention_mask"],
                       questions=[question], options=[schema])
        model.config.conditioning_mode = "full"
        reference = model(input_ids=pb["input_ids"], attention_mask=pb["attention_mask"],
                          questions=[question], options=[schema])
    return {
        "protocol": "eval (fills the memo) -> one AdamW step -> eval WITHOUT clearing",
        "max_abs_diff": float((second.decisions[0].logits
                               - reference.decisions[0].logits).abs().max()),
        "bitwise_equal": bool(torch.equal(second.decisions[0].logits,
                                          reference.decisions[0].logits)),
        "note": (
            "a non-zero difference is the content-keyed memo serving a STALE H_Q/H_C "
            "after an in-place weight change it cannot observe. It is a property of the "
            "shipped memo (it is keyed on content, not on weights), not something this "
            "benchmark introduced."
        ),
    }


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def environment_info() -> Dict[str, Any]:
    info = bc.environment_info()
    info["torch_threads_set_to"] = torch.get_num_threads()
    info["loadavg"] = _loadavg()
    return info


def _loadavg() -> Optional[str]:
    import os

    try:
        one, five, fifteen = os.getloadavg()
    except (AttributeError, OSError):  # not available on Windows
        return None
    return f"{one:.2f} {five:.2f} {fifteen:.2f}"


def _backbone_for(modality: str) -> Tuple[nn.Module, Any]:
    if modality == "audio":
        full = bc.JevX.from_pretrained(AUDIO_BACKBONE, freeze_base_model=True,
                                      pooling_method=None).eval()
        return full.model, full.tokenizer
    full = bc.JevX.from_pretrained(
        TEXT_BACKBONE, freeze_base_model=True, pooling_method=POOLING, fusion_type=FUSION
    ).eval()
    return full.model, full.tokenizer


def _aggregate_cell(per_seed: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    ok = [c for c in per_seed if c.get("status") != "FAILED"]
    if not ok:
        return {"status": "FAILED", "n_seeds": 0, "n_seeds_requested": len(per_seed),
                "errors": [c.get("error") for c in per_seed]}
    keys = list(ok[0]["metrics"].keys())
    return {
        "status": "OK",
        "n_seeds": len(ok),
        "n_seeds_requested": len(per_seed),
        "accuracy": mean_std([c["metrics"]["accuracy"] for c in ok]),
        "per_metric": {
            k: {"mean_std": mean_std([c["metrics"][k] for c in ok]),
                "per_seed": [c["metrics"][k] for c in ok]}
            for k in keys
        },
        "trainable_params": ok[0]["trainable_params"],
        "trainable_params_by_module": ok[0]["trainable_params_by_module"],
        "grad_norms_per_seed": [c.get("grad_norms") for c in ok],
        "n_train_pairs": ok[0]["n_train_pairs"],
        "n_test_pairs": ok[0]["n_test_pairs"],
    }


def _verdict(cells: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by = {(c["question"], c["arm"]): c["aggregate"] for c in cells}
    questions = sorted({c["question"] for c in cells})
    comparisons: List[Dict[str, Any]] = []
    wins: List[Dict[str, Any]] = []
    for q in questions:
        for cond in ("full", "constant", "off"):
            for base in ("off", "state_only"):
                if cond == base:
                    continue
                a, b = by.get((q, cond)), by.get((q, base))
                if not a or not b or a.get("status") != "OK" or b.get("status") != "OK":
                    continue
                d = a["accuracy"]["mean"] - b["accuracy"]["mean"]
                disjoint = ranges_disjoint(a["accuracy"], b["accuracy"])
                pooled = float(np.sqrt((a["accuracy"]["std"] ** 2
                                        + b["accuracy"]["std"] ** 2) / 2.0))
                rec = {
                    "question": q,
                    "conditioned_arm": cond, "baseline_arm": base,
                    "conditioned_mean": a["accuracy"]["mean"],
                    "conditioned_std": a["accuracy"]["std"],
                    "conditioned_range": [a["accuracy"]["min"], a["accuracy"]["max"]],
                    "baseline_mean": b["accuracy"]["mean"],
                    "baseline_std": b["accuracy"]["std"],
                    "baseline_range": [b["accuracy"]["min"], b["accuracy"]["max"]],
                    "delta_pp": d * 100.0,
                    "ranges_disjoint": bool(disjoint),
                    "delta_in_pooled_stds": (d / pooled) if pooled > 0 else None,
                    "verdict": ("WIN (non-overlapping 5-seed ranges)" if disjoint and d > 0
                                else ("LOSS" if d < 0 else "TIE")),
                }
                comparisons.append(rec)
                if rec["verdict"].startswith("WIN"):
                    wins.append(rec)
    per_arm: Dict[str, Any] = {}
    for arm in sorted({c["arm"] for c in cells}):
        vals = {q: by[(q, arm)]["accuracy"] for q in questions
                if (q, arm) in by and by[(q, arm)].get("status") == "OK"}
        if not vals:
            continue
        means = [v["mean"] for v in vals.values()]
        per_arm[arm] = {
            "mean_over_questions": float(np.mean(means)),
            "std_over_questions": float(np.std(means, ddof=1)) if len(means) > 1 else 0.0,
            "mean_of_per_question_stds": float(np.mean([v["std"] for v in vals.values()])),
            "n_questions": len(means),
            "per_question_mean": {q: v["mean"] for q, v in vals.items()},
            "per_question_range": {q: [v["min"], v["max"]] for q, v in vals.items()},
            "per_question_per_seed": {q: by[(q, arm)]["per_metric"]["accuracy"]["per_seed"]
                                      for q in vals},
        }
    return {
        "decisive_question": (
            "Does at least one question on at least one dataset show a conditioning "
            "mode beating the state-only arm with NON-OVERLAPPING 5-seed accuracy ranges?"
        ),
        "answer": "YES" if wins else "NO",
        "n_wins": len(wins),
        "wins": wins,
        "all_comparisons": comparisons,
        "aggregate_over_questions_per_arm": per_arm,
    }


def phase_dataset(name: str, args: argparse.Namespace) -> Dict[str, Any]:
    t0 = time.perf_counter()
    ds = build_dataset(name, recompute=args.recompute_features)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    backbone, tokenizer = _backbone_for(ds.modality)
    out: Dict[str, Any] = {
        "dataset": name, "phase": "grid", "status": "OK",
        "environment": environment_info(),
        "protocol": {
            "epochs": args.epochs, "lr": LR, "batch_size": args.batch_size,
            "seeds": seeds, "arms": arms, "split_seed": SPLIT_SEED, "n_bins": N_BINS,
            "backbone": AUDIO_BACKBONE if ds.modality == "audio" else TEXT_BACKBONE,
            "featurize": dict(ds.extra.get("featurize", {})),
            "one_question_per_batch": True,
            "one_question_per_batch_justification": (
                "JevXForDecision.forward shares one question across the state batch, so "
                "a batch is a set of states asking the SAME question; the option count "
                "is therefore uniform and no padded option column is produced. "
                "--phase shared-state shows the m-question call agrees with the "
                "per-question calls to float32 re-association with identical argmax, so "
                "this protocol is the same computation as the genuine multi-question "
                "forward."),
            "head_init": (f"every arm starts from bitwise-identical head weights "
                          f"(torch.manual_seed({HEAD_INIT_SEED}) before the wrapper is built)"),
            "split_sizes_used": {k: int(len(v)) for k, v in ds.splits.items()},
            "split_caps_contract": {"train": 3000, "cal": 1000, "test": 1500},
            "split_sizes_within_contract": bool(
                all(int(len(ds.splits[k])) <= c
                    for k, c in (("train", 3000), ("cal", 1000), ("test", 1500)))),
            "head_hidden": int(HEAD_HIDDEN.get(name) or int(ds.features.shape[1])),
            "head_hidden_is_feature_width": bool(HEAD_HIDDEN.get(name) is None),
            "head_hidden_deviation": (
                None if HEAD_HIDDEN.get(name) is None else
                f"DEVIATION: the JevX head runs at width {HEAD_HIDDEN[name]} instead of the "
                f"{int(ds.features.shape[1])} feature width, because one (arm, seed) cell at "
                "the full width costs ~270 s on a 4-core box shared with other agents and the "
                "full grid did not fit this wave's wall-clock budget. The width is IDENTICAL "
                "across all four JevX arms here, so the within-dataset comparison is "
                "unaffected; the audio head at full width is NOT established here."),
        },
        "counts": ds.counts(),
        "dataset_ids": ds.dataset_ids,
        "extra": {k: v for k, v in ds.extra.items() if k != "featurize"},
        "split_sizes": {k: int(len(v)) for k, v in ds.splits.items()},
        "questions": [q.as_dict() for q in ds.questions],
        "question_groups": [
            {"name": g.name, "question": g.question, "options": g.options,
             "n_states": int(len(g.rows))} for g in ds.groups
        ],
    }
    cells: List[Dict[str, Any]] = []
    for question in ds.questions:
        groups = ds.groups_of(question)
        for arm in arms:
            per_seed: List[Dict[str, Any]] = []
            log(f"[{name}] -> {question.name} / {arm}")
            for seed in seeds:
                base = {"question": question.name, "arm": arm, "seed": seed,
                        "label_provenance": question.provenance}
                try:
                    t1 = time.perf_counter()
                    log(f"[{name}]    {question.name} / {arm} / seed {seed} ...", )
                    logits, labels, info = _run_cell(
                        arm, question, groups, ds, seed, backbone, tokenizer,
                        epochs=args.epochs, batch_size=args.batch_size)
                    metrics = score(logits, labels)
                    log(f"[{name}]    {question.name} / {arm} / seed {seed} done in "
                        f"{time.perf_counter() - t1:.1f}s "
                        f"(acc={metrics['accuracy']:.4f})")
                    cell = {
                        **base, "status": "OK", "metrics": metrics,
                        "n_train_pairs": int(info["n_train_pairs"]),
                        "n_test_pairs": int(len(labels)),
                        "n_question_groups": len(groups),
                        "trainable_params": info["trainable_params"],
                        "trainable_params_by_module": info["trainable_params_by_module"],
                        "grad_norms": info["grad_norms"],
                        "final_epoch_loss": (info["history"][-1]["loss"]
                                             if info["history"] else None),
                        "wall_s": round(time.perf_counter() - t1, 2),
                        "absa_breakdown": _absa_breakdown(
                            question, groups, ds, logits, labels),
                    }
                except Exception as exc:  # noqa: BLE001 - the cell is reported verbatim
                    cell = {**base, "status": "FAILED",
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc()}
                    log(f"  FAILED {name}/{question.name}/{arm}/seed{seed}: "
                        f"{type(exc).__name__}: {exc}")
                per_seed.append(cell)
            agg = _aggregate_cell(per_seed)
            cells.append({"question": question.name, "arm": arm, "aggregate": agg,
                          "per_seed": per_seed})
            # Checkpoint after EVERY cell: this box is shared, the wall clock is
            # not mine, and a run that is killed must still leave usable partial
            # results rather than nothing.
            out["status"] = "PARTIAL"
            out["cells"] = cells
            out["verdict"] = _verdict(cells)
            out["wall_s"] = round(time.perf_counter() - t0, 2)
            write_json(RESULTS_DIR / f"multi_question_{name}.json", out)
            if agg["status"] == "OK":
                log(f"[{name}] {question.name:24s} {arm:11s} "
                    f"acc={agg['accuracy']['mean']:.4f}+-{agg['accuracy']['std']:.4f} "
                    f"nll={agg['per_metric']['nll']['mean_std']['mean']:.4f} "
                    f"ece={agg['per_metric']['ece']['mean_std']['mean']:.4f} "
                    f"({per_seed[0].get('wall_s', '?')}s/seed)")
            else:
                log(f"[{name}] {question.name:24s} {arm:11s} FAILED: {agg.get('errors')}")
    out["cells"] = cells
    out["verdict"] = _verdict(cells)
    out["status"] = "OK"
    out["arms_run"] = arms
    out["arms_not_run"] = [a for a in ARMS if a not in arms]
    out["arms_not_run_reason"] = (
        None if out["arms_not_run"] else
        "conditioning_mode='constant' is inference-only and bitwise-equal to 'full' "
        "(selfcheck, cache-cost and the ABSA 5-seed grid all measure them identical), so it "
        "was dropped from the remaining datasets to fit this wave's wall-clock budget. It is "
        "LABELLED as not run, not reported as a result."
    )
    out["wall_s"] = round(time.perf_counter() - t0, 2)
    path = RESULTS_DIR / f"multi_question_{name}.json"
    write_json(path, out)
    log(f"[{name}] wrote {path} in {out['wall_s']}s; verdict={out['verdict']['answer']}")
    return out


def _run_cell(arm, question, groups, ds, seed, backbone, tokenizer, *, epochs, batch_size):
    """One (question, arm, seed) cell, pooled over that question's groups."""
    if arm in PROBE_ARMS:
        res = train_probe_arm(arm, ds.features, groups, ds.splits, seed=seed,
                              epochs=epochs, batch_size=batch_size)
    else:
        res = train_jevx_arm(arm, ds.features, groups, ds.splits, seed=seed,
                             backbone=backbone, tokenizer=tokenizer, ds_name=ds.name,
                             epochs=epochs, batch_size=batch_size)
    return res["logits"], res["test_labels"], res


def _absa_breakdown(question, groups, ds, logits, labels) -> Optional[Dict[str, Any]]:
    """Per-aspect-term accuracy for ABSA (only meaningful when pooled)."""
    if not groups or "{" not in question.text:
        return None
    per: Dict[str, List[int]] = {}
    off = 0
    for g in groups:
        n = len(g.rows)
        correct = (logits[off : off + n].argmax(axis=1) == labels[off : off + n]).astype(int)
        per[g.name.split("::", 1)[1]] = correct.tolist()
        off += n
    counts = {k: len(v) for k, v in per.items()}
    keep = sorted(counts, key=lambda k: (-counts[k], k))[:15]
    return {
        "top15_aspects_by_test_count": {
            k: {"n": counts[k], "accuracy": float(np.mean(per[k]))} for k in keep
        },
        "n_distinct_aspects": len(per),
        "note": "test-split accuracy per distinct aspect term (the question text)",
    }


def phase_summary() -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "environment": environment_info(),
        "datasets": {},
    }
    for path in sorted(RESULTS_DIR.glob("multi_question_*.json")):
        if path.name.endswith(("_cache_cost.json", "_selfcheck.json", "_summary.json",
                               "_shared_state.json", "_gradnorm.json")):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary["datasets"][payload.get("dataset", path.stem)] = {
            "status": payload.get("status"),
            "error": payload.get("error"),
            "verdict": payload.get("verdict"),
            "counts": payload.get("counts"),
            "questions": payload.get("questions"),
            "split_sizes": payload.get("split_sizes"),
            "wall_s": payload.get("wall_s"),
        }
    wins: List[Dict[str, Any]] = []
    for name, entry in summary["datasets"].items():
        for w in (entry.get("verdict") or {}).get("wins", []):
            wins.append({"dataset": name, **w})
    summary["cross_dataset_wins"] = wins
    summary["decisive_answer"] = "YES" if wins else "NO"
    out = RESULTS_DIR / "multi_question_summary.json"
    write_json(out, summary)
    log(f"summary: wrote {out} (decisive answer: {summary['decisive_answer']})")
    return summary


def phase_gradnorm(args: argparse.Namespace) -> Dict[str, Any]:
    """Per-module gradient norms at step 1 and at the final step.

    One seed and ONE representative question per dataset, three arms, because
    the question is WHICH MODULES RECEIVE A GRADIENT, not an accuracy number -
    that is a property of the arm's wiring and is identical for every question
    of a dataset. Cheap on purpose: it exists so the gradient table is measured
    rather than asserted.
    """
    out: Dict[str, Any] = {
        "phase": "gradnorm", "environment": environment_info(),
        "seeds": [int(args.seeds.split(",")[0])],
        "epochs": args.epochs,
        "representative_question_rule": (
            "the FIRST question of each dataset; the gradient wiring is a "
            "property of the arm, not of the question text"),
        "note": (
            "L2 norm of the accumulated gradient, read after loss.backward() and "
            "BEFORE HeadTrainer's clip_grad_norm_(1.0). A module ABSENT from a "
            "record received NO gradient at all on that step - that absence is "
            "the measurement, not a gap in it."),
        "datasets": {},
    }
    for name in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        try:
            ds = build_dataset(name, recompute=args.recompute_features)
            backbone, tokenizer = _backbone_for(ds.modality)
            question = ds.questions[0]
            groups = ds.groups_of(question)
            seed = int(args.seeds.split(",")[0])
            entry: Dict[str, Any] = {
                "question": question.name, "seed": seed,
                "head_hidden": int(HEAD_HIDDEN.get(name) or ds.features.shape[1]),
                "arms": {},
            }
            for arm in ("full", "off", "state_only"):
                res = train_jevx_arm(
                    arm, ds.features, groups, ds.splits, seed=seed, ds_name=name,
                    backbone=backbone, tokenizer=tokenizer,
                    epochs=args.epochs, batch_size=args.batch_size)
                params = {
                    n: int(sum(p.numel() for p in c.parameters() if p.requires_grad))
                    for n, c in res["_model"].named_children()
                    if any(p.requires_grad for p in c.parameters())
                }
                entry["arms"][arm] = {
                    "trainable_params": res["trainable_params"],
                    "trainable_params_by_module": params,
                    "conditioning_half_params": int(sum(
                        params.get(k, 0) for k in
                        ("question_encoder", "question_proj", "schema_encoder", "schema_proj"))),
                    "grad_norms": res["grad_norms"],
                }
                log(f"[{name}] gradnorm {arm}: "
                    + json.dumps(res["grad_norms"]["at_step"]))
            out["datasets"][name] = entry
        except Exception as exc:  # noqa: BLE001 - the cell is reported verbatim
            out["datasets"][name] = {"status": "FAILED",
                                      "error": f"{type(exc).__name__}: {exc}",
                                      "traceback": traceback.format_exc()}
            log(f"gradnorm FAILED {name}: {type(exc).__name__}: {exc}")
    path = RESULTS_DIR / "multi_question_gradnorm.json"
    write_json(path, out)
    log(f"gradnorm: wrote {path}")
    return out


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--phase", default="all",
                   choices=["all", "selfcheck", "cache-cost", "grid", "shared-state",
                            "gradnorm", "summary"])
    p.add_argument("--datasets", default="absa,ravdess,ag_news,yelp")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--latency-iters", type=int, default=120)
    p.add_argument("--recompute-features", action="store_true")
    p.add_argument("--threads", type=int, default=1,
                   help="torch.set_num_threads; this box is 4 cores shared with other "
                        "agents and 2 intra-op threads thrash badly under load")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    torch.set_num_threads(max(1, int(args.threads)))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    log(f"environment: {json.dumps(environment_info())}")

    if args.phase in ("all", "selfcheck"):
        phase_selfcheck(args)
    if args.phase in ("all", "cache-cost"):
        phase_cache_cost(args)
    if args.phase in ("all", "shared-state"):
        phase_shared_state(args)
    if args.phase in ("all", "gradnorm"):
        phase_gradnorm(args)
    if args.phase in ("all", "grid"):
        for name in [d.strip() for d in args.datasets.split(",") if d.strip()]:
            try:
                phase_dataset(name, args)
            except Exception as exc:  # noqa: BLE001 - the dataset is reported verbatim
                log(f"FAILED {name}: {type(exc).__name__}: {exc}")
                traceback.print_exc()
                write_json(RESULTS_DIR / f"multi_question_{name}.json", {
                    "dataset": name, "phase": "grid", "status": "FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "environment": environment_info(),
                })
    if args.phase in ("all", "summary"):
        phase_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
