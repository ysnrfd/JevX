# Copyright (c) 2026 ysnrfd
"""Real CPU latency benchmarks for JevX (phases 55-56, 88).

WHAT IT MEASURES (every number written to benchmarks/results/latency_*.json):

* (a) stage breakdown of ONE .choice call (state ~64 tokens, 1 question, K=4)
  for hf-internal-testing/tiny-random-bert, plus one cross-check run on
  sshleifer/tiny-gpt2 (decoder family). Stages are timed via SEPARATE calls
  to the public internals (adapter.prepare_inputs / extract_representation /
  question_encoder.encode_texts / schema_encoder.encode / fusion / head /
  calibration+confidence) — the model code is NOT modified and carries no
  instrumentation hooks (no return_timings flag needed);
* (b) model load: JevX.from_pretrained cold (empty cache dir) vs warm
  (pre-populated cache dir, second load in the same process);
* (c) T(m) for m in {1,2,4,8,16,32} questions over ONE state with fixed K=4
  (shared-state mode) AND the naive baseline of m separate .choice calls
  (each re-running the backbone); marginal cost = OLS slope of the medians +
  R^2; speedup factor table; peak RSS of both modes (psutil sampler thread);
* (d) batching: B in {1,2,4,8} states x 1 question -> throughput (pairs/s);
* (e) stage table that isolates JevX-head cost vs backbone cost (phase 88).

METHOD: time.perf_counter, warmup >= 10 iters, >= 30 timed iters per cell,
median + p95 + IQR reported. This host runs CONCURRENT AGENTS, so wall-clock
cells are noisy: every cell records median/p95/IQR (IQR is the noise
indicator), and NO claim is made at sub-millisecond precision.

Idempotent: re-running overwrites only benchmarks/results/latency_*.json.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jevx import JevX, JevXForDecision  # noqa: E402

BERT_ID = "hf-internal-testing/tiny-random-bert"
GPT2_ID = "sshleifer/tiny-gpt2"

RESULTS_DIR = Path(__file__).resolve().parent / "results"
WARMUP_ITERS = 10
TIMED_ITERS = 30

# ---------------------------------------------------------------------------
# Workload construction
# ---------------------------------------------------------------------------


def make_state(target_tokens: int, tokenizer: Any) -> str:
    """A text state whose tokenized length is as close to target as possible."""
    # Build from tokenizer vocab entries so every word is in-vocabulary.
    vocab = tokenizer.get_vocab()
    words = [
        w for w in sorted(vocab)
        if w not in {tokenizer.pad_token, tokenizer.cls_token, tokenizer.sep_token,
                     tokenizer.bos_token, tokenizer.eos_token, tokenizer.unk_token,
                     tokenizer.mask_token, None}
        and not w.startswith("[") and not w.startswith("##")
    ]
    if not words:
        words = ["hello", "world"]
    text_parts: List[str] = []
    for i in range(100_000):
        text_parts.append(words[i % len(words)])
        if i % 8 == 7:
            length = len(tokenizer(" ".join(text_parts))["input_ids"])
            if length >= target_tokens:
                return " ".join(text_parts)
    return " ".join(text_parts)


def make_questions(m: int) -> List[str]:
    return [f"question number {i} about the topic" for i in range(m)]


def make_options(m: int, k: int) -> List[List[str]]:
    return [[f"opt{j}-{c}" for c in range(k)] for j in range(m)]


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------


def percentile(sorted_vals: Sequence[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    index = q * (len(sorted_vals) - 1)
    lo = int(index)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = index - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def summarize(samples_ms: List[float]) -> Dict[str, Any]:
    ordered = sorted(samples_ms)
    med = statistics.median(ordered)
    q1 = percentile(ordered, 0.25)
    q3 = percentile(ordered, 0.75)
    return {
        "median_ms": med,
        "p95_ms": percentile(ordered, 0.95),
        "iqr_ms": q3 - q1,
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "n": len(ordered),
        # IQR/median: relative dispersion — the run-noise indicator for this
        # multi-agent host. >0.25 flags a noisy cell; medians stay usable.
        "relative_iqr": (q3 - q1) / med if med > 0 else 0.0,
        "noisy": bool(med > 0 and (q3 - q1) / med > 0.25),
    }


def time_call(fn: Callable[[], Any], warmup: int = WARMUP_ITERS,
              iters: int = TIMED_ITERS) -> Dict[str, Any]:
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        samples: List[float] = []
        for _ in range(iters):
            start = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - start) * 1000.0)
    stats = summarize(samples)
    # noisy cells are retried once; keep the less noisy of the two batches
    if stats["noisy"]:
        with torch.no_grad():
            samples2: List[float] = []
            for _ in range(iters):
                start = time.perf_counter()
                fn()
                samples2.append((time.perf_counter() - start) * 1000.0)
        retry = summarize(samples2)
        stats = min((stats, retry), key=lambda s: s["iqr_ms"])
        stats["retried"] = True
    else:
        stats["retried"] = False
    return stats


class RssSampler:
    """Peak-RSS sampler thread (psutil when importable, else ctypes fallback)."""

    def __init__(self, interval: float = 0.005) -> None:
        self.interval = interval
        self.peak_bytes = 0
        self.available = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._read: Callable[[], int]
        try:
            import psutil

            process = psutil.Process()
            self._read = lambda: int(process.memory_info().rss)
            self.available = True
        except Exception:  # pragma: no cover - psutil present in this env
            try:
                import ctypes

                class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                    _fields_ = [
                        ("cb", ctypes.c_uint32),
                        ("PageFaultCount", ctypes.c_uint32),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t),
                    ]

                k32 = ctypes.windll.kernel32
                GetCurrentProcess = k32.GetCurrentProcess
                GetProcessMemoryInfo = k32.GetProcessMemoryInfo
                GetProcessMemoryInfo.restype = ctypes.c_bool

                def _read() -> int:
                    counters = PROCESS_MEMORY_COUNTERS()
                    counters.cb = ctypes.sizeof(counters)
                    ok = GetProcessMemoryInfo(
                        GetCurrentProcess(), ctypes.byref(counters), counters.cb
                    )
                    if not ok:
                        return 0
                    return int(counters.WorkingSetSize)

                self._read = _read
                self.available = True
            except Exception:  # pragma: no cover - documented fallback
                self._read = lambda: 0

    def _loop(self) -> None:
        while not self._stop.is_set():
            rss = self._read()
            if rss > self.peak_bytes:
                self.peak_bytes = rss
            time.sleep(self.interval)

    def __enter__(self) -> "RssSampler":
        self.peak_bytes = self._read()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    @property
    def peak_mb(self) -> float:
        return self.peak_bytes / 1e6


def rss_mb() -> float:
    sampler = RssSampler(interval=10.0)
    return sampler._read() / 1e6


# ---------------------------------------------------------------------------
# Environment block
# ---------------------------------------------------------------------------


def environment_block() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "cpu_count_logical": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "device": "cpu",
        "gpu": "none (CPU-only build: torch 2.14.0+cpu)",
    }
    try:
        info["processor"] = platform.processor() or platform.machine()
    except Exception:
        info["processor"] = "unknown"
    try:
        import psutil

        info["ram_total_mb"] = round(psutil.virtual_memory().total / 1e6, 1)
        info["cpu_brand"] = psutil.cpu_info().brand_raw
    except Exception:
        info["ram_total_mb"] = None
    info["timing_note"] = (
        "Concurrent agents share this host: wall-clock cells are noisy. "
        "All figures are medians over >=30 timed iterations after >=10 warmup "
        "iterations, with p95 and IQR reported; noisy cells (IQR/median > 0.25) "
        "are retried once. No sub-millisecond precision is claimed."
    )
    return info


# ---------------------------------------------------------------------------
# Load benchmark (b)
# ---------------------------------------------------------------------------


def benchmark_load(repeats: int = 3) -> Dict[str, Any]:
    """Cold (empty cache dir) vs warm (populated cache dir) load time."""
    results: Dict[str, Any] = {}
    cold_dir = Path(tempfile.mkdtemp(prefix="jevx_cold_cache_"))
    warm_dir = Path(tempfile.mkdtemp(prefix="jevx_warm_cache_"))
    try:
        # warm cache: first load populates, second load is the measured warm cell
        JevX.from_pretrained(BERT_ID, cache_dir=str(warm_dir))
        warm_samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            model = JevX.from_pretrained(BERT_ID, cache_dir=str(warm_dir))
            warm_samples.append((time.perf_counter() - start) * 1000.0)
            del model
        results["warm_cache_ms"] = summarize(warm_samples)

        # cold cache: every sample downloads weights+tokenizer into a fresh dir
        cold_samples = []
        for i in range(max(1, repeats - 1)):
            fresh = cold_dir / f"run{i}"
            start = time.perf_counter()
            model = JevX.from_pretrained(BERT_ID, cache_dir=str(fresh))
            cold_samples.append((time.perf_counter() - start) * 1000.0)
            del model
        results["cold_cache_ms"] = summarize(cold_samples)
        results["cold_includes_download"] = True
        results["model_id"] = BERT_ID
    finally:
        shutil.rmtree(cold_dir, ignore_errors=True)
        shutil.rmtree(warm_dir, ignore_errors=True)
    return results


# ---------------------------------------------------------------------------
# Stage breakdown (a, e)
# ---------------------------------------------------------------------------


def benchmark_stages(model: JevXForDecision, state: str, k: int = 4) -> Dict[str, Any]:
    """Time each pipeline stage via separate calls to public internals."""
    adapter = model.adapter
    inputs = adapter.prepare_inputs(state, tokenizer=model.tokenizer,
                                    processor=model.processor, model=model.model)
    question_texts = make_questions(1)
    options = make_options(1, k)
    from jevx.questions import Option

    schema_rows = [[Option(key=key, label=key) for key in row] for row in options]

    stages: Dict[str, Any] = {}

    stages["1_preprocess_tokenizer"] = time_call(
        lambda: adapter.prepare_inputs(state, tokenizer=model.tokenizer,
                                       processor=model.processor, model=model.model)
    )

    def backbone_forward() -> Any:
        return adapter.extract_representation(
            model.model, inputs, strategy=model.config.pooling_method
        )

    stages["2_backbone_forward_and_adapter_pool"] = time_call(backbone_forward)
    representation = backbone_forward()
    pooled = representation.pooled
    assert pooled is not None

    stages["3_state_projection"] = time_call(lambda: model.state_proj(pooled))

    stages["4_question_encode"] = time_call(
        lambda: model.question_proj(
            model.question_encoder(model.question_encoder.encode_texts(
                question_texts, tokenizer=model.tokenizer))
        )
    )

    stages["5_schema_encode"] = time_call(
        lambda: model.schema_proj(model.schema_encoder.encode(
            schema_rows, tokenizer=model.tokenizer)[0])
    )

    state_rows = model.state_proj(pooled)
    question_rows = model.question_proj(
        model.question_encoder(model.question_encoder.encode_texts(
            question_texts, tokenizer=model.tokenizer))
    )
    option_features = model.schema_proj(
        model.schema_encoder.encode(schema_rows, tokenizer=model.tokenizer)[0])
    option_mask = torch.ones(1, k, dtype=torch.bool)

    stages["6_fusion"] = time_call(
        lambda: model.fusion(state_rows, question_rows, option_features, option_mask)
    )
    fused_output = model.fusion(state_rows, question_rows, option_features, option_mask)
    head = model.heads["choice"]

    stages["7_head_choice"] = time_call(
        lambda: head(fused_output.fused, fused_output.option_features,
                     option_mask=option_mask)
    )
    head_output = head(fused_output.fused, fused_output.option_features,
                       option_mask=option_mask)

    stages["8_calibration_and_confidence"] = time_call(
        lambda: (
            model.calibration_engine.apply(head_output.logits, "logits")
            if model.calibration_engine.method != "none"
            and model.calibration_engine.is_fitted else head_output.logits,
            model.confidence(head_output.probabilities, logits=head_output.logits,
                             option_mask=option_mask, fused=fused_output.fused),
        )
    )

    # whole-call reference: the full .choice API
    stages["0_full_choice_call"] = time_call(
        lambda: model.choice(state, question_texts, options=options)
    )
    return stages


# ---------------------------------------------------------------------------
# T(m) shared-state vs naive m requests (c)
# ---------------------------------------------------------------------------


def benchmark_question_scaling(model: JevXForDecision, state: str,
                               m_values: Sequence[int] = (1, 2, 4, 8, 16, 32),
                               k: int = 4) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for m in m_values:
        questions = make_questions(m)
        options = make_options(m, k)

        shared_sampler = RssSampler()
        with shared_sampler:
            shared = time_call(
                lambda: model.choice(state, questions, options=options)
            )
        rss_before = rss_mb()

        naive_sampler = RssSampler()
        per_question = []
        with naive_sampler:
            # naive baseline: m SEPARATE .choice calls, each re-running the backbone
            for _ in range(WARMUP_ITERS):
                for j in range(m):
                    model.choice(state, [questions[j]], options=[options[j]])
            for _ in range(TIMED_ITERS):
                total = 0.0
                for j in range(m):
                    start = time.perf_counter()
                    model.choice(state, [questions[j]], options=[options[j]])
                    total += (time.perf_counter() - start) * 1000.0
                per_question.append(total)
        naive_stats = summarize(per_question)
        if naive_stats["noisy"]:
            samples2 = []
            for _ in range(TIMED_ITERS):
                total = 0.0
                for j in range(m):
                    start = time.perf_counter()
                    model.choice(state, [questions[j]], options=[options[j]])
                    total += (time.perf_counter() - start) * 1000.0
                samples2.append(total)
            retry = summarize(samples2)
            naive_stats = min((naive_stats, retry), key=lambda s: s["iqr_ms"])
            naive_stats["retried"] = True

        rows.append({
            "m": m,
            "shared_ms": shared,
            "naive_ms": naive_stats,
            "speedup": naive_stats["median_ms"] / shared["median_ms"]
            if shared["median_ms"] > 0 else None,
            "shared_peak_rss_mb": shared_sampler.peak_mb,
            "naive_peak_rss_mb": naive_sampler.peak_mb,
            "rss_before_mb": rss_before,
        })

    # OLS over the shared-mode medians: T(m) = a + b*m
    xs = [float(r["m"]) for r in rows]
    ys = [r["shared_ms"]["median_ms"] for r in rows]
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx else 0.0
    intercept = mean_y - slope * mean_x
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot else 1.0

    # Observed vs the shared-state complexity claim T(m) ~ T(1) + (m-1)*slope:
    # residual = measured - predicted (reported, no unmeasured claims).
    t1 = ys[0]
    residuals = [
        {"m": int(x), "measured_ms": y,
         "predicted_ms": t1 + (x - 1) * slope,
         "residual_ms": y - (t1 + (x - 1) * slope)}
        for x, y in zip(xs, ys)
    ]
    return {
        "rows": rows,
        "ols": {
            "slope_ms_per_question": slope,
            "intercept_ms": intercept,
            "r_squared": r_squared,
            "t1_ms": t1,
        },
        "shared_state_claim_residuals": residuals,
        "baseline": "m separate .choice calls, one question each (backbone re-runs per call)",
    }


# ---------------------------------------------------------------------------
# Batch throughput (d)
# ---------------------------------------------------------------------------


def benchmark_batching(model: JevXForDecision, states: Sequence[str],
                       b_values: Sequence[int] = (1, 2, 4, 8)) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    question = ["throughput question"]
    options = [["a", "b", "c", "d"]]
    for b in b_values:
        batch = list(states[:b])
        if len(batch) < b:
            batch = (batch * b)[:b]
        stats = time_call(lambda: model.choice(batch, question, options=options))
        median_s = stats["median_ms"] / 1000.0
        rows.append({
            "B": b,
            "latency_ms": stats,
            "pairs_per_s": b / median_s if median_s > 0 else None,
        })
    return {"rows": rows}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(model_id: str, tag: str) -> Dict[str, Any]:
    output: Dict[str, Any] = {
        "environment": environment_block(),
        "model_id": model_id,
        "warmup_iters": WARMUP_ITERS,
        "timed_iters": TIMED_ITERS,
    }
    print(f"[{tag}] loading {model_id} ...")
    load_start = time.perf_counter()
    model = JevX.from_pretrained(model_id)
    model.eval()
    output["first_load_s"] = round(time.perf_counter() - load_start, 3)

    tokenizer = model.tokenizer
    if tokenizer is None:
        raise RuntimeError(f"{model_id} resolved no tokenizer; cannot build text state")
    state = make_state(64, tokenizer)
    token_len = len(tokenizer(state)["input_ids"])
    output["state_tokens"] = token_len
    print(f"[{tag}] state tokenized to {token_len} tokens (target 64)")

    print(f"[{tag}] (a) stage breakdown ...")
    output["stages"] = benchmark_stages(model, state)

    print(f"[{tag}] (c) T(m) shared vs naive ...")
    output["question_scaling"] = benchmark_question_scaling(model, state)

    print(f"[{tag}] (d) batch sweep ...")
    states = [state] + [
        make_state(64, tokenizer) for _ in range(7)
    ]
    output["batching"] = benchmark_batching(model, states)

    del model
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-load", action="store_true",
                        help="skip the cold/warm cache load benchmark")
    parser.add_argument("--quick", action="store_true",
                        help="fewer iterations (smoke run)")
    args = parser.parse_args()
    if args.quick:
        globals()["WARMUP_ITERS"] = 3
        globals()["TIMED_ITERS"] = 10

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    bert_result = run(BERT_ID, "tiny-random-bert")
    if not args.skip_load:
        print("[load] cold vs warm cache ...")
        bert_result["model_load"] = benchmark_load()

    gpt2_result = run(GPT2_ID, "tiny-gpt2")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for payload, name in (
        (bert_result, f"latency_tiny-random-bert_{stamp}.json"),
        (gpt2_result, f"latency_tiny-gpt2_{stamp}.json"),
    ):
        target = RESULTS_DIR / name
        target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"wrote {target}")

    # also write a stable "latest" alias for the report generator
    (RESULTS_DIR / "latency_tiny-random-bert_latest.json").write_text(
        json.dumps(bert_result, indent=2, default=str), encoding="utf-8")
    (RESULTS_DIR / "latency_tiny-gpt2_latest.json").write_text(
        json.dumps(gpt2_result, indent=2, default=str), encoding="utf-8")

    # console summary
    ols = bert_result["question_scaling"]["ols"]
    print("\nT(m) shared-state (tiny-random-bert):")
    print(f"{'m':>4} {'shared med ms':>14} {'IQR':>8} {'naive med ms':>13} {'speedup':>8}")
    for row in bert_result["question_scaling"]["rows"]:
        print(f"{row['m']:>4} {row['shared_ms']['median_ms']:>14.3f} "
              f"{row['shared_ms']['iqr_ms']:>8.3f} "
              f"{row['naive_ms']['median_ms']:>13.3f} "
              f"{row['speedup']:>8.2f}x")
    print(f"slope = {ols['slope_ms_per_question']:.4f} ms/question, "
          f"R^2 = {ols['r_squared']:.4f}")


if __name__ == "__main__":
    main()
