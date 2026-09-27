# Copyright (c) 2026 ysnrfd
"""CPU memory footprint benchmarks for JevX (phase 89).

WHAT IT MEASURES (written to benchmarks/results/memory_*.json, MB = 10^6 bytes):

* model load footprint: RSS before/after JevX.from_pretrained + python-side
  tracemalloc peak during the load;
* steady-state forward peak (batch 1): tracemalloc peak AND RSS delta across
  repeated eval-mode .choice calls after warmup;
* forward peak with m=32 questions in ONE call vs m=32 SEPARATE calls
  (shared-state vs naive): tracemalloc peak + RSS delta for both modes;
* python-side peak via tracemalloc (allocation accounting for Python objects
  and tensor metadata) alongside RSS (process-resident bytes, which include
  PyTorch's native tensor storage that tracemalloc cannot see).

HONEST SCOPE: this host is CPU-only (torch 2.14.0+cpu, no CUDA device), so
CUDA memory and optimizer-state memory are N/A — no GPU allocations exist to
measure and none are simulated here. tracemalloc instruments the Python
allocator: tensors allocated by torch's C++ allocator are visible in RSS but
mostly NOT in tracemalloc; both numbers are reported side by side rather than
pretending either one is complete.

Idempotent: re-running overwrites only benchmarks/results/memory_*.json.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jevx import JevX  # noqa: E402

BERT_ID = "hf-internal-testing/tiny-random-bert"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def rss_mb() -> float:
    """Process resident set size in MB (psutil; ctypes fallback)."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1e6
    except Exception:  # pragma: no cover
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
            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(counters)
            k32.GetProcessMemoryInfo(
                k32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
            )
            return counters.WorkingSetSize / 1e6
        except Exception:
            return float("nan")


def tracemalloc_peak_mb(fn) -> tuple[Any, float]:
    """Run fn with tracemalloc, return (result, python-side peak MB delta)."""
    gc.collect()
    tracemalloc.start()
    baseline_current, _ = tracemalloc.get_traced_memory()
    result = fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # peak is the max absolute traced bytes during the call; delta over the
    # pre-call baseline isolates what the call itself allocated.
    return result, max(0.0, peak - baseline_current) / 1e6


def environment_block() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "cpu_count_logical": os.cpu_count(),
        "gpu": "none (CPU-only: torch 2.14.0+cpu)",
        "cuda_available": torch.cuda.is_available(),
        "cuda_memory": "N/A — no CUDA device; CUDA allocator stats do not exist here",
        "optimizer_memory": "N/A — inference-only benchmark; no optimizer is constructed",
        "units": "MB = 10^6 bytes",
        "scope_note": (
            "tracemalloc covers the Python allocator (objects, tensor metadata); "
            "torch tensor storage lives in torch's native allocator and shows up "
            "in RSS but mostly not in tracemalloc — both are reported."
        ),
    }
    try:
        import psutil

        info["ram_total_mb"] = round(psutil.virtual_memory().total / 1e6, 1)
        info["cpu_brand"] = psutil.cpu_info().brand_raw
        info["rss_source"] = "psutil"
    except Exception:
        info["rss_source"] = "ctypes fallback"
    return info


def build_texts(model, count: int, tokens: int = 64) -> List[str]:
    tokenizer = model.tokenizer
    vocab = tokenizer.get_vocab()
    words = [
        w for w in sorted(vocab)
        if w not in {tokenizer.pad_token, tokenizer.cls_token, tokenizer.sep_token,
                     tokenizer.bos_token, tokenizer.eos_token, tokenizer.unk_token,
                     tokenizer.mask_token, None}
        and not w.startswith("[") and not w.startswith("##")
    ] or ["hello", "world"]
    texts = []
    for item in range(count):
        parts = [words[(item * 7 + i) % len(words)] for i in range(tokens)]
        texts.append(" ".join(parts))
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=int, default=32)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output: Dict[str, Any] = {"environment": environment_block(), "model_id": BERT_ID}

    # ---- 1. model load footprint -----------------------------------------
    gc.collect()
    rss_before = rss_mb()
    model, load_py_peak = tracemalloc_peak_mb(lambda: JevX.from_pretrained(BERT_ID))
    model.eval()
    gc.collect()
    rss_after_load = rss_mb()
    param_mb = sum(
        p.numel() * p.element_size() for p in model.parameters()
    ) / 1e6
    buffer_mb = sum(
        b.numel() * b.element_size() for b in model.buffers()
    ) / 1e6
    output["model_load"] = {
        "rss_before_mb": rss_before,
        "rss_after_mb": rss_after_load,
        "rss_delta_mb": rss_after_load - rss_before,
        "python_peak_during_load_mb": load_py_peak,
        "param_bytes_mb": param_mb,
        "buffer_bytes_mb": buffer_mb,
    }
    print(f"load: rss_delta={rss_after_load - rss_before:.1f} MB, "
          f"py_peak={load_py_peak:.1f} MB, params={param_mb:.1f} MB")

    # ---- 2. steady-state forward peak (batch 1) --------------------------
    state = build_texts(model, 1)[0]
    question = ["memory probe question"]
    options = [["a", "b", "c", "d"]]
    with torch.no_grad():  # warmup
        for _ in range(10):
            model.choice(state, question, options=options)
    gc.collect()
    rss_pre_steady = rss_mb()

    def steady_forward() -> None:
        with torch.no_grad():
            for _ in range(50):
                model.choice(state, question, options=options)

    _, steady_py_peak = tracemalloc_peak_mb(steady_forward)
    rss_post_steady = rss_mb()
    output["steady_forward_batch1"] = {
        "iterations": 50,
        "python_peak_mb": steady_py_peak,
        "rss_before_mb": rss_pre_steady,
        "rss_after_mb": rss_post_steady,
        "rss_delta_mb": rss_post_steady - rss_pre_steady,
        "note": "RSS delta after 50 identical eval calls ~ allocator retention, "
                "not per-call growth (a leak would grow linearly).",
    }
    print(f"steady: py_peak={steady_py_peak:.2f} MB, "
          f"rss_delta={rss_post_steady - rss_pre_steady:.2f} MB")

    # ---- 3. m=32 single call vs 32 separate calls ------------------------
    m = args.questions
    m_texts = [f"question number {i} about the topic" for i in range(m)]
    m_options = options * m
    with torch.no_grad():  # warmup both modes
        model.choice(state, m_texts, options=m_options)
        for j in range(m):
            model.choice(state, [m_texts[j]], options=[options[0]])

    gc.collect()
    rss_pre_shared = rss_mb()

    def shared_call():
        with torch.no_grad():
            return model.choice(state, m_texts, options=m_options)

    shared_out, shared_py_peak = tracemalloc_peak_mb(shared_call)
    rss_post_shared = rss_mb()

    gc.collect()
    rss_pre_naive = rss_mb()

    def naive_calls():
        with torch.no_grad():
            last = None
            for j in range(m):
                last = model.choice(state, [m_texts[j]], options=[options[0]])
            return last

    naive_out, naive_py_peak = tracemalloc_peak_mb(naive_calls)
    rss_post_naive = rss_mb()

    # shared call answers all m questions in one output; the naive mode ends
    # with the LAST single-question call (1 decision) — both are expected.
    assert len(shared_out.decisions) == m
    assert len(naive_out.decisions) == 1
    output["m_questions"] = {
        "m": m,
        "shared_single_call": {
            "python_peak_mb": shared_py_peak,
            "rss_before_mb": rss_pre_shared,
            "rss_after_mb": rss_post_shared,
            "rss_delta_mb": rss_post_shared - rss_pre_shared,
        },
        "naive_m_separate_calls": {
            "python_peak_mb": naive_py_peak,
            "rss_before_mb": rss_pre_naive,
            "rss_after_mb": rss_post_naive,
            "rss_delta_mb": rss_post_naive - rss_pre_naive,
        },
        "python_peak_ratio_shared_over_naive": (
            shared_py_peak / naive_py_peak if naive_py_peak > 0 else None
        ),
    }
    print(f"m={m}: shared py_peak={shared_py_peak:.2f} MB vs "
          f"naive py_peak={naive_py_peak:.2f} MB; "
          f"shared rss_delta={rss_post_shared - rss_pre_shared:.2f} MB, "
          f"naive rss_delta={rss_post_naive - rss_pre_naive:.2f} MB")

    output["final_rss_mb"] = rss_mb()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = RESULTS_DIR / f"memory_tiny-random-bert_{stamp}.json"
    target.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"wrote {target}")
    (RESULTS_DIR / "memory_tiny-random-bert_latest.json").write_text(
        json.dumps(output, indent=2, default=str), encoding="utf-8")

    del model
    gc.collect()


if __name__ == "__main__":
    main()
