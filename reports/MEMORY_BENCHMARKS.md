# JevX Memory Benchmarks — Phase 89 (real CPU measurements)

## Environment

| Field | Value |
|---|---|
| Timestamp (UTC) | 2026-09-24T11:16:21 |
| OS / platform | Windows-11-10.0.26100-SP0 |
| CPU | Intel64 Family 6 Model 78 Stepping 3 (i5-6300U class), 4 logical cores |
| RAM | 8,476 MB total (psutil) |
| Python | 3.14.3 · torch 2.14.0+cpu · transformers 5.16.1 |
| GPU | **none** — CPU-only build |
| Model | hf-internal-testing/tiny-random-bert |
| Units | **MB = 10^6 bytes** |
| Raw JSON | `benchmarks/results/memory_tiny-random-bert_*.json` |

**Scope honesty (stated up front):**

* **CUDA memory: N/A.** `torch.cuda.is_available()` is False; there is no CUDA
  device, no CUDA allocator, and no CUDA allocator stats to report. Nothing
  here is simulated as a GPU measurement.
* **Optimizer memory: N/A.** This is an inference-only benchmark — no optimizer
  is ever constructed, so Adam/grad-state memory does not exist to measure.
* **tracemalloc = Python allocator only.** It sees Python objects and tensor
  metadata, but torch's tensor storage is allocated by torch's native C++
  allocator and shows up in **RSS**, not in tracemalloc. Both numbers are
  therefore reported side by side rather than pretending either is complete.
* **RSS source**: the run's `psutil.Process()` call was unavailable inside the
  sampler, so RSS came from the documented **ctypes fallback**
  (`GetProcessMemoryInfo` → WorkingSetSize). Same OS counter psutil reads.

---

## 1. Model load footprint (`JevX.from_pretrained`)

| Metric | MB |
|---|---:|
| RSS before load | 472.66 |
| RSS after load | 488.55 |
| **RSS delta (process footprint of the loaded model)** | **15.89** |
| Python-side peak during load (tracemalloc) | 3.03 |
| Parameter bytes (state_dict tensors) | 1.45 |
| Buffer bytes | 0.008 |

The RSS delta (15.9 MB) exceeds raw parameter bytes (1.45 MB): it includes
framework import/initialization pages, tokenizer files, Python wrapper
objects, and torch allocator arenas — one-time costs, not per-call growth.

## 2. Steady-state forward peak (batch 1)

50 identical eval-mode `.choice` calls (1 state, 1 question, K=4) after
warmup:

| Metric | MB |
|---|---:|
| Python-side peak during 50 calls (tracemalloc) | 0.054 |
| RSS before | 493.167 |
| RSS after | 493.175 |
| **RSS delta over 50 calls** | **0.008** |

An 8 KB RSS change over 50 forward calls is allocator retention noise — a
per-call leak would grow linearly (50 x per-call size). **No forward-time
growth observed.**

## 3. m=32 questions: one shared call vs 32 separate calls

| Mode | Python peak (tracemalloc) MB | RSS delta MB |
|---|---:|---:|
| shared single `.choice` call (m=32) | 0.104 | 0.037 |
| 32 separate single-question calls | 0.050 | 0.045 |
| **ratio shared/naive (python peak)** | **2.07x** | — |

Interpretation, measured:

* The shared call holds all m=32 decision outputs (32 DecisionOutput objects
  with their [1,K] tensors) alive **simultaneously** → ~2x the Python-side
  peak of a mode that allocates one decision at a time and drops it. Both
  numbers are tiny in absolute terms (**0.10 MB vs 0.05 MB**).
* RSS deltas (0.037 vs 0.045 MB) are indistinguishable at this scale — the
  backbone activations dominate neither mode differently because the state
  batch is identical; the question-side tensors are negligibly small.
* Final process RSS after the whole script: **493.50 MB** (baseline import
  footprint ~470 MB is torch/transformers itself, not JevX).

## Summary table

| Scenario | python-side peak MB | RSS delta MB |
|---|---:|---:|
| model load | 3.03 (during load) | 15.89 |
| steady forward, batch 1 (x50) | 0.054 | 0.008 |
| m=32 shared single call | 0.104 | 0.037 |
| m=32 as 32 separate calls | 0.050 | 0.045 |
| CUDA / optimizer memory | **N/A (CPU-only env, inference-only run)** | — |

## Reproducing

```
python benchmarks/run_memory.py             # ~40 s; overwrites memory_*.json only
python benchmarks/run_memory.py --questions 64
```
