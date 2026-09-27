# Environment-Blocked Claims — JevX (written 2026-09-26)

**This is a declaration, not a benchmark.** Nothing in this file is a
measurement of a thing that could not be measured. Every "why not" below is
either a machine-read fact of the host (re-checked 2026-09-26, commands
quoted) or a pointer to a report that already carries the fact. No simulated
GPU number, emulated device, mocked accelerator or placeholder test appears
here or anywhere it points to.

The claims listed here are **repeated, not dropped**. They are blocked by the
environment, they are reported as blocked everywhere else in the repository,
and they remain blocked until the host or the dependency set changes. Nothing
here is counted as passing, and no assertion was weakened or skipped to make a
suite go green.

## 1. The host, machine-read (re-checked 2026-09-26)

```
python 3.14.3
torch 2.14.0+cpu
cuda_available        False      (torch.cuda.is_available())
mps_available         False      (torch.backends.mps.is_available())
xpu_available         False      (torch.xpu.is_available())
torch.version.cuda    None       (CPU wheel: no CUDA build)
cuda_device_count     0
nvidia-smi            not found on PATH (shutil.which -> None)
bitsandbytes          ModuleNotFoundError: No module named 'bitsandbytes'
                      (importlib.util.find_spec('bitsandbytes') -> None)
transformers 5.16.1   datasets 4.4.1
```

Consistent with `docs/LIMITATIONS.md` §1 (Accelerators: "none:
`torch 2.14.0+cpu`, `cuda.is_available() = False`, no MPS, no `nvidia-smi`")
and `REPOSITORY_RECONNAISSANCE.md`. Windows 11 (10.0.26100), Intel i5-6300U
class, 4 logical cores.

## 2. The canonical environment-blocked table, reproduced VERBATIM

This is the table from `reports/FAILURE_ANALYSIS.md`, section
"Environment-blocked rows (no fake tests)" — **same rows, same wording, same
reasons, same unblock conditions**, copied without edit:

| Row | Blocker | What would unblock |
|-----|---------|--------------------|
| 15 quantized inference | CPU-only torch build; `bitsandbytes` raises `ModuleNotFoundError` | GPU host + `pip install bitsandbytes` |
| 11 cross-device input | no second torch device (CUDA/MPS/XPU all absent) | any second device; the test auto-enables and then asserts a device-naming error |

The two anchors it refers to, also quoted verbatim from the same report's
16-row table:

| # | Scenario | Result | Asserted contract |
|---|----------|--------|-------------------|
| 11 | Wrong device | **SKIP (documented)** | Device-generic test builds a cross-device mismatch only when a second torch device exists. This host: `torch.cuda.is_available()` False, no MPS/XPU → skip reason names the absent devices. Not a fake pass. |
| 15 | Quantized model (bitsandbytes) | **ENVIRONMENT-BLOCKED** | `import bitsandbytes` → `ModuleNotFoundError`; host is CPU-only (torch +cpu build) and bitsandbytes is absent, so no quantized checkpoint can be constructed. **No fake test written** — this row is testable only on a GPU host with bitsandbytes installed. |

## 3. The three blocked claims, in full

`reports/FINAL_VERIFICATION.md` carries exactly **one** exhibit still marked
NOT PRODUCED after its 2026-09-26 refresh:

> `| GPU / quantized / cross-device | NOT PRODUCED | NOT PRODUCED (environment-blocked, unchanged) — never faked |`

That single exhibit decomposes into the three claims below. Each is given as:
the claim that cannot be produced **here**, the exact environmental reason, and
what would be required to produce it.

### 3.1 Quantized inference

* **Claim that cannot be produced here.** That a JevX-wrapped checkpoint can
  be loaded and run through the full decision path in a **quantized** form
  (bitsandbytes 4-bit/8-bit, or any other quantized kernel) with the extracted
  representation and decision outputs compared against the unquantized
  baseline. Concretely: quantized load → representation extraction → fusion →
  choice logits → calibration → confidence, and the numerical gap against the
  same checkpoint unquantized.
* **Exact environmental reason.** `import bitsandbytes` raises
  `ModuleNotFoundError: No module named 'bitsandbytes'`
  (`importlib.util.find_spec` → `None`), and the installed torch is the
  `2.14.0+cpu` wheel, so no CUDA kernel the quantizer needs can load even if
  the package were installed. Verbatim from
  `reports/FAILURE_ANALYSIS.md` row 15: *"host is CPU-only (torch +cpu build)
  and bitsandbytes is absent, so no quantized checkpoint can be constructed.
  No fake test written — this row is testable only on a GPU host with
  bitsandbytes installed."*
* **What would be required.** **A GPU host** (a CUDA device visible to torch,
  i.e. a CUDA-enabled torch build rather than `+cpu`) **plus
  `pip install bitsandbytes`**. Both, not either: on this host bitsandbytes has
  no kernel to load. Once present, row 15's test is written and executed like
  the other 15 rows; until then it stays declared-untested.
* **Status today.** `reports/FAILURE_ANALYSIS.md` row 15 = ENVIRONMENT-BLOCKED,
  no test file. `docs/FAILURE_MODES.md` §2 repeats it. QUALITY_AUDIT Known #11
  lists it as ENVIRONMENT-BLOCKED.

### 3.2 Cross-device inputs

* **Claim that cannot be produced here.** That a JevX wrapper whose parameters
  sit on one torch device raises a **device-naming error** when a caller passes
  a pre-encoded question tensor on a *different* device — i.e. the cross-device
  guard's behaviour, measured rather than only asserted to be unreachable.
* **Exact environmental reason.** **No second torch device exists on this
  host**: `torch.cuda.is_available()` False, `torch.backends.mps.is_available()`
  False, `torch.xpu.is_available()` False, `torch.cuda.device_count()` 0. A
  cross-device mismatch is not constructible with one device, so the test
  cannot enter its `pytest.raises` body. This is the suite's **single skip**.
* **What would be required.** **Any second torch device** — one CUDA device
  (which also needs a CUDA-enabled torch build, not `+cpu`), an Apple MPS
  device, or an `xpu` device. The test is already written and device-generic:
  it enumerates available devices, `pytest.skip`s with a reason that names the
  absent ones when fewer than two exist, and otherwise puts the model on
  `devices[0]` and the questions tensor on `devices[1]` and asserts a
  `(RuntimeError, ValueError)` matching `"device"`. **No new test code is
  needed** — the moment a second device appears the row self-enables. That is
  the one blocked row in this table whose unblock requires hardware only and no
  new code.
* **Status today.** `reports/FAILURE_ANALYSIS.md` row 11 = SKIP (documented).
  The live skip reason in `tests/unit/test_failure_modes.py::
  test_cross_device_input_raises_device_error` is: *"only one torch device is
  available (CPU-only environment: torch.cuda.is_available() is False, no
  MPS/XPU) — a cross-device input mismatch cannot be constructed on this
  host"*.

### 3.3 CUDA / MPS measurement

* **Claim that cannot be produced here.** Any **GPU** measurement:
  CUDA-device latency (the `T(m)` curves of `reports/LATENCY_BENCHMARKS.md`
  re-measured on an accelerator), CUDA allocator memory figures
  (`torch.cuda.max_memory_allocated` and the rest of the `reports/
  MEMORY_BENCHMARKS.md` memory table), and any MPS-backend timing or memory
  figure.
* **Exact environmental reason.** The installed torch is the **`+cpu` wheel**:
  `torch.version.cuda` is `None`, `torch.cuda.is_available()` is False,
  `torch.cuda.device_count()` is 0, and `nvidia-smi` is not on PATH. There is
  no CUDA device, therefore no CUDA allocator, therefore no CUDA allocator
  stats; and no Apple-silicon MPS backend on a Windows x86 host. Verbatim from
  `reports/MEMORY_BENCHMARKS.md` §1: *"CUDA memory: N/A. `torch.cuda.
  is_available()` is False; there is no CUDA device, no CUDA allocator, and no
  CUDA allocator stats to report. Nothing here is simulated as a GPU
  measurement."*
* **What would be required.** **A CUDA-capable host plus a CUDA-enabled torch
  build** (`torch.version.cuda` non-`None`, `torch.cuda.is_available()` True)
  for the CUDA rows; **an Apple-silicon host with a torch build that has the
  MPS backend** for the MPS row. Note the two are independent: a CUDA machine
  does not unblock the MPS numbers, and vice versa.
* **Status today.** Declared N/A in `reports/MEMORY_BENCHMARKS.md` (§1 and the
  memory table's "CUDA / optimizer memory = **N/A (CPU-only env, inference-only
  run)**" row) and in `reports/LATENCY_BENCHMARKS.md` §1 ("GPU: **none** —
  CPU-only build, no CUDA/MPS/XPU device") and its allocator-stats note. Every
  latency and memory number published in this repository is a **CPU** number on
  this host, and is labeled as such; none is a GPU number.

## 4. Drift audit against `reports/FAILURE_ANALYSIS.md` rows 11 and 15

Checked field by field, because a blocked row that drifts into a different
reason becomes a false statement:

| Field | `reports/FAILURE_ANALYSIS.md` (canonical) | Verdict |
|---|---|---|
| Row 11 blocker | `no second torch device (CUDA/MPS/XPU all absent)` | **consistent** with the host: cuda/mps/xpu all measured False, device_count 0 |
| Row 11 result label | `SKIP (documented)`, reason names the absent devices | **consistent** — the live skip message names `torch.cuda.is_available()` and MPS/XPU |
| Row 15 blocker | `CPU-only torch build; bitsandbytes raises ModuleNotFoundError` | **consistent** — both re-measured 2026-09-26, §1 above |
| Row 15 result label | `ENVIRONMENT-BLOCKED`, **no test file** | **consistent** — no quantized test exists in `tests/`; QUALITY_AUDIT Q-017 records "no files for … quantized" as closed by documentation, not by a test |

**No reason drifted.** Every blocker string matches the machine-read host
facts, and both labels match the code as written.

**One wording drift exists, in the "what would unblock" column of row 11**,
between the report and its doc mirror. It is cosmetic but it is a difference,
so it is recorded rather than smoothed:

| Source | Row 11 unblock text |
|---|---|
| `reports/FAILURE_ANALYSIS.md` (canonical, used in §2 above) | `any second device; the test auto-enables and then asserts a device-naming error` |
| `docs/FAILURE_MODES.md` §2 | `any second device — the test auto-enables and asserts a device-naming error` |

**Correct wording — the report's:** *"any second device; the test auto-enables
and then asserts a device-naming error"*. The two differ only in the separator
(`;` vs em-dash) and the dropped "and then"; the technical content is the same
in both, and the report's phrasing is adopted here as canonical. Editing
`docs/FAILURE_MODES.md` to match is a one-character-class change left to
whoever owns that file; it changes no claim.

**One structural gap, not a wording drift:** the CUDA/MPS row (§3.3) has **no
anchor in `reports/FAILURE_ANALYSIS.md`**. That report's environment-blocked
table holds only rows 15 and 11, because they are *failure-mode rows*; CUDA/MPS
is a *measurement exhibit*, not a failure mode, so it is anchored instead in
`docs/LIMITATIONS.md` §1, `reports/FINAL_VERIFICATION.md` (the "GPU / quantized /
cross-device" exhibit) and QUALITY_AUDIT Known #11. It is a legitimate row
here; the point to keep straight is that citing "FAILURE_ANALYSIS rows 11/15"
does **not** cover it.

## 5. What is deliberately NOT in this table, and why

Stated so the table's completeness is checkable rather than assumed:

* **Packaging** — `reports/FINAL_VERIFICATION.md` marked it NOT PRODUCED at the
  2026-09-24 audit and **VERIFIED** at the 2026-09-25 refresh (`pyproject.toml`
  PEP 621 + `LICENSE` + `pip install -e .` executed). It is a gap that was
  closed, not a blocked claim, so it is not a row here.
* **Distributed / multi-device execution** — recorded as an absent test
  category in `docs/TESTING.md` §1 and in QUALITY_AUDIT Q-017, and blocked by
  the same single-device host as §3.2. It is not listed as a separate row
  because the repository does not mark it NOT PRODUCED in
  `reports/FINAL_VERIFICATION.md`, and adding rows that no source claims would
  inflate this table.
* **Host disk exhaustion (2026-09-26)** — a real, separately documented host
  condition: `tests/integration/test_qwen35_hybrid.py::test_07_save_load_roundtrip`
  failed with `SafetensorError: … not enough space on the disk (os error 112)`
  at 0.46 GB free on C:, and passed on a re-run with `--basetemp` on a volume
  with 26 GB free (`1 passed in 102.05s`). It is a **test failure attributed
  and re-run**, not a NOT PRODUCED exhibit, so it does not belong in this
  table; it lives in `reports/FINAL_VERIFICATION.md`'s 2026-09-26 refresh. This
  wave's own artifacts were written with ~15.6 GB of headroom and did not
  touch it.

## 6. Cross-references

* `reports/FAILURE_ANALYSIS.md` — rows 11 and 15, and the
  "Environment-blocked rows (no fake tests)" table reproduced in §2.
* `docs/FAILURE_MODES.md` §2 "Environment-blocked cases (never faked)" — the
  doc mirror; carries the row-11 wording drift noted in §4.
* `docs/LIMITATIONS.md` §1 — the environment constraint table that
  "wins any conflict".
* `reports/FINAL_VERIFICATION.md` — the one NOT PRODUCED exhibit, and its
  per-refresh verdicts.
* `reports/MEMORY_BENCHMARKS.md` §1 / `reports/LATENCY_BENCHMARKS.md` §1 — the
  N/A declarations for the accelerator columns.
* `docs/TESTING.md` §1 / §3 — the absent test categories and the single
  documented skip.
* `reports/QUALITY_AUDIT.md` — Known #11, recorded ENVIRONMENT-BLOCKED and
  never counted as passing.

**Standing rule.** These rows are blocked by the environment and are reported
as blocked. They are never counted as passing, never simulated, and never
replaced by a test that does not exercise the blocked path. If the host gains a
second device, a CUDA-capable GPU or `bitsandbytes`, the corresponding row is
executed and its verdict changes on the evidence of that execution — not before.
