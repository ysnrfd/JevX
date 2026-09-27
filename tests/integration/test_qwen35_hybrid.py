# Copyright (c) 2026 ysnrfd
"""JEVX-ON-QWEN3.5 validation battery — REAL execution on the hybrid checkpoint.

Runs the full JevX battery against the user's downloaded Qwen3.5-0.8B HYBRID
checkpoint (transformers 5.16.1, no remote code): 18 GatedDeltaNet linear-
attention + 6 full-attention layers (``layer_types = [linear, linear, linear,
full] x 6``, ``full_attention_interval=4``), hidden 1024, 24 layers, 8Q/2KV
heads head_dim 256, vocab 248320, mrope interleaved, 853M params bf16.

Battery items (every PASS here is an executed pass on the real checkpoint,
per the "every PASS executed" rule):

1. LOAD — wrap via ``JevX.from_pretrained(<explicit snapshot>,
   local_files_only=True)``; record backbone/adapter class, capability report,
   wall time, RSS.
2. EXTRACTION SEMANTICS — adapter pooled vs an INDEPENDENT manual reference
   (direct backbone forward with ``use_cache=False`` + own fp32 masked-mean
   math); field/strategy actually taken; also isolates pool-math vs
   cache-mode differences (A/C/B decomposition below).
3. BASE PRESERVATION — ``base_forward`` vs a directly-loaded raw ``AutoModel``
   (computed in phase 0, model freed before the wrap so at most ONE 853M
   copy is alive at a time on this 8 GB CPU box).
4. DECISIONS — ``.choice`` 2 real questions x variable K {2, 5} on 2 real
   sentences; ``.score`` bounds; ``.noul`` range; mixed-type call.
5. SHARED-STATE — forward-hook counter: m=4 questions -> backbone runs once.
6. HYBRID-SPECIFIC — (a) use_cache / DynamicCache behavior + cache drift,
   (b) right-pad batch vs alone (JevX bar: not worse than backbone-alone),
   (c) left-pad vs right-pad (measured, both paths; no unconditional
   equality is asserted for this hybrid).
7. SAVE/LOAD — save_pretrained -> from_pretrained -> same ``.choice`` probs;
   sidecars; reloaded backbone class == saved ``architectures`` entry.
8. REAL-DATA SMOKE — 40 SST-2 pairs end-to-end. UNTRAINED HEAD: pipeline
   verification only, NOT a quality claim.
   Optional: HeadTrainer choice-head training (featurize once via the real
   adapter, frozen backbone, 200 samples) reported separately.
9. FAILURE TRANSPARENCY — pixel_values/image-key states and garbage input
   types: error quality recorded verbatim.

Measured numbers accumulate into ``benchmarks/results/
qwen35_jevx_validation.json`` after every test.

Memory protocol (8 GB CPU): phase 0 loads the raw AutoModel, computes small
reference tensors, frees it; then the JevX wrap loads its own copy; the only
two-copy window is the save/load test (original + reloaded), freed within the
test. All inputs are short (<=128 tokens).

bf16 tolerance rationale (empirical, this checkpoint):
* probability rows are stored bf16 (``DecisionOutput.probabilities`` dtype =
  torch.bfloat16); measured max |sum - 1| = 1.953e-3, bf16 eps at 1.0 is
  7.8125e-3 -> assert atol 4e-3 (2x the measurement, still < 1 ulp at 1.0).
* bf16 ulp at magnitude v is ``2^(floor(log2 v) - 7)``; cross-path pooled
  comparisons are asserted against a dynamic 2-ulp bound of the observed
  magnitude (plus the mission's explicit < 1e-2 where same-call determinism
  is expected: base_forward, save/load).
"""

from __future__ import annotations

import gc
import inspect
import json
import math
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pytest
import torch
from transformers import AutoModel, AutoTokenizer

from jevx import JevX, JevXForDecision
from jevx.modeling_jevx import _BACKBONE_LOAD_KWARGS
from matrix_common import ForwardCounter, is_environment_error

ROOT = Path(__file__).resolve().parents[2]

#: User-provided checkpoint (never fetched by the suite — local only).
QWEN35_MODEL_ID = "Qwen/Qwen3.5-0.8B"
#: Pinned snapshot revision, quoted in the skip message for diagnosis.
QWEN35_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
#: Skip reason when the checkpoint is absent on this host (Q-005: an absent
#: checkpoint is an environment problem -> classified pytest.skip, never a
#: bare assert — repo convention docs/TESTING.md §3).
_QWEN35_MISSING = (
    f"{QWEN35_MODEL_ID} snapshot not present locally (user-provided checkpoint; "
    f"revision {QWEN35_REVISION})"
)


def _resolve_snapshot() -> Path | None:
    """Locate the Qwen3.5-0.8B snapshot WITHOUT touching the network.

    Ladder (Q-005 portability fix — no hardcoded host paths):

    1. ``huggingface_hub.snapshot_download(QWEN35_MODEL_ID,
       local_files_only=True)`` — honors ``HF_HOME``/``HF_HUB_CACHE``;
    2. walk ``HF_HUB_CACHE`` / ``HF_HOME``/hub / ``HUGGINGFACE_HOME`` /
       ``~/.cache/huggingface/hub`` for ``models--Qwen--Qwen3.5-0.8B/
       snapshots/*`` (the pinned revision preferred, else any cached one);

    Returns ``None`` if neither resolves → callers ``pytest.skip``.
    """
    try:
        from huggingface_hub import snapshot_download

        resolved = snapshot_download(QWEN35_MODEL_ID, local_files_only=True)
        candidate = Path(resolved)
        if candidate.is_dir():
            return candidate
    except Exception:  # noqa: BLE001 - offline/absent falls through to the walk
        pass

    hub_dirs = []
    for env_var in ("HF_HUB_CACHE", "HF_HOME", "HUGGINGFACE_HOME"):
        value = os.environ.get(env_var)
        if value:
            base = Path(value)
            hub_dirs.append(base / "hub" if env_var == "HF_HOME" else base)
    hub_dirs.append(Path.home() / ".cache" / "huggingface" / "hub")
    repo_dir_name = "models--Qwen--Qwen3.5-0.8B"
    for hub in hub_dirs:
        snapshots = hub / repo_dir_name / "snapshots"
        if not snapshots.is_dir():
            continue
        pinned = snapshots / QWEN35_REVISION
        if pinned.is_dir():
            return pinned
        cached = sorted(p for p in snapshots.iterdir() if p.is_dir())
        if cached:
            return cached[-1]
    return None


SNAPSHOT = _resolve_snapshot()
REFS_MAIN = (SNAPSHOT.parent.parent / "refs" / "main") if SNAPSHOT else None


def _require_snapshot() -> Path:
    """The resolved checkpoint, or ``pytest.skip`` when absent (Q-005)."""
    if SNAPSHOT is None or not SNAPSHOT.is_dir():
        pytest.skip(_QWEN35_MISSING)
    return SNAPSHOT
RESULTS_PATH = ROOT / "benchmarks" / "results" / "qwen35_jevx_validation.json"

#: Empirical bf16 tolerance for probability-row sums (see module docstring).
BF16_SUM_ATOL = 4e-3
#: Same-call comparisons the mission pins to < 1e-2 (bf16).
SAME_CALL_ATOL = 1e-2

TEXT_SHORT = (
    "The maintenance crew inspected the bridge deck and confirmed that the "
    "support damage had been repaired before the storm arrived."
)
TEXT_LONG = (
    "Analysts expect the quarterly report to show steady revenue growth across "
    "all regions this year, driven mainly by services and cloud subscriptions."
)
QUESTIONS = ["Is this content safe to deploy?", "Which disposition fits this item?"]
OPTIONS = [["safe", "unsafe"], ["approve", "review", "reject", "defer", "escalate"]]
SCORE_QUESTIONS = ["How would you rate the severity of this issue?"]
SCORE_OPTIONS = [["1", "2", "3", "4", "5"]]
NOUL_QUESTIONS = ["Does this text contain a repair guarantee?"]

_RESULTS: Dict[str, Any] = {"battery": {}}


def record(key: str, payload: Dict[str, Any]) -> None:
    """Store one battery result and flush the JSON artifact (as-you-go)."""
    _RESULTS["battery"][key] = payload
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(_RESULTS, indent=2, default=str, ensure_ascii=True), encoding="utf-8"
    )


def _environment_block() -> Dict[str, Any]:
    import platform
    import sys

    import psutil
    import transformers

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or "cpu",
        "cpu_count": psutil.cpu_count(logical=True),
        "torch_num_threads": torch.get_num_threads(),
        "psutil": psutil.__version__,
    }


def _rss() -> Dict[str, Any]:
    """Current + peak RSS via psutil (peak_wset is the Windows process peak)."""
    import psutil

    mi = psutil.Process().memory_info()
    out = {
        "rss_mb": round(mi.rss / 1e6, 1),
        "peak_wset_mb": round(getattr(mi, "peak_wset", 0) / 1e6, 1)
        if getattr(mi, "peak_wset", None)
        else None,
    }
    return out


def bf16_ulp(value: float) -> float:
    """bf16 ulp at ``value``: ``2^(floor(log2|v|) - 7)`` (0 for 0.0)."""
    v = abs(float(value))
    if v == 0.0 or not math.isfinite(v):
        return 0.0
    return 2.0 ** (math.floor(math.log2(v)) - 7)


def manual_pool_fp32(seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Independent masked-mean reference in fp32 (NOT ``jevx.pool``)."""
    s = seq.float()
    m = mask.to(torch.bool).unsqueeze(-1)
    return (s * m).sum(dim=1) / m.sum(dim=1).clamp(min=1).float()


def _tokenize(tok: Any, texts: Any, *, padding_side: str = "right") -> Dict[str, torch.Tensor]:
    """Tokenize with an explicit padding side (side restored to ``right``)."""
    old = tok.padding_side
    tok.padding_side = padding_side
    try:
        return dict(tok(texts, return_tensors="pt", padding=True))
    finally:
        tok.padding_side = old


# ---------------------------------------------------------------------------
# Phase 0 — directly-loaded raw AutoModel references (model then FREED)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def raw_refs() -> Dict[str, Any]:
    """Small reference tensors from a directly-loaded raw ``AutoModel``.

    Runs BEFORE the JevX wrap so only one 853M copy is ever resident here;
    the raw model is freed before returning (memory protocol in the module
    docstring).
    """
    snapshot = _require_snapshot()
    t0 = time.perf_counter()
    raw = AutoModel.from_pretrained(str(snapshot), local_files_only=True)
    raw_load_s = time.perf_counter() - t0
    raw.eval()
    assert type(raw).__name__ == "Qwen3_5Model", type(raw).__name__

    tok = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token or tok.unk_token

    inputs_s = dict(tok(TEXT_SHORT, return_tensors="pt"))
    inputs_l = dict(tok(TEXT_LONG, return_tensors="pt"))
    with torch.no_grad():
        # Same call base_forward will replay later (kwargs identical).
        out_default = raw(**dict(inputs_s))
        out_nocache = raw(**{**dict(inputs_s), "use_cache": False})
        seq_default = out_default.last_hidden_state.detach().clone()
        seq_nocache = out_nocache.last_hidden_state.detach().clone()
        # Backbone-alone padding references (use_cache=False, own fp32 pool).
        pad_pooled: Dict[str, torch.Tensor] = {}
        for name, batch in {
            "solo_short": inputs_s,
            "solo_long": inputs_l,
            "right_batch": _tokenize(tok, [TEXT_SHORT, TEXT_LONG], padding_side="right"),
            "left_batch": _tokenize(tok, [TEXT_SHORT, TEXT_LONG], padding_side="left"),
        }.items():
            out = raw(**{**batch, "use_cache": False})
            pad_pooled[name] = manual_pool_fp32(
                out.last_hidden_state, batch["attention_mask"]
            ).clone()
            if name in ("right_batch", "left_batch"):
                pad_pooled[name + "_ids"] = batch["input_ids"].clone()
                pad_pooled[name + "_mask"] = batch["attention_mask"].clone()
    raw_facts = {
        "raw_load_seconds": round(raw_load_s, 3),
        "raw_class": type(raw).__name__,
        "raw_params": sum(p.numel() for p in raw.parameters()),
        "raw_dtype": str(next(raw.parameters()).dtype),
        "rss_after_raw_load": _rss(),
    }
    del out_default, out_nocache, raw, tok
    gc.collect()

    sha = (
        REFS_MAIN.read_text(encoding="utf-8").strip()
        if REFS_MAIN is not None and REFS_MAIN.is_file()
        else None
    )
    return {
        "facts": raw_facts,
        "sha": sha,
        "inputs_short": inputs_s,
        "inputs_long": inputs_l,
        "seq_default": seq_default,
        "seq_nocache": seq_nocache,
        "pool_nocache_short": manual_pool_fp32(seq_nocache, inputs_s["attention_mask"]),
        "pad_pooled": pad_pooled,
    }


@pytest.fixture(scope="module")
def jevx_case(raw_refs: Dict[str, Any]) -> SimpleNamespace:
    """The JevX wrap of the real checkpoint (battery item 1's subject)."""
    snapshot = _require_snapshot()
    t0 = time.perf_counter()
    wrapper = JevX.from_pretrained(str(snapshot), local_files_only=True)
    load_s = time.perf_counter() - t0
    wrapper.eval()
    rep = wrapper.capability_report
    facts = {
        "model_id_passed": str(snapshot),
        "load_kwargs_passed": {"local_files_only": True},
        "local_files_only_in_backbone_whitelist": "local_files_only"
        in _BACKBONE_LOAD_KWARGS,
        "load_seconds": round(load_s, 3),
        "backbone_class": type(wrapper.model).__name__,
        "adapter_class": type(wrapper.adapter).__name__,
        "adapter_name": getattr(wrapper.adapter, "name", None),
        "adapter_default_pooling": getattr(wrapper.adapter, "default_pooling", None),
        "tokenizer_class": type(wrapper.tokenizer).__name__
        if wrapper.tokenizer is not None
        else None,
        "tokenizer_padding_side": getattr(wrapper.tokenizer, "padding_side", None)
        if wrapper.tokenizer is not None
        else None,
        "processor_class": type(wrapper.processor).__name__
        if wrapper.processor is not None
        else None,
        "pad_token_id": getattr(wrapper.tokenizer, "pad_token_id", None)
        if wrapper.tokenizer is not None
        else None,
        "capability": rep.to_dict(),
        "capability_mentions_layer_types": any(
            "layer_types" in reason for reason in rep.reasons
        ),
        "decision_hidden_size": wrapper.config.decision_hidden_size,
        "pooling_method_config": wrapper.config.pooling_method,
        "base_model_type": wrapper.config.base_model_type,
        "backbone_config_architectures": (wrapper.config.backbone_config or {}).get(
            "architectures"
        ),
        "wrapper_dtype": str(next(wrapper.model.parameters()).dtype),
        "params_total": sum(p.numel() for p in wrapper.parameters()),
        "params_backbone": sum(p.numel() for p in wrapper.model.parameters()),
        "text_config_hidden_size": getattr(
            getattr(wrapper.model.config, "text_config", None), "hidden_size", None
        ),
        "text_config_layer_types": list(
            getattr(wrapper.model.config.text_config, "layer_types", [])
        ),
        "full_attention_interval": getattr(
            wrapper.model.config.text_config, "full_attention_interval", None
        ),
        "rss_after_wrap_load": _rss(),
        "checkpoint_sha": raw_refs["sha"],
    }
    return SimpleNamespace(wrapper=wrapper, facts=facts, raw_refs=raw_refs)


@pytest.fixture(scope="module")
def extracted(jevx_case: SimpleNamespace) -> Dict[str, Any]:
    """Battery 2/6a material: extraction (A), direct default (C + C2 drift
    copy), direct use_cache=False (B) — one forward each, results stored."""
    wrapper = jevx_case.wrapper
    inputs = jevx_case.raw_refs["inputs_short"]
    adapter = wrapper.adapter
    with torch.no_grad():
        # C and C2: identical sequential default-mode forwards (cache drift).
        out_c = wrapper.model(**dict(inputs))
        out_c2 = wrapper.model(**dict(inputs))
        # B: direct forward exactly as battery 2 prescribes.
        out_b = wrapper.model(**{**dict(inputs), "use_cache": False})
        # A: adapter extraction exactly as the wrapper's forward does it.
        rep_a = adapter.extract_representation(
            wrapper.model,
            dict(inputs),
            strategy=wrapper.config.pooling_method,
            output_hidden_states=False,
        )
    return {
        "rep_a": rep_a,
        "seq_c": out_c.last_hidden_state,
        "seq_c2": out_c2.last_hidden_state,
        "seq_b": out_b.last_hidden_state,
        "cache_c": out_c.past_key_values,
        "pool_b": manual_pool_fp32(out_b.last_hidden_state, inputs["attention_mask"]),
        "use_cache_declared": "use_cache"
        in inspect.signature(type(wrapper.model).forward).parameters,
    }


# ---------------------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------------------


def test_01_load_and_capability(jevx_case: SimpleNamespace, raw_refs) -> None:
    wrapper = jevx_case.wrapper
    facts = jevx_case.facts
    rep = wrapper.capability_report

    # Documented load contract: explicit snapshot path + whitelisted kwarg.
    assert facts["local_files_only_in_backbone_whitelist"], (
        "local_files_only missing from _BACKBONE_LOAD_KWARGS"
    )
    # Backbone class chosen by the AutoModel chain, cross-checked against the
    # independently-loaded raw model from phase 0.
    assert facts["backbone_class"] == raw_refs["facts"]["raw_class"] == "Qwen3_5Model"
    assert facts["params_backbone"] == raw_refs["facts"]["raw_params"] == 852_985_920
    assert facts["wrapper_dtype"] == "torch.bfloat16"
    # Adapter resolution on the hybrid wrapper (verified, not assumed):
    # the AutoModel wrapper carries a vision tower (config.vision_config) ->
    # the multimodal arm of the built-in chain wins over DecoderAdapter.
    assert facts["adapter_class"] in ("MultimodalAdapter", "DecoderAdapter")
    assert facts["adapter_name"] in ("multimodal", "decoder")
    # Detector output for the hybrid.
    assert rep.jevx_conversion in ("supported", "experimental")
    assert rep.representation_field == "last_hidden_state"
    assert rep.output_class == "Qwen3_5ModelOutputWithPast"
    assert rep.pooler is False  # Qwen3_5 has no pooler -> sequence pooling path
    # decision_hidden_size follows the nested text_config.hidden_size = 1024.
    assert facts["decision_hidden_size"] == 1024 == facts["text_config_hidden_size"]
    # Hybrid config reached JevX intact.
    assert len(facts["text_config_layer_types"]) == 24
    assert facts["text_config_layer_types"] == [t for t in
        ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 6]
    assert facts["full_attention_interval"] == 4
    assert facts["backbone_config_architectures"] == ["Qwen3_5Model"]
    # Featurizer: text path must be usable despite vision side.
    assert facts["tokenizer_class"] is not None
    # Known finding #1 FLIPPED (was: "capability reasons do NOT mention
    # layer_types", recorded-only): the detector now parses the nested
    # text_config.layer_types and must report PRESENCE + the hybrid summary.
    assert facts["capability_mentions_layer_types"] is True, (
        "capability reasons must mention layer_types for the hybrid config"
    )
    assert rep.attention_layout == "hybrid:18xlinear_attention+6xfull_attention"
    assert any(
        reason.startswith("hybrid attention layout")
        and "18 linear_attention + 6 full_attention" in reason
        for reason in rep.reasons
    )
    # Informational field only: verdicts unchanged (no upgrade/downgrade).
    assert rep.jevx_conversion in ("supported", "experimental")
    measured = (
        f"backbone={facts['backbone_class']}, adapter={facts['adapter_class']}, "
        f"load={facts['load_seconds']}s (raw {raw_refs['facts']['raw_load_seconds']}s), "
        f"jevx_conversion={rep.jevx_conversion}, "
        f"representation_field={rep.representation_field}, "
        f"decision_hidden_size={facts['decision_hidden_size']}, "
        f"layer_types_in_reasons={facts['capability_mentions_layer_types']}"
    )
    record(
        "1_load",
        {
            **{k: v for k, v in facts.items() if k != "capability"},
            "capability": rep.to_dict(),
            "raw_load_seconds": raw_refs["facts"]["raw_load_seconds"],
            "measured": measured,
        },
    )
    print(f"[qwen35-1] {measured}")


# ---------------------------------------------------------------------------
# 2. EXTRACTION SEMANTICS
# ---------------------------------------------------------------------------


def test_02_extraction_semantics(jevx_case: SimpleNamespace, extracted) -> None:
    wrapper = jevx_case.wrapper
    rep_a = extracted["rep_a"]
    inputs = jevx_case.raw_refs["inputs_short"]

    assert rep_a.field == "last_hidden_state"
    assert wrapper.capability_report.representation_field == rep_a.field, (
        "detector representation_field disagrees with the extracted field"
    )
    assert rep_a.layout == "btd"
    assert tuple(rep_a.pooled.shape) == (1, 1024)
    assert rep_a.pooled.dtype == torch.bfloat16
    effective_strategy = (
        wrapper.config.pooling_method or wrapper.adapter.default_pooling
    )
    assert wrapper.config.pooling_method is None
    assert effective_strategy in ("masked_mean", "last_valid")

    # A vs C: identical call (adapter adds nothing for MultimodalAdapter on
    # text inputs) -> must be bitwise identical (pool over the same sequence).
    seq_a = rep_a.sequence_output
    a_c_seq_err = float((seq_a.float() - extracted["seq_c"].float()).abs().max())
    assert torch.equal(seq_a, extracted["seq_c"]), a_c_seq_err

    # C vs B: cache-mode (DynamicCache built vs use_cache=False) effect on
    # the sequence — the hybrid's cache must not change the math.
    c_b_seq_err = float(
        (extracted["seq_c"].float() - extracted["seq_b"].float()).abs().max()
    )
    pool_c = manual_pool_fp32(extracted["seq_c"], inputs["attention_mask"])
    a_pool_vs_c_pool = float(
        (rep_a.pooled.float() - pool_c.float()).abs().max()
    )

    # Battery number: adapter pooled vs INDEPENDENT manual reference
    # (direct forward with use_cache=False + own fp32 masked-mean math).
    manual = extracted["pool_b"]
    extract_err = float((rep_a.pooled.float() - manual).abs().max())
    # bf16 quantization floor: pooled is STORED bf16; a value of magnitude m
    # rounds to ulp(m). Bound = 2 ulp at the observed max magnitude.
    ulp = bf16_ulp(float(rep_a.pooled.float().abs().max()))
    bound = max(2.0 * ulp, 1e-6)
    assert extract_err <= bound, (
        f"adapter pooled diverged from independent manual reference by "
        f"{extract_err:.3e} > 2-ulp bf16 bound {bound:.3e}"
    )
    # A vs C pool: same sequence, same fp32 math — only the STORE-to-bf16
    # rounding of the adapter's pooled tensor separates them (<= 1 ulp).
    assert a_pool_vs_c_pool <= max(ulp, 1e-6), a_pool_vs_c_pool
    # Cache-mode invariance of the sequence itself (hybrid GDN + full attn).
    assert c_b_seq_err <= max(2.0 * bf16_ulp(
        float(extracted["seq_c"].float().abs().max())
    ), 1e-6), c_b_seq_err

    record(
        "2_extraction",
        {
            "field": rep_a.field,
            "strategy_effective": effective_strategy,
            "pooling_method_config": wrapper.config.pooling_method,
            "adapter_default_pooling": wrapper.adapter.default_pooling,
            "pooled_shape": list(rep_a.pooled.shape),
            "pooled_dtype": str(rep_a.pooled.dtype),
            "decision_hidden_size": wrapper.config.decision_hidden_size,
            "pooler_present": wrapper.capability_report.pooler,
            "extract_vs_manual_fp32_max_abs_error": extract_err,
            "extract_vs_direct_default_pool_max_abs_error": a_pool_vs_c_pool,
            "extract_seq_vs_direct_default_max_abs_error": a_c_seq_err,
            "cache_vs_nocache_seq_max_abs_error": c_b_seq_err,
            "bf16_ulp_at_pooled_max": ulp,
            "bound_2ulp": bound,
            "manual_reference": "direct backbone forward use_cache=False + independent fp32 masked_mean",
            "first3": [round(float(x), 5) for x in rep_a.pooled[0, :3].tolist()],
        },
    )
    print(
        f"[qwen35-2] extraction field={rep_a.field} strategy={effective_strategy} "
        f"pooled={tuple(rep_a.pooled.shape)} manual_err={extract_err:.3e} "
        f"(2-ulp bound {bound:.3e}) cache_vs_nocache_seq={c_b_seq_err:.3e}"
    )


# ---------------------------------------------------------------------------
# 3. BASE PRESERVATION
# ---------------------------------------------------------------------------


def test_03_base_preservation(jevx_case: SimpleNamespace, raw_refs) -> None:
    wrapper = jevx_case.wrapper
    inputs = raw_refs["inputs_short"]
    with torch.no_grad():
        wrapped_base = wrapper.base_forward(**inputs)
    err = float(
        (wrapped_base.last_hidden_state.float() - raw_refs["seq_default"].float())
        .abs()
        .max()
    )
    bitwise = bool(
        torch.equal(wrapped_base.last_hidden_state, raw_refs["seq_default"])
    )
    assert err < SAME_CALL_ATOL, f"base_forward diverged by {err:.3e} (bf16 atol {SAME_CALL_ATOL})"
    record(
        "3_base_preservation",
        {
            "max_abs_error": err,
            "bitwise_equal": bitwise,
            "tolerance": SAME_CALL_ATOL,
            "reference": "directly-loaded AutoModel.from_pretrained(snapshot, local_files_only=True)",
            "measured": f"max_abs_error={err:.3e}, bitwise_equal={bitwise}",
        },
    )
    print(f"[qwen35-3] base_forward vs raw AutoModel max_abs_error={err:.3e} bitwise={bitwise}")


# ---------------------------------------------------------------------------
# 4. DECISIONS
# ---------------------------------------------------------------------------


def test_04_decision_invariants(jevx_case: SimpleNamespace) -> None:
    wrapper = jevx_case.wrapper
    states = [TEXT_SHORT, TEXT_LONG]
    with torch.no_grad():
        choice_out = wrapper.choice(state=states, questions=QUESTIONS, options=OPTIONS)
    assert len(choice_out.decisions) == 4  # B=2 x m=2

    sum_devs = []
    first_probs: list = []
    predicted_values = []
    for pair, (dec, keys) in enumerate(zip(choice_out.decisions, OPTIONS * 2)):
        probs = dec.probabilities
        k = len(keys)
        assert probs.dtype == torch.bfloat16  # stored in the model dtype
        assert (probs >= 0).all()
        # variable-K mask: the K=2 question must show exact zeros at 2..4
        # (the choice group pads both questions to K_max=5 in one head batch)
        if probs.shape[-1] > k:
            assert bool((probs[0, k:] == 0).all()), "variable-K mask must be exact zeros"
        dev = abs(float(probs.float().sum()) - 1.0)
        sum_devs.append(dev)
        assert dev <= BF16_SUM_ATOL, (
            f"pair {pair}: probability row sum deviates {dev:.3e} > bf16 atol "
            f"{BF16_SUM_ATOL} (measured reference 1.953e-3, bf16 eps 7.8125e-3)"
        )
        assert 0 <= int(dec.predicted_index[0]) < k, (int(dec.predicted_index[0]), k)
        assert dec.predicted_value in keys, (dec.predicted_value, keys)
        assert 0.0 <= float(dec.confidence[0]) <= 1.0
        if pair == 0:
            first_probs.append([round(float(x), 6) for x in probs[0].tolist()])
        predicted_values.append(dec.predicted_value)
        assert tuple(choice_out.state_representation.shape) == (2, 1024)

    with torch.no_grad():
        score_out = wrapper.score(
            state=TEXT_SHORT, questions=SCORE_QUESTIONS, options=SCORE_OPTIONS
        )
        noul_out = wrapper.noul(state=TEXT_SHORT, questions=NOUL_QUESTIONS)
        mixed = wrapper(
            state=TEXT_SHORT,
            questions=["Is it safe?", "Rate it", "Any guarantee?"],
            decision_type=["choice", "score", "noul"],
            options=[["yes", "no"], ["0", "1", "2", "3"], None],
        )
    es = float(score_out.decisions[0].expected_score[0])
    assert -1e-6 <= es <= 4 + 1e-6, es
    assert abs(float(score_out.decisions[0].probabilities.float().sum()) - 1.0) <= BF16_SUM_ATOL
    p_yes = float(noul_out.decisions[0].probabilities[0, 0])
    assert 0.0 < p_yes < 1.0, p_yes
    assert 0.0 <= float(noul_out.decisions[0].confidence[0]) <= 1.0
    mixed_types = [d.schema_metadata["decision_type"] for d in mixed.decisions]
    assert mixed_types == ["choice", "score", "noul"], mixed_types

    record(
        "4_decisions",
        {
            "choice_states": 2,
            "choice_questions": 2,
            "K_by_question": [2, 5],
            "n_decisions": len(choice_out.decisions),
            "row_sum_abs_devs": sum_devs,
            "row_sum_max_abs_dev": max(sum_devs),
            "bf16_sum_atol": BF16_SUM_ATOL,
            "bf16_sum_atol_rationale": (
                "probabilities stored bf16; measured max |sum-1| 1.953e-3; "
                "bf16 eps at 1.0 = 7.8125e-3 -> atol 4e-3 = 2x measured"
            ),
            "first_probs_question1_state1": first_probs[0] if first_probs else None,
            "predicted_values": predicted_values,
            "confidence_range": [
                min(float(d.confidence[0]) for d in choice_out.decisions),
                max(float(d.confidence[0]) for d in choice_out.decisions),
            ],
            "score_expected_score": es,
            "score_in_bounds": True,
            "noul_p_yes": p_yes,
            "noul_confidence": float(noul_out.decisions[0].confidence[0]),
            "mixed_types": mixed_types,
            "prob_dtype": str(choice_out.decisions[0].probabilities.dtype),
        },
    )
    print(
        f"[qwen35-4] choice sums max|dev|={max(sum_devs):.3e} "
        f"first_probs={first_probs[0] if first_probs else None} "
        f"score_es={es:.4f} noul_p={p_yes:.4f} mixed={mixed_types}"
    )


# ---------------------------------------------------------------------------
# 5. SHARED-STATE (m=4 -> exactly one backbone forward)
# ---------------------------------------------------------------------------


def test_05_shared_state_single_backbone_forward(jevx_case: SimpleNamespace) -> None:
    wrapper = jevx_case.wrapper
    counter = ForwardCounter([wrapper.model])
    try:
        m = 4
        with torch.no_grad():
            wrapper.choice(
                state=TEXT_SHORT,
                questions=[f"Question {i}: is this acceptable?" for i in range(m)],
                options=[[f"opt{i}-a", f"opt{i}-b"] for i in range(m)],
            )
    finally:
        counter.close()
    assert counter.calls == 1, (
        f"state featurization ran the backbone {counter.calls} times for m=4 "
        "(contract: exactly once)"
    )
    record(
        "5_shared_state",
        {"m": 4, "backbone_forward_calls": counter.calls, "expected": 1},
    )
    print(f"[qwen35-5] m=4 backbone forward count = {counter.calls}")


# ---------------------------------------------------------------------------
# 6a. HYBRID: use_cache / DynamicCache behavior + drift
# ---------------------------------------------------------------------------


def test_06a_hybrid_cache_behavior(jevx_case: SimpleNamespace, extracted) -> None:
    wrapper = jevx_case.wrapper
    cache = extracted["cache_c"]
    # Two identical sequential default-mode forwards must be drift-free.
    drift = float(
        (extracted["seq_c"].float() - extracted["seq_c2"].float()).abs().max()
    )
    assert torch.equal(extracted["seq_c"], extracted["seq_c2"]), (
        f"sequential identical forwards drifted: {drift:.3e}"
    )
    # FIX REGRESSION GUARD (jevx/adapters/base.py extract_representation):
    # Qwen3_5Model declares use_cache only via **kwargs, so the shared
    # extraction path must inject use_cache=False for kwargs-only signatures
    # too — otherwise the config default builds a fresh DynamicCache on every
    # extraction call. Post-fix the adapter extraction output carries NO cache.
    extraction_cache = extracted["rep_a"].raw_output.past_key_values
    assert extraction_cache is None, (
        "extraction built a cache "
        f"({type(extraction_cache).__name__}) — extract_representation must pass "
        "use_cache=False for kwargs-only forwards (§5/§8.7)"
    )
    payload = {
        "use_cache_declared_in_forward_signature": extracted["use_cache_declared"],
        "jevx_injects_use_cache_false_for_kwargs_forwards": True,
        "extraction_output_past_key_values": None,
        "raw_default_call_built_cache": cache is not None,
        "raw_default_cache_type": type(cache).__name__ if cache is not None else None,
        "two_sequential_forwards_bitwise_equal": True,
        "two_sequential_forwards_max_abs_error": drift,
        "cache_vs_use_cacheFalse_seq_max_abs_error": float(
            (extracted["seq_c"].float() - extracted["seq_b"].float()).abs().max()
        ),
        "finding": (
            "Qwen3_5Model.forward declares use_cache only via **kwargs "
            "(TransformersKwargs), which the original explicit-parameter gate "
            "missed: the config default (use_cache=true) built a DynamicCache "
            "per extraction (bug found by this battery, fixed in "
            "jevx/adapters/base.py). Raw default-mode wrapper calls still build "
            "a DynamicCache (model default) and are drift-free: identical "
            "sequential forwards are bitwise equal and cache/no-cache sequences "
            "agree exactly (0.0)."
        ),
    }
    record("6a_cache", payload)
    print(
        f"[qwen35-6a] extraction cache={payload['extraction_output_past_key_values']} "
        f"raw_default_cache={payload['raw_default_cache_type']} "
        f"use_cache_declared={payload['use_cache_declared_in_forward_signature']} "
        f"drift={drift:.3e}"
    )


# ---------------------------------------------------------------------------
# 6b/6c. HYBRID: right-pad batch vs alone; left-pad vs right-pad
# ---------------------------------------------------------------------------


def test_06b_c_hybrid_padding(
    jevx_case: SimpleNamespace, extracted, raw_refs
) -> None:
    wrapper = jevx_case.wrapper
    adapter = wrapper.adapter
    pad_refs = raw_refs["pad_pooled"]

    solo_s = raw_refs["inputs_short"]
    solo_l = raw_refs["inputs_long"]
    with torch.no_grad():
        # JevX-side pooled vectors through the real adapter path.
        # solo_short == the extraction fixture's input (same tensors) -> reuse.
        jevx_solo_s = extracted["rep_a"].pooled
        jevx_solo_l = adapter.extract_representation(wrapper.model, dict(solo_l)).pooled
        jevx_rb = adapter.extract_representation(
            wrapper.model,
            {"input_ids": pad_refs["right_batch_ids"], "attention_mask": pad_refs["right_batch_mask"]},
        ).pooled
        jevx_lb = adapter.extract_representation(
            wrapper.model,
            {"input_ids": pad_refs["left_batch_ids"], "attention_mask": pad_refs["left_batch_mask"]},
        ).pooled

    def err(a: torch.Tensor, b: torch.Tensor) -> float:
        return float((a.float() - b.float()).abs().max())

    ulp = bf16_ulp(max(float(jevx_rb.float().abs().max()), 1.0))
    noise_floor = 2.0 * ulp  # bf16 kernel-rounding noise floor for batch-vs-alone

    measured = {
        # (b) right-pad batch of 2 different-length texts vs each alone
        "jevx_right_batch_vs_alone": {
            "row0_short": err(jevx_rb[0], jevx_solo_s[0]),
            "row1_long": err(jevx_rb[1], jevx_solo_l[0]),
        },
        "backbone_right_batch_vs_alone": {
            "row0_short": err(pad_refs["right_batch"][0], pad_refs["solo_short"][0]),
            "row1_long": err(pad_refs["right_batch"][1], pad_refs["solo_long"][0]),
        },
        # (c) left-pad vs right-pad, same content
        "jevx_left_vs_right_batch": err(jevx_lb, jevx_rb),
        "backbone_left_vs_right_batch": err(
            pad_refs["left_batch"], pad_refs["right_batch"]
        ),
        "jevx_left_batch_vs_alone": {
            "row0_short": err(jevx_lb[0], jevx_solo_s[0]),
            "row1_long": err(jevx_lb[1], jevx_solo_l[0]),
        },
        "backbone_left_batch_vs_alone": {
            "row0_short": err(pad_refs["left_batch"][0], pad_refs["solo_short"][0]),
            "row1_long": err(pad_refs["left_batch"][1], pad_refs["solo_long"][0]),
        },
        "bf16_ulp_at_batch_max": ulp,
        "noise_floor_2ulp": noise_floor,
    }

    # JevX's bar (b): NOT WORSE than backbone-alone above the bf16 noise floor.
    for row in ("row0_short", "row1_long"):
        jevx_err = measured["jevx_right_batch_vs_alone"][row]
        bb_err = measured["backbone_right_batch_vs_alone"][row]
        assert jevx_err <= max(bb_err, noise_floor) + 1e-6, (
            f"right-pad {row}: JevX {jevx_err:.3e} worse than backbone {bb_err:.3e} "
            f"(noise floor {noise_floor:.3e})"
        )
    # (c): measured only + not-worse-than-backbone (no unconditional equality
    # is asserted for this hybrid — the GPT-2 left/right claim belongs to
    # DecoderAdapter, which does not resolve on the vision-carrying wrapper).
    assert measured["jevx_left_vs_right_batch"] <= max(
        measured["backbone_left_vs_right_batch"], noise_floor
    ) + 1e-6, measured

    measured["interpretation"] = (
        "GDN path zeroes padded tokens pre-recurrence (conv bias=False, so pad "
        "steps inject exact zeros); full-attention layers use RoPE whose "
        "attention scores depend only on RELATIVE positions, and a uniform "
        "padding shift preserves all intra-row offsets — measured left/right "
        "and batch/alone differences therefore sit at bf16 kernel-rounding "
        "scale. JevX bar: not worse than backbone-alone."
    )
    record("6bc_padding", measured)
    print(
        f"[qwen35-6bc] right-batch vs alone jevx={measured['jevx_right_batch_vs_alone']} "
        f"backbone={measured['backbone_right_batch_vs_alone']} | "
        f"left-vs-right jevx={measured['jevx_left_vs_right_batch']:.3e} "
        f"backbone={measured['backbone_left_vs_right_batch']:.3e} "
        f"(2-ulp floor {noise_floor:.3e})"
    )


# ---------------------------------------------------------------------------
# 7. SAVE / LOAD
# ---------------------------------------------------------------------------


def test_07_save_load_roundtrip(jevx_case: SimpleNamespace, tmp_path) -> None:
    wrapper = jevx_case.wrapper
    save_dir = tmp_path / "jevx-qwen35-hybrid"

    with torch.no_grad():
        before = wrapper.choice(state=TEXT_SHORT, questions=QUESTIONS, options=OPTIONS)
    before_probs = [d.probabilities.clone() for d in before.decisions]

    t0 = time.perf_counter()
    wrapper.save_pretrained(str(save_dir))
    save_s = time.perf_counter() - t0

    meta = save_dir / "jevx_metadata.json"
    calib = save_dir / "jevx_calibration.pt"
    assert meta.is_file(), "jevx_metadata.json sidecar missing"
    assert not calib.is_file(), (
        "calibration sidecar must NOT be written when calibration_method == 'none'"
    )
    saved_config = json.loads((save_dir / "config.json").read_text(encoding="utf-8"))
    saved_arch = saved_config["backbone_config"]["architectures"]
    assert saved_arch == [type(wrapper.model).__name__] == ["Qwen3_5Model"], saved_arch

    t0 = time.perf_counter()
    reloaded = JevXForDecision.from_pretrained(str(save_dir))
    reload_s = time.perf_counter() - t0
    reloaded.eval()
    try:
        # Reload Auto-class choice: reloaded backbone class must equal the
        # saved architectures entry (the earlier architectures-mismatch class).
        reloaded_arch = (reloaded.config.backbone_config or {}).get("architectures")
        assert type(reloaded.model).__name__ == "Qwen3_5Model", type(reloaded.model).__name__
        assert reloaded_arch == ["Qwen3_5Model"], reloaded_arch
        assert type(reloaded.adapter).__name__ == type(wrapper.adapter).__name__
        assert reloaded.tokenizer is not None

        with torch.no_grad():
            after = reloaded.choice(state=TEXT_SHORT, questions=QUESTIONS, options=OPTIONS)
        after_probs = [d.probabilities for d in after.decisions]
        err = max(
            float((a.float() - b.float()).abs().max())
            for a, b in zip(before_probs, after_probs)
        )
        assert err < SAME_CALL_ATOL, (
            f"save/load changed decisions by {err:.3e} (bf16 atol {SAME_CALL_ATOL})"
        )
        metadata = json.loads(meta.read_text(encoding="utf-8"))
        record(
            "7_save_load",
            {
                "max_abs_error": err,
                "tolerance": SAME_CALL_ATOL,
                "save_seconds": round(save_s, 3),
                "reload_seconds": round(reload_s, 3),
                "jevx_metadata_sidecar": True,
                "calibration_sidecar": False,
                "calibration_method": reloaded.calibration_engine.method,
                "saved_architectures": saved_arch,
                "reloaded_backbone_class": type(reloaded.model).__name__,
                "reloaded_backbone_config_architectures": reloaded_arch,
                "original_backbone_class": type(wrapper.model).__name__,
                "reloaded_adapter_class": type(reloaded.adapter).__name__,
                "measured": f"max_abs_error={err:.3e}, backbone={type(reloaded.model).__name__}",
            },
        )
        print(
            f"[qwen35-7] save/load max_abs_error={err:.3e} "
            f"arch={saved_arch} reload={reload_s:.1f}s"
        )
    finally:
        del reloaded
        gc.collect()


# ---------------------------------------------------------------------------
# 8a. REAL-DATA SMOKE — 40 SST-2 pairs through JevX (UNTRAINED head)
# ---------------------------------------------------------------------------


def _load_glue_sst2():
    """``nyu-mll/glue``/sst2 validation dataset: cached (offline) first, then
    online; both environment-blocked -> ``pytest.skip`` (repo convention).

    Rationale: the hub can enter SSL-handshake retry storms for minutes on
    script-era HEAD requests; the local arrow cache serves the identical
    rows deterministically. Returns ``(dataset, source_label)``.
    """
    import datasets as hf_datasets

    attempts = []
    try:
        return (
            hf_datasets.load_dataset(
                "nyu-mll/glue",
                "sst2",
                split="validation",
                download_config=hf_datasets.DownloadConfig(local_files_only=True),
            ),
            "offline-cache",
        )
    except Exception as exc:  # noqa: BLE001 - fall through to the online attempt
        attempts.append(f"offline-cache: {type(exc).__name__}: {exc}")
    try:
        return hf_datasets.load_dataset("nyu-mll/glue", "sst2", split="validation"), "online"
    except Exception as exc:  # noqa: BLE001 - classified below
        attempts.append(f"online: {type(exc).__name__}: {exc}")
        if is_environment_error(exc):
            pytest.skip("nyu-mll/glue/sst2 unavailable: " + " | ".join(attempts))
        raise


def _load_sst2():
    """40 SST-2 validation rows. Primary ``stanfordnlp/sst2`` (60 s budget —
    the script-era repo can stall on SSL handshakes); fallback
    ``nyu-mll/glue``/sst2 (identical rows, offline-cache ladder). Network/HTTP
    failures on ALL sources skip (repo convention: environment problems never
    fail the suite); implementation problems still raise. Returns
    ``(rows, source)``."""
    import datasets as hf_datasets

    box: Dict[str, Any] = {}

    def _worker() -> None:
        try:
            box["ds"] = hf_datasets.load_dataset("stanfordnlp/sst2", split="validation")
        except Exception as exc:  # noqa: BLE001 - recorded, then fallback
            box["err"] = f"{type(exc).__name__}: {exc}"

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join(timeout=60)
    if "ds" in box:
        ds, source = box["ds"], "stanfordnlp/sst2"
    else:
        reason = "timeout" if thread.is_alive() else box.get("err", "unknown")[:200]
        ds, how = _load_glue_sst2()
        source = f"nyu-mll/glue/sst2 [{how}] (primary stanfordnlp/sst2 unavailable: {reason})"
    rows = [
        {"text": row["sentence"].strip(), "label": int(row["label"])} for row in ds
    ]
    assert all(r["label"] in (0, 1) for r in rows[:40])
    return rows[:40], source


def _metrics(probs: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """acc / macro-F1 / NLL / 10-bin top-label ECE from probability rows."""
    p = probs.float().clamp(min=1e-12)
    conf, pred = p.max(dim=-1)
    correct = pred.eq(labels)
    acc = float(correct.float().mean())
    # macro-F1 over observed classes (jevx.training.data convention)
    f1s = []
    for cls in labels.unique().tolist():
        tp = int(((pred == cls) & (labels == cls)).sum())
        fp = int(((pred == cls) & (labels != cls)).sum())
        fn = int(((pred != cls) & (labels == cls)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    macro_f1 = float(sum(f1s) / len(f1s))
    nll = float(-p.gather(1, labels.view(-1, 1)).log().mean())
    ece = 0.0
    edges = torch.linspace(0, 1, 11)
    n = labels.shape[0]
    for b in range(10):
        in_bin = (conf >= edges[b]) & (
            conf <= edges[b + 1] if b == 9 else conf < edges[b + 1]
        )
        if bool(in_bin.any()):
            ece += float(in_bin.float().mean()) * abs(
                float(correct[in_bin].float().mean()) - float(conf[in_bin].mean())
            )
    return {"accuracy": acc, "macro_f1": macro_f1, "nll": nll, "ece_10bin": ece}


def test_08a_sst2_smoke_untrained(jevx_case: SimpleNamespace) -> None:
    wrapper = jevx_case.wrapper
    rows, source = _load_sst2()
    assert len(rows) == 40, len(rows)
    question = "Does this sentence express positive sentiment?"
    schema = [["positive", "negative"]]  # label 1 -> positive, 0 -> negative

    texts = [r["text"] for r in rows]
    labels = torch.tensor([r["label"] for r in rows], dtype=torch.long)
    # Short inputs, explicit <=128-token cap (SST-2 sentences are well below).
    tok = wrapper.tokenizer
    lens = [
        len(ids)
        for ids in tok(texts, truncation=True, max_length=128)["input_ids"]
    ]
    assert max(lens) <= 128, max(lens)
    # Length-sorted bucketing: padded token-slots are the dominant cost on
    # this bf16-CPU machine (~0.4 s/slot measured), so process similar-length
    # rows together and restore the original row order for the labels.
    order = sorted(range(40), key=lambda i: lens[i])
    probs_rows: list = [None] * 40
    chunk = 8
    t0 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, 40, chunk):
            idx = order[start : start + chunk]
            batch = dict(
                tok(
                    [texts[i] for i in idx],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128,
                )
            )
            out = wrapper.choice(state=batch, questions=[question], options=schema)
            assert len(out.decisions) == len(idx)
            for slot, (original, dec) in enumerate(zip(idx, out.decisions)):
                assert 0.0 <= float(dec.confidence[0]) <= 1.0
                probs_rows[original] = dec.probabilities.float().squeeze(0)
    wall = time.perf_counter() - t0
    probs = torch.stack(probs_rows, dim=0)  # [40, 2]
    assert probs.shape == (40, 2)
    assert bool(torch.isfinite(probs).all())
    row_sums = probs.sum(dim=-1)
    sum_dev = float((row_sums - 1.0).abs().max())
    assert sum_dev <= BF16_SUM_ATOL, sum_dev

    m = _metrics(probs, labels)
    assert 0.0 <= m["accuracy"] <= 1.0 and 0.0 <= m["macro_f1"] <= 1.0
    assert math.isfinite(m["nll"]) and m["nll"] >= 0.0
    assert 0.0 <= m["ece_10bin"] <= 1.0

    record(
        "8a_sst2_smoke",
        {
            "n": 40,
            "dataset_source": source,
            "question": question,
            "schema": schema[0],
            "label_mapping": {"1": "positive", "0": "negative"},
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "nll": m["nll"],
            "ece_10bin": m["ece_10bin"],
            "row_sum_max_abs_dev": sum_dev,
            "wall_seconds": round(wall, 2),
            "chunk_size": chunk,
            "disclaimer": (
                "UNTRAINED HEAD — pipeline verification only, NOT a quality claim."
            ),
        },
    )
    print(
        f"[qwen35-8a] SST-2 N=40 source={source!r} acc={m['accuracy']:.3f} "
        f"f1={m['macro_f1']:.3f} nll={m['nll']:.4f} ece={m['ece_10bin']:.3f} "
        f"({wall:.1f}s) -- untrained head, pipeline verification only"
    )


# ---------------------------------------------------------------------------
# 9. FAILURE TRANSPARENCY
# ---------------------------------------------------------------------------


def test_09_failure_transparency(jevx_case: SimpleNamespace) -> None:
    wrapper = jevx_case.wrapper
    inputs = jevx_case.raw_refs["inputs_short"]
    outcomes: Dict[str, str] = {}

    def _catch(label: str, fn) -> None:
        try:
            fn()
            outcomes[label] = "NO ERROR"
        except Exception as exc:  # noqa: BLE001 - error quality is the subject
            outcomes[label] = f"{type(exc).__name__}: {exc}"

    _catch(
        "pixel_values_only_state",
        lambda: wrapper.choice(
            state={"pixel_values": torch.randn(1, 3, 16, 16)},
            questions=["q?"],
            options=[["a", "b"]],
        ),
    )
    _catch(
        "input_ids_plus_pixel_values",
        lambda: wrapper.choice(
            state={
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "pixel_values": torch.randn(1, 3, 16, 16),
            },
            questions=["q?"],
            options=[["a", "b"]],
        ),
    )
    _catch("garbage_state_int", lambda: wrapper.choice(state=123, questions=["q?"], options=[["a", "b"]]))
    _catch("garbage_state_int_list", lambda: wrapper.choice(state=[1, 2, 3], questions=["q?"], options=[["a", "b"]]))
    _catch("blank_text_state", lambda: wrapper.choice(state="   ", questions=["q?"], options=[["a", "b"]]))
    _catch("garbage_questions_int", lambda: wrapper.choice(state=TEXT_SHORT, questions=123, options=[["a", "b"]]))
    _catch("noul_with_options", lambda: wrapper.noul(state=TEXT_SHORT, questions=["q?"], options=[["a", "b"]]))

    # Everything must raise — no silent pass-through, no empty messages.
    for label, message in outcomes.items():
        assert message != "NO ERROR", f"{label}: garbage input was accepted silently"
        assert len(message) > 20, f"{label}: error message too terse: {message!r}"
    # The JevX-owned input-contract paths must carry actionable text.
    for label in ("garbage_state_int", "garbage_state_int_list", "blank_text_state",
                  "garbage_questions_int", "noul_with_options"):
        assert outcomes[label].startswith(("TypeError:", "ValueError:")), outcomes[label]

    record(
        "9_failure_transparency",
        {
            "errors": outcomes,
            "quality_notes": {
                "pixel_values_only": "upstream ValueError naming input_ids/inputs_embeds — actionable but does not mention pixel_values",
                "input_ids_plus_pixel_values": "wrapped by call_backbone (Q-013): ValueError 'JevX could not call the backbone Qwen3_5Model ...' naming the model class, forwarded keys and the processor/modality fix; the original upstream AttributeError ('NoneType' object has no attribute 'device', image_grid_thw missing) is preserved as __cause__",
                "garbage_types": "JevX TypeErrors/ValueErrors name the argument and expected types",
            },
        },
    )
    for label, message in outcomes.items():
        print(f"[qwen35-9] {label}: {message[:160]}")


# ---------------------------------------------------------------------------
# 8b. OPTIONAL — HeadTrainer choice-head training (featurize once)
# ---------------------------------------------------------------------------


class _FeaturizedChoiceModel(torch.nn.Module):
    """Test-local stand-in so HeadTrainer can drive the REAL JevX pipeline.

    ``model(batch)`` matches ``make_supervised_step_fn``'s duck-typed call:
    ``batch['state]`` holds integer row indices into a pre-featurized
    ``[N, D]`` buffer (featurization ran ONCE through the real adapter +
    frozen backbone — zero backbone forwards during training). The step then
    executes JevX's own state_proj / question encoder / schema encoder /
    fusion / choice head, so the optimized loss IS the JevX choice loss over
    real JevX components.
    """

    def __init__(self, wrapper: JevXForDecision, feats: torch.Tensor) -> None:
        super().__init__()
        self.wrapper = wrapper
        self.register_buffer("feats", feats)  # [N, D], bf16, no grad

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        from jevx import Option

        idx = torch.as_tensor(batch["state"], dtype=torch.long)
        rows = self.wrapper.state_proj(self.feats[idx])  # [B, D]
        question_texts = list(batch["questions"][0])  # m=1 shared question
        raw_q = self.wrapper.question_encoder.encode_texts(
            question_texts, tokenizer=self.wrapper.tokenizer
        )
        raw_q = self.wrapper.question_encoder(raw_q)
        question = (
            self.wrapper.question_proj(raw_q)
            .unsqueeze(0)
            .expand(rows.shape[0], -1, -1)
            .reshape(rows.shape[0], -1)
        )  # m=1 replicated per row
        # Records carry raw string schemas; coerce to Option like forward does.
        schemas = [
            [
                [opt if isinstance(opt, Option) else Option(key=opt, label=str(opt))
                 for opt in row]
                for row in per_row
            ]
            for per_row in batch["options"]
        ]
        flat_schemas = [row[0] for row in schemas]  # m=1: one schema per row
        features, mask = self.wrapper.schema_encoder.encode(
            flat_schemas, tokenizer=self.wrapper.tokenizer
        )
        option_features = self.wrapper.schema_proj(features)
        option_mask = mask if not bool(mask.all()) else None
        fused = self.wrapper.fusion(rows, question, option_features, option_mask)
        head = self.wrapper.heads["choice"]
        return head(fused.fused, fused.option_features, option_mask=option_mask).logits


def _featurize(wrapper: JevXForDecision, texts: list, *, chunk: int = 8) -> torch.Tensor:
    """Run the real adapter extraction once per chunk -> [N, D] pooled feats.

    Rows are bucketed by token length before chunking: on this bf16-only-CPU
    machine padded token-slots dominate wall time (measured ~0.4 s/slot), and
    sorted chunks of 8 cut padding waste from 8552 to 4696 slots over the 200
    SST-2 train rows versus naive fixed chunking. Rows stay independent
    (per-row masks), so bucketing does not change the pooled values beyond
    bf16 kernel noise; output order is restored to the input order.
    """
    tok = wrapper.tokenizer
    lens = [
        len(ids)
        for ids in tok(texts, truncation=True, max_length=128)["input_ids"]
    ]
    order = sorted(range(len(texts)), key=lambda i: lens[i])
    pooled: list = [None] * len(texts)
    with torch.no_grad():
        for start in range(0, len(order), chunk):
            idx = order[start : start + chunk]
            batch = dict(
                tok(
                    [texts[i] for i in idx],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128,
                )
            )
            rep = wrapper.adapter.extract_representation(
                wrapper.model, batch, strategy=wrapper.config.pooling_method
            )
            for slot, original in enumerate(idx):
                pooled[original] = rep.pooled[slot : slot + 1]
    return torch.cat(pooled, dim=0)


def test_08b_optional_head_training(jevx_case: SimpleNamespace) -> None:
    """OPTIONAL protocol: 200 SST-2 samples, 1 epoch, featurize-once, frozen
    backbone, via HeadTrainer. Reported separately from the smoke battery."""
    from torch.utils.data import DataLoader

    from jevx.training import (
        HeadTrainer,
        accuracy,
        decision_collate_fn,
        make_supervised_step_fn,
    )

    glue_ds, glue_how = _load_glue_sst2()
    glue_rows = [
        {"text": r["sentence"].strip(), "label": int(r["label"])} for r in glue_ds
    ]
    train_rows = glue_rows[40:240]  # 200 rows (disjoint from the 8a smoke slice 0..39)
    assert len(train_rows) == 200

    question = "Does this sentence express positive sentiment?"
    schema = [["positive", "negative"]]  # label 1 -> positive, 0 -> negative

    wrapper = jevx_case.wrapper
    t0 = time.perf_counter()
    # Featurize ONCE through the real adapter + frozen backbone.
    feats = _featurize(wrapper, [r["text"] for r in train_rows])
    featurize_s = time.perf_counter() - t0
    assert tuple(feats.shape) == (200, 1024), tuple(feats.shape)

    records = [
        {
            "state": i,
            "questions": [question],
            "decision_type": "choice",
            "options": [list(schema[0])],
            "labels": [int(row["label"])],
        }
        for i, row in enumerate(train_rows)
    ]

    model = _FeaturizedChoiceModel(wrapper, feats)

    def _eval_logits_labels(m, loader):
        m.eval()
        logits, labels = [], []
        with torch.no_grad():
            for batch in loader:
                logits.append(m(batch))
                labels.append(torch.as_tensor(batch["labels"]).view(-1))
        return torch.cat(logits, dim=0), torch.cat(labels, dim=0)

    # IN-SAMPLE accuracy on the 200 featurized rows (pre vs post fit) — no
    # second featurization pass; labeled in-sample in the recorded protocol.
    eval_loader = DataLoader(
        records, batch_size=20, shuffle=False, collate_fn=decision_collate_fn
    )
    pre_logits, train_labels = _eval_logits_labels(model, eval_loader)
    pre_acc = accuracy(pre_logits, train_labels)

    torch.manual_seed(42)
    trainer = HeadTrainer(model, lr=1e-3, seed=42, grad_clip=1.0)
    # Default step is make_supervised_step_fn's decision-type dispatch; the
    # explicit build documents which step the trainer runs.
    step = make_supervised_step_fn(model)
    assert callable(step)
    train_loader = DataLoader(
        records,
        batch_size=10,
        shuffle=True,
        collate_fn=decision_collate_fn,
        generator=trainer.generator,
    )
    history = trainer.fit(train_loader, epochs=1)
    losses = history.losses
    assert len(history) == 1
    assert all(math.isfinite(v) for v in losses)

    post_logits, post_labels = _eval_logits_labels(model, eval_loader)
    assert torch.equal(train_labels, post_labels)
    post_acc = accuracy(post_logits, post_labels)

    record(
        "8b_optional_head_training",
        {
            "executed": True,
            "protocol": {
                "n_train": 200,
                "epochs": 1,
                "batch_size": 10,
                "lr": 1e-3,
                "optimizer": "AdamW (HeadTrainer default)",
                "grad_clip": 1.0,
                "seed": 42,
                "backbone": "frozen; featurized ONCE via the real adapter (0 backbone forwards during training)",
                "trainable": "JevX-side parameters (state_proj/question/schema/fusion/choice head)",
                "train_rows": "glue sst2 validation rows 40..239",
                "eval": "IN-SAMPLE accuracy on the same 200 featurized rows (pre vs post)",
                "featurize_seconds": round(featurize_s, 1),
                "dataset_source": f"nyu-mll/glue/sst2 [{glue_how}]",
            },
            "epoch1_loss": losses[0],
            "pre_train_in_sample_accuracy": pre_acc,
            "post_train_in_sample_accuracy": post_acc,
            "note": (
                "protocol evidence only — 1 epoch, frozen random-init features; "
                "NOT a quality claim"
            ),
        },
    )
    print(
        f"[qwen35-8b] HeadTrainer 1 epoch / 200 samples: loss={losses[0]:.4f} "
        f"in-sample acc {pre_acc:.3f} -> {post_acc:.3f} "
        f"(featurize {featurize_s:.1f}s, frozen backbone)"
    )


# ---------------------------------------------------------------------------
# Environment header (populated at import; battery keys added by record())
# ---------------------------------------------------------------------------

_RESULTS.setdefault("format", "qwen35-jevx-validation-v1")
try:
    _RESULTS["environment"] = _environment_block()
except Exception as exc:  # noqa: BLE001 - environment block is informational
    _RESULTS["environment"] = {"error": f"{type(exc).__name__}: {exc}"}
