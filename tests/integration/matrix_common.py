# Copyright (c) 2026 ysnrfd
"""Shared harness for the REAL pretrained validation matrix (Phases 20 +
46-48 + 70-71 + 92-94).

One entry point — :func:`execute_matrix_case` — runs the full §92 per-model
protocol against a REAL hub checkpoint and returns a JSON-serializable
record (also merged into ``vendor/reports/_jevx_matrix_results.json`` at
interpreter exit):

1. load facts (params, capability report, adapter, wall-clock load time);
2. REAL modality inputs (text sentences / PIL image fixture / soundfile wav
   fixture / processor-built multimodal batches);
3. decision invariants on ``.choice`` / ``.score`` / ``.noul`` (probability
   sums, variable-K exact zeros, bounds, confidence range) — these ARE the
   forward validation;
4. ``save_pretrained`` → ``from_pretrained`` → identical decisions
   (``max_abs_error``);
5. ``base_forward`` vs a freshly loaded raw ``AutoModel`` (``max_abs_error``,
   contract < 1e-5);
6. shared-state property (Def. 11 / T7): exactly ONE backbone forward for
   m = 4 questions (depth-tracked forward counter);
7. §71 extract-representation correctness: adapter-boundary ``pooled``
   vs an INDEPENDENT manual recomputation (own forward + own pooling math —
   never ``jevx.pool``), plus the projected ``state_representation``
   vs ``state_proj(manual)``.

Environment/network failures skip with an explicit reason (mirroring
``test_jevx_e2e_tiny.py``); implementation bugs re-raise and fail.

Fixture assets (deterministic, generated once, checked in):

* ``tests/golden/fixtures/input_image_224.png`` — 224×224 RGB pattern
  (seeded arithmetic pattern + circle), consumed by every vision family;
* ``tests/golden/fixtures/input_audio_1s.wav`` — 1 s / 16 kHz two-tone
  waveform written with ``soundfile``, read back with ``soundfile`` for
  every audio family (documented explicitly: real file → array, not a
  synthetic tensor).
"""

from __future__ import annotations

import atexit
import inspect
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import AutoModel

from jevx import JevX, JevXForDecision

ROOT = Path(__file__).resolve().parents[2]
VERIFIED_IDS_FILE = ROOT / "vendor" / "reports" / "_hf_verified_ids.json"
ENV_PROBE_FILE = ROOT / "vendor" / "reports" / "_matrix_env_probe.json"
RESULTS_FILE = ROOT / "vendor" / "reports" / "_jevx_matrix_results.json"
FIXTURES_DIR = ROOT / "tests" / "golden" / "fixtures"
IMAGE_FIXTURE = FIXTURES_DIR / "input_image_224.png"
WAV_FIXTURE = FIXTURES_DIR / "input_audio_1s.wav"

#: Real, fixed English sentences used as the text state everywhere.
REAL_TEXT = (
    "The maintenance crew inspected the bridge deck and confirmed that the "
    "support damage had been repaired before the storm arrived."
)
REAL_TEXTS = [
    REAL_TEXT,
    "Analysts expect the quarterly report to show steady revenue growth across all regions.",
]

#: Realistic question schemas for the invariant pass (m=2, variable K).
QUESTIONS = ["Is this content safe to deploy?", "Which disposition fits this item?"]
OPTIONS = [["safe", "unsafe"], ["approve", "review", "reject"]]
SCORE_QUESTIONS = ["How would you rate the severity?"]
SCORE_OPTIONS = [["1", "2", "3", "4", "5"]]
NOUL_QUESTIONS = ["Does this text contain a guarantee?"]

#: Substrings marking an environment/network problem (skip-eligible) rather
#: than an implementation bug (which must fail loudly).
_ENV_MARKERS = (
    "401",
    "403",
    "404",
    "connection",
    "timed out",
    "timeout",
    "max retries",
    "couldn't connect",
    "could not fetch",
    "offline",
    "not found",
    "gated",
    "permission",
    "proxy",
    "name resolution",
    "temporary failure in name resolution",
    "no space left",
)


def is_environment_error(exc: BaseException) -> bool:
    """True iff ``exc`` smells like network/HTTP/disk, not a JevX bug."""
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _ENV_MARKERS)


def verified_ids() -> Dict[str, Any]:
    """The Phase-1 HF-API-verified id map (empty dict when unavailable)."""
    if VERIFIED_IDS_FILE.is_file():
        return json.loads(VERIFIED_IDS_FILE.read_text(encoding="utf-8"))
    return {}


def resolve_sha(model_id: str) -> Optional[str]:
    """Resolved commit sha: env-probe cache first, live HF API as fallback."""
    if ENV_PROBE_FILE.is_file():
        probe = json.loads(ENV_PROBE_FILE.read_text(encoding="utf-8"))
        sha = (probe.get(model_id) or {}).get("sha")
        if sha:
            return str(sha)
    try:  # pragma: no cover - network fallback
        import urllib.request

        with urllib.request.urlopen(
            f"https://huggingface.co/api/models/{model_id}", timeout=30
        ) as resp:
            return str(json.load(resp).get("sha"))
    except Exception:  # noqa: BLE001 - sha is metadata, never a test gate
        return None


class ForwardCounter:
    """Counts outermost backbone forward calls (depth-tracked).

    For encoder-decoder children the adapter calls ``model.encoder`` directly
    (audit Q6), so the encoder submodule is instrumented too; depth tracking
    ensures a nested encoder call inside a full forward counts once.
    """

    def __init__(self, modules) -> None:
        self.depth = 0
        self.calls = 0
        self._handles = []
        for module in modules:
            self._handles.append(module.register_forward_pre_hook(self._pre))
            self._handles.append(module.register_forward_hook(self._post))

    def _pre(self, module, args) -> None:
        if self.depth == 0:
            self.calls += 1
        self.depth += 1

    def _post(self, module, args, output) -> None:
        self.depth -= 1

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()


def max_abs_error(a: Any, b: Any) -> float:
    """Max abs difference over parallel output structures (tensors/dicts)."""
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.shape == b.shape, (
            f"output structure differs: {type(a)} {getattr(a, 'shape', None)} vs "
            f"{type(b)} {getattr(b, 'shape', None)}"
        )
        if a.numel() == 0:
            return 0.0
        return float((a.float() - b.float()).abs().max())
    if isinstance(a, dict):
        assert isinstance(b, dict)
        keys_a = [k for k, v in a.items() if v is not None]
        keys_b = [k for k, v in b.items() if v is not None]
        assert keys_a == keys_b, f"output fields differ: {keys_a} vs {keys_b}"
        return max((max_abs_error(a[k], b[k]) for k in keys_a), default=0.0)
    if isinstance(a, (tuple, list)):
        assert isinstance(b, (tuple, list)) and len(a) == len(b)
        return max((max_abs_error(x, y) for x, y in zip(a, b)), default=0.0)
    return 0.0  # None / foreign objects (e.g. empty past_key_values caches)


# ---------------------------------------------------------------------------
# Fixture assets (deterministic)
# ---------------------------------------------------------------------------


def ensure_fixture_assets() -> Path:
    """Create the image/wav fixtures if missing (identical bytes every run)."""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    if not IMAGE_FIXTURE.is_file():
        from PIL import Image

        seed = 0
        yy, xx = np.mgrid[0:224, 0:224]
        r = ((xx * 5 + seed) % 256).astype(np.uint8)
        g = ((yy * 11 + seed * 3) % 256).astype(np.uint8)
        b = (((xx + yy) * 7 + seed * 7) % 256).astype(np.uint8)
        # deterministic filled circle at a fixed center
        cx, cy, rad = 112, 96, 48
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= rad**2
        r[mask], g[mask], b[mask] = 240, 40, 40
        Image.fromarray(np.stack([r, g, b], axis=-1), mode="RGB").save(IMAGE_FIXTURE)
    if not WAV_FIXTURE.is_file():
        import soundfile as sf

        sr = 16000
        t = np.arange(sr, dtype=np.float64) / sr  # exactly 1.0 s
        envelope = np.minimum(1.0, t / 0.05) * np.minimum(1.0, (1.0 - t) / 0.05)
        wave = (0.6 * np.sin(2 * np.pi * 440.0 * t) + 0.3 * np.sin(2 * np.pi * 880.0 * t))
        wave = (wave * envelope).astype(np.float32)
        sf.write(str(WAV_FIXTURE), wave, sr, subtype="PCM_16")
    return FIXTURES_DIR


def fixture_image():
    """Deterministic 224×224 PIL image (generated on demand)."""
    ensure_fixture_assets()
    from PIL import Image

    return Image.open(IMAGE_FIXTURE).convert("RGB")


def fixture_waveform() -> np.ndarray:
    """1 s / 16 kHz float32 waveform READ BACK FROM THE WAV FIXTURE.

    Documented input path: real ``.wav`` file on disk → ``soundfile.read``
    → ``np.ndarray`` → processor/feature extractor (the audit's "array
    input" path, now exercised through an actual file).
    """
    ensure_fixture_assets()
    import soundfile as sf

    data, sr = sf.read(str(WAV_FIXTURE), dtype="float32", always_2d=False)
    assert sr == 16000 and data.shape[0] == 16000, (sr, data.shape)
    return data


# ---------------------------------------------------------------------------
# Independent (manual) reference math for §71
# ---------------------------------------------------------------------------


def _align_mask(model: Any, mask: Any, seq: torch.Tensor) -> Optional[torch.Tensor]:
    """Shape-align an input mask to ``seq``'s frame axis (family hook, duck-typed)."""
    if isinstance(mask, torch.Tensor) and tuple(mask.shape) == tuple(seq.shape[:2]):
        return mask
    if isinstance(mask, torch.Tensor):
        hook = getattr(model, "_get_feature_vector_attention_mask", None)
        if callable(hook):
            try:
                aligned = hook(seq.shape[1], mask)
                if isinstance(aligned, torch.Tensor) and tuple(aligned.shape) == tuple(
                    seq.shape[:2]
                ):
                    return aligned
            except Exception:  # noqa: BLE001 - fall through to no-mask
                return None
    return None


def manual_pool(
    seq: torch.Tensor, mask: Optional[torch.Tensor], strategy: Optional[str]
) -> torch.Tensor:
    """INDEPENDENT pooling reference — deliberately NOT ``jevx.pool``.

    Implements the documented formulas directly (mask-aware mean / first
    valid / last valid / max; plain mean when no mask is available) so an
    accidental change to ``jevx/representations/pooling.py`` shows up as a
    §71 extraction error instead of silently agreeing with itself.
    """
    if strategy is None:
        strategy = "masked_mean"
    if strategy == "encoder_pool":
        strategy = "masked_mean"
    if strategy == "native":  # native implies a pooler field — handled by caller
        raise ValueError("manual_pool: 'native' is not a sequence strategy")
    if mask is None:
        if strategy in ("masked_mean", "mean"):
            return seq.mean(dim=1)
        if strategy == "cls":
            return seq[:, 0, :]
        if strategy == "last_valid":
            return seq[:, -1, :]
        if strategy == "max":
            return seq.max(dim=1).values
        raise ValueError(f"manual_pool: unknown strategy {strategy!r}")
    m = mask.to(torch.bool)
    if strategy == "mean":
        return seq.mean(dim=1)
    if strategy == "masked_mean":
        mf = m.unsqueeze(-1).to(seq.dtype)
        return (seq * mf).sum(dim=1) / m.sum(dim=1).clamp(min=1).unsqueeze(-1).to(seq.dtype)
    if strategy == "cls":
        big = torch.full_like(m, seq.shape[1], dtype=torch.int64)
        idx = torch.where(m, m.to(torch.int64), big).argmin(dim=-1)
        return seq[torch.arange(seq.shape[0], device=seq.device), idx]
    if strategy == "last_valid":
        idx = seq.shape[1] - 1 - m.to(torch.int64).flip(-1).argmax(dim=-1)
        return seq[torch.arange(seq.shape[0], device=seq.device), idx]
    if strategy == "max":
        very_neg = torch.finfo(seq.dtype).min
        return seq.masked_fill(~m.unsqueeze(-1), very_neg).max(dim=1).values
    raise ValueError(f"manual_pool: unknown strategy {strategy!r}")


def _filtered(model: Any, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Filter ``inputs`` to the model's own forward signature (independent copy)."""
    params = set(inspect.signature(model.forward).parameters)
    allowed = {
        name
        for name, p in inspect.signature(model.forward).parameters.items()
        if name != "self"
        and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    } | {"output_hidden_states", "output_attentions", "return_dict"}
    fwd = {k: v for k, v in inputs.items() if k in allowed}
    if "use_cache" in params:
        fwd["use_cache"] = False
    return fwd


def manual_expected_representation(wrapper: JevXForDecision, inputs: Dict[str, Any]) -> torch.Tensor:
    """§71 reference: pooled state vector computed WITHOUT JevX extraction code.

    Runs the backbone (or its encoder directly for encoder-decoder families,
    audit Q6) with signature-filtered inputs, reads the raw field the
    adapter claims, and pools with :func:`manual_pool` when no native field
    is involved.
    """
    model = wrapper.model
    config_strategy = wrapper.config.pooling_method
    native_ok = config_strategy in (None, "native")

    def _effective() -> Optional[str]:
        if config_strategy not in (None, "native"):
            return config_strategy
        return getattr(wrapper.adapter, "default_pooling", "masked_mean")

    # Encoder-decoder adapters extract encoder states by default (the wrapper
    # forward never passes mode=, so extract_representation runs
    # mode="encoder_state" → a direct encoder call, audit Q6).
    if getattr(wrapper.adapter, "name", None) == "encoder_decoder":
        get_encoder = getattr(model, "get_encoder", None)
        encoder = get_encoder() if callable(get_encoder) else getattr(model, "encoder", None)
        assert encoder is not None, "encoder-decoder model exposes no encoder route"
        enc_fwd = _filtered(encoder, inputs)
        enc_fwd.pop("use_cache", None)
        seq = encoder(**enc_fwd).last_hidden_state
        mask = _align_mask(model, inputs.get("attention_mask"), seq)
        return manual_pool(seq, mask, _effective())

    out = model(**_filtered(model, inputs))
    if native_ok:
        if getattr(out, "pooler_output", None) is not None:
            return out.pooler_output
        for semantic in ("sentence_embedding", "text_embeds", "image_embeds"):
            value = getattr(out, semantic, None)
            if value is not None:
                return value
    seq = getattr(out, "last_hidden_state", None)
    assert seq is not None, f"no usable raw output field on {type(model).__name__}"
    mask = _align_mask(model, inputs.get("attention_mask"), seq)
    return manual_pool(seq, mask, _effective())


def extract_via_adapter(wrapper: JevXForDecision, inputs: Dict[str, Any]):
    """Adapter-boundary extraction exactly as the wrapper's forward does it."""
    return wrapper.adapter.extract_representation(
        wrapper.model,
        inputs,
        strategy=wrapper.config.pooling_method,
        output_hidden_states=False,
    )


# ---------------------------------------------------------------------------
# State preparation (REAL modality inputs)
# ---------------------------------------------------------------------------


def build_state(kind: str) -> Any:
    """Raw state for one modality kind (see module docstring for provenance)."""
    if kind == "text":
        return REAL_TEXT
    if kind == "vision":
        return {"images": fixture_image()}
    if kind == "audio":
        return {"audio": fixture_waveform()}
    if kind == "multimodal":  # CLIP: real image + real text through the processor
        return {"images": fixture_image(), "texts": "a red circle drawn on a checkered background"}
    if kind == "vlm":  # LLaVA-NeXT: real image + real prompt with <image> placeholder
        return {"images": fixture_image(), "texts": "<image> Describe this image briefly."}
    if kind == "siglip":
        raise ValueError("siglip state needs the wrapper (explicit max_length padding)")
    raise ValueError(f"unknown state kind {kind!r}")


def build_siglip_state(wrapper: JevXForDecision) -> Dict[str, Any]:
    """SigLIP real inputs — text padded to ``max_length`` (model docstring
    requirement: 'as that's how the model was trained'; the generic adapter
    call passes no processor kwargs, so tensors are prepared explicitly
    here and handed to JevX as an already-processed dict)."""
    proc = wrapper.processor
    assert proc is not None, "siglip wrapper resolved no processor"
    # Repo quirk: the processor output carries NO attention_mask, so build the
    # pad mask explicitly — the SigLIP convention (model docstring: padding
    # "as that's how the model was trained") still applies.
    tok = wrapper.tokenizer
    assert tok is not None and tok.pad_token_id is not None, "siglip needs a pad token"
    batch = proc(
        text="a red circle drawn on a checkered background",
        images=fixture_image(),
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = batch["input_ids"]
    max_len = int(getattr(tok, "model_max_length", 0) or 0)
    if 0 < max_len < 1_000_000:
        assert int(input_ids.shape[-1]) == max_len, (
            f"siglip text must be padded to max_length={max_len}, got {input_ids.shape[-1]}"
        )
    assert int(input_ids[0, -1]) == int(tok.pad_token_id), (
        "siglip text pooler reads the LAST position — it must be a pad token "
        "(padding=max_length convention)"
    )
    result = dict(batch)
    result["attention_mask"] = (input_ids != int(tok.pad_token_id)).long()
    return result


def build_ast_state(wrapper: JevXForDecision) -> Dict[str, Any]:
    """AST real inputs — feature extractor REBUILT from the MODEL config.

    The tiny-random-ASTModel repo ships a preprocessor_config.json whose
    geometry (feature_size=1, max_length=1024, num_mel_bins=128) is
    inconsistent with its own model config (num_mel_bins=16,
    max_length=24 → 96 patches + 2 special tokens = 98 position
    embeddings). Using the repo's preprocessor verbatim overflows the
    position table; the AST docs prescribe max_length/num_mel_bins matching
    the checkpoint, so both are taken from ``model.config`` here. Still a
    REAL waveform → real kaldi-style fbank (numpy path, no torchaudio).
    """
    from transformers import ASTFeatureExtractor

    cfg = wrapper.model.config
    fe = ASTFeatureExtractor(
        feature_size=1,  # mono waveform channel
        sampling_rate=16000,
        num_mel_bins=int(cfg.num_mel_bins),
        max_length=int(cfg.max_length),
        do_normalize=True,
        return_attention_mask=False,
    )
    batch = fe(fixture_waveform(), sampling_rate=16000, return_tensors="pt")
    values = batch["input_values"]
    assert tuple(values.shape[-2:]) == (int(cfg.max_length), int(cfg.num_mel_bins)), (
        f"AST input geometry {tuple(values.shape)} does not match model config "
        f"(max_length={cfg.max_length}, num_mel_bins={cfg.num_mel_bins})"
    )
    return dict(batch)


def build_whisper_state(wrapper: JevXForDecision) -> Dict[str, Any]:
    """Whisper real inputs — processor on the wav fixture, mel trimmed to the
    length the checkpoint's own convolution accepts.

    The tiny-random checkpoint sets ``max_source_positions`` such that the
    encoder demands mel length ``max_source_positions * conv1.stride *
    conv2.stride`` (= 60), while ``WhisperProcessor`` always emits the
    standard 3000-frame mel; upstream modeling_whisper.py:616 hard-raises on
    the mismatch. Trimming the REAL mel to the model-required length is the
    minimal correct input adaptation (documented, not a JevX code path).
    """
    proc = wrapper.processor
    assert proc is not None, "whisper wrapper resolved no processor"
    batch = proc(audio=fixture_waveform(), sampling_rate=16000, return_tensors="pt")
    features = batch["input_features"]
    model = wrapper.model
    encoder = model.encoder  # conv1/conv2 live on the encoder stack
    expected = (
        int(model.config.max_source_positions)
        * int(encoder.conv1.stride[0])
        * int(encoder.conv2.stride[0])
    )
    if int(features.shape[-1]) != expected:
        assert int(features.shape[-1]) > expected, (
            f"whisper mel shorter than required: {features.shape[-1]} < {expected}"
        )
        features = features[..., :expected]
    return {"input_features": features}


def build_vlm_state(wrapper: JevXForDecision) -> Dict[str, Any]:
    """LLaVA-NeXT real inputs — real image + real prompt through the processor.

    Repo quirk handled here: the tiny checkpoint's ``preprocessor_config.json``
    ships ``patch_size: null`` (its image processor cannot count image tokens
    without it — upstream ``TypeError: int // NoneType`` at
    processing_llava_next.py:100), so the patch size is taken from the
    model's own ``vision_config.patch_size`` before processing. Everything
    else (grid pinpoints, crop size) comes from the repo config.
    """
    proc = wrapper.processor
    assert proc is not None, "llava wrapper resolved no processor"
    model_cfg = wrapper.model.config
    # The tiny repo ships processor_config.json without patch_size; the
    # processor cannot count image tokens without it (upstream TypeError:
    # int // NoneType at processing_llava_next.py:100). Probe-verified
    # (vendor/reports/_probe_llava_tokens.py): patch_size from the model's
    # vision_config + the repo's own vision_feature_select_strategy=None /
    # num_additional_image_tokens=0 give exactly matching image-token and
    # feature counts (1155 == 1155); overriding the strategy to the model
    # config's "default" breaks the count by one.
    image_processor = getattr(proc, "image_processor", None)
    if image_processor is not None and getattr(image_processor, "patch_size", None) is None:
        image_processor.patch_size = int(model_cfg.vision_config.patch_size)
    if getattr(proc, "patch_size", None) is None:
        proc.patch_size = int(model_cfg.vision_config.patch_size)
    return {"images": fixture_image(), "texts": "<image> Describe this image briefly."}


def prepared_inputs(wrapper: JevXForDecision, state: Any) -> Dict[str, Any]:
    """Model kwargs for ``base_forward``/extraction — through the adapter."""
    return wrapper.adapter.prepare_inputs(
        state, tokenizer=wrapper.tokenizer, processor=wrapper.processor, model=wrapper.model
    )


# ---------------------------------------------------------------------------
# Invariants (step 3)
# ---------------------------------------------------------------------------


def run_invariants(wrapper: JevXForDecision, state: Any) -> Dict[str, Any]:
    """choice/score/noul output invariants — the forward validation proper."""
    flags: Dict[str, Any] = {}
    with torch.no_grad():
        choice_out = wrapper.choice(state=state, questions=QUESTIONS, options=OPTIONS)
    assert len(choice_out.decisions) == 2, len(choice_out.decisions)
    for dec, keys in zip(choice_out.decisions, OPTIONS):
        probs = dec.probabilities
        k = len(keys)
        assert (probs >= 0).all()
        assert bool((probs[0, k:] == 0).all()), "variable-K mask must be exact zeros"
        assert float(probs[0, :k].sum()) == _approx(1.0)
        assert 0 <= int(dec.predicted_index[0]) < k
        assert dec.predicted_value in keys
        assert 0.0 <= float(dec.confidence[0]) <= 1.0
    flags["choice_questions"] = len(QUESTIONS)

    with torch.no_grad():
        score_out = wrapper.score(
            state=state, questions=SCORE_QUESTIONS, options=SCORE_OPTIONS
        )
        noul_out = wrapper.noul(state=state, questions=NOUL_QUESTIONS)
    es = float(score_out.decisions[0].expected_score[0])
    assert -1e-6 <= es <= 4 + 1e-6, es
    p_yes = float(noul_out.decisions[0].probabilities[0, 0])
    assert 0.0 < p_yes < 1.0, p_yes
    assert 0.0 <= float(noul_out.decisions[0].confidence[0]) <= 1.0
    flags["score_expected_score"] = es
    flags["noul_p_yes"] = p_yes

    # Batched text states (B=2) — only meaningful for str/list-of-str states.
    if isinstance(state, str):
        batch_state = REAL_TEXTS
        with torch.no_grad():
            batch_out = wrapper.choice(
                state=batch_state, questions=QUESTIONS, options=OPTIONS
            )
        assert len(batch_out.decisions) == 4, len(batch_out.decisions)
        assert tuple(batch_out.state_representation.shape) == (2, wrapper.state_proj.out_dim)
        flags["batch_B2_decisions"] = 4
    return flags


def _approx(value: float, abs_tol: float = 1e-6):
    import pytest

    return pytest.approx(value, abs=abs_tol)


# ---------------------------------------------------------------------------
# Results collection
# ---------------------------------------------------------------------------

_RESULTS: Dict[str, Dict[str, Any]] = {}


def record_result(family: str, record: Dict[str, Any]) -> None:
    """Store a per-family record and print the harvestable result line."""
    _RESULTS[family] = record
    print(
        "MATRIX RESULT | family={family} | id={model_id} | sha={sha} | params={params} | "
        "load_s={load_seconds:.2f} | adapter={adapter} | conversion={jevx_conversion} | "
        "rep_field={representation_field} | save_load={save_load_max_abs_error:.3e} | "
        "base_fwd={base_forward_max_abs_error:.3e} | fwd_calls_m4={backbone_forward_calls_m4} | "
        "extract_err={extract_max_abs_error:.3e} | state_repr_err={state_representation_max_abs_error:.3e}".format(
            **record
        ),
        flush=True,
    )


def _write_results() -> None:
    """Merge this process's records into the JSON artifact (best effort)."""
    if not _RESULTS:
        return
    try:
        existing: Dict[str, Any] = {}
        if RESULTS_FILE.is_file():
            existing = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
        families = existing.get("families", {})
        families.update(_RESULTS)
        payload = {
            "format": "jevx-matrix-results-v1",
            "environment": {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "transformers": __import__("transformers").__version__,
                "platform": platform.platform(),
                "processor": platform.processor() or "cpu",
            },
            "families": families,
        }
        RESULTS_FILE.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 - artifact writing must never fail a run
        print(f"WARNING: could not write {RESULTS_FILE}: {exc}", flush=True)


atexit.register(_write_results)


# ---------------------------------------------------------------------------
# The per-model protocol (§92)
# ---------------------------------------------------------------------------


def load_wrapper(model_id: str, **jevx_kwargs: Any):
    """``JevX.from_pretrained`` with load-time measurement; env errors → skip."""
    start = time.perf_counter()
    attempts: List[str] = []
    try:
        wrapper = JevX.from_pretrained(model_id, **jevx_kwargs)
    except Exception as exc:  # noqa: BLE001 - classified below
        attempts.append(f"{model_id}: {type(exc).__name__}: {exc}")
        if not is_environment_error(exc):
            raise  # implementation bug — fail, do not mask as blocked
        import pytest

        pytest.skip(
            f"{model_id}: environment-blocked load ({type(exc).__name__}: {exc}). "
            "Attempts: " + " | ".join(attempts)
        )
    load_seconds = time.perf_counter() - start
    wrapper.eval()
    return wrapper, load_seconds


def execute_matrix_case(
    family: str,
    model_id: str,
    kind: str,
    wrapper: JevXForDecision,
    load_seconds: float,
    tmp_path: Path,
    *,
    test_file: str,
    notes: str = "",
) -> Dict[str, Any]:
    """Run steps 1-7 of the per-model protocol; return the record."""
    # -- 1. load facts ------------------------------------------------------
    n_params = sum(p.numel() for p in wrapper.parameters())
    report = wrapper.capability_report
    conversion = report.jevx_conversion
    assert conversion in ("supported", "experimental"), (
        f"{model_id}: unexpected jevx_conversion={conversion!r} (reasons: {report.reasons})"
    )
    expected_adapter = {
        "text": ("encoder", "encoder_decoder", "decoder"),
        "vision": ("vision",),
        "audio": ("audio",),
        "ast": ("audio",),
        "whisper": ("encoder_decoder",),
        "multimodal": ("multimodal",),
        "vlm": ("multimodal",),
        "siglip": ("multimodal",),
    }[kind]
    adapter_name = getattr(wrapper.adapter, "name", type(wrapper.adapter).__name__)
    assert adapter_name in expected_adapter, (
        f"{family}: adapter {adapter_name!r} not in expected {expected_adapter}"
    )

    # -- 2. real inputs -----------------------------------------------------
    if kind == "siglip":
        state = build_siglip_state(wrapper)
    elif kind == "ast":
        state = build_ast_state(wrapper)
    elif kind == "whisper":
        state = build_whisper_state(wrapper)
    elif kind == "vlm":
        state = build_vlm_state(wrapper)
    else:
        state = build_state(kind)

    # -- 3. invariants ------------------------------------------------------
    invariant_flags = run_invariants(wrapper, state)

    # -- 4. save -> reload: identical decisions ----------------------------
    save_dir = tmp_path / f"jevx-matrix-{family}"
    with torch.no_grad():
        before = wrapper.choice(state=state, questions=QUESTIONS, options=OPTIONS)
    wrapper.save_pretrained(save_dir)
    reloaded = JevXForDecision.from_pretrained(save_dir)
    reloaded.eval()
    with torch.no_grad():
        after = reloaded.choice(state=state, questions=QUESTIONS, options=OPTIONS)
    save_load_err = max(
        float((a.probabilities - b.probabilities).abs().max())
        for a, b in zip(before.decisions, after.decisions)
    )
    assert save_load_err <= 1e-6, (
        f"{family}: save/load changed decisions by {save_load_err:.3e}"
    )
    del reloaded

    # -- 5. base_forward vs a freshly loaded raw model ----------------------
    inputs = prepared_inputs(wrapper, state)
    raw_model = AutoModel.from_pretrained(model_id)
    raw_model.eval()
    assert type(raw_model) is type(wrapper.model), (
        f"{family}: raw AutoModel is {type(raw_model).__name__} but the wrapper embedded "
        f"{type(wrapper.model).__name__} — base_forward comparison would be meaningless"
    )
    with torch.no_grad():
        wrapped_base = wrapper.base_forward(**inputs)
        raw_base = raw_model(**inputs)
    base_err = max_abs_error(wrapped_base, raw_base)
    assert base_err < 1e-5, (
        f"{model_id}: base_forward diverged from the raw checkpoint by {base_err:.3e}"
    )
    del raw_model, raw_base, wrapped_base

    # -- 6. shared-state: exactly ONE backbone forward for m=4 -------------
    extra = []
    if report.encoder_decoder and hasattr(wrapper.model, "encoder"):
        extra.append(wrapper.model.encoder)
    counter = ForwardCounter([wrapper.model, *extra])
    try:
        m = 4
        start_fwd = time.perf_counter()
        with torch.no_grad():
            wrapper.choice(
                state=state,
                questions=[f"question {i}?" for i in range(m)],
                options=[[f"q{i}-a", f"q{i}-b"] for i in range(m)],
            )
        forward_seconds = time.perf_counter() - start_fwd
    finally:
        counter.close()
    assert counter.calls == 1, (
        f"{model_id}: state featurization ran the backbone {counter.calls} times "
        "for m=4 (contract: exactly once)"
    )

    # -- 7. §71 extract-representation correctness --------------------------
    with torch.no_grad():
        rep = extract_via_adapter(wrapper, inputs)
        if wrapper.config.pooling_method is None:
            # With the default strategy the detector's preferred field and the
            # field actually extracted must agree (alignment bug guard).
            assert report.representation_field == rep.field, (
                f"{family}: capability_report.representation_field="
                f"{report.representation_field!r} but extraction selected {rep.field!r}"
            )
        expected = manual_expected_representation(wrapper, inputs)
    extract_err = float((rep.pooled.float() - expected.float()).abs().max())
    assert extract_err < 1e-5, (
        f"{family}: adapter pooled ({rep.field}) diverged from the independent manual "
        f"reference by {extract_err:.3e}"
    )
    # projected state representation through a real forward call
    one_question = [QUESTIONS[0]]
    one_options = [OPTIONS[0]]
    with torch.no_grad():
        out_fw = wrapper.choice(state=state, questions=one_question, options=one_options)
        projected = wrapper.state_proj(expected)
        state_repr_err = float(
            (out_fw.state_representation.float() - projected.float()).abs().max()
        )
    assert state_repr_err < 1e-4, (
        f"{family}: JevXOutput.state_representation diverged from "
        f"state_proj(manual pooled) by {state_repr_err:.3e}"
    )

    record: Dict[str, Any] = {
        "family": family,
        "model_id": model_id,
        "sha": resolve_sha(model_id),
        "params": n_params,
        "load_seconds": round(load_seconds, 3),
        "adapter": adapter_name,
        "jevx_conversion": conversion,
        "representation_field": report.representation_field,
        "extracted_field": rep.field,
        "output_class": report.output_class,
        "model_type": report.model_type,
        "architecture": report.architecture,
        "input_kind": kind,
        "state_repr_layout": f"{rep.layout}: {tuple(rep.pooled.shape)}",
        "invariants": invariant_flags,
        "save_load_max_abs_error": save_load_err,
        "base_forward_max_abs_error": base_err,
        "backbone_forward_calls_m4": counter.calls,
        "forward_seconds_m4": round(forward_seconds, 4),
        "extract_max_abs_error": extract_err,
        "state_representation_max_abs_error": state_repr_err,
        "test": test_file,
        "notes": notes,
    }
    record_result(family, record)
    return record
