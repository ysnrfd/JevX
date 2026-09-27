# Copyright (c) 2026 ysnrfd
"""Phase 46 probe: verify candidate matrix checkpoint ids + featurizer reachability.

Runs WITHOUT pytest. For every candidate id:
  * GET https://huggingface.co/api/models/<id> -> exists / resolved commit sha
    / whether the repo ships tokenizer.json (fast path, no sentencepiece needed)
  * AutoConfig resolution -> model_type, architectures, hidden width
  * featurizer ladder as JevX._resolve_featurizers does it
    (text-only -> AutoTokenizer; else AutoProcessor -> AutoFeatureExtractor)
  * optional libs presence (soundfile / sentencepiece)

Output: JSON map written to vendor/reports/_matrix_env_probe.json and a
human-readable table on stdout. Never downloads full weights (config +
tokenizer files only), so it is cheap on RAM/disk.
"""

from __future__ import annotations

import concurrent.futures
import importlib.util
import json
import sys
import traceback
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "_matrix_env_probe.json"

CANDIDATES = {
    # family -> list of ids (primary first)
    "bert": ["hf-internal-testing/tiny-random-bert", "hf-internal-testing/tiny-random-BertModel"],
    "t5": ["hf-internal-testing/tiny-random-t5", "hf-internal-testing/tiny-random-T5Model"],
    "gpt2": ["sshleifer/tiny-gpt2", "hf-internal-testing/tiny-random-GPT2Model"],
    "roberta": ["hf-internal-testing/tiny-random-RobertaModel"],
    "distilbert": ["hf-internal-testing/tiny-random-DistilBertModel"],
    "albert": ["hf-internal-testing/tiny-random-AlbertModel"],
    "deberta-v2": ["hf-internal-testing/tiny-random-DebertaV2Model"],
    "qwen2": ["yujiepan/qwen2-tiny-random"],
    "llama": ["hf-internal-testing/tiny-random-LlamaForCausalLM"],
    "bart": ["hf-internal-testing/tiny-random-BartForConditionalGeneration"],
    "semantic-bert": ["sentence-transformers/all-MiniLM-L6-v2"],
    "vit": ["hf-internal-testing/tiny-random-ViTModel"],
    "clip": ["hf-internal-testing/tiny-random-CLIPModel"],
    "siglip": ["hf-internal-testing/tiny-random-SiglipModel"],
    "wav2vec2": ["hf-internal-testing/tiny-random-Wav2Vec2Model"],
    "ast": ["hf-internal-testing/tiny-random-ASTModel"],
    "whisper": ["hf-internal-testing/tiny-random-WhisperForConditionalGeneration"],
    "llava_next": ["katuni4ka/tiny-random-llava-next"],
}


def api_check(model_id: str) -> dict:
    try:
        req = urllib.request.Request(f"https://huggingface.co/api/models/{model_id}")
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.load(resp)
        siblings = [s.get("rfilename", "") for s in data.get("siblings") or []]
        return {
            "http": resp.status,
            "exists": True,
            "sha": data.get("sha"),
            "pipeline_tag": data.get("pipeline_tag"),
            "has_tokenizer_json": "tokenizer.json" in siblings,
            "has_processor_config": "processor_config.json" in siblings,
            "has_preprocessor_config": "preprocessor_config.json" in siblings,
            "has_fe_config": "feature_extractor_config.json" in siblings,
        }
    except Exception as exc:  # noqa: BLE001 - probe reports, never raises
        code = getattr(exc, "code", None)
        return {"http": code or type(exc).__name__, "exists": False, "error": str(exc)[:200]}


def featurizer_ladder(model_id: str, is_text_only: bool) -> dict:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer

    result: dict = {}
    if is_text_only:
        try:
            tok = AutoTokenizer.from_pretrained(model_id)
            result["tokenizer"] = f"ok:{type(tok).__name__}"
        except Exception as exc:  # noqa: BLE001
            result["tokenizer"] = f"FAIL:{type(exc).__name__}: {str(exc)[:160]}"
    try:
        proc = AutoProcessor.from_pretrained(model_id)
        result["processor"] = f"ok:{type(proc).__name__}"
    except Exception as exc:  # noqa: BLE001
        result["processor"] = f"FAIL:{type(exc).__name__}: {str(exc)[:160]}"
    if not result.get("processor", "").startswith("ok"):
        try:
            fe = AutoFeatureExtractor.from_pretrained(model_id)
            result["feature_extractor"] = f"ok:{type(fe).__name__}"
        except Exception as exc:  # noqa: BLE001
            result["feature_extractor"] = f"FAIL:{type(exc).__name__}: {str(exc)[:160]}"
    return result


def config_check(model_id: str) -> dict:
    from transformers import AutoConfig

    try:
        cfg = AutoConfig.from_pretrained(model_id)
        return {
            "ok": True,
            "model_type": getattr(cfg, "model_type", None),
            "architectures": getattr(cfg, "architectures", None),
            "is_encoder_decoder": bool(getattr(cfg, "is_encoder_decoder", False)),
            "vision_config": hasattr(cfg, "vision_config"),
            "text_config": hasattr(cfg, "text_config"),
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}


def main() -> int:
    from transformers import AutoConfig

    results: dict = {}
    for family, ids in CANDIDATES.items():
        for model_id in ids:
            entry = {"family": family, "api": api_check(model_id)}
            if entry["api"].get("exists"):
                entry["config"] = config_check(model_id)
                cfg_ok = entry["config"].get("ok")
                if cfg_ok:
                    model_type = entry["config"]["model_type"]
                    # text-only families (no vision/audio in config) use the
                    # AutoTokenizer-first branch of JevX._resolve_featurizers.
                    text_only = model_type in {
                        "bert", "roberta", "distilbert", "albert", "deberta-v2",
                        "gpt2", "llama", "qwen2", "mistral", "phi3", "t5", "bart",
                    } and not entry["config"]["vision_config"]
                    entry["featurizers"] = featurizer_ladder(model_id, text_only)
            results[model_id] = entry
            print(
                f"{model_id}: http={entry['api'].get('http')} "
                f"sha={(entry['api'].get('sha') or '')[:12]} "
                f"tokjson={entry['api'].get('has_tokenizer_json')} "
                f"cfg={entry.get('config', {}).get('model_type')} "
                f"featurizers={entry.get('featurizers')}",
                flush=True,
            )
    libs = {
        name: importlib.util.find_spec(name) is not None
        for name in ("soundfile", "sentencepiece", "librosa", "torchaudio")
    }
    results["_libs"] = libs
    results["_python"] = sys.version
    OUT.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT} | libs={libs}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        raise SystemExit(1)
