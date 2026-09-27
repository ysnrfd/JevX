# Copyright (c) 2026 ysnrfd
"""Diagnose the wav2vec2 reload meta-tensor failure (diff against bert)."""

from __future__ import annotations

import contextlib
import shutil
import traceback
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from jevx import JevX, JevXForDecision

CASES = {
    "wav2vec2": "hf-internal-testing/tiny-random-Wav2Vec2Model",
    "bert": "hf-internal-testing/tiny-random-bert",
}


def diagnose(name: str, model_id: str) -> None:
    print(f"\n===== {name} =====")
    wrapper = JevX.from_pretrained(model_id)
    wrapper.eval()
    bc = wrapper.config.backbone_config
    print("backbone_config type:", type(bc).__name__)
    if isinstance(bc, dict):
        print("backbone_config model_type:", bc.get("model_type"),
              "| architectures:", bc.get("architectures"))
        print("backbone_config keys sample:", sorted(bc.keys())[:12])
    tmp = Path(tempfile.mkdtemp(prefix=f"jevx-diag-{name}-"))
    try:
        wrapper.save_pretrained(tmp)
        import json

        saved = json.loads((tmp / "config.json").read_text(encoding="utf-8"))
        sbc = saved.get("backbone_config")
        print("SAVED backbone_config model_type:", None if not isinstance(sbc, dict) else sbc.get("model_type"))
        cfg = JevXForDecision._peek_config(tmp, False, {})[0]
        print("peeked model_type:", cfg.model_type, "| dtype:", getattr(cfg, "dtype", None))

        # Replicate _build_backbone under the from_pretrained init contexts.
        from transformers.modeling_utils import local_torch_dtype

        with contextlib.ExitStack() as stack:
            stack.enter_context(local_torch_dtype(torch.float32, "JevXForDecision"))
            stack.enter_context(torch.device("meta"))
            backbone = JevXForDecision._build_backbone(cfg)
        first = next((p for p in backbone.parameters() if p.is_floating_point()), None)
        print("rebuilt backbone first float param:",
              None if first is None else (first.device, first.dtype, tuple(first.shape)))

        with contextlib.ExitStack() as stack:
            stack.enter_context(local_torch_dtype(torch.float32, "JevXForDecision"))
            stack.enter_context(torch.device("meta"))
            model = JevXForDecision(cfg)
        sp = next(model.state_proj.parameters())
        ref = next(p for p in model.model.parameters() if p.is_floating_point())
        print("state_proj param:", sp.device, sp.dtype)
        print("backbone first float:", ref.device, ref.dtype)
        print("construction OK")
        del model
    except Exception as exc:  # noqa: BLE001
        print(f"DIAG FAILURE: {type(exc).__name__}: {exc}")
        print(traceback.format_exc())
        traceback.print_exc()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    for case_name, case_id in CASES.items():
        diagnose(case_name, case_id)
