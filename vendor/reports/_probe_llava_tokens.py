# Copyright (c) 2026 ysnrfd
"""Find processor settings whose image-token count matches model features."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from jevx import JevX

MODEL_ID = "katuni4ka/tiny-random-llava-next"
PROMPT = "<image> Describe this image briefly."


def make_image():
    sys.path.insert(0, str(ROOT / "tests" / "integration"))
    import matrix_common

    return matrix_common.fixture_image()


def main() -> int:
    torch.set_grad_enabled(False)
    wrapper = JevX.from_pretrained(MODEL_ID)
    wrapper.eval()
    proc = wrapper.processor
    model = wrapper.model
    cfg = model.config
    image = make_image()
    img_tok = int(getattr(proc, "image_token_id", 0) or 0)
    print("model vision_feature_select_strategy:", cfg.vision_feature_select_strategy)
    print("processor initial: patch_size=", proc.patch_size,
          "| strategy=", proc.vision_feature_select_strategy,
          "| additional=", proc.num_additional_image_tokens,
          "| image_token_id=", img_tok)

    for strategy, additional in itertools.product(
        (None, "default", "full"), (0, 1)
    ):
        proc.vision_feature_select_strategy = strategy
        proc.num_additional_image_tokens = additional
        if proc.patch_size is None:
            proc.patch_size = int(cfg.vision_config.patch_size)
        try:
            batch = proc(text=PROMPT, images=image, return_tensors="pt")
            tokens = int((batch["input_ids"] == img_tok).sum())
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                pixel_values=batch["pixel_values"],
                image_sizes=batch.get("image_sizes"),
                use_cache=False,
            )
            seq = int(out.last_hidden_state.shape[1])
            print(f"strategy={strategy!r} additional={additional}: "
                  f"tokens={tokens} seq_len={seq} -> FORWARD OK")
        except Exception as exc:  # noqa: BLE001 - probing combinations
            msg = str(exc).replace("\n", " ")[:150]
            print(f"strategy={strategy!r} additional={additional}: {type(exc).__name__}: {msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
