# Copyright (c) 2026 ysnrfd
"""Apply the two real-model-matrix bug fixes to jevx/modeling_jevx.py.

1. _hidden_width: composite configs (clip/siglip/llava_next) resolve width
   via projection_dim or nested text_config/vision_config;
2. __init__ alignment loop: never .to() meta tensors (wav2vec2 escapes the
   meta construction context, leaving reference=CPU + modules=meta).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "jevx" / "modeling_jevx.py"

OLD1 = '''def _hidden_width(config: Any) -> Optional[int]:
    """First positive int among :data:`_HIDDEN_WIDTH_KEYS` on a config/dict."""
    getter = config.get if isinstance(config, dict) else lambda k: getattr(config, k, None)
    for key in _HIDDEN_WIDTH_KEYS:
        value = getter(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None'''

NEW1 = '''def _hidden_width(config: Any) -> Optional[int]:
    """Width of the pooled state vector for a config/dict.

    Order (first positive int wins):

    1. :data:`_HIDDEN_WIDTH_KEYS` on the config itself (plain families);
    2. ``projection_dim`` — composite contrastive configs (clip) where the
       adapter extracts the POST-projection semantic field
       (``text_embeds``, adapters/base SEMANTIC_FIELDS), so the pooled
       width is the projection width, not the encoder hidden size;
    3. nested ``text_config`` / ``vision_config`` (llava_next-style
       composites whose pooled vector comes from interleaved
       ``last_hidden_state`` tokens of the language tower).

    Config and model are always built from the same config object, so the
    picked width agrees with what extraction actually returns (probe-
    verified for clip/siglip/llava_next in the real-model matrix).
    """
    getter = config.get if isinstance(config, dict) else lambda k: getattr(config, k, None)
    for key in _HIDDEN_WIDTH_KEYS:
        value = getter(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    projection = getter("projection_dim")
    if isinstance(projection, int) and not isinstance(projection, bool) and projection > 0:
        return projection
    for nested_key in ("text_config", "vision_config"):
        nested = getter(nested_key)
        if isinstance(nested, (dict, PreTrainedConfig)):
            width = _hidden_width(nested)
            if width is not None:
                return width
    return None'''

OLD2 = '''        reference = next(
            (p for p in self.model.parameters() if p.is_floating_point()), None
        )
        if reference is not None:
            for module in ('''

NEW2 = '''        reference = next(
            (p for p in self.model.parameters() if p.is_floating_point()), None
        )
        # The loop below aligns wrapper params with a PRE-LOADED materialized
        # backbone (wrap path). On the reload path transformers constructs the
        # whole tree under the meta device and materializes afterwards; torch
        # refuses to move meta tensors (NotImplementedError: "Cannot copy out
        # of meta tensor") and some backbones (wav2vec2) even escape the meta
        # context so reference lands on CPU while these modules are still
        # meta. Skip alignment whenever either side is meta — checkpoint
        # weights for these modules are loaded (or re-initialized for truly
        # missing keys) by transformers' own finalization, and the calibrator
        # is re-materialized by _materialize_calibrator().
        if reference is not None and not reference.is_meta:
            for module in ('''

OLD3 = '''                module.to(device=reference.device, dtype=reference.dtype)'''

NEW3 = '''                if any(p.is_meta for p in module.parameters()):
                    continue  # reload path: transformers materializes later
                module.to(device=reference.device, dtype=reference.dtype)'''

OLD4 = '''                self.calibrator,
            ):
                module.to(device=reference.device, dtype=reference.dtype)'''

NEW4 = '''                self.calibrator,
            ):
                if any(p.is_meta for p in module.parameters()):
                    continue  # reload path: transformers materializes later
                module.to(device=reference.device, dtype=reference.dtype)'''


def main() -> int:
    src = PATH.read_text(encoding="utf-8")
    checks = [
        ("_hidden_width", OLD1, NEW1, True),
        ("alignment head", OLD2, NEW2, True),
        ("alignment tail", OLD4, NEW4, False),
    ]
    for label, old, new, required in checks:
        if old in src:
            src = src.replace(old, new, 1)
            print(f"applied: {label}")
        elif required:
            print(f"REQUIRED pattern missing: {label}")
            return 1
        else:
            print(f"optional pattern missing (checking variant): {label}")
    if "if any(p.is_meta for p in module.parameters()):" not in src:
        # fallback: apply the generic tail replacement
        if OLD3 in src:
            src = src.replace(OLD3, NEW3, 1)
            print("applied: alignment tail (generic)")
        else:
            print("alignment tail not applied")
            return 1
    PATH.write_text(src, encoding="utf-8")
    print(f"wrote {PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
