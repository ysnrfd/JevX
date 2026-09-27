# Copyright (c) 2026 ysnrfd
"""Extend the reload meta-guard to buffers (confidence engine buffers have
no parameters, so a params-only check let module.to() hit a meta buffer)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "jevx" / "modeling_jevx.py"

OLD = '''        if reference is not None and not reference.is_meta:
            for module in ('''

NEW = '''        def _has_meta_tensors(module: nn.Module) -> bool:
            return any(p.is_meta for p in module.parameters()) or any(
                buffer is not None and buffer.is_meta for buffer in module.buffers()
            )

        if reference is not None and not reference.is_meta and not _has_meta_tensors(self.model):
            for module in ('''

OLD_TAIL = '''                if any(p.is_meta for p in module.parameters()):
                    continue  # reload path: transformers materializes later
                module.to(device=reference.device, dtype=reference.dtype)'''

NEW_TAIL = '''                if _has_meta_tensors(module):
                    continue  # reload path: transformers materializes later
                module.to(device=reference.device, dtype=reference.dtype)'''


def main() -> int:
    src = PATH.read_text(encoding="utf-8")
    applied = 0
    for old, new in ((OLD, NEW), (OLD_TAIL, NEW_TAIL)):
        if old in src:
            src = src.replace(old, new, 1)
            applied += 1
    if applied != 2:
        print(f"expected 2 replacements, applied {applied}")
        return 1
    PATH.write_text(src, encoding="utf-8")
    print("meta-guard extended to buffers (+ backbone meta check)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
