# Copyright (c) 2026 ysnrfd
"""One-shot: add PreTrainedConfig import + make diag print traceback to stdout."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

p = ROOT / "jevx" / "modeling_jevx.py"
src = p.read_text(encoding="utf-8")
if "    PreTrainedConfig,\n" not in src:
    old = "    AutoTokenizer,\n    PreTrainedModel,\n)"
    new = "    AutoTokenizer,\n    PreTrainedConfig,\n    PreTrainedModel,\n)"
    assert old in src, "import block pattern not found"
    src = src.replace(old, new, 1)
    p.write_text(src, encoding="utf-8")
    print("import fixed")
else:
    print("import already present")

d = ROOT / "vendor" / "reports" / "_debug_reload_meta.py"
t = d.read_text(encoding="utf-8")
if "traceback.format_exc()" not in t:
    old = 'print(f"DIAG FAILURE: {type(exc).__name__}: {exc}")'
    new = 'print(f"DIAG FAILURE: {type(exc).__name__}: {exc}")\n        print(traceback.format_exc())'
    assert old in t, "diag except pattern not found"
    t = t.replace(old, new, 1)
if "import traceback" not in t:
    t = t.replace("import contextlib", "import contextlib\nimport traceback", 1)
d.write_text(t, encoding="utf-8")
print("diag updated")
