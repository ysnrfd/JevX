# Copyright (c) 2026 ysnrfd
"""DIAGNOSTIC (scratch, WAVE-4): why did the roberta-base Phase-92 gate fail?

benchmarks/run_level4.py (another session) recorded for FacebookAI/roberta-base:
    AssertionError: base_forward diverged from the raw checkpoint by 1.077e+00
and for the "native" (pooler_output) quality cell featurize-vs-full-model
equivalence diffs of 5.8 / 8.7 / 23.6, where float re-association is ~1e-6.

Hypothesis under test: FacebookAI/roberta-base ships NO pooler weights, so
RobertaModel.add_pooling_layer=True random-initialises pooler.dense on EVERY
load. Two separately loaded models then disagree on pooler_output (hence both
symptoms) while last_hidden_state is bit-identical.

Writes nothing into vendor/ and nothing into jevx/.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

spec = importlib.util.spec_from_file_location(
    "jevx_matrix_common", REPO_ROOT / "tests" / "integration" / "matrix_common.py")
mc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mc)

BID = sys.argv[1] if len(sys.argv) > 1 else "FacebookAI/roberta-base"
POOL = sys.argv[2] if len(sys.argv) > 2 else "cls"

out: dict = {"backbone": BID, "pooling": POOL}
wrapper, load_s = mc.load_wrapper(BID, freeze_base_model=True,
                                  pooling_method=POOL, fusion_type="concat")
raw = AutoModel.from_pretrained(BID)
raw.eval()
out["load_s"] = round(load_s, 2)
out["wrapper_model_class"] = type(wrapper.model).__name__
out["raw_model_class"] = type(raw).__name__
out["wrapper_training"] = wrapper.model.training
out["raw_dtype"] = str(next(raw.parameters()).dtype)
out["n_params"] = int(sum(p.numel() for p in wrapper.model.parameters()))
out["config_add_pooling_layer"] = getattr(wrapper.model.config, "add_pooling_layer", None)

sd_w = wrapper.model.state_dict()
sd_r = raw.state_dict()
out["param_key_symmetric_diff"] = sorted(set(sd_w) ^ set(sd_r))[:20]
pdiff = {}
for k in sorted(set(sd_w) & set(sd_r)):
    a, b = sd_w[k].float(), sd_r[k].float()
    if a.shape != b.shape:
        pdiff[k] = "SHAPE " + str(tuple(a.shape)) + " vs " + str(tuple(b.shape))
        continue
    d = float(np.abs(a - b).max())
    if d > 0:
        pdiff[k] = d
out["param_max_abs_diffs"] = pdiff
out["params_bitwise_identical"] = len(pdiff) == 0

state = mc.build_state("text")
inputs = mc.prepared_inputs(wrapper, state)
out["prepared_input_keys"] = sorted(inputs)
out["prepared_input_meta"] = {k: [list(v.shape), str(v.dtype)]
                              for k, v in inputs.items() if torch.is_tensor(v)}
out["tokenizer"] = type(wrapper.tokenizer).__name__
out["pad_side"] = getattr(wrapper.tokenizer, "padding_side", None)
out["decoded"] = wrapper.tokenizer.decode(inputs["input_ids"][0])

with torch.no_grad():
    wb = wrapper.base_forward(**inputs)
    rb = raw(**inputs)
if torch.is_tensor(wb):
    wb = {"last_hidden_state": wb}
out["wrapped_output_fields"] = {k: list(v.shape) for k, v in wb.items() if torch.is_tensor(v)}
out["raw_output_fields"] = {k: list(v.shape) for k, v in rb.items() if torch.is_tensor(v)}
fe = {}
for k, v in wb.items():
    other = getattr(rb, k, None)
    if torch.is_tensor(v) and other is not None:
        fe[k] = float(np.abs(v.float() - other.float()).max())
out["field_errors"] = fe
out["overall_base_error"] = mc.max_abs_error(wb, rb)

rep = mc.extract_via_adapter(wrapper, inputs)
out["adapter_field"] = rep.field
out["adapter_pooled_shape"] = list(rep.pooled.shape)
out["adapter_pooled_absmax"] = float(np.abs(rep.pooled).max())
try:
    man = mc.manual_expected_representation(wrapper, inputs)
    out["manual_absmax"] = float(np.abs(man).max())
    out["extract_error"] = float(np.abs(rep.pooled.float() - man.float()).max())
except Exception as exc:  # noqa: BLE001
    out["extract_error"] = type(exc).__name__ + ": " + str(exc)

from benchmarks import bench_common as bc  # noqa: E402

texts = mc.REAL_TEXTS
feats = bc.featurize(wrapper, texts, batch_size=2, max_length=128,
                     strategies=(None, "cls"))
out["featurize_keys"] = sorted(feats)
per = {}
for key in sorted(feats):
    for strat in sorted(feats):
        sub = bc.featurize(wrapper, texts, batch_size=2, max_length=128,
                           strategies=(strat,))[strat]
        per[key + " vs " + strat] = {
            "absmax": round(float(np.abs(sub).max()), 5),
            "l2": round(float(np.linalg.norm(sub)), 5),
            "max_abs_diff": round(float(np.abs(sub - feats[key]).max()), 5),
        }
out["featurize_vs_default"] = per
man2 = mc.manual_expected_representation(wrapper, inputs).detach().cpu().numpy()
out["cls_feats_vs_manual_max_abs"] = round(float(np.abs(feats["cls"] - man2).max()), 6)

print("DIAG_JSON_BEGIN")
print(json.dumps(out, indent=2, default=str))
print("DIAG_JSON_END")
