# How JevX Uses Real transformers Abstractions

Grounded in [reports/TRANSFORMERS_SOURCE_AUDIT.md](../reports/TRANSFORMERS_SOURCE_AUDIT.md)
(source of truth: `vendor/transformers` @ tag **v5.16.1**, SHA
`93c8b7b485963a10800c91f55304db6be211c2bd`; installed package verified equal).
Every `file:line` below is quoted from that audit and refers to files under
`vendor/transformers/src/transformers/`. Statements are [FACT] (audited/probed)
versus [DESIGN] (JevX choices) as labeled.

## 1. What JevX builds on (public abstractions only)

| transformers abstraction | How JevX uses it | Evidence |
|---|---|---|
| **`PreTrainedModel`** | `JevXForDecision` subclasses it (`jevx/modeling_jevx.py`): sets `config_class = JevXConfig`, `base_model_prefix = "model"`, calls `post_init()` last in `__init__` (audit requirement, `modeling_utils.py:1376`), inherits `save_pretrained`/`from_pretrained` machinery | audit §1.2, §6/Q1 |
| **`PreTrainedConfig`** | `JevXConfig` subclasses it with `model_type = "jevx"` and composition (base config stored as a nested dict, never duplicated); follows the v5 strict-dataclass conventions (`configuration_utils.py:148,359-372`) | audit §1.1; `jevx/configuration_jevx.py` docstring |
| **`ModelOutput`** | `JevXOutput` / `DecisionOutput` are `@dataclass` ModelOutputs: one required field, `None` fields omitted, trailing non-tensor metadata (v5 rule: no `to_dict`, first field must be a tensor) | audit §1.3; `jevx/modeling_outputs.py` docstring |
| **`AutoConfig`** | (a) wrap-path classification in `_peek_config`; (b) session-local `AutoConfig.register("jevx", JevXConfig)` in `jevx/auto/modeling_auto.py` so saved JevX dirs resolve our class without remote code | audit §2 (`configuration_auto.py:439-455`) |
| **`AutoModel` / `AutoModelForCausalLM` / `AutoModelForSeq2SeqLM`** | backbone load chain in `_wrap_base_model` (first success wins) and rebuild chain in `_build_backbone` (prefers the class matching saved `architectures`); session-local `AutoModel.register(JevXConfig, JevXForDecision)` as a documented extra | audit §2/Q3 (`auto_factory.py:261`); `jevx/modeling_jevx.py` |
| **`AutoTokenizer`** | text-only featurizer resolution (families with `text` and no vision/audio have no processor mapping at all) | audit §4/Q8 (`tokenization_auto.py:655`) |
| **`AutoProcessor`** | everything else first (unified `ProcessorMixin.__call__(images, text, videos, audio)`) | audit §4/Q8 (`processing_utils.py:601,656-716`) |
| **`AutoFeatureExtractor`** | last-resort featurizer (AST-like repos ship only feature extractors — AST is absent from `PROCESSOR_MAPPING`, present in `FEATURE_EXTRACTOR_MAPPING` at `auto_mappings.py:1022`) | audit §4/Q8 |
| **`save_pretrained` / `from_pretrained`** | standard checkpoint round trip: config.json (+ nested `backbone_config`), `model.safetensors` with tied-weight removal handled by transformers (`remove_tied_weights_from_state_dict`, `modeling_utils.py:437`), `architectures` auto-written (`:3418`), meta-device construction + `local_torch_dtype` on load (`:3791-3820`) | audit §3/Q1/Q5 |
| **safetensors** | the only weight format written; declared child ties (e.g. `lm_head.weight`) are removed pre-write and restored by transformers' `tie_weights` on load — probe-verified with a tied `LlamaForCausalLM` wrapper | audit §3, Q1 probe |
| **session-local `AutoConfig.register`** | performed at `import jevx.auto`; idempotent (`try/except ValueError`); **process-local only** | audit Q3 option 2; `jevx/auto/modeling_auto.py` |
| **tuple `return_dict` mode** | `forward(..., return_dict=False)` returns `output.to_tuple()`; `None` → `config.return_dict`. Same semantics transformers v5 enforces via `can_return_tuple`/`capture_outputs` (`utils/generic.py:924-944`, `utils/output_capturing.py:233`) | audit Q2; `tests/unit/test_outputs_jevx.py::test_v5_tuple_mode_via_can_return_tuple` |
| **hook-based `output_hidden_states`** | JevX never collects hidden states itself — it passes `output_hidden_states=True` (only when requested) and lets v5's `_can_record_outputs` forward-hook capture do it (lazy install, no recomputation, `utils/output_capturing.py:196-213`); extraction priority `hidden_states[-1]` is step 6 and **opt-in only** | audit Q4; `jevx/adapters/base.py` |
| **`dtype=` canonical** | backbone-loader kwarg list accepts `dtype` (canonical, `modeling_utils.py:4130`) with `torch_dtype` BC fallback — if both are passed, `torch_dtype` is dropped so exactly one spelling reaches the loader; wrapper parameters follow the dtype context automatically | audit Q5; `jevx/modeling_jevx.py::_BACKBONE_LOAD_KWARGS` |
| **`PreTrainedModel.can_generate()`** | capability detection for `generation` (v5 configs have no `can_generate`; the config-level callable is only tried first when present) — `modeling_utils.py:1561` | audit §1.2; `jevx/capabilities/detector.py` |
| **embedding accessors** | `get_input_embeddings`/`set_input_embeddings` are **overridden** to delegate to the backbone, because the default name heuristics raise `NotImplementedError` for most wrapper children (probe: even a `model`-named BERT child fails) | audit risk #2 / §1.2 probe |

## 2. Session-local registration and its split-brain limitation [DESIGN]

`jevx/auto/modeling_auto.py` runs, at import time:

```python
try: AutoConfig.register("jevx", JevXConfig)       # except ValueError: pass (idempotent)
try: AutoModel.register(JevXConfig, JevXForDecision)  # except Exception: pass
```

**Split-brain limitation (documented, deliberate):** registration is
**process-local**. Inside a process that imported `jevx.auto`, plain
`AutoConfig.from_pretrained(jevx_dir)` resolves `JevXConfig`; in a *fresh*
process that has not imported it, `AutoConfig` raises `ValueError: Unrecognized
model ... 'jevx'`. JevX therefore **never requires** Auto registration to load a
saved checkpoint: `JevXForDecision.from_pretrained` peeks the config directly and
takes the reload path from the raw `model_type == "jevx"` (`_peek_config`
fallback reads the raw dict via `PreTrainedConfig.get_config_dict`), so explicit
loading works without any registration. The persistent/Hub route
(`auto_map` + `trust_remote_code`) was deliberately **not** used: JevX ships as
a library, not as remote code, and `trust_remote_code` never defaults to `True`
(audit §2: on Windows the SIGALRM prompt does not exist —
`dynamic_module_utils.py:774-779`).

The `AutoModel.register` extra is best-effort (`except Exception` swallowed) so a
mapping-constraint failure can never mask the core exports.

## 3. What JevX does NOT fork or monkey-patch [DESIGN, verified]

- **No vendored edits**: `vendor/transformers` is a read-only clone; no file
  under it is modified (the clone is used for audit citations only).
- **No monkey-patching**: `jevx/**` never patches, replaces or wraps any
  transformers function/class at runtime (no assignment into `transformers.*`
  namespaces anywhere in the package; `monkey_patching.py` is never imported).
- **No subclasses of Auto internals, no custom `from_pretrained` load path** —
  the wrap path *calls* `AutoModel.from_pretrained` and the reload path *calls*
  `PreTrainedModel.from_pretrained` unmodified.
- **No `auto_map`/`register_for_auto_class`**: saved configs do not ship remote
  code hooks; nothing lands in `sys.modules` from a checkpoint.
- **No generation machinery**: JevX runs no decoding; `GenerationMixin` is not
  involved (audit §5: v5 no longer inherits it from `PreTrainedModel`).
- **No custom attention/masking code**: masks come from each backbone's own
  forward; JevX only validates `attention_mask` values at its input boundary.
- **No pooling inside transformers** — v5 ships none (audit Q7), so JevX
  implements the operators once in `jevx/representations/pooling.py` and routes
  every adapter through them ([DESIGN], prompted by the audit).

## 4. Extension points JevX uses (all public)

| Extension point | Where used |
|---|---|
| `AutoConfig.register` / `AutoModel.register` | session-local, `jevx/auto` (documented extra) |
| `PreTrainedConfig` dataclass subclassing | `JevXConfig` (`__init_subclass__` kw-only wrapper) |
| `PreTrainedModel` subclass + `post_init()` aggregation | `JevXForDecision`; child ties aggregated with prefix so safetensors tie-removal covers the embedded backbone (audit §6/Q1 probe) |
| `ModelOutput` dataclass fields | `JevXOutput`, `DecisionOutput` |
| `_can_record_outputs` hook capture | opted into via `output_hidden_states` passthrough (audit Q4) |
| `ProcessorMixin`/`Auto*` featurizer resolution ladders | `_resolve_featurizers` implements exactly the audited Q8 ladder |
| `config.to_diff_dict()` | nested `backbone_config` serialization (`configuration_utils.py:1011`, nested `model_type` preserved at `:1043-1045`) |

## 5. v5-specific gotchas the integration honors (audit §8/§9)

1. Undeclared shared tensors raise `RuntimeError` at save (`modeling_utils.py:508-511`)
   — JevX declares no aliasing of its own; child ties flow through `post_init`
   aggregation.
2. `get_input_embeddings`/`set_input_embeddings` must be overridden (risk #2) — done.
3. Auto classes refuse unregistered `model_type`s — handled by direct config peek
   + session registration (§2 above).
4. Windows `trust_remote_code` — explicit opt-in only, default `False`.
5. `torch_dtype` → `dtype` migration — `dtype` canonical in the loader kwargs;
   config.json stores `dtype`.
6. `output_attentions=True` requires eager attention — JevX passes the flag
   through and documents the constraint (it never needs attentions itself).
7. Decoder-only `use_cache` defaults `True` — extraction passes `use_cache=False`
   to skip `DynamicCache` allocation (`jevx/adapters/base.py`,
   audit §8.7 / `modeling_llama.py:383-384`).
8. `hidden_states[-1]` tie differs by family (post-norm for LM-likes, pre-norm
   for vision/audio with `tie_last_hidden_states=False`) — JevX treats
   `hidden_states[-1]` as an opt-in fallback only and prefers named fields.
9. `PreTrainedConfig` is a strict kw-only dataclass — `JevXConfig` declares no
   custom `__init__`, so the BC kwargs wrapper applies.
10. AST has no `AutoProcessor` entry — the featurizer ladder falls through to
    `AutoFeatureExtractor` (executed green on the AST row).
11. `torch.jit.script` FutureWarning noise under Python 3.14 (torch 2.14) is
    third-party, observed in the suite's warnings summary, harmless.

## 6. Where each claim is enforced

| Claim | Test |
|---|---|
| Auto registration resolves JevX config/model | `tests/unit/test_auto_factory.py::test_auto_config_registration_resolves_jevx_config`, `::test_auto_model_registration_resolves_wrapper`, `::test_auto_jevx_from_saved_dir` |
| Wrap path through real Auto classes | `::test_auto_jevx_wraps_real_base_checkpoint` |
| `trust_remote_code` defaults False | `::test_trust_remote_code_defaults_false` |
| tuple `return_dict` mode | `tests/unit/test_outputs_jevx.py::test_v5_tuple_mode_via_can_return_tuple` |
| `base_forward` bit-identical to raw checkpoint | `tests/unit/test_modeling_jevx.py::test_base_forward_bitwise_equal_to_raw_backbone` + per-checkpoint `base_forward_max_abs_error` in `tests/integration/matrix_common.py` |
| safetensors tie round trip | `tests/unit/test_serialization_jevx.py::test_tied_causal_lm_roundtrip` |
| reload without Auto registration | `tests/unit/test_serialization_jevx.py::test_reload_with_explicit_config` |
| hook-based hidden states not requested unless asked | extraction priority step 6 in `jevx/adapters/base.py`; capability `hidden_states` flag tests in `tests/unit/test_capabilities.py` |
