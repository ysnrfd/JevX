# Universal Conversion (§63): How Any HF Checkpoint Becomes a Decision Model

This documents the conversion factory **as implemented** in
`jevx/modeling_jevx.py::_wrap_base_model` and `jevx/adapters/registry.py`, the
manual adapter API, capability detection, the no-silent-fallback rules, and the
support-level system. The **live source** for per-family support levels is
[TRANSFORMERS_COMPATIBILITY_MATRIX.md](../TRANSFORMERS_COMPATIBILITY_MATRIX.md)
(this document summarizes it; where they differ, the matrix wins).

## 1. The factory flow as implemented

`JevX.from_pretrained(model_id, ...)` (alias of
`JevXForDecision.from_pretrained`) branches on the config peek, then runs the
wrap path for any base checkpoint:

```text
JevX.from_pretrained(id)
│
├─ _peek_config: AutoConfig.from_pretrained(id, trust_remote_code=False, hub-peek kwargs)
│    model_type == "jevx" ──► RELOAD path: PreTrainedModel.from_pretrained
│    AutoConfig refuses ──► read raw dict (PreTrainedConfig.get_config_dict);
│                            JEVX-reload ONLY if raw model_type == "jevx",
│                            else re-raise AutoConfig's error
│
└─ _wrap_base_model (base checkpoint):
     1. backbone load chain: AutoModel → AutoModelForCausalLM → AutoModelForSeq2SeqLM
        (first success wins; every failure recorded; all fail → ValueError listing attempts)
     2. capability detection: JevXCapabilityDetector.detect(backbone, config)  [introspection only]
     3. featurizer resolution (when tokenizer= and processor= were not passed):
        text-only family → AutoTokenizer; else/failure → AutoProcessor;
        else/failure → AutoFeatureExtractor; processor-only → expose its embedded
        tokenizer as `tokenizer` (wrap/reload symmetry — the measured save/load
        drift fix, reports/REAL_MODEL_VALIDATION.md bug 4);
        missing pad_token → reuse EOS, then UNK (GPT-2-style tokenizers)
     4. kwarg partition: JevXConfig fields (any dataclass field, e.g.
        pooling_method=, fusion_type=, decision_hidden_size=) | backbone loader
        kwargs (dtype/torch_dtype/revision/...) | factory kwargs
        (decision_type/trust_remote_code/tokenizer/processor/adapter);
        anything else → TypeError naming all three groups
     5. hidden-width resolution: hidden_size → d_model → n_embd → ... →
        projection_dim → nested text_config/vision_config; none → ValueError
        (pass decision_hidden_size=)
     6. JevXConfig built with base_model_name_or_path, base_model_type,
        backbone_config (= to_diffdict + architectures pinned to the EMBEDDED class),
        decision/question/schema widths = backbone width (unless overridden)
     7. JevXForDecision(config, backbone=backbone, ...):
        capability detection again, adapter resolution (§2), projections /
        encoders / fusion / heads built, wrapper params aligned to backbone
        device/dtype, post_init(), freeze_base_model applied, model.eval()
```

Documented fallbacks inside this flow are **enumerated ladders with recorded
errors** (Auto-class chain, featurizer chain, config-peek fallback) — the wrap
either produces a working wrapper or raises with the full attempt list. There is
no path that silently substitutes a different model, pooling, or decision type.

The complementary **manual construction** path exists too:
`JevXForDecision(JevXConfig(...), backbone=<already-loaded model>)` — used by the
unit tests with in-memory fixtures, and it embeds the base config into
`backbone_config` itself so save/load stays self-contained.

## 2. Adapter resolution chain

From `jevx/adapters/registry.py` — order matters, most specific first; every
`can_handle` is **capability introspection only** (no forward pass, no
model-name matching anywhere):

1. **User-registered adapters** (`USER_ADAPTERS`, in registration order) — the
   first whose `can_handle(model)` is True wins.
2. **Built-in chain** (`BUILTIN_ADAPTER_CHAIN`):
   `MultimodalAdapter` → `EncoderDecoderAdapter` → `VisionAdapter` →
   `AudioAdapter` → `DecoderAdapter` → `EncoderAdapter` → `GenericAdapter`.
3. **`GenericAdapter`** — `can_handle` is always `True`, so resolution never
   fails (the defensive `RuntimeError` path is unreachable unless a user chain is
   broken).

Family predicates and defaults:

| Adapter | `can_handle` (introspection) | `default_pooling` |
|---|---|---|
| `multimodal` | vision **and** text (CLIP/LLaVA composites) | `masked_mean` |
| `encoder_decoder` | `config.is_encoder_decoder` (T5/BART/Whisper) | `encoder_pool` (= masked_mean) |
| `vision` | vision, no text | `masked_mean` |
| `audio` | audio, no text | `masked_mean` |
| `decoder` | cache-capable causal stack, not enc-dec, not vision/audio | `last_valid` |
| `encoder` | encoder direction, none of the above | `cls` |
| `generic` | always | `masked_mean` |

Family coverage per the matrix (levels: see §5): encoders
(bert/roberta/albert/distilbert/deberta-v2), decoders (llama/qwen2/mistral/gemma/
gpt2/phi3), enc-dec (t5/bart/whisper), audio (wav2vec2/ast), vision (vit),
contrastive (clip/siglip), VLM (llava_next) — all resolve to a specific adapter
by capability, never by name. `available_adapters()` lists names in resolution
order.

## 3. Manual adapter API

```python
from jevx import register_adapter
from jevx.adapters import BaseAdapter, resolve_adapter, available_adapters

@register_adapter                     # class decorator (highest priority)
class MyAdapter(BaseAdapter):
    name = "my.adapter"               # registry key; duplicate name → ValueError
    default_pooling = "masked_mean"    # used when strategy=None and no native field

    def can_handle(self, model) -> bool: ...           # introspection only
    def prepare_inputs(self, state, *, tokenizer=None, processor=None, **kw) -> dict: ...
```

`BaseAdapter` contract (abstract `can_handle` + `prepare_inputs`; inherited
`extract_representation`, `describe_capabilities`, `validate`):

- `extract_representation(model, inputs, *, strategy=None,
  output_hidden_states=False, mode=None) -> Representation` — field priority
  `pooler_output` → semantic fields (`sentence_embedding`, `text_embeds`,
  `image_embeds`) → `last_hidden_state` → `encoder_last_hidden_state` →
  decoder-state tuple (enc-dec `mode`) → `hidden_states[-1]` (opt-in only);
  `strategy="native"` *requires* a native field (else
  `UnsupportedRepresentationError` naming families without poolers).
- `validate(model, sample_state)` — the single permitted `no_grad` dry run;
  converts any failure into a structured `UnsupportedRepresentationError`.
- `describe_capabilities(model)` — never runs a forward.

Registration is idempotent per class; duplicate *names* across different classes
raise `ValueError` at decoration time. Override at the factory with
`JevX.from_pretrained(id, adapter=MyAdapter)` (instance or class), or per-model
via `model.register_adapter(...)`. Resolution priority is asserted in
`tests/unit/test_adapters.py::test_user_registered_adapter_wins` and
`tests/unit/test_auto_factory.py::test_register_adapter_priority_and_restore`.

## 4. Capability detection

`JevXCapabilityDetector.detect` (jevx/capabilities/detector.py) is **pure
introspection**: forward-signature modality flags (`input_ids`/`pixel_values`/
`input_values`/`input_features`), output-class resolution from the return
annotation (unions unwrapped, PEP-563 strings resolved), decoder detection from
output fields + `is_decoder` + submodules (never name-based), pooler scan,
`can_generate()` via the model classmethod, native/fallback representation route,
`processor_required`, and the `jevx_conversion` verdict:

| `jevx_conversion` | Meaning (detector docstring) |
|---|---|
| `supported` | a standard extraction route is introspectable (native field, or standard field + established pooling) |
| `experimental` | route exists but semantics need a pinned convention (interleaved multimodal pooling — matrix level-2 rows such as llava_next) |
| `unsupported` | no representation field identifiable (or neither config nor `PreTrainedModel`) |

`probe(model, dummy_inputs)` optionally confirms with exactly one cautious
forward; failures land in `reasons`, never raised. The report is serialized into
the checkpoint's `jevx_metadata.json`.

## 5. No-silent-fallback behavior

Facts enforced by code and asserted by tests:

| Situation | Behavior (never silent) | Test |
|---|---|---|
| All three Auto classes fail to load the backbone | `ValueError` listing every attempt (`AutoModel: ... : ...`) | factory chain in `_wrap_base_model` |
| Config without `model_type` (e.g. `prajjwal1/bert-tiny`) | re-raises **AutoConfig's** actionable error — no misclassification as a JEVX checkpoint | `tests/unit/test_auto_factory.py::test_peek_config_rejects_config_without_model_type`, `::test_peek_config_fallback_requires_raw_jevx_model_type` |
| `trust_remote_code` | defaults `False`; never implied; remote-code failure does not execute the module (no `sys.modules` entry, no artifacts) | `::test_trust_remote_code_defaults_false`; `tests/unit/test_failure_modes.py` row 14 |
| Unknown kwarg to `from_pretrained` | `TypeError` naming config/loader/factory kwarg groups | `jevx/modeling_jevx.py::_wrap_base_model` |
| Hidden width undeterminable | `ValueError` naming probed keys + remedy | same |
| Unknown decision type | `ValueError` listing `decision_primitives` + registered heads | `tests/unit/test_modeling_jevx.py::test_unknown_decision_type_lists_registered_heads` |
| State longer than a learned position table | `ValueError` naming lengths — "JevX never silently truncates a state" | `tests/unit/test_numerical_edge.py` (rows 6), `tests/unit/test_failure_modes.py` |
| Invalid `attention_mask` values (float/negative) | `ValueError` with binarize remedy | `tests/unit/test_failure_modes.py` rows 8a/8b |
| No extractable representation | `UnsupportedRepresentationError` with detected/attempted/failed/suggestion | `tests/unit/test_failure_modes.py` row 12; `tests/unit/test_capabilities.py` |
| `strategy="native"` without a native field | `UnsupportedRepresentationError` suggesting explicit strategies | `tests/unit/test_pooling.py::test_native_requires_pooler_output` |
| Pooling over a fully-masked row | `ValueError` (axiom D1.1) — never NaN | `tests/unit/test_pooling.py::test_all_masked_row_raises` |
| Calibration sidecar method ≠ config method | `ValueError` on reload | `tests/unit/test_serialization_jevx.py::test_calibration_method_mismatch_raises` |
| Featurizer absent for a vision/audio state | `ValueError` naming `processor=` | `tests/unit/test_failure_modes.py` row 13 |
| Network/HTTP failure in the test harness | documented `pytest.skip` with attempts — **never** converted into a pass; implementation bugs re-raise and fail | `tests/integration/test_jevx_e2e_tiny.py`, golden env-marker skips |

## 6. Support-level system (current matrix summary)

Level definitions are the matrix's
([TRANSFORMERS_COMPATIBILITY_MATRIX.md](../TRANSFORMERS_COMPATIBILITY_MATRIX.md),
"Support levels" table): **4** native pooled/verified route incl. HF-API-verified
checkpoint **plus**, per the Phase 92-94 evidence overlay, a real-dataset quality
result; **3** native states + established standard pooling, executed
(load → forward → invariants → save/load equality → real input) on the cited
checkpoint; **2** convention-dependent (lossy pooling) — mechanics standard, but
the pooled semantics require a pinned convention; **1** source-verified only;
**0** no standard extraction path (no audited family landed here).

Current distribution (measured executions in
reports/REAL_MODEL_VALIDATION.md; machine-readable
`vendor/reports/_jevx_matrix_results.json`):

- **Level 3 (13):** bert, roberta, albert, distilbert, deberta-v2, t5, bart,
  wav2vec2, whisper, ast, vit, clip, siglip — all 13 executed with every
  numerical gate at 0.0 (except llava-next's documented 5.66e-7 base_forward,
  which is level 2 anyway).
- **Level 2 (7):** llama, qwen2, gpt2 (executed; lossy-pooling semantics
  unchanged by execution), mistral, gemma, phi3 (Phase-1 probe evidence only —
  not re-executed in Phase 92, noted in the matrix for transparency),
  llava_next (executed including the with-image path; interleaved pooling is
  `jevx_conversion=experimental`).
- **Levels 1/0:** none.

Former level-4 cells were **demoted to 3** when executed, because level 4 now
requires a real-dataset quality result owned by the benchmark suite — evidence
overlay documented in the matrix header and reports/REAL_MODEL_VALIDATION.md
Appendix C. Levels are per-family, per-level and evidence-based: the
709-record architecture inventory is **inventory, not support**
([SUPPORTED_MODELS.md](SUPPORTED_MODELS.md)).
