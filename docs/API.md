# JevX Public API Reference

Verified against the source in `jevx/**` (every signature below was read from the
implementation, not invented). Module paths are as imported. Language rules of
§76 apply: statements are *measured*/*observed* where empirical, otherwise
*supported*/*unsupported* behavior of the code.

Top-level import surface (`jevx/__init__.py`, `__all__`):

```python
from jevx import (
    JevX, JevXForDecision, JevXConfig, JevXOutput, DecisionOutput, Option,
    CalibrationEngine, ConfidenceEngine, JevXCapabilityDetector,
    UnsupportedRepresentationError, build_decision_head, register_decision_head,
    register_adapter, build_fusion, pool, risk_coverage_curve, coverage_at_risk,
    __version__,          # "0.1.0"
)
```

The Auto layer is a separate import: `from jevx.auto import AutoJevX` (importing
it also performs the session-local `AutoConfig`/`AutoModel` registration).

---

## 1. `JevX` / `JevXForDecision`

`JevX` **is** `JevXForDecision` — a plain alias of the same class (so saved
`architectures` stays `["JevXForDecision"]`). Subclass of
`transformers.PreTrainedModel`; `config_class = JevXConfig`,
`base_model_prefix = "model"` (the backbone is the child module `model`).

### 1.1 Constructor

```python
JevXForDecision(
    config: JevXConfig,
    *,
    backbone: nn.Module | None = None,   # pre-loaded backbone to embed (factory path)
    adapter: BaseAdapter | type | None = None,
    tokenizer=None,
    processor=None,
    **_,                                   # extra kwargs silently ignored (HF convention)
)
```

- `backbone=None` rebuilds the backbone from `config.backbone_config` via the
  Auto classes (checkpoint-reload path).
- A supplied `backbone` is marked `_is_hf_initialized` so `post_init()` never
  re-draws pretrained values; wrapper parameters are aligned to the backbone's
  device/dtype (skipped while either side holds meta tensors).
- `adapter=None` resolves through `jevx.adapters.resolve_adapter`; an adapter
  *class* is instantiated.
- Raises `ValueError` when the backbone's hidden width cannot be determined from
  config keys (`hidden_size`, `d_model`, `n_embd`, …, `projection_dim`, nested
  `text_config`/`vision_config`) and `decision_hidden_size` was not supplied.
- Reads `config.freeze_base_model` (default `True`) → freezes the backbone at the
  end of construction; heads are built from `config.decision_primitives`.

### 1.2 Factory: `from_pretrained`

```python
JevXForDecision.from_pretrained(
    pretrained_model_name_or_path,
    *args,                                  # POSITIONAL ARGS REJECTED (TypeError)
    decision_type: str | None = None,       # pin config.decision_type
    trust_remote_code: bool = False,        # never defaults to True
    tokenizer=None, processor=None,         # featurizer overrides
    adapter=None,                           # adapter override (instance or class)
    **kwargs,
)
```

**Wrap-vs-reload semantics** (§45/§63):

1. `_peek_config` classifies the path with `AutoConfig.from_pretrained` (hub-peek
   kwargs only: `cache_dir`, `force_download`, `local_files_only`, `revision`,
   `token`, `subfolder`). If `AutoConfig` rejects the config, the raw dict is read
   via `PreTrainedConfig.get_config_dict` and the JEVX-reload path is taken **only
   if** the file declares `model_type == "jevx"`; otherwise the original
   `AutoConfig` error is re-raised (so e.g. `prajjwal1/bert-tiny`, whose config
   lacks `model_type`, fails with AutoConfig's actionable message).
2. **Reload path** (`model_type == "jevx"`): standard
   `PreTrainedModel.from_pretrained` with the explicit `JevXConfig` — **no Auto
   class registration required** — then `_attach_reload_assets` re-attaches
   `jevx_metadata.json`, tokenizer/processor (auto-resolved when present), the
   adapter override, materializes calibrator buffers off the meta device and
   restores the `jevx_calibration.pt` sidecar (raises `ValueError` if the
   sidecar's format marker or method disagrees with the config).
3. **Wrap path** (`_wrap_base_model`): loads the backbone through the documented
   chain `AutoModel → AutoModelForCausalLM → AutoModelForSeq2SeqLM` (first success
   wins; raises `ValueError` listing all attempts if every class fails) → runs
   capability detection → resolves featurizers (text-only → `AutoTokenizer`,
   else `AutoProcessor`, fallback `AutoFeatureExtractor`; a processor's embedded
   tokenizer is exposed as `tokenizer` for wrap/reload symmetry) → sets a missing
   `pad_token` to EOS/UNK → builds `JevXConfig` → instantiates and returns the
   wrapper in `eval()` mode.

**Kwargs on the wrap path** are partitioned three ways; anything outside the
partition raises `TypeError` naming the accepted names:

- backbone loader: `cache_dir`, `force_download`, `local_files_only`, `revision`,
  `token`, `subfolder`, `proxies`, `resume_download`, `dtype` (canonical; a
  simultaneous `torch_dtype` is dropped), `torch_dtype` (BC), `attn_implementation`,
  `ignore_mismatched_sizes`, `use_safetensors`, `variant`, `device_map`;
- `JevXConfig` fields (`pooling_method`, `fusion_type`, `decision_hidden_size`, …
  — any dataclass field of `JevXConfig`);
- factory keywords: `decision_type`, `trust_remote_code`, `tokenizer`,
  `processor`, `adapter`.

Raises: `TypeError` (extra positional args; unknown kwargs), `ValueError`
(backbone chain failure; undeterminable hidden width; broken `backbone_config` on
reload: missing `model_type` and `base_model_name_or_path`; unregistered
`model_type` without `base_model_name_or_path`).

**`trust_remote_code` policy:** default `False`, passed through to
config/backbone/featurizer loading; it is never implied. On Windows the interactive
prompt path does not exist (no SIGALRM), so only an explicit `True` opts in
(audit §2; `tests/unit/test_auto_factory.py::test_trust_remote_code_defaults_false`).

### 1.3 `forward` — one call, B states × m questions

```python
forward(
    state=None,                  # str | list[str] | dict of tensors | raw tensor
    questions=None,              # str | list[str|dict] | [m, D_q] tensor | [B, m, D_q]
    decision_type=None,          # "choice"|"score"|"noul" | list (len 1, m, or B*m) | None
    options=None,                # schema per question: list-of-lists (len 1, m, B*m)
                                 # or one flat schema when B*m == 1
    confidence_threshold=None,   # float in [0, 1] → abstention
    return_base_output=False,    # include raw backbone output in `base_output`
    *,
    input_ids=None, attention_mask=None, token_type_ids=None,
    output_hidden_states=None,   # None → config flag
    output_attentions=None,      # None → config flag (attentions need eager attention)
    condition_options=None,     # bool | None → STATE-ONLY (False) or conditioned
    # (True) scoring; None → _resolve_condition(None), i.e. the config
    #          MODE, read on every call: "off" (DEFAULT) → False,
    #          "constant" → True, "full" → bool(config.conditioning_enabled)
    return_dict=None,            # None → config.return_dict; False → tuple form
    **kwargs,                    # extra TENSOR model inputs (e.g. pixel_values);
                                 # non-tensor kwargs raise TypeError
) -> JevXOutput | tuple
```

Execution (documented in the module docstring): the state batch is featurized and
passed through the backbone **exactly once** per call; all m questions encode in
one batch; schemas encode with variable K + bool mask; fusion/heads run per
(state, question) row grouped by decision type (probability spaces are never
concatenated); calibration (when configured **and** fitted) is applied to head
logits; confidence sees the calibrated probabilities; abstention clears predicted
fields per question without touching probabilities.

****`condition_options` (per call, never mutates the heads):** `None` ->
`JevXForDecision._resolve_condition(None)`, i.e. the config MODE: `"off"`
(the shipped default) -> `False`; `"full"` -> `bool(config.conditioning_enabled)`,
read on **every** call so a runtime flip (e.g. `conditioning_gate.apply_gate`)
takes effect; `"constant"` -> `True`.

`False` makes the call **state-only in both halves of the conditioning path**:

* the option interaction is switched off inside the head for this call (the
  residual is exactly zero - it is not even computed - so no gradient reaches
  the option rows),
* the schema rows are removed *before* fusion (fusion receives `options=None`),
  and the head then receives exact zeros of the original `[B, K, D]` shape so the
  K-column contract (mask and schema metadata) is unchanged;
* the **question rows handed to fusion are exact zeros** as well, so the fused
  state does not depend on the question either. (Since P0-2. Shapes are
  unchanged - the fused state is still `[B, D]`.)

Because the question path is neutralized too, the state-only call's fused state
is NOT byte-identical to the conditioned call's: the option branch still never
feeds back into `fused` (a state-only call is schema-invariant - see the
duplicate-tie scope rule in `MATHEMATICAL_SPECIFICATION.md` (5.h)), but the
question half does. `True` forces the conditioned call for one call. A non-bool
non-`None` value raises `TypeError`. Passing `condition_options=False`
additionally passes `duplicate_tie=False` to the head - see section 10 and (5.h).


**`decisions` layout:** a flat list of `B*m` single-row `DecisionOutput` objects,
**state-major** — entry `b*m + j` is state `b` answered with question `j`
(`logits` keep head batch semantics with leading dim 1, shape `[1, K]`).

Resolution rules: `decision_type=None` → argument → `config.decision_type` →
`"choice"`. `state=None` requires `input_ids=` (else `ValueError`); passing both
`state=` and `input_ids=` raises `ValueError`.

Raises (`ValueError`/`TypeError`, each message names the offending argument):

| Condition | Error |
|---|---|
| unknown decision type | `ValueError` — lists `decision_primitives` and registered heads |
| `decision_type` list length not in {1, m, B*m} | `ValueError` |
| missing schema for `choice`/`score` (or `options=[[]]`) | `ValueError` (axiom D3.1: K ≥ 1) |
| options supplied to `noul` | `ValueError` ("noul is binary and takes no options") |
| empty questions / empty option rows / bad option entry types | `ValueError`/`TypeError` |
| schema count not in {1, m, B*m}, or a flat schema with B*m > 1 | `ValueError` |
| `confidence_threshold` outside [0, 1] or non-numeric | `ValueError`/`TypeError` |
| `attention_mask` batch ≠ state batch | `ValueError` naming both sizes |
| `attention_mask` float dtype, or integer values outside {0,1} | `ValueError` (binarize remedy in message) |
| state token length > learned position-table limit | `ValueError` — "JevX never silently truncates a state" |
| non-finite state/question/schema tensors | `ValueError` from `check_finite` (NaN/+Inf/−Inf counts + first index) |
| pre-encoded questions of non-float dtype | `TypeError` |
| pre-encoded `[B, m, D_q]` with wrong B; tensor rank wrong | `ValueError` |
| no batch dimension inferable from prepared inputs | `ValueError` |
| adapter cannot extract a representation | `UnsupportedRepresentationError` (structured) |
| non-tensor non-control kwargs | `TypeError` |

### 1.4 Convenience methods

| Method | Signature | Semantics |
|---|---|---|
| `decide(...)` | `(state, questions, decision_type=None, options=None, **kw) -> JevXOutput` | general entry; accepts 1 state or a batch |
| `choice(...)` | `(state, questions, options=None, **kw)` | `decision_type="choice"` (options required per question) |
| `score(...)` | `(state, questions, options=None, **kw)` | `decision_type="score"` — options are levels 0..K-1 |
| `noul(...)` | `(state, questions, **kw)` | `decision_type="noul"` — schema forbidden; forwards a caller-supplied `options=` to reach the documented `ValueError` |
| `base_forward(*args, **kwargs)` | → raw backbone output | pass-through `self.model(...)`; in eval mode bit-identical to loading the base checkpoint directly (verified per checkpoint: `base_forward_max_abs_error`, 0.0 everywhere except llava-next 5.66e-7 — reports/REAL_MODEL_VALIDATION.md) |
| `trainable_parameters()` | generator of `nn.Parameter` | yields every parameter **not** under `model.` (structural split used by `freeze_backbone`/`unfreeze_backbone`); for Mode B iterate `model.parameters()` |
| `unfreeze_last_n_layers(n)` | → `self` (chainable) | Mode B helper; layer discovery by convention (`encoder.layer` / `h` / `layers` under `model.`) |
| `register_adapter(adapter_cls)` | static method | registry passthrough — highest resolution priority |
| `get_input_embeddings()` / `set_input_embeddings(v)` | delegate to the backbone | required overrides (audit risk #2) |

### 1.5 Save / load

```python
model.save_pretrained(save_directory, *args, **kwargs)  # → super().save_pretrained result
```

Writes into `save_directory`:

| File | Contents |
|---|---|
| `config.json` | `JevXConfig` incl. nested `backbone_config` dict (rebuild without re-downloading the base checkpoint), `architectures`, `dtype` |
| `model.safetensors` | wrapper + embedded backbone; declared child ties (e.g. causal `lm_head.weight`) removed before writing and restored on load (safetensors has no aliasing) |
| `jevx_metadata.json` | format `jevx-metadata-v1`: capability report, adapter identity, `pooling_method`, `fusion_type`, `base_model_name_or_path`, `decision_type`, `jevx_version` |
| `jevx_calibration.pt` | written **only when** `calibration_method != "none"`: format `jevx-calibration-v1`, method name, calibrator `state_dict` (fit-data-shaped buffers cannot ride safetensors — v5's loader drops shape-mismatched tensors) |
| tokenizer/processor files | when attached |

`state_dict()` **filters out** `calibrator.*` keys (symmetric with
`from_pretrained`'s expectations); `load_state_dict(state, strict=True, assign=False)`
additionally accepts a dict *containing* `calibrator.*` keys by reallocating
fit-shaped buffers to the checkpoint's shapes first. Confidence fit state
(`energy_lo/hi`) has fixed shapes and rides the regular state_dict.
Tests: `tests/unit/test_serialization_jevx.py` (5 tests incl. sidecar round-trip,
method-mismatch `ValueError`, tied causal-LM round-trip).

---

## 2. `AutoJevX` (`jevx.auto`)

```python
from jevx.auto import AutoJevX
model = AutoJevX.from_pretrained(path, **kwargs)   # → JevXForDecision
```

Thin factory: `model_type == "jevx"` → direct reload; anything else → the wrap
path. Importing `jevx.auto` performs idempotent **session-local** registration:
`AutoConfig.register("jevx", JevXConfig)` and
`AutoModel.register(JevXConfig, JevXForDecision)` — process-local only (must run
again in every new process); no `trust_remote_code` / `auto_map` is needed for
local directories. `AutoJevX` forwards `trust_remote_code` as passed (default
`False`). See [TRANSFORMERS_INTEGRATION.md](TRANSFORMERS_INTEGRATION.md) for the
split-brain limitation.

---

## 3. `JevXConfig` (`model_type = "jevx"`)

Subclass of `PreTrainedConfig` (v5 strict-dataclass conventions), decorated
`@strict`. Fields (defaults read from `configuration_jevx.py`):

| Field | Type | Default | Meaning / validation |
|---|---|---|---|
| `base_model_name_or_path` | `str \| None` | `None` | wrapped checkpoint id |
| `base_model_type` | `str \| None` | `None` | e.g. `"bert"` |
| `representation_strategy` | `str` | `"native"` | metadata-only (extraction strategy is `pooling_method`) |
| `pooling_method` | `str \| None` | `None` | sequence pooling for `H_S`; `None` = adapter default |
| `decision_hidden_size` | `int` | `768` | width D of fused state / heads — must be ≥ 1 (`ValueError`) |
| `question_hidden_size` | `int` | `768` | `E_Q` width — ≥ 1 |
| `schema_hidden_size` | `int` | `768` | `E_C` width — ≥ 1 |
| `fusion_type` | `str` | `"concat"` | see §7 fusion kinds |
| `decision_primitives` | `list[str]` | `["choice", "score", "noul"]` | heads built at construction; must be a list/tuple of str (`TypeError`) |
| `calibration_method` | `str` | `"none"` | one of `("none", "temperature", "vector", "isotonic")` — validated eagerly, unknown → `ValueError` |
| `confidence_method` | `str` | `"max_probability"` | see §6 |
| `freeze_base_model` | `bool` | `True` | freeze backbone at construction |
| `runtime_validation` | `bool` | `True` | run the `jevx.validation` ValidationEngine checks from `JevXForDecision.forward` (fix-wave contract; `False` disables the runtime checks) |
| `state_pathway` | `str` | `"linear"` | kind of the PRIMARY state score of the decision heads: `"linear"` (`z_k = sᵀW[k,:]+b[k]`), `"mlp"` (`h_s = act(W1·s+b1)`, `z_k = h_sᵀW2[k,:]+b2[k]`). Unknown → `ValueError` listing the roster |
| `state_hidden_size` | `int \| None` | `None` | hidden width `H` of the `"mlp"` pathway; `None` = `decision_hidden_size`. Non-positive or non-int → `ValueError`/`TypeError`. Ignored by `"linear"` |
| `state_activation` | `str` | `"gelu"` | elementwise activation of the `"mlp"` pathway. Unknown → `ValueError` |
| `conditioning_scale_init` | `float` | `0.0` | initial `λ` of the option-interaction residual. Non-finite → `ValueError`. **See the mapping below** |
| `conditioning_mode` | `str` | `"off"` | SOURCE OF TRUTH for the conditioning path; one of `CONDITIONING_MODES = ("full", "constant", "off")`. Unknown → `ValueError` listing the roster. **See "Default conditioning mode" below** |
| `conditioning_enabled` | `bool` | `True` (class default) — a fresh `JevXConfig()` derives **`False`** from the default mode | DERIVED ALIAS of `conditioning_mode`, never an independent switch. When it ends up `False` the residual is EXACTLY zero on every call, no gradient can reach the option rows, and the question rows entering fusion are exactly zero. Non-bool → `TypeError` |
| `decision_type` | `str \| None` | `None` | default decision type for calls that pass none. `None` = **no pin** and `forward(decision_type=...)` falls back to `"choice"`. A name is validated against the live head registry at construction (`ValueError` naming the roster + how to register one; `TypeError` for a non-`str`/`None`), and round-trips through `save_pretrained`/`from_pretrained`. It is the same value the `JevX.from_pretrained(decision_type=...)` factory kwarg pins. |
| `schema_encoder_width` | `int \| None` | `None` | width the **option** encoder `E_C` runs at; `None` = `schema_hidden_size` (today's width). Implemented in `JevXForDecision.__init__`: both `schema_encoder.hidden_size` and `schema_proj.in_dim` become this value, so the option encoder is *narrowed* without touching `E_Q`. Non-positive / non-int → `ValueError`; wider than `schema_hidden_size` → `ValueError` (the field exists to run the encoder at a **fraction** of the stack width). The default reproduces today's module tree and state_dict bit for bit. |
| `shared_question_schema_encoder` | `bool` | `False` | **SPECIFIED, NOT IMPLEMENTED — RESERVED.** Non-bool → `TypeError`. **`True` is REJECTED in `__post_init__` with a named `ValueError`** (and re-raised defensively at `JevXForDecision.__init__`, `# pragma: no cover`): the config can never silently accept a switch that does nothing. Intended semantics: one shared `QuestionEncoder` with a learnable type tag (`is_question`/`is_option`) instead of two encoders. **Blocking reason is serialization, not the encoder** — one shared module appears twice in `state_dict()` (`question_encoder.*` and `schema_encoder.text_encoder.*`) and transformers refuses to serialize shared tensors unless the alias is declared in `_tied_weights_keys`, which on this wrapper also carries the backbone's own tied weights. The supporting pieces exist and are tested alone (`QuestionEncoder(type_tags=True)`, `SchemaEncoder(text_encoder=...)`); what is missing is the model-level wiring + the tied-weight declaration. |
| `version` | `str` | `"0.1.0"` | checkpoint format metadata |
| plus inherited | | | `output_hidden_states=False`, `output_attentions=False`, `return_dict=True`, `dtype`, `architectures`, `transformers_version` (not redeclared — redeclaring would shadow base properties) |

`CALIBRATION_METHODS` is exported from `jevx.configuration_jevx`.

### 3.1 Default conditioning mode

**A freshly built `JevXConfig()` has `conditioning_mode == "off"`.**

```python
>>> JevXConfig().conditioning_mode
"off"
>>> JevXConfig().conditioning_enabled      # derived alias, follows the mode
False
```

**Revert it in one line** — the `conditioning_mode` field default in `jevx/configuration_jevx.py`:

```python
conditioning_mode: str = "off"    # shipped default
conditioning_mode: str = "full"   # <- revert: this one line is the whole change
```

**If you want question conditioning, set it explicitly:**

```python
model = JevX.from_pretrained("...", conditioning_mode="full")   # one kwarg
# or per call:
out = model(state=s, questions=q, options=o, condition_options=True)
```

**Why** (measured, `reports/MULTI_QUESTION.md` §5–§6; 11 questions, 4 datasets, 5 seeds,
frozen features, every arm from bitwise-identical initial weights):

| measurement | value |
|---|---|
| pooled accuracy over 11 questions, `full` | 0.7046 |
| pooled accuracy over 11 questions, `off` | 0.7199 |
| pooled accuracy over 11 questions, `state_only` reference (= today's `off`) | 0.7294 |
| paired comparisons won by any conditioning mode | **0 / 35** |
| only non-overlapping comparison (a LOSS for `full`) | `ravdess_gender`, **−16.92 pp** (`full` 0.7296 ± 0.0687 vs `off` 0.8988 ± 0.0535) |
| trainable parameters in the conditioning half | **80.8 – 89.7 %** of the stack |
| gradient those parameters receive in the `off` arm at optimizer step 1 | **exactly 0** |

This is a DEFAULT, not a removal: `full` and `constant` are unchanged and fully supported, and
the `"constant"` H_Q/H_C memo is still bitwise-equal to `"full"` on a cold call, on a warm
repeated call with the same content, and after a content change, **provided the encoder
parameters have not moved** — the memo is weight-keyed (see below). It also does **not**
establish that conditioning never helps (§7 of the same report) — see
`MATHEMATICAL_SPECIFICATION.md` §5.i.

**Precedence table** (`conditioning_mode` is the source of truth, `conditioning_enabled` is
derived; total over all six pairs, no silent third outcome):

| `conditioning_mode` | `conditioning_enabled` requested | outcome |
|---|---|---|
| `"full"` | `True` | agree — `True` |
| `"full"` | `False` | `ValueError` naming BOTH fields |
| `"constant"` | `True` | agree — `True` |
| `"constant"` | `False` | `ValueError` naming BOTH fields |
| `"off"` | `True` | **coerced to `False`** — the mode wins, no error |
| `"off"` | `False` | agree — `False` |

The `("off", True)` pair coerces because "off" and `conditioning_enabled=False` are the same
request, not a contradiction. `forward(condition_options=...)` still overrides the mode for one
call.

**The `"constant"` memo is weight-keyed.** Its entries are invalidated whenever any parameter or
buffer tensor of `question_encoder` / `schema_encoder` has been mutated since the entry was
computed, checked through `torch.Tensor._version` counters (one integer read per TENSOR). A
`eval -> optimizer.step() -> eval` loop therefore recomputes instead of serving an `H_Q`/`H_C`
### 3.2 The `conditioning_scale_init` -> `lambda` mapping

`conditioning_scale_init` is the INITIAL `lambda` of the option-interaction residual in
`z_k = state_k + lambda * r_k` (spec (5.g)). The head holds `lambda` in a persistent BUFFER
(`interaction_scale`), never as an optimizer parameter — it is a curriculum control, not a
weight. [FACT — `JevXForDecision._state_pathway_kwargs`, `jevx/decisions/generic.py`]

| `config.conditioning_scale_init` | forwarded to the head as | resulting head `interaction_scale` |
|---|---|---|
| `0.0` (**the default**) | **not forwarded** — read as *"unspecified"* | `1.0` (the head's own default) |
| any other finite value, e.g. `0.25` | that value | that value |

**Why `0.0` means "unspecified" and not "no residual".** Forcing the constructor value to
`0.0` would silently change what a **pre-existing checkpoint** computes:
`interaction_scale` is a PERSISTENT buffer, a checkpoint written before the buffer existed
loads with `strict=False` and therefore KEEPS the constructor value — and would lose its
trained residual. Mapping the config default `0.0` to the head default `1.0` keeps a
default-constructed head bit-identical to the pre-flag head.

`lambda = 0.0` remains reachable at RUNTIME by either `head.set_interaction_scale(0.0)` or
`forward(condition_options=False)` — and the latter does not even COMPUTE `r_k` ((5.g1)), so
no gradient can reach the option rows and a non-finite option feature cannot become
`0 * inf = NaN`. A curriculum can start a ramp from any non-zero `conditioning_scale_init`
(`jevx.training.curriculum`). Non-finite / non-numeric -> `ValueError` in `__post_init__`
("an inf/NaN lambda would poison every logit"). Spec statement: `MATHEMATICAL_SPECIFICATION.md`
(5.g2); pinned by `tests/unit/test_head_next.py::test_default_config_passes_no_head_kwargs` and
`tests/unit/test_state_only_path.py`.

**Related — `_state_pathway_kwargs` at the shipped defaults.** [FACT] It returns `{}` (no head
kwargs at all) only for `conditioning_mode="full"`. With the shipped default `"off"` the
DERIVED `conditioning_enabled` is `False`, so it returns an explicit dict whose
`conditioning_enabled` is `False` and whose `conditioning_scale_init` is `1.0` — the same
pre-flag head, reached by a different route. Both modes are asserted in
`tests/unit/test_head_next.py::test_default_config_passes_no_head_kwargs`.
computed from the previous weights. `clear_condition_cache()` is still available (and is still
called by `load_state_dict`) but is no longer required after an in-place parameter update.

---

## 4. Output dataclasses (`jevx.modeling_outputs`)

Both are `transformers.utils.ModelOutput` subclasses: at most one required field,
`None` fields omitted from dict/tuple views, `return_dict=False` → `to_tuple()`
(non-`None` fields in declaration order).

### 4.1 `DecisionOutput` — one decision

| Field | Type / shape | Notes |
|---|---|---|
| `logits` **(required)** | `[B, K]` tensor | raw, **unmasked** (calibration sees true logits) |
| `probabilities` | `[B, K]` or `None` | mask-aware; rows sum to 1 (softmax) or `sigmoid` for noul; masked positions exactly 0 |
| `confidence` | `[B]` in [0,1] or `None` | filled by `ConfidenceEngine`, not by heads |
| `predicted_index` | `long [B]` or `None` | in `[0, K)` (noul: `1[p ≥ 0.5]`); cleared on abstention |
| `expected_score` | `[B]` or `None` | score head only: `Σ k·p_k ∈ [0, K-1]` |
| `predicted_value` | any or `None` | schema-projected value (`project_to_schema`); `None` on abstention |
| `abstained` | `bool [B]` tensor or `None` | set when `confidence < confidence_threshold` |
| `calibration_metadata` | dict-like or `None` | e.g. `{"method": "temperature", "applied": True}` |
| `schema_metadata` | dict-like or `None` | `{"decision_type": ..., "option_keys": [...]}` |

### 4.2 `JevXOutput` — one forward call

| Field | Type / shape | Notes |
|---|---|---|
| `decisions` **(required)** | `list[DecisionOutput]` (or stacked tensor) | flat `B*m`, state-major `b*m + j` |
| `state_representation` | `[B, D]` or `None` | shared projected `H_S` |
| `question_representations` | `[B, m, D]` or `None` | per-state-expanded field |
| `schema_representations` | `[B*m, K_max, D]` or `None` | `None` for schema-less groups (noul) |
| `base_output` | raw backbone output or `None` | when `return_base_output=True` |
| `hidden_states` | tuple of `[B, T, D]` or `None` | when `output_hidden_states` |
| `attentions` | tuple or `None` | when `output_attentions` (needs eager attention) |
| `capability_metadata` | dict or `None` | model_type, architecture, representation_field, adapter, pooling/fusion, decision_types, confidence_method, calibration, batch sizes, `pair_ordering`, `backbone_forward_calls` |

---

## 5. Schema: `Option` (`jevx.questions`)

```python
Option(key, label="", description="", metadata={})
Option.text()   # "label: description" | either side | str(key)  (never empty)
```

`forward(options=...)` additionally accepts per-entry `str` (key=label), `dict`
(only `key`/`label`/`description`/`metadata`; missing `key` → `ValueError`),
numbers (ordinal levels, key preserved), and `bool` (rejected — ambiguous,
`TypeError`). `metadata` is never read by any JevX map (axiom D2.2).

---

## 6. Capability detection (`jevx.capabilities`)

### 6.1 `CapabilityReport` fields

`text`, `vision`, `audio`, `encoder`, `decoder`, `encoder_decoder`,
`hidden_states`, `pooler`, `generation`, `native_representation`,
`fallback_representation` (bools) · `jevx_conversion`
(`"supported" | "experimental" | "unsupported"`) · `representation_field`,
`output_class`, `model_type`, `architecture` (str or None) · `input_fields`
(list[str]) · `processor_required` (bool) · `attention_layout` (str or None:
summary of the declared `config.layer_types` — also read from a nested
`config.text_config.layer_types`, e.g. Qwen3.5; `None` when not declared
(parse-only, never inferred), `"full"` when all `full_attention`, `"linear"`
when all `linear_attention`, else `"hybrid:<count>x<type>+<count>x<type>"`
with per-type counts in config order; a hybrid layout also appends one
`reasons` line — informational only, never affects `jevx_conversion`) ·
`reasons` (list[str], one per decision). `to_dict()` → plain JSON-serializable
dict.

### 6.2 `JevXCapabilityDetector`

- `detect(model, config=None) -> CapabilityReport` — pure introspection (forward
  signature, config, module tree); **never runs the model**, never raises for odd
  models (unidentifiable routes report `jevx_conversion == "unsupported"` with
  populated `reasons`).
- `probe(model, dummy_inputs) -> CapabilityReport` — detect + exactly one
  cautious `eval()`/`no_grad` forward; failures are appended to `reasons`, never
  raised.

### 6.3 `UnsupportedRepresentationError`

Fields (all mirror the constructor 1:1, always populated):
`model_name`, `architecture`, `detected_outputs` (list), `attempted_strategies`
(list), `failed_strategies` (list, `"label: reason"` entries), `suggestion`,
`message`. Raised by adapter extraction/validation and by `forward` when no
usable representation exists (`tests/unit/test_failure_modes.py` row 12 asserts
the structured fields).

---

## 7. Adapter registry (`jevx.adapters`)

```python
register_adapter(cls)      # class decorator; BaseAdapter subclass required (TypeError);
                           # duplicate name → ValueError; re-registering the same class is idempotent
resolve_adapter(model)     # user adapters (registration order) → built-in chain → GenericAdapter
available_adapters()       # names in resolution order
BUILTIN_ADAPTER_CHAIN      # [Multimodal, EncoderDecoder, Vision, Audio, Decoder, Encoder, Generic]
USER_ADAPTERS              # list of user-registered classes (highest priority)
```

Built-in chain order and family `can_handle` predicates (introspection only, no
forward): **Multimodal** (vision AND text) → **EncoderDecoder**
(`config.is_encoder_decoder`) → **Vision** (vision, no text) → **Audio** (audio,
no text) → **Decoder** (cache-capable causal stack, no enc-dec, no vision/audio) →
**Encoder** (encoder direction, none of the above) → **Generic** (always `True` —
resolution never fails).

Adapter contract (`BaseAdapter`): `can_handle(model)`, `prepare_inputs(state, *,
tokenizer=None, processor=None, **kw) -> dict`,
`extract_representation(model, inputs, *, strategy=None, output_hidden_states=False,
mode=None) -> Representation`,
`describe_capabilities(model) -> CapabilityReport`,
`validate(model, sample_state)` (the one cautious `no_grad` dry run; converts
failures into `UnsupportedRepresentationError`). Extraction priority:
`pooler_output` → semantic fields (`sentence_embedding`, `text_embeds`,
`image_embeds`) → `last_hidden_state` → `encoder_last_hidden_state` →
`decoder_state` (tuple outputs, enc-dec `mode="decoder_state"`) →
`hidden_states[-1]` (only when `output_hidden_states=True` was requested).
Family `default_pooling`: encoder `cls`, decoder `last_valid`,
encoder_decoder `encoder_pool`, vision/audio/multimodal/generic `masked_mean`.
`strategy="native"` requires a native field — otherwise `UnsupportedRepresentationError`
with the remedy in `suggestion`.

Exports also include `Representation` (fields: `sequence_output`, `pooled`,
`attention_mask`, `raw_output`, `field`, `layout ∈ {"btd","bd"};` raises
`ValueError` on layout/consistency violations), `ENC_DECODER_MODES`
(`"encoder_state" | "decoder_state" | "encoder_decoder_fusion"`), text helpers
(`tokenize_texts`, `is_text_state`, `TEXT_PROCESSOR_KEYS`), `forward_kwargs`,
`observed_output_fields`, `SEMANTIC_FIELDS`, `FORWARD_CONTROL_FIELDS`.

---

## 8. Pooling (`jevx.representations.pooling`)

```python
available_pooling_strategies()  # sorted names
pool(sequence_output, attention_mask, strategy, *, pooler_output=None, hidden_states=None)
```

Registry `POOLING_STRATEGIES` (all return `[B, D]`, input dtype/device):
`attention_pool`, `cls`, `encoder_pool` (documented alias of `masked_mean`),
`last_valid`, `learned_pool`, `max`, `masked_mean`, `mean`, `native`.

Invariants (tested in `tests/unit/test_pooling.py`, `tests/numerical`): padding
invariance for every mask-aware strategy (left/right/unpadded identical);
fp16/bf16 pooled in fp32 then cast back; stateless `attention_pool`/`learned_pool`
re-init is forked-seed deterministic (caller RNG untouched); fully-masked rows
raise `ValueError` (axiom D1.1). Raises: `TypeError` (non-str strategy, bad mask
type, non-float sequence), `ValueError` (unknown strategy — message lists the
registry; mask shape mismatch; fully-masked row). `strategy="native"` with no
`pooler_output` raises `ValueError` (never silently pools).

Also exported: `AttentionPooler(hidden_size)`, `LearnerPooler(hidden_size,
attn_hidden=None)` — trainable modules owning their parameters.

---

## 9. Fusion (`jevx.representations.fusion`)

```python
build_fusion(kind: str, hidden_size: int, **kwargs) -> Fusion   # TypeError (non-str),
                                                                # ValueError (unknown kind, lists registry)
available_fusions()   # ['add', 'additive', 'bilinear', 'concat', 'cross_attention', 'dot', 'gated', 'mlp']
```

| kind | context formula |
|---|---|
| `concat` (default) | `c = W_cat [s ; q] + b` |
| `additive` (alias `add`) | `c = s + W_q q` |
| `gated` | `g = σ(W_g [s;q] + b); c = g⊙s + (1−g)⊙W_q q` |
| `bilinear` | `c = s ⊙ (W_q q)` |
| `cross_attention` | question embedding as query over the state **sequence** `[B, T, D]` as K/V (spec 5.e): multi-head `c = W_o concat_h(softmax_t(Q_h K_hᵀ/√d_h + B(M)) V_h)`; per-option rows `e_k = g⊙u_k + (1−g)⊙c` with `u_k = CrossAttn(W_cq[o_k; c], K, V)` — fed via the keyword-only `state_sequence=`/`state_mask=` inputs |
| `dot` | `c = s ⊙ q` |
| `mlp` | `c = MLP([s ; q])` (2 Linear + GELU) |

All kinds share the per-option gate formula
`ô_k = o_k / max(‖o_k‖₂, ε)`, `e_k = g ⊙ W_o ô_k + (1−g) ⊙ c` (spec §5.f —
masked positions `masked_fill`ed to exact 0). The option row is reduced to a
unit DIRECTION before `W_o` (`Fusion.__init__(..., option_normalize=True)`,
the default and the normative contract; `OPTION_NORM_EPS = 1e-12` so an all-zero
padded row normalizes to exact 0, never NaN). Passing
`option_normalize=False` restores the pre-§5.f formula `e_k = g ⊙ W_o o_k +
(1−g) ⊙ c` verbatim — required to load checkpoints trained before the
normalization landed. `CrossAttentionFusion` applies the same `ô_k` to its
option query (`W_cq [ô_k ; c]`).
`Fusion.forward(state, question, options=None, option_mask=None, *, state_sequence=None,
state_mask=None) -> FusionOutput`
(fields `fused [B,D]`, `option_features [B,K,D] | None`; the keyword-only
`state_sequence`/`state_mask` inputs are consumed by `cross_attention` only —
any other kind receiving them raises `ValueError` naming the kind; without a
sequence `cross_attention` degrades to a `T=2` sequence `[s ; q]`). Raises `TypeError`
(non-tensor, non-bool mask) and `ValueError` (rank/width/batch/mask-shape
mismatch, mask without its tensor, all-masked sequence row, non-finite inputs).

`Projection(in_dim, out_dim, layer_norm=True, mlp=False)` — `y = LN(W x (+ b))`
or the 2-layer MLP form; raises `ValueError` (non-positive dims, last-dim
mismatch) / `TypeError` (non-tensor input, non-bool flags).

---

## 10. Decision-head registry (`jevx.decisions`)

```python
DECISION_HEAD_REGISTRY            # dict[str, type[DecisionHead]]
register_decision_head(name)      # class decorator; duplicate name (other class) → ValueError
get_decision_head(name)           # TypeError (non-str) / ValueError (unknown — lists roster)
build_decision_head(name, **kw)   # instantiate
available_decision_heads()        # sorted: ['choice', 'multi_label', 'noul', 'regression', 'score']
project_to_schema(indices, option_keys)   # ValueError: empty schema or index outside [0, K)
mask_from_num_options(num_options, width, batch_size=None)
```

Built-ins: **`choice`** (softmax over K options; interactions `dot`/`bilinear`
(default)/`mlp`; requires `option_features` → `ValueError` without),
**`score`** (levels 0..K-1 + expected score asserted inside `[0, K−1]` at
tolerance 1e-4 — raises `RuntimeError` on escape rather than clamping; two
schemas: variable-K option features, or fixed `num_levels=` linear head),
**`noul`** (single sigmoid logit; `p` clamped to `[eps, 1−eps]` so stored
probabilities are strictly interior for every finite fp-representable logit —
PROOFS Remark 4.1; rejects any schema input with `ValueError`), plus
**experimental** `multi_label` (independent sigmoids, `predicted_index` stays
`None`) and `regression` (scalar `predicted_value`, no probabilities).

`DecisionHead.forward(fused [B,D], option_features [B,K,D] | None, *,
option_mask=None, num_options=None, **kw) -> DecisionOutput` — validates shapes
(batch/rank/finiteness/mask dtype/counts; `TypeError`/`ValueError` messages name
the tensor), computes logits, applies `masked_softmax` (invalid positions exactly
0; fully-masked row → `ValueError`), returns logits/probs/predicted_index.
`DecisionHead.project(probabilities, option_keys=None)` → index tensor or schema
values (membership validated).

Interaction helpers accept only `dot`/`bilinear`/`mlp` — anything else raises
`ValueError` listing them.

### 10.1 State pathway, `λ` scale, and the duplicate-tie switch

```python
DecisionHead.set_interaction_scale(value: float) -> None   # @torch.no_grad; sets λ
DecisionHead.get_interaction_scale -> float               # property; current λ
head.state_pathway_name -> str        # the SELECTOR: "linear" | "mlp" | "none"
head.state_pathway       -> nn.Module | None   # the MODULE (or None); ships as
                                              # a state_dict prefix, e.g.
                                              # "state_pathway.weight"
```

`set_interaction_scale` is the hook a `λ` ramp drives. A value of exactly `0.0`
makes the residual exactly zero on the NEXT forward (it is not computed at all);
a non-finite value raises `ValueError`. On a head that has NO interaction
(`noul`/`multi_label`/`regression`, whose scorer IS the state pathway) the buffer
is registered **non-persistent** and stays out of the `state_dict`, so a
classifier transplant sees exactly the tensors that define its scoring function.

Scoring: `z_k = state_k + λ·r_k`, the state term PRIMARY and the option
interaction the additive RESIDUAL it is scaled by.

| `state_pathway` | `state_k` | residual scored against |
|---|---|---|
| `"linear"` (default) | `sᵀW[k,:] + b[k]` | the fused state `s` |
| `"mlp"` | `h_sᵀW2[k,:] + b2[k]`, `h_s = act(W1·s+b1)` | `h_s` when `state_hidden_size == decision_hidden_size`, else `s` |
| `"none"` | absent (interaction-only ablation arm) | — |

`ChoiceHead(state_pathway=<bool>)` also accepts the shipped legacy bool spelling
(`True` = linear, `False` = the pre-2026-09-24 interaction-only head) so the v1-vs-v2
ablation stays reproducible from the shipped class; a string selects the pathway
kind. The selector is always readable as `state_pathway_name` because
`ChoiceHead.state_pathway` is the pathway MODULE (a shipped `state_dict` key).

**`conditioning_scale_init 0.0 → 1.0` on a fresh model.** [documented + tested]
A `JevXConfig` at its shipped defaults passes **no** head-constructor kwargs at
all (`JevXForDecision._state_pathway_kwargs` returns `{}`), and the heads' own
`conditioning_scale_init` default is `1.0`, so a fresh `JevXForDecision` starts
at **`λ == 1.0`** even though its config says `0.0`. A config value of exactly
`0.0` is read as "unspecified" and mapped to `1.0`. This is deliberate:
`interaction_scale` is a PERSISTENT buffer, so a checkpoint written before the
flag existed loads with `strict=False`, keeps the CONSTRUCTOR value, and would
silently lose its residual if that value were forced to `0.0`. To start at a
different `λ`, set a NON-ZERO `conditioning_scale_init` (what a curriculum does,
and it also sets the scale explicitly at every step). `0.0` remains reachable at
RUNTIME through `set_interaction_scale(0.0)` / `forward(condition_options=False)`.
Pinned by `tests/unit/test_head_next.py` and `tests/unit/test_state_only_path.py`.

**`duplicate_tie` (per call).** `ChoiceHead.compute_logits(..., duplicate_tie=None)`:
`False` skips the duplicate-option tie for this call, `True` forces it, `None`
(the default) applies the documented per-schema rule. It reaches the head through
`DecisionHead.forward`'s `**kwargs`. `JevXForDecision.forward` passes
`duplicate_tie=condition`, so a state-only call suppresses the tie. Rationale and
the exact degeneracy it avoids: `MATHEMATICAL_SPECIFICATION.md` (5.h).

---

## 11. Calibration (`jevx.calibration`)

```python
# registry
CALIBRATOR_REGISTRY               # {"none", "temperature", "vector", "isotonic"}
register_calibrator(name)         # decorator; duplicate name → ValueError
get_calibrator(name)              # TypeError / ValueError (roster in message)
build_calibrator(name, **kw)
available_calibrators()
validate_fit_inputs(logits, labels)  # [N,K] float + [N] long; K=1 → labels in {0,1};
                                     # else labels in [0,K); TypeError/ValueError otherwise
```

`BaseCalibrator(nn.Module)`:

| Method | Signature / behavior |
|---|---|
| `fit(logits, labels) -> Self` | **on a CALIBRATION split — never test**; subclass-specific |
| `calibrate_logits(z)` | `[N,K] → [N,K]`, same shape/dtype/K (schema invariant) |
| `calibrate_probs(p)` | default: `log p` → `calibrate_logits` → `softmax` |
| `calibrate(x, input_type="logits"|"probs")` | dispatch; bad `input_type` → `ValueError` |
| `evaluate(logits, labels, n_bins=15)` | `{"nll", "ece", "mce", "brier"}` floats (sigmoid path for K=1) |
| `is_fitted` | property — persisted `_fitted` buffer |
| `save(path)` / `BaseCalibrator.load(path)` | `torch.save/load`, format marker `jevx-calibration-v1`; wrong file → `ValueError` |
| `load_state_dict(sd, strict=True, assign=False)` | reallocates fit-shaped buffers/params to the saved shapes before the strict load |

Methods: **`none`** (`IdentityCalibrator`), **`temperature`** (deterministic
log-grid T ∈ [1e−2, 1e2], 400 points + 6 refinement rounds, RNG-free;
`calibrate_logits = z / T` — **argmax preserved exactly** for T > 0),
**`vector`** (per-class affine `scale·z + bias`, full-batch LBFGS seeded via
`torch.manual_seed(seed=0)`; fitted-width mismatch at apply time → `ValueError`;
may move argmax in pathological cases — documented, not clamped),
**`isotonic`** (top-label PAVA on `(confidence, correctness)` with tie grouping;
apply via piecewise-linear interpolation; `calibrate_logits` returns `log p'`
(log-odds for K=1); unfitted apply → `RuntimeError("call fit(...) first")`;
argmax preservation is **not** claimed).

`CalibrationEngine`:

```python
CalibrationEngine(method="none", **method_kwargs)
CalibrationEngine.from_config(config)          # reads config.calibration_method
engine.fit(logits, labels) -> self             # calibration split only
engine.apply(x, input_type=...)                # alias: engine.calibrate(...)
engine.evaluate(logits, labels, n_bins=15) -> {"nll","ece","mce","brier"}
engine.is_fitted -> bool ; engine.method -> str
engine.state_dict() / engine.load_state_dict(sd) -> self
engine.save(path) / CalibrationEngine.load(path)
repr -> "CalibrationEngine(method=..., fitted=...)"
```

**Invariant (fact + tests):** every registered method maps `[N,K] → [N,K]` with
identical dtype and class count — calibration changes values, never the schema or
the decision type (`tests/unit/test_calibration.py::test_calibration_preserves_schema`,
`::test_temperature_preserves_argmax`; `tests/unit/test_benchmark_smoke.py::test_temperature_preserves_accuracy_isotonic_may_not`;
`tests/numerical/test_theorem_verification.py::test_t8_calibration_then_project_preserves_membership`).

In the model, calibration runs **before** confidence and before projection, only
when `config.calibration_method != "none"` **and** `is_fitted`; applied decisions
carry `calibration_metadata = {"method", "applied": True}`.

---

## 12. Confidence (`jevx.confidence`)

```python
CONFIDENCE_REGISTRY
register_confidence_method(name)   # decorator; duplicate → ValueError
get_confidence_method(name)        # TypeError / ValueError (roster)
available_confidence_methods()     # ['energy', 'entropy', 'learned', 'margin', 'max_probability']
```

`ConfidenceEngine(nn.Module)`:

```python
ConfidenceEngine(method="max_probability", **method_kwargs)
ConfidenceEngine.from_config(config)                  # reads config.confidence_method
engine.fit(logits, option_mask=None) -> self          # no-op except "energy"
engine(probabilities, *, logits=None, option_mask=None, fused=None) -> [B]
engine.method_name / engine.method_class
```

Methods (all return `[B]`; single-column rows — Noul/Threshold with one sigmoid — compute
`c_max = max(p, 1 − p) ∈ [0.5, 1]` per spec Def. 9(e), so a threshold can fire; genuine
K = 1 *choice* rows where `p == 1` still yield 1): `max_probability` (`max_k p_k`),
`margin` (`p1 − p2`, width-1 → 1),
`entropy` (`1 − H(p)/log K_eff` with per-row valid counts; K_eff=1 → 1),
`energy` (`E = −logsumexp(logits)`; **before fit** returns raw unbounded energy
plus a `UserWarning` — `ValueError` if `logits=` omitted; **after** `fit` on a
calibration batch maps `(hi − E)/(hi − lo)` into [0,1], degenerate `hi == lo` →
0.5; masked options pushed to `finfo.min` first), `learned` (MLP over
`[fused, max_prob, H_norm, margin]` → sigmoid; `fused=` required → `ValueError`;
`fused_dim < 1` → `ValueError` at construction).

Validation: `probabilities` must be a finite 2-D float tensor (`TypeError`/
`ValueError` via `assert_shape`/`check_finite`); `option_mask` must be bool and
shape-identical (`TypeError`/`ValueError`). Fit state for `energy` is scalar
buffers `energy_lo`/`energy_hi` **and an explicit `_fitted` bool buffer** (`False` at
construction, set `True` by `fit`) — all three ride the model's regular `state_dict`.
This follows the calibrator convention (`BaseCalibrator._fitted`,
`jevx/calibration/base.py:149`): "never fitted" is a persisted VALUE rather than a magic
NaN, so a checkpoint that has been fit and one that has not are distinguishable without
inspecting the buffer payload. [FACT — `jevx/confidence/base.py:266`, `EnergyConfidence.is_fitted`]

**Backward compatibility (measured, not assumed).** A checkpoint written BEFORE `_fitted`
existed carries only `energy_lo`/`energy_hi`, whose `energy_lo` NaN sentinel was the old fit
marker. `EnergyConfidence._load_from_state_dict` handles that: when `<prefix>_fitted` is
absent it DERIVES the flag from that buffer (not-NaN -> fitted) and drops the synthetic miss
from `missing_keys`, so a **strict** load of a legacy state_dict succeeds unchanged.
Pinned by `tests/unit/test_confidence.py` (G13): fit state is a persisted bool, a
save+load round-trips the affine mapping bit-for-bit, a pre-`_fitted` two-buffer dict loads
unchanged, and a legacy dict whose `energy_lo` is still NaN loads as NOT fitted.

---

## 13. Risk–coverage utilities (§62)

```python
risk_coverage_curve(confidences, correct_mask) -> (coverage [n], risk [n])   # float32
coverage_at_risk(confidences, correct_mask, max_risk: float) -> float        # in [0, 1]
```

Both coerce array-likes via `torch.as_tensor`; require 1-D inputs of equal length
(`ValueError` naming the argument), finite confidences (`check_finite`), and
`max_risk ∈ [0, 1]` (`TypeError` for non-numeric/bool, `ValueError` out of
range). Coverage `γ_i = (i+1)/n` along a stable descending-confidence ranking
(ties broken by original order — deterministic); `n == 0` → two empty tensors /
`0.0`. Risk is *not* guaranteed monotone (documented non-claim, tested as such).

---

## 14. Training API (`jevx.training`)

### 14.1 Losses (pure functions over raw head logits)

```python
choice_loss(logits, target, option_mask=None, *, reduction="mean")    # masked CE
score_loss(logits, target, option_mask=None, *, mode="classification"|"expectation"|"hybrid",
           expectation_weight=1.0, reduction="mean")
noul_loss(logits, target, *, reduction="mean")                        # BCE-with-logits
multilabel_loss(logits, target, mask=None, *, reduction="mean")       # EXPERIMENTAL
```

`reduction ∈ {"mean","sum","none"}` — anything else → `ValueError`. Invalid
options are excluded by filling logits with `finfo(dtype).min` (never `-inf`), so
masked gradients are exactly 0 and extreme logits stay finite. Target validation:
non-integral floats rejected (`TypeError`), out-of-range or masked-position
targets → `ValueError`; `score_loss` unknown `mode` → `ValueError`.

### 14.2 Objective

```python
DecisionObjective(lambda_decision=1.0, lambda_calibration=0.0,
                  lambda_representation=0.0, lambda_regularization=0.0)
objective.compute(decision_loss, *, calibration_loss=None,
                  representation_loss=None, regularization_loss=None)
                  -> (total, components)
objective.lambdas  -> {"decision", "calibration", "representation", "regularization"}
l2_regularization(module, weight) -> Tensor      # weight * Σ ||p||²
```

Non-finite or negative lambdas → `ValueError`; non-numeric → `TypeError`;
`compute` without `decision_loss` → `ValueError`.

### 14.3 Freeze control

```python
freeze_module(m) / unfreeze_module(m) -> m        # TypeError on non-module
freeze_backbone(model) / unfreeze_backbone(model) -> model
unfreeze_last_n_layers(model, n) -> model          # invalid n → ValueError;
                                                   # no layer list → ValueError
count_parameters(module) -> {"total", "trainable", "frozen"}  (+ rejects non-modules)
```

`freeze_backbone` resolves the backbone by duck-typing (child module named
`model`, or the complement of `model.trainable_parameters()` when present) — it
never imports `modeling_jevx`.

### 14.4 Data & metrics

```python
DecisionDataset(records: list[dict])   # each record: state, questions (list[str]),
                                       # decision_type (str), options (list-of-schemas),
                                       # labels (list) — missing key → ValueError,
                                       # wrong container types → TypeError
decision_collate_fn(batch) -> {key: [per-sample values]}
make_supervised_step_fn(model, loss_fn=None) -> step(batch) -> (loss, components)
accuracy(logits, labels) -> float
macro_f1(logits, labels) -> float
confusion_matrix(logits, labels) -> Tensor [C, C]
extract_logits(output) -> Tensor                    # from JevXOutput / DecisionOutput / tensor
```

Labels absent from a batch → `ValueError` from the step fn; a custom `loss_fn`
must return a tensor (`TypeError`).

### 14.5 `HeadTrainer`

```python
HeadTrainer(model, optimizer=None, *, lr=1e-3, seed=42, device=None,
            grad_clip=1.0, objective=None)
trainer.fit(train_loader, *, epochs=1, eval_loader=None, loss_fn=None,
            callback=None) -> History          # callback receives each epoch record
trainer.evaluate(loader, metric_fns=None, *, objective_aware=False)
                                               # None → DEFAULT_METRICS {"accuracy"}
trainer.save_checkpoint(path, *, save_trainable_only=False, config=None)
                                               # {model_state, optimizer_state, history, config}
trainer.load_checkpoint(path, *, load_optimizer=True, strict=True)
trainer.generator               # seeded torch.Generator for DataLoader shuffling
```

Defaults: `AdamW` over `requires_grad` parameters, `torch.manual_seed(seed)`,
grad clipping at 1.0. Raises: `TypeError` (non-module model, bad objective/
grad_clip types), `ValueError` (no trainable parameters, `lr <= 0`, non-finite/
non-positive `grad_clip`, NaN loss reported with epoch/step context). `History`
records per-epoch dicts; `history.losses` (property) → list[float].

### 14.6 Staged curriculum (`jevx.training.curriculum`)

```python
CurriculumConfig(stage_a_epochs=4, stage_b_epochs=4,
                 lambda_start=0.0, lambda_end=1.0,
                 lambda_schedule="linear"|"cosine",
                 lr=1e-3, weight_decay=0.0, grad_clip=1.0, seed=42,
                 decision_type="choice")
CurriculumConfig.from_config(source=None, **overrides) -> CurriculumConfig
CurriculumConfig.to_dict() -> dict

interaction_scale(step_index, total_steps, *, lambda_start=0.0,
                  lambda_end=1.0, schedule="linear") -> float   # the pure ramp

run_curriculum(model, train_loader, eval_loaders, *,
               curriculum: CurriculumConfig, loss_fn=None) -> CurriculumHistory

CurriculumHistory(records=[], curriculum=None)
  .stage_a / .stage_b / .append / __len__ / __iter__ / __getitem__
  .to_dict() / .from_dict(data) / .save(path) -> Path / .load(path)
EpochRecord(stage, epoch, global_epoch, train_loss, interaction_scale,
            interaction_scale_first, steps, metrics={}, condition_options=False)
  .to_dict() / .from_dict(data)
read_interaction_scale(model) -> float
```

**Stage A** (`stage_a_epochs`) calls the model with `condition_options=False` on
every training step: the option/schema pathway is out of the graph, its parameters
keep `grad is None`, and `λ` is pinned to `0.0`. **Stage B** (`stage_b_epochs`)
re-enables it and ramps `λ` from `lambda_start` to `lambda_end` PER OPTIMIZER STEP.
One AdamW and one learning rate serve both stages — the run is a single
optimization, not two.

The ramp is a pure function of the step's position in Stage B, with
`step_index = epoch * steps_per_epoch + step` (both 0-based) and
`total_steps = stage_b_epochs * steps_per_epoch`; see
`MATHEMATICAL_SPECIFICATION.md` (12.a)–(12.b). The first Stage-B step always uses
`lambda_start`, the last always `lambda_end`, and `total_steps == 1` uses
`lambda_start`.

The per-call flag and the scale are written by the SAME object
(`_StageController`) once per epoch — so the epoch's evaluation inherits its
training flag — and again before every training step.
`EpochRecord.condition_options` records the flag each epoch ran under, read back
off the model view. A payload written before the flag existed defaults to
`True`, so an old record can never claim the pathway was closed.

**No-test-leakage guard.** `run_curriculum` refuses any `eval_loaders` entry whose
name (the mapping key, or a `split`/`split_name`/`name` attribute on the loader or
its dataset) contains `test`. There is no override.

`run_curriculum` raises `TypeError` (non-module model, non-`CurriculumConfig`,
non-callable `loss_fn`, a `forward` that cannot take `condition_options`, a
non-map-style loader) and `ValueError` (a test split in `eval_loaders`, an empty
loader, no trainable parameters, no conditioning head for Stage B, a batch whose
decision type disagrees with the config, a batch breaking the
one-question-per-record contract). `CurriculumConfig` validates every field
eagerly, naming the offending one.

### 14.7 Conditioning gate (`jevx.training.conditioning_gate`)

```python
ConditioningGateConfig(tau=None, rule="state_only_if_delta_nll_exceeds",
                       split_name="calibration", min_samples=32,
                       tie_tolerance=0.0, decision="pending",
                       delta_nll=None, tau_source=None, z=1.0,
                       decision_type="choice",
                       forbidden_split_tokens=LEAKY_SPLIT_TOKENS)
ConditioningGate(model=None, config=None, *, loss_fn=None)
  .measure(loader, *, max_batches=None) -> GateMeasurement
  .fit_threshold() -> float
  .decide(loader, *, max_batches=None) -> "state_only" | "conditioned"
  .condition_options_for(model=None) -> bool        # RuntimeError while "pending"
  .set_tau(tau: float | None) -> None               # pin/un-pin an external τ
  .to_dict() / .from_dict(payload, model=None) / .summary() -> str
  properties: .model .decision .delta_nll .tau

measure_modes(model, loader, *, loss_fn=None, decision_type="choice",
              split_name="calibration", max_batches=None) -> GateMeasurement
measure_delta_nll(...) -> float                 # just ΔNLL
fit_tau(per_batch_delta, *, z=1.0, min_batches=2) -> TauFit
assert_split_not_leakage(split_name, *, forbidden_tokens=LEAKY_SPLIT_TOKENS)
apply_gate(model, gate, *, sync_interaction_scale=False) -> bool

GateMeasurement(delta_nll, conditioned_nll, state_only_nll,
                conditioned_accuracy, state_only_accuracy,
                per_batch_delta, batch_sizes, num_samples, num_batches,
                decision_type)
TauFit(tau, standard_error, z, num_batches, source)
LEAKY_SPLIT_TOKENS = ("test","tests","testing","testset","testsplit",
                      "holdout","eval","evals","evaluation","final")
```

Protocol: the loader is drained ONCE and the batch list replayed for
`condition_options=True` then `False`, in `eval()` mode under `no_grad`, so both
passes see the same weights, batches, order and conditions and a repeated
`decide` is bit-identical. `ΔNLL = NLL_conditioned − NLL_state_only`, per example
and example-weighted over batches. `τ = z·SE` with
`SE = std(d_b, ddof=1)/sqrt(B)` over the CALIBRATION batches only; with fewer than
`min_batches` batches `SE = 0` and `tau_source` says `se_unestimable`.
Verdict: `state_only` iff `ΔNLL > τ + tie_tolerance`, else `conditioned` (a tie
keeps the shipped behaviour); below `min_samples` the gate refuses to switch and
records why in `tau_source`.

The verdict is **task level**, never per sample. The threshold is fitted on the
calibration split alone — no sampling, no search over any evaluation number — and
any split name carrying a whole-token test-like token is refused at config
construction, in `measure` and in `decide` (`contest` is not rejected; `RAVDESS_test` is).

`apply_gate` writes `model.config.conditioning_enabled` and returns the flag; an
explicit per-call `condition_options=` still wins. It is meant to be applied AFTER
the head is trained (Stage B): flipping it changes only the option pathway at
inference, and no learned weight is touched. `RuntimeError` while the gate is
still `pending`. `sync_interaction_scale=True` also sets `λ` to 1.0/0.0 so the two
mechanisms cannot disagree (`RuntimeError` on a model with no
`set_interaction_scale`). A `τ` this module FITTED is re-fit on the next decision
when a gate is restored from a dict; only a caller-PINNED `τ` (`set_tau`, or
`tau_source="explicit"`) stays authoritative.

---

## 15. Exception summary

| Exception | Raised when |
|---|---|
| `UnsupportedRepresentationError` | no extractable representation (structured fields, see §6.3) |
| `ValueError` | input-contract violations (tables above), unknown enum/registry names (message lists the roster), shape/finiteness failures, sidecar mismatch |
| `TypeError` | wrong Python types (positional `from_pretrained` args, non-tensor where tensor required, bool options, non-str registry names, float masks at the fusion boundary) |
| `RuntimeError` | ScoreHead expected-score escaping `[0, K−1]` (numerics guard); unfitted isotonic apply; `resolve_adapter` defensive unreachable path |
| `NotImplementedError` | abstract `DecisionHead.compute_logits` / confidence-method base (programming errors) |

Every documented raise site is asserted by tests **except the three
defensive by-construction paths** — the `ScoreHead` convex-hull
`RuntimeError` (unreachable for simplex inputs; carried by T1/T3 rather
than a forcing test), the `resolve_adapter` unreachable fallback
(`pragma: no cover`), and the abstract-base `NotImplementedError`s
(programming-error paths) — see
[FAILURE_MODES.md](FAILURE_MODES.md) for the failure-row → test mapping and
[TESTING.md](TESTING.md) for the suite taxonomy.

---

## 16. Weight conversion (`jevx.conversion`)

```python
from jevx import convert, ConversionMode, ConversionReport, EquivalenceKind
# full surface: from jevx.conversion import (transplant_classifier_head,
#   transplant_state_encoder_proj, probe_head_compat, check_base_equivalence,
#   check_representation_equivalence, check_decision_roundtrip,
#   check_functional_equivalence, derive_probe_inputs, CompatReport)

model, report = convert(model_or_id, *, decision_type=None,
                        equivalence_check=True, source_head=None,
                        probe_inputs=None, **from_pretrained_kwargs)
#   -> (JevXForDecision, ConversionReport)
```

`convert` wraps a checkpoint through the §1.2 factory (mode
`representation_reuse` by default: backbone weights reused verbatim, JevX
heads freshly initialized) and, with `equivalence_check=True`, runs the
executable equivalence probes and records their results in the report.

**Honesty invariants** (audit Q-001; enforced by
`tests/unit/test_conversion.py`, 21 tests):

| Object | Contract |
|---|---|
| `ConversionMode` | `representation_reuse` / `weight_transplant` / `head_attachment` / `architectural_conversion` (the last is a reserved constant — never auto-assigned today) |
| `EquivalenceKind` | `functional` / `behavioral` / `none` — a report **cannot claim an equivalence it did not measure**, and never claims a transplant that mapped nothing |
| `ConversionReport` | fields `mode, equivalence, source, target, mapped_params, unmapped_source, new_params, checks, warnings, jevx_version`; skipped checks are not passes |
| `transplant_classifier_head(source_head, target_head, *, mapping="auto")` | shape-checked graft of a source task head into an existing JevX head; returns a `weight_transplant` report (exact-shape match required, semantics re-checked — reported as transplant, never as identity) |
| `transplant_state_encoder_proj(source_proj_like, target, *, key_map=None)` | same discipline for state-encoder projections |
| `probe_head_compat(source_head, target_head) -> CompatReport` | read-only compatibility probe (`compatible` flag + `reasons`), no mutation |

**Equivalence checkers** (each returns a result dict; none is asserted
without being run): `check_base_equivalence(jevx_model, base_model, inputs,
*, atol=0.0)` (base-forward agreement), `check_representation_equivalence(jevx_model,
inputs, *, pool_fn=None, atol=None)` (pooled-representation agreement),
`check_decision_roundtrip(model, *args, **kwargs)` (decision stability
across a save/reload round trip), `check_functional_equivalence(jevx_model,
base_model=None, inputs=None, *, atol=0.0, tokenizer=None)` (aggregates the
available probes), and `derive_probe_inputs(model, *, tokenizer=None,
max_len=8, seed=0)` (builds probe inputs when the caller has none).

Semantics ground in [ARCHITECTURE.md](ARCHITECTURE.md) §5 (representation
reuse vs weight transplant); scope in
[UNIVERSAL_CONVERSION.md](UNIVERSAL_CONVERSION.md).
