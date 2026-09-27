# Phase 1 — Transformers v5.16.1 Source Audit

Scope: real source at `vendor/transformers` (tag v5.16.1, SHA 93c8b7b485963a10800c91f55304db6be211c2bd); installed package matches (verified: `transformers.__version__ == "5.16.1"`). Python 3.14, torch 2.14.0+cpu, CPU-only.
All `file:line` citations below refer to files under `vendor/transformers/src/transformers/`. Claims marked **[probe]** were executed in this environment (scripts: `vendor/reports/_probe_roundtrip.py`, `_probe_register.py`, `_probe_tied_wrapper.py`, `_probe_family_outputs.py`; results in this document and `vendor/reports/_probe_family_outputs.json` — all 20 audited families forward-verified, 20/20 OK). Claims marked **[HF-API]** were verified with `GET https://huggingface.co/api/models/<id>` (53 IDs returned 200, 4 returned HTTP 401 → not verified; raw map: `vendor/reports/_hf_verified_ids.json`).

Companion artifacts:
- `vendor/reports/architecture_inventory.json` — 709 records, built programmatically from `CONFIG_MAPPING` + 53 `MODEL_*_MAPPING_NAMES` (generator: `vendor/reports/_build_inventory.py`).
- Pre-existing `vendor/reports/auto_model_mapping.json` — keys `base` (540), `configs` (707), `tasks` (dict); generated earlier, not modified.
- `TRANSFORMERS_COMPATIBILITY_MATRIX.md` (repo root).

---

## 1. Core abstractions

### 1.1 PreTrainedConfig — `configuration_utils.py`
- Class: `configuration_utils.py:148` — now a **strict dataclass** (`@dataclass_transform`, `@strict(accept_kwargs=True)`, `@dataclass`); `__init_subclass__` re-applies `dataclass(cls, kw_only=True)` to every subclass and wraps classes **without** a custom `__init__` to accept arbitrary kwargs (`configuration_utils.py:359-372`). A custom `__init__` (common in remote code) bypasses the kwargs wrapper.
- `model_type: ClassVar[str] = ""` (`configuration_utils.py:262`); serialized by `to_dict` (`:1084-1085`) and used by `AutoConfig` dispatch.
- Common fields: `output_hidden_states=False`, `return_dict=True` (`:267-268`), `dtype: str | torch.dtype | None` (`:269`), `is_encoder_decoder` (`:271`), `architectures` (`:264`).
- `torch_dtype` is a **deprecated property** that warns and delegates to `dtype` (`:461-474`); `__post_init__` still accepts a `torch_dtype` kwarg for BC and converts str→`torch.dtype` (`:278-289`).
- Attribute aliasing via `attribute_map` in `__setattr__`/`__getattribute__` (`:476-484`).
- Round-trip APIs: `from_pretrained` (`:619`) → `get_config_dict` (`:730`) → `from_dict` (`:863`); `to_dict` (`:1076`), `to_diff_dict` (`:1011`, keeps nested configs' `model_type` at `:1043-1045`), `to_json_string(use_diff=True)` (`:1126`), `save_pretrained` (`:556`).
- `model_type` mismatch on load only warns (`:710-725`); `register_for_auto_class` (`:1255`) sets `_auto_class` so `save_pretrained` copies source + writes `auto_map` (`:595-596`).

### 1.2 PreTrainedModel — `modeling_utils.py`
- Class: `modeling_utils.py:1180`. Key class attrs: `config_class` (`:1203`, also auto-derived from a `config` type annotation via `__init_subclass__`, `:1305-1327`), `base_model_prefix=""` (`:1206`), `main_input_name="input_ids"` (`:1211`), `_no_split_modules` (`:1217`), `_tied_weights_keys: dict[str,str]` (**dict of `{target: source}`, not a list of regexes** — `:1228`), `_keys_to_ignore_on_load_missing/unexpected/save` (`:1230-1234`).
- `__init__(config, ...)` (`:1329`): requires a `PreTrainedConfig`, sets `self.config`, validates attention implementation, builds `generation_config` if `can_generate()` (`:1561`), records `loss_type`.
- `post_init()` (`:1376`) **must be called at the end of a subclass `__init__`**: aggregates child properties (children's `all_tied_weights_keys` with `name.` prefix, `_keep_in_fp32_modules`, `_no_split_modules`, `_keys_to_ignore_*` — `:1408-1423`), then `init_weights()` → `initialize_weights()` + `tie_weights(recompute_mapping=False)` (`:3174-3181`).
- `_init_weights` (`:2374`): normal(std from `initializer_range`/`init_std`/`initializer_factor`, default 0.02) for Linear/Conv/Embedding; ones for LayerNorm/RMSNorm (by class-name substring).
- `tie_weights` (`:2606`): symmetric tie logic with swap-on-missing during `from_pretrained`; performs `setattr(parent, name, source_param)`.
- `base_model` property = `getattr(self, self.base_model_prefix, self)` (`:1556-1558`).
- Embedding accessors: default `get_input_embeddings` (`:1081`) resolves by **name heuristic**: direct `embed_tokens` attr → `self.embeddings.embed_tokens` → `self.model.embed_tokens` → `self.language_model` → `base_model`; else `NotImplementedError`. `set_input_embeddings` analogous (`:1120`). **[probe]** a wrapper whose child module is named `backbone` raises `NotImplementedError`; child named `model` works only if the child exposes `embed_tokens` (Llama child: works; Bert child: `NotImplementedError` because BERT uses `embeddings.word_embeddings`). ⇒ JevX **must override** `get_input_embeddings`/`set_input_embeddings`.
- `save_pretrained` (`:3300`): writes `config.dtype` as str (`:3414`) and `config.architectures = [ClassName]` (`:3418`); `custom_object_save` for remote-code classes (`:3423`); drops `_keys_to_ignore_on_save` (`:3499-3504`); `remove_tied_weights_from_state_dict` (`:3513`, def `:437` — safetensors has no aliasing; undeclared shared tensors raise `RuntimeError` at `:508-511`); shards via `huggingface_hub.split_torch_state_dict_into_shards` (`:3530`, default `max_shard_size="50GB"`); writes each shard with `safe_save_file` (`:3605`) + `model.safetensors.index.json` (`:3620`). Quantized-but-not-serializable models raise (`:3364-3371`).
- `from_pretrained` (`:3881`): pops `dtype`/`torch_dtype` (`:4130-4131`; BC merge `:4159-4161`; default `dtype="auto"` `:4162`); resolves dtype via `_get_dtype` (`:815`, order: explicit → `config.dtype` → sharded metadata → first floating weight → fp32); model constructed **on meta device under `local_torch_dtype(dtype)`** (`get_init_context` `:3791-3820`, meta at `:3818`, dtype context `:3795`) then weights loaded by `_load_pretrained_model` (`:4436`) and finalized by `_finalize_model_loading` (`:4542`: move missing from meta → re-init → `tie_weights(missing_keys=...)` `:4565` → log report). Sets `model.eval()` (`:4395`). Multi-device/disk `device_map` triggers `accelerate_dispatch` (`:4412-4413`).
- `get_encoder()` helper (`:2275`) resolves `text_encoder`/`encoder` attributes (also `vision_tower`/`visual`/`vision_model` with `modality="image"`).
- Padding warning: `warn_if_padding_and_no_attention_mask` (`:4625`).

### 1.3 ModelOutput semantics in v5 — `utils/generic.py`, `modeling_outputs.py`
- Base class: `ModelOutput(OrderedDict)` at `utils/generic.py:415`; `to_tuple()` (`:548-551`) returns non-`None` fields in declaration order; `__getitem__` supports int/slice/str (`:521-526`).
- **`return_dict=False` is still honored** — two mechanisms:
  1. `can_return_tuple` decorator (`utils/generic.py:924-944`): reads `return_dict` from kwarg or `self.config.return_dict`, calls `output.to_tuple()` when False. Applied to head classes (e.g. `LlamaForCausalLM.forward` `models/llama/modeling_llama.py:436`, all `BertModel`-family heads).
  2. `capture_outputs` decorator (`utils/output_capturing.py:215-283`): pops `return_dict` at `:233` — **inner modules always return a `ModelOutput`**; conversion to tuple happens only at the decorated boundary (`:282`).
  3. Legacy models do it manually: DebertaV2 `if not return_dict: return (sequence_output,) + ...` (`models/deberta_v2/modeling_deberta_v2.py:793`).
- **[probe]** `return_dict=False` on `BertModel` → plain `tuple` (len 2: `last_hidden_state`, `pooler_output` — `None` fields skipped); on `LlamaModel` → `tuple`; `config.return_dict=False` also honored; default returns the dataclass (`BaseModelOutputWithPoolingAndCrossAttentions`, `BaseModelOutputWithPast`).
- Field definitions (`modeling_outputs.py`): `BaseModelOutput` (`:20`; fields `last_hidden_state`, `hidden_states`, `attentions`), `BaseModelOutputWithPooling` (`:69`, + `pooler_output`), `BaseModelOutputWithPast` (`:116`, + `past_key_values: Cache`), `BaseModelOutputWithPastAndCrossAttentions` (`:238`, + `cross_attentions`), `BaseModelOutputWithPoolingAndCrossAttentions` (`:192`), `Seq2SeqModelOutput` (`:452`; fields: `last_hidden_state` (decoder), `past_key_values: EncoderDecoderCache`, `decoder_hidden_states/attentions`, `cross_attentions`, **`encoder_last_hidden_state`, `encoder_hidden_states`, `encoder_attentions`**), `CausalLMOutputWithPast` (`:610`), `Wav2Vec2BaseModelOutput` (`:1297`; `last_hidden_state`, `extract_features`), `BaseModelOutputWithPoolingAndProjection` (`:1385`).
- `hidden_states` structure: tuple of length `num_layers + 1` — index 0 = embedding output, 1..L = post-layer outputs. **[probe]** 2-layer BERT/Llama → len 3; with default `tie_last_hidden_states=True` the last element is **overwritten with `last_hidden_state`** (`utils/output_capturing.py:271-281`) — verified `torch.equal(last_hidden_state, hidden_states[-1]) is True`. Vision/audio models disable the tie (e.g. ViT `@capture_outputs(tie_last_hidden_states=False)` `models/vit/modeling_vit.py:354`, AST `models/audio_spectrogram_transformer/modeling_audio_spectrogram_transformer.py:~275`).

### 1.4 Output capture (v5 mechanism for hidden states) — `utils/output_capturing.py`
- Models declare `_can_record_outputs = {"hidden_states": <DecoderLayerClass>, "attentions": <AttentionClass>}` (e.g. `models/bert/modeling_bert.py:540`, `models/llama/modeling_llama.py:340`).
- `capture_outputs` installs forward hooks **lazily once** (`maybe_install_capturing_hooks` `:196-213`, thread-safe) when a capture flag is first requested; an inactive hook (no collector) is an immediate no-op (`:108-110`). When active it appends references to already-computed tensors — **no recomputation**; cost is keeping L+1 activations alive (memory) plus per-layer Python hook calls.
- Base outputs are unchanged: **[probe]** with `output_hidden_states=True` the returned keys were `last_hidden_state`, `pooler_output`, `hidden_states`; toggling `output_hidden_states=False` later returned `hidden_states=None`; `config.output_hidden_states=True` works identically. `output_attentions=True` requires `attn_implementation="eager"` or it raises (`configuration_utils.py:406-415, 486-491`).
- `hidden_states` last-element tie (`utils/output_capturing.py:271-281`): with default `tie_last_hidden_states=True` the tuple's last element is replaced by `outputs.last_hidden_state` after the forward; vision/audio models pass `tie_last_hidden_states=False` and keep the pre-norm value.

---

## 2. Auto classes, registration, trust_remote_code — `models/auto/`, `dynamic_module_utils.py`

- `AutoConfig` (`models/auto/configuration_auto.py:278`): `from_pretrained` (`:303`) computes `has_remote_code = "auto_map" in config_dict and "AutoConfig" in config_dict["auto_map"]` (`:389`) vs `has_local_code = model_type in CONFIG_MAPPING`; remote path loads the class via `get_class_from_dynamic_module` (`dynamic_module_utils.py:516`) after `resolve_trust_remote_code` (`dynamic_module_utils.py:712`). `AutoConfig.register(model_type, config, exist_ok=False)` (`configuration_auto.py:439-455`) validates `config.model_type == model_type`.
- Model Auto classes share `_BaseAutoModelClass` (`models/auto/auto_factory.py:194`): `from_config` (`:206`), `from_pretrained` (`:261`), resolution `_get_model_class` (`:178`, uses `config.architectures` to pick among multiple models per config). `register(config_class, model_class, exist_ok)` (`:411-429`) validates `model_class.config_class.__name__ == config_class.__name__`. 47 `AutoModelFor*` classes exist (`models/auto/modeling_auto.py:2178-2506`), including `AutoModel` (`:2198`) and **`AutoModelForTextEncoding`** (`:2190`, mapping `:1954` → bert/roberta/deberta-v2/albert/distilbert base models + `T5EncoderModel`).
- Mappings are `_LazyAutoMapping` (`auto_factory.py:575`) over `MODEL_MAPPING_NAMES` (`modeling_auto.py:41`) + per-task `MODEL_FOR_*_MAPPING_NAMES`. `.register` on the lazy mapping **skips keys whose `__module__` starts with `transformers.`** (`auto_factory.py:676-684`) — i.e. binding a *native* config class to a custom model is silently ignored; a **custom config class** registers fine.
- `trust_remote_code` flow: `auto_map` entries like `"AutoModel": "modeling_jevx.JevXModel"` (+ optional `repo--` prefix for upstream repos) → `get_class_from_dynamic_module` downloads/copies module into the HF module cache → class registered on the fly (`auto_factory.py:376-383` in `from_pretrained`, `:227-230` in `from_config`: `cls.register(... exist_ok=True)` + `register_for_auto_class` + `add_generation_mixin_to_remote_model` `:543`). `resolve_trust_remote_code` (`dynamic_module_utils.py:712-790`): if `None`, prompts on POSIX via `SIGALRM` (`:764-773`); **on Windows (no `SIGALRM`) it raises and demands an explicit `trust_remote_code=True`** (`:774-779`).
- `custom_object_save` (`dynamic_module_utils.py:626-699`): copies the defining `.py` + relative imports into the save dir and writes `auto_map[obj._auto_class] = "module.Class"`; **refuses classes defined in `__main__`** with a warning (`:640-648`) — **[probe]** `auto_map` stayed `null` for a `__main__`-defined class (warning text observed).
- `PreTrainedConfig.register_for_auto_class` (`configuration_utils.py:1255`), `PreTrainedModel.register_for_auto_class` (`modeling_utils.py:4604`).

---

## 3. Serialization details JevX needs

- **config.json contents** (probe of a wrapper save): `model_type`, `architectures=[<class name>]` (auto-written), `dtype="float32"` (**key is `dtype`, not `torch_dtype`**), `transformers_version`, plus a nested dict for the inner backbone config that **preserves its own `model_type`** (probe: `backbone.model_type == "bert"`; kept because `to_diff_dict` retains nested `model_type`, `configuration_utils.py:1043-1045`). `auto_map` only appears if `register_for_auto_class` was used *and* the class is not in `__main__`.
- **State-dict key prefixes** are plain module attribute paths — a child named `backbone` yields `backbone.embeddings.word_embeddings.weight`, etc. No name rewriting on save.
- **Tied weights**: `remove_tied_weights_from_state_dict` (`modeling_utils.py:437`) groups tensors by storage pointer, deletes aliases covered by `_tied_weights_keys` (aggregated with child prefix by `post_init` `:1411-1414`), clones harmless disjoint sharing, and **raises `RuntimeError` for undeclared shared tensors** (`:508-511`). **[probe]** wrapper holding `LlamaForCausalLM(tie_word_embeddings=True)` saved fine (file contains `embed_tokens` but **not** `lm_head`), reloaded with the tie restored (`tie_weights` at load `:4565`), weights equal, forward OK. Undeclared alias (`self.alias = self.b.embeddings.word_embeddings`) → exact `RuntimeError: The weights trying to be saved contained shared tensors [{'alias.weight', 'b.embeddings.word_embeddings.weight'}] ...`.
- **Sharding**: `split_torch_state_dict_into_shards` (huggingface_hub) at `modeling_utils.py:3530`; index `model.safetensors.index.json` (`:3620`); load resolves shards via `_get_resolved_checkpoint_files` (`:534`) and streams slices through `safe_open` without materializing (`:4504-4520` region of `_load_pretrained_model`).
- `.bin` fallback still loads (`torch.load(..., weights_only=True)` default, `from_pretrained(weights_only=True)` `:3893`).

---

## 4. Processors / tokenizers interface contract

- **`ProcessorMixin`** (`processing_utils.py:601`): unified `__call__(images=None, text=None, videos=None, audio=None, **kwargs) -> BatchFeature` (`:656-716`); sub-objects (`tokenizer`, `image_processor`, `feature_extractor`, `video_processor`) validated by class in `__init__` (`:638-654`). `from_pretrained` (`:1687`) = load `processor_config.json`/`preprocessor_config.json` + each sub-processor, then `from_args_and_dict` (`:1460`). `save_pretrained` writes `processor_class` (`:1080`).
- **`AutoProcessor`** (`models/auto/processing_auto.py:116`) resolution order (`:200-330`): (1) `processor_config.json` → `processor_class` / `auto_map.AutoProcessor`; (2) `preprocessor_config.json` (image) → same; (3) video config; (4) feature-extractor config; (5) model `config.json` → `processor_class`/`auto_map` (`:284-297`); (6) `PROCESSOR_MAPPING[type(config)]` (`:307-309`); remote code via `get_class_from_dynamic_module` (`:321-327`). `AutoProcessor.register(config_class, processor_class)` (`:354`).
- **`AutoTokenizer`** (`models/auto/tokenization_auto.py:639`, `from_pretrained` `:655`) resolution: explicit `tokenizer_type` kwarg (`:740,751`) → `tokenizer_config.json` `tokenizer_class`/`auto_map` (`:782-793`) → registered-class corrections / hub-class overrides (`:796-863`) → remote `auto_map` + `trust_remote_code` (`:905-917`) → `tokenizer_config_class` (`:919`) → `config.tokenizer_class` (`:931`) → `TOKENIZER_MAPPING[config]` with **`TokenizersBackend` fallback** (`:952`, `:963-975`) → error. `AutoTokenizer.register` (`:981`).
- `PreTrainedTokenizerBase` (`tokenization_utils_base.py:962`), `from_pretrained` (`:1489`), `register_for_auto_class` (`:2970`).
- Feature extraction / images: `FeatureExtractionMixin` (`feature_extraction_utils.py:266`, `from_pretrained` `:287`), `AutoFeatureExtractor` (`models/auto/feature_extraction_auto.py:217`), `ImageProcessingMixin` (`image_processing_base.py:61`), `BaseImageProcessor` (`image_processing_utils.py:60`).
- **Mapping facts for audited families** (from `models/auto/auto_mappings.py`, verified by import): `PROCESSOR_MAPPING_NAMES` contains `clip` (`:1067`), `siglip` (`:1172`), `wav2vec2` (`:1190`), `whisper` (`:1192`), `llava_next` (`:1121`) — **no `audio-spectrogram-transformer`** (AST is in `FEATURE_EXTRACTOR_MAPPING_NAMES` → `ASTFeatureExtractor`, `:1022`, so AST uses `AutoFeatureExtractor`); text families (bert/t5/bart/…) are absent from `PROCESSOR_MAPPING` → use `AutoTokenizer`. `TOKENIZER_MAPPING_NAMES`: `llama→None`, `siglip→None`, `vit→None`, `mistral/phi3/llava_next→TokenizersBackend` (probe) — these resolve through `tokenizer_config.json`/`config.tokenizer_class` fallbacks.

---

## 5. Cache / masking for hidden-state extraction (no generation)

- Mask construction is centralized: `create_causal_mask` (`masking_utils.py:864`), `create_bidirectional_mask` (`:999`), sliding-window variants (`:1102`); padding is folded into the mask (`prepare_padding_mask` `:205`, `padding_mask_function` `:168`). Models call these in `forward` (e.g. `models/bert/modeling_bert.py:697-711`, `models/llama/modeling_llama.py:397-402`).
- `use_cache`: config default is **`True`** for decoder-only (`models/llama/configuration_llama.py:74`, `models/gpt2/configuration_gpt2.py:96`); `LlamaModel.forward` allocates `DynamicCache` when `use_cache and past_key_values is None` (`models/llama/modeling_llama.py:383-384`). `merge_with_config_defaults` (`utils/generic.py:1000-1040`) fills `use_cache` from config and forces `False` under training + gradient checkpointing. **Recommendation for JevX extraction: pass `use_cache=False` explicitly** to avoid cache allocation; output then has `past_key_values=None`.
- Cache classes: `Cache` (`cache_utils.py:1269`), `DynamicCache` (`:1737`), `EncoderDecoderCache` (`:1947`).
- Padding: right-padding + `attention_mask` is correct for encoder-only and for decoder-only **mean/CLS pooling** (mask excludes pads); for **last-token** pooling on decoder-only use left padding or gather at `attention_mask`-derived last-valid index (positions come from `position_ids` arithmetic, unaffected for mean pooling). `warn_if_padding_and_no_attention_mask` (`modeling_utils.py:4625`) warns if pads present without a mask.
- `GenerationMixin` is no longer inherited by `PreTrainedModel` (`can_generate` detection, `modeling_utils.py:1561-1591`); irrelevant for JevX (no generation).

---

## 6. Composition: wrapper `PreTrainedModel` holding another HF model

Verified by probe (Bert-wrapper and tied-Llama-wrapper round-trips): see §3 and Q1 below. Summary of mechanics:
- Child models are ordinary submodules; `state_dict()` prefixes with the child attribute name; `save_pretrained`/`from_pretrained` work unchanged (**[probe]** weights bitwise-equal after reload for both wrappers).
- `post_init()` on the wrapper aggregates the child's tied-weight mapping with prefixes (e.g. `{'inner.lm_head.weight': 'inner.model.embed_tokens.weight'}` — **[probe]** observed) and its `_no_split_modules`/ignore-key sets (`modeling_utils.py:1408-1423`), so `tie_weights`/device_map logic sees the whole tree.
- Pitfalls: (a) undeclared shared tensors → `RuntimeError` at save; (b) `get_input_embeddings`/`set_input_embeddings` heuristics fail for non-`embed_tokens` children (see §1.2) — must override; (c) wrapper inherits `base_model_prefix=""` default — set it or override `get_input_embeddings`; (d) inner config must be serialized into `JevXConfig` (nested dict) so `from_pretrained` can rebuild the child (probe did this via `BertConfig.from_dict(config.backbone)`); (e) `dtype` casting covers the whole tree because construction happens under `local_torch_dtype` (**[probe]** `dtype=bf16` → wrapper head **and** child in bfloat16).

---

## 7. Answers to the implementation questions

### Q1 — Can a custom `PreTrainedModel` hold a loaded HF model as a child and round-trip?
**Yes.** **[probe]** Two wrappers round-tripped: (a) `JevXModel(BertModel child + Linear head)` → `save_pretrained` produced `config.json` + `model.safetensors`; `from_pretrained` restored bitwise-equal weights; (b) wrapper around `LlamaForCausalLM(tie_word_embeddings=True)` → saved (lm_head correctly dropped from the safetensors file as a declared tie), reloaded with the tie restored, forward OK.
Must declare:
- `JevXConfig(PreTrainedConfig)` with a **unique `model_type`** and the inner config stored as a nested dict (probe: `backbone: {…, "model_type": "bert"}` preserved on save).
- `JevXModel(PreTrainedModel)` with `config_class = JevXConfig` (or a `config: JevXConfig` annotation), call `self.post_init()` last in `__init__` (`modeling_utils.py:1376`).
- Override `get_input_embeddings`/`set_input_embeddings` (**[probe]** default heuristics raise `NotImplementedError` for a child named `backbone`, and even for `model`-named BERT children).
- For Auto-class loading: `AutoConfig.register(model_type, JevXConfig)` + `AutoModel.register(JevXConfig, JevXModel)` (**[probe]** both succeed and `AutoModel.from_pretrained(dir)` then returns the wrapper; **without** registration `AutoModel.from_pretrained` raises `ValueError: … model type 'jevx_tiny' … does not recognize this architecture`), or ship `auto_map` via `register_for_auto_class` (persists in config.json for Hub/remote-code consumers; requires a real module, not `__main__` — `dynamic_module_utils.py:640`).
v5 pitfalls: safetensors **shared-tensor `RuntimeError`** for undeclared aliases (`modeling_utils.py:508-511`); ties must be declared in `_tied_weights_keys` **or** be exactly the child's keys so `post_init` aggregation covers them (it does, with prefix — probe confirmed); `device_map` is only accelerate-dispatched when >1 device or disk (`:4412`) — on this CPU box just `.to(device)`; `dtype` handling is automatic through `from_pretrained(dtype=…)` (probe: whole tree bf16); quantized children require a serializable hf_quantizer or save raises (`:3364-3371`).

### Q2 — Does `forward()` still honor `return_dict=False` / tuple outputs?
**Yes.** Default returns a `ModelOutput` dataclass; `return_dict=False` (kwarg or `config.return_dict=False`) returns a plain `tuple` of the non-`None` fields. Mechanisms: `can_return_tuple` decorator (`utils/generic.py:924-944`), `capture_outputs` (`utils/output_capturing.py:233` pops the kwarg so *inner* modules always return `ModelOutput`, conversion at `:282`), and manual legacy branches (DebertaV2 `modeling_deberta_v2.py:793`). **[probe]** verified on BERT (tuple len 2), Llama (tuple), and config-level `return_dict=False`.

### Q3 — Official way to expose AutoJevX-style classes?
All three exist; choose by distribution channel:
1. **`auto_map` in config.json + `trust_remote_code=True`** — the Hub/remote-code mechanism: `AutoConfig.from_pretrained` checks `config_dict["auto_map"]["AutoConfig"]` (`configuration_auto.py:389`), model Auto classes check `config.auto_map[cls.__name__]` (`auto_factory.py:208/355`); classes loaded via `get_class_from_dynamic_module` (`dynamic_module_utils.py:516`). `auto_map` is written automatically when saving a class registered with `register_for_auto_class` (`configuration_utils.py:595`, `modeling_utils.py:3423`, writer `dynamic_module_utils.py:626-699`; **not** for `__main__` classes). On Windows, `trust_remote_code` must be passed explicitly (`dynamic_module_utils.py:774-779`).
2. **`AutoConfig.register("jevx", JevXConfig)` + `AutoModel.register(JevXConfig, JevXModel)`** (`configuration_auto.py:439`, `auto_factory.py:411`) — in-process/session registration; **[probe]** works for a local save dir (both Auto classes return JevX classes). Caveat: `_LazyAutoMapping.register` ignores registrations whose *config* class lives in `transformers.*` (`auto_factory.py:676-684`) — JevX's own config class is fine.
3. **Subclassing `AutoModel` itself is not the mechanism** — Auto classes raise on direct instantiation (`auto_factory.py:197-203`) and are factories; you register *onto* them. `PreTrainedModel.register_for_auto_class` (`modeling_utils.py:4604`) pairs with (1).
Recommended for JevX: (2) for local/library use, plus (1) for Hub checkpoints.

### Q4 — Reading hidden states from a frozen backbone: is `output_hidden_states=True` sufficient?
**Yes.** Pass `output_hidden_states=True` per-call or set `config.output_hidden_states=True` (**[probe]** both work). It returns `hidden_states` as a tuple of length `num_layers+1` (embedding output + each layer); by default `hidden_states[-1] is last_hidden_state` (post-final-norm tie, `utils/output_capturing.py:275-281`; disabled in vision/audio models — then `[-1]` is pre-norm). Base outputs are unchanged (same fields, `hidden_states` added). Perf: hooks are installed lazily once and are no-ops when inactive (`utils/output_capturing.py:107-111`); when active there is **no recomputation** — only references to already-computed tensors are stored (memory ≈ keeping L+1 activation tensors alive; a few Python hook calls per layer). With a frozen backbone run under `torch.no_grad()`, memory is the dominant cost; for a single final representation prefer `last_hidden_state`/`pooler_output` and skip `output_hidden_states` entirely. No behavioral change to other outputs was observed. (Avoid `output_attentions=True` on non-eager attention — it raises, `configuration_utils.py:406-415`.)

### Q5 — dtype/device conventions in v5?
- `from_pretrained(..., dtype=...)` is the canonical kwarg (`modeling_utils.py:4130`); **`torch_dtype` still accepted for BC** and merges as fallback (`:4131, 4159-4161`) — **[probe]** both `dtype=torch.bfloat16` and `torch_dtype=torch.bfloat16` loaded bf16. Default is `"auto"` (`:4162`) → `config.dtype` → sharded-metadata dtype → first floating weight → fp32 (`_get_dtype` `:815-895`).
- Config field is **`dtype`** (`configuration_utils.py:269`); `config.torch_dtype` is a deprecated property that warns (`:461-474`). Saved config.json contains `"dtype": "float32"` and **no `torch_dtype` key** (**[probe]**). `save_pretrained` writes `config.dtype` from actual weights (`modeling_utils.py:3414`).
- Devices: model params are constructed under `local_torch_dtype(dtype)` on meta device then materialized (`:3791-3820`) — **a wrapper's own head parameters automatically match the requested dtype** (**[probe]** wrapper `head` and child both bfloat16 after `from_pretrained(..., dtype=torch.bfloat16)`). `device_map` only matters with multiple devices/disk (`:4412-4413`); on this CPU machine use plain `.to("cpu")` / default. If JevX creates its head *outside* `from_pretrained` (e.g. `from_config` path), `_from_config` also applies `local_torch_dtype` (`:1489-1512`) — otherwise construct under `torch.set_default_dtype` or cast explicitly.

### Q6 — Encoder-decoder (`t5`/`bart`): what does `forward()` return; can we call `.encoder` directly?
`T5Model.forward` returns **`Seq2SeqModelOutput`** (`models/t5/modeling_t5.py:884-892`; class `modeling_outputs.py:452`) with `encoder_last_hidden_state` / `encoder_hidden_states` / `encoder_attentions` alongside decoder-side fields; `BartModel` likewise (`models/bart/modeling_bart.py:784-791`). The `encoder_outputs` input field accepts either a `BaseModelOutput` or a legacy tuple (auto-wrapped, `models/t5/modeling_t5.py:860-866`).
**Calling the encoder directly is supported and conventional**: `T5Model` itself invokes `self.encoder(...)` internally (`:853-858`); `model.encoder(input_ids=…)` returns `BaseModelOutputWithPastAndCrossAttentions` with `last_hidden_state` (**[probe]**). Equivalence with the full model's `encoder_last_hidden_state` holds in eval mode (**[probe]** `torch.equal → True` after `model.eval()`; differs in train mode due to dropout). There is also the generic `PreTrainedModel.get_encoder()` (`modeling_utils.py:2275`) and, for T5, a dedicated `T5EncoderModel` (`models/t5/modeling_t5.py:1073`, mapped under `AutoModelForTextEncoding`).

### Q7 — Pooling utilities vs per-model poolers?
**transformers core ships no generic pooling module** — `dir(transformers)` exposes no `Pooler`/`Pooling` utility (**[probe]**; only unrelated `PoolFormer*` classes; `sentence_transformers` not installed). Pooling exists only as per-model modules/inline math:
| family | ships a pooler? | where |
|---|---|---|
| bert | yes — `BertPooler` (Linear+Tanh on CLS) | `models/bert/modeling_bert.py:451`, built when `add_pooling_layer=True` `:609`, output `pooler_output` `:682` |
| roberta | yes — `RobertaPooler` | `models/roberta/modeling_roberta.py:524/566`, output `:639-644` |
| albert | yes — inline `nn.Linear`+Tanh on CLS | `models/albert/modeling_albert.py:352-353`, output `:405` |
| deberta_v2 | **no** on base model (`ContextPooler` only inside `ForSequenceClassification`) | base returns `BaseModelOutput` only, `models/deberta_v2/modeling_deberta_v2.py:795`; ContextPooler `:981`, used `:1017/:1069`. **[probe]** `hasattr(DebertaV2Model, 'pooler') → False` |
| distilbert | **no** | returns `BaseModelOutput`, `models/distilbert/modeling_distilbert.py:282/385`. **[probe]** False |
| llama/qwen2/mistral/gemma/phi3 | **no** | returns `BaseModelOutputWithPast` (`llama:414`, `qwen2:399`, `mistral:387`, `gemma:436`, `phi3:418`). **[probe]** False |
| gpt2 | **no** | returns `BaseModelOutputWithPastAndCrossAttentions`, `models/gpt2/modeling_gpt2.py:624`. **[probe]** False |
| t5/bart/whisper | no pooler — use `encoder_last_hidden_state` | §Q6 |
| clip | text: EOS-position gather; vision: CLS + `post_layernorm`; projections in `get_text/image_features` | `models/clip/modeling_clip.py:559-570, 641-642, 674, 710` |
| siglip | text: last-token `[:, -1, :]` + Linear head; vision: `SiglipMultiheadAttentionPoolingHead` | `models/siglip/modeling_siglip.py:537-544, 614, 622` |
| vit | yes — `ViTPooler` (CLS+Linear+act) | `models/vit/modeling_vit.py:289-301`, output `:388` |
| wav2vec2 | no in base; masked mean pooling only inside `ForSequenceClassification` | `models/wav2vec2/modeling_wav2vec2.py:1730-1735`; base output `:1304` |
| ast | inline `(hidden[0]+hidden[1])/2` returned as `pooler_output` | `models/audio_spectrogram_transformer/modeling_audio_spectrogram_transformer.py:304-306` |
| llava_next | no pooler — `LlavaNextModelOutputWithPast.last_hidden_state` | `models/llava_next/modeling_llava_next.py:154` |

⇒ JevX needs its own mask-aware pooling policy for: distilbert, deberta-v2, all decoder-only, wav2vec2, llava_next, and (optionally) encoder-decoder encoder states.

### Q8 — ProcessorMixin / AutoProcessor resolution; minimal robust generic input prep?
Resolution order is documented in §4 (AutoProcessor: processor_config → preprocessor_config → model config → `PROCESSOR_MAPPING` → remote code; AutoTokenizer: tokenizer_config → registered class → remote code → `TOKENIZER_MAPPING`/`TokenizersBackend` fallback).
Minimal robust pattern **[probe]**: `AutoProcessor.from_pretrained(id)` then `processor(images=…, text=…, audio=…, return_tensors="pt")` — `ProcessorMixin.__call__` signature is exactly `(images, text, videos, audio, **kwargs)` (verified via `inspect`), returning a `BatchFeature` you splat into `model(**batch)`. Text-only families: `AutoTokenizer.from_pretrained(id)(texts, return_tensors="pt", padding=True)` — no processor exists for them. AST-like audio: `AutoFeatureExtractor.from_pretrained(id)` (AST absent from `PROCESSOR_MAPPING`). Vision-only (ViT): image processor from `IMAGE_PROCESSOR_MAPPING` (`ViTImageProcessor…`, `auto_mappings.py:1320`). Fall back per-modality if `AutoProcessor` raises (some repos ship only feature extractors).

---

## 8. Risks / gotchas (implementation-relevant)

1. **Safetensors shared-tensor `RuntimeError`** on save for any undeclared aliasing (`modeling_utils.py:508-511`) — never assign the same `Parameter` object under two names unless it flows through a declared child tie.
2. **`get_input_embeddings`/`set_input_embeddings` must be overridden** in the wrapper (default heuristics only understand `embed_tokens`/`embeddings`/`model`/`language_model` naming — **[probe]** NotImplementedError otherwise). Needed by `resize_token_embeddings` and PEFT-style flows.
3. **`AutoConfig`/`AutoModel` refuse unregistered `model_type`s** with `ValueError` (**[probe]**) — registration is process-local; `auto_map`+`trust_remote_code` is the persistent route, and `auto_map` is skipped for `__main__`-defined classes (`dynamic_module_utils.py:640`).
4. **Windows `trust_remote_code`**: interactive prompt path is unavailable (no `SIGALRM`) → explicit `True` required (`dynamic_module_utils.py:774-779`).
5. **`torch_dtype` → `dtype` migration**: config.json now stores `dtype`; reading old checkpoints still works (BC in `__post_init__`), but code touching `config.torch_dtype` emits deprecation warnings (`configuration_utils.py:461-474`).
6. **`output_attentions=True` forces eager attention or raises** (`configuration_utils.py:406-415`) — avoid; JevX doesn't need attentions.
7. **Decoder-only `use_cache` defaults to `True`** — pass `use_cache=False` for extraction to skip `DynamicCache` allocation (`models/llama/modeling_llama.py:383`).
8. **Last-element tie of `hidden_states`** differs between LM-like (tied to `last_hidden_state`) and vision/audio (pre-norm) models (`utils/output_capturing.py:275-281`) — don't assume `hidden_states[-1] == last_hidden_state` universally.
9. **`PreTrainedConfig` is a strict kw-only dataclass** — unknown kwargs to `__init__` of a custom config with a custom `__init__` behave differently than the BC wrapper path (`configuration_utils.py:359-372`); keep `JevXConfig.__init__` either absent (let the wrapper accept extras) or explicitly consuming kwargs.
10. **AST has no `AutoProcessor` entry** — use `AutoFeatureExtractor` (`auto_mappings.py:1022`); llama/siglip/vit have no `TOKENIZER_MAPPING` entry — `AutoTokenizer` relies on `tokenizer_config.json` fallbacks.
11. **CPU-only / Python 3.14**: `torch.jit.script` FutureWarning noise observed at import/run time (torch 2.14 + py3.14); harmless but pollutes stderr.

## 9. v5 vs v4 gotchas

| Area | v4 (typical) | v5.16.1 (this source) |
|---|---|---|
| Attention/masking | `src/transformers/attention/` package (in some 4.x) / `modeling_attn_mask_utils` | no `attention/` dir; masks built by `masking_utils.create_causal_mask/create_bidirectional_mask` (`masking_utils.py:864/999`); backend via `config._attn_implementation` property (`configuration_utils.py:417-437`) |
| dtype kwarg | `torch_dtype` everywhere | `dtype` canonical (`modeling_utils.py:4130`); `torch_dtype` BC + deprecated property (`configuration_utils.py:461-474`); config.json key `dtype` (`modeling_utils.py:3414`) |
| Config base | plain class | **strict dataclass** w/ `kw_only` subclassing + BC kwargs wrapper (`configuration_utils.py:145-147, 359-372`) |
| Hidden-state collection | manual tuple append in layer loops | **forward-hook capture**: `_can_record_outputs` + `capture_outputs` decorator (`utils/output_capturing.py:215`); `hidden_states[-1]` tied to `last_hidden_state` by default (`:275-281`) |
| `GenerationMixin` | inherited from `PreTrainedModel` | **not inherited**; `can_generate()` detection + auto-mixin for remote code (`modeling_utils.py:1561`, `auto_factory.py:543`) |
| `_tied_weights_keys` | list of regex strings | **dict `{target: source}`** (`modeling_utils.py:1226-1228`); `tie_weights` is symmetric with swap-on-missing (`:2606-2704`) |
| Save flow | `save_pretrained` + manual shard split | tied-weight removal by storage pointer before safetensors write (`:437/:3513`), `huggingface_hub.split_torch_state_dict_into_shards` (`:3530`), auto-written `architectures` + `dtype` (`:3414/:3418`) |
| Load flow | `_load_state_dict_into_model` | meta-device construction under `local_torch_dtype` (`:3791-3820`), `_load_pretrained_model` + `_finalize_model_loading` (`:4436/:4542`), `weights_only=True` default (`:3893`) |
| Auto registration | `AutoX.register`, `auto_map` | same, **plus** `_LazyAutoMapping.register` silently skips `transformers.*`-native config keys (`auto_factory.py:676-684`); 47 `AutoModelFor*` incl. `AutoModelForTextEncoding` (`modeling_auto.py:2190`) |
| Processors | separate `__call__` signatures | unified `ProcessorMixin.__call__(images, text, videos, audio)` (`processing_utils.py:656`); `TokenizersBackend` is the tokenizer fallback backend (`tokenization_auto.py:952/975`) |
| `use_cache` in forwards | often `None`→config | `merge_with_config_defaults` decorator injects config defaults incl. `use_cache` (`utils/generic.py:1000`) |
| return_dict handling | per-forward boilerplate | decorators (`can_return_tuple`/`capture_outputs`) pop `return_dict` at the boundary; inner modules always return `ModelOutput` (`utils/output_capturing.py:233`) |
| No `src/transformers/attention/` | — | confirmed: not present at this tag (task premise verified by directory listing) |
