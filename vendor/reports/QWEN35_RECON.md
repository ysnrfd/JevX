# Qwen3.5-0.8B Checkpoint + transformers v5.16.1 Hybrid Recon (FACTS ONLY)

Recon date: 2026-09-24. No code changes made.

## Checkpoint location

- ABSOLUTE PATH: `C:\Users\ysnrfd\.cache\huggingface\hub\models--Qwen--Qwen3.5-0.8B`
- Found at first candidate location; `HF_HOME` unset (default cache used).
- `refs/main` commit sha: `2fc06364715b967f1860aea9cf38778875588b17`
- Snapshot dir: `C:\Users\ysnrfd\.cache\huggingface\hub\models--Qwen--Qwen3.5-0.8B\snapshots\2fc06364715b967f1860aea9cf38778875588b17`
- Same hub also contains `models--Qwen--Qwen3.5-27B` and `models--Qwen--Qwen3.8-27B` (not inspected beyond listing).

## Snapshot files (all are symlinks into ..\..\blobs\; sizes = resolved sizes)

| file | bytes |
|---|---|
| chat_template.jinja | 7,755 |
| config.json | 2,907 |
| merges.txt | 3,353,259 |
| model.safetensors-00001-of-00001.safetensors | 1,746,942,600 |
| model.safetensors.index.json | 50,900 (metadata.total_size = 1,746,882,752) |
| preprocessor_config.json | 390 |
| tokenizer.json | 12,807,982 |
| tokenizer_config.json | 16,709 |
| video_preprocessor_config.json | 385 |
| vocab.json | 6,722,759 |

- Weight total: 1,746,942,600 bytes = 1.627 GiB (single shard, 473 tensors; index metadata total_size 1,746,882,752).
- NO `*.py` remote-code files in the snapshot (no trust_remote_code assets present).
- Weight names include `model.language_model.layers.N.linear_attn.in_proj_qkv.weight` etc. (multimodal wrapper `model.visual.*` + `model.language_model.*`).

## Config verbatim (config.json, 2907 bytes — full)

```json
{
    "architectures": [
        "Qwen3_5ForConditionalGeneration"
    ],
    "image_token_id": 248056,
    "model_type": "qwen3_5",
    "text_config": {
        "attention_bias": false,
        "attention_dropout": 0.0,
        "attn_output_gate": true,
        "dtype": "bfloat16",
        "eos_token_id": 248044,
        "full_attention_interval": 4,
        "head_dim": 256,
        "hidden_act": "silu",
        "hidden_size": 1024,
        "initializer_range": 0.02,
        "intermediate_size": 3584,
        "layer_types": [
            "linear_attention", "linear_attention", "linear_attention", "full_attention",
            "linear_attention", "linear_attention", "linear_attention", "full_attention",
            "linear_attention", "linear_attention", "linear_attention", "full_attention",
            "linear_attention", "linear_attention", "linear_attention", "full_attention",
            "linear_attention", "linear_attention", "linear_attention", "full_attention",
            "linear_attention", "linear_attention", "linear_attention", "full_attention"
        ],
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 16,
        "linear_value_head_dim": 128,
        "max_position_embeddings": 262144,
        "mlp_only_layers": [],
        "model_type": "qwen3_5_text",
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": false,
        "num_attention_heads": 8,
        "num_hidden_layers": 24,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-06,
        "tie_word_embeddings": true,
        "use_cache": true,
        "vocab_size": 248320,
        "mamba_ssm_dtype": "float32",
        "rope_parameters": {
            "mrope_interleaved": true,
            "mrope_section": [11, 11, 10],
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25
        }
    },
    "tie_word_embeddings": true,
    "transformers_version": "4.57.0.dev0",
    "video_token_id": 248057,
    "vision_config": {
        "deepstack_visual_indexes": [],
        "depth": 12,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 768,
        "in_channels": 3,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "model_type": "qwen3_5",
        "num_heads": 12,
        "num_position_embeddings": 2304,
        "out_hidden_size": 1024,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2
    },
    "vision_end_token_id": 248054,
    "vision_start_token_id": 248053
}
```

Hybrid-specific fields: `model_type: qwen3_5` (top), `text_config.model_type: qwen3_5_text`,
`full_attention_interval: 4`, `layer_types` (24 entries: 18 linear_attention + 6 full_attention, full every 4th),
`linear_conv_kernel_dim: 4`, `linear_key_head_dim: 128`, `linear_value_head_dim: 128`,
`linear_num_key_heads: 16`, `linear_num_value_heads: 16`, `attn_output_gate: true`,
`mamba_ssm_dtype: "float32"`, `mlp_only_layers: []`, `mtp_num_hidden_layers: 1`,
`num_attention_heads: 8` with `num_key_value_heads: 2`, `head_dim: 256` (note: 8*256=2048 != hidden 1024 for full attn proj layout), RoPE via nested `rope_parameters` (theta 1e7, partial_rotary_factor 0.25, mrope), `dtype: "bfloat16"` inside text_config, `transformers_version: "4.57.0.dev0"`.
No `attn_implementation` field in config; no `num_token_per_block`; no top-level `hidden_size` (it lives in text_config).

## Load probes (installed transformers = 5.16.1, python 3.14.3 from PATH)

Environment: `transformers 5.16.1` at `C:\Users\ysnrfd\AppData\Roaming\Python\Python314\site-packages\transformers\__init__.py` (vendor copy `JevX\vendor\transformers` also reports `__version__ = 5.16.1` and contains `models\qwen3_5`; installed site-packages also has `models/qwen3_5`).

- **4a. AutoConfig**: `AutoConfig.from_pretrained(<snapshot>, local_files_only=True)` → **OK**: `AutoConfig OK: Qwen3_5Config model_type= qwen3_5`. No error, no warning, no trust_remote_code needed.
- **4b. CONFIG mapping**: `"qwen3_5" in CONFIG_MAPPING` → `True`; `CONFIG_MAPPING["qwen3_5"]` → `<class 'transformers.models.qwen3_5.configuration_qwen3_5.Qwen3_5Config'>`. Sibling keys include `qwen3_5_text`, `qwen3_5_moe`, `qwen3_next`. Config class source: `vendor\transformers\src\transformers\models\qwen3_5\configuration_qwen3_5.py` (class `Qwen3_5TextConfig`, `model_type = "qwen3_5_text"`).
- **4c. AutoModel (default dtype)**: `AutoModel.from_pretrained(path, local_files_only=True)` → **OK**:
  - `MODEL LOAD OK: Qwen3_5Model`
  - `num params: 852985920 (853.0M)`, `dtype: torch.bfloat16`
  - children: `['Qwen3_5VisionModel', 'Qwen3_5TextModel']`
  - Weight load bar: 473/473 tensors, ~1s (no missing/unexpected-key errors printed).
- **4c-extra. AutoModel float32**: `dtype="float32"` → **OK**: `F32 LOAD OK: Qwen3_5Model`, dtype torch.float32, params 852,985,920.
- **4c-extra. Task heads**: `AutoModelForCausalLM` → **OK** `Qwen3_5ForCausalLM`; `AutoModelForImageTextToText` → **OK** `Qwen3_5ForConditionalGeneration` (matches config architectures).
- **4d. AutoTokenizer**: `AutoTokenizer.from_pretrained(path, local_files_only=True)` → **OK**:
  - class: `transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer`
  - `vocab_size: 248044`, `len(tok): 248077`
  - `all_special_tokens: ['<|eot_id|>', '<|end_of_text|>', '<|audio_start|>', '<|audio_end|>', 'Ӝ', 'PathComponent', '\grid', '鸟成长', 'เปิดเผย']` — NOTE: list continues with 5 image/video placeholder glyphs whose exact multi-byte characters were mangled in stdout transport (not reliably captured); presence of visual/audio specials confirmed
  - **pad_token present**: `pad_token: '<|end_of_text|>'`, `pad_token_id: 248044`
  - eos: `'<|eot_id|>'` id 248046; bos: None; unk: None
- No probe produced an error; **trust_remote_code is NOT required** for any probe.

## Transformers v5.16.1 hybrid inventory (vendor: `C:\Users\ysnrfd\Documents\Transformer\JevX\vendor\transformers\src\transformers`)

### qwen3_5 model dir EXISTS
`models/qwen3_5/`: `configuration_qwen3_5.py` (8,266 B), `modeling_qwen3_5.py` (88,587 B), `modular_qwen3_5.py` (28,816 B), `tokenization_qwen3_5.py` (3,127 B), `__init__.py`. Also `models/qwen3_5_moe/` (configuration/modeling/modular). **This is the dir that implements config's model_type.**

### qwen3_5 hybrid mechanics (file:line)
- Layer marking: config field **`layer_types`** (configuration_qwen3_5.py:103, derived from **`full_attention_interval`** at :113-116: `"linear_attention" if bool((i + 1) % interval_pattern) else "full_attention"`).
- Decoder dispatch: modeling_qwen3_5.py:747-751 — `self.block_type = config.layer_types[layer_idx]`; `linear_attention` → `self.linear_attn = Qwen3_5GatedDeltaNet(...)`; `full_attention` → `self.self_attn = Qwen3_5Attention(...)`.
- Forward dispatch: modeling_qwen3_5.py:770-786 (block_type if/elif in `Qwen3_5DecoderLayer.forward`).
- Per-layer mask selection: modeling_qwen3_5.py:1197-1200 builds `causal_mask_mapping = {"full_attention": create_causal_mask(...), "linear_attention": create_recurrent_attention_mask(...)}`; consumed at :1209 `attention_mask=causal_mask_mapping[self.config.layer_types[i]]`.
- Linear backend: `Qwen3_5GatedDeltaNet` (modeling_qwen3_5.py:383-387, `@use_kernel_forward_from_hub("Qwen3_5GatedDeltaNet")`, `@use_kernelized_func([torch_recurrent_gated_delta_rule, torch_chunk_gated_delta_rule, causal_conv1d_fn, causal_conv1d_update])`) — GDN (Gated DeltaNet) + short depthwise causal conv (kernel `linear_conv_kernel_dim=4`), gated RMSNorm out (`Qwen3_5RMSNormGated`), `A_log`/`dt_bias` params (Mamba-style discretization).

### qwen3_next (the GDN template qwen3_5 inherits from)
- modular_qwen3_5.py:45 `Qwen3NextGatedDeltaNet` imported; :207 `class Qwen3_5GatedDeltaNet(Qwen3NextGatedDeltaNet)`.
- Same structure: configuration_qwen3_next.py:123/131-134 (`layer_types` / `full_attention_interval`); modeling_qwen3_next.py:807-811 (block_type dispatch), :958 (mask map), :968 (per-layer dispatch), :939 `DynamicCache(config=self.config)`; GDN class :512, GDN forward :591-695.

### Cache for mixed layer types
- No `HybridCache` class exists anywhere in v5.16.1 (`class HybridCache` → NONE). Mixed handling is per-layer inside **`DynamicCache`** (`cache_utils.py:1737`):
  - `DynamicCache.__init__` (cache_utils.py:1780-1793): `get_layer_types_and_kwargs(decoder_config)` (:1701) then `layers = [DYNAMIC_LAYER_TYPE_MAPPING[layer_type](**layer_kwargs) for layer_type in layer_types]` (:1793).
  - Mapping `DYNAMIC_LAYER_TYPE_MAPPING` (cache_utils.py:1223-1244): `"full_attention"→DynamicLayer` (KV), `"linear_attention"→LinearAttentionLayer` (recurrent/conv state only, no per-token KV), also `"hybrid"→LinearAttentionAndFullAttentionLayer` (:1094), `"conv"→LinearAttentionLayer`, sparse types → `DynamicIndexedLayer`.
  - State APIs: `LinearAttentionLayer.update_conv_state` (cache_utils.py:1037), `update_recurrent_state` (:1077), mixin `LinearAttentionCacheLayerMixin` (:891), combined hybrid layer classes (:1094-1198). `Cache.has_previous_state` (cache_utils.py:1532).
  - qwen3_5 GDN reads states via `cache_params.layers[self.layer_idx].conv_states[0]` / `record_past` (modeling_qwen3_5.py:457-470) and `cache_params.update_conv_state(...)`; model instantiates `past_key_values = DynamicCache(config=self.config)` (modeling_qwen3_5.py:1171).

### Attention mask / padding for linear layers (CRITICAL)
- Mask producer: `create_recurrent_attention_mask` — **`masking_utils.py:1474-1504`**. Returns the **2D padding mask trimmed to the current segment** (`attention_mask[:, -inputs_embeds.shape[1]:].contiguous()`, :1504); returns `None` if mask missing/not-2D (:1496), if single-token decode (:1499), or all-ones (:1501). Docstring (:1493-1494) explicitly: "padding inside the new segment must still be zeroed out, otherwise it leaks into the recurrent state."
- Consumer: `apply_mask_to_padding_states` defined per model — qwen3_5 **`modeling_qwen3_5.py:187-196`** (`hidden_states = hidden_states * attention_mask[:, :, None]`), called at GDN forward entry **`modeling_qwen3_5.py:440`**; qwen3_next same at **`modeling_qwen3_next.py:313-322`, called :595**.
- Answer in 1 line: **padding tokens do NOT pollute the recurrent state content-wise — they are zeroed elementwise before entering the GDN block (masking_utils.py:1474 → modeling_qwen3_5.py:440/187-196), so padded positions inject zeros; caveat: only if the caller passes a 2D attention_mask (mask=None ⇒ no masking), and state decay still advances through padded steps.**
- Generic dispatch table: `LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING` masking_utils.py:1507-1520 maps `"linear_attention"→create_recurrent_attention_mask`, `"hybrid"→{full: create_causal_mask, linear: create_recurrent_attention_mask}`.

### Model dirs matching hybrid/linear-attention survey (keyword scan of models/*)
Dirs containing at least one of {GatedDeltaNet, lightning_attention/LightningAttention, linear_attention, HybridCache, MambaMixer, minimax, jamba, falcon_h1, RWKV, GDN} (25): `auto, bamba, falcon_h1, falcon_mamba, glm5_next, granitemoehybrid, hy_v3, inkling, jamba, lfm2, lfm2_moe, mamba, mamba2, minimax, minimax_m2, minimax_m3_vl, nemotron_h, olmo_hybrid, pvt_v2, qwen3_5, qwen3_5_moe, qwen3_next, qwen4_exp, rwkv, step3p7, zamba, zamba2`.
Specific keyword homes: GatedDeltaNet → `qwen3_5, qwen3_5_moe, qwen3_next, qwen4_exp, olmo_hybrid, glm5_next`; LightningAttention → `minimax`; MambaMixer → `mamba, jamba, bamba, falcon_mamba, zamba, zamba2, nemotron_h`; RWKV → `rwkv`; no `HybridCache` anywhere.
Requested-name check: `Mamba`(dir `mamba` + `mamba2`) present; `minimax`, `cohere2`(+`cohere2_moe`) present; `jamba` present; `falcon_h1` present; `rwkv` present; `deepseek*` below.

### deepseek dirs (exact list)
`deepseek_ocr2, deepseek_v2, deepseek_v3, deepseek_v32, deepseek_v4, deepseek_vl, deepseek_vl_hybrid` — i.e. **deepseek_v3 AND deepseek_v4 both exist** (plus v2, v32, vl variants).

## Memory / dtype note
- Weights on disk: 1.627 GiB (bf16 storage). 853.0M params → ~1.63 GB in bf16, ~3.41 GB in fp32 (before optimizer/activations/logits; logits over 248k vocab are sizeable in fp32).
- On 8 GB CPU RAM: load in **bf16** (the config default, confirmed working) or keep fp32 only for short forwards; fp32 load probe also succeeded but costs ~2x resident memory.

## Verdict
**LOADS NATIVELY.** transformers v5.16.1 ships `models/qwen3_5` with `Qwen3_5Config` registered in CONFIG_MAPPING; AutoConfig, AutoModel (bf16 and fp32), AutoModelForCausalLM, AutoModelForImageTextToText, and AutoTokenizer all succeed with `local_files_only=True` and **no trust_remote_code, no remote code files in the snapshot**. Hybrid (GDN linear-attention + full attention) is handled in-tree: `layer_types` config field, per-layer block dispatch (modeling_qwen3_5.py:747/770), dual mask map (modeling_qwen3_5.py:1197-1209), mixed-state `DynamicCache` layer classes (cache_utils.py:1223-1244, 1793), and padding zeroing for linear layers (masking_utils.py:1474 → modeling_qwen3_5.py:440). No unsupported-model error exists to capture — every probe passed on first attempt.
