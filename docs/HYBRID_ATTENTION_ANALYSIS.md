# Hybrid Attention Architectures: Linear/Full Interleaving in transformers v5.16.1

**Scope.** Analysis only; no `jevx/**` or `tests/**` files were modified. Every code claim carries a
`file:line` citation into `vendor/transformers/src/transformers` (v5.16.1) or the local
Qwen3.5-0.8B checkpoint recon (`vendor/reports/QWEN35_RECON.md`). Web sources are tagged
**VERIFIED** (fetched and read on 2026-09-24) or **UNVERIFIED** (claim not backed by a primary
source we read). Empirical results come from a seeded local script (torch 2.14.0+cpu,
transformers 5.16.1; the optional `kernels` package is *not* installed, so the pure-torch
reference kernels ran — `kernels_installed: false`).

---

## 1. What "hybrid model" means here

A "hybrid" transformer in this tree is not one thing. We distinguish five independent axes:

1. **Layer-type interleave (mixer hybrid).** A `layer_types: list[str]` config field assigns each
   decoder layer a token-mixer: `"linear_attention"` (recurrent/kernel attention), `"full_attention"`
   (softmax attention), `"sliding_attention"`, `"conv"`, `"moe"`/`"mlp"` placeholders, sparse types,
   or `"hybrid"`. The field is declared at `configuration_qwen3_5.py:103` and derived from
   `full_attention_interval` when absent (`configuration_qwen3_5.py:110-117`: linear iff
   `(i+1) % interval != 0`). Dispatch happens at layer construction
   (`modeling_qwen3_5.py:747-751`), at forward time (`modeling_qwen3_5.py:770-786`), and — the
   load-bearing part — at **mask** and **cache** construction (below).
2. **Cache-type mix (consequence of axis 1).** One `DynamicCache` holds heterogeneous per-layer
   objects: `DYNAMIC_LAYER_TYPE_MAPPING` maps `"full_attention"→DynamicLayer` (per-token KV),
   `"linear_attention"→LinearAttentionLayer` (conv + recurrent state, **no KV**),
   `"hybrid"→LinearAttentionAndFullAttentionLayer` (both), sparse types → `DynamicIndexedLayer`
   (`cache_utils.py:1223-1244`). Layers self-register via `_layer_type` in
   `CacheLayerMixin.__init_subclass__` (`cache_utils.py:36-42`), which is how DeepSeek-V4's
   compressor caches join the map. There is **no `HybridCache` class** in v5.16.1 (recon verified);
   mixing happens inside `DynamicCache.__init__`
   (`cache_utils.py:1780-1793`, building `layers = [DYNAMIC_LAYER_TYPE_MAPPING[lt](...) for lt in layer_types]` at `:1793`).
3. **Within-attention schedule hybridity.** Both mixers are softmax attention, but the *kind*
   varies per layer: sliding-window vs compressed-sparse (DeepSeek-V4), sliding vs full
   (Cohere2), sparse-indexed vs plain (DeepSeek-V3.2, GLM5-Next's "full" layers).
4. **Modality hybrid.** `Qwen3_5ForConditionalGeneration` wraps a vision tower + text stack
   (`modeling_qwen3_5.py:1675-1685`; checkpoint `model_type: qwen3_5` with `vision_config`,
   `image_token_id`, `video_token_id` — recon config dump). `deepseek_vl_hybrid` is hybrid in a
   completely different sense: **two vision encoders** (low-res + high-res SAM-style) fused by an
   aligner (`modeling_deepseek_vl_hybrid.py:192-214`, `:253-267`), with an ordinary (default
   Llama) text model (`configuration_deepseek_vl_hybrid.py:54-80`). Its attention is not hybrid at all.
5. **MoE schedule (orthogonal).** `mlp_layer_types` interleaves `hash_moe`/`moe` (DeepSeek-V4
   `configuration_deepseek_v4.py:35,278-282`) or dense/sparse (V3.2
   `configuration_deepseek_v32.py:117,134-136`) — orthogonal to how attention layers are typed.

### Taxonomy of in-tree families

| Family (dirs) | Linear/recurrent mixer | Full/sparse mixer | Schedule field + default | Key cites |
|---|---|---|---|---|
| **qwen3_5 / qwen3_5_moe** | `Qwen3_5GatedDeltaNet` (gated delta rule + depthwise conv) | `Qwen3_5Attention` (softmax, GQA 8/2, output gate) | `layer_types`, interval 4 → `[L,L,L,F]×6` | `configuration_qwen3_5.py:113-116`; `modeling_qwen3_5.py:387,632,747-751` |
| **qwen3_next** | `Qwen3NextGatedDeltaNet` (parent GDN) | `Qwen3NextAttention` | interval-4 interleave | `modeling_qwen3_next.py:512,807-811` |
| **qwen4_exp** | `Qwen4ExpTextGatedDeltaNet` | full layers may be `qwen_sparse_attention` | interval 4; allowed types `{linear, qwen_sparse}` | `modeling_qwen4_exp.py:403,1197`; `configuration_qwen4_exp.py:175-191` |
| **olmo_hybrid** | `OlmoHybridGatedDeltaNet` (separate q/k/v/a/b + `g_proj`, `allow_neg_eigval` β∈[0,2]) | `OlmoHybridAttention` | defaults all-linear, forces both types present | `modeling_olmo_hybrid.py:501-509`; `configuration_olmo_hybrid.py:124-149` |
| **glm5_next** | `Glm5NextTextLinearAttention` (Kimi-style **KDA** + conv) | `deepseek_sparse_attention` on every 4th layer | `idx%4 != 3 → linear else sparse` | `modeling_glm5_next.py:584-625`; `configuration_glm5_next.py:165-174` |
| **minimax** | `MiniMaxLightningAttention` (exponential-slope decay, block recurrent, output gate) | `MiniMaxAttention` | odd→full, even→linear (1:1) | `modeling_minimax.py:122,542`; `configuration_minimax.py:124-127` |
| **minimax_m2 / m3_vl** | — (m2: no `layer_types`, no Lightning, no recurrent mask — plain attention + MoE) | `MiniMaxM2Attention` | n/a | `configuration_minimax_m2.py` (no `layer_types`); `modeling_minimax_m2.py:287,356` |
| **mamba / mamba2 / falcon_mamba** | `MambaMixer` / `Mamba2Mixer` / `FalconMambaMixer` (pure SSM, no attention layers) | none | single-type (recurrence-only family) | `modeling_mamba.py:285`; `modeling_mamba2.py:360`; `modeling_falcon_mamba.py:300` |
| **jamba** | `JambaMambaMixer` | `JambaAttention` | attention every 8th, offset 4; experts every 2nd | `configuration_jamba.py:32-35,79-110`; `modeling_jamba.py:441,707,745` |
| **zamba / zamba2** | mamba layers | shared attention in `"hybrid"` layers | zamba2: `["linear"]*5+["hybrid"]` ×7 + singles | `configuration_zamba.py:42-45,100-110`; `configuration_zamba2.py:126-140`; `modeling_zamba.py:724` |
| **bamba** | mamba2 layers | `BambaAttention` at explicit `attn_layer_indices` | list-driven | `configuration_bamba.py:96-101` |
| **falcon_h1** | `FalconH1Mixer` (Mamba2-derived, per docstring) | attention folded per `"hybrid"` block type | `layers_block_type = ["hybrid"]*n` | `modeling_falcon_h1.py:588-590,654`; `configuration_falcon_h1.py:133-134` |
| **nemotron_h** | `NemotronHMamba2Mixer` | `NemotronHAttention` (imports DeepSeek-V3 MLA code) | cycle `["linear_attention","moe","full_attention","mlp"]` | `configuration_nemotron_h.py:163-165`; `modeling_nemotron_h.py:356,827` |
| **granitemoehybrid** | mamba layer | `GraniteMoeHybridAttention` | `layers_block_type` (default all-linear; real ckpts supply pattern) | `configuration_granitemoehybrid.py:106-109`; `modeling_granitemoehybrid.py:940-954,1095-1108` |
| **rwkv** | `RwkvLinearAttention` (WKV recurrence, CUDA kernel) | none (single-type) | n/a | `modeling_rwkv.py:56,217` |
| **lfm2 / lfm2_moe** | `"conv"` layer type (short conv only) | `full_attention` | `full_attn_idxs` list, else conv | `configuration_lfm2.py:81-88`; `modeling_lfm2.py:395,521` |
| **cohere2 / cohere2_moe** | none | sliding vs full softmax | sliding pattern 4: `sliding` iff `(i+1)%4 != 0` | `configuration_cohere2.py:88-102`; `modeling_cohere2.py:198` |
| **deepseek_v32** | none | MLA + DSA indexer sparse mask on **all** layers | `layer_types = ["deepseek_sparse_attention"]*n` | `configuration_deepseek_v32.py:126-139`; `modeling_deepseek_v32.py:351-353,756` |
| **deepseek_v4** | none | shared-KV MQA + sliding/CSA/HCA compressors | `layer_types ∈ {sliding, CSA, HCA}`, 2×HCA bootstrap then interleave | `configuration_deepseek_v4.py:21-32,270-276` |
| **step3p7** | none | local (sliding) vs full attention | `layer_types` | `modeling_step3p7.py:702` |
| **llt** | — | — | **directory does not exist in this tree** (`models/llt` → ENOENT) | — |

Mask plumbing shared by all axis-1 hybrids: `LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING`
(`masking_utils.py:1507-1520`) maps `linear_attention→create_recurrent_attention_mask` and
`full_attention→create_causal_mask`, with per-model dicts built at forward time
(`modeling_qwen3_5.py:1197-1200`, consumed per-layer at `:1209`).

---

## 2. Qwen3.5-0.8B dissection

### 2.1 Layer layout and dataflow

Checkpoint facts (recon): 24 layers, `layer_types = [linear,linear,linear,full]×6`
(18 GDN + 6 softmax), hidden 1024, 8 Q heads / 2 KV heads / `head_dim` 256, GDN 16 key +
16 value heads × dim 128, conv kernel 4, vocab 248320, bf16 weights, `mtp_num_hidden_layers: 1`
(recon config dump; interval derivation at `configuration_qwen3_5.py:113-116`).

Forward pass (`Qwen3_5TextModel.forward`, `modeling_qwen3_5.py:1154-1221`):

1. `input_ids → embed_tokens` (`:1167-1168`).
2. `use_cache and past_key_values is None → DynamicCache(config)` (`:1170-1171`), which
   instantiates 18 `LinearAttentionLayer` + 6 `DynamicLayer` objects purely from `layer_types`
   (`cache_utils.py:1791-1793`).
3. Position ids default to a flat `arange` expanded to **4 rows** (text/t/h/w rows; `:1173-1177`)
   — the first row is split off as `text_position_ids` for masks, the remaining 3 feed interleaved
   MRoPE (`:1181-1183`).
4. Two masks are built once (`:1197-1200`): `full → create_causal_mask` (4-D additive/bool mask),
   `linear → create_recurrent_attention_mask` (a **trimmed 2-D padding mask or `None`**,
   §6). Each decoder layer receives `causal_mask_mapping[layer_types[i]]` (`:1209`).
5. Per layer (`Qwen3_5DecoderLayer.forward`, `:756-796`): RMSNorm → token mixer chosen by
   `self.block_type` (`:770-786`; block built at `:747-751`) → residual → RMSNorm → SiLU-gated
   MLP (`:718-720`; `mlp_only_layers` empty, so every layer has an MLP) → residual. Full-attention
   layers additionally receive `position_embeddings` (`:1208`, used at `:679-680`); **GDN layers
   never receive position embeddings** (`modeling_qwen3_5.py:770-776` — order enters only via the
   causal conv and the recurrence).
6. Final RMSNorm (`:1216`) → `Qwen3_5ModelOutputWithPast` with `last_hidden_state` and
   `past_key_values` (`:1218-1221`).

### 2.2 GDN math **as implemented**

Projections (`Qwen3_5GatedDeltaNet.__init__`, `modeling_qwen3_5.py:388-430`): a single
`in_proj_qkv` (1024 → 6144 = 2·2048 q/k + 2048 v), gate `in_proj_z` (1024 → 2048), forget gate
`in_proj_b` (1024 → 16), data rate `in_proj_a` (1024 → 16); a depthwise causal `conv1d` over all
6144 channels with kernel 4 (`:404-412`); Mamba-style discretization parameters `dt_bias` and
`A_log ~ log U(0.01, 16)` (`:414-420`). Forward (`:433-546`):

1. **Zero padding first**: `hidden_states = apply_mask_to_padding_states(hidden, mask)`
   (`:440`; implementation `:187-196` multiplies by `attention_mask[:, :, None]`).
2. Project, then run the causal conv. With a cache, the conv state window is spliced in via
   `cache_params.update_conv_state` and trimmed back to the live segment
   (`:467-483`); single-token decode uses the in-place `causal_conv1d_update`
   (`:457-466`; implementation `:199-216`).
3. Split into `query, key, value` (`:486-498`); gates:
   `β = σ(b)` (`:500`) and per-head log-decay
   `g = −exp(A_log) · softplus(a + dt_bias) < 0` (`:502`, computed in fp32 by construction).
4. Kernel selection: cached single token → `torch_recurrent_gated_delta_rule`
   (`:508-520`), otherwise the chunkwise `torch_chunk_gated_delta_rule` (`:521-533`). Both are
   decorated with `use_qk_l2norm_in_kernel=True` (L2-normalized q/k) and scale q by `d_k^{-1/2}`
   (`:278, :352`).

The recurrence, traced line-by-line from `torch_recurrent_gated_delta_rule`
(`modeling_qwen3_5.py:346-375`; all state math `.to(torch.float32)` at `:346-348`), per head
with state `S_t ∈ R^{128×128}`, `λ_t = exp(g_t)`:

$$
\tilde S_t = \lambda_t \odot S_{t-1}, \qquad
\delta_t = \beta_t \odot \big(v_t - k_t^{\top}\tilde S_t\big), \qquad
S_t = \tilde S_t + k_t\,\delta_t^{\top}, \qquad
o_t = S_t\, q_t
$$

(β broadcast over the value dim: `:369`; decay `:371`; retrieval `k^T S` via
`(S * k).sum(dim=-2)` `:372`; delta-rule write as outer product `k.unsqueeze(-1) * delta.unsqueeze(-2)`
`:374`; readout `:375`.) The subtractive term inside `δ` is the delta rule's *erase*: writing a new
association for key `k_t` first removes the component of old values that `k_t` retrieves, which is
what separates DeltaNet/GDN from additive linear attention (arXiv:2406.06484, **VERIFIED** abstract).
The chunk kernel (`:248-327`) is the parallel training form: cumulative decay within a 64-token
chunk (`:291-292`), a fixed-point triangular correction (`:294-298`), and inter-chunk state carry
(`:317-320`); note the torch fallback loops over chunks sequentially (`:310`) — full chunk
parallelism lives in the optional hub/FLA kernels (UNVERIFIED: not installed here).

**Output gate** (`:540-545`): the attention output is RMSNorm-ed **in fp32** and multiplied by
`silu(z)` from `in_proj_z` inside `Qwen3_5RMSNormGated` (`:167-184`), then `out_proj` maps
2048 → 1024 (`:423`). Cache write-back: `update_recurrent_state` (`:536-537`).

### 2.3 Full-attention layer specifics

`Qwen3_5Attention` (`modeling_qwen3_5.py:632-704`):

- **The 8×256 = 2048 ≠ 1024 question, resolved in source.** `q_proj` maps
  1024 → `num_attention_heads · head_dim · 2 = 4096` (`:644-646`); the output is viewed as
  `(B,S,8,512)` and **chunked in half along the last dim** into `query` and a same-shaped **gate**
  (`:670-673`). So the attention sublayer runs at width 2048 (8 heads × 256), and the *second half
  of the q projection* is a per-position output gate applied as `attn_output · σ(gate)` after the
  softmax product (`:700-701`), before `o_proj: 2048 → 1024` (`:653-655`, `:703`). The config flag
  `attn_output_gate: true` has **no consumer anywhere in transformers 5.16.1** (repo-wide grep: 0
  hits) — the gate is unconditional in this class; treat the flag as descriptive.
- GQA 8/2: `num_key_value_groups = 8/2 = 4` (`:640`), KV projections 1024 → 512
  (`:647-652`), `repeat_kv` expansion (`:595-604`), per-head QK RMSNorm on `head_dim=256`
  (`:656-657`).
- **Partial MRoPE**: `partial_rotary_factor 0.25 × 256 = 64` rotary dims
  (`configuration_qwen3_5.py` rope params; computed at `modeling_qwen3_5.py:117-124`; the model
  card states "rotary embeddings of dimension 64" — **VERIFIED**). Interleaved MRoPE sections
  `[11,11,10]` sum to 32 = 64/2 (consistency check against `:101, :149-164`). `apply_rotary_pos_emb`
  rotates only the first `rotary_dim` and passes the rest through (`:580-591`).
- Backend: `ALL_ATTENTION_FUNCTIONS.get_interface(config._attn_implementation, eager)`
  (`:685-687`); the eager fallback upcasts softmax to fp32 (`:624`). `_supports_flash_attn` and
  `_supports_sdpa` are True (`:805-806`). KV enters the cache via `past_key_values.update`
  (`:682-683`) → `DynamicLayer` (`cache_utils.py:1224`).

### 2.4 MTP layer

The checkpoint contains 15 `mtp.*` tensors (`mtp.fc`, `mtp.layers.0.*`, `mtp.norm`,
`mtp.pre_fc_norm_{embedding,hidden}` — safetensors index, verified locally): one transformer draft
layer used for speculative decoding at serving time (model card documents
`--speculative-config {"method":"qwen3_next_mtp",...}` for vLLM and `NEXTN` for SGLang —
**VERIFIED** as card text; effectiveness UNVERIFIED). The HF classes **ignore** these weights:
`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]` (`modeling_qwen3_5.py:807, :1584`). MTP is
therefore invisible to representation extraction.

### 2.5 Dtype policy

- Weights/activations: bf16 (recon: `dtype: "bfloat16"` in `text_config`; 1.63 GiB shard).
- **Delta-rule state math: always fp32 in the reference kernels**, independent of input dtype —
  explicit casts at `modeling_qwen3_5.py:265-267, :346-348` and `g = -A_log.float().exp() *
  softplus(a.float() + dt_bias)` at `:502`; gated RMSNorm upcasts at `:177`.
- The config field `mamba_ssm_dtype: "float32"` has **no consumer in transformers 5.16.1**
  (repo-wide grep: 0 hits). The fp32 guarantee that matters comes from the kernel casts, not the
  field. Whether the optional fused hub kernels honor `mamba_ssm_dtype` is **UNVERIFIED**
  (package absent in this environment).

---

## 3. DeepSeek-V4 (and V3.2) from local source

### 3.1 DeepSeek-V4: NOT a linear-attention hybrid

Code-verified facts:

- **Attention types.** `DEEPSEEK_V4_LAYER_TYPES = ("sliding_attention",
  "compressed_sparse_attention", "heavily_compressed_attention")`
  (`configuration_deepseek_v4.py:21-25`); default schedule = 2×HCA bootstrap then alternating
  HCA/CSA (`:270-276`); compression rates 4 (CSA) and 128 (HCA) (`:159`); `sliding_window: 128`
  (`:167`). There is **no GDN, Mamba, or linear-attention class anywhere in `models/deepseek_v4`**
  (class scan: zero hits).
- **One softmax attention class, three cache/mask variants.** `DeepseekV4Attention`
  (`modeling_deepseek_v4.py:746-864`) is shared-KV MQA (`num_key_value_heads = 1`, single
  `kv_proj` read as both K and V — docstring `:748-751`, `kv_proj` `:781-816`), LoRA-style queries
  (`q_lora_rank 1024`, `:777-780`), `head_dim 512`/64 heads (`configuration:143-145`), partial
  RoPE (64/512, `configuration:147`), per-head learnable **attention sinks** (`:787`, passed as
  `s_aux` to the backend `:849`), and a grouped low-rank output projection (`:783-786, :861-863`).
  CSA/HCA layers attach a `compressor` (`:788-790`) that appends **compressed KV entries** to the
  KV axis with a per-block causal/indexer `block_bias` concatenated onto the mask
  (`:818-835`). A **Lightning Indexer** (`DeepseekV4Indexer`, `:453-`; top-512 per query,
  `configuration:170-172`) selects which compressed entries each query may attend.
- **Caches.** `DeepseekV4HCACache(DynamicSlidingWindowLayer)` with `_layer_type =
  "heavily_compressed_attention"` (`modeling:162-185`) and `DeepseekV4CSACache`
  (`:246-265`, `_layer_type = "compressed_sparse_attention"`) auto-register into
  `DYNAMIC_LAYER_TYPE_MAPPING` via `cache_utils.py:36-42`. The sliding window update stores
  `self.values = self.keys` (shared KV) and keeps only the last `window+1` entries
  (`:195-207`); the compressor buffers `window-1`-tail tokens and emits one pooled entry per
  `compress_rate` source tokens (`:209-243`); CSA additionally persists two-series overlap state
  (`:246-291`).
- **Decoder residual is mHC**, not a plain skip: `hc_mult = 4` parallel streams mixed by
  Sinkhorn-Knopp-projected doubly-stochastic matrices (`configuration:161-163`;
  `DeepseekV4DecoderLayer` `modeling:1096-1144`).
- **Mask** is one `create_sliding_window_causal_mask` for all layer types (`modeling:1297-1309`);
  per-type masks from `create_masks_for_generate` are accepted as a dict (`:1300-1301`).
- Checkpoint surface: HF repo `deepseek-ai/DeepSeek-V4-Flash-Base` exists, 292B parameters,
  **no model card** (**VERIFIED** fetch, 2026-09-24); in-tree config defaults
  (43 layers etc., `configuration_deepseek_v4.py:139-173`) are class defaults, not necessarily
  the shipped checkpoint's values (**UNVERIFIED** for the actual config.json).

Web/press claims, separated:

- The DeepSeek-V4 paper (arXiv:2606.19348, **VERIFIED** abstract) describes exactly what the code
  implements: a "hybrid attention design that combines Compressed Sparse Attention (CSA) and
  Heavily Compressed Attention (HCA)", mHC hyper-connections, MoE (V4-Pro 1.6T/49B active,
  V4-Flash 284B/13B active, 1M context), and claims "27% of single-token inference FLOPs and 10%
  of KV cache compared with DeepSeek-V3.2" at 1M context (**VERIFIED** as abstract text; full
  paper not read).

**Verdict.** DeepSeek-V4 **is** a hybrid — but a hybrid of *sparse/compressed softmax attention
variants* (sliding + CSA + HCA) plus a hash-MoE schedule. It is **not** a linear/full-attention
interleaved model. V4's "hybrid" ≠ Qwen3.5's "hybrid"; the shared word hides different axes
(axis 3 vs axis 1 of §1).

### 3.2 DeepSeek-V3.2

V3.2 is **uniformly sparse**: `layer_types = ["deepseek_sparse_attention"] * num_hidden_layers`
(`configuration_deepseek_v32.py:126-139`) — every layer gets DeepSeek-V3 **MLA**
(`kv_lora_rank 512`, `q_lora_rank 1536`, `v_head_dim 128`, `configuration:98-101`) plus a **DSA
indexer** whose top-k sparse mask is folded into the attention mask
(`modeling_deepseek_v32.py:160-232, :351-353`; indexer uses non-interleaved half-split RoPE,
`:232`; the sparse model caches expanded K/V, `:460`). Its only interleaving is at the MLP level:
dense MLPs for the first layers, MoE afterwards (`configuration:134-136`). Cache:
`DynamicIndexedLayer` (`cache_utils.py:1236`); mask: `create_causal_mask` for the sparse type
(`modeling:756`; mapping `masking_utils.py:1514`).

### 3.3 deepseek_vl_hybrid

Hybrid = **dual vision encoders** (siglip-style low-res + SAM-style high-res with
`global_attn_indexes`) whose outputs are concatenated by `DeepseekVLHybridAligner`
(`modeling_deepseek_vl_hybrid.py:192-214`), over an ordinary text decoder (default Llama,
`configuration_deepseek_vl_hybrid.py:75-80`). Attention mechanism: not hybrid.

---

## 4. Linear attention vs full attention — formal comparison

### 4.1 Full softmax attention

$$
o_t = \sum_{j \le t} \frac{\exp\!\big(q_t^{\top}k_j/\sqrt d + M_{tj}\big)}{\sum_{i \le t} \exp\!\big(q_t^{\top}k_i/\sqrt d + M_{ti}\big)}\, v_j
$$

- **Training:** $O(T^2 d)$ FLOPs; the $T\times T$ weight matrix is materialized (eager/sdpa) or
  re-computed tile-wise (FlashAttention, $O(T)$ auxiliary memory). Fully parallel over $t$ — every
  position is one matmul stack, which is the property SSM/linear models spend cleverness to
  recover (state-space duality: attention *is* a structured semiseparable matrix product —
  arXiv:2405.21060, **VERIFIED** abstract).
- **Decode:** $O(Td)$ FLOPs per token and a KV cache growing linearly in $T$ per layer:
  bytes/token $= 2 \cdot n_{kv} \cdot d_h \cdot 2\text{B}$ (K and V, bf16).
- **Retrieval:** weights form a normalized, content-addressed distribution over *all* past
  positions — softmax can concentrate $1-\varepsilon$ on any single earlier token at any distance;
  this is "exact recall" capacity, and it is what pure-linear models underperform on
  (DeltaNet paper motivation: linear transformers/SSMs "still underperform transformers especially
  on tasks that require in-context retrieval" — arXiv:2406.06484, **VERIFIED** abstract).
- **Masking:** a $T\times T$ (or banded) mask matrix per forward; positions from explicit RoPE.
- **Expressivity notes:** softmax weights sum to 1 (nothing is lost) and are unbounded-ratio —
  logit gaps translate directly into sharp distributions, which sharpens decision-relevant logit
  separation.

### 4.2 Kernel / linear attention (and GDN specifically)

Generic feature-map form: $S_t = \sum_{j\le t} \phi(k_j) v_j^{\top}$, $o_t = \phi(q_t)^{\top} S_t$ —
training $O(T d^2)$ (with $m=\dim\phi \approx d$), decode state fixed at $O(d_k d_v)$ per head,
$O(d^2)$ work per token regardless of $T$. Position sensitivity comes from the conv + recurrence,
not from an addressable index. The cost of constant state: the entire prefix passes through one
$d_k\times d_v$ matrix per head, so retrievable content is bounded and interference-prone;
without a normalizer there is no "probability over the past" — outputs are bounded by
$\|\phi(q)\|\cdot\|S\|$ rather than being a convex combination.

**GDN as coded** (§2.2) adds two mechanisms on top:

- **Forgetting:** $\lambda_t = e^{g_t}$ with $g_t = -e^{A_h}\,\mathrm{softplus}(a_t + \mathrm{dt\_bias}_h)$
  (`modeling_qwen3_5.py:502`) — data-dependent, head-specific exponential decay (the "gate" of
  Gated DeltaNet; arXiv:2412.06464 **VERIFIED**: gated delta rule "combines gating's rapid memory
  erasure with the delta rule's targeted updates" and outperforms Mamba2/DeltaNet).
- **Targeted overwrite:** $\delta_t = \beta_t \odot (v_t - k_t^{\top}\tilde S_t)$ — the erase term
  $-k_t^{\top}\tilde S_t$ makes writes *replace* rather than accumulate, which buys associative
  recall a pure additive updater (RetNet arXiv:2307.08621, GLA arXiv:2312.06635 — both
  **VERIFIED** abstracts) cannot match — while remaining capacity-bounded by $128\times128$ per
  head.

### 4.3 Side-by-side (Qwen3.5-0.8B concrete numbers, bf16 weights, batch 1)

| Property | Full softmax (`Qwen3_5Attention`, 6 layers) | Linear GDN (`Qwen3_5GatedDeltaNet`, 18 layers) |
|---|---|---|
| Train compute | $O(T^2 d)$ per layer | $O(T d^2)$ chunkwise (`modeling_qwen3_5.py:248-327`) |
| Decode compute/token | $O(T \cdot 2048)$ (+masking) | $O(16\cdot128\cdot128)$ state ops + conv — constant in $T$ |
| Cache state | per-token KV: $2 \text{ (K,V)} \cdot 2 \text{ kv heads} \cdot 256 \cdot 2\text{B} = \mathbf{2{,}048\ B/token/layer}$ → **12,288 B/token** across 6 layers, unbounded growth | recurrent: $16 \cdot 128 \cdot 128 \cdot 4\text{B (fp32)} = \mathbf{1{,}048{,}576\ B/layer}$ fixed + conv: $6144 \cdot 4 \cdot 2\text{B (bf16)} = \mathbf{49{,}152\ B/layer}$ fixed → **18 MiB + 864 KiB = 19,759,104 B total, constant in $T$** |
| Hybrid total cache at length $T$ | $12{,}288\,T + 19{,}759{,}104$ bytes | — (break-even vs an all-full 24-layer layout: $19{,}759{,}104 / (49{,}152-12{,}288) = \mathbf{536}$ **tokens**) |
| Parallel training | native matmuls over $T$ | chunk algorithm; torch fallback loops chunks (`:310`), decode loops steps (`:364`) |
| Masking semantics | $T\times T$ causal mask / flash varlen | **zero-before-recurrence** 2-D mask, `None` shortcuts (§6) |
| Position encoding | interleaved MRoPE, partial rotary 64/256 (`:117-124, :149-164`) | **none passed** — order via conv(4) + decay (`:770-776`) |
| Retrieval regime | exact content addressing at any distance; sharp, normalized distributions | capacity- and decay-limited recall; overwrite dynamics mitigate but do not remove interference; long-$T$ degradation concentrates on mid-context exact recall |
| Where it wins | retrieval-in-the-middle, precise logit separation, short/medium $T$ | long-$T$ memory footprint and per-token latency, constant-state streaming |

Verification of cache shapes: state shapes are fixed by the kernels
(`modeling_qwen3_5.py:358-361` → `[B,16,128,128]` fp32; conv state `cache_utils.py:1019-1023` →
`[B,6144,4]` in input dtype) and by `LinearAttentionLayer`'s static-shape design
(`cache_utils.py:998-1000`, `get_max_length() == -1`). Beam/repeat handling: `reorder_cache`
`index_select`s conv and recurrent states along batch (`cache_utils.py:954-961`); KV reorders per
`DynamicLayer`; mixed caches reorder each layer with its own logic (`cache_utils.py:1216-1219`).

---

## 5. Why models interleave (the 1:3 pattern)

The Qwen3.5-0.8B model card (**VERIFIED**, https://huggingface.co/Qwen/Qwen3.5-0.8B) states the
layout verbatim as "6 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))" and claims
"Gated Delta Networks combined with sparse Mixture-of-Experts deliver high-throughput inference
with minimal latency and cost overhead". It reports benchmarks against *other models*
(Qwen3-4B-2507, Qwen3-1.7B, Qwen3.5-2B), **not** against a same-size pure-attention ablation — so
no primary hybrid-vs-pure number exists on the card. The qwen.ai blog (`?id=qwen3.5`) is
JS-rendered and returned no text to our fetch — **UNVERIFIED**.

What the full layers plausibly restore, grounded in what we *did* read:

1. **Exact retrieval and sharp distributions.** The GDN state is a bounded $128\times128$ buffer
   per head with data-dependent decay; the DeltaNet paper's own framing is that linear models
   underperform "on tasks that require in-context retrieval" (arXiv:2406.06484, **VERIFIED**).
   Periodic softmax layers reintroduce a normalized, all-distance addressable read every 4th
   layer, capping how far any fact must survive inside recurrent state alone.
2. **Logit-space precision.** Softmax weights are unbounded-ratio and sum to 1; linear readouts
   are bounded and interference-summed. Full layers restore the ability to *sharpen* — relevant
   for both next-token logits and, downstream, for representation geometry JevX consumes.
3. **Engineering precedent.** MiniMax-M1 bills itself as "the world's first open-weight,
   large-scale hybrid-attention reasoning model" (arXiv:2506.13585, **VERIFIED** abstract), and
   MiniMax-01 interleaves lightning attention with full attention 1:1
   (`configuration_minimax.py:124-127`; arXiv:2501.08313, **VERIFIED** abstract). The 3:1 (Qwen)
   vs 1:1 (MiniMax) vs sparse-4 (GLM5-Next) ratios are design choices; no primary source we read
   derives the ratio — that specific rationale is **UNVERIFIED** (analyst inference from the
   retrieval argument above).

---

## 6. Masking / padding semantics deep-dive (JevX-critical)

### 6.1 What the code does

`create_recurrent_attention_mask` (`masking_utils.py:1474-1504`) returns the **2-D padding mask
trimmed to the trailing `inputs_embeds.shape[1]` positions** (`:1504`), or `None` when: the mask
is missing or 4-D (`:1496-1497`); the forward is a single decoded token (`:1498-1500`); or the
mask is all-ones (`:1501-1502`). Its docstring is explicit about the danger: "padding inside the
new segment must still be zeroed out, otherwise it leaks into the recurrent state"
(`:1493-1494`). The consumer is `apply_mask_to_padding_states`
(`modeling_qwen3_5.py:187-196`, `hidden * attention_mask[:, :, None]`), called at GDN entry
(`:440`) — i.e., **pad hidden states are zeroed elementwise before projection, convolution, and
recurrence**. Two implications follow from the code alone:

- **Zeros ≠ skipped steps.** A zeroed token still *advances* the recurrence: the state multiplies
  by $\lambda_t<0$-log (i.e. decays) each padded step, the conv window still shifts, and — for the
  first `kernel-1 = 3` padded positions — the causal conv's left context still contains real
  trailing tokens, so padded positions are not perfectly inert content-wise even when masked.
- **`None` means no zeroing at all.** Missing/4-D masks short-circuit to `None` (`:1496-1497`) and
  GDN proceeds unmasked; single-token decode skips masking only because a decoded token is never
  padding (`:1498-1499`).

### 6.2 Precise claim about left vs right padding

> **Claim.** Given identical content tokens and a 2-D attention mask:
> (a) outputs at *content positions* are padding-side-invariant — left-pad ≡ right-pad ≡ unpadded;
> (b) the **final recurrent state is not**: left-pad ≡ unpadded, but right-pad decays the state
> through every trailing padded step, so left and right padding yield *different* end states for
> identical content; (c) without a mask, left-padding corrupts content-position outputs directly,
> while right-padding corrupts only the carried state.

### 6.3 Empirical verification (seeded, local)

Setup: fresh `Qwen3_5TextConfig` (hidden 64, 4 heads × dim 16, conv kernel 4,
`full_attention_interval=4` → `layer_types=['linear_attention']`), one `Qwen3_5GatedDeltaNet`,
`torch.manual_seed(0/7/123)`, fp32, `L=6` content tokens, `P=4` pad tokens with *strongly nonzero*
pad inputs (`randn*3+5`). Per-head init decay `g ∈ [-4.71, -6.41, -6.35, -16.33]`
(random `A_log`, `dt_bias=1`; trained checkpoints decay more slowly — numbers below demonstrate
the mechanism, they are not checkpoint-faithful magnitudes).

| Case | max &#124;Δ&#124; content outputs vs unpadded | final recurrent state vs unpadded | final conv state |
|---|---|---|---|
| A: no padding | — | ‖S‖ = 0.5790 | reference |
| B: right-pad **+ mask** | **0.0** (exact) | ‖S‖ = 0.00146 (**ratio ≈ 2.5·10⁻³**; predicted pure decay $e^{4g}$ ≤ 6.5·10⁻⁹ per head — residual is conv-bleed writes) | **0.0** (pads zeroed) |
| C: left-pad **+ mask** | 2.98·10⁻⁸ | 3.73·10⁻⁹ (≡ unpadded) | 0.0 (≡ unpadded) |
| D: left-pad **no mask** | **0.502** | 4.39·10⁻⁷ (state recovers: content delta-rule writes + $e^{L g}$ decay erase pad garbage) | differs |
| E: right-pad **no mask** | 0.0 (causality protects earlier outputs) | **1.99** (pad content written into state) | 10.65 (pad values cached) |

Mask-factory checks, same run: trimmed output shape `[1,4]` with values `[0,0,0,0]` for input mask
`[1,1,1,0,0,0,0,0]` (trailing-slice semantics, `masking_utils.py:1504`); `None` for single-token,
all-ones, and 4-D masks (`:1496-1501`).

**Verdict.** Left-padding with a mask is state-exact; right-padding with a mask preserves
content-position outputs exactly but *attenuates the carried recurrent state* (and zeroes the conv
state). For identical content, left vs right padding **change the final hybrid state** — they do
**not** change the hidden states that JevX pools. The mechanism is visible in the kernel: content
outputs only depend on $g_i - g_j$ differences within the chunk (prefix decay cancels), while the
final state depends on absolute cumulative decay including trailing pad steps
(`modeling_qwen3_5.py:291-320, :364-375`).

### 6.4 Implications for pooling

- `last_valid` (`jevx/representations/pooling.py:18-19`) gathers at
  `max{t : mask=1}` — a content position — so claim (a) gives padding-side invariance (P4.3),
  empirically exact/right and ≤3·10⁻⁸/left.
- `masked_mean` sums only masked-in positions — same invariance.
- Both hold **only if the 2-D mask is actually passed** (case D: 0.50 corruption without it) and
  the run is single-forward (JevX sets `use_cache=False`, §7, so no state survives to be
  decay-sensitive anyway).
- Any *stateful* workflow (chunked prefill, generation continuation after a right-padded batch)
  starts from an attenuated state (case B) — transformers' convention of left-padding generated
  batches exists for exactly this reason.

---

## 7. JevX implications

Cross-referencing `reports/TRANSFORMERS_SOURCE_AUDIT.md` §5 (cache/masking for extraction,
`:87-93`), `vendor/reports/QWEN35_RECON.md`, and the adapter sources:

1. **Capability detection.** `qwen3_5` need not set `config.is_decoder` (absent on this config):
   the detector falls back to "cache-capable output + not enc-dec + no pooler"
   (`jevx/capabilities/detector.py:289-325`) — `Qwen3_5TextModel` returns
   `Qwen3_5ModelOutputWithPast(past_key_values=...)` (`modeling_qwen3_5.py:1218-1221`), so
   `decoder=True`. `layer_types` should be *recorded* (it determines mask/cache dispatch) but must
   not gate support. Detection order caveat: the multimodal wrapper's forward exposes
   `pixel_values`/`image_grid_thw` (`modeling_qwen3_5.py:1720-1726`), so vision detection
   (`detector.py:275-286`) fires on `Qwen3_5Model`/`...ForConditionalGeneration` — for clean text
   extraction prefer `Qwen3_5TextModel` (or ensure the multimodal adapter handles a text-only
   call).
2. **Adapter choice.** `DecoderAdapter.can_handle` requires decoder ∧ ¬enc_dec ∧ ¬vision ∧ ¬audio
   (`jevx/adapters/decoder.py:45-53`) → applies to the *text* stack; pooling default `last_valid`
   (`:43`), chosen because decoder families ship no pooler (audit Q7,
   `TRANSFORMERS_SOURCE_AUDIT.md:139-153`).
3. **`use_cache=False` — verified.** The shared extraction path injects `use_cache=False` whenever
   the forward declares it (`jevx/adapters/base.py:250-266`, esp. `:265-266`; policy comment
   `decoder.py:12-14`). With `use_cache` falsy, `DynamicCache` is never allocated
   (`modeling_qwen3_5.py:1170-1171`), GDN runs `cache_params=None` → no conv/recurrent state is
   read or written (`:444-446, :536-537`) and the chunk kernel returns no final state
   (`output_final_state=cache_params is not None`, `:529`). Representation extraction is therefore
   pure-functional on this family. **Recommendation stands: keep `use_cache=False`.**
4. **2-D `attention_mask` is mandatory for padded batches.** Full layers fold it into the causal
   mask; linear layers depend on it for pad zeroing (`§6`); passing `None`/4-D leaks pad content
   (empirical: 0.50 output corruption). JevX already passes masks for pooling — keep doing so for
   *any* hybrid/SSM family, not just qwen3_5.
5. **`position_ids` from the mask.** `DecoderAdapter._postprocess_forward_kwargs` derives
   `position_ids = cumsum(mask)-1` (`decoder.py:55-75`), which makes left-pad positions match
   unpadded ones for the RoPE'd full-attention layers (default `arange` at
   `modeling_qwen3_5.py:1174-1177` would shift them). GDN layers ignore positions entirely, so
   this fix matters only for the 6 full layers — but those 6 are exactly where MRoPE runs.
6. **bf16 → fp32 pooling windows** are already in place: `pooling.py:81, :118, :289-308` and
   `jevx/utils/numerical.py:66+` upcast low-precision inputs before softmax-weighted math and cast
   back — consistent with the model's own fp32-inside conventions (`modeling_qwen3_5.py:177, :346`).
7. **MTP / conditional-generation surface.** `mtp.*` weights are ignored by HF classes
   (`modeling_qwen3_5.py:807, :1584`) — nothing to strip. The wrapper is genuinely multimodal
   (`image_token_id`/`video_token_id`, `preprocessor_config.json` +
   `video_preprocessor_config.json` in the snapshot — recon file list): JevX's modality detection
   will see vision on the wrapper; a text-only probe should target the language submodel.
8. **Risks to carry.**
   - *State decay + padding*: harmless for single-forward pooling (§6), relevant if anyone ever
     enables caching or chunked inputs.
   - *Backends*: full layers accept eager/sdpa/flash (`:685-687, :805-806`); `output_attentions=True`
     forces eager and raises otherwise (audit §8.6, `configuration_utils.py:406-415`) — JevX never
     requests attentions, keep it that way.
   - *Output gate & head geometry*: probes must not assume `head_dim = hidden/heads`
     (`head_dim=256` is configurable per-layer, `:639`), and should know the sigmoid gate at
     `:701` is unconditional (config flag unconsumed, §2.3).
   - *Dead config fields*: `attn_output_gate` and `mamba_ssm_dtype` have zero consumers in
     transformers 5.16.1 — do not build behavior on them.
9. **What JevX must NOT assume (spec/audit §5 lineage).** The audit's §5 was written for uniform
   softmax stacks; hybrids show the uniformity assumption is false in three ways: (i) **mask type
   varies per layer** — a 2-D mask for `linear_attention`, a 4-D causal mask for `full_attention`,
   both derived from the *same* user 2-D mask (`modeling_qwen3_5.py:1197-1209`); (ii) **cache
   state varies per layer** — KV vs fixed recurrent state inside one `DynamicCache`
   (`cache_utils.py:1223-1244`); (iii) **position encoding varies per layer** — MRoPE on full
   layers, none on linear layers (`:770-786`). Downstream math (Definition 4/5 pooling and fusion)
   is agnostic to these differences *provided* the mask contract holds; the layer-type knowledge
   lives entirely in transformers' mask/cache dispatch, and JevX's job is only to hand it a correct
   2-D mask, correct `position_ids`, and `use_cache=False`.

---

## 8. Citations and non-verification register

### Code (all `vendor/transformers/src/transformers/...` unless noted)

- `models/qwen3_5/configuration_qwen3_5.py:103,110-121`; `models/qwen3_5/modeling_qwen3_5.py:167-196,248-380,387-546,557-604,607-704,743-796,805-807,1151-1221,1578-1584,1675-1726`
- `models/qwen3_next/modeling_qwen3_next.py:512-699,807-811,946-973`
- `cache_utils.py:36-42,891-1000,1003-1091,1094-1198,1215-1259,1500-1560,1701-1793`
- `masking_utils.py:1474-1520`
- `models/deepseek_v4/configuration_deepseek_v4.py:21-35,99-173,246-300`;
  `models/deepseek_v4/modeling_deepseek_v4.py:162-293,453-,741-864,1096-1144,1256-1328`
- `models/deepseek_v32/configuration_deepseek_v32.py:26-145`; `models/deepseek_v32/modeling_deepseek_v32.py:160-232,351-497,756`
- `models/deepseek_vl_hybrid/configuration_deepseek_vl_hybrid.py:33-87`;
  `models/deepseek_vl_hybrid/modeling_deepseek_vl_hybrid.py:192-267`
- Taxonomy rows as cited in §1 table.
- JevX: `jevx/adapters/decoder.py:12-19,34-75`; `jevx/adapters/base.py:250-266`;
  `jevx/capabilities/detector.py:275-340`; `jevx/representations/pooling.py:1-140`;
  `reports/TRANSFORMERS_SOURCE_AUDIT.md:87-96,139-153,166-184`; `vendor/reports/QWEN35_RECON.md`.
- Empirical: seeded script run 2026-09-24 (torch 2.14.0+cpu, transformers 5.16.1, `kernels`
  absent); numbers in §6.3.

### Web (accessed 2026-09-24)

1. **VERIFIED** — Qwen3.5-0.8B model card: https://huggingface.co/Qwen/Qwen3.5-0.8B (hybrid
   layout quote, GDN/attention head counts, rotary dim 64, MTP multi-steps + serving configs,
   262,144 context, benchmarks-vs-other-models).
2. **UNVERIFIED** — Qwen3.5 blog: https://qwen.ai/blog?id=qwen3.5 (JS-rendered; fetch returned no
   article text — no rationale/benchmark claims could be read).
3. **VERIFIED** — DeepSeek-V4 paper abstract: https://arxiv.org/abs/2606.19348 (CSA+HCA hybrid,
   mHC, Muon, sizes, 27%-FLOPs/10%-KV claim vs V3.2).
4. **VERIFIED** — DeepSeek-V4-Flash-Base repo: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Base
   (exists; 292B params; "No model card").
5. **VERIFIED** — Gated Delta Networks: https://arxiv.org/abs/2412.06464 (Yang, Kautz,
   Hatamizadeh; ICLR 2025; gated delta rule = gating + targeted updates).
6. **VERIFIED** — DeltaNet: https://arxiv.org/abs/2406.06484 (parallel training over sequence
   length; retrieval shortcoming of linear transformers/SSMs).
7. **VERIFIED** — RetNet: https://arxiv.org/abs/2307.08621 (parallel/recurrent/chunkwise
   retention, O(1) inference).
8. **VERIFIED** — Mamba2/SSD: https://arxiv.org/abs/2405.21060 (attention–SSM duality).
9. **VERIFIED** — GLA: https://arxiv.org/abs/2312.06635 (data-dependent gating,
   hardware-efficient training).
10. **VERIFIED** — MiniMax-01 / MiniMax-M1: https://arxiv.org/abs/2501.08313 and
    https://arxiv.org/abs/2506.13585 (lightning attention; M1's "hybrid-attention" self-description).

### What we did not verify

- The Qwen3.5 design *rationale* and any hybrid-vs-pure-attention ablation numbers (blog
  unfetchable; card has no same-size ablation) — §5 tags these UNVERIFIED.
- Behavior of optional fused hub kernels (`use_kernel_forward_from_hub("Qwen3_5GatedDeltaNet")`,
  FLA kernels) — `kernels` package not installed; all findings describe the pure-torch reference
  path. Whether they consume `mamba_ssm_dtype`/`attn_output_gate` (both grep-dead in transformers)
  is unknown.
- The shipped DeepSeek-V4 checkpoint config.json (only class defaults + HF metadata were read);
  the full V4 paper body (abstract only); third-party V4 follow-up papers (e.g. claims of a
  37-layer 56/8 Local/Global split from arXiv:2609.07008 conflict with the in-tree default of 43
  layers — not adjudicated).
- Quantitative decay-collapse magnitudes in §6.3 come from *random-init* parameters; trained
  checkpoints' $A$, `dt_bias` (and hence per-step $\lambda$) were not measured — structural
  claims are kernel-exact, numeric factors are illustrative.
- Trained-model benchmark reproduction of any kind (no training runs, no long-context evals).
- `minimax_m2`/`minimax_m3_vl` internals beyond confirming m2 has no linear layers;
  `hy_v3`/`inkling`/`step3p7` patterns were surveyed only at keyword level.
