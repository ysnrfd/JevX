# JevX Architecture

**Scope.** Layered architecture of the JevX conversion+decision pipeline, the eleven named
components, twelve diagrams (conversion paths per modality, conditioning, parallelism,
calibration, abstention, training), the two training modes, and the honest distinction between
*representation reuse* and *weight transplant*. Labels: [FACT] observable in this
repository/vendor tree; [DESIGN] a choice of this project; [HYPOTHESIS] — see
``RESEARCH.md`` `12`. Mathematical objects referenced as Def. *n* / T*n* are defined in
``MATHEMATICAL_SPECIFICATION.md`` and proved in ``PROOFS.md``.

**Repository status note [FACT — updated 2026-09-24]:** the Phase-2 note that ``jevx/`` was
an empty skeleton is stale: the package is now implemented and the design below describes the
code as built (reference: docs/API.md; suite counts in docs/TESTING.md §1 — the pre-fix-wave
runs quoted there measured 768 passed, 1 skipped; the fix wave collects 828 and its verbatim
summary is refreshed in reports/FINAL_VERIFICATION.md).

---

## 1. Layers

| Layer | Name | Contents | Failure if absent |
|---|---|---|---|
| L0 | **Substrate** | Python 3.14, torch 2.14.0+cpu, ``transformers`` 5.16.1 (audited vendor tree ``vendor/transformers`` [FACT]) | — |
| L1 | **Model & capability** | `CapabilityRegistry`, `ModelAdapter` — which model, what modality, what pooling, support level 0–4 | silent misuse of an incompatible backbone |
| L2 | **Input & representation** | `InputAdapter` (tokenize/featurize + masks, Def. 1), `RepresentationAdapter` (backbone forward + pooling choice, Def. 4) | garbage hidden states; padding leaks |
| L3 | **Question & schema & fusion** | `QuestionEncoder` (Def. 2→4), `SchemaEncoder` (Def. 3→4), `DecisionFusion` (Def. 5) | unconditioned decisions; wrong option wiring |
| L4 | **Typed decision** | `TypedDecisionHeads` ($G_{\text{type}}$, $\Pi_{\text{type}}$, Defs. 6–7) | outputs outside the declared type |
| L5 | **Post-processing & guardrails** | `CalibrationEngine` (Def. 8), `ConfidenceEngine` (Def. 9, abstention Def. 10), `ValidationEngine` (invariants D8.I, (11.3), T7 tolerances) | uncalibrated thresholds; schema drift after calibration; uncaught type escapes |
| L6 | **Training** | `training` subpackage: Mode A / Mode B loops (`6` below) | heads untrainable/untunable |
| L7 | **Auto layer** | `auto`: model-class resolution via the vendor auto mappings [FACT: ``vendor/reports/auto_model_mapping.json`` keyed by ``model_type``] | hand-wired model loading, brittle across 510 families |

Data flows downward-to-upward as: raw state/question/schema → tensors → hidden states →
logits → probabilities → calibrated typed decision (+ confidence/abstention side-channel).

---

## 2. Components (eleven named)

| Component | Responsibility | Consumes → produces | Owns invariants |
|---|---|---|---|
| **CapabilityRegistry** | Probes a model class for: input modality, hidden-state availability, mask support, pooling options; assigns support level 0–4 (`7`) | model id → capability record | conversion refused below level 1 |
| **ModelAdapter** | Loads backbone weights (reuse path), places in eval mode, exposes one `forward_states(X, M) → (Z, M)` interface across architectures | checkpoint → uniform hidden states | eval-mode determinism (A7.1) |
| **InputAdapter** | Modality featurization: tokenization / image processing / audio feature extraction; padding + masks (Def. 1) | raw $S$, $Q$ text → $(X, M)$ | axioms D1.1–D1.4 |
| **RepresentationAdapter** | Chooses pooling per capability record (Def. 4a–e); retains unpooled sequence when cross-attention needs it | $(Z, M)$ → $H_S$ / $H_S^{\text{seq}}$ | mask respect P4.2, padding invariance P4.3 |
| **QuestionEncoder** | Encodes $Q.\textit{text}$ — default hashed bag-of-tokens projection (never touches the backbone, preserving the one-forward property); optional `method="backbone"` mode runs a shared/separate text backbone (extra forward — spec Def. 4), pools to $h_Q$ | $Q$ → $h_Q$ | metadata non-interference D2.2 |
| **SchemaEncoder** | Encodes each option independently ($k \mapsto H_{C,k}$), schema order preserved; option-axis mask for $K$-padding | $C$ → $H_C \in \mathbb{R}^{K\times d}$ | schema axioms D3.1–D3.4 |
| **DecisionFusion** | One of five variants (Def. 5a–e — concat, gated, additive, bilinear, cross-attention — all registered under `fusion_type`, plus `dot`/`mlp` extras; `FUSION_REGISTRY` in `jevx/representations/fusion.py`); pure function of $(H_S, h_Q, H_C)$ (cross-attention additionally consumes the retained state sequence). The $s_C$ schema-summary summands of 5.a–5.d are carried by the per-option formula ($e_k = g \odot W_o H_{C,k} + (1-g)\odot c$) instead — no $\phi_C(s_C)$ term (spec 5: "implementation form is normative") | states → logits $z \in \mathbb{R}^K$ | purity D5.1 (feeds T7) |
| **TypedDecisionHeads** | $G_{\text{type}}$ (softmax/sigmoid/cumulative/PL, Def. 6) and $\Pi_{\text{type}}$ (Def. 7) selected by $Q.\textit{type}$ | logits → typed decision | T1–T4, T8 membership |
| **CalibrationEngine** | fit (8.3/8.4) on held-out cal set; calibrate at inference; evaluate (8.5) | $(z, y)_{\text{cal}}$ → fitted calibrator; $p \to p'$ | type/schema preservation D8.I |
| **ConfidenceEngine** | Functionals (9.1)–(9.4); abstention overlay (10.1)–(10.4) | $p'$ (or $z$) → $c$, risk–coverage stats | confidence ≠ correctness (non-claim #2) |
| **ValidationEngine** | Runtime assertions: schema-hash equality across calibration; type-tag equality; per-question $\sum_i p_i = 1 \pm$ tol; masked $p = 0$; T7 batch-vs-loop tolerance checks in tests | every boundary call | D8.I, (11.3), implementation invariants |

Module locations reserved [DESIGN]: `jevx/adapters`, `jevx/representations`,
`jevx/questions`, `jevx/decisions`, `jevx/calibration`, `jevx/confidence`,
`jevx/capabilities`, `jevx/training`, `jevx/conversion`, `jevx/auto`, `jevx/utils`.

---

## 3. Diagrams (12)

### 3.1 Overall JevX

```mermaid
flowchart LR
  subgraph EXT["External / L0-L1"]
    CKPT["Pretrained checkpoint (HF)"]
    REG["CapabilityRegistry: probe + support level 0-4"]
    MA["ModelAdapter: load weights, eval mode"]
    CKPT --> REG --> MA
  end
  subgraph IN["Inputs"]
    S["State S"]
    Q["Question Q: text, type, constraints, metadata"]
    C["Schema C: options O_1..O_K, K variable"]
  end
  subgraph L2L4["L2-L4 pipeline (per call)"]
    IA["InputAdapter: featurize + mask"]
    RA["RepresentationAdapter: backbone + pooling -> H_S"]
    QE["QuestionEncoder -> h_Q"]
    CE["SchemaEncoder -> H_C"]
    FUS["DecisionFusion (Def. 5a-5e variants + dot/mlp) -> logits z"]
    HEAD["TypedDecisionHeads: G_type, Pi_type"]
    IA --> RA --> FUS
    QE --> FUS
    CE --> FUS
    HEAD --> FUS
  end
  subgraph L5["L5 post-processing"]
    CALM["CalibrationEngine"]
    VAL["ValidationEngine"]
    CFGE["ConfidenceEngine"]
    ABS["Abstention rule vs kappa"]
  end
  D["Typed decision D"]
  S --> IA
  Q --> QE
  C --> CE
  MA -.->|hidden states interface| RA
  FUS --> CALM --> D
  CALM -.->|D8.I schema/type check| VAL
  D -.-> CFGE -.-> ABS
```

*Caption.* Solid arrows: data path. Dotted: control/monitoring. Note the single fan-in of
`H_S` into `FUS` (computed once — Def. 11) and that `CalibrationEngine` sits strictly
between probability construction and typed projection (Eq. 7.9).

### 3.2 Decoder-only conversion (GPT-family, causal LMs)

```mermaid
flowchart TD
  TXT["Token ids"] --> TOKPOS["Token + position embeddings"]
  TOKPOS --> BLK["Causal decoder blocks x N: masked self-attention (prefix only)"]
  BLK --> Hs["Hidden states Z: n x d, causal context"]
  Hs --> POOL["RepresentationAdapter: P_last (last-valid index i_b = max t where mask=1)"]
  POOL --> HS1["H_S pooled vector (optionally keep Z seq)"]
  HS1 --> HEADS["Question/schema conditioning (3.8)"]
  NOTE1["No bidirectional attention: every position sees only its prefix; pooling must use the final valid position to carry full context"]
  NOTE1 -.-> POOL
```

*Notes.* [FACT] Causal LMs expose last hidden states through the common output contract
(``BaseModelOutput``, ``vendor/transformers/src/transformers/modeling_outputs.py``).
[DESIGN] `P_last` (Def. 4d) is the default; `P_mean` over valid positions is a fallback
probed by the registry.

### 3.3 Encoder conversion (BERT-family, bidirectional)

```mermaid
flowchart TD
  TXT["Token ids"] --> EMB["Token + type + position embeddings"]
  EMB --> BLK["Bidirectional transformer blocks x N (full attention + mask)"]
  BLK --> Hs["last_hidden_state: n x d"]
  Hs --> P1["Option A: P_first (CLS index t_min)"]
  Hs --> P2["Option B: P_mean over valid positions"]
  Hs --> P3["Option C: native pooler (dense + tanh on first token)"]
  P1 --> HS1["H_S"]
  P2 --> HS1
  P3 --> HS1
  NOTE2["FACT: BertPooler pools the first-token hidden state through dense+tanh (vendor/transformers/src/transformers/models/bert/modeling_bert.py line 451)"]
  NOTE2 -.-> P3
```

*Notes.* [DESIGN] registry probes which of A/B/C the model provides; mean-pooling is the
portable default, native poolers are used when exposed.

### 3.4 Encoder-decoder conversion (BART/T5-family)

```mermaid
flowchart TD
  SRC["Source tokens"] --> ENC["Encoder stack: bidirectional self-attention"]
  ENC --> EOUT["Encoder hidden states H_enc"]
  EOUT --> POOL["Pooling over source positions (P_mean / P_first)"]
  POOL --> HS1["H_S (representation-reuse path)"]
  DEC["Decoder stack + cross-attention + LM head"] -.->|unused in Mode A reuse: no generation, no target sequence| X["not part of JevX decision path"]
  EOUT -.->|optional alternative: decoder-side forward over source only, capability-probed| POOL
  HS1 --> HEADS["Question/schema conditioning (3.8)"]
  NOTE3["DESIGN: JevX consumes the ENCODER representation; the autoregressive decoder is out of scope for typed decisions (contrast 10.3 of RESEARCH.md)"]
  NOTE3 -.-> DEC
```

*Notes.* [DESIGN] two capability-probed sub-options: (a) encoder forward only (default),
(b) full model forward with a minimal prefix to exercise decoder-side layers — never
free-text generation. Registry records which sub-option the checkpoint supports.

### 3.5 Vision conversion (ViT-family, patch models)

```mermaid
flowchart TD
  IMG["Image"] --> PROC["Image processor: resize, normalize, patchify"]
  PROC --> PE["Patch embeddings + CLS token + pos embeddings"]
  PE --> BLK["Vision transformer blocks x N"]
  BLK --> Hs["last_hidden_state: num_patches(+1) x d"]
  Hs --> P1["P_first (CLS entry)"]
  Hs --> P2["P_mean over patch positions"]
  P1 --> HS1["H_S"]
  P2 --> HS1
  NOTE4["FACT: ViT exposes pooler-style and classification-head paths (vendor/transformers/src/transformers/models/vit/modeling_vit.py, ViTForImageClassification)"]
  NOTE4 -.-> P1
```

*Notes.* Question text still flows through `QuestionEncoder`; only the *state* tower is
vision (the question stays textual — [DESIGN] default; image-conditioned questions are a
registry-flagged extension, unimplemented).

### 3.6 Audio conversion (spectrogram/frame models)

```mermaid
flowchart TD
  WAV["Waveform"] --> FE["Feature extractor: frames / mel spectra, pad + mask"]
  FE --> EMB["Frame embeddings (+positional)"]
  EMB --> BLK["Temporal encoder blocks x N"]
  BLK --> Hs["last_hidden_state: num_frames x d"]
  Hs --> P1["P_mean over valid frames"]
  Hs --> P2["P_last final valid frame (causal variants)"]
  Hs --> P3["P_att learned attention pooling over frames"]
  P1 --> HS1["H_S"]
  P2 --> HS1
  P3 --> HS1
  NOTE5["DESIGN: frame-axis mask is the Def-1 mask; pooling choice keyed by capability record (AST-style models: mean; causal streaming models: last-valid)"]
  NOTE5 -.-> P2
```

*Notes.* [FACT] audio model families present in the audited tree include
``audio_spectrogram_transformer``, ``wav2vec2``, ``whisper``, ``encodec`` under
``vendor/transformers/src/transformers/models/``. Support level still 0 until probed.

### 3.7 Multimodal conversion (dual-tower, CLIP-family)

```mermaid
flowchart TD
  subgraph T1["Text tower"]
    TX["Text tokens"] --> TB["Text transformer blocks"] --> TP["P_first / native pooler -> t"]
  end
  subgraph T2["Vision tower"]
    IM["Image"] --> VB["Patch blocks"] --> VP["P_cls or P_mean -> v"]
  end
  TP --> PROJ1["Text projection"]
  VP --> PROJ2["Vision projection"]
  PROJ1 --> FUSE["State fusion: concat [t v] -> MLP (or gated sum)"]
  PROJ2 --> FUSE
  FUSE --> HS1["H_S (multimodal state representation)"]
  NOTE6["FACT: CLIP exposes get_text_features / get_image_features (vendor/transformers/src/transformers/models/clip/modeling_clip.py lines 674, 710)"]
  NOTE6 -.-> PROJ1
```

*Notes.* [DESIGN] two sub-modes: (a) *joint-state*: towers fused once into a single $H_S$
shown here; (b) *tower-selective*: capability record maps question modality to the tower
that conditions it. Both keep Def. 11 semantics (one shared $H_S$ per state, per-question
purity).

### 3.8 Question and schema conditioning

```mermaid
flowchart LR
  subgraph QP["Question path"]
    QT["Q.text"] --> QTOK["tokenizer -> ids, mask"] --> QENC["QuestionEncoder: backbone or shared tower"] --> HQ["h_Q (pooled)"]
    QC["Q.type, constraints"] --> ROUTE["type router: selects G_type and Pi_type"]
  end
  subgraph CP["Schema path, per option k = 1..K"]
    OK["O_k: key, label, description, meta"] --> OENC["Option encoder g_psi (shared weights)"] --> HK["H_C row k"]
    OK -.-> KEYMAP["index k <-> key (bijection, D3.2)"]
  end
  HS["H_S (shared state)"] --> FUS["DecisionFusion: concat / gated / additive / bilinear / cross-attention / dot / mlp"]
  HQ --> FUS
  HK --> FUS
  FUS --> LOG["logits z_1..z_K (option-wise scoring psi_phi)"]
  LOG --> ROUTE --> OUT["G_type -> p ; Pi_type -> typed decision over C"]
```

*Notes.* [DESIGN] Option encoder weights are shared across options and questions (variable-K
friendly, Def. 5.0). The type router is where `Q.type` first touches the math: it selects
both the probability map and the projection, so an incorrectly declared type is a declared
contract violation (AS8), caught by schema/type validation rather than silently coerced.

### 3.9 Parallel inference (shared H_S, J questions)

```mermaid
flowchart TD
  S["State S"] --> IME["InputAdapter + backbone (ONCE)"] --> HS["H_S"]
  subgraph BATCH["Batch of J parallel-safe questions (option axis padded to K_max)"]
    Q1["Q_1, C_1"] --> P1["path 1"]
    Q2["Q_2, C_2"] --> P2["path 2"]
    QJ["Q_J, C_J"] --> PJ["path J"]
  end
  HS --> P1
  HS --> P2
  HS --> PJ
  P1 --> D1["D_1"]
  P2 --> D2["D_2"]
  PJ --> DJ["D_J"]
  INV["Invariants: each path depends only on H_S, Q_j, C_j (T7 A7.3); masked softmax per question (11.2); sum over options = 1 PER QUESTION, never across j (11.3); batched == loop within tolerance (T7-ii)"]
  INV -.-> BATCH
```

*Notes.* [DESIGN] Functional independence only: outputs are not statistically independent —
shared `H_S` and shared parameters couple them (P11.2; see LIMITATIONS).

### 3.10 Calibration

```mermaid
flowchart LR
  subgraph OFF["Offline fit (after heads trained, before serving)"]
    TR["Training set (fit phi)"] --> HD["Trained heads produce p on held-out CAL set"]
    HD --> FIT["fit: T* = argmin sum CE  (or PAVA isotonic, or vector scaling)"]
    FIT --> CALP["fitted calibrator params (per head / per type bucket)"]
  end
  subgraph ON["Online apply (per decision)"]
    Z["logits z"] --> P["p = G_type(z)"] --> APPLY["p' = cal(p)  (same K, same schema, same index order)"] --> PROJ["Pi_type(p', C) -> D"]
  end
  CALP -.->|stored params| APPLY
  APPLY --> EVAL["evaluate: ECE_B, NLL, Brier, reliability diagram on disjoint eval split"]
  GUARD["ValidationEngine: schema hash before == after; type tag unchanged (D8.I); argmax-change report for non-order-preserving fits"]
  APPLY -.-> GUARD
```

*Notes.* [DESIGN] Calibration is *post-hoc and disjoint*: never trained jointly with heads
(DC-10); the fit/eval protocol is Def. 8. The guard exists because D8.I is an invariant to
enforce, not merely to hope for.

### 3.11 Abstention (confidence threshold, risk-coverage)

```mermaid
flowchart TD
  PPRIME["p' (calibrated)"] --> CONF["ConfidenceEngine: c in {max-prob, margin, norm-entropy, energy}"]
  CONF --> CMP{"c >= kappa ?"}
  CMP -->|yes| DEC["emit typed decision D"]
  CMP -->|no| ABST["emit abstain symbol (perp); record reason code"]
  DEC --> LOG1["(loss l, confidence c, covered=1) -> online ledger"]
  ABST --> LOG2["(covered=0) -> online ledger"]
  LOG1 --> RC["risk-coverage curve; coverage-at-risk rho*(alpha) per (10.2)-(10.4)"]
  LOG2 --> RC
  KSET["kappa chosen on validation split; selection method always reported"]
  KSET -.-> CMP
```

*Notes.* [DESIGN] Threshold semantics per (10.1); the ledger records everything needed for
Def. 10 reporting without recomputation. [STATISTICAL] Any risk number read off this curve
is a sample estimate under exchangeability — not a per-instance guarantee (non-claim #2).

### 3.12 Training (Mode A and Mode B)

```mermaid
flowchart TD
  DATA["(S, Q, C, y): labels for typed tasks"] --> LOSSF["Typed loss: CE for Choice/Score levels; BCE for Noul/MultiLabel; PL for Ranking (strictly proper family, T5/T6)"]
  subgraph MODEA["Mode A: frozen backbone (default)"]
    BB1["Backbone theta: FROZEN (no grads, eval stats)"] --> RA1["RepresentationAdapter"] --> AD1["Trainable: encoders/fusion/heads phi"] --> LOSSF
  end
  subgraph MODEB["Mode B: fine-tune (optional)"]
    BB2["Backbone theta: TRAINABLE, small LR, optional layer-wise unfreeze"] --> RA2["RepresentationAdapter"] --> AD2["Trainable phi (usually lower LR than phi)"] --> LOSSF
  end
  LOSSF --> OPT["optimizer step on phi (and theta in Mode B)"]
  OPT --> DONE["trained heads -> freeze -> CALIBRATION (3.10, fit on disjoint split) -> eval"]
  DONE -.->|never backprop through calibration| GAP["calibrator fitted after, not with"]
```

*Notes.* [DESIGN] One loss line, two modes: the loss never sees the calibrator; calibration
always comes after (D8.I placement). Mode B adds optimizer state over the backbone — on the
CPU-only box this caps practical model size (LIMITATIONS).

---

## 4. Training modes: Mode A (frozen backbone) vs Mode B (fine-tuned)

| Dimension | Mode A — representation reuse, frozen | Mode B — fine-tuned |
|---|---|---|
| Backbone parameters $\theta$ | frozen: not in optimizer, `eval` mode, gradients blocked | trainable: included in optimizer, usually smaller LR than heads; optional layer-wise unfreeze schedule |
| Trainable parameter count | adapters + encoders + fusion + heads ($O(d^2)$-scale at most) | + backbone ($O(\text{params})$) |
| What is preserved | pretrained representation *by construction* (identical weights, identical outputs for identical inputs) | representation drifts by gradient descent — measurable via probe/regression suite |
| Catastrophic forgetting | impossible for $\theta$ (never updated) | possible; controlled only empirically (H3) |
| Compute on this box (CPU-only) | feasible for adapter-scale training on small/medium backbones | constrained: gradient+optimizer memory over full backbone; large models impractical [FACT: 4 threads, ~8 GB RAM] |
| Calibration | fitted post-hoc either way | fitted post-hoc either way (never joint) |
| Default | **yes (DC-2)** | opt-in per experiment, must report forgetting checks |

[DESIGN] Both modes share the identical L2–L5 stack; only the optimizer's parameter groups
differ, which keeps ablations apples-to-apples (H2).

---

## 5. Representation reuse vs weight transplant (the honest distinction)

**Claim in one line:** JevX performs **representation reuse** with *surgical* weight
transplant where a genuine counterpart exists; a true 1:1 weight conversion of the *whole
system* is **impossible**, because most of the decision system has no pretrained weights to
convert.

**What "conversion" can and cannot mean here.**

1. **Backbone (possible, 1:1).** Loading a checkpoint's encoder/decoder weights into the
   same architecture is a genuine identity map: every pretrained tensor maps to itself
   (`safetensors` keys unchanged). [FACT] `transformers` itself performs weight conversion
   only *between related architectures/versions* through explicit mapping tables (observed:
   ``vendor/transformers/src/transformers/conversion_mapping.py``, ``core_model_loading.py``,
   ``safetensors_conversion.py``) — none of these invents weights; they relocate existing ones.
2. **New heads (impossible, 1:1 by construction).** `DecisionFusion`, `TypedDecisionHeads`,
   `CalibrationEngine` (fitted parameters), `ConfidenceEngine` thresholds, and the encoders'
   pooling projections are **new modules with no counterpart in the source checkpoint**. There
   is no pretrained tensor named for a schema-conditioned bilinear fusion of
   $(H_S, h_Q, H_C)$ because the checkpoint never saw questions or schemas. Their weights are
   *initialized* (fresh) and *learned* (Mode A/B) — not converted. Claiming a bijection here
   would be a category error, and this project does not make it.
3. **Head grafting (possible, narrow).** When the source checkpoint *does* contain a
   compatible task head (e.g., a fine-tuned classifier with matching output dimension, or a
   pooler with matching shape), JevX may transplant those tensors as initialization for the
   corresponding adapter piece — a *warm start*, still subject to validation. This is
   reported as transplant, never as identity: shapes must match exactly, semantics are
   re-checked, and the transplanted head is re-fit or frozen by explicit config.
4. **Vocabulary/tokenizer and processor** are reused as-is (they are data-contract, not
   weights), pinned by version to the audited tree. [FACT]

**Terminology used in JevX docs** [DESIGN]:
`conversion` = capability probe + architecture wrap + backbone load (+ optional grafting)
+ fresh head init — *not* "the checkpoint becomes a decision model bit-for-bit". Where a
simpler statement is needed: *the backbone is transplanted; the decision system is built
around it.*

---

## 6. CapabilityRegistry and support levels (0–4)

[DESIGN, referenced by LIMITATIONS] Each probed `model_type` (from the audited auto-mapping
of 510 families [FACT]) gets exactly one level.

**Normative scale — one source of truth.** The definitions of levels 0–4 live in the
"Support levels (defined precisely)" table of
[TRANSFORMERS_COMPATIBILITY_MATRIX.md](../TRANSFORMERS_COMPATIBILITY_MATRIX.md) (repo
root). That table wins any conflict with any other document; this section and
``LIMITATIONS.md`` §7 only re-state it. Quoted once, condensed:

| Level | Meaning (condensed from the matrix's normative table) |
|---|---|
| **0 — Unsupported** | No standard extraction path through the public API (no audited family landed here). |
| **1 — Source-verified only** | Contract read in source but not forward-probed here, or the environment could not exercise the input pipeline, or the checkpoint is not HF-API verified. |
| **2 — Convention-dependent (lossy pooling)** | Extraction mechanics fully standard and probe-verified, but the pooled-vector semantics depend on a convention JevX must pin (which pooling, which layer, which positions); usable, quality is checkpoint- and pooling-choice-dependent. |
| **3 — Native states + standard pooling** | Forward output probe-verified; JevX applies one generic mask-aware pooling that is the established convention for the family; checkpoint HF-API verified — plus, per the Phase 92-94 evidence overlay, load → forward → save/load equality → output invariants → real-data input **executed on the cited checkpoint**. |
| **4 — Native pooled vector + quality gate** | A trained/pooled representation is a named output field, probe-verified, checkpoint HF-API verified — **plus a real-dataset quality result** (Accuracy/F1/NLL/Brier/ECE) for that family's checkpoint from the benchmark suite. |

Rules: levels are monotone (4 implies 3 implies 2 implies 1); a failure demotes to the
highest passing level; level 4 is always scoped to the environment it was measured in
(this CPU-only box — see LIMITATIONS). A registry entry is a **[FACT] only at the level the
evidence supports**; aspirational levels are never pre-assigned. Current headline
distributions (13 families at level 3, 7 at level 2, none at 1/0) are computed on this
matrix scale — see the matrix and ``reports/REAL_MODEL_VALIDATION.md``.

---

## 7. Grounding map (reserved module ↔ specification)

| Reserved location [FACT: directory exists] | Specification anchor |
|---|---|
| ``jevx/adapters`` | ModelAdapter, InputAdapter, RepresentationAdapter; Def. 1, 4 |
| ``jevx/representations`` | pooling operators P4.1–P4.3; Def. 4 |
| ``jevx/questions`` | QuestionEncoder; Def. 2 |
| ``jevx/decisions`` | SchemaEncoder, DecisionFusion, TypedDecisionHeads; Defs. 3, 5–7 |
| ``jevx/calibration`` | CalibrationEngine; Def. 8, D8.I |
| ``jevx/confidence`` | ConfidenceEngine, abstention; Defs. 9–10 |
| ``jevx/capabilities`` | CapabilityRegistry; levels 0–4 (section 6) |
| ``jevx/training`` | Mode A / Mode B; section 4 |
| ``jevx/conversion`` | conversion + grafting semantics; section 5 |
| ``jevx/auto`` | auto class resolution; `vendor/reports/auto_model_mapping.json` |
| ``vendor/transformers`` | audited L0 dependency, tag v5.16.1 |

All twelve diagrams above are design intent; no diagram implies code exists today.
