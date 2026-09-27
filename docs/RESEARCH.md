# JevX Research Foundation

**Purpose.** This document states the research basis of JevX: what is known, what is not,
what must be assumed, and why each architectural choice was made. It is deliberately split
into three audited sections — **JEV FACTS**, **JEVX DESIGN CHOICES**, **JEVX HYPOTHESES** —
plus the background that motivates them.

**Rules of evidence used here.**

* [FACT] = publicly documented / directly observable, with a citation you can check.
  Citations use numbered references [R#] listed at the end. Nothing is cited that the author
  cannot support; where knowledge is absent, the text says so plainly.
* [DESIGN] = a decision of this project; defensible, revisable, not evidence.
* [HYPOTHESIS] = a falsifiable expectation; only an experiment upgrades it to [EMPIRICAL].
* Reverse-engineered inferences about any proprietary system are never presented as facts.
* Banned vocabulary: *revolutionary, unbeatable, universally superior, guaranteed better.*

---

## 1. JEV FACTS — what is publicly documented about "Jev"

**Q1 (What is Jev, conceptually?) — answer: nothing public is known.** [FACT about knowledge
state]

1. As of this writing, this project has **no access to any public documentation of a system
   named "Jev"** — no paper, technical report, package, model card, or specification under
   that name is known to us that describes a probabilistic typed decision system of the kind
   JevX targets. If "Jev" is an internal/proprietary codename, its architecture, training
   procedure, training data, and objectives are **unknown to this document**, and we do not
   speculate about them. Any later claim that "JevX reproduces Jev" would require evidence
   this repository does not have.
2. What *is* observable in this repository (all [FACT], from
   ``REPOSITORY_RECONNAISSANCE.md``, observed 2026-09-24): the working directory
   ``C:/Users/ysnrfd/Documents/Transformer/JevX`` contained, at reconnaissance time, a documentation skeleton
   (``jevx/`` package with empty ``__init__.py`` files, empty ``tests/`` and ``docs/``
   directories), a shallow clone of Hugging Face ``transformers`` at tag **v5.16.1**
   (``vendor/transformers``, commit ``93c8b7b4…``), and an auto-model mapping extract
   (``vendor/reports/auto_model_mapping.json``, keyed by ``model_type`` with base-model and
   config class names). The environment: Python 3.14.3, ``transformers 5.16.1``,
   ``torch 2.14.0+cpu`` (no CUDA, no MPS), 4 logical CPU threads, ~8 GB RAM. A sibling
   directory ``JevX_Transformer`` exists in the parent folder; its contents are unrelated to
   the facts above and were not mined for assumptions here.
3. Consequently, everything below that describes *JevX's* behavior is a property of **this
   design**, not of any "Jev" system. The conceptual identity JevX commits to is modest and
   self-contained: *convert a pretrained Hugging Face model into a frozen-or-tuned
   representation engine, and evaluate typed, schema-constrained probabilistic decisions with
   it — in one non-autoregressive pass, optionally calibrated and optionally selective
   (abstaining).*

**Summary of the JEV FACTS section:** the only facts available are environment/repository
facts and the plain statement of ignorance above. There is nothing else to report honestly.

---

## 2. What is known [Q2] — background with citations

The following are established results JevX builds on; each is standard and citable, none is a
claim about JevX itself.

1. **Softmax neural classifiers are often mis-calibrated post training.** [FACT] Guo et al.
   [R1] showed modern CNNs can be strikingly overconfident and that **post-hoc temperature
   scaling** — a single scalar fitted on held-out data — recovers most of the calibration gap
   without touching accuracy (temperature scaling preserves argmax, cf. T-remarks in
   ``MATHEMATICAL_SPECIFICATION.md`` `8`). Vector scaling (per-class affine on logits) and
   isotonic regression appear there as baselines; isotonic regression for classifier-score
   calibration goes back to Zadrozny & Elkan [R5] and Niculescu-Mizil & Caruana [R6];
   Platt scaling [R7] for binary SVM scores. Mis-calibration patterns persist across modern
   architectures and modalities (vision: Minderer et al. [R9]; NLP: Desai & Durrett [R8]),
   and uncertainty quality degrades under shift (Ovadia et al. [R10]).
2. **Proper scoring rules.** [FACT] The log score and the Brier score [R2] are strictly
   proper: expected score is uniquely minimized by reporting the truth (T5, T6; Gneiting &
   Raftery [R4]). Murphy's vector partition gives the reliability–resolution–uncertainty
   decomposition used in T6 [R3].
3. **Calibration measurement has known estimator pitfalls.** [FACT] ECE with fixed bins is a
   biased, binning-dependent estimator [R26, R27]; calibration should be evaluated with
   stated binning and ideally multiple metrics [R28]; multiclass variants beyond temperature
   (matrix/Dirichlet) exist but add parameters that can overfit small calibration sets [R29].
4. **Selective prediction / reject options are classical.** [FACT] Chow's reject rule and the
   accuracy–reject tradeoff date to the 1950s–70s [R12]; selective classification for deep
   networks formalizes risk–coverage curves [R11]; abstention is a first-class decision type
   there and in conformal/prediction-set lines of work [R24, R25].
5. **Logit/activation-based OOD signals exist but are score-like, not probabilities.**
   [FACT] Energy scores [R13] and earlier activation baselines [R30] rank (in)confidence;
   they require thresholding/calibration before any probabilistic reading — which is exactly
   how JevX Definition 9(d) treats them.
6. **Encoders and dual-encoder scoring are mature.** [FACT] Transformers [R31]; BERT-style
   bidirectional encoders [R32]; decoder-only GPT-2-style models [R33]; Sentence-BERT [R16]
   and SimCSE [R17] for semantic similarity as embedding-space scoring; CLIP [R15] for
   joint image–text scoring. Hugging Face ``transformers`` [R34] exposes hidden states and
   pooling conventions uniformly (observed in v5.16.1: ``BaseModelOutput.last_hidden_state``
   at ``vendor/transformers/src/transformers/modeling_outputs.py:24``; ``BertPooler`` first-
   token pooling at ``.../models/bert/modeling_bert.py:451``; CLIP ``get_text_features`` /
   ``get_image_features`` at ``.../models/clip/modeling_clip.py:674,710``).
7. **Adjacent "decision" literatures exist and mean different things.** [FACT] The
   **Decision Transformer** [R14] is reinforcement learning via return-conditioned sequence
   modeling over trajectories — unrelated in mechanism to a single-pass typed choice over a
   schema (see `10`). Reward models in RLHF [R22, R23] are scalar scorers trained on human
   preference data. Learning-to-rank (RankNet [R18], LambdaMART [R19], ListNet [R21],
   nDCG [R20]) scores items listwise/pairwise for ordering.

**What is known, in one sentence:** the individual ingredients (calibrated probabilities,
proper scores, reject options, encoders, schema-scoring) are well understood *separately*;
their integration into one typed decision layer is this project's engineering, not a known
result.

---

## 3. What is not known [Q3]

Stated as open items — none is answered by documentation; each is an experiment or a gap:

1. Whether JevX's adapter/head stack preserves the *task performance* of the underlying
   backbone relative to task-specific heads (H2/H3, `12`).
2. Which fusion variant (concat/gated/additive/bilinear/cross-attention) is best per backbone
   family — no theory predicts this; it is a hyperparameter question (H4).
3. Whether **one temperature per head** suffices across question types and schema sizes, or
   per-type/per-K calibrators are needed (H7).
4. Whether confidence functionals remain informative **under distribution shift** — the
   literature says raw confidence degrades [R10]; JevX has no mechanism that prevents this.
5. How far the CapabilityRegistry's support levels (0–4, `LIMITATIONS.md`) actually extend
   across the 510 ``model_type`` directories observed in v5.16.1 — unknown until probed.
6. Everything about "Jev" beyond `1` (no public knowledge).
7. Whether CPU-only latency/throughput is acceptable for interactive use — unmeasured
   (machine-specific; [EMPIRICAL] slot reserved in `reports/`).
8. The statistical behavior of jointly reporting many parallel questions from one state
   (error correlations, multiple-testing effects on thresholds) — unanalyzed here.

---

## 4. Required assumptions [Q4]

JevX's claims are conditional on the following; violating any voids the corresponding claim.

| # | Assumption | Kind | If violated |
|---|---|---|---|
| AS1 | The backbone exposes a hidden-state sequence (or pooled vector) with a usable mask — i.e., it is **HF Transformers-compatible with a meaningful differentiable representation** | [DESIGN] scope | CapabilityRegistry marks support 0; no conversion |
| AS2 | That representation is *task-relevant*: nearby/aligned states carry information the question heads can exploit | [HYPOTHESIS], probe-tested | Heads learn noise; no better than random baselines |
| AS3 | Calibration/evaluation data are **exchangeable** (i.i.d.-style) with deployment data | [STATISTICAL] | Temperature/isotonic fits and ECE estimates do not transfer |
| AS4 | Train/calibration/test splits are disjoint; no leakage | [DESIGN] protocol | All reported statistics invalid |
| AS5 | Evaluation-mode determinism (dropout off, fixed weights, deterministic kernels) | [DESIGN] (A7.1) | T7 equivalence breaks; outputs nondeterministic |
| AS6 | **Closed-world schemas:** for Choice questions, the correct answer is a member of $C$ | [DESIGN] contract | No in-schema projection can be correct; type safety (T2) still holds, correctness impossible by construction |
| AS7 | Option text (label/description) is informative for the encoder being used | [HYPOTHESIS] | Schema conditioning adds noise, not signal (H6) |
| AS8 | The declared type is adequate for the task (e.g., no hidden ranking structure in a flat Choice) | [DESIGN] contract | Miscalibrated decision semantics regardless of fit quality |
| AS9 | Evaluation loss matches the decision policy's loss (0-1 ↔ argmax; squared ↔ expectation) | [DESIGN] | Right probabilities, wrong decisions (or vice versa) |
| AS10 | Inputs are not adversarially crafted against the heads | [HYPOTHESIS] | Confidence/calibration guarantees in the statistical sense may fail per-instance |
| AS11 | binary32 precision suffices for the logit/softmax ranges used (cf. Remarks 1.2, 4.1) | [MATH boundary] | Overflow/underflow → clamps engaged, small biases |
| AS12 | Head training is feasible on the available CPU budget (small trainable parameter count) | [DESIGN] constraint | Mode A infeasible in practice → larger models untestable here |

---

## 5–9. The five "why" questions [Q5–Q9]

Each answer is labeled [DESIGN] (this is why *we* did it) with pointers to evidence status.

### 5. Why an adapter layer? [Q5] — [DESIGN]

A pretrained checkpoint is a *representation* artifact, not a decision system: it has no
notion of questions, schemas, types, thresholds, or calibration. JevX inserts an explicit
adapter stack (InputAdapter → RepresentationAdapter → QuestionEncoder/SchemaEncoder →
DecisionFusion → TypedDecisionHeads → CalibrationEngine) between the frozen-or-tuned backbone
and the user-facing decision for four stated reasons:

1. **Non-invasiveness.** The backbone's weights and code path stay untouched (Mode A), so the
   pretrained representation is preserved by construction and no per-model forks of 510 model
   families are required — adapters consume the common interfaces (``BaseModelOutput``,
   masks) that HF models already expose [FACT, `2.6`].
2. **Trainable-parameter economy.** Only the adapter is fit on CPU-class hardware (AS12): the
   head stack is $O(d^2)$ at most and typically $O(d)$ per fusion variant, versus
   $O(\text{backbone})$ — the precondition for Mode A feasibility on a 2015 dual-core CPU.
   *(That small capacity suffices is H1, `12` — not known.)*
3. **Invariant placement.** Type/schema validation, calibration, and confidence live at a
   single seam where the ValidationEngine can enforce D8.I and (11.3) on every call —
   invariants are cheap at one boundary, expensive scattered across model code.
4. **Comparability.** Swapping fusion variants, calibrators, or confidence functionals
   changes *only* the adapter, so ablations isolate the component under test.

### 6. Why schema conditioning matters? [Q6] — [DESIGN]

Conditioning the decision on the declared candidate set $C$ (via $E_C$ and $F$'s $H_C$
argument) rather than generating a free-form answer:

1. **Type safety becomes structural, not procedural.** The decision is an index into $C$
   (T2): a JevX Choice answer *cannot* be an undeclared label — unlike generated text, which
   requires external constrained-decoding machinery to even approximate this property.
2. **Comparable scores over candidates.** Logits are computed per option from
   (state, question, option) — a normalized comparison over a fixed alternative set, which is
   what classification, ranking (7.6–7.7), and expected-score (7.2) semantics require.
3. **Variable $K$ without variable parameters.** Option-wise shared scoring $\psi_\phi$
   (5.0) makes schema size a data property, not an architecture property — enabling the
   padding+mask batching of (11.2) and the T7 equivalence.
4. **Auditability.** Keys, not strings the model invented, are what gets logged and validated.
   *Empirical status: whether description text improves discrimination over label-only
   conditioning is H6.*

### 7. Why typed outputs? [Q7] — [DESIGN]

1. **Decision-theoretic correctness of the readout.** Each type pairs the probability object
   with its Bayes decision under the matching loss: Choice + 0-1 → argmax (T2);
   Score + squared error → expectation (T3, Remark 3.1); Noul + threshold loss →
   $\mathbf{1}[p \ge \tau]$. Untyped pipelines force one readout (usually argmax/verbalized
   text) onto tasks whose proper decision is different (e.g., expecting a 0–4 severity level,
   where the expected score uses information argmax discards).
2. **Bounded output spaces give provable closure.** Simplex/box membership and
   projection-invariance are theorems (T8) instead of hopes; downstream consumers can rely on
   ranges.
3. **Interfaces compose.** A Ranking type yields permutations by construction (T8-case 7); a
   MultiLabel type yields a $[0,1]^K$ box (T8-case 4) — consumers need no ad-hoc parsing or
   retry loops that generative formats entail.
   *Empirical status: format-error reduction vs generative output is H8.*

### 8. Why probability calibration? [Q8] — [DESIGN]

1. **Downstream consumers read probabilities numerically.** Abstention thresholds
   (10.1), risk–coverage reporting (10.2–10.4), and confidence-then-threshold policies are
   statements about probability *values*; uncalibrated softmax values make "$\kappa = 0.95$"
   meaningless (Definition 10 silently assumes calibrated inputs to be interpretable as
   frequencies).
2. **Training with a proper score is not calibration.** [MATH/STATISTICAL boundary] T5/T6
   concern the population optimum; finite data, model misspecification, and regularization
   leave residual miscalibration even at low test loss — the empirical finding of [R1] that
   motivates post-hoc methods in the first place.
3. **Cheap and accuracy-preserving where it matters.** Temperature scaling is one parameter,
   fit in seconds on CPU, and provably preserves Choice argmax decisions (`8` properties) —
   it changes reliability statements, not decisions. Non-order-preserving families
   (vector/isotonic) are available but their decision effects are surfaced, not hidden.
   *Honest scope: calibration addresses reliability of probabilities on matched data (a
   statistical property) — never correctness of individual answers (see non-claims in
   PROOFS.md).*

### 9. Why non-autoregressive decision evaluation? [Q9] — [DESIGN]

1. **Cost structure.** One forward pass per (state, question-batch) instead of a decode loop
   of $O(\text{output tokens})$ passes — decisive on CPU-only hardware (AS12), and it makes
   parallel questions (Def. 11) share the expensive $H_S$ exactly once (amortization factor
   $\approx J$).
2. **Determinism and equivalence testing.** No sampling, no exposure bias, no token-error
   compounding: outputs are a pure function of inputs (D5.1), which is what makes T7's
   loop-vs-batch equivalence statement possible at all.
3. **Typed output by construction, not by parsing.** Decision = index/expectation/threshold
   over declared objects (T2, T8); no grammar, no regex extraction, no format retries.
4. **Honest counterweight** [DESIGN/known limitation]: autoregressive models can *reason
   through* a question (multi-step inference, tool-like decomposition) before answering;
   JevX's single-pass scoring does not. For tasks where that reasoning is decisive, a
   generative pipeline may outperform this design — a scoped, empirical comparison JevX does
   not claim to win (and per `4` AS2 must be tested, not assumed).

---

## 10. Relation to prior art and adjacent fields [Q10]

Honest positioning: what is shared, what differs. No superiority comparisons — mechanism
comparisons only.

**10.1 Standard classifiers.** [FACT]
A softmax classifier maps a fixed label set (fixed $W \in \mathbb{R}^{K\times d}$) to simplex
points; T1/T2 are its elementary properties, and the calibration literature [R1, R8, R9]
measures exactly its outputs. JevX's $\mathsf{Choice}$ type *is* that pattern with three
changes: (i) the alternative set is **per-call data** (schema $C$, variable $K$) rather than
fixed weights — scored option-wise by shared $\psi_\phi$ (5.0); (ii) the schema is a
validated, hashed, first-class object the projection must return into (T2, D8.I); (iii)
calibration, confidence, and abstention are attached engines with defined protocols
(Defs. 8–10), not ad-hoc post-processing. Everything else (softmax, argmax, cross-entropy
training) is deliberately identical so that classical results transfer unchanged.

**10.2 Semantic encoders / bi-encoders.** [FACT]
Sentence-BERT [R16], SimCSE [R17], and CLIP-style scoring [R15] encode two objects
independently and compare by a fixed similarity (cosine/dot). JevX's $E_Q$, $E_C$ are
bi-encoder-like, but the score is **triadic** — a function of $(H_S, H_Q, H_{C,k})$ through a
learned fusion (Def. 5). Degenerate relations are exact: bilinear fusion with identity maps
reduces to dot-product scoring; dropping $H_S$ reduces to question–option bi-encoder
retrieval. Difference: the state conditions every score, so "same question, different state"
can lawfully produce different decisions.

**10.3 LLM generation (autoregressive).** [FACT + DESIGN]
Generation factorizes $p(y \mid \text{prompt})$ over token sequences: expressive free-text
outputs, iterative decode cost $O(\text{output length})$ passes, format errors unless
constrained decoding/grammars are imposed, and (unless greedy) sampling randomness. JevX:
single pass, output is a typed object over a declared schema by construction (T2/T8),
deterministic given inputs (D5.1, A7.1). Shared: same backbones (decoder-only models enter
via last-valid pooling — Def. 4d). Honest asymmetry [DESIGN/known]: generation can chain
multi-step reasoning before answering; JevX's single-pass scoring cannot — see Q9.4 and H9;
this document claims no advantage over reasoning-heavy generation, only a different
cost/guarantee profile.

**10.4 Decision transformers.** [FACT — name collision warning]
The Decision Transformer [R14] is an offline-RL architecture: it casts RL as sequence
modeling, conditioning token-level *action* generation on target returns, trained on
trajectories of an MDP. JevX shares only the slogan "express decisions with a transformer":
there is no return conditioning, no trajectory dataset, no control loop, no Bellman
objective; the output is a single typed decision over an exogenous schema, not a policy.
**Do not conflate the two literatures.** (JevX's phrase "typed decision" is defined entirely
in Definitions 2, 3, 7 of this project.)

**10.5 Ranking models (learning-to-rank).** [FACT]
RankNet [R18], LambdaMART [R19], ListNet [R21], and nDCG objectives [R20] score items to
produce orderings. JevX's $\mathsf{Ranking}$ type is a score-and-sort readout (7.6) with an
optional Plackett–Luce distribution (7.7) — mathematically inside the ListNet/PL family.
Differences: candidates are schema options scored jointly with state+question; JevX v1 does
not claim an LTR-specific ranking loss or nDCG optimization (evaluation metrics like nDCG
remain available for reporting [R20]).

**10.6 Reward models.** [FACT]
RLHF reward models [R22, R23] are scalar scorers trained on human preference pairs, used to
rank generations. JevX's $\mathsf{Score}$ type is structurally adjacent but different: a
softmax over ordered levels with the decision = expectation (7.2), hence **bounded by
construction** (T3) and interpretable as an expected category, versus unbounded scalar
regression; $\mathsf{Noul}$ is adjacent to a binary preference head. JevX claims no RLHF
loop, no preference-data recipe, and no human-feedback training here — the heads are fit on
task labels (Mode A/B, `ARCHITECTURE.md`).

**10.7 Calibration methods.** [FACT]
Temperature/vector/matrix scaling, isotonic regression, and binning originate in the cited
literature [R1, R5, R6, R7]; evaluation pitfalls (binning-dependence of ECE) are documented
in [R26–R29]. JevX adopts these methods *verbatim* inside a CalibrationEngine with a defined
fit/calibrate/evaluate protocol (Def. 8) and adds two engineering invariants the standalone
papers do not need: schema/type preservation across calibration (D8.I) and explicit reporting
when a non-order-preserving calibrator changes argmax decisions (`8`). Conformal prediction
[R24, R25] is noted as the principled route to finite-sample risk statements under
exchangeability — not implemented in v1, listed as future work, not claimed.

**10.8 Selective prediction.** [FACT]
Chow's reject tradeoff [R12] and deep selective classification [R11] define exactly the
risk–coverage and coverage-at-risk quantities of Definition 10. JevX contributes integration,
not new theory: confidence functionals, thresholding, abstention type, and reporting protocol
(w loss, sample size, κ-selection method stated) are wired into the same typed pipeline, and
the theorems invoked are T1–T4/T7–T8 only — no theorem about selective risk is claimed here.
Whether the resulting risk–coverage curves are *good* is H10, an empirical question.

---

## 11. JEVX DESIGN CHOICES (consolidated)

*The audited list of decisions this project made. Each is revisable; none is evidence.*

| # | Design choice | Primary rationale / doc |
|---|---|---|
| DC-1 | Backbone = any HF Transformers-compatible model with a differentiable representation; `transformers` v5.16.1 as the audited reference tree | `2.6`, AS1 |
| DC-2 | Adapter stack between backbone and decision; **Mode A (frozen backbone)** default, Mode B (fine-tune) optional | Q5, `ARCHITECTURE.md` |
| DC-3 | **Representation reuse**, not weight transplant, for all new components (fusion, heads, calibrators, confidence) — true 1:1 conversion is impossible where no pretrained weights exist | `ARCHITECTURE.md` |
| DC-4 | Questions and schemas are explicit typed inputs: $Q=(\text{text,type,constraints,metadata})$; options carry key/label/description/metadata | Defs. 2–3 |
| DC-5 | Option-wise shared scoring for variable $K$; padding+mask batching; **never normalize across questions** | Def. 5.0, (11.2)–(11.3) |
| DC-6 | One shared $H_S$ per state; per-question graph separation (functional, not statistical) enforced architecturally | Def. 11, T7 |
| DC-7 | Interchangeable fusion variants — the five Def. 5 forms (concat/gated/additive/bilinear/cross-attention) are registered as `FUSION_REGISTRY` kinds in `jevx/representations/fusion.py` (cross-attention: state sequence as K/V, question embedding as query, masked stable softmax; `dot`/`mlp` registered as extras); default selected by validation, not by doctrine | Def. 5, H4 |
| DC-8 | Probability semantics chosen per type (softmax/sigmoid/cumulative/PL) — no free-text outputs anywhere | Def. 6 |
| DC-9 | Deterministic readouts: min-index argmax, stable sort by schema index | Defs. 7, T2 |
| DC-10 | Post-hoc calibration stage with schema/type-preservation invariant; temperature default; decision-change reporting for non-order-preserving fits | Def. 8, D8.I |
| DC-11 | Confidence = explicit functionals (max/margin/normalized-entropy/energy), applied post-calibration by default; never equated with correctness | Def. 9, non-claims |
| DC-12 | Optional abstention overlay with threshold κ; risk–coverage reporting protocol (loss, N, κ-selection always stated) | Def. 10 |
| DC-13 | Runtime ValidationEngine asserts: schema-hash equality across calibration, type-tag equality, per-question normalization, mask zeroing, FP tolerances | D8.I, (11.3) |
| DC-14 | CapabilityRegistry with support levels 0–4; conversion attempted only after a hidden-state/mask probe passes | `LIMITATIONS.md` |
| DC-15 | CPU-first feasibility: small trainable parameter counts in Mode A; no GPU-dependent claims; all latency numbers machine-tagged | AS12, environment facts |
| DC-16 | Documentation discipline: [FACT]/[DESIGN]/[HYPOTHESIS]/[EMPIRICAL] labels; banned marketing vocabulary; explicit non-claims list | PROOFS.md |

---

## 12. JEVX HYPOTHESES (consolidated)

*Falsifiable expectations. None is a result; each names the measurement that would refute it.*

| # | Hypothesis | Falsifier / measurement |
|---|---|---|
| H1 | Adapter-only training (Mode A) reaches useful decision quality on in-domain tasks without touching backbone weights | Mode A at chance/majority baselines while Mode B ≫ chance → adapter capacity insufficient |
| H2 | On standard datasets, JevX typed heads match task-specific reference heads within a pre-declared tolerance ε | Statistically clear gap to reference heads under identical data/splits |
| H3 | Mode B (fine-tuning) improves in-domain quality over Mode A but can degrade the general representation (forgetting) | No in-domain gain, or backbone probe performance drops with no compensating decision gain |
| H4 | The best fusion variant depends on backbone family (no single variant dominates) | One variant wins across all families within ε → collapse to that default |
| H5 | Last-valid pooling suffices for decoder-only; CLS/mean suffice for bidirectional encoders | Probing shows decision-relevant information lost by the registry's default pooling |
| H6 | Including option *descriptions* improves discriminability over label-only conditioning | Label-only ablation ties or beats full conditioning across tasks |
| H7 | Scalar temperature captures most calibration improvement for fixed-$K$ heads; some head classes need vector/isotonic | Post-temperature residual ECE remains large (vs pre-fit baseline) on held-out sets |
| H8 | Typed projection reduces output-format failures relative to generative answering of comparable models | Format/parse-failure rate of typed output ≥ generative baseline's retry-adjusted rate |
| H9 | For shallow decision tasks, non-autoregressive accuracy ≈ generation-with-parsing, at lower latency | Tasks where reasoning-augmented generation beats typed scoring decisively at matched compute |
| H10 | Confidence-thresholded abstention yields risk–coverage curves better (lower AURC) than random-selection baselines | AURC indistinguishable from or worse than random selection on held-out data |
| H11 | Batched and looped evaluation agree within a declared fp32 tolerance across supported models (T7-ii) | Measured ε exceeds the declared tolerance on any supported configuration |
| H12 | Most probed `model_type` entries reach CapabilityRegistry support ≥ 2 (load + forward pass + pooling) | Probe pass rate below a pre-declared floor → universality claim (scoped as it is) must shrink further |

An [EMPIRICAL] label is earned only by attaching numbers, machine, versions, and splits to
one of H1–H12. Until then these remain hypotheses.

---

## References

*[R#] entries are citations the author can support; no source is listed that is not known
to exist in the public literature.*

* **[R1]** Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. (2017). *On Calibration of Modern
  Neural Networks.* ICML 2017. (post-hoc temperature scaling; vector/matrix scaling and
  isotonic baselines)
* **[R2]** Brier, G. W. (1950). *Verification of Forecasts Expressed in Terms of
  Probability.* Monthly Weather Review 78(1):1–3.
* **[R3]** Murphy, A. H. (1973). *A New Vector Partition of the Probability Score.*
  Journal of Applied Meteorology 12(4):595–600.
* **[R4]** Gneiting, T., Raftery, A. E. (2007). *Strictly Proper Scoring Rules, Prediction,
  and Estimation.* Journal of the American Statistical Association 102(477):359–378.
* **[R5]** Zadrozny, B., Elkan, C. (2002). *Transforming Classifier Scores into Accurate
  Multiclass Probability Estimates.* KDD 2002.
* **[R6]** Niculescu-Mizil, A., Caruana, R. (2005). *Predicting Good Probabilities with
  Supervised Learning.* ICML 2005.
* **[R7]** Platt, J. (1999). *Probabilistic Outputs for Support Vector Machines and
  Comparisons to Regularized Likelihood Methods.* in Advances in Large Margin Classifiers.
* **[R8]** Desai, M. S., Durrett, G. (2020). *Calibration of Pre-trained Transformers.*
  EMNLP 2020.
* **[R9]** Minderer, M., Djolonga, J., Romijnders, R., et al. (2021). *Revisiting the
  Calibration of Modern Neural Networks.* NeurIPS 2021.
* **[R10]** Ovadia, Y., Fertig, E., Ren, J., et al. (2019). *Can You Trust Your Model's
  Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift.* NeurIPS 2019.
* **[R11]** Geifman, Y., El-Yaniv, R. (2017). *Selective Classification for Deep Neural
  Networks.* NeurIPS 2017.
* **[R12]** Chow, C. K. (1970). *On Optimum Recognition Error and Reject Tradeoff.* IEEE
  Transactions on Information Theory 16(1):41–46.
* **[R13]** Liu, W., Wang, X., Owens, J. D., Li, Y. (2020). *Energy-based Out-of-distribution
  Detection.* NeurIPS 2020.
* **[R14]** Chen, L., Lu, K., Rajeswar, A., et al. (2021). *Decision Transformer:
  Reinforcement Learning via Sequence Modeling.* NeurIPS 2021.
* **[R15]** Radford, A., Kim, J. W., Hallacy, C., et al. (2021). *Learning Transferable
  Visual Models From Natural Language Supervision (CLIP).* ICML 2021.
* **[R16]** Reimers, N., Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using
  Siamese BERT-Networks.* EMNLP-IJCNLP 2019.
* **[R17]** Gao, T., Yao, X., Chen, D. (2021). *SimCSE: Unsupervised Contrastive Learning
  for Sentence Embeddings.* EMNLP 2021.
* **[R18]** Burges, C. J. C., Shaked, T., Renshaw, E., et al. (2005). *Learning to Rank
  using Gradient Descent (RankNet).* ICML 2005.
* **[R19]** Burges, C. J. C. (2010). *From RankNet to LambdaRank to LambdaMART to LTR — an
  Overview.* Microsoft Research Technical Report.
* **[R20]** Järvelin, K., Kekäläinen, J. (2002). *Cumulated Gain-Based Evaluation of IR
  Techniques.* ACM TOIS 20(4):422–446.
* **[R21]** Cao, Z., Qin, T., Liu, T.-Y., et al. (2007). *Learning to Rank: From Linear
  Approach to Nonlinear Analysis (ListNet).* ECML 2007.
* **[R22]** Stiennon, N., Ouyang, L., Wu, J., et al. (2020). *Learning to Summarize with
  Human Feedback.* NeurIPS 2020.
* **[R23]** Ouyang, L., Wu, J., Jiang, X., et al. (2022). *Training Language Models to
  Follow Instructions with Human Feedback (InstructGPT).* NeurIPS 2022.
* **[R24]** Vovk, V., Gammerman, A., Shafer, G. (2005). *Algorithmic Learning in a Random
  World.* Springer.
* **[R25]** Angelopoulos, A. N., Bates, S. (2023). *Conformal Prediction: A Gentle
  Introduction.* Foundations and Trends in Machine Learning 16(4):494–591.
* **[R26]** Naeini, M. P., Cooper, G., Hauskrecht, M. (2015). *Obtaining Well Calibrated
  Probabilities Using Bayesian Binning.* AAAI 2015.
* **[R27]** Nixon, M. E., Dusenberry, M. W., Zhang, L., et al. (2019). *Measuring
  Calibration in Deep Learning.* CVPR Workshops 2019.
* **[R28]** Vaicenavicius, J., Widmann, D., Andersson, C., et al. (2019). *Evaluating Model
  Calibration in Classification.* AISTATS 2019.
* **[R29]** Kull, M., Silva Filho, T. C., Flach, P. (2019). *Beyond Temperature Scaling:
  Obtaining Well-Calibrated Multiclass Probabilities with Dirichlet Calibration.* NeurIPS
  2019.
* **[R30]** Hendrycks, D., Gimpel, K. (2017). *A Baseline for Detecting Misclassified and
  Out-of-Distribution Examples in Neural Networks.* ICLR 2017.
* **[R31]** Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). *Attention Is All You Need.*
  NeurIPS 2017.
* **[R32]** Devlin, J., Chang, M.-W., Lee, K., Toutanova, K. (2019). *BERT: Pre-training of
  Deep Bidirectional Transformers for Language Understanding.* NAACL-HLT 2019.
* **[R33]** Radford, A., Wu, J., Child, R., et al. (2019). *Language Models are
  Unsupervised Multitask Learners (GPT-2).* OpenAI technical report.
* **[R34]** Wolf, T., Debut, L., Sanh, V., et al. (2020). *Transformers: State-of-the-Art
  Natural Language Processing.* EMNLP 2020 System Demonstrations (Hugging Face);
  version-specific facts here are quoted from the vendored source tree at tag v5.16.1 with
  file:line pointers (`2.6`).
