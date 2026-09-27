# JevX Proofs — Theorems T1–T8

**Scope.** Rigorous proofs of the mathematical claims used throughout JevX documentation.
Each theorem states (i) the quantified statement, (ii) its assumptions with unique labels,
(iiii) the proof, and (iv) numerical-model remarks. The closing sections separate the five
kinds of guarantees JevX deals in, and list claims that are explicitly *not* made.

**Global conventions.**

* Logarithms: $\ln$ (base $e$). Conventions $0\ln 0 = 0$, $\ln 0 = -\infty$,
  $x/0 = +\infty$ for $x > 0$.
* Arithmetic model: **RM** = exact real arithmetic; **FP** = IEEE-754 binary32/binary64 with
  unit roundoff $u$ ($2^{-24}$ for binary32, $2^{-53}$ for binary64). Theorems are RM results;
  FP deviations are isolated in remarks and never silently promoted to theorems.
* $[K] = \{1,\dots,K\}$; $e_i$ denotes the $i$-th standard basis vector (one-hot);
  $\Delta^{K-1} = \{p \ge 0 : \sum_i p_i = 1\}$.
* $\operatorname{softmax}(z)_i = e^{z_i}/\sum_j e^{z_j}$;
  $\sigma(z) = 1/(1+e^{-z})$; $\operatorname{argmax}^{\min}$ = minimum-index argmax.
* Labels: [MATH] deductive under the listed assumptions; [EMPIRICAL] must be backed by
  measurements; [STATISTICAL] needs data assumptions; [DESIGN] a policy choice.

---

## T1 — Softmax validity

**Theorem T1.**
$\forall K \in \mathbb{N}$, $\forall z \in \mathbb{R}^K$ with all components finite:
$p = \operatorname{softmax}(z)$ satisfies

$$
\text{(i)}\ \forall i \in [K]:\ p_i > 0,
\qquad\qquad
\text{(ii)}\ \sum_{i=1}^{K} p_i = 1 .
$$

**Assumptions.**

* **A1.1** $K \ge 1$ is finite.
* **A1.2** Each $z_i \in \mathbb{R}$ is finite (no $\pm\infty$, no NaN).
* **A1.3** The model of computation is exact real arithmetic (RM); $\exp$ is the real
  exponential with its standard properties.

**Proof.**

*Lemma 1 (positivity of the exponential).* $\forall x \in \mathbb{R}: e^{x} > 0$.
Indeed $e^{x} e^{-x} = e^{0} = 1$, so $e^{x}$ has a multiplicative inverse and hence
$e^{x} \ne 0$; moreover $e^{x} = \left(e^{x/2}\right)^{2} \ge 0$. A nonzero real that is a
square is strictly positive, so $e^{x} > 0$. $\square_{\text{Lemma 1}}$

1. *(Positivity.)* Let $D = \sum_{j=1}^{K} e^{z_j}$. By Lemma 1 every summand is $>0$, so
   $D \ge e^{z_1} > 0$ (a finite sum of positives is at least any one of them, and is positive).
   For each $i$, $p_i = e^{z_i}/D$ is a quotient of a positive numerator and a positive
   denominator, hence $p_i > 0$. This proves (i).
2. *(Normalization.)* By distributivity of multiplication over finite sums and $D \ne 0$:
   $\sum_{i=1}^{K} p_i = \frac{\sum_{i=1}^{K} e^{z_i}}{D} = \frac{D}{D} = 1$. This proves (ii).
   $\blacksquare$

**Remark 1.1 — max-subtraction is an exact invariance (RM).** [MATH]
$\forall c \in \mathbb{R}: \operatorname{softmax}(z + c\mathbf{1}) = \operatorname{softmax}(z)$.
Proof: $e^{z_i + c} = e^{c} e^{z_i}$ for every $i$, so numerator and denominator are both
multiplied by the same scalar $e^{c} > 0$ (Lemma 1), which cancels exactly in the quotient.
Choosing $c = -\max_j z_j$ yields the stable form of Definition 6:
$\operatorname{softmax}(z)_i = e^{z_i - m}/\sum_j e^{z_j - m}$ with all exponents $\le 0$.
The word *exact* refers to RM: no approximation is involved in the identity itself.

**Remark 1.2 — what survives in IEEE-754 (FP honesty).** [MATH boundary]
In FP the identity of Remark 1.1 is not bit-exact: the subtraction $z_i - m$ is exact only
under conditions of the Sterbenz lemma (e.g. $z_i \in [m/2, 2m]$ for $m > 0$), the $\exp$ and
the division are each rounded, and $\sum_j e^{z_j-m}$ accumulates $O(K)$ roundings. Standard
forward-error analysis bounds the computed components classwise:
$|\hat p_i - p_i| \le \gamma_{cK}\, p_i$ for a small constant $c$ and
$\gamma_{cK} = \frac{cKu}{1-cKu}$ (valid while $cKu < 1$). Consequences actually used by JevX:

* the *raw* form overflows for $z_i \gtrsim 88$ (binary32) / $709$ (binary64) — computed as
  $e^{z_i} = \texttt{inf}$, giving $p_i = \text{inf}/\text{inf} = \text{NaN}$;
* the *stable* form removes overflow, but tiny terms can still underflow: $e^{z_i - m}$ rounds
  to exact $0$ when $z_i - m \lesssim -104$ (binary32), so the computed $\hat p_i$ can be
  exactly $0$ although the RM value is $> 0$;
* $\sum_i \hat p_i = 1$ holds up to rounding unless the implementation renormalizes by a final
  division (JevX's ValidationEngine checks $\left|\sum_i \hat p_i - 1\right| \le \text{tol}$ at runtime when `JevXConfig.runtime_validation` is on — `jevx/validation.py`, called from `JevXForDecision.forward`; the same bound is also asserted by the test suites).

Theorem T1 itself is a statement about RM; the numbers above are implementation facts
[EMPIRICAL constants from IEEE-754 range limits].

---

## T2 — Choice type safety (schema-constrained projection cannot escape the declared type)

**Theorem T2.**
Let $C = (O_1,\dots,O_K)$ be a schema, $z \in \mathbb{R}^{\tilde K}$ the (possibly batch-padded)
logit vector with option mask selecting exactly the valid coordinates $[K] \subseteq [\tilde K]$,
$p$ the masked-softmax output of Definition 11 (11.2) (valid coordinates as in T1, masked
coordinates exactly $0$), and $i^* = \operatorname{argmax}^{\min}(p)$ over all $\tilde K$
coordinates. Then

$$
\text{(a)}\ i^* \in [K], \qquad
\text{(b)}\ O_{i^*} \in C, \qquad
\text{(c)}\ \Pi_{\mathsf{Choice}}(p, C) = (p,\, O_{i^*}) \in \mathcal{D}_{\mathsf{Choice}}(C),
$$

and no masked coordinate is ever selected. In words: the projected Choice decision is an
element of the declared schema; the image of $\Pi_{\mathsf{Choice}}$ is contained in the
declared type space.

**Assumptions.**

* **A2.1** Schema axioms D3.1–D3.2 hold: $K \ge 1$; option list has length exactly $K$;
  keys are pairwise distinct (injectivity of $k \mapsto O_k$ on identities).
* **A2.2** Every *valid* coordinate has a finite logit; the number of valid coordinates is
  $\ge 1$; masked coordinates are excluded from normalization and assigned probability exactly
  $0$ (equivalently: their pre-softmax logit is $-\infty$ per (11.2)).
* **A2.3** The output assembly reads options *only* by index from $C$
  (schema-anchored construction: the decision slot contains the object $O_{i^*}$, not a newly
  constructed candidate).
* **A2.4** Tie-breaking is $\operatorname{argmax}^{\min}$ (minimum index), so the decision is
  single-valued.
* **A2.5** No NaN is present (rest of A1.2), so comparisons in $\max$ are total.

**Proof.**

1. *(Support of $p$.)* Restricted to the valid coordinates, T1(i) applies to the softmax of
   the valid subvector: $p_i > 0$ for all $i \in [K]$; by (11.2) masked coordinates satisfy
   $p_k = 0$. Also $\sum_{i \in [K]} p_i = 1$ (T1(ii)), hence
   $M := \max_{k \in [\tilde K]} p_k \ge \frac{1}{\sum_{i\in[K]} 1} = \frac{1}{K} > 0$
   (the maximum over a nonempty finite set is at least the mean). Therefore every coordinate
   attaining the maximum has $p_k = M > 0$, which excludes all masked coordinates
   ($0 \not> 0$). The argmax set $A = \{k : p_k = M\}$ is nonempty (finite, nonempty index
   set, real-valued) and satisfies $A \subseteq [K]$. By A2.4, $i^* = \min A \in A \subseteq [K]$.
   This proves (a) and the "never selects masked coordinates" clause.
2. *(Schema membership.)* $C$ is a list of length $K$, so the map $\iota : [K] \to C$,
   $\iota(k) = O_k$, is total (every index in $[K]$ names exactly one stored option). By (a),
   $O_{i^*} = \iota(i^*)$ is defined and is an element of $C$. Distinct keys (A2.1) guarantee
   that distinct chosen indices correspond to distinct option identities — the decision key is
   unambiguous. This proves (b).
3. *(Type-space containment.)* By definition
   $\mathcal{D}_{\mathsf{Choice}}(C) = \{(p, O_{i^*}) : p \in \Delta^{K-1},\ i^* = \operatorname{argmax}^{\min}(p)\}$.
   From step 1 and T1(ii), $p \in \operatorname{int}\Delta^{K-1} \subseteq \Delta^{K-1}$;
   from steps 1–2 the second component is in $C$ and is the min-index argmax of $p$; A2.3
   ensures nothing else can appear in the decision slot. Hence the image of
   $\Pi_{\mathsf{Choice}}$ is a subset of $\mathcal{D}_{\mathsf{Choice}}(C)$: the projection is
   *total into* the declared type and cannot emit an object outside it. This proves (c).
   $\blacksquare$

**Remark 2.1 — what "cannot escape" does and does not mean.** [MATH / boundary]
T2 is a statement about the *codomain of a function*, not about semantic correctness: the
chosen $O_{i^*}$ can be the *wrong option for the task*; T2 only guarantees it is *an* option
that was declared. Type safety constrains the output space; it says nothing about which
element of that space is correct (see the guarantee taxonomy below).

**Remark 2.2 — ties, NaN, empty schemas.** [MATH boundary]
* Ties: $|A| \ge 1$ may exceed 1; $\min A$ is still in $[K]$ — no extra assumption needed for
  membership, only for uniqueness of *semantics* (A2.4 supplies determinism).
* NaN: if some valid $z_i$ were NaN, $\max$ comparisons with NaN are false in IEEE-754 and the
  argmax result becomes platform-dependent; A2.5 excludes this (input validation is the
  implementation counterpart).
* $K = 0$: excluded by A2.1/D3.1 — with no valid coordinate, $M$ over the empty set is
  undefined and no decision exists.

---

## T3 — Score boundedness (expectation lies inside the level range)

**Theorem T3.**
Let $K \ge 1$, let the level set be $I = \{\ell_1 < \ell_2 < \dots < \ell_K\} \subset \mathbb{R}$
(identified with $\{0,1,\dots,K-1\}$ under the default identification of D3.3), and let
$p \in \Delta^{K-1}$ be any distribution over $I$, i.e. $p_i \ge 0$, $\sum_{i=1}^K p_i = 1$.
Define the expected score

$$
s \;=\; \sum_{i=1}^{K} \ell_i\, p_i .
$$

Then

$$
\text{(a)}\ \ \min_{i} \ell_i \;\le\; s \;\le\; \max_i \ell_i
\qquad\Big(\text{default: } 0 \le s \le K-1\Big),
$$

with equality on the left **iff** $\operatorname{supp}(p) \subseteq \{\min_i \ell_i\}$ and
equality on the right **iff** $\operatorname{supp}(p) \subseteq \{\max_i \ell_i\}$.

More generally (the form actually used): if a random variable $X$ has support in a bounded
interval $[a, b]$ (i.e. $X \in [a,b]$ almost surely), then $a \le \mathbb{E}[X] \le b$.

**Assumptions.**

* **A3.1** $p$ is a probability vector: $p_i \ge 0$ for all $i$ and $\sum_i p_i = 1$
  (nonnegativity and normalization are both used).
* **A3.2** $I$ is a finite set of reals; $K \ge 1$ (the sum is finite).
* **A3.3** Exact real arithmetic (RM).

**Proof.**

Write $a = \min_i \ell_i$ and $b = \max_i \ell_i$. Both exist: a nonempty finite subset of
$\mathbb{R}$ attains its min/max.

1. *(Lower bound.)* For every $i$, $\ell_i - a \ge 0$ by minimality of $a$; and $p_i \ge 0$
   by A3.1. Hence every product $(\ell_i - a) p_i \ge 0$, so the finite sum is $\ge 0$:

$$
s - a \;=\; \sum_{i=1}^{K} (\ell_i - a)\, p_i \;\ge\; 0
\quad\Longrightarrow\quad s \ge a .
$$

   (Equivalently: $s = \sum_i \ell_i p_i \ge \sum_i a\, p_i = a \sum_i p_i = a$, using both
   $\ell_i \ge a$ and normalization.)
2. *(Upper bound.)* Symmetrically, $b - \ell_i \ge 0$ gives
   $b - s = \sum_i (b - \ell_i) p_i \ge 0$, so $s \le b$.
3. *(Equality cases.)* $s - a = 0$ iff the finite sum of nonnegative terms
   $(\ell_i - a)p_i$ vanishes iff every term vanishes (a finite sum of nonnegative reals is
   $0$ iff each summand is $0$) iff for each $i$: $p_i = 0$ or $\ell_i = a$, i.e. iff
   $\operatorname{supp}(p) = \{i : p_i > 0\} \subseteq \{i : \ell_i = a\}$. The case
   $s = b$ is symmetric. (Distinctness of the $\ell_i$ makes these index sets singletons.)
4. *(General form.)* Identical argument with $a, b$ replaced by the essential bounds of the
   support of $X$: $\mathbb{E}[X] - a = \mathbb{E}[X - a] \ge 0$ since $X - a \ge 0$ a.s.
   $\blacksquare$

**Remark 3.1 — decision-rule semantics.** [MATH / DESIGN boundary]
$s = \sum_i \ell_i p_i$ is the Bayes action for squared-error loss
$\mathbb{E}[(X - d)^2]$ minimized at $d = \mathbb{E}[X]$; the modal level
$\ell_{\operatorname{argmax} p}$ is the Bayes action for 0-1 loss on the level. Both lie in
$[a, b]$ by T3. Which one JevX returns is a declared policy (default: expectation,
Definition 7), not a mathematical necessity.

**Remark 3.2 — floating point.** [MATH boundary]
The computed sum $\hat s = \sum_i \hat\ell_i \hat p_i$ in FP satisfies
$|\hat s - s| \le \gamma_K \sum_i |\ell_i| p_i + \text{terms from } \hat p$ rounding, so
$\hat s$ can leave $[a,b]$ by $O(\gamma_K \max_i |\ell_i|)$ even when $s \in [a,b]$ exactly.
If membership of the *returned scalar* in the box is required bit-exactly, the implementation
clamps $\hat s \leftarrow \min(\max(\hat s, a), b)$ — that clamp is an **implementation
invariant**; the exact membership claim (a) is an RM theorem.

---

## T4 — Binary validity ($0 < \sigma(z) < 1$ for finite $z$)

**Theorem T4.**
$\forall z \in \mathbb{R}$ (finite): $\ 0 \;<\; \sigma(z) \;<\; 1$, where
$\sigma(z) = \dfrac{1}{1+e^{-z}}$. Moreover $\sigma$ is strictly increasing, and
$\inf_{z\in\mathbb{R}} \sigma(z) = 0$, $\sup_{z\in\mathbb{R}} \sigma(z) = 1$ are approached only
in the limits $z \to -\infty$ / $z \to +\infty$ — never attained at any finite $z$.

**Assumptions.**

* **A4.1** $z$ is a finite real number.
* **A4.2** Exact real arithmetic (RM); Lemma 1 of T1 ($e^{x} > 0$) is reused.
* **A4.3** The sigmoid is defined by the closed form (6.3), not by a truncated series.

**Proof.**

1. *(Strict lower bound.)* By A4.2, $e^{-z} > 0$, so $1 + e^{-z} > 1 > 0$. The reciprocal of
   a number greater than $1$ lies in $(0,1)$: explicitly,
   $\sigma(z) = \frac{1}{1+e^{-z}} > 0$ (quotient of positives) and
   $\sigma(z) < 1 \iff 1 < 1 + e^{-z} \iff 0 < e^{-z}$, which holds by Lemma 1.
   Equivalently from the second form: $\sigma(z) = \frac{e^{z}}{1+e^{z}}$ with
   $e^{z} > 0 \Rightarrow$ numerator $> 0$ and denominator $= 1 + e^{z} > e^{z} = $ numerator
   $\Rightarrow \sigma(z) < 1$.
2. *(Strict monotonicity.)* $\sigma'(z) = \sigma(z)(1 - \sigma(z))$; by step 1 both factors are
   strictly positive for finite $z$, so $\sigma'(z) > 0$: strictly increasing.
3. *(Non-attainment of the limits.)* By step 1 the bounds hold at every finite $z$, so the
   inf/sup are limits only. Concretely: as $z \to +\infty$, $e^{-z} \to 0^+$ and
   $\sigma(z) \to 1^-$; as $z \to -\infty$, $e^{-z} \to +\infty$ and $\sigma(z) \to 0^+$.
   Monotonicity (step 2) plus strict inequality at every finite point give that neither value
   is attained. $\blacksquare$

**Remark 4.1 — what FP computes at extreme logits.** [MATH boundary / implementation]
With the stable branch (6.4), for $z \ge 0$: $e^{-z} \in (0,1]$ never overflows, and for
$z < 0$: $e^{z} \in (0,1)$ never overflows. However **underflow** still occurs:
$e^{z}$ rounds to exact $0$ for $z \lesssim -104$ in binary32 (smallest subnormal
$\approx 1.48\times10^{-45}$, $\ln$ thereof $\approx -103.97$) and $z \lesssim -745$ in
binary64. Then the *computed* value is exactly $\hat\sigma(z) = 0$ (for $z \lesssim -104$) or
exactly $\hat\sigma(z) = 1$ (by the symmetric argument for $z \gtrsim 104$), violating the
strict inequalities of T4 at the level of *bits*. The naive branch
$1/(1+e^{-z})$ for $z = -1000$ is worse: $e^{1000} = \texttt{inf}$ and $1/(1+\texttt{inf}) = 0$
also yields $0$, while some intermediate cases produce $\text{inf}/\text{inf} = \text{NaN}$.
JevX therefore (i) uses branch (6.4), and (ii) if strict $(0,1)$ membership of *stored*
outputs is required, clamps to $[\varepsilon, 1-\varepsilon]$ with a declared
$\varepsilon$ — an **implementation invariant** whose bias is then bounded by
$\varepsilon$. The theorem is about RM.

---

## T5 — The log score is a strictly proper scoring rule

**Theorem T5.**
Let $\Omega = [K]$ be finite, $p, q \in \Delta^{K-1}$. Define, with the conventions of the
header,

$$
\mathrm{CE}(p,q) \;=\; \mathbb{E}_{p}[-\ln q] \;=\; -\sum_{i=1}^{K} p_i \ln q_i
\quad(\in [0, +\infty]),\qquad
H(p) \;=\; -\sum_{i=1}^{K} p_i \ln p_i \in [0, \ln K],
$$

$$
\mathrm{KL}(p \| q) \;=\; \sum_{i=1}^{K} p_i \ln \frac{p_i}{q_i} \;\in\; [0, +\infty] .
$$

Then:

$$
\text{(a)}\ \ \mathrm{CE}(p,q) = H(p) + \mathrm{KL}(p\|q)\ \ \text{for all } p,q \in \Delta^{K-1};
$$

$$
\text{(b)}\ \ \mathrm{KL}(p\|q) \ge 0,\ \text{ with equality iff } p = q;
$$

$$
\text{(c)}\ \ \forall p:\ \ \arg\min_{q \in \Delta^{K-1}} \mathrm{CE}(p,q) = \{p\},
\qquad \min_q \mathrm{CE}(p,q) = H(p).
$$

I.e., expected negative log-likelihood decomposes into irreducible entropy plus nonnegative
divergence, and it is minimized **if and only if** $q = p$: the log score is a strictly
proper scoring rule.

**Assumptions.**

* **A5.1** $\Omega$ finite; $p, q$ are probability vectors (nonnegative, sum to 1).
* **A5.2** Natural logarithm; conventions $0\cdot\ln 0 = 0$, and for $p_i > 0, q_i = 0$ the
  $i$-th terms of CE and KL are defined as $+\infty$ (consistent with
  $\lim_{\varepsilon\downarrow 0} -\ln\varepsilon = +\infty$).
* **A5.3** $q$ ranges over the **closed** simplex $\Delta^{K-1}$ in (c) — boundary points
  allowed (this matters for one-hot empirical targets, e.g. $p = e_y$).
* **A5.4** Exact real arithmetic.

**Proof.**

**(a) Additivity.** Work indexwise with the conventions of A5.2.

* If $p_i > 0$ and $q_i > 0$: then $\ln q_i = \ln p_i - \ln(p_i/q_i)$, so multiplying by
  $-p_i$: $-p_i\ln q_i = -p_i \ln p_i + p_i \ln(p_i/q_i)$.
* If $p_i = 0$: every term ($-p_i\ln q_i$, $-p_i\ln p_i$, $p_i\ln(p_i/q_i)$) is $0$ by
  convention; also the limit of $x\ln(x/q_i)$ as $x \downarrow 0$ is $0$, so the convention
  matches the continuous extension.
* If $p_i > 0$ and $q_i = 0$: LHS term $= +\infty$; KL term $= p_i \ln(+\infty) = +\infty$;
  $H$ term finite; $+\infty = \text{finite} + +\infty$ holds.

Summing the valid indexwise identities over the finite set $[K]$: if no term is infinite we
may split sums freely; if at least one term is $+\infty$ then CE $= +\infty$ and
KL $= +\infty$ (KL is a sum of nonnegative extended-real terms with one $+\infty$), so both
sides equal $+\infty$. Hence (a) holds in all cases.

**(b) Nonnegativity with sharp equality.** First, the elementary inequality

$$
\forall x > 0:\quad \ln x \le x - 1,\qquad \text{equality iff } x = 1 .
\tag{$\star$}
$$

Proof of $(\star)$: let $g(x) = x - 1 - \ln x$ on $(0,\infty)$; $g'(x) = 1 - 1/x$, so $g$
decreases on $(0,1)$, increases on $(1,\infty)$, and $g(1) = 0$; hence $g \ge 0$ everywhere
with the unique zero at $x = 1$.

Now split by the support $S = \{i : p_i > 0\}$. For $i \in S$ apply $(\star)$ to
$x = q_i/p_i > 0$ and multiply by $p_i > 0$ (order-preserving):

$$
p_i \ln \frac{q_i}{p_i} \;\le\; p_i\Big(\frac{q_i}{p_i} - 1\Big) \;=\; q_i - p_i,
\qquad \text{equality iff } q_i = p_i .
$$

Sum over $i \in S$ (finite):

$$
\sum_{i \in S} p_i \ln \frac{q_i}{p_i} \;\le\; \sum_{i \in S} q_i - \sum_{i \in S} p_i
\;=\; \sum_{i \in S} q_i - 1 \;\le\; 1 - 1 \;=\; 0,
$$

using $\sum_{i\in S} p_i = 1$ and $\sum_{i \in S} q_i \le \sum_{i \in [K]} q_i = 1$.
Indices $i \notin S$ contribute $0$ to KL (with the convention of A5.2; if additionally
$q_i = 0$ the term is $+\infty \ge 0$, which only strengthens the inequality). Therefore

$$
\mathrm{KL}(p\|q) \;=\; -\sum_{i \in S} p_i \ln \frac{q_i}{p_i} \;\ge\; 0 .
$$

*Equality analysis.* Suppose $\mathrm{KL}(p\|q) = 0$. Then no term is $+\infty$, so
$q_i > 0$ whenever $p_i = 0$ would be violated — concretely, any index with
$p_i = 0, q_i = 0$ contributes $+\infty \ne 0$, so $q_i > 0$ there; and equality must hold in
every application of $(\star)$, giving $q_i = p_i$ for all $i \in S$. It remains to check
$i \notin S$: $\sum_{i \notin S} q_i = 1 - \sum_{i \in S} q_i = 1 - \sum_{i \in S} p_i = 0$,
and since $q_i > 0$ on such indices (else $+\infty$), we get $i \notin S \Rightarrow$ no such
index exists with $q_i > 0$, i.e. $q_i = 0$ for $i \notin S$. Together: $q = p$.
Conversely $q = p$ gives $\mathrm{KL} = \sum_i p_i \ln 1 = 0$. This proves (b), including the
boundary cases of A5.3.

**(c) Minimization.** By (a), for every $q$: $\mathrm{CE}(p,q) - H(p) = \mathrm{KL}(p\|q)
\ge 0$ with equality iff $q = p$ by (b). Hence $H(p)$ is attained exactly at $q = p$, which is
therefore the unique minimizer. $\blacksquare$

**Remark 5.1 — the statistical gap (why T5 does not say "the model is calibrated").**
[MATH vs STATISTICAL]
T5 concerns the *population* functional with the true $p$. Training minimizes an *empirical*
sum $\frac1N\sum_n -\ln q_{y^{(n)}}$ over a restricted function class; the empirical minimizer
$\hat q$ satisfies $\mathrm{KL}(p\|\hat q) = O(\text{approximation} + \text{estimation error})$
only under data assumptions, and nothing in T5 transfers to held-out or shifted data without
them. Proper propriety of the loss is a property of the loss, not of any fitted model.

**Remark 5.2 — one-hot boundary case.** [MATH]
For $p = e_y$ (empirical labels): $\mathrm{CE}(e_y, q) = -\ln q_y$ and $H(e_y) = 0$, so the
minimum over the closed simplex is $0$, attained at $q = e_y$ (a boundary point — permitted by
A5.3). On the *open* simplex the infimum $0$ is not attained; this is why (c) is stated over
the closed simplex.

---

## T6 — Brier score: decomposition and optimality

**Theorem T6.**
Let $\{(q^{(n)}, o^{(n)})\}_{n=1}^{N}$ be forecasts and realized outcomes with
$q^{(n)} \in \Delta^{K-1}$ and $o^{(n)} \in \{e_1,\dots,e_K\}$ one-hot. Let
$\{B_1,\dots,B_M\}$ be a partition ("bins") of $\{1,\dots,N\}$ with sizes $n_k$, and define

$$
\text{Brier:}\quad \mathrm{BS} \;=\; \frac{1}{N}\sum_{n=1}^{N} \big\| q^{(n)} - o^{(n)} \big\|_2^2 ,
$$

$$
\bar q_k = \text{mean forecast in } B_k,\qquad
\bar p_k = \frac{1}{n_k}\sum_{n \in B_k} o^{(n)}\ \ (\text{observed outcome frequencies}),\qquad
w_k = \frac{n_k}{N},\qquad
\bar p = \sum_{k} w_k \bar p_k \ \ (\text{climatology}),
$$

$$
\mathrm{Rel} = \sum_k w_k \|\bar q_k - \bar p_k\|^2,\quad
\mathrm{Res} = \sum_k w_k \|\bar p_k - \bar p\|^2,\quad
\mathrm{Unc} = \sum_{i=1}^{K} \bar p_i (1 - \bar p_i) = 1 - \|\bar p\|^2 .
$$

**(a) Decomposition.** If forecasts are constant within each bin
($\forall n \in B_k: q^{(n)} = \bar q_k$ — the canonical choice "bins = distinct forecast
values" makes this exact by construction), then

$$
\boxed{\ \mathrm{BS} \;=\; \mathrm{Rel} \;-\; \mathrm{Res} \;+\; \mathrm{Unc}\ }
$$

exactly (RM), with $\mathrm{Rel} \ge 0$, $\mathrm{Res} \ge 0$, $\mathrm{Unc} \ge 0$.
Consequently $\mathrm{Rel} + \mathrm{Unc} \ge \mathrm{Res}$ for every such partition
(since $\mathrm{BS} \ge 0$).

**(b) Optimality at the truth.** For any fixed true distribution $p \in \Delta^{K-1}$ with
$Y \sim p$ (so $\mathbb{E}[o] = p$),

$$
\mathbb{E}\big[\|q - o\|^2\big] \;=\; \|q - p\|^2 + 1 - \|p\|^2
\;=\; \|q - p\|^2 + \mathrm{Unc}(p),
\qquad\text{hence}\quad
\arg\min_{q \in \Delta^{K-1}} \mathbb{E}\big[\|q - o\|^2\big] = \{p\},
$$

with minimum value $1 - \|p\|^2$; the Brier score is strictly proper, uniquely optimized by
reporting the truth.

**Assumptions.**

* **A6.1** As in the statement: one-hot outcomes; forecasts on the simplex; $N \ge 1$, $K \ge 1$;
  Euclidean norm.
* **A6.2** *(needed for (a) only)* Forecasts are constant within each bin. If bins pool
  heterogeneous forecasts, see Remark 6.2 for the exact extra terms — they do not vanish in
  general, which is why A6.2 is an assumption and not decoration.
* **A6.3** *(needed for (b) only)* A fixed true $p$ with $\mathbb{E}[o] = p$; the expectation
  is over the outcome draw only ($q$ is held fixed — this is the standard propriety setup).
* **A6.4** Exact real arithmetic; $\bar p_k \in \Delta^{K-1}$ (it is a convex combination of
  one-hot vectors), so in particular $\sum_i \bar p_{k,i} = 1$ and
  $\sum_i \bar p_i = 1$.

**Proof.**

**(a)** *Step 1 (per-bin identity).* Fix bin $B_k$. Since forecasts are constant there
($q^{(n)} = \bar q_k$ for all $n \in B_k$, A6.2), expand for each $n \in B_k$:

$$
\|\bar q_k - o^{(n)}\|^2
= \|\bar q_k - \bar p_k\|^2 \;+\; 2\,\langle \bar q_k - \bar p_k,\ \bar p_k - o^{(n)} \rangle
\;+\; \|\bar p_k - o^{(n)}\|^2 .
$$

Average over $n \in B_k$. The middle term: $\bar q_k - \bar p_k$ is a *constant vector* with
respect to $n$ (this uses A6.2 — with varying forecasts it would not factor out), and
$\frac{1}{n_k}\sum_{n \in B_k}(\bar p_k - o^{(n)}) = \bar p_k - \bar p_k = 0$, so the average of
the middle term is $0$. Last term, using $o_i \in \{0,1\} \Rightarrow o_i^2 = o_i$ and
$\frac{1}{n_k}\sum_{n\in B_k} o^{(n)} = \bar p_k$:

$$
\frac{1}{n_k}\sum_{n \in B_k} \|\bar p_k - o^{(n)}\|^2
= \sum_{i=1}^{K} \Big( \bar p_{k,i} - 2\bar p_{k,i}^2 + \bar p_{k,i}^2 \Big)
= \sum_i \bar p_{k,i} - \|\bar p_k\|^2 \;=\; 1 - \|\bar p_k\|^2 .
$$

Hence the bin average is $\|\bar q_k - \bar p_k\|^2 + 1 - \|\bar p_k\|^2$, and weighting by
$w_k$ and summing over bins:

$$
\mathrm{BS} \;=\; \mathrm{Rel} \;+\; 1 \;-\; \sum_k w_k \|\bar p_k\|^2 .
\tag{6a}
$$

*Step 2 (Pythagoras for means).* The $\{\bar p_k\}$ are cluster means around their weighted
mean $\bar p$; expanding:

$$
\mathrm{Res} = \sum_k w_k \|\bar p_k - \bar p\|^2
= \sum_k w_k \|\bar p_k\|^2 - 2\Big\langle \sum_k w_k \bar p_k,\ \bar p \Big\rangle + \|\bar p\|^2
= \sum_k w_k \|\bar p_k\|^2 - \|\bar p\|^2 .
\tag{6b}
$$

*Step 3.* Substituting (6b) into (6a):
$\mathrm{BS} = \mathrm{Rel} + 1 - \big(\|\bar p\|^2 + \mathrm{Res}\big)
= \mathrm{Rel} - \mathrm{Res} + (1 - \|\bar p\|^2)
= \mathrm{Rel} - \mathrm{Res} + \mathrm{Unc}$, where the last equality uses
$\mathrm{Unc} = \sum_i \bar p_i - \sum_i \bar p_i^2 = 1 - \|\bar p\|^2$ (A6.4).
Nonnegativity of Rel, Res (sums of squared norms times $w_k \ge 0$) and of Unc
(each $\bar p_i \in [0,1]$ so each factor nonnegative) hold by inspection; the corollary
$\mathrm{Rel} + \mathrm{Unc} \ge \mathrm{Res}$ follows from the identity and $\mathrm{BS} \ge 0$.

**(b)** With $o = e_Y$, $Y \sim p$, and $q$ fixed:

$$
\mathbb{E}\|q - o\|^2 = \mathbb{E}\big[\|q\|^2 - 2 q^{\top} o + \|o\|^2\big]
= \|q\|^2 - 2\, q^{\top} \mathbb{E}[o] + \mathbb{E}\|o\|^2 .
$$

Now $q^\top \mathbb{E}[o] = q^\top p$ and $\|o\|^2 = 1$ for one-hot $o$, so
$\mathbb{E} = \|q\|^2 - 2 q^\top p + 1$. Evaluate the same expression at $q = p$:
$\|p\|^2 - 2\|p\|^2 + 1 = 1 - \|p\|^2$. Subtracting:

$$
\mathbb{E}\|q - o\|^2 - \mathbb{E}\|p - o\|^2
= \|q\|^2 - 2 q^\top p + \|p\|^2 \;=\; \|q - p\|^2 \;\ge\; 0,
$$

with equality iff $q = p$ (Euclidean norm vanishes only at coincidence). Hence $q = p$ is the
unique minimizer with value $1 - \|p\|^2 = \mathrm{Unc}(p)$. $\blacksquare$

**Remark 6.1 — reading the signs.** Reliability (bad) and uncertainty (fixed by the data's
base rate) *add*; resolution (good — the forecaster's spread of outcome frequencies around
climatology) *subtracts*. Perfect reliability ($\mathrm{Rel}=0$) means
$\bar q_k = \bar p_k$: each bin's average forecast equals its observed frequency. The three
terms are nonnegative, and Brier's propriety in (b) mirrors T5: two distinct strictly proper
scores (log, Brier) agree on the unique truth-telling optimum.

**Remark 6.2 — what happens without A6.2 (honest accounting).** [MATH boundary]
If forecasts vary within a bin, re-run Step 1: the middle term no longer factors to zero.
Writing $V_k = \frac{1}{n_k}\sum_{n\in B_k}\|q^{(n)} - \bar q_k\|^2 \ge 0$ and
$\mathrm{Cov}_k = \frac{1}{n_k}\sum_{n \in B_k}\langle q^{(n)} - \bar q_k,\ o^{(n)} - \bar p_k\rangle$
(the forecast–outcome covariance inside the bin), the exact identity becomes

$$
\mathrm{BS} \;=\; \mathrm{Rel} - \mathrm{Res} + \mathrm{Unc}
\;+\; \sum_k w_k \big( V_k - 2\,\mathrm{Cov}_k \big),
$$

and both corrections vanish iff forecasts are constant within bins ($V_k = 0 \Rightarrow$
$\mathrm{Cov}_k = 0$). Since $\mathrm{Cov}_k$ can have either sign, coarse bins with
heterogeneous forecasts break the clean three-term reading — which is exactly why the theorem
is stated under A6.2 and why binning choices must be reported with any empirical decomposition
[EMPIRICAL reporting rule].

---

## T7 — Parallel-state equivalence (batched ≡ per-question)

**Setup.** Fix a state $S$ and $J \ge 1$ questions $\{(Q_j, C_j, T_j)\}_{j=1}^{J}$ (parallel-safe,
D2.3). Two evaluation procedures:

* **Loop $L$:** compute $H_S = E_M(S)$ once; then for each $j = 1..J$ run
  $z^{(j)} = F(H_S, E_Q(Q_j), E_C(C_j))$, $p^{(j)} = G_{T_j}(z^{(j)})$,
  $p'^{(j)} = \mathrm{cal}(p^{(j)})$, $D_j = \Pi_{T_j}(p'^{(j)}, C_j)$.
* **Batch $B$:** compute $H_S = E_M(S)$ once; process all $J$ questions in one padded,
  masked invocation of the same maps (option axis padded to $K_{\max}$ per (11.2), token axis
  padded per the input axioms), returning $\{\tilde p^{(j)}, \tilde D_j\}_{j=1}^{J}$.

**Theorem T7.**

**(i) Exact equivalence (RM).** Under assumptions A7.1–A7.5 below, in exact real arithmetic,

$$
\forall j \in [J]:\qquad \tilde p^{(j)} \;=\; p^{(j)}
\ \ \text{(componentwise on all coordinates: valid equal, masked } = 0 \text{ on both sides)},
\qquad
\tilde D_j \;=\; D_j .
$$

**(ii) Tolerance equivalence (FP).** Under A7.1, A7.2, A7.3, A7.5 in IEEE-754 (relaxing
A7.4 to its inexact form), there is a configuration-dependent tolerance
$\varepsilon = \varepsilon(\text{model}, \text{shapes}, \text{dtype}, \text{threading})$ such
that $\|\hat{\tilde p}^{(j)} - \hat p^{(j)}\|_\infty \le \varepsilon$ for all $j$. The value of
$\varepsilon$ is **not derived a priori** and must be *measured* per configuration:
(ii) is an [EMPIRICAL] claim template ("measured maximum deviation $= \hat\varepsilon \le
10^{-c}$ on configuration X"), not a theorem. Equality of the *floating-point numbers* is not
claimed.

**(iii) Decision agreement needs a margin.** If additionally, for every $j$, the loop-evaluated
probability gap between the top-1 and top-2 entries exceeds $2\varepsilon(j)$, then
$\tilde D_j = D_j$ as discrete decisions even under (ii). Without a positive margin, flipped
argmax decisions between $B$ and $L$ are possible in principle.

**Assumptions (the exact T7 assumption list).**

* **A7.1 — Determinism.** All encoders/backbone run in evaluation mode: dropout and any
  stochastic sublayer disabled; no RNG consumed inside a forward pass; identical weights
  $\Theta$ (no optimizer step, no in-place parameter or buffer update between or during the two
  runs); identical tokenizer/featurizer versions (D1.3); identical dtype (default binary32);
  deterministic kernels/algorithms and fixed thread count for a *given* configuration.
* **A7.2 — Single shared state.** $H_S = E_M(S)$ is computed exactly once per run from
  identical inputs; the value $H_S$ handed to each question is the same value (no per-question
  recomputation drift) and is **not mutated** between uses (no in-place writes, no aliasing
  reuse of its buffer for something else). Under A7.1 both runs obtain the same $H_S$.
* **A7.3 — No cross-question coupling.** For every $j$ and every $k \ne j$, the subgraph
  producing output $j$ has no incoming edge from any node encoding $(Q_k, C_k)$: no
  attention/matmul/normalization/softmax over the question axis; question-axis padding masked;
  no batch-statistics layer in *train* mode (eval-mode batchnorm uses frozen running stats,
  hence batch-composition-independent); head/calibrator parameters fixed for the whole run
  (A7.1) and question-independent except through inputs (D5.1, parallel-safety D2.3).
* **A7.4 — Equivalent numerics.** *(a)* In RM: $B$ and $L$ compute the same real-valued
  functions of the same arguments — automatic given A7.1–A7.3, A7.5. *(b)* In FP: exact
  bit-equality would additionally require identical operation *order* (association, reduction
  tree, kernel tiling) in both executions. This is **not** assumed in (ii); it is precisely
  what is relaxed, at the cost of the tolerance claim. (Motivation: floating-point addition is
  not associative — see the counterexample in Remark 7.2.)
* **A7.5 — Masking correctness for variable $K$.** (a) Every question has $K_j \ge 1$ valid
  options (D3.1); option-axis padding to $K_{\max}$ uses (11.2): masked logits excluded from
  the partition function, masked probabilities exactly $0$, never eligible for argmax (T2).
  (b) Token-axis padding is excluded from every pooling (P4.3) and every cross-attention
  (mask $B(M_S)$ in (5.e)), and masked contributions are *exactly* zero (or $\max$ is taken
  over valid entries only), so padding changes no value on either side. (c) The masks used by
  $B$ are elementwise identical to the validity sets used by $L$.

**Proof.**

*Lemma 7.1 (padding exactness).* Appending coordinates that (i) are excluded from every
$\max$/softmax denominator and (ii) contribute exactly $+0$ to every sum leaves all computed
values unchanged — in RM trivially (adding $0$ is the additive identity; excluding entries
from a max over a superset with $-\infty$/exclusion recovers the subset max), and in FP too,
because $x \oplus 0.0 = x$ exactly for every finite binary $x$ (IEEE-754 addition of exact
zero is exact; masked softmax coordinates are set to exactly $0.0$ by construction). Padding
itself is therefore error-free on both sides; it is *not* a source of $B$ vs $L$ divergence.
$\square_{\text{Lemma 7.1}}$

1. *(Equality of the shared state.)* By A7.1 and A7.2, both runs compute
   $H_S = E_M(S)$ as the same deterministic function of the same input, yielding the same
   value $h$; no subsequent mutation (A7.2) changes it.
2. *(Equality of each per-question subgraph.)* Fix $j$. Consider the computation DAG
   $\mathcal{G}_j$ of the map
   $(h, Q_j, C_j) \mapsto z^{(j)} \mapsto p^{(j)} \mapsto p'^{(j)} \mapsto D_j$.
   By A7.3 (and Lemma 7.1 with A7.5 for padded coordinates), the nodes evaluated for output
   $j$ inside $B$ are exactly the nodes of $\mathcal{G}_j$ with inputs
   $(h, Q_j, C_j)$ — padding and other questions contribute nothing. Both runs evaluate the
   *same* pure maps ($E_Q$, $E_C$, $F$, $G$, $\mathrm{cal}$, $\Pi$ — pure by A7.1/A7.3) on the
   *same* argument values. Induction on a topological ordering of $\mathcal{G}_j$: base case —
   input nodes hold equal values (step 1, and $Q_j, C_j$ are shared by construction);
   inductive step — a node computing $v = \phi(v_1, \dots, v_r)$ with a fixed pure $\phi$
   receives equal inputs from the induction hypothesis and therefore produces equal output
   (referential transparency in RM). Hence all of $\mathcal{G}_j$'s nodes agree, giving
   $\tilde p^{(j)} = p^{(j)}$ and $\tilde D_j = D_j$. Since $j$ was arbitrary, (i) holds.
   $\blacksquare$ (for (i))

**Remark 7.1 — what A7.3 forbids (and why the guarantee would die without it).**
[MATH boundary] If the heads contained any softmax *across* questions (e.g., flattening the
$(j,k)$ grid into one simplex — explicitly forbidden by (11.3)), or if a layernorm mixed the
question axis, output $j$ would depend on $Q_k$ and step 2's induction would fail: the batched
and looped outputs would differ *systematically*, not just by rounding. P11.1–P11.4 of the
specification are exactly the structural preconditions of T7(i).

**Remark 7.2 — why only a tolerance claim survives in floating point.**
[MATH boundary / EMPIRICAL]
IEEE-754 addition is not associative. Concrete binary32 example: with
$a = 10^{10}$ (exactly representable: $10^{10} = 9765625 \cdot 2^{10}$, significand
$9765625 < 2^{24}$; ulp $= 2^{33-23} = 1024$), we get $(1 \oplus a) \ominus a = 0$
(the increment $1$ rounds away) while $1 \oplus (a \ominus a) = 1$. Two correct orderings of
the same sum disagree by $1$. Batched kernels (matrix-multiply tiling, thread-parallel
reduction over $J \times K_{\max}$ tensors) and looped kernels (reduction over $K_j$) generally
choose *different* orderings, and shape-dependent kernel selection differs, so
$\hat{\tilde p}^{(j)}$ and $\hat p^{(j)}$ are distinct bit patterns in general — each a
rounding of the same RM value with $O(\gamma_{cK})$-class error (cf. Remark 1.2). Hence:
RM equality (i) is a theorem; FP equality is false in general; the FP statement is (ii), a
measured tolerance. Lemma 7.1 shows the variable-$K$ padding contributes *zero* divergence —
the residual $\varepsilon$ comes from reduction/kernel differences elsewhere in the forward
pass, not from padding.

**Remark 7.3 — why decisions need a margin.** [MATH]
Let $\hat p = p + \delta$ with $\|\delta\|_\infty \le \varepsilon$ (componentwise). If
$p_{(1)} - p_{(2)} > 2\varepsilon$, then any competitor satisfies
$\hat p_{(2)} \le p_{(2)} + \varepsilon < p_{(1)} - \varepsilon \le \hat p_{(1)}$, so the
top-1 index is preserved strictly — hence also under the min-index tie-break (strict
inequality means the perturbed top-1 is unique). If the gap is $\le 2\varepsilon$, a
perturbation of size $\varepsilon$ can swap or tie the leaders, and the two procedures may
legitimately return different schema keys. JevX test practice [DESIGN]: assert
probabilities within a declared tolerance, assert decisions equal **only** for cases with a
recorded margin above that tolerance (or assert decision equality on the dataset level and
report disagreement counts).

**Remark 7.4 — the measured tolerance rule (what (ii) means numerically).** [EMPIRICAL]
(ii) is a claim template, and a template still has to say what a caller should compare against.
JevX therefore fixes one rule, derived from a seeded measurement sweep and implemented as
`jevx.utils.numerical.batch_equivalence_tolerance`:

```
tol = safety(dtype) * eps(dtype) * sqrt(T * stages) * scale
```

* `eps(dtype)` — machine epsilon of the **compute** dtype;
* `T` — the padded sequence width that differs between the two runs: the reduction length whose
  *ordering* differs, i.e. the width of the wider call, not the count of valid tokens;
* `stages` — sequential accumulation stages: `num_backbone_layers + 5` for a full
  `JevXForDecision` call (`JEVX_DECISION_ACCUMULATION_STAGES` = state projection, fusion
  context, shared per-option formula, head, probability map), or `1` for a single primitive;
* `scale` — the magnitude the deviation is measured against: `1.0` for probabilities
  (which live in $[0,1]$), or `max|x|` for unbounded quantities;
* `safety(dtype)` — the measured constant (`BATCH_EQUIVALENCE_SAFETY`): **8.0** for float32 and
  **0.5** for bfloat16/float16. Each is the smallest power of two that is at least 3x the worst
  ratio observed between a measured deviation and the bare `eps * sqrt(T*stages) * scale` term.

The anchors behind those constants, from `reports/NUMERICAL_EQUIVALENCE_STUDY.md` (seeded sweep,
no network; 96 pooled-state cells plus 102 end-to-end wrapper cells):

| anchor | worst measured deviation | bare term | ratio | safety |
|---|---:|---:|---:|---:|
| pooled `H_S`, float32, worst of 48 cells (`attention_pool`/L1/T4) | 4.768e-07 | 6.189e-07 | 0.77 | 8.0 |
| pooled `H_S`, bfloat16, worst of 48 cells (`masked_mean`/L1/T8) | 3.906e-03 | 4.402e-02 | 0.089 | 0.5 |
| real 1500-state scenario, 6-layer 27.6M backbone, T=128, stages=11 (`benchmarks/results/real_scenario.json`) | **1.0967e-05** | 4.473e-06 | 2.45 | 8.0 |

The third row is what settles float32. That real walk measured **1.0967e-05** against a
previously hard-coded `1e-5` contract, i.e. it **failed** a bound that was not a bound. Under the
rule above the same walk sits inside `tol = 8 * eps32 * sqrt(128 * 11) = 3.579e-05` — a factor of
**3.26** above the observed value, the margin the derivation asks for. A bare constant cannot do
this job: the deviation grows with the reduction width and with the number of accumulation
stages, and shrinks with precision, so one number is simultaneously too tight (fp32, wide) and
needlessly loose (bf16, narrow).

Two things this rule does **not** claim, consistent with Remarks 7.2 and 7.3:

1. **Not** a proof. The `eps * sqrt(...)` shape is a heuristic accumulation model and the safety
   factor is empirical, per configuration class. A new backbone, dtype, kernel or thread count can
   move the deviation; re-measure rather than inherit these constants.
2. **Not** a decision guarantee. (iii) still governs argmax. The sweep recorded **zero** argmax
   flips across all cells, and the smallest top-1/top-2 margin observed stayed far above `2*tol`,
   but that is an observation on the swept configurations — not what (iii) proves for an arbitrary
   margin. A flip remains possible in principle, and it is a correctness question, not a
   tolerance question.

---

## T8 — Typed-output invariant ($\Pi_T(z) \in \mathcal{D}_T(C)$ for all $z$)

**Theorem T8.**
Let $T$ range over the types of Definition 7 with input domain $Z_T$ and declared decision
space $\mathcal{D}_T(C)$ for a schema $C$ with $K \ge 1$ options. Under the component
theorems' assumptions applied pointwise (A8.1 below), for every $z \in Z_T$:

$$
\Pi_T(z) \;\in\; \mathcal{D}_T(C),
$$

and, moreover, composing with any calibrator satisfying the type-preservation invariant D8.I
($\mathrm{Cal}:\mathcal{P}_K \to \mathcal{P}_K$) preserves membership:

$$
\Pi_T\big(\mathrm{Cal}(z)\big) \;\in\; \mathcal{D}_T(C).
$$

**Assumptions.**

* **A8.1** Pointwise validity of the component results: finite logits (A1.2/A2.5), $K \ge 1$
  (D3.1), schema axioms (D3.x), masks well formed (A7.5) — i.e., exactly the hypotheses under
  which T1–T4 hold, checked per call by input validation.
* **A8.2** $\Pi_T$ is exactly the map defined in Definition 7 (no alternate projection path
  with a wider codomain; a branch that could emit, e.g., a free-text label would invalidate
  (a)).
* **A8.3** (for the composition clause) $\mathrm{Cal}$ preserves the probability object's
  dimension and range: simplex $\to$ simplex, box $\to$ box (D8.I), with nonnegative-
  affine-softmax families outputting simplex points, and PAVA isotonic fitted to $\{0,1\}$
  labels outputting block means in $[0,1]$ (convex combinations of $0/1$ values stay in
  $[0,1]$).
* **A8.4** Exact real arithmetic for the membership claims; FP membership is restored by the
  documented clamps (Remarks 3.2, 4.1) — those clamps are implementation invariants, not
  assumptions of the RM theorem.

**Proof** (case analysis over $\mathcal{T}$; each case names the component theorem).

1. **Choice.** By T1, $p \in \operatorname{int}\Delta^{K-1} \subseteq \Delta^{K-1}$. By T2,
   $i^* = \operatorname{argmax}^{\min}(p) \in [K]$ and the schema slot holds
   $O_{i^*} \in C$. Hence $(p, O_{i^*}) \in \mathcal{D}_{\mathsf{Choice}}(C)$ by (7.1).
2. **Score.** By T1 the level distribution $p$ is a simplex point; by T3,
   $s = \sum_i \ell_i p_i \in [\min I, \max I]$. Membership in (7.2).
3. **Noul / Threshold.** By T4, $p = \sigma(z) \in (0,1) \subset [0,1]$; the decision
   $\mathbf{1}[p \ge \tau] \in \{0,1\}$ by construction of the indicator. Membership in
   (7.3)/(Threshold row).
4. **MultiLabel.** Componentwise application of T4 to each of the $K$ independent sigmoids:
   each $p_k \in (0,1)$, so $p \in (0,1)^K \subseteq [0,1]^K$; decisions in $\{0,1\}^K$.
   Membership in (7.4).
5. **Ordinal.** Structural monotonicity: $a_1 \ge \dots \ge a_{K-1}$ by the parameterization
   of (7.5); $\sigma$ strictly increasing (T4, step 2 of its proof) preserves the order, so
   $c_1 \ge \dots \ge c_{K-1}$ with $c_j = \sigma(a_j) \in (0,1)$; extending by
   $c_0 = 1 \ge c_1$ and $c_{K-1} \ge c_K = 0$, the cumulative vector is nonincreasing on
   $[0,1]$; then $p_i = c_{i-1} - c_i \ge 0$ and $\sum_i p_i = c_0 - c_K = 1$ (telescoping), so
   the induced level distribution lies in $\Delta^{K-1}$. Membership in (7.5).
6. **Regression.** Unbounded: $\Pi$ is the identity into $\mathbb{R}$ — membership is the
   axiom of $\mathbb{R}$. Bounded $[a,b]$: $\mu = a + (b-a)\sigma(z)$; by T4
   $\sigma(z) \in (0,1)$, so $(b-a)\sigma(z) \in (0, b-a)$ and $\mu \in (a,b) \subseteq [a,b]$.
7. **Ranking.** $\operatorname{argsort}^{\text{stable}}$ with any deterministic total order
   (ties broken by schema index) produces a total order on the finite index set $[K]$, and a
   total order on $[K]$ *is* a permutation $\pi \in S_K$ by definition (bijective ranking of
   indices). Hence $\Pi_{\mathsf{Ranking}}(z) \in S_K = \mathcal{D}_{\mathsf{Ranking}}(C)$.
   (In PL-distribution mode, (7.7) is a product of nonnegative ratios summing to $1$ over
   $S_K$: each factor is a softmax-type ratio $> 0$, and the telescoping normalization
   $\sum_{\pi} P(\pi) = 1$ is the standard Luce property — membership in the simplex over
   $S_K$.)
8. **Abstention.** By (7.8), the branch returns either $\Pi_{T_0}(z) \in \mathcal{D}_{T_0}(C)$
   (cases 1–7 by induction on the base type) or the symbol $\bot$; both are elements of
   $\mathcal{D}_{T_0}(C) \cup \{\bot\}$.
9. **Composition with calibration.** Let $p' = \mathrm{Cal}(p)$. By A8.3,
   $p'$ remains in the same $\mathcal{P}_K$ (simplex/box), so the preconditions of cases 1–8
   still hold for $p'$ (e.g., a simplex vector fed to T2/T3 remains eligible; a $[0,1]$ value
   remains eligible for clamping). Therefore $\Pi_T(p') \in \mathcal{D}_T(C)$.
   $\blacksquare$

**Corollary 8.1 (pipeline closure).** [MATH under A8.x] For every validated input
$(S, Q, C)$ (finite tensors, $K \ge 1$, type/schema consistent), the full pipeline of (7.9)
satisfies $J_\Theta(S,Q,C) \in \mathcal{D}_T(C)$.

**Corollary 8.2 (what T8 does *not* say).** [boundary]
Membership is a *shape/range* property. T8 is consistent with: arbitrarily wrong decisions,
perfectly miscalibrated probabilities, adversarially nonsense inputs — none of these are
excluded by being inside the type space. T8 upgrades nothing into a correctness claim.

---

## Guarantee taxonomy — five distinct kinds of statements

JevX documentation labels every load-bearing claim with exactly one of these kinds.

| Kind | What makes it true | Examples in this repo | Failure mode |
|---|---|---|---|
| **Mathematical guarantee** | Deduction from stated assumptions in a stated arithmetic model (RM unless noted) | T1–T8; padding exactness (Lemma 7.1); per-question normalization (11.3); D8.I composition | Assumptions violated (NaN, $K=0$, cross-question edges) or FP deviations ignored |
| **Empirical guarantee** | Reproducible measurements with a stated setup and tolerance | FP deviations $\le \hat\varepsilon$ (T7-ii); test-suite pass rates; latency/throughput numbers on the stated machine; support-level status | Configuration changes; finite test coverage hides a defect |
| **Statistical property** | Probability statements that require data assumptions (i.i.d./exchangeability, fixed distributions) and carry sampling error | ECE/NLL/Brier on a held-out set (8.5); risk–coverage / coverage-at-risk (10.2)–(10.4); "temperature scaling improves reliability on distribution $\mathcal{D}$" | Distribution shift; small calibration sets; estimator bias (binning); no distribution-free finite-sample claim for post-hoc calibrators |
| **Implementation invariant** | True because runtime validation asserts it on every call (the `jevx/validation.py` ValidationEngine called from `JevXForDecision.forward` behind `JevXConfig.runtime_validation`, default on: per-question $\lvert\sum_i \hat p_i - 1\rvert \le \text{tol}$, shape/$K$ preservation across the calibration step (calibrated probabilities keep the logits' shape — the D8.I dimension check), noul strict interiority, confidence $\in [0,1]$, predicted-index range; masked probabilities $= 0$ additionally hold by construction — masked softmax — and every row below is also asserted by the test suites) | $\lvert\sum_i \hat p_i - 1\rvert \le \text{tol}$ per question (never across questions); $K$-preservation across calibration (D8.I shape check); type-directed invariants dispatched on `schema_metadata["decision_type"]` (choice/score simplex, noul single column in $(0,1)$, i.e. the clamps to $[\varepsilon, 1-\varepsilon]$); masked probabilities $= 0$ | A code path bypassing the validator; validator bug (mitigated by tests, i.e., by empirics) |
| **Model-performance claim** | Only from scoped benchmarks with baselines, data, and uncertainty stated | Scoped claims exist only where the benchmark suite ran: cross-family structural validation of 18 checkpoints (reports/REAL_MODEL_VALIDATION.md) plus text-quality benchmarks for `google/bert_uncased_L-2_H-128_A-2` on SST-2 and their ablations/latency/memory companions (reports/DATASET_BENCHMARKS.md, reports/ABLATIONS.md, reports/LATENCY_BENCHMARKS.md, reports/MEMORY_BENCHMARKS.md; JSON in `benchmarks/results/`). **No per-family quality benchmark exists for the matrix's cited checkpoints — hence no level 4 is claimed** | Fails on other data/tasks; incomparable setups; selection bias |

## Explicit non-claims (forbidden statements)

The following are **not** claimed anywhere in JevX, and any document or output asserting them
is wrong:

1. **"A calibrated model is always correct."** False: calibration is an *aggregate reliability
   property of probabilities* (e.g., among cases where the model reports $0.9$, about $90\%$
   are correct — measured on a stated distribution). It says nothing about any individual
   decision; a perfectly calibrated model is still wrong on the cases where its probability
   says it is. (T5/T6 give propriety of *scores*, not correctness of *fits*.)
2. **"Confidence guarantees correctness."** False: $c_{\max}$, margin, entropy, and energy
   (Definition 9) are deterministic functions of logits. Only after *separate, measured*
   validation (calibration or selective-risk analysis on matched data, Definitions 8–10) does
   a confidence threshold bound an *expected* error rate on *that* distribution — a
   statistical statement, never a per-instance guarantee. Unvalidated confidence can be
   uninformative (e.g., constant confidence has zero resolution).
3. **"JevX universally superior"** (and synonyms: *revolutionary, unbeatable, guaranteed
   better, universally superior* — banned vocabulary). Performance is task-, data-, and
   compute-dependent; JevX makes no superiority claim of any kind, and the universality of the
   *mechanism* (scope: LIMITATIONS.md) is not a claim about the *quality* of its outputs.
4. **"Batched and looped results are bit-identical."** Not claimed: only RM equality (T7-i)
   and a measured FP tolerance (T7-ii).
5. **"Parallel answers are statistically independent."** Not claimed: only functional/graph
   separation (P11.1), with explicit dependence through shared $H_S$, shared parameters, and
   correlated inputs (P11.2).

To *upgrade* any non-claim into a legitimate claim requires: for (1)–(2), held-out
measurements under stated data assumptions with uncertainty reported; for (3), a pre-specified
benchmark against named baselines with uncertainty; for (4), bitwise determinism studies; for
(5), a data-level independence analysis with its own assumptions.
