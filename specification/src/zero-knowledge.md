# Zero-Knowledge

Formally, a proof system is zero-knowledge if there is an efficient simulator $\mathsf{S}$ capable of producing
transcripts without knowledge of a secret witness and even when no witness exists, such that no efficient distinguisher
$\mathsf{D}$ can distinguish simulated transcripts from authentic ones. This page describes the steps the Triton VM
prover undertakes to ensure that the proofs it produces satisfy this property, and furthermore proves that these
techniques do in fact realize that intention.

Specifically, Triton VM achieves zero-knowledge through three core mechanism pillars: batch randomization, trace polynomial randomization, and quotient table randomization, all of which apply at the interactive oracle proof level. The next step is to show that the zero-knowledge is retained under the transformations that send the interactive oracle proof to a non-interactive proof. The following sections detail these randomization techniques, construct an explicit, efficient simulator $\mathsf{S}$, and mathematically bound the probability of information leakage.

## Notation

Throughout this document we will use the following symbols.

| Symbol | Domain / Type | Definition & Context |
| :--- | :--- | :--- |
| $\mathsf{S}$ | Simulator algorithm | An efficient simulator capable of generating valid transcripts without a secret witness. |
| $\mathsf{D}$ | Distinguisher algorithm | A computationally bounded adversary attempting to distinguish simulated transcripts from authentic ones. |
| $\mathbb{F}$ | Base field | The primary finite field over which the main trace polynomials $t_i(X)$ are defined. |
| $\mathbb{E}$ | Extension field | Extension field of degree $[\mathbb{E} : \mathbb{F}] = e$ over which auxiliary trace polynomials and DEEP challenges live. |
| $e$ | Integer | Extension degree $[\mathbb{E} : \mathbb{F}] = e$. |
| $N$ | Integer | Trace domain length / number of rows in the execution trace. |
| $D$ | Set / Subgroup | LDT (Low-Degree Test) domain, disjoint from the trace domain. |
| $\rho$ | Scalar in $(0, 1)$ | Rate of the Reed-Solomon code; $\rho \|D\|$ bounds segment polynomial degrees. |
| $h$ | Integer | Number of randomizer coefficients injected into trace polynomials ($h = t + kef + 1$). |
| $\sigma$ | Integer | Number of coefficients of the quotient table randomizer $s_k(X)$; $\sigma := (h+1)(k+1)$. |
| $t$ | Integer | Number of in-domain rows queried during the low-degree test. |
| $f$ | Integer | Fan-in of the AIR (Algebraic Intermediate Representation) circuit. |
| $\mathsf{m}$ | Integer | Total number of main trace polynomials. |
| $\mathsf{w}$ | Integer | Total number of randomized trace polynomials (main + auxiliary), not counting the batch randomizer. |
| $k$ | Integer | Number of quotient segments into which the quotient polynomial $q(X)$ is split. |
| $\zeta$ | Field element | Fixed STARK parameter used to randomize the quotient table segments. |
| $\alpha$ | Element in $\mathbb{E}$ | Out-of-domain challenge point sampled during the DEEP-ALI step. |
| $Z(X)$ | Polynomial | Zerofier (vanishing polynomial) for the trace domain. |
| $Q$ | Integer | Maximum number of random oracle invocations made by a bounded distinguisher. |
| $P$ | Probability | Upper bound on the probability of a distinguisher gleaning witness information from un-salted Merkle structures. |

## Zero-Knowledge of the Interactive Oracle Proofs

Triton VM applies three randomization steps, summarized as follows.

1. A uniformly random polynomial, called the *batch randomizer*, is added to the batch of polynomials. With the addition
   of the batch randomizer, any linear combination of the trace and quotient polynomials is itself uniform and therefore
   perfectly independent of the witness. All codewords that arise in the course of the low-degree test are downstream
   from this linear combination and therefore share this property.
2. All trace polynomials (both main and auxiliary) contain $h$ field elements worth of entropy. The variable $h$ is
   chosen such that all in-domain and out-of-domain rows that are queried as part of the batch-check of the low-degree
   test are uniform.
3. The quotient table is extended with one column and the entire table is randomized in a way that preserves the ability
   to extract the value of the quotient at a certain out-of-domain point while perfectly hiding the values of the
   individual segments for each revealed row.

### Batch-Randomizer

> **Intuition:** The batch randomizer acts like a "one-time pad" for the low-degree test. By adding a uniformly random polynomial to the random linear combination of trace and quotient polynomials, the resulting combination codeword becomes perfectly independent of the witness. That independence percolates to all subsequent codewords in the low-degree test.

The batch randomizer is a uniformly random polynomial of degree less than $\rho |D|$ that is included into the random
linear combination of polynomials in the batching step, in preparation for the low-degree test (or, depending on your
perspective, as its initialization step). Its codeword (specifically, its list of evaluations on the trace domain) is
adjoined to the auxiliary trace but left unconstrained by all AIR constraints.

The effect of adding the batch randomizer is that all codewords sent in the course of the low-degree test are perfectly
independent of the witness. To demonstrate witness-independence, let $\mathbf{c}$ be the first combination codeword in
a given accepting transcript, and let $\{ \hat{t}_i(X) \}_i$ be *any* choice for the randomized trace polynomials and
$\{ \hat{q}_i(X) \}_i$ *any* choice for the randomized quotient segment polynomials. Isolating the batch-randomizer term
$b(D)$ in the batch equation yields

$$ \sum_{i=0}^{\mathsf{w}-1} w_i \cdot \hat{t}_i(D) + \sum_{i = 0}^{k} w_{\mathsf{w}+1+i} \cdot \hat{q}_i(D)
+ w_\mathsf{w} \cdot b(D) = \mathbf{c} \enspace , $$

where:

 - $D$ is the LD domain.
 - The first sum runs over all $\mathsf{w}$ randomized trace polynomials and the second over all $k+1$ randomized
   quotient segment polynomials.
 - $b(D)$ is the batch randomizer. It is a column of the auxiliary table – the column of index $\mathsf{w}$, which the
   count $\mathsf{w}$ does not include – trace-randomized differently from the trace columns, as detailed in the note
   under "Determining $h$" below. The weight
   $w_\mathsf{w}$ multiplying it is zero with probability at most $1/\vert{}\mathbb{E}\vert{}$, which is folded directly
   into the soundness error.

The right hand side, $\boldsymbol{c}$, must be a Reed-Solomon codeword because it is a linear combination of
Reed-Solomon codewords. Consequently, there must be some low degree polynomial $c(X)$ that agrees with $\boldsymbol{c}$
on $D$.

Decompose $c(X)$ into a sum of three contributions:

$$c(X) = c_\mathsf{t}(X) + c_\mathsf{s}(X) + b(X) $$

1. $c_\mathsf{t}$ is the contribution from everything that depends on the trace – not just the (randomized) trace polynomials themselves $\{\hat{t_i}(X)\}_{i=0}^{\mathsf{w}-1}$ but also the raw (unrandomized) quotient segments $\{q_i(X)\}_{i=0}^{k-1}$. The trace polynomials have degree less than $N + h$; but the quotient segments have degree less than $\rho |D|$, and the latter dominates.
2. $c_\mathsf{s}$ is the contribution of the quotient table randomizer $s_k(X)$. The degree is likewise less than $\rho |D|$.
3. $b$ is the batch randomizer. By construction, the degree is less than $\rho |D|$.

Since $b$ is uniform over the space that contains the other two contributions, the sum is uniform and independent of the trace.

### Randomized Trace Polynomials

> **Intuition:** The prover must reveal certain rows of the execution trace during the low-degree test. To prevent these rows from leaking information about the secret witness, we inject $h$ random elements into each trace column. As long as the verifier queries no more than $h$ rows, the revealed values appear perfectly random.

#### Mechanism

Let $\{t_i(X)\}_{i=0}^{\mathsf{m}-1}$ be the main trace polynomials (defined over base field $\mathbb{F}$) and
$\{t_i(X)\}_{i=\mathsf{m}}^{\mathsf{w}-1}$ the auxiliary trace polynomials (defined over extension field
$\mathbb{E}$ with extension degree $[\mathbb{E} : \mathbb{F}] = e$). To randomize these polynomials we compute

$$ \hat{t}_i(X) = t_i(X) + Z(X) \cdot r_i(X) \enspace , $$

where:

 - $t_i(X)$ is the original trace polynomial.
 - each $r_i(X)$ is a uniformly random polynomial over the respective field of degree less than $h$.
 - $Z(X)$ is the zerofier (vanishing polynomial) for the trace domain.

By definition, $Z(X)$ evaluates to zero on the trace domain. Therefore, randomization by means of adding $Z(X) \cdot r_i(X)$ does not alter the values of the trace polynomials on the trace domain. However, outside the trace domain, the $h$ randomizer coefficients guarantee that up to $h$ evaluations of $\hat{t}_i(X)$ are perfectly independent of $t_i(X)$. Formally, for any trace polynomial $t_i(X)$, evaluating $\hat{t}_i(X)$ at $h$ distinct points from outside the domain uniquely determines $r_i(X)$, which can be found through interpolation. This method for obtaining $r_i(X)$ works for *any* trace polynomial $t_i(X)$.

#### Determining $h$

The number of randomized coefficients $h$ must be large enough to cover all rows disclosed to the verifier. The total
number of revealed rows comes from multiple protocol steps: 

 - In-Domain Queries ($t$): The low-degree test (LDT) queries $t$ rows from the LDE domain, which is disjoint from the
   trace domain.
 - Out-of-Domain (OOD) Queries ($ef$): During the DEEP-ALI protocol, the prover must supply $f$ rows to evaluate the
   AIR circuit (where $f$ is the fan-in). Because the OOD challenge $\alpha$ lives in the extension field, one OOD row
   of the main table is equivalent to $e$ base field rows. This step requires $ef$ equivalent rows for the main trace
   and $f$ rows for the auxiliary trace.
 - Quotient Table Margin ($(k-1)ef$): An additional margin of $(k-1)ef$ coefficients is required to account for the
   randomized quotient table. (To be explained in more detail below.)
 - Merkle Tree Margin ($1$): One final coefficient is added to secure the Merkle trees. (Also more details below.)

For simplicity, Triton VM uses a single value for $h$ that works for both the main and auxiliary traces. Summing these
components yields:

$$h = t + kef + 1$$

> **Note:** The batch randomizer $b(X)$ is trace-randomized like every other column, but with $N$ randomizer
coefficients instead of $h$. The randomized trace domain, whose length is $\rho |D|$, is exactly twice as long as the
trace domain, so $\rho |D| = 2N$. Writing $b(X) = a(X) + Z(X) \cdot r(X)$ with $\deg a < N$ and $\deg r < N$, the pair
$(a, r)$ has $2N$ uniform coefficients and the map $(a, r) \mapsto a + Zr$ is injective: were $a + Zr = 0$ with
$r \neq 0$, then $Zr = -a$ would have degree at least $N$ and less than $N$ simultaneously. The map is therefore a
bijection onto the polynomials of degree less than $\rho |D|$, which is what makes $b(X)$ uniform over that entire
space. The $h$ coefficients used for every other column would not achieve this, since $N + h < \rho |D|$; and $N$
coefficients are the most that fit, since $\deg(Zr) = N + \deg r$ must stay below $\rho |D|$.

> **Note:** Besides requiring that the LDT and trace domains are disjoint sets, it is also important to disallow
verifiers who sample $\alpha$ from the trace domain. In the non-interactive case, the probability of this event is
negligible and may safely be ignored, but interactively the prover must guard against malicious verifiers employing
this strategy.

### Quotient Table Randomization

> **Intuition:** The quotient polynomial $q(X)$ is too large to process in a single piece, so it is segmented into $k$
smaller polynomials. However, evaluating these raw segments directly would leak information about the quotient (and
thus the trace). To prevent this, we construct a "chain" of randomized segments, where a single high-degree random
polynomial $s_k(X)$ cascades its entropy down through all the lower segments.

#### Segmentation

Segmentation is the process of splitting the large quotient polynomial $q(X)$ into $k$ segments $q_i(X)$ of degree less
than $\rho |D|$ such that

$$ q(X) = \sum_{i=0}^{k-1} X^i q_i(X^k) .$$

> **Note:** The degree of a quotient segment is less than $\rho |D|$, and that number is larger than $N$ (trace
length) because the trace polynomials are randomized.

#### Randomization

To mask these raw segments, we construct $k+1$ randomized segments $s_i(X)$ as follows:

1. Sample $s_k(X)$ uniformly of degree less than $\sigma := (h+1)(k+1)$. The length $\rho |D|$ of the randomized trace
   domain is chosen such that $\sigma \leqslant \rho |D|$, so all $k+1$ segments respect the degree bound that the
   low-degree test enforces.
2. For $0 \leqslant i < k$, iteratively define:

$$s_i(X) := q_i(X) - \zeta^i s_{i+1}(\zeta^k X) .$$

The constant $\zeta$ is a fixed parameter of the STARK with the following constraints:

1. $\zeta^k$ has multiplicative order larger than $k$.
2. No powers in the set $\{\zeta^{mk}\}_{m=1}^k$ belong to the field's 2-adic subgroups.

Furthermore, define

$$ \begin{align*}
p(X) &:= \sum_{i=0}^{k-1} X^i s_i(X^k) \\
r(X) &:= \sum_{i=0}^{k-1} \zeta^i X^i s_{i+1}(\zeta^k X^k) . \\
\end{align*} $$

By summing them and substituting the definition of $s_i(X)$, the randomizers perfectly cancel out, leaving the original quotient:

$$ \begin{align*}
p(X) + r(X) &= \left( \sum_{i=0}^{k-1} X^i s_i(X^k) \right) +
\left( \sum_{i=0}^{k-1}\zeta^i X^i s_{i+1}(\zeta^k X^k) \right) \\
&= \sum_{i=0}^{k-1} X^i \left(s_i(X^k) + \zeta^i s_{i+1}(\zeta^k X^k)\right) \\
&= \sum_{i=0}^{k-1} X^i q_i(X^k) \\
&= q(X) .\\
\end{align*} $$

The *randomized quotient table* consists of the $k+1$ segments' codewords: $\{s_i(D)\}_{i=0}^{k}$. During the DEEP-ALI
protocol, the prover reveals *two* out-of-domain rows of $k$ elements each:

 - First out-of-domain row: $\{s_i(\alpha^k)\}_{i=0}^{k-1}$.
 - Second out-of-domain row: $\{s_i(\zeta^k \alpha^k)\}_{i=1}^{k}$.

These out-of-domain rows allow the verifier to compute $p(\alpha)$ and $r(\alpha)$ and hence $q(\alpha)$.
The DEEP-ALI verifier equates $q(\alpha)$ to the value of the AIR constraints applied to
the revealed out-of-domain trace rows, after dividing out the zerofier.

Two DEEP updates (single-point quotients) link the two out-of-domain rows to the randomized quotient table,
establishing the integrity of $p(\alpha)$ and $r(\alpha)$.

> **Note:** The two rows jointly determine more than just that sum: pairing them entrywise recovers each $q_i(\alpha^k)$
separately, since $q_i(\alpha^k) = s_i(\alpha^k) + \zeta^i s_{i+1}(\zeta^k\alpha^k)$, and hence $k$ evaluations of the quotient $q(X)$. This surplus of revealed information is what the margin $(k-1)ef$ on
the number of trace randomizers $h$ is for.

#### In-Domain Rows and First Out-of-Domain Row

We must show that observing $t$ in-domain rows of the quotient table and the first out-of-domain row reveals nothing
about the raw quotient $q(X)$.

Given $t$ in-domain rows the distinguisher observes $\{s_i(x_j)\}$ for each of the $k+1$ segments and for all
indeterminates $\{x_0, \ldots, x_{t-1}\}$. Using the recursive definition of $s_i(X)$ for $i < k$, we can substitute
$s_i(x_j)$ with $-\zeta^i s_{i+1}(\zeta^k x_j) + \langle\!$ *terms that only depend on* $q(X) \rangle$.

With every substitution, the indeterminate picks up another factor $\zeta^k$, sending

$$x_j \mapsto \zeta^k x_j \mapsto \zeta^{2k} x_j \mapsto \ldots .$$

Ultimately, every in-domain row becomes an invertible affine transformation of the vector
$\{s_k(\zeta^{mk}x_j)\}_{m=0}^k$. The concrete transformation depends on $q(X)$ and $\zeta$, but the underlying
variables are solely evaluations of the top randomizer segment $s_k(X)$.

Applying this exact same substitution process to the first out-of-domain row, the set $\{s_i(\alpha^k)\}_{i=0}^{k-1}$
becomes $\{s_k(\zeta^{(k-i)k}\alpha^k)\}_{i=0}^{k-1} = \{s_k(\zeta^{mk}\alpha^k)\}_{m=1}^{k}$.

##### Uniformity

Let $G$ be the complete set of indeterminates at which $s_k$ is evaluated across all these rows:

$$ G := \{\{\zeta^{mk} x_j\}_{m=0}^k\}_{j=0}^{t-1} \cup \{\zeta^{mk}\alpha^k\}_{m=1}^{k} . $$

In total, there are $g := t(k+1) + k$ elements revealed by the $t$ in-domain and 1 out-of-domain row. As long as all
elements in $G$ are distinct (meaning $|G| = g$), these revealed elements uniquely determine $g$ points on $s_k(X)$ for
any fixed quotient $q(X)$ and admissible $\zeta$.

As long as $g \leqslant \sigma$, the map sending $s_k(X)$ to its vector of evaluations on $G$ is surjective, so that
vector is uniform. Therefore, this combination of $t$ in-domain and 1 out-of-domain revealed rows is perfectly masked by
$s_k(X)$ and is entirely independent of the quotient.

> **Note:** The choice $\sigma = (h+1)(k+1)$ implies this inequality with room to spare. Since $h = t + kef + 1$ and
$g = t(k+1) + k$, it follows that $\sigma - g - k - 1 = (h + 1 - t)(k+1) - 2k - 1 = (kef + 2)(k+1) - 2k - 1 > 0$. So
$s_k(X)$ has at least $k+1$ more coefficients than the transcript pins evaluations. This margin will be used in the
section "Zero-Knowledge IOP to Zero-Knowledge IP".

##### No collisions in $G$

To ensure $s_k(X)$ provides exactly $g$ degrees of freedom, we must prove that $G$ contains no duplicates. The constraints placed on $\zeta$ guarantee this property for the in-domain rows:

1. No collisions within one row. The set $\{\zeta^{mk} x_j\}_{m=0}^k$ has exactly $k+1$ distinct elements because the
   multiplicative order of $\zeta^k$ is strictly greater than $k$.
2. No collision between rows. Because all in-domain $x_j$ are sampled from the same coset, a collision
   $\zeta^{mk}x_j = \zeta^{m^\prime k}x_j^\prime$  would imply that $\zeta^{(m - m^\prime)k} = x_j^{-1} x_j^\prime$ is
   an element of a 2-adic subgroup, violating the second constraint on $\zeta$. (Multiplicative subgroups are closed
   under inversion, so that constraint covers negative exponents $m - m^\prime$ as well.)

For the out-of-domain challenge $\alpha$, the prover must explicitly reject any combination of $\alpha$ and in-domain points for which $\{\zeta^{mk} \alpha^k\}_{m=1}^k$ has a non-empty intersection with the in-domain points $\{\{\zeta^{mk} x_j\}_{m=0}^k\}_{j=0}^{t-1}$. This intersection is non-empty precisely when $\alpha^k = \zeta^{\Delta k} x_j$ for some revealed $x_j$ and some $-k \leqslant \Delta < k$. For any other combination, the OOD row safely contributes $k$ distinct elements to $G$. In the non-interactive case, the probability of a conflict is negligible; the prover performs the check regardless.

#### Second Out-of-Domain Row

We cannot rely on the interpolation argument for the *second* out-of-domain row $\{s_{i}(\zeta^k\alpha^k)\}_{i=1}^{k}$.
After substitutions it maps to
$\{s_k(\zeta^{(k-i+1)k} \alpha^k)\}_{i=1}^{k} = \{s_k(\zeta^{mk} \alpha^k)\}_{m=1}^{k}$, which is the very same set of
evaluations of $s_k$ that the first out-of-domain row already gives rise to. It therefore contributes no new elements
to $G$.

Evaluating the definition of the randomized segments at $\alpha^k$ yields

$$q_i(\alpha^k) = s_i(\alpha^k) + \zeta^i s_{i+1}(\zeta^k \alpha^k) .$$

Because the first OOD row provides $s_i(\alpha^k)$ and the second provides $s_{i+1}(\zeta^k \alpha^k)$, the two out-of-domain rows jointly reveal *all* $k$ unrandomized values $\{q_i(\alpha^k)\}_{i=0}^{k-1}$ – concretely more information than the single value $q(\alpha)$ that the verifier needs.

##### Bijective Resolution

To retain zero-knowledge, we must prove these $k$ leaked values are themselves independent of the trace.

Consider a modified protocol where the prover sends $k$-many $f$-tuples of out-of-domain trace rows corresponding to
the preimages of $\{q(\omega^i \alpha)\}_{i=0}^{k-1}$ for a primitive $k$-th root of unity $\omega$.

The extra margin $(k-1)ef$ on the
bound on the number of trace randomizers $h$ guarantees that even these $k \times f$ extension-field rows are
independent of the trace (as well as uniform). Consequently, the images $\{q(\omega^i \alpha)\}_{i=0}^{k-1}$ of these
$f$-tuples are independent of the trace (though not necessarily uniform).

From the segmentation equation

$$ q(X) = \sum_{i=0}^{k-1} X^i q_i(X^k) ,$$

one obtains, for any $X \neq 0$, a bijection between the $k$ quotient evaluations $\{q(\omega^i X)\}_{i=0}^{k-1}$ and
the evaluations of the $k$ raw (unrandomized) quotient segments $\{q_i(X^k)\}_{i=0}^{k-1}$. To see this, evaluate the
segmentation equation in all of $\{\omega^i X\}_{i=0}^{k-1}$ to obtain $k$ equations or one equation of $k$-vectors
involving an invertible matrix and an invertible Hadamard product.

Considering the first out-of-domain row fixed, there is also a bijective equivalence between
$\{q_i(\alpha^k)\}_{i=0}^{k-1}$ and the second out-of-domain row $\{s_{i}(\zeta^k \alpha^k)\}_{i=1}^{k}$. It follows
that even this second out-of-domain row is a deterministic image of a variable that is
independent of the trace.

Consequently, the distinguisher in the zero-knowledge game for this modified protocol cannot use this second
out-of-domain row to his advantage. It follows that the distinguisher in the zero-knowledge game for the original
protocol, which has strictly less information, cannot use it either.

### Simulation

The simulator $\mathsf{S}$ is given the claim but no witness. This section treats the interactive oracle proof, in which
the DEEP challenge $\alpha$ is a message from the honest verifier; $\mathsf{S}$ is free to sample it itself and to build
the rest of the transcript around it. The section “Zero-Knowledge IP to Zero-Knowledge NIP” below explains how the same
simulator survives the Fiat-Shamir transform, which is where the freedom to choose $\alpha$ first stops being free.

$\mathsf{S}$ proceeds as follows.

1. Take an arbitrary main and auxiliary trace of the correct dimensions – the all-zero trace will do – which need
   satisfy no AIR constraint whatsoever. Adjoin a uniformly random batch-randomizer column, randomize all trace
   polynomials authentically with freshly sampled trace randomizers, low-degree extend, and commit. This produces the
   main and auxiliary Merkle roots. Call the randomized trace $\hat{T}^\star$.
2. Sample $\alpha \in \mathbb{E}$ uniformly.
3. For each $0 \leqslant i < k$, obtain an $f$-tuple of out-of-domain rows at $\omega^i\alpha$ for a primitive $k$-th
   root of unity $\omega$: for $i = 0$ read the tuple off $\hat{T}^\star$, and for $i > 0$ sample it uniformly. Apply
   the AIR and divide out the zerofiers to obtain $k$ values $v_i$ – the values a real transcript's quotient would take
   at $\{\omega^i\alpha\}_{i=0}^{k-1}$.
4. Sample $\tilde{q}(X)$ uniformly among polynomials of degree less than $\rho |D| k$, subject to the $k$ linear
   constraints $\tilde{q}(\omega^i\alpha) = v_i$. Segment it, randomize the segmentation with a freshly sampled uniform
   $s_k(X)$ exactly as the honest prover does, low-degree extend, and commit.
5. Emit the $\alpha$ of step 2 as the verifier's challenge, and run the remainder of the honest prover verbatim: the two
   out-of-domain quotient rows, the DEEP combination, the low-degree test, and the openings at the queried indices.
   Abort exactly where the honest prover would, namely if the sampled indices conflict with $\alpha$ as described
   above. The remaining challenges are sampled uniformly, as an honest verifier would.

> **Note:** $\mathsf{S}$ is efficient: it runs the prover once and solves a small linear system. It never touches a witness, and
nothing in the above requires one to exist.

#### Why the Verifier Accepts

The verifier accepts because $\tilde{q}(X)$ is a genuine low-degree polynomial and $\tilde{q}(\alpha) = v_0$ is by
construction the value that the out-of-domain trace rows of $\hat{T}^\star$ demand of it.

Note that this is the *only* place the AIR is checked. At the in-domain indices, the verifier checks the DEEP update
(single-point quotient) which yields a codeword that is batched in the low-degree test. So $\mathsf{S}$ is free to set
the trace to an arbitrary value because no in-domain checks will ever test that it satisfies the AIR.

#### Hybrid Argument

It remains to be shown that the two distributions, of honest transcripts on the one hand, and of simulated ones on 
the other, are indistinguishable. The following hybrid argument establishes this fact.

Define a sequence of distributions $G_0, G_1, G_2, \ldots$ where the first is identically the distribution of honest
transcripts, and every next distribution accumulates one incremental change. If in every consecutive pair the first
is indistinguishable from the second, then the endpoints of the sequence are likewise indistinguishable from each
other. To apply this strategy we must define incremental changes and argue why the resulting distribution is
indistinguishable from its predecessor. Every distribution computes everything downstream of its change honestly.
All of the verifier's randomness is sampled honestly and identically across distributions.

 1. $G_0$ is the distribution of authentic transcripts.
 2. $G_1$: uniform combination codeword.
     - **Change:** Sample $c(X)$ uniformly at random subject to $\deg(c) < \rho |D|$. Compute $b(X)$ as a function of $c(X)$ and the other terms in the batch equation.
     - **Indistinguishable because:** There is a bijection between $b(X)$ and $c(X)$ given by the equation from Section "Batch-Randomizer": $c(X) = c_{\mathsf{t}}(X) + c_{\mathsf{s}}(X) + b(X)$. Sample $b$ then compute the bijection; or sample $c$ and invert it – both produce exactly the same distribution.
 3. $G_2$: uniform view of quotient.
     - **Change:** Sample the revealed $t$ in-domain quotient rows and the first out-of-domain quotient row uniformly at random, and sample $s_k(X)$ uniformly subject to the evaluations these revealed values pin.
     - **Indistinguishable because:** By the argument in "Quotient Table Randomization", for every quotient $q(X)$ and every admissible $\zeta, \alpha, \{x_j\}_{j=0}^{t-1}$, the vector of all $g = t(k+1) + k$ revealed values is the image under an affine invertible map of a vector of as many evaluations of $s_k(X)$ at distinct indeterminates. The same logic applies: sample the argument, compute the bijection; or sample the image, and invert it – both sequences result in identical distributions.
 4. $G_3$: bogus quotient.
     - **Change:** Sample $\tilde{q}$ uniformly of degree less than $\rho |D| \cdot k$ and subject to $\tilde{q}(\omega^i \alpha) = v_i$ for $i \in \{0,\ldots,k-1\}$ where $v_i = q(\omega^i \alpha)$. Populate the quotient table with $\tilde{q}$ instead of $q$.
     - **Indistinguishable because:** At this point (after $G_1$ and $G_2$) only the second out-of-domain quotient row depends on the quotient. Conditioned on the first row, which is independently uniform, the second out-of-domain row is a deterministic bijective image of $\{v_i\}_{i=0}^{k-1}$, which is explicitly pinned by the change. Same bijection argument; identical distributions.
 5. $G_4$: uniform view of trace.
     - **Change:** Except for the batch randomizer column, sample the rows at $\{x_j\}_{j=0}^{t-1}$ and the $k$ $f$-tuples at $\{\omega^i \alpha\}_{i=0}^{k-1}$ uniformly at random, and sample the $r_i$ uniformly subject to the evaluations these revealed values pin. Compute $v_i$ by applying the AIR to the $i$-th $f$-tuple and dividing out the zerofier evaluated at $\omega^i \alpha$.
     - **Indistinguishable because:** By the argument in Section "Randomized Trace Polynomials", every main column reveals $t + kef = h - 1$ evaluations, and every auxiliary column $t + kf$, all at indeterminates outside the trace domain. For every column, the vector of revealed values is a bijective image of a vector of as many evaluations of $r_i(X)$ at distinct indeterminates. The $v_i$ are computed from values whose distribution has not changed.
 6. $G_5$: bogus trace.
     - **Change:** Replace the authentic trace with an all-zero trace, or an arbitrary one.
     - **Indistinguishable because:** There is nothing left in the transcript that depends on the trace. The trace still determines the unqueried values of the trace oracles; however, those are not part of the *verifier's view*. (In the compiled protocol those values do reach the verifier, through the Merkle roots and authentication paths. The section "Zero-Knowledge IOP to Zero-Knowledge IP" bounds the resulting distinguisher advantage by $P$.)

Finally, $G_5$ is precisely the distribution of transcripts produced by the simulator $\mathsf{S}$. The simulator
samples the trace randomizers $r_i(X)$, the top quotient randomizer $s_k(X)$, and the batch randomizer $b(X)$
directly and reads the revealed values off them. $G_5$ draws the revealed values and samples those polynomials
subject to them. These are the same distributions, by the same bijection argument that justifies $G_1$, $G_2$,
and $G_4$, but applied in the opposite direction. The endpoints of the sequence are therefore indeed
indistinguishable.

## Zero-Knowledge of the Interactive and Non-Interactive Proofs

### Zero-Knowledge IOP to Zero-Knowledge IP

[BCS, §6 & §7](https://eprint.iacr.org/2016/116.pdf) presents and analyzes the canonical IOP-to-NIROP transformation.
The analysis shows that that transformation retains zero-knowledge.

Triton VM uses Merkle trees without salts (or any other form of hiding commitment scheme) but otherwise the same
transformation. However, the BCS argument that zero-knowledge is retained fails under this change. So below
we articulate a new one. This argument is specific to the context of STARKs and may fail in a more general IOP context.

The objects that are introduced by the BCS transform and that are *a priori* capable of leaking information are the
Merkle authentication paths and Merkle roots. These objects are images of data that should remain hidden. Something
else – not salts, since there are none – must establish that they leak no information.

Distinguish on the one hand Merkle trees built from the batch codeword, or from codewords downstream from it; from on
the other hand, Merkle trees built from low-degree-extended trace or quotient tables. Since the batch contains a batch
randomizer, the batch codeword is a uniformly random Reed-Solomon codeword and perfectly independent of the trace.
Therefore, even if the complete list of leaf-preimages to this Merkle root or of any Merkle root of a codeword
downstream from it were released, that release would leak no information about the trace. It follows that the Merkle
roots and authentication paths (or authentication structures, in case they are compressed) corresponding to the batch
codeword or codewords downstream from it are independent of the trace.

What remains is the Merkle trees built out of the low-degree-extended trace and quotient tables, with the important
feature that both are randomized using the respective strategies described above. For these Merkle trees we
quantitatively bound the probability $P$ that information is leaked and show that this probability is negligible and
concretely irrelevant.

Consider instead of the internal Merkle nodes and Merkle roots, the complete list of leafs, which are hashes of rows in
the low-degree-extended trace and quotient tables. Clearly the Merkle authentication paths and roots can be computed
from this list of leaf digests, so it suffices to bound the amount of information leaked by this list instead.

Consider the event in which the distinguisher recognizes one of these leaf digests because the digest appears as the
image field of a list $L$ of preimage-and-image pairs produced by invoking the random oracle $Q$ times. (These
invocations may even happen *after* the distinguisher receives the transcript.) If this event happens then all bets
about concealing the trace are off because the distinguisher can infer the value from one more row than accounted for.
If this event does not happen, then all observed leaf digests are new and therefore leak no information about their
corresponding preimages.

Consider first the low-degree-extended (and randomized) *main* trace. To recognize a row by its hash, the distinguisher
must have queried the random oracle on the exact row that produced that hash. That means guessing the entire set of
main trace randomizers correctly. (We consider the trace fixed.) The remaining 1
coefficient of margin on the number $h$ of randomizers *per column* means that there are $\mathsf{m}$ (number of main
columns) variables left to guess. Any one preimage-image pair from the list $L$ corresponds to $|D|$-many *triples*
consisting of one row index, one preimage, and one image. So for every hash invocation, the distinguisher can test $|D|$
distinct hypotheses for the randomizers' values. The distinguisher is successful if any one of his hypotheses is true.
There are $Q$ invocations of the random oracle, so $|D| \times Q$ hypotheses in total. (The corresponding
events are not independent but the union bound, which is used next, does not need independence.)
The search space is $\mathbb{F}^\mathsf{m}$ so the probability
of the distinguisher gleaning information from the main trace is no greater
than $\frac{|D| Q}{|\mathbb{F}|^\mathsf{m}}$.

An analogous argument holds for the auxiliary trace, where the number of columns is $\mathsf{w}-\mathsf{m}$ and the
columns contain extension field elements. So the probability that the distinguisher gleans information from the
auxiliary table is no greater than $\frac{|D| Q}{|\mathbb{F}|^{e(\mathsf{w}-\mathsf{m})}}$.

For the quotient table, the relevant randomness is not the trace randomizers but the freedom left in $s_k(X)$. By the
substitution argument of the section "In-Domain Rows and First Out-of-Domain Row", the row of the randomized quotient
table at an index $x \in D$ is an invertible affine image of the vector $\{s_k(\zeta^{mk}x)\}_{m=0}^{k}$. Let $x$ be
any index other than the $t$ revealed ones. The argument of the section "No collisions in $G$" – "no collision between
rows" applies verbatim to $x$: if $\zeta^{mk}x$ were equal to $\zeta^{m^\prime k}x_j$ for some revealed $x_j$, then
$\zeta^{(m-m^\prime)k} = x^{-1}x_j$ would lie in a 2-adic subgroup, which the constraints on $\zeta$ forbid unless
$m = m^\prime$ and hence $x = x_j$. The block $\{\zeta^{mk}x\}_{m=0}^{k}$ is therefore disjoint from the in-domain part
of $G$. Furthermore, it can coincide with the out-of-domain part $\{\zeta^{mk}\alpha^k\}_{m=1}^{k}$ in at most $k$ of
its $k+1$ elements. So at least $1$ of these $k+1$ elements lies outside of $G$.

Since $g + k + 1 \leqslant \sigma$ – see the note in section "Quotient Table Randomization" – the evaluation of
$s_k(X)$ at that element (and generically at all $k+1$ elements) is uniform and independent of everything in the
transcript. A guessed quotient row therefore agrees with the true one with probability at most $1/|\mathbb{E}|$, and
the probability that the distinguisher recognizes a quotient-table leaf digest in any of his $|D| \cdot Q$ attempts is
bounded by
$$
\frac{|D| \cdot Q}{|\mathbb{E}|} .
$$

In conclusion, the probability that the *computationally bounded* distinguisher, who is restricted to $Q$ invocations
of the random oracle, finds in the transcript one of the responses to his random oracle queries, is bounded by

$$ P \leq |D| \cdot Q \cdot \Biggl(
\frac{1}{|\mathbb{F}|^\mathsf{m}} +
\frac{1}{|\mathbb{F}|^{e(\mathsf{w}-\mathsf{m})}} +
\frac{1}{|\mathbb{F}|^{e}}
\Biggr) .
$$

> **Note:** This argument is not complete. A complete argument would present a simulator and show that it produces
transcripts that are indistinguishable from authentic ones. We do not do that. What the preceding sections do establish
is the substance of that argument, which follows the skeleton of the hybrid argument laid out above with one
important difference: the justification why $G_5$ produces the same distribution. That justification covers the
verifier's view and "The unqueried values of the trace oracles are not part of the verifier's view." This quoted proposition is true for the interactive oracle proof but is false after BCS compilation because the Merkle roots and
authentication paths are functions of exactly that information. The bound on $P$ is what licenses this step in the
compiled setting.

### Zero-Knowledge IP to Zero-Knowledge NIP

Applying the Fiat-Shamir transform does not affect the distribution of the transcript. It does, however, affect the
simulator who needs to know the value of the challenge $\alpha$ before producing the commitments from which it is
pseudorandomly derived. To achieve this, the simulator is defined relative to the *programmable random oracle model*,
which enables him to specify (a sparse list of) query-response pairs that the random oracle must abide by.