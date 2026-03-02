# Zero-Knowledge

Formally, a proof system is zero-knowledge if there is an efficient simulator $\mathsf{S}$ capable of producing
transcripts without knowledge of a secret witness and even when no witness exists, such that no efficient distinguisher
$\mathsf{D}$ can distinguish simulated transcripts from authentic ones. This page describes the steps the Triton VM
prover undertakes to ensure that the proofs it produces satisfy this property, and furthermore proves that these
techniques do in fact realize that intention.

## Zero-Knowledge of the Interactive Oracle Proofs

Triton VM applies three randomization steps, summarized as follows.

1. A uniformly random polynomial, called the *batch randomizer*, is added to the batch of polynomials. With the addition of the batch randomizer, any
   linear combination of the trace and quotient polynomials is itself uniform and therefore perfectly independent of the
   witness. All codewords that arise in the course of the low-degree test are downstream from this linear combination
   and therefore share this property.
2. All trace polynomials (both main and auxiliary) contain $h$ field elements worth of entropy. The variable $h$ is
   chosen such that all in-domain and out-of-domain rows that are queried as part of the batch-check of the low-degree
   test are uniform.
3. The quotient table is extended with one column and the entire table is randomized in a way that preserves the ability to extract
   the value of the quotient at a certain out-of-domain point while perfectly hiding the values of the individual segments for each
   revealed row.

### Batch-Randomizer

The batch randomizer is a uniformly random polynomial that is included into the random linear combination of polynomials
in the batching step, in preparation for the low-degree test (or, depending on your perspective, as its initialization step). Its codeword (specifically, its list of evaluations on the trace domain) is adjoined to the auxiliary trace but left
unconstrained by all AIR constraints.

The effect of adding the batch randomizer is that all codewords sent in the course of the low-degree test are perfectly
independent of the witness. To see this, let $\mathbf{c}$ be the first combination codeword in a given accepting
transcript, and let $\{ \hat{t}_i(X) \}_i$ be *any* choice for the randomized trace polynomials and
$\{ \hat{q}_i(X) \}_i$ *any* choice for the randomized quotient segment polynomials. Isolate the batch-randomizer term
$r$ in the batch equation

$$ \sum_{i=0}^{\mathsf{w}-1} w_i \cdot \hat{t}_i(D) + \sum_{i = 0}^{k} w_{\mathsf{w}+i} \cdot \hat{q}_i(D) + r(D)
= \mathbf{c} \enspace , $$

where $D$ is the LDT domain and the sums run over all $\mathsf{w}$ randomized trace polynomials and all $k+1$ randomized
quotient segment polynomials. The right hand side, $\boldsymbol{c}$ must be a Reed-Solomon codeword because it is a linear combination of Reed-Solomon codewords. Consequently, there must be some low degree polynomial $c(X)$ that agrees with $\boldsymbol{c}$ on $D$. The distribution of $c(X)$ equals that of $r(X)$ up to translation, and even this "up to translation" is unnecessary because the distribution is in fact uniform over polynomials of bounded degree. This argument establishes that
$\mathbf{c}$ is a uniformly random Reed-Solomon codeword and, consequently, that no distinguisher has any advantage over a random
guess at distinguishing simulated from authentic transcripts based on the codewords from the low-degree test alone
because that would entail distinguishing distributions that are identical.

### Randomized Trace Polynomials

Let $\{t_i(X)\}_{i=0}^{\mathsf{m}-1}$ be the main trace polynomials (defined over base field $\mathbb{F}$) and
$\{t_i(X)\}_{i=\mathsf{m}}^{\mathsf{w}-1}$ the auxiliary trace polynomials (defined over extension field
$\mathbb{E}$ with extension degree $[\mathbb{E} : \mathbb{F}] = e$). To randomize these polynomials we compute

$$ \hat{t}_i(X) = t_i(X) + Z(X) \cdot r_i(X) \enspace , $$

where each $r_i(X)$ is a uniformly random polynomial over the respective field of degree less than $h$, and where $Z(X)$
is the zerofier (vanishing polynomial) for the trace domain.

The addition of $h$-many randomizer coefficients ensures that $h$ values of $\hat{t}_i(X)$ are independent of $t_i(X)$,
provided that $X$ is drawn from the same field and does not live in the trace domain. To see this, let
$\{x_0, \ldots, x_{h-1}\}$ be distinct indeterminates satisfying these criteria. Restricting the above equation to
$\{x_0, \ldots, x_{h-1}\}$ gives $h$ points that $r_i(X)$ must agree with, and $r_i(X)$ may be found through
interpolation. This method for obtaining $r_i(X)$ works for *any* trace polynomial $t_i(X)$.

Since the target domain of the low-degree extension step, also known as the LDT domain, is a disjoint set from the trace domain, it follows that $h$ rows of the low-degree extended trace are independent of the trace. Let $t$ be the number of
in-domain rows queried by the low-degree test. Obviously, $h \geqslant t$ must be true for independence, but this alone is not enough because there
are also out-of-domain rows released in the course of the DEEP-ALI protocol. For these rows we must take into account
the extension degree $e$ as well as the fan-in $f$ of the AIR circuit. Since the out-of-domain indeterminate $\alpha$
lives in the extension field, one out-of-domain row of the main table is equivalent to $e$ base field rows. Furthermore,
to admit one evaluation of the AIR, $f$ rows must be supplied. So for the main trace, the prover discloses an equivalent
of $t + ef$ and for the auxiliary trace an equivalent of $t + f$ rows. Consequently, $h \geqslant t + ef$ for the main
trace polynomials and $h \geqslant t + f$ for the auxiliary trace polynomials.

For simplicity, we choose to have only one $h$ that works for both main and auxiliary traces. Furthermore, we anticipate the need for a margin of $(k-1)ef$ coefficients originating from the randomized quotient table. So: $h = t + kef$.

**Note:** The batch-randomizer $\hat{r}(X)$ is also trace-randomized:
$\hat{r}(X) = r(X) + Z(X) \cdot r_{\mathsf{w}}(X)$, where $r(X)$ is the *unrandomized* batch-randomizer of $N$ uniform
coefficients, $N$ is the length of the trace domain, and $r_{\mathsf{w}}(X)$ is the trace randomizer for the
batch-randomizer which adds another $h$ uniform coefficients.

**Note:** Besides requiring that the LDT and trace domains are disjoint sets, it is also important to disallow verifiers who sample $\alpha$ from the trace domain. In the non-interactive case, the probability of this event is negligible and may safely be ignored, but interactively the prover must guard against malicious verifiers employing this strategy.

### Quotient Table Randomization

Segmentation is the process of splitting the large quotient polynomial $q(X)$ into $k$ segments $q_i(X)$ of degree less than $\rho |D|$ such that

$$ q(X) = \sum_{i=0}^{k-1} X^i q_i(X^k) .$$

Note that $\rho$ is the rate of the Reed-Solomon code and $D$ its domain of definition. Moreover, note that $\rho D > N$ (where $N$ is the trace length) because of trace polynomial randomization.

To use a randomized quotient table instead of these raw segments, construct $k+1$ segments $s_i(X)$ as follows:

1. Sample $s_k(X)$ uniformly of degree less than $\rho |D|$.
2. For $0 \leqslant i < k$, define $s_i(X) := q_i(X) - \zeta^i s_{i+1}(\zeta X)$.

The constant $\zeta$ is a fixed parameter of the STARK with the following constraints:

1. $\zeta$ has multiplicative order larger than $2k$, and
2. none of the powers $\{\zeta^i\}_{i=1}^k$ are an element of the field's 2-adic subgroups.

The reasons for this is expanded upon below.

Furthermore, define

$$ \begin{align*}
p(X) &:= \sum_{i=0}^{k-1} X^i s_i(X^k) \\
r(X) &:= \sum_{i=0}^{k-1} \zeta^i X^i s_{i+1}(\zeta^k X^k) \\
\end{align*} $$

and observe that

$$ \begin{align*}
p(X) + r(X) &= \left( \sum_{i=0}^{k-1} X^i s_i(X^k) \right) +
\left( \sum_{i=0}^{k-1}\zeta^i X^i s_{i+1}(\zeta^k X^k) \right) \\
&= \sum_{i=0}^{k-1} X^i \left(s_i(X^k) + \zeta^i s_{i+1}(\zeta^k X^k)\right) \\
&= \sum_{i=0}^{k-1} X^i q_i(X^k) \\
&= q(X) .\\
\end{align*} $$

The “randomized quotient table” consists of the $k+1$ segments' codewords: $\{s_i(D)\}_{i=0}^{k}$. There are two
out-of-domain rows of $k$ elements each: $\{s_i(\alpha^k)\}_{i=0}^{k-1}$ and $\{s_i(\zeta^k \alpha^k)\}_{i=1}^{k}$.
These out-of-domain rows allow the verifier to compute $p(\alpha)$ and $r(\alpha)$ and hence $q(\alpha)$. The DEEP-ALI
verifier equates $q(\alpha)$ to the value of the AIR constraints applied to the revealed out-of-domain trace rows, after
dividing out the zerofier.

Two DEEP updates (single-point quotients) link the two out-of-domain rows to the randomized quotient table,
establishing the integrity of $p(\alpha)$ and $r(\alpha)$.

Given $t$ rows of the quotient table, the distinguisher observes $\{s_i(x_j)\}$ for each of the $k+1$ segments and
indeterminates $\{x_0, \ldots, x_{t-1}\}$. Using the definition of $s_i(X)$ for $i < k$, we can replace
$s_i(x_j)$ by $-\zeta^i s_{i+1}(\zeta x_j) + \langle\!$ *terms that only depend on* $q(X) \rangle$ and ultimately by
$(-1)^{k-i} \zeta^i \cdots \zeta^k s_k(\zeta^{k-i} x_j) + \langle\!$ *terms that only depend on* $q(X) \rangle$. With
every replacement, the indeterminate is sent to a new value $x_j \mapsto \zeta^{k-i} x_j \mapsto \ldots$. It follows
that every row in-domain row is an invertible affine transformation of the vector $\{s_k(\zeta^{k-i} x_j)\}_{i=0}^k$,
or equivalently, of the vector $\{s_k(\zeta^i x_j)\}_{i=0}^k$, where the concrete transformation depends on the quotient
$q(X)$ and on $\zeta$.

Consider the *first* out-of-domain row. After applying all possible substitutions, $\{s_i(\alpha^k)\}_{i=0}^{k-1}$ become s
$\{s_k(\zeta^{k-i}\alpha^k)\}_{i=0}^{k-1} = \{s_k(\zeta^{i}\alpha^k)\}_{i=1}^{k}$. Let

$$
G := \{\{\zeta^i x_j\}_{i=0}^k\}_{j=0}^{t-1}
\cup \{\zeta^i\alpha^k\}_{i=1}^{k}
$$

the set of indeterminates in which $s_k$ is evaluated.

In total, there are $g := t(k+1) + k$ elements revealed by the $t$ in-domain and 1 out-of-domain row. As long as
$|G| = g$, the revealed elements uniquely determine $g$ points on $s_k(X)$, for any fixed quotient $q(X)$ and any
admissible choice of $\zeta$. As long as $g \leqslant \rho |D|$, $s_k(X)$ can be found by interpolation. It follows that
under these conditions, any revealed combintation of $t$ in-domain and 1 out-of-domain row are independent of the quotient.

Finally, we show that $|G| = g$. Because of the constraints on $\zeta$, $t$ distinct in-domain indeterminates $x_j$ give
rise to $t(k+1)$ distinct indeterminates for $s_k$. In particular,

1. $\{\zeta^i x_j\}_{i=0}^k$ is of size $(k+1)$ because of the multiplicative order of $\zeta$, and
2. since all $x_j$ are sampled from the same coset, there is no $x_j^\prime \neq x_j$ such that
   $\zeta^ix_j = x_j^\prime$, since that would imply $\zeta^i = x_j^{-1} x_j^\prime$ be an element of a 2-adic subgroup,
   violating the second constraint on $\zeta$.

The prover must reject $\alpha$ such that $\{\zeta^i\alpha^k\}_{i=1}^k$ and $\{\{\zeta^i x_j\}_{i=0}^k\}_{j=0}^{t-1}$ have a non-empty intersection. In this case, $\{\zeta^i\alpha^k\}_{i=1}^k$ contributes another $k$ distinct elements to $G$, proving $|G| = g$. In the non-interactive case, the probability of sampling a malicious $\alpha$ is negligible and may safely by ignored.

Note that we cannot apply the same argument to the *second* out-of-domain row $\{s_{i}(\zeta^k\alpha^k)\}_{i=1}^{k}$ because after substitutions it maps to $\{s_k(\zeta^{2k-i} \alpha^k)\}_{i=1}^{k} = \{s_k(\zeta^{i} \alpha^k)\}_{i=k}^{2k-1}$ which repeats $s_k(\zeta^k \alpha^k)$. Therefore, we must have a second argument to show that the second out-of-domain row is *also* independent of the trace, which follows below.

Consider the modified protocol where the prover sends not one $f$-tuple of out-of-domain trace rows corresponding to the preimage of $q(\alpha)$, but $k$-many $f$-tuples of out-of-domain trace rows corresponding to the preimages of $\{q(\omega^i \alpha)\}_{i=0}^{k-1}$ for a primitive $k$-th root of unity $\omega$. The extra margin $(k-1)ef$ on the bound on the number of trace randomizers $h$ guarantees that even these $k \times f$ extension-field rows are independent of the trace (as well as uniform). Consequently, the images $\{q(\omega^i \alpha)\}_{i=0}^{k-1}$ of these $f$-tuples are independent of the trace (though not necessarily uniform).

From the segmentation equation 

$$ q(X) = \sum_{i=0}^{k-1} X^i q_i(X^k) ,$$

one obtains a bijection between $\{q(\omega^i X)\}_{i=0}^{k-1}$ and $\{q_i(X^k)\}_{i=0}^{k-1}$. To see this, substitute $X$ for all of $\{\omega^i X\}_{i=0}^{k}$ to obtain $k$ equations or one equation of $k$-vectors involving an invertible matrix and an invertible Hadamard product.

It follows that the set $\{q(\omega^i \alpha)\}_{i=0}^{k-1}$ is bijectively equivalent to $\{q_i(\alpha^k)\}_{i=0}^{k-1}$. According to the definition of the randomized segments, $\{q_i(\alpha^k)\}_{i=0}^{k-1} = \{s_i(\alpha^k) + \zeta^i s_{i+1}(\zeta^k \alpha^k)\}$ and note that $\{s_i(\alpha^k)\}_{i=0}^{k-1}$ is exactly the first out-of-domain row of the randomized quotient table, which was already established to be independent of the trace (as well as uniform). Considering this first out-of-domain row fixed, there is also a bijective equivalence between $\{q_i(\alpha^k)\}_{i=0}^{k-1}$ and $\{s_{i+1}(\zeta^k \alpha^k)\}_{i=0}^{k-1} = \{s_{i}(\zeta^k \alpha^k)\}_{i=1}^{k}$, which is exactly the second out-of-domain row. It follows that even this second out-of-domain row is a deterministic image of a variable that is independent of the trace.

Consequently, the distinguisher in the zero-knowledge game for this modified protocol cannot use this second out-of-domain row to his advantage. It follows that the distinguisher in the zero-knowledge game for the original protocol, which has strictly less information, cannot use it either.

### Simulation

The simulator takes the trace, adjoins the batch-randomizer, and randomizes the trace polynomials authentically. There
is no guarantee that the resulting quotient $q(X)$ is low degree. However, after computing $q(X)$, the simulator factors
$q(X) - v$ for random values $v \in \mathbb{E}$ until a linear factor $X-\alpha$ is found. The simulator proceeds to use
this $\alpha$ for the DEEP challenge along with a random quotient $q^\star(X)$ of the right degree such that
$q^\star(\alpha) = v$ to populate the quotient table with. The simulator randomizes the quotient table. All field
elements in the transcript are, by the above arguments, guaranteed to be independent of the trace. Consequently, no
distinguisher has any advantage over a random guess. Moreover, the switch between $q(X)$ and $q^\star(X)$ ensures that
the verifier accepts.

## Zero-Knowledge of the Interactive and Non-Interactive Proofs

### Zero-Knowledge IOP to Zero-Knowledge IP

[BCS, §6 & §7](https://eprint.iacr.org/2016/116.pdf) presents and analyzes the canonical IOP-to-NIROP transformation.
The analysis shows that that transformation retains zero-knowledge.

Triton VM uses unsalted Merkle trees but otherwise the same transformation. However, the argument that zero-knowledge is
retained is affected by this change.

TODO.

### Zero-Knowledge IP to Zero-Knowledge NIP
