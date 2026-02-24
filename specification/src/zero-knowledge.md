# Zero-Knowledge

Formally, a proof system is zero-knowledge if there is an efficient simulator $\mathsf{S}$ capable of producing
transcripts without knowledge of a secret witness and even when no witness exists, such that no efficient distinguisher
$\mathsf{D}$ can distinguish simulated transcripts from authentic ones. This page describes the steps the Triton VM
prover undertakes to ensure that the proofs it produces satisfy this property, and furthermore proves that these
techniques do in fact realize that intention.

## Zero-Knowledge of the Interactive Oracle Proofs

Triton VM applies three randomization steps, summarized as follows.

1. A uniformly random polynomial is added to the batch of polynomials. With the addition of the batch randomizer, any
   linear combination of the trace and quotient polynomials is itself uniform and therefore perfectly independent of the
   witness. All codewords that arise in the course of the low-degree test are downstream from this linear combination
   and therefore share this property.
2. All trace polynomials (both main and auxiliary) contain $h$ field elements worth of entropy. The variable $h$ is
   chosen such that all in-domain and out-of-domain rows that are queried as part of the batch-check of the low-degree
   test are uniform.
3. The quotient table is extended with one column and the entire table is randomized in a way that preserves extracting
   the value of the quotient at a certain point while perfectly hiding the value of individual segments for each
   revealed row.

### Batch-Randomizer

The batch randomizer is a uniformly random polynomial that is included into the random linear combination of polynomials
in the batching step, in preparation for the low-degree test. Its codeword is adjoined to the auxiliary trace but left
unconstrained by all AIR constraints.

The effect of adding the batch randomizer is that all codewords sent in the course of the low-degree test are perfectly
independent of the witness. To see this, let $\mathbf{c}$ be the first combination codeword in a given accepting
transcript, and let $\{ \hat{t}_i(X) \}_i$ be *any* choice for the randomized trace polynomials and
$\{ \hat{q}_i(X) \}_i$ *any* choice for the randomized quotient segment polynomials. Isolate the batch-randomizer term
$\mathbf{r}$ in the batch equation

$$ \sum_{i=0}^{\mathsf{w}-1} w_i \cdot \hat{t}_i(D) + \sum_{i = 0}^{k} w_{\mathsf{w}+i} \cdot \hat{q}_i(D) + \mathbf{r}
= \mathbf{c} \enspace , $$

where $D$ is the LDT domain and the sums run over all $\mathsf{w}$ randomized trace polynomials and all $k+1$ randomized
quotient segment polynomials. The vector $\mathbf{r}$ is indeed a Reed-Solomon codeword if $\mathbf{c}$ is, in which
case $\mathbf{r}$ agrees with some low-degree polynomial $\hat{r}(X)$ on $D$. This argument establishes that
$\mathbf{c}$ is uniform if $\hat{r}(X)$ is and, consequently, that no distinguisher has any advantage over a random
guess at distinguishing simulated from authentic transcripts based on the codewords from the low-degree test alone
because that would entail distinguishing distributions that are identical.

### Randomized Trace Polynomials

Let $\{t_i(X)\}_{i=0}^{\mathsf{m}-1}$ be the main trace polynomials (defined over base field $\mathbb{F}$) and
$\{t_i(X)\}_{i=\mathsf{m}}^{\mathsf{w}-1}$ the auxiliary trace polynomials (defined over extension field
$\mathbb{E}$ with extension degree $[\mathbb{E} : \mathbb{F}] = e$). To randomize these polynomials we compute

$$ \hat{t}_i(X) = t_i(X) + Z(X) \cdot r_i(X) \enspace , $$

where each $r_i(X)$ is a uniformly random polynomial over the respective field of degree less than $h$, and where $Z(X)$
is the zerofier (vanishing polynomial) for the trace domain.

The addition of $h$-coefficient randomizers ensures that $h$ values of $\hat{t}_i(X)$ are independent of $t_i(X)$,
provided that $X$ is drawn from the same field and does not live in the trace domain. To see this, let
$\{x_0, \ldots, x_{h-1}\}$ be distinct indeterminates satisfying these criteria. Restricting the above equation to
$\{x_0, \ldots, x_{h-1}\}$ gives $h$ points that $r_i(X)$ must agree with, and $r_i(X)$ may be found through
interpolation. This method for obtaining $r_i(X)$ works for *any* trace polynomial $t_i(X)$.

It follows that $h$ rows of the low-degree extended trace are independent of the trace. Let $t$ be the number of
in-domain rows queried by the low-degree test. Obviously, $h \geqslant t$, but this alone is not enough because there
are also out-of-domain rows released in the course of the DEEP-ALI protocol. For these rows we must take into account
the extension degree $e$ as well as the fan-in $f$ of the AIR circuit. Since the out-of-domain indeterminate $\alpha$
lives in the extension field, one out-of-domain row of the main table is equivalent to $e$ base field rows. Furthermore,
to admit one evaluation of the AIR, $f$ rows must be supplied. So for the main trace, the prover discloses an equivalent
of $t + ef$ and for the auxiliary trace an equivalent of $t + f$ rows. Consequently, $h \geqslant t + ef$ for the main
trace polynomials and $h \geqslant t + f$ for the auxiliary trace polynomials.

For simplicity we choose to have only one $h$ that works for both main and auxiliary traces. Furthermore, we anticipate
another term $kef$ originating from the quotient table randomization. So: $h = t + ef + kef$.

**Note:** The batch-randomizer $\hat{r}(X)$ is also trace-randomized:
$\hat{r}(X) = r(X) + Z(X) \cdot r_{\mathsf{w}}(X)$, where $r(X)$ is the *unrandomized* batch-randomizer of $N$ uniform
coefficients, $N$ is the length of the trace domain, and $r_{\mathsf{w}}(X)$ is the trace randomizer for the
batch-randomizer which adds another $h$ uniform coefficients.

**Note:** In order for the above argument to work, the low-degree trace domain $D$ must be disjoint from the trace
domain. Otherwise, $Z(X)$ may cancel some contributions of $r_i(X)$.

### Quotient Table Randomization

Segmentation is the process of splitting the large quotient polynomial $q(X)$ into $k$ segments $q_i(X)$ such that

$$ q(X) = \sum_{i=0}^{k-1} X^i q_i(X^k) .$$

For quotient randomization, construct $k+1$ segments $s_i(X)$ as follows:

1. Sample $s_k(X)$ uniformly of degree less than $\rho |D|$.
2. For $0 \leqslant i < k$, define $s_i(X) := q_i(X) - \zeta^{-i} s_{i+1}(\zeta^{-k}X)$.

The constant $\zeta$ is an almost-arbitrary, fixed parameter of the STARK. The one constraint is that
$\zeta^{-k}$ has a multiplicative cycle larger than $k$; the reason for this is expanded upon below.

Furthermore, define

$$ \begin{align*}
p(X) &:= \sum_{i=0}^{k-1} X^i s_i(X^k) \\
r(X) &:= \sum_{i=0}^{k-1} \zeta^{-i} X^i s_{i+1}(\zeta^{-k} X^k) \\
\end{align*} $$

and observe that

$$ \begin{align*}
p(X) + r(X) &= \left( \sum_{i=0}^{k-1} X^i s_i(X^k) \right) +
\left( \sum_{i=0}^{k-1}\zeta^{-i} X^i s_{i+1}(\zeta^{-k} X^k) \right) \\
&= \sum_{i=0}^{k-1} X^i \left(s_i(X^k) + \zeta^{-i} s_{i+1}(\zeta^{-k} X^k)\right) \\
&= \sum_{i=0}^{k-1} X^i q_i(X^k) \\
&= q(X) .\\
\end{align*} $$

The “randomized quotient table” consists of the $k+1$ segments' codewords: $\{s_i(D)\}_{i=0}^{k}$. There are two
out-of-domain rows of $k$ elements each: $\{s_i(\alpha^k)\}_{i=0}^{k-1}$ and $\{s_i(\zeta^{-k} \alpha^k)\}_{i=1}^{k}$.
These out-of-domain rows allow the verifier to compute $p(\alpha)$ and $r(\alpha)$ and hence
$q(\alpha) = p(\alpha) + r(\alpha)$. The DEEP-ALI verifier equates $q(\alpha)$ to the value of the AIR constraints
applied to the revealed out-of-domain trace rows, after dividing out the zerofier.

Two DEEP updates (single-point quotients) suffice to link the two out-of-domain rows to the randomized quotient table,
establishing the integrity of $p(\alpha)$ and $r(\alpha)$ as well. (As a practical performance matter, it is prudent to
release two rows of $k+1$ elements each so that batching marries well with the DEEP-update. The extra elements, which
are not needed for computing $p(\alpha)$ or $r(\alpha)$, are also covered by the following proof of the zero-knowledge
property.)

Given $t+1$ rows of the quotient table, the distinguisher observes $\{s_i(x_j)\}$ for each of the $k+1$ segments and
indeterminates $\{x_0, \ldots, x_{t}\}$.

Using the definition of $s_i(X)$ for $0 \leqslant i < k$, we can replace $s_i(x_j)$ by
$-\zeta^i s_{i+1}(\zeta^{-k}x_j) + <$ *some terms that depend on $q(X)$* $>$ and ultimately by
$(-1)^{k-i} (\zeta^{-1})^{\sum_{\iota = i}^k \iota} s_k(\zeta^{-(k-i)k} x_j) + <$ *some terms that depend on*
$q(X) >$. With every replacement, the indeterminate is sent to a new value
$x_j \mapsto (\zeta^{-k})^{k-i} x_j \mapsto \ldots$. It follows that every row (whether in-domain or out-of-domain) is
an invertible affine transformation of the vector $\{s_k((\zeta^{-1})^{k(k-i)}x_j)\}_{i=0}^k$, where the concrete
transformation depends on the quotient $q(X)$ and $\zeta$.

Unless the set of indeterminates contains a pair $(x', x'')$ such that $\zeta^{-k(k-i)} x' = x''$ for some
$i \in \{0, \ldots, k-1\}$, the $(t+1)(k+1)$ elements revealed by $(t+1)$ rows uniquely determine $(t+1)(k+1)$ points on
$s_k(X)$, for any fixed quotient $q(X)$ and any admissible choice of $\zeta$. As long as
$(t+1)(k+1) \leqslant \rho |D|$, $s_k(X)$ can be found by interpolation. It follows that under these conditions any set
of $t+1$ revealed rows is independent of the quotient.

This argument covers all in-domain rows and at most one out-of-domain row but not both out-of-domain rows. Indeed, the
indeterminates for the out-of-domain rows are $\alpha^k$ and $\zeta^{-k}\alpha^k$ and are apart by a factor
$\zeta^{-k}$, and therefore violate the above clause.

A closer inspection shows that the first coefficient of the second out-of-domain row, $s_0(\zeta^{-k} \alpha^k)$
substitutes to $\pm (\zeta^{-1})^{\sum_{\iota = 0}^k \iota} s_k(\zeta^{-k^2} \zeta^{-k} \alpha^{k}) + <$ *some terms
that depend on* $q(X) >$. That the indeterminate $\zeta^{-(k+1)k} \alpha^{k}$ is not contained in the set
$\{\zeta^{-(k-i)k}\alpha^k\}_{i=0}^k \cup \{\{\zeta^{-k(k-i)} x_j\}_i\}_{j=0}^{t-1}$ of indeterminates resulting
from the first out-of-domain row and all $t$ in-domain rows (unless for very unlikely choices of $\alpha$). As a result,
the argument from interpolating $s_k(X)$ from $(t+1)(k+1) + 1$ fixes points covers all in-domain rows, the entire first
out-of-domain row, and the first coefficient of the second out-of-domain row. The requirement is that
$(t+1)(k+1) + 1 \leqslant \rho |D|$.

To show that the remainder of the second out-of-domain row is *also* independent of the trace, consider the $k$-fold
segmentation equation applied to $r(X)$ and its segments $\{s_{i+1}(X)\}_{i=0}^{k-1}$:

$$ \left( r(\omega^i \alpha) \right)_{i=0}^{k-1} =
\left( \sum_{j=0}^{k-1} \omega^{ij} \zeta^{-j} \alpha^{j} s_{j+1}(\zeta^{-k} \alpha^k) \right)_{i=0}^{k-1} $$

where $\omega$ is a primitive $k$-th root of unity.
Ignoring the first element, we have a bijection between $\{s_{i+1}(\zeta^{-k}\alpha^k)\}_{i=0}^{k-1}$ and
$\{r(\omega^i \alpha)\}_{i=0}^{k-1}$. Likewise, from the first $k$ elements of the penultimate row one obtains
$\{p(\omega^i \alpha)\}_{i=0}^{k-1}$. Considering this information fixed (as it was already established to be
independent of the trace), it follows that $\{r(\omega^i \alpha)\}_{i=0}^{k-1}$ is bijectively equivalent to
$\{q(\omega^i \alpha)\}_{i=0}^{k-1}$. Therefore, in order to show that $\{s_{i+1}(\zeta^{-k}\alpha^k)\}_{i=0}^{k-1}$ is
independent of the trace, it suffices to show that $\{q(\omega^i \alpha)\}_{i=0}^{k-1}$ is independent of the trace.

Consider the distinguisher that receives, as a supplementary hint in addition to the transcript, the authentic preimages
under the AIR evaluation map to $\{q(\omega^i \alpha)\}_{i=0}^{k-1}$. These preimages are the $f$-tuples of
out-of-domain trace rows corresponding to $\{\omega^i \alpha\}_{i=0}^{k-1}$. Since we had $kef$ coefficients of margin
in our choice for $h$, it follows that even these $k$-many $f$-tuples of (degree-$e$ extension field) rows are
independent of the trace, and the same must be true for any image of them.

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
