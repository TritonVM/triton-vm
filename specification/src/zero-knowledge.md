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

For simplicity, we choose to have only one $h$ that works for both main and auxiliary traces. Furthermore, we anticipate the need for a margin of $(k-1)ef$ coefficients originating from the randomized quotient table, and 1 coefficient for the Merkle trees. So: $h = t + kef + 1$.

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

Triton VM uses Merkle trees without salts (or any other form of hiding commitment scheme) but otherwise the same transformation. However, the BCS argument that zero-knowledge is retained, fails when this change is applied. So below we articulate a new one. This argument is specific to the context of STARKs and may fail in a more general IOP context.

The objects that are introduced by the BCS transform and that are *a priori* capable of leaking information are the Merkle authentication paths and Merkle roots. Without salts, these objects are images of data that should remain hidden.

Distinguish on the one hand Merkle trees built from the batch codeword, or from codewords downstream from it; from on the other hand, Merkle trees built from low-degree-extended trace or quotient tables. Since the batch contains a batch randomizer, the batch codeword is a uniformly random Reed-Solomon codeword and perfectly independent of the trace. Therefore, even if the complete list of leaf-preimages to this Merkle root or of any Merkle root of a codeword downstream from it were released, that release would leak no information about the trace. It follows that the Merkle roots and authentication paths (or authentication structures, in case they are compressed) corresponding to the batch codeword or codewords downstream from it are independent of the trace.

What remains is the Merkle trees built out of the low-degree-extended trace and quotient tables, with the important feature that both are randomized using the respective strategies described above. For these Merkle trees we quantitatively bound the the probability $P$ that information is leaked and show that this probability is negligible and concretely irrelevant.

Consider instead of the internal Merkle nodes and Merkle roots, the complete list of leafs, whose preimages are rows in the low-degree-extended trace and quotient tables. Clearly the Merkle authentication paths and roots can be computed from this information, so it suffices to bound the amount of information leaked by these leafs instead.

Consider the event in which the distinguisher recognizes one of these leaf digests because the digest appears as the image field of a list $L$ of preimage-and-image pairs produced by invoking the random oracle $Q$ times. (These invocations may even happen *after* the distinguisher receives the transcript.) If this event happens then all bets about concealing the trace are off because the distinguisher can infer the value from one more row than accounted for. If this event does not happen, then all observed leaf digests are new and therefore leak no information about their corresponding preimages.

Consider first the low-degree-extended (and randomized) main trace. In order for the distinguisher to sample a preimage-and-image pair and row index whose image is observed as the leaf digest for that row, the distinguisher must make a correct guess at the entire set of main trace randomizers. (We consider the trace fixed.) The remaining 1 coefficient of margin on the number $h$ of randomizers *per column* means that there are $\mathsf{m}$ (number of columns) variables left to guess. Any one preimage-image pair from the list $L$ corresponds to $|D|$-many triples consisting of one preimage, one image, and one row index. So for every hash invocation, the distinguisher can test $|D|$ distinct guesses for the randomizers. The distinguisher is successful if he samples the correct set of main trace randomizers in any one of these $|D| \times Q$ guesses. The search space is $\mathbb{F}^\mathsf{m}$ so the probability of the distinguisher gleaning information from the main trace is no greater than $\frac{|D| Q}{|\mathbb{F}|^\mathsf{m}}$.

An analogous argument holds for the auxiliary trace, where the number of columns is $\mathsf{w}-\mathsf{m}$ and the columns contain extension field elements. So the probability that the distinguisher gleans information from the auxiliary table is no greater than $\frac{|D| Q}{|\mathbb{F}|^{e(\mathsf{w}-\mathsf{m})}}$.

For the quotient table, and considering the randomizers used for quotient table randomization fixed, any given row corresponds to $k$ quotient segments $\{q_i(X)\}_{i=0}^{k-1}$ and, via the segmentation equation, to $k$ values of the quotient $\{q(\omega^i X)\}_{i=0}^{k-1}$. That quotient value itself corresponds to $M$-many potential values of the $\mathsf{w}$-vector of randomized trace polynomials as preimages, where $M$ is at least $|\mathcal{C}|$, the number of constraints. The use of the term *preimage* here denotes not the inverse image of hash function evaluation but evaluation of the map that composes a) evaluation of the AIR constraints; b) division of the zerofiers; and c) taking the weighted linear sum.

If the AIR circuit is not injective, then $M > |\mathcal{C}|$ and we bound $M$ as follows. Let $\mathsf{d}$ be the degree of the AIR. Then there are ${f\cdot\mathsf{w} + \mathsf{d} \choose \mathsf{d}}$ monomials of degree $\mathsf{d}$ or less in $f \cdot \mathsf{w}$ variables which correspond to a $f$-tuple of in-or-out-of-domain rows. To see this, consider that a there are $\# \star + \, \# |$ slots on a line to fill with $\# \star$ stars and $\# |$ bars and the resulting arrangement corresponds bijectively to a monomial of exactly degree $\# \star$ in $\# | + 1$ variables. Interpret the one extra variable as an imaginary one used to homogenize all monomials of non-max degree; then there are ${\#\star + \, \#| \choose \# \star}$ monomials of degree at most $\# \star$ in $\# |$ variables.

Any value of the quotient $q(X)$ is a linear combination of these monomials. It follows that the number $M$ of corresponding $f$-tuples of $\mathsf{w}$-vectors of randomized trace values is bounded by ${f\cdot\mathsf{w} + \mathsf{d} \choose \mathsf{d}} - 1$.

Therefore, whenever the distinguisher tests a candidate quotient row and finds that its hash is different from any leaf digest in the transcript, he can discount at most $k \cdot M \cdot |D|$ candidate $\mathsf{w}$-vectors of trace randomizers as viable candidates. The total number of discountable candidates is at most $k \cdot ({f\cdot\mathsf{w} + \mathsf{d} \choose \mathsf{d}} - 1) \cdot |D| \cdot Q$, which sounds large until you realize that the search space is $\mathbb{F}^{\mathsf{m} + e(\mathsf{w}-\mathsf{m})}$. The probability of the distinguisher gleaning information from the quotient table is therefore bounded by $\frac{k ({f\cdot\mathsf{w} + \mathsf{d} \choose \mathsf{d}} - 1) |D| Q}{|\mathbb{F}|^{\mathsf{m} + e(\mathsf{w}-\mathsf{m})}}$.

In conclusion, the probability that the *computationally bounded* distinguisher, who limits himself to $Q$ invocations of the random oracle, finds in the transcript one of the responses to his random oracle queries, is bounded by

$$ P \leq |D| \cdot  Q \cdot \left(\frac{1}{|\mathbb{F}|^\mathsf{m}} + \frac{1}{|\mathbb{F}|^{e(\mathsf{w}-\mathsf{m})}} + \frac{k \cdot ({f\cdot\mathsf{w} + \mathsf{d} \choose \mathsf{d}} - 1)}{|\mathbb{F}|^{\mathsf{m} + e(\mathsf{w}-\mathsf{m})}} \right).$$

### Zero-Knowledge IP to Zero-Knowledge NIP

Applying the Fiat-Shamir transform does not affect the distribution of the transcript. It does, however, affect the simulator who needs to know the value of the challenge $\alpha$ before producing the commitments from which it is pseudorandomly derived. To achieve this, the simulator is defined relative to the *programmable random oracle model*, which enables him to specify (a sparse list of) query-response pairs that the random oracle must abide by.