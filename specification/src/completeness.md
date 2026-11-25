# Completeness

A proof system has *completeness error $\varepsilon_\mathsf{C}$* if, for every true claim and every witness for it, the honest prover, given the claim and the witness, produces a proof that the honest verifier accepts, except with probability at most $\varepsilon_\mathsf{C}$. The probability is over the prover's randomness and the verifier's challenges. In the non-interactive proof, the challenges are outputs of the Tip5 sponge, which this page models as a random oracle, so they are uniformly random, as in the interactive protocol.

The claim and the witness are as defined in the chapter on [Soundness](soundness.md). This page considers only true claims whose execution has padded height $N$ with $\log_2 N \leqslant 28$, because the verifier rejects larger $N$. The honest prover *fails* if it returns an error, if it panics, or if the verifier rejects or panics on its proof. This page lists every way in which the honest prover can fail, bounds the probability of each, and sums the bounds. The result is $\varepsilon_\mathsf{C} < 2^{-155}$, far below $2^{-80}$.

This page uses the notation of the chapter on [Soundness](soundness.md): $\mathbb{E}$ is the extension field with $\vert \mathbb{E} \vert = p^3 \approx 2^{192}$, $H$ the trace domain, $D$ the domain of the committed words, $\alpha$ the out-of-domain point, $k = 4$ the number of quotient segments, $\zeta$ the constant of the randomized segments, and $D_R$ the domain of the last FRI codeword, with $\vert D_R \vert = 512$. For the default parameters, $\vert D \vert = 8 \vert H \vert$, and $\vert H \vert = N$ for every $N \geqslant 2^9$. For smaller $N$, $\vert H \vert = 2^9 > N$. Every table of the honest prover has at most $N \leqslant \vert H \vert$ rows.

## Failures That Do Not Depend on Randomness

The following failures cannot occur for a true claim with a witness and $\log_2 N \leqslant 28$, whatever the randomness.

- The prover returns `ProgramDigestMismatch`, `PublicOutputMismatch`, or an error of the VM only if the witness does not match the claim.
- The prover returns `PaddedHeightTooLarge` only if $\log_2 N > 28$.
- Every other check of the verifier, listed in [Verifier Checks](soundness.md#verifier-checks), passes on an honest proof for every value of the challenges.

## Failures That Depend on Randomness

Every remaining failure is a division by zero at a value that depends on a uniformly random challenge in $\mathbb{E}$, except the zero-knowledge conflict below, which is an equality. A non-zero polynomial of degree $d$ vanishes at a uniformly random point with probability at most $d / \vert \mathbb{E} \vert$; this is the Schwartz–Zippel lemma, as in the chapter on [Soundness](soundness.md#round-1-table-linking-challenges).

**Lookup denominators.** Each [lookup argument](lookup-argument.md) adds a fraction $m / (x - e)$ for each entry $e$ on the server side and $1 / (x - e)$ for each entry on the client side, where $x$ is the argument's challenge. The honest prover computes these fractions while building the auxiliary table, and the inversion panics if some $x - e$ is zero. The entry $e$ is a linear combination of main-table cells with challenges other than $x$ as coefficients, and the main table is fixed before the challenges are sampled. So $x - e$ is a polynomial of degree 1 in the challenges, and it is non-zero because of the term $x$. It vanishes with probability at most $1 / \vert \mathbb{E} \vert$. The number of denominators per row, client plus server, is that of the table in [Round 1](soundness.md#round-1-table-linking-challenges):

| Lookup argument | Denominators per row |
| :--- | ---: |
| Instruction lookup | $1 + 1$ |
| Hash table to cascade table | $16 + 1$ |
| Cascade table to lookup table | $2 + 1$ |
| Processor to U32 table | $2 + 1$ |
| Clock jump differences | $3 + 1$ |

In total, there are at most $29 N \leqslant 29 \vert H \vert$ denominators. A union bound gives

$$ \varepsilon_\mathsf{lookup} := \frac{29 \vert H \vert}{\vert \mathbb{E} \vert} . $$

Permutation and evaluation arguments do not divide, so they cannot fail this way.

**Out-of-domain point in the trace domain.** If $\alpha \in H$, both the prover and the verifier divide by zero. The verifier inverts the zerofiers $\alpha - 1$, $\alpha^{\vert H \vert} - 1$, and $\alpha - g^{-1}$, where $g$ generates $H$. The prover computes the out-of-domain rows by barycentric evaluation on $H$ at $\alpha$ and $g\alpha$, which divides by $\alpha - h$ and $g\alpha - h$ for every $h \in H$. Each of these is zero only if $\alpha \in H$, which has probability

$$ \varepsilon_H := \frac{\vert H \vert}{\vert \mathbb{E} \vert} . $$

**Out-of-domain point in the evaluation domain.** The DEEP quotients divide by $x - \alpha$, $x - g\alpha$, $x - \alpha^k$, and $x - (\zeta\alpha)^k$. The prover divides for every point $x$ of the shorter of $D$ and the quotient domain. These two domains are cosets with the same offset whose orders are powers of 2, so the shorter one is contained in $D$. The verifier, in `Stark::deep_update`, divides for every queried point $x \in D$. Both fail only if one of the four points lies in $D$. The conditions $\alpha \in D$ and $g\alpha \in D$ hold for $\vert D \vert$ values of $\alpha$ each. Each of the conditions $\alpha^k \in D$ and $(\zeta\alpha)^k \in D$ holds for at most $k \vert D \vert$ values, because each point of $D$ has at most $k$ $k$-th roots in $\mathbb{E}$. So

$$ \varepsilon_D := \frac{(2k + 2) \vert D \vert}{\vert \mathbb{E} \vert} . $$

**Zero-knowledge conflict.** After sampling the query indices, the prover returns `ZeroKnowledgeViolation` if revealing the queried rows of the randomized quotient table would leak information; the function `Prover::zero_knowledge_is_violated` performs the check. This happens exactly if $\alpha^k = \zeta^{\Delta k} x$ for some queried point $x \in D$ and some integer $-k \leqslant \Delta < k$. The right-hand side takes at most $2k \vert D \vert$ values, and for each, at most $k$ values of $\alpha$ satisfy the equation. So

$$ \varepsilon_\mathsf{ZK} := \frac{2k^2 \vert D \vert}{\vert \mathbb{E} \vert} . $$

**Last FRI round.** The verifier evaluates the interpolant of the last codeword at a random point $\beta \in \mathbb{E}$ by barycentric evaluation. The formula treats the codeword as if it lay on the subgroup of order $\vert D_R \vert$, of which $D_R$ is a coset, and divides by $\beta - d$ for each element $d$ of that subgroup. Such a division by zero happens with probability

$$ \varepsilon_\mathsf{last} := \frac{\vert D_R \vert}{\vert \mathbb{E} \vert} . $$

No other step of the prover or the verifier divides by a value that depends on a challenge, or compares such a value with a fixed one. In particular, the prover inverts the sum $\sum_{h \in H} h / (\alpha - h) = \vert H \vert / (\alpha^{\vert H \vert} - 1)$, which is non-zero for $\alpha \notin H$, and likewise at $g\alpha$; the FRI folds divide by $2x$ for $x \in D_r$, which is non-zero; and the query indices are drawn by rejection sampling, which terminates with probability 1.

## The Completeness Error

A union bound over the five events gives

$$
\varepsilon_\mathsf{C} \leqslant \varepsilon_\mathsf{lookup} + \varepsilon_H + \varepsilon_D + \varepsilon_\mathsf{ZK} + \varepsilon_\mathsf{last}
= \frac{29 \vert H \vert + \vert H \vert + (2k+2) \vert D \vert + 2k^2 \vert D \vert + \vert D_R \vert}{\vert \mathbb{E} \vert} .
$$

With $k = 4$ and $\vert D \vert = 8 \vert H \vert$, the numerator is $366 \vert H \vert + 512$. All errors are given as $\log_2$:

| Quantity | $N = 2^{28}$ | $N = 2^{21}$ |
| :--- | ---: | ---: |
| $\varepsilon_\mathsf{lookup}$ | -159.1 | -166.1 |
| $\varepsilon_H$ | -164.0 | -171.0 |
| $\varepsilon_D$ | -157.7 | -164.7 |
| $\varepsilon_\mathsf{ZK}$ | -156.0 | -163.0 |
| $\varepsilon_\mathsf{last}$ | -183.0 | -183.0 |
| $\varepsilon_\mathsf{C}$ | -155.5 | -162.5 |

Every term except $\varepsilon_\mathsf{last}$ grows with $N$, and $\varepsilon_\mathsf{last}$ is constant, so $N = 2^{28}$ is the worst case. There, $\varepsilon_\mathsf{C} \approx 2^{-155.5} < 2^{-80}$.

> **Note:** The honest prover does not retry. After a failure, a fresh run with new randomness succeeds with the same probability $1 - \varepsilon_\mathsf{C}$. Only the zero-knowledge conflict is an error that the prover reports deliberately; the other failures are panics of the prover or the verifier.
