# Copy-Constraints

Triton VM, possibly in addition to other VMs, uses several different tables. Each table has its own AIR. However, the tables a *not* independent. There are constraints that require that some pairs of tables have some identical rows. This note explains how to realize these copy-constraints at the Polynomial IOP level.

## Permutation Argument

Let $T$ and $S$ be single-column tables of $2^k$ rows. We describe a protocol that proves that $S$ and $T$ are permutations of one another. The key is to take the product of all elements, after offsetting them by some random $\beta$. Clearly the polynomials

$$ (X - T_{[0]}) \cdot (X - T_{[1]}) \cdots (X - T_{[2^k-1]}) $$

and

$$ (X - S_{[0]}) \cdot (X - S_{[1]}) \cdots (X - S_{[2^k-1]}) $$

are equal iff $\{T_{[i]}\}_i = \{S_{[i]}\}_i$. The offset by $\beta$ amounts to evaluating these polynomials in a random point. The subsequent scalar equality test constitutes a randomized polynomial identity test and by the Schwartz-Zippel lemma produces a false positive with probability at most $\frac{2^k}{|\mathbb{F}|}$.

To establish that the product is correctly computed, the Prover needs to supply running product tables, $T_ p$ and $S_ p$. These pairs of tables satisfy the following AIR:
 - $T_ {p,[0]} = T_ {[0]}$ and $S_ {p,[0]} = S_ {[0]}$
 - For all $i \in \{0, \ldots, 2^k-2\}$: $T_ {p,[i+1]} = T_ {p,[i]} \cdot T_ {[i+1]}$ and $S_ {p,[i+1]} = S_ {p,[i]} \cdot S_ {[i+1]}$
 - $T_ {p, [2^k-1]} = S_ {p, [2^k-1]}$

**Protocol Permutation**
 - Verifier is already in possession of polynomial oracles $[t(X)]$ and $[s(X)]$ that agree with $T$ and $S$ on a subgroup $H$ of order $2^k$.
 - Verifier samples $\beta \xleftarrow{\$} \mathbb{F}$ and sends it to Prover.
 - Prover computes running product polynomials, $t_ p(X)$ and $s_ p(X)$ of degree at most $|H|-1$ and sends them to Verifier.
 - Prover and Verifier run an AIR sub-protocol to establish that the AIR is satisfied for $(t(X), s(X), t_ p(X), s_ p(X))$.

 