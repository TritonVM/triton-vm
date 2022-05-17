# Inverse Sumcheck

A [sum-check](sum-check.md) argument establishes that a given polynomials integrates to a given scalar $s \in \mathbb{F}$ on a subgroup $H \subset \mathbb{F} \backslash \{0\}$. An *inverse sum-check* argument establishes that the sum of inverses of values taken by a given polynomial on a subgroup $H$ sums to $S \in \mathbb{F}$. Let $f(X)$ be the polynomial in question, and ignore cases where $f(X) = 0$. Then the arguments establish that

$$ s = \sum_{h \in H} f(h) $$

and

$$ S = \sum_{h \in H} \frac{1}{f(h)}  \enspace .$$

Recall that the sum-check protocol is a Polynomial IOP whereby the Verifier already possesses the oracle $[f(X)]$. The prover sends two polynomials of bounded degree, $g^*(X)$ and $h(X)$. The verifier then samples $[f(X)]$, $[g^*(X)]$ and $[h(X)]$ in a random point $z \xleftarrow{\$} \mathbb{F}$ to test the polynomial identity

$$ f(X) - \frac{s}{|H|} = X \cdot g^*(X) + Z_H(X) \cdot h(X) \enspace ,$$

where $Z_H(X)$ is the zerofier for $H$.

The inverse sum-check protocol uses the sum-check as a subprotocol.
 1. Prover computes $f^\star(X)$, which is the unique polynomial of degree $|H|-1$ that agrees with $\frac{1}{f(X)}$ on $H$. Prover sends $f^\star(X)$ to Verifier.
 2. Prover and Verifier execute a subprotocol to establish that $f(X) \cdot f^\star(X) = 1$ on $H$.
 3. Prover and Verifier execute a sum-check subprotocol to establish that $f^\star(X)$ integrates to $S$ on $H$.

Let's compile this using FRI. The Prover has already committed to $f(X)$. The question is which polynomials the Verifier can simulate and which polynomials the Prover must additionally commit to.

Observe that the agreement on $H$, $f(X) \cdot f^\star(X) = 1$ is not a polynomial identity (because there is no guarantee for $X \not \in H$). However, given access to $[f(X)]$ and $[f^\star(X)]$, the Verifier can simulate the quotient $q(X) = \frac{f(X) \cdot f^\star(X) - 1}{Z_H(X)}$, whose bounded degree establishes the agreement.

One concludes that the Prover must additionally commit to $f^\star(X)$. In a FRI-based proof, this necessitates a second stage. In order to eliminate the second stage, we need to find an expression in terms of $f(X)$ and *only* other easy-to-evaluate polynomials, such that this expression has a low degree if and only if the agreement on $H$ holds. It is not clear that algebraic manipulations can provide this expression.

