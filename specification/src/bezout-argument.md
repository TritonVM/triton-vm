# Bézout Argument

The Bézout Argument establishes that some list $A = (a_0, \dots, a_n)$ does not contain any duplicate elements.
It uses [Bézout's identity for univariate polynomials](https://en.wikipedia.org/wiki/Polynomial_greatest_common_divisor#B%C3%A9zout's_identity_and_extended_GCD_algorithm).
Concretely, the prover shows that for the polynomial $f_A(X)$ with all elements of $A$ as roots, _i.e._,
$$
f_A(X) = \prod_{i=0}^n X - a_i
$$
and its formal derivative $f'_A(X)$, there exist $u(X)$ and $v(X)$ with appropriate degrees such that
$$
u(X) \cdot f_A(X) + v(X) \cdot f'_A(X) = 1.
$$
In other words, the Bézout Argument establishes that the greatest common divisor of $f_A(X)$ and $f'_A(X)$ is 1.
This implies that all roots of $f_A(X)$ have multiplicity 1, which holds if and only if there are no duplicate elements in list $A$.

For an example, a more in-depth explanation, the arithmetization, as well as proofs of completeness, soundness, and Zero-Knowledge, we refer to the later [section on continuity of the RAM pointer regions](contiguity-of-memory-pointer-regions.md#contiguity-for-ram-table).
