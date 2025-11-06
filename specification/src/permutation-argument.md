# Permutation Argument

The Permutation Argument establishes that two lists $A = (a_0, \dots, a_n)$ and $B = (b_0, \dots, b_n)$ are permutations of each other.
To achieve this, the lists' elements are interpreted as the roots of polynomials $f_A(X)$ and $f_B(X)$, respectively:

$$
\begin{aligned}
f_A(X) &= \prod_{i=0}^n X - a_i \\
f_B(X) &= \prod_{i=0}^n X - b_i
\end{aligned}
$$

The two lists $A$ and $B$ are a permutation of each other if and only if the two polynomials $f_A(X)$ and $f_B(X)$ are identical.
By the [Schwartz–Zippel lemma](https://en.wikipedia.org/wiki/Schwartz%E2%80%93Zippel_lemma), probing the polynomials in a single point $\alpha$ establishes the polynomial identity with high probability.[^1]

In Triton VM, the Permutation Argument is generally applied to show that the rows of one table appear in some other table without enforcing the rows' order in relation to each other.
To establish this, the prover

- commits to the main column in question,[^2]
- samples a random challenge $\alpha$ through the Fiat-Shamir heuristic,
- computes the _running product_ of $f_A(\alpha)$ and $f_B(\alpha)$ in the respective tables' auxiliary column.

For example, in Table A:

| main column A | auxiliary column A: running product                |
|--------------:|:---------------------------------------------------|
|             0 | $(\alpha - 0)$                                     |
|             1 | $(\alpha - 0)(\alpha - 1)$                         |
|             2 | $(\alpha - 0)(\alpha - 1)(\alpha - 2)$             |
|             3 | $(\alpha - 0)(\alpha - 1)(\alpha - 2)(\alpha - 3)$ |

And in Table B:

| main column B | auxiliary column B: running product                |
|--------------:|:---------------------------------------------------|
|             2 | $(\alpha - 2)$                                     |
|             1 | $(\alpha - 2)(\alpha - 1)$                         |
|             3 | $(\alpha - 2)(\alpha - 1)(\alpha - 3)$             |
|             0 | $(\alpha - 2)(\alpha - 1)(\alpha - 3)(\alpha - 0)$ |

It is possible to establish a subset relation by skipping over certain elements when computing the running product.
The running product must incorporate the same elements in both tables.
Otherwise, the Permutation Argument will fail.

An example of a subset Permutation Argument can be found between the [U32 Table](u32-table.md#auxiliary-columns) and the [Processor Table](processor-table.md#auxiliary-columns).

[^1]: This depends on the length $n$ of the lists $A$ and $B$ as well as the field size.
For Triton VM, $n < 2^{32}$.
The polynomials $f_A(X)$ and $f_B(X)$ are evaluated over the auxiliary field with $p^3 \approx 2^{192}$ elements.
The false positive rate is therefore $n / |\mathbb{F}_{p^3}| \leqslant 2^{-160}$.

[^2]: See “[Compressing Multiple Elements](table-linking.md#compressing-multiple-elements).”
