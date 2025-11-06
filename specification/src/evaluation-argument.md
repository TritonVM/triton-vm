# Evaluation Argument

The Evaluation Argument establishes that two lists $A = (a_0, \dots, a_n)$ and $B = (b_0, \dots, b_n)$ are identical.
To achieve this, the lists' elements are interpreted as the coefficients of polynomials $f_A(X)$ and $f_B(X)$, respectively:

$$
\begin{aligned}
f_A(X) &= X^{n+1} + \sum_{i=0}^n a_{n-i}X^i \\
f_B(X) &= X^{n+1} + \sum_{i=0}^n b_{n-i}X^i \\
\end{aligned}
$$

The two lists $A$ and $B$ are identical if and only if the two polynomials $f_A(X)$ and $f_B(X)$ are identical.
By the [Schwartz–Zippel lemma](https://en.wikipedia.org/wiki/Schwartz%E2%80%93Zippel_lemma), probing the polynomials in a single point $\alpha$ establishes the polynomial identity with high probability.[^1]

In Triton VM, the Evaluation Argument is generally used to show that (parts of) some row appear in two tables in the same order.
To establish this, the prover

- commits to the main column in question,[^2]
- samples a random challenge $\alpha$ through the Fiat-Shamir heuristic,
- computes the _running evaluation_ of $f_A(\alpha)$ and $f_B(\alpha)$ in the respective tables' auxiliary column.

For example, in both Table A and B:

| main column | auxiliary column: running evaluation                       |
|------------:|:-----------------------------------------------------------|
|           0 | $\alpha^1 + 0\alpha^0$                                     |
|           1 | $\alpha^2 + 0\alpha^1 + 1\alpha^0$                         |
|           2 | $\alpha^3 + 0\alpha^2 + 1\alpha^1 + 2\alpha^0$             |
|           3 | $\alpha^4 + 0\alpha^3 + 1\alpha^2 + 2\alpha^1 + 3\alpha^0$ |

It is possible to establish a subset relation by skipping over certain elements when computing the running evaluation.
The running evaluation must incorporate the same elements in both tables.
Otherwise, the Evaluation Argument will fail.

Examples for subset Evaluation Arguments can be found between the [Hash Table](hash-table.md#auxiliary-columns) and the [Processor Table](processor-table.md#auxiliary-columns).

[^1]: This depends on the length $n$ of the lists $A$ and $B$ as well as the field size.
For Triton VM, $n < 2^{32}$.
The polynomials $f_A(X)$ and $f_B(X)$ are evaluated over the auxiliary field with $p^3 \approx 2^{192}$ elements.
The false positive rate is therefore $n / |\mathbb{F}_{p^3}| \leqslant 2^{-160}$.

[^2]: See “[Compressing Multiple Elements](table-linking.md#compressing-multiple-elements).”
