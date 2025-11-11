# Lookup Argument

The [Lookup Argument](https://eprint.iacr.org/2023/107.pdf) establishes that all elements of list $A = (a_0, \dots, a_\ell)$ also occur in list $B = (b_0, \dots, b_n)$.
In this context, $A$ contains the values that are being looked up, while $B$ is the lookup table.[^1]
Both lists $A$ and $B$ may contain duplicates.
However, it is inefficient if $B$ does, and is therefore assumed not to.

The example at the end of this section summarizes the necessary computations for the Lookup Argument.
The rest of the section derives those computations.
The first step is to interpret both lists' elements as the roots of polynomials $f_A(X)$ and $f_B(X)$, respectively:

$$
\begin{aligned}
f_A(X) &= \prod_{i=0}^\ell X - a_i \\
f_B(X) &= \prod_{i=0}^n X - b_i
\end{aligned}
$$

By counting the number of occurrences $m_i$ of unique elements $a_i \in A$ and eliminating duplicates, $f_A(X)$ can equivalently be expressed as:

$$
f_A(X) = \prod_{i=0}^n (X - a_i)^{m_i}
$$


The next step uses
- the formal derivative $f'_A(X)$ of $f_A(X)$, and
- the multiplicity-weighted formal derivative $f'^{(m)}_B(X)$ of $f_B(X)$.

The former is the familiar formal derivative:

$$
f'_A(X) = \sum_{i=0}^n m_i (X - a_i)^{m_i - 1} \prod_{j \neq i}(X - a_j)^{m_j}
$$

The multiplicity-weighted formal derivative uses the lookup-multiplicities $m_i$ as additional factors:

$$
f'^{(m)}_B(X) = \sum_{i=0}^n m_i \prod_{j \neq i}(X - b_j)
$$

Let $g(X)$ the greatest common divisor of $f_A(X)$ and $f'_A(X)$.
The polynomial $f_A(X) / g(X)$ has the same roots as $f_A(X)$, but all roots have multiplicity 1.
This polynomial is identical to $f_B(X)$ if and only if all elements in list $A$ also occur in list $B$.

A similar property holds for the formal derivatives:
The polynomial $f'_A(x) / g(X)$ is identical to $f'^{(m)}_B(X)$ if and only if all elements in list $A$ also occur in list $B$.
By solving for $g(X)$ and equating, the following holds:

$$
f_A(X) \cdot f'^{(m)}_B(X) = f'_A(X) \cdot f_B(X)
$$

## Optimization through Logarithmic Derivatives

The [logarithmic derivative](https://eprint.iacr.org/2022/1530.pdf) of $f(X)$ is defined as $f'(X) / f(X)$.
Furthermore, the equality of [monic polynomials](https://en.wikipedia.org/wiki/Monic_polynomial) $f(X)$ and $g(X)$ is equivalent to the equality of their logarithmic derivatives.[^2]
This allows rewriting above equation as:

$$
\frac{f'_A(X)}{f_A(X)} = \frac{f'^{(m)}_B(X)}{f_B(X)}
$$

Using logarithmic derivatives for the equality check decreases the computational effort for both prover and verifier.
Concretely, instead of four running products – one each for $f_A$, $f'_A$, $f_B$, and $f'^{(m)}_B$ – only two logarithmic derivatives are needed.
Namely, considering once again list $A$ _containing_ duplicates, above equation can be written as:

$$
\sum_{i=0}^\ell \frac{1}{X - a_i} = \sum_{i=0}^n \frac{m_i}{X - b_i}
$$

To compute the sums, the lists $A$ and $B$ are main columns in the two respective tables.
Additionally, the lookup multiplicity is recorded explicitly in a main column of the lookup table.

## Example

In Table A:

| main column A | auxiliary column A: logarithmic derivative                           |
|--------------:|:---------------------------------------------------------------------|
|             0 | $\frac{1}{\alpha - 0}$                                               |
|             2 | $\frac{1}{\alpha - 0} + \frac{1}{\alpha - 2}$                        |
|             2 | $\frac{1}{\alpha - 0} + \frac{2}{\alpha - 2}$                        |
|             1 | $\frac{1}{\alpha - 0} + \frac{2}{\alpha - 2} + \frac{1}{\alpha - 1}$ |
|             2 | $\frac{1}{\alpha - 0} + \frac{3}{\alpha - 2} + \frac{1}{\alpha - 1}$ |

And in Table B:

| main column B | multiplicity | auxiliary column B: logarithmic derivative                           |
|--------------:|-------------:|:---------------------------------------------------------------------|
|             0 |            1 | $\frac{1}{\alpha - 0}$                                               |
|             1 |            1 | $\frac{1}{\alpha - 0} + \frac{1}{\alpha - 1}$                        |
|             2 |            3 | $\frac{1}{\alpha - 0} + \frac{1}{\alpha - 1} + \frac{3}{\alpha - 2}$ |

It is possible to establish a subset relation by skipping over certain elements when computing the logarithmic derivative.
The logarithmic derivative must incorporate the same elements with the same multiplicity in both tables.
Otherwise, the Lookup Argument will fail.

An example for a Lookup Argument can be found between the [Program Table](program-table.md) and the [Processor Table](processor-table.md#auxiliary-columns).

[^1]: The lookup table may represent a mapping from one or more “input” elements to one or more “output” elements – see “[Compressing Multiple Elements](table-linking.md#compressing-multiple-elements).”

[^2]: See Lemma 3 in [this paper](https://eprint.iacr.org/2022/1530.pdf) for a proof.
