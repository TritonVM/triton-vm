# Contiguity of Memory-Pointer Regions

## Contiguity for Op Stack Table

In each cycle, the memory pointer for the Op Stack Table, `op_stack_pointer`, can only ever increase by one, remain the same, or decrease by one.
As a result, it is easy to enforce that the entire table is sorted for memory pointer using one initial constraint and one transition constraint.

 - Initial constraint: `op_stack_pointer` starts with 16[^op_stack], so in terms of polynomials the constraint is `op_stack_pointer - 16`.
 - Transition constraint: the new `op_stack_pointer` is either the same as the previous or one larger.
    The polynomial representation for this constraint is `(op_stack_pointer' - op_stack_pointer - 1) · (op_stack_pointer' - op_stack_pointer)`.

## Contiguity for Jump Stack Table

Analogously to the Op Stack Table, the Jump Stack's memory pointer `jsp` can only ever decrease by one, remain the same, or increase by one, within each cycle. As a result, similar constraints establish that the entire table is sorted for memory pointer.

 - Initial constraint: `jsp` starts with zero, so in terms of polynomials the constraint is `jsp`.
 - Transition constraint: the new `jsp` is either the same as the previous or one larger. The polynomial representation for this constraint is `(jsp' - jsp - 1) * (jsp' - jsp)`.

## Contiguity for RAM Table

The *Contiguity Argument* for the RAM table establishes that all RAM pointer regions start with distinct values.
It is easy to ignore _consecutive_ duplicates in the list of all RAM pointers using one additional main column.
This allows identification of the RAM pointer values at the regions' boundaries, $A$.
The Contiguity Argument then shows that the list $A$ contains no duplicates.
For this, it uses [Bézout's identity for univariate polynomials](https://en.wikipedia.org/wiki/Polynomial_greatest_common_divisor#B%C3%A9zout's_identity_and_extended_GCD_algorithm).
Concretely, the prover shows that for the polynomial $f_A(X)$ with all elements of $A$ as roots, _i.e._,
$$
f_A(X) = \prod_{i=0}^n X - a_i
$$
and its formal derivative $f'_A(X)$, there exist $u(X)$ and $v(X)$ with appropriate degrees such that
$$
u(X) \cdot f_A(X) + v(X) \cdot f'_A(X) = 1.
$$
In other words, the Contiguity Argument establishes that the greatest common divisor of $f_A(X)$ and $f'_A(X)$ is 1.
This implies that all roots of $f_A(X)$ have multiplicity 1, which holds if and only if there are no duplicate elements in list $A$.

The following columns and constraints are needed for the Contiguity Argument:

 - Main column `iord` and two deterministic transition constraints enable conditioning on a changed memory pointer.
 - Main columns `bcpc0` and `bcpc1` and two deterministic transition constraints contain and constrain the symbolic Bézout coefficient polynomials' coefficients.
 - Auxiliary column `rpp` is a running product similar to that of a conditioned [permutation argument](permutation-argument.md). A randomized transition constraint verifies the correct accumulation of factors for updating this column.
 - Auxiliary column `fd` is the formal derivative of `rpp`. A randomized transition constraint verifies the correct application of the product rule of differentiation to update this column.
 - Auxiliary columns `bc0` and `bc1` build up the Bézout coefficient polynomials based on the corresponding main columns, `bcpc0` and `bcpc1`.
Two randomized transition constraints enforce the correct build-up of the Bézout coefficient polynomials.
 - A terminal constraint takes the weighted sum of the running product and the formal derivative, where the weights are the Bézout coefficient polynomials, and equates it to one. This equation asserts the Bézout relation.

The following table illustrates the idea.
Columns not needed for establishing memory consistency are not displayed.

| `ramp` | `iord`       | `bcpc0` | `bcpc1` | `rpp`                   | `fd`   | `bc0`    | `bc1`               |
|:-------|:-------------|:--------|:--------|:------------------------|:-------|:---------|:--------------------|
| $a$    | 0            | $0$     | $\ell$  | $(X - a)$               | $1$    | $0$      | $\ell$              |
| $a$    | $(b-a)^{-1}$ | $0$     | $\ell$  | $(X - a)$               | $1$    | $0$      | $\ell$              |
| $b$    | 0            | $j$     | $m$     | $(X - a)(X - b)$        | $p(X)$ | $j$      | $\ell X + m$        |
| $b$    | 0            | $j$     | $m$     | $(X - a)(X - b)$        | $p(X)$ | $j$      | $\ell X + m$        |
| $b$    | $(c-b)^{-1}$ | $j$     | $m$     | $(X - a)(X - b)$        | $p(X)$ | $j$      | $\ell X + m$        |
| $c$    | 0            | $k$     | $n$     | $(X - a)(X - b)(X - c)$ | $q(X)$ | $jX + k$ | $\ell X^2 + mX + n$ |
| $c$    | -            | $k$     | $n$     | $(X - a)(X - b)(X - c)$ | $q(X)$ | $jX + k$ | $\ell X^2 + mX + n$ |

The values contained in the auxiliary columns are undetermined until the verifier's challenge $\alpha$ is known; before that happens it is worthwhile to present the polynomial expressions in $X$, anticipating the substitution $X \mapsto \alpha$. The constraints are articulated relative to `α`.

The inverse of RAMP difference `iord` takes the inverse of the difference between the current and next `ramp` values if that difference is non-zero, and zero else. This constraint corresponds to two transition constraint polynomials:

 - `(ramp' - ramp) ⋅ ((ramp' - ramp) ⋅ iord - 1)`
 - `iord ⋅ (ramp' - ramp) ⋅ iord - 1)`

The running product `rpp` starts with $X - \mathsf{ramp}$ initially, which is enforced by an initial constraint.
It accumulates a factor $X - \mathsf{ramp}'$ in every pair of rows where `ramp ≠ ramp'`. This evolution corresponds to one transition constraint: `(ramp' - ramp) ⋅ (rpp' - rpp ⋅ (α - ramp')) + (1 - (ramp' - ramp) ⋅ di) ⋅ (rpp' - rp)`

Denote by $f_{\mathsf{rp}}(X)$ the polynomial that accumulates all factors $X - \mathsf{ramp}'$ in every pair of rows where $\mathsf{ramp} \neq \mathsf{ramp}'$.

The column `fd` contains the “formal derivative” of the running product with respect to $X$.
The formal derivative is initially $1$, which is enforced by an initial constraint.
The transition constraint applies the product rule of differentiation conditioned upon the difference in `ramp` being nonzero; in other words, if $\mathsf{ramp} = \mathsf{ramp}'$ then the same value persists; but if $\mathsf{ramp} \neq \mathsf{ramp}'$ then $\mathsf{fd}$ is mapped as
$$\mathsf{fd} \mapsto \mathsf{fd}' = (X - \mathsf{ramp}') \cdot \mathsf{fd} + \mathsf{rp} \enspace .$$
This update rule is called the *product rule of differentiation* because, assuming $\mathsf{ramp}' \neq \mathsf{ramp}$, then
$$\begin{aligned}
\frac{\mathsf{d}  \mathsf{rp}'}{\mathsf{d}   X} &= \frac{\mathsf{d}  (X - \mathsf{ramp}') \cdot \mathsf{rp}}{\mathsf{d}   X} \\
&= (X - \mathsf{ramp}') \cdot \frac{\mathsf{d}   \mathsf{rp}}{\mathsf{d}   X} + \frac{\mathsf{d}  ( X - \mathsf{ramp}')}{\mathsf{d}   X} \cdot \mathsf{rp} \\
&= (X - \mathsf{ramp}') \cdot \mathsf{fd} +\mathsf{rp} \enspace .
\end{aligned}$$

The transition constraint for `fd` is `(ramp' - ramp) ⋅ (fd' - rp - (α - ramp') ⋅ fd) + (1 - (ramp' - ramp) ⋅ iord) ⋅ (fd' - fd)`.

The polynomials $f_\mathsf{bc0}(X)$ and $f_\mathsf{bc1}(X)$ are the Bézout coefficient polynomials satisfying the relation
$$f_\mathsf{bc0}(X) \cdot f_{\mathsf{rp}}(X) + f_\mathsf{bc1}(X) \cdot f_{\mathsf{fd}}(X) = \gcd(f_{\mathsf{rp}}(X), f_{\mathsf{fd}}(X)) \stackrel{!}{=} 1 \enspace .$$
The prover finds $f_\mathsf{bc0}(X)$ and $f_\mathsf{bc1}(X)$ as the minimal-degree Bézout coefficients as returned by the extended Euclidean algorithm.
Concretely, the degree of $f_\mathsf{bc0}(X)$ is smaller than the degree of $f_\mathsf{fd}(X)$, and the degree of $f_\mathsf{bc1}(X)$ is smaller than the degree of $f_\mathsf{rp}(X)$.

The (scalar) coefficients of the Bézout coefficient polynomials are recorded in main columns `bcpc0` and `bcpc1`, respectively.
The transition constraints for these columns enforce that the value in one such column can only change if the memory pointer `ramp` changes.
However, unlike the conditional update rule enforced by the transition constraints of `rp` and `fd`, the new value is unconstrained.
Concretely, the two transition constraints are:

 - `(1 - (ramp' - ramp) ⋅ iord) ⋅ (bcpc0' - bcpc0)`
 - `(1 - (ramp' - ramp) ⋅ iord) ⋅ (bcpc1' - bcpc1)`

Additionally, `bcpc0` must initially be zero, which is enforced by an initial constraint.
This upper-bounds the degrees of the Bézout coefficient polynomials, which are built from main columns `bcpc0` and `bcpc1`.
Two transition constraints enforce the correct build-up of the Bézout coefficient polynomials:

 - `(1 - (ramp' - ramp) ⋅ iord) ⋅ (bc0' - bc0) + (ramp' - ramp) ⋅ (bc0' - α ⋅ bc0 - bcpc0')`
 - `(1 - (ramp' - ramp) ⋅ iord) ⋅ (bc1' - bc1) + (ramp' - ramp) ⋅ (bc1' - α ⋅ bc1 - bcpc1')`

Additionally, `bc0` must initially be zero, and `bc1` must initially equal `bcpc1`.
This is enforced by initial constraints `bc0` and `bc1 - bcpc1`.

Lastly, the verifier verifies the randomized AIR terminal constraint `bc0 ⋅ rpp + bc1 ⋅ fd - 1`.

### Completeness.
For honest provers, the gcd is guaranteed to be one. As a result, the protocol has perfect completeness. $\square$

### Soundness.
If the table has at least one non-contiguous region, then $f_{\mathsf{rp}}(X)$ and $f_{\mathsf{fd}}(X)$ share at least one factor.
As a result, no Bézout coefficients $f_\mathsf{bc0}(X)$ and $f_\mathsf{bc1}(X)$ can exist such that $f_\mathsf{bc0}(X) \cdot f_{\mathsf{rp}}(X) + f_\mathsf{bc1}(X) \cdot f_{\mathsf{fd}}(X) = 1$.
The verifier therefore probes unequal polynomials of degree at most $2T - 2$, where $T$ is the length of the execution trace, which is upper bounded by $2^{32}$.
According to the Schwartz-Zippel lemma, the false positive probability is at most $(2T - 2) / \vert \mathbb{F} \vert$. $\square$

### Zero-Knowledge.
Since the contiguity argument is constructed using only AET and AIR, zero-knowledge follows from Triton VM's zk-STNARK engine. $\square$

## Summary of constraints

We present a summary of all constraints.

### Initial

 - `bcpc0`
 - `bc0`
 - `bc1 - bcpc1`
 - `fd - 1`
 - `rpp - (α - ramp)`

### Consistency

None.

### Transition

 - `(ramp' - ramp) ⋅ ((ramp' - ramp) ⋅ iord - 1)`
 - `iord ⋅ ((ramp' - ramp) ⋅ iord - 1)`
 - `(1 - (ramp' - ramp) ⋅ iord) ⋅ (bcpc0' - bcpc0)`
 - `(1 - (ramp' - ramp) ⋅ iord) ⋅ (bcpc1' - bcpc1)` 
 - `(ramp' - ramp) ⋅ (rpp' - rpp ⋅ (α - ramp')) + (1 - (ramp' - ramp) ⋅ iord) ⋅ (rpp' - rpp)`
 - `(ramp' - ramp) ⋅ (fd' - rp - (α - ramp') ⋅ fd) + (1 - (ramp' - ramp) ⋅ iord) ⋅ (fd' - fd)`
 - `(1 - (ramp' - ramp) ⋅ iord) ⋅ (bc0' - bc0) + (ramp' - ramp) ⋅ (bc0' - α ⋅ bc0 - bcpc0')`
 - `(1 - (ramp' - ramp) ⋅ iord) ⋅ (bc1' - bc1) + (ramp' - ramp) ⋅ (bc1' - α ⋅ bc1 - bcpc1')`

### Terminal

 - `bc0 ⋅ rpp + bc1 ⋅ fd - 1`

[^op_stack]: See [data structures](data-structures.md#operational-stack) and [registers](registers.md#stack) for explanations of the specific value 16.
