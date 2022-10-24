# Memory-Consistency

Triton has three memory-like units: the RAM, the JumpStack, and the OpStack. Each such unit has a corresponding table. *Memory-consistency* is the property that whenever the processor reads a data element from these units, the value that it receives corresponds to the value sent the last time when that cell was written. Memory consistency follows from two intermediate properties.

 1. Contiguity of regions of constant memory pointer. After selecting from each table all the rows with any given memory pointer, the resulting sublist is contiguous, which is to say, there are no gaps and two such sublists can never interleave.
 2. Inner-sorting within contiguous regions. Within each such contiguous region, the rows are sorted in ascending order by clock cycle.

The contiguity of regions of constant memory pointer is established differently for the RAM Table than for the OpStack or JumpStack Tables. The OpStack and JumpStack Tables enjoy a particular property whereby the memory pointer can only every increase or decrease by one (or stay the same). As a result, simple AIR constraints can enforce the correct sorting by memory pointer. In contrast, the memory pointer for the RAM Table can jump arbitrarily. As explained below, an argument involving formal derivatives and Bézout's relation establishes contiguity.

The correct inner sorting is establish the same way for all three memory-like tables. The set of all clock jump differences – differences greater than 1 of clock cycles within regions of constant memory pointer – is shown to be contained in the set of all clock cycles. Under reasonable assumptions about the running time, this fact implies that all clock jumps are directed forwards, as opposed to backwards, which in turn implies that the rows are sorted for clock cycle.

The next sections elaborate on these constructions. Another section shows that these two properties do indeed suffice to prove memory-consistency.

*Historical note.* The constructions documented here correspond to Triton Improvement Proposals (TIPs) [0001](https://github.com/TritonVM/triton-vm/blob/master/tips/tip-0001/tip-0001.md) and [0003](https://github.com/TritonVM/triton-vm/blob/master/tips/tip-0003/tip-0003.md). This note attempts to provide a self-contained summary.

## Contiguity for OpStack Table

In each cycle, the memory pointer for the OpStack Table, `osp`, can only ever increase by one, remain the same, or decrease by one. As a result, it is easy to enforce that the entire table is sorted for memory pointer using one initial boundary constraint and one transition constraint.

 - Initial boundary constraint: `osp` starts with zero, so in terms of polynomials the constraint is `osp`.
 - Transition constraint: the new `osp` is either the same as the previous or one larger. The polynomial representation for this constraint is `(osp' - osp - 1) * (osp' - osp)`.

## Contiguity for JumpStack Table

Analogously to the OpStack Table, the JumpStack's memory pointer `jsp` can only ever decrease by one, remain the same, or increase by one, within each cycle. As a result, similar constraints establish that the entire table is sorted for memory pointer.

 - Initial boundary constraint: `jsp` starts with zero, so in terms of polynomials the constraint is `jsp`.
 - Transition constraint: the new `jsp` is either the same as the previous or one larger. The polynomial representation for this constraint is `(jsp' - jsp - 1) * (jsp' - jsp)`.

 ## Contiguity for RAM Table

 This *contiguity argument* is a collection of three base columns, four extension columns, four deterministic initial constraints, one randomized initial constraint, four deterministic transition constraints, four randomized transition constraints, and one randomized terminal constraint.

 - The first base column `iord` and two deterministic transition constraints enable conditioning on a changed memory pointer.
 - The second and third base columns, `bcpc0` and `bcpc1`, and the other two deterministic transition constraints contain and constrain the symbolic Bézout coefficient polynomials' coefficients.
 - The first extension column `rpp` is a running product similar to that of a conditioned permutation argument. The first randomized transition constraint verifies the correct accumulation of factors for updating this column.
 - The second extension column `fd` is the formal derivative of the first. The second randomized transition constraint verifies the correct application of the product rule of differentiation to update this column.
 - The third and fourth extension columns, `bc0` and `bc1`, build up the Bézout coefficient polynomials based on the corresponding base columns.
The remaining two randomized transition constraints enforce the correct build-up of the Bézout coefficient polynomials.
 - The terminal constraint takes the weighted sum of the running product and the formal derivative, where the weights are the Bézout coefficient polynomials, and equates it to one. This equation asserts the Bézout relation. It can only be satisfied if the greatest common divisor of the running product and its formal derivative is one – implying that no change in the memory pointer resets it to a value used earlier.


The following table illustrates the idea.
Columns not needed for establishing memory-consistency are not displayed.

| `ramp` | `iord`         | `bcpc0` | `bcpc1` | `rpp`                    | `fd`   | `bc0`    | `bc1`               |
|:-------|:-------------|:--------|:--------|:------------------------|:-------|:---------|:--------------------|
| $a$    | 0            | $0$     | $\ell$  | $(X - a)$               | $1$    | $0$      | $\ell$              |
| $a$    | $(b-a)^{-1}$ | $0$     | $\ell$  | $(X - a)$               | $1$    | $0$      | $\ell$              |
| $b$    | 0            | $j$     | $m$     | $(X - a)(X - b)$        | $p(X)$ | $j$      | $\ell X + m$        |
| $b$    | 0            | $j$     | $m$     | $(X - a)(X - b)$        | $p(X)$ | $j$      | $\ell X + m$        |
| $b$    | $(c-b)^{-1}$ | $j$     | $m$     | $(X - a)(X - b)$        | $p(X)$ | $j$      | $\ell X + m$        |
| $c$    | 0            | $k$     | $n$     | $(X - a)(X - b)(X - c)$ | $q(X)$ | $jX + k$ | $\ell X^2 + mX + n$ |
| $c$    | -            | $k$     | $n$     | $(X - a)(X - b)(X - c)$ | $q(X)$ | $jX + k$ | $\ell X^2 + mX + n$ |

The values contained in the extension columns are undetermined until the verifier's challenge $\alpha$ is known; before that happens it is worthwhile to present the polynomial expressions in $X$, anticipating the substitution $X \mapsto \alpha$. The constraints are articulated relative to `α`.

The inverse of RAMP difference `iord` takes the inverse of the difference between the current and next `ramp` values if that difference is non-zero, and zero else. This constraint corresponds to two transition constraint polynomials:

 - `(ramp' - ramp) ⋅ ((ramp' - ramp) ⋅ iord - 1)`
 - `iord ⋅ (ramp' - ramp) ⋅ iord - 1)`

The running product `rp` starts with $X - \mathsf{ramp}$ initially, which is enforced by an initial constraint.
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

The (scalar) coefficients of the Bézout coefficient polynomials are recorded in base columns `bcpc0` and `bcpc1`, respectively.
The transition constraints for these columns enforce that the value in one such column can only change if the memory pointer `ramp` changes.
However, unlike the conditional update rule enforced by the transition constraints of `rp` and `fd`, the new value is unconstrained.
Concretely, the two transition constraints are:

 - `(1 - (ramp' - ramp) ⋅ iord) ⋅ (bcpc0' - bcpc0)`
 - `(1 - (ramp' - ramp) ⋅ iord) ⋅ (bcpc1' - bcpc1)`

Additionally, `bcpc0` must initially be zero, which is enforced by an initial constraint.
This upper-bounds the degrees of the Bézout coefficient polynomials, which are built from base columns `bcpc0` and `bcpc1`.
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
The verifier therefore probes unequal polynomials of degree at most $2T - 2$.
According to the Schwartz-Zippel lemma, the false positive probability is at most $(2T - 2) / \vert \mathbb{F} \vert$. $\square$

### Zero-Knowledge.
Since the contiguity argument is constructed using only AET and AIR, zero-knowledge follows from Triton VM's zk-STNARK engine. $\square$

## Summary of constraints

We present a summary of all constraints.

### Initial

- $\mathsf{bcpc0}$
- $\mathsf{bc0}$
- $\mathsf{bc1} - \mathsf{bcpc1}$
- $\mathsf{fd} - 1$
- $\mathsf{rp} - (X - \mathsf{mp})$

### Consistency
None.

### Transition

- $(\mathsf{ramp}^\star - \mathsf{ramp})\cdot((\mathsf{ramp}^\star - \mathsf{ramp}) \cdot \mathsf{di} - 1)$
- $\mathsf{di}\cdot((\mathsf{ramp}^\star - \mathsf{ramp}) \cdot \mathsf{di} - 1)$
- $(1 - (\mathsf{ramp}^\star - \mathsf{ramp}) \cdot \mathsf{di}) \cdot (\mathsf{bcpc0}^\star - \mathsf{bcpc0})$
- $(1 - (\mathsf{ramp}^\star - \mathsf{ramp}) \cdot \mathsf{di}) \cdot (\mathsf{bcpc1}^\star - \mathsf{bcpc1})$
- $(\mathsf{ramp}^\star - \mathsf{ramp}) \cdot (\mathsf{rp}^\star - \mathsf{rp} \cdot (X - \mathsf{ramp}^\star)) + (1 -(\mathsf{ramp}^\star -\mathsf{ramp}) \cdot \mathsf{di}) \cdot (\mathsf{rp}^\star - \mathsf{rp})$
- $(\mathsf{ramp}^\star - \mathsf{ramp}) \cdot (\mathsf{fd}^\star - \mathsf{rp} - (X - \mathsf{ramp}^\star) \cdot \mathsf{fd}) + (1 -(\mathsf{ramp}^\star -\mathsf{ramp}) \cdot \mathsf{di}) \cdot (\mathsf{fd}^\star - \mathsf{fd})$
- $(1 - (\mathsf{ramp}^\star - \mathsf{ramp}) \cdot \mathsf{di}) \cdot (\mathsf{bc0}^\star - \mathsf{bc0}) + (\mathsf{ramp}^\star - \mathsf{ramp}) \cdot (\mathsf{bc0}^\star - X\cdot\mathsf{bc0} - \mathsf{bcpc0}^\star)$
- $(1 - (\mathsf{ramp}^\star - \mathsf{ramp}) \cdot \mathsf{di}) \cdot (\mathsf{bc1}^\star - \mathsf{bc1}) + (\mathsf{ramp}^\star - \mathsf{ramp}) \cdot (\mathsf{bc1}^\star - X\cdot\mathsf{bc1} - \mathsf{bcpc1}^\star)$

### Terminal

- $\mathsf{bc0} \cdot {\mathsf{rp}} + \mathsf{bc1} \cdot {\mathsf{fd}} - 1$
