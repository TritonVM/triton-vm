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

### Summary of constraints

We present a summary of all constraints.

#### Initial

 - `bcpc0`
 - `bc0`
 - `bc1 - bcpc1`
 - `fd - 1`
 - `rpp - (α - ramp)`

#### Consistency
None.

#### Transition

 - `(ramp' - ramp) ⋅ ((ramp' - ramp) ⋅ iord - 1)`
 - `iord ⋅ ((ramp' - ramp) ⋅ iord - 1)`
 - `(1 - (ramp' - ramp) ⋅ iord) ⋅ (bcpc0' - bcpc0)`
 - `(1 - (ramp' - ramp) ⋅ iord) ⋅ (bcpc1' - bcpc1)` 
 - `(ramp' - ramp) ⋅ (rpp' - rpp ⋅ (α - ramp')) + (1 - (ramp' - ramp) ⋅ iord) ⋅ (rpp' - rpp)`
 - `(ramp' - ramp) ⋅ (fd' - rp - (α - ramp') ⋅ fd) + (1 - (ramp' - ramp) ⋅ iord) ⋅ (fd' - fd)`
 - `(1 - (ramp' - ramp) ⋅ iord) ⋅ (bc0' - bc0) + (ramp' - ramp) ⋅ (bc0' - α ⋅ bc0 - bcpc0')`
 - `(1 - (ramp' - ramp) ⋅ iord) ⋅ (bc1' - bc1) + (ramp' - ramp) ⋅ (bc1' - α ⋅ bc1 - bcpc1')`

#### Terminal

 - `bc0 * rpp + bc1 * fd - 1`

## Clock Jump Differences and Inner Sorting

The previous sections show how it is proven that in the JumpStack, OpStack, and RAM Tables, the regions of constant memory pointer are contiguous. The next step is to prove that within each contiguous region of constant memory pointer, the rows are sorted for clock cycle. That is the topic of this section.

The problem arises from *clock jumps*, which describes the phenomenon when the clock cycle increases by more than 1 even though the memory pointer does not change. If arbitrary jumps were allowed, nothing would prevent the cheating prover from using a table where higher rows correspond to later states, giving rise to an exploitable attack. So it must be shown that every clock jump is directed forward and not backward.

Our strategy is to show that the *difference*, *i.e.*, the next clock cycle minus the current clock cycle, is itself a clock cycle.
Recall that the Processor Table's clock cycles run from 0 to $T-1$.
Therefore, a forward directed clock jump difference is in $F = \lbrace 2, \dots, T - 1 \rbrace \subseteq \mathbb{F}_p$, whereas a backward directed clock jump's difference is in $B = \lbrace -f \mid f \in F \rbrace = \lbrace 1 - T, \dots, -2 \rbrace \subseteq \mathbb{F}_p$.
No other clock jump difference can occur.
If $T \ll p/2$, there is no overlap between sets $F$ and $B$.
As a result, in this regime, showing that a clock jump difference is in $F$ guarantees that the jump is forward-directed.

The set of values in the Processor Table's clock cycle column is $F \cup \lbrace 0,1 \rbrace$.
Standard subset arguments can show that the clock jump differences are elements of that column.
However, it is cumbersome to repeat this argument for three separate tables. What is described here is a construction that combines all three memory-like tables and generates one lookup in the Processor Table's `clk` column. It introduces

 - one base column in each memory-like table;
 - one extension column in each memory-like table;
 - four extra base columns in the Processor Table; and
 - three extension columns in the Processor Table.

## Intuition

 - In order to treat clock jump differences of magnitude 1 separately, each memory-like table needs an extra base column `clk_di`, which holds the inverse of two consecutive rows' cycle count minus 1, *i.e.*, `clk' - clk - 1`, if that inverse exists, and 0 otherwise.
 - A multi-table permutation argument establishes that all clock jump differences (*cjd*s) greater than 1 are contained in a new column `cjd` of the Processor Table.
 Every memory-like table needs one extension column `rpcjd` and the Processor Table needs one matching extension column `rpm` to effect this permutation argument.
 - In addition to the extension column computing the running product, the Processor Table needs an inverse column `invm` to help select all *nonzero* `cjd`s, and thus skip padding rows. The abbreviation *invm* is short for inverse-with-multiplicities.
 - An inverse column `invu` in the Processor Table allows for selecting the first row of every contiguous region of `cjd`.
 The abbreviation *invu* is short for unique-inverse.
 - An evaluation argument establishes that a selection of clock cycles and the unique clock jump differences are identical lists.
 This evaluation argument requires two more extension columns on the Processor Table: `rer` computes the running evaluation for the *relevant* clock cycles, whereas `reu` computes the running evaluation for the *unique* clock jump differences.

![](img/cjd-relations-diagram.svg)

### Memory-like Tables

Here are the constraints for the RAM Table. The constraints for the other two tables are analogous and are therefore omitted from this section. Where necessary, the suffices `_ram`, `_js`, and `_os` disambiguate between the RAM Table, JumpStack Table, and OpStack Table, respectively.

Use `mp` to abstractly refer to the memory pointer. Depending on the table, that would be `ramp`, `jsp`, or `osp`. The first extension column, `rpcjd`, computes a running product. It starts with 1, giving rise to the boundary constraint `rpcjd - 1`.

The transition constraint enforces the accumulation of a factor `(α - clk' + clk)` whenever the memory pointer is the same or the clock jump difference is greater than 1.
If the memory pointer is changed or the clock jump difference is exactly 1, the same running product is carried to the next row.
Expressed in Boolean logic:

```
    clk' - clk ≠ 1 /\ mp' = mp => rpcjd' = rpcjd ⋅ (α - (clk' - clk))
    clk' - clk = 1 \/ mp' ≠ mp => rp' = rp
```

The corresponding transition constraint is

```
(clk' - clk - 1) ⋅ (1 - (mp' - mp) ⋅ iord) ⋅ (rpcjd' - rpcjd ⋅ (α - (clk' - clk)))
  + (1 - (clk' - clk - 1) ⋅ clk_di) ⋅ (rpcjd' - rpcjd)
  + (mp' - mp) ⋅ (rpcjd' - rpcjd).
```

Note that `iord` is the difference inverse of the RAM Table but for the other two tables this factor can be dropped since the corresponding memory pointer can only change by either 0 or 1 between consecutive rows.

The column `clk_di` contains the inverse-or-zero of the two consecutive clocks, minus one.
This consistency requirement induces two transition constraints:

 - `(clk' - clk - 1) ⋅ (1 - (clk' - clk - 1) ⋅ clk_di)`
 - `clk_di ⋅ (1 - (clk' - clk - 1) ⋅ clk_di)`

### Clock Jump Differences with Multiplicities in the Processor Table

All clock jump differences (that are greater than 1) of all the memory-like tables are listed in the `cjd` column of the Processor Table.
The values are sorted and the padding inserts “0” rows at the bottom.

This cross-table relation comes with another extension column, this time in the Processor Table, that computes a running product. This column is denoted by `rpm`.
This running product accumulates a factor `(α - cjd)` in every row where `cjd ≠ 0`.
Column `invm` (for *inverse-with-multiplicities*), which is the inverse-or-zero of `cjd`, allows writing inequality `cjd ≠ 0` as a polynomial of low degree.

The first factor is accumulated in the first row, giving rise to boundary constraint `cjd ⋅ (rpm - (α - cjd)) + (1 - invm ⋅ cjd) ⋅ (rpm - 1)`.

The transition constraint is `cjd ⋅ (rpm' - rpm ⋅ (α - cjd)) + (1 - invm ⋅ cjd) ⋅ (rpm' - rpm)`.

The consistency constraints for the inverse are

 - `cjd ⋅ (1 - cjd ⋅ invm)`
 - `invm ⋅ (1 - cjd ⋅ invm)`.

The terminal value of this column must be equal to the terminal values of the matching running products of the memory-like tables. The cross-table terminal boundary constraint is therefore: `rpm - rpcjd_ram ⋅ rpcjd_js ⋅ rpcjd_os`.

#### Total Number of Clock Jump Differences with Multiplicities

Recall that the Processor Table has length $T$.
An honest prover can convince the verifier only if the total number of clock jump differences accumulated by the running product `rpm` is no greater than $T$, independent of the executed program.

If, in the Processor Table, some memory pointer does not change between two consecutive clock cycles, the clock jump difference this produces in the corresponding memory-like table is 1.
Clock jump differences of exactly 1 are treated explicitly and do not require a lookup, *i.e.*, do not contribute a factor to `rpm`.
Thus, `rpm` accumulates at most $T$ factors if all instructions change at most one of the three memory pointers. This is indeed the case.
The table [“Modified Memory Pointers by Instruction” in the appendix](#modified-memory-pointers-by-instruction) lists all instructions and the memory pointers they change.

### Unique Clock Jump Differences in the Processor Table

As described earlier, `invu` is used to select the first row of regions of constant `cjd`. The expression `invu * (cjd' - cjd)` is 1 in such rows.

Using this indicator, we build a running evaluation that accumulates one step of evaluation relative to `cjd` for each contiguous region, excluding the padding region. The clock jump differences accumulated in this manner are unique, giving rise to the column's name: `reu`, short for *running evaluation* over *unique* cjd's.

The first clock jump difference is accumulated in the first row, giving rise to the boundary constraint `reu - β - cjd`.

The running evaluation accumulates one step of evaluation whenever the indicator bit is set and the new clock jump difference is not padding.
Otherwise, the running evaluation does not change.
Expressed in Boolean logic:

```
    invu ⋅ (cjd' - cjd) = 1 /\ cjd' ≠ 0 => reu' = β ⋅ reu + cjd'
           (cjd' - cjd) = 0 \/ cjd' = 0 => reu' = reu
```

The following transition constraint captures this transition.

```
    (cjd' - cjd) ⋅ cjd' ⋅ (reu' - β ⋅ reu - cjd')
  + (1 - invu ⋅ (cjd' - cjd)) ⋅ (reu' - reu)
  + (1 - cjd' ⋅ invm') ⋅ (reu' - reu)
```

To verify that the indicator is correctly indicating the first row of every contiguous region, we need `invu` to contain the inverse-or-zero of every consecutive pair of $\mathsf{cjd}$ values. This consistency induces two transition constraints:

 - `invu ⋅ (1 - invu ⋅ (cjd' - cjd))`
 - `(cjd' - cjd) ⋅ (1 - invu ⋅ (cjd' - cjd))`

### Relevant Clock Cycles in the Processor Table

Assume the prover knows when the clock cycle `clk` is also *some* jump in a memory-like table and when it is not. Then it can apply the right running evaluation step as necessary. The prover computes this running evaluation in a column called `rer`, short for *running evaluation* over *relevant* clock cycles.

Since 0 is never a valid clock jump difference, the initial value is 1, giving rise to the initial boundary constraint: `rer - 1`.

In every row, either the running evaluation step is applied, or else the running evaluation remains the same: `(rer' - rer) ⋅ (rer' - β ⋅ rer - clk)`.

The terminal value must be identical to the running evaluation of "Relevant Clock Jumps". This gives rise to the terminal boundary constraint:  `rer - reu`

Whether to apply the evaluation step or not does not need to be constrained since if the prover fails to include certain rows he will have a harder (not easier) time convincing the verifier.

## Memory-Consistency

Whenever the Processor Table reads a value "from" a memory-like table, this value appears nondeterministically and is unconstrained by the base table AIR constraints. However, there is a permutation argument that links the Processor Table to the memory-like table in question. *The construction satisfies memory-consistency if it guarantees that whenever a memory cell is read, its value is consistent with the last time that cell was written.*

The above is too informal to provide a meaningful proof for. Let's put formal meanings on the proposition and premises, before reducing the former to the latter.

Let $P$ denote the Processor Table and $M$ denote the memory-like table. Both have height $T$. Both have columns `clk`, `mp`, and `val`. For $P$ the column `clk` coincides with the index of the row. $P$ has another column `ci`, which contains the current instruction, which is `write`, `read`, or `any`. Obviously, `mp` and `val` are abstract names that depend on the particular memory-like table, just like `write`, `read`, and `any` are abstract instructions that depend on the type of memory being accessed. In the following math notation we use $\mathtt{col}$ to denote the column name and $\mathit{col}$ to denote the value that the column might take in a given row.

**Definition 1 (contiguity):** The memory-like table is *contiguous* iff all sublists of rows with the same memory pointer `mp` are contiguous. Specifically, for any given memory pointer $\mathit{mp}$, there are no rows with a different memory pointer $\mathit{mp}'$ in between rows with memory pointer $\mathit{mp}$.

$$ \forall i < j < k \in \lbrace 0, \ldots, T-1 \rbrace : \mathit{mp} \stackrel{\triangle}{=} M[i][\mathtt{mp}] = M[k][\mathtt{mp}] \Rightarrow M[j][\mathtt{mp}] = \mathit{mp} $$

**Definition 2 (regional sorting):** The memory-like table is *regionally sorted* iff for every contiguous region of constant memory pointer, the clock cycle increases monotonically.

$$ \forall i < j \in \lbrace 0, \ldots, T-1 \rbrace : M[i][\mathtt{mp}] = M[j][\mathtt{mp}] \Rightarrow M[i][\mathtt{clk}] <_{\mathbb{Z}} M[j][\mathtt{clk}] $$

The symbol $<_{\mathbb{Z}}$ denotes the integer less than operator, after lifting the operands from the finite field to the integers.

**Definition 3 (memory-consistency):** A Processor Table $P$ has *memory-consistency* if whenever a memory cell at location $\mathit{mp}$ is read, its value corresponds to the previous time the memory cell at location $\mathit{mp}$ was written. Specifically, there are no writes in between the write and the read, that give the cell a different value.

$$ \forall k \in \lbrace 0 , \ldots, T-1 \rbrace : P[k][\mathtt{ci}] = \mathit{read} \, \Rightarrow \left( (1) \, \Rightarrow \, (2) \right)$$

$$ (1) \exists i  \in \lbrace 0 , \ldots, k \rbrace : P[i][\mathtt{ci}] = \mathit{write} \, \wedge \, P[i+1][\mathtt{val}] = P[k][\mathtt{val}] \, \wedge \, P[i][\mathtt{mp}] = P[k][\mathtt{mp}]$$

$$ (2) \nexists j \in \lbrace i+1 , \ldots, k-1 \rbrace : P[j][\mathtt{ci}] = \mathit{write} \, \wedge \, P[i][\mathtt{mp}] = P[k][\mathtt{mp}] $$

**Theorem 1 (memory-consistency):** Let $P$ be a Processor Table. If there exists a memory-like table $M$ such that

 - selecting for the columns `clk`, `mp`, `val`, the two tables' lists of rows are permutations of each other; and
 - $M$ is contiguous and regionally sorted; and
 - $M$ has no changes in `val` that coincide with clock jumps;

then $P$ has memory-consistency.

*Proof.* For every memory pointer value $\mathit{mp}$, select the sublist of rows $P_{\mathit{mp}} \stackrel{\triangle}{=} \lbrace P[k] \, | \, P[k][\mathtt{mp}] = \mathit{mp} \rbrace$ in order. The way this sublist is constructed guarantees that it coincides with the contiguous region of $M$ where the memory pointer is also $\mathit{mp}$.

Iteratively apply the following procedure to $P_{\mathit{mp}}$: remove the bottom-most row if it does not correspond to a row $k$ that constitutes a counter-example to memory consistency. Specifically, let $i$ be the clock cycle of the previous row in $P_{\mathit{mp}}$.

 - If $i$ satisfies $(1)$ then by construction it also satisfies $(2)$. As a result, row $k$ is not part of a counter-example to memory-consistency. We can therefore remove the bottom-most row and proceed to the next iteration of the outermost loop.
 - If $P[i][\mathtt{ci}] \neq \mathit{write}$ then we can safely ignore this row: if there is no clock jump, then the absence of a $\mathit{write}$-instruction guarantees that $\mathit{val}$ cannot change; and if there is a clock jump, then by assumption on $M$, $\mathit{val}$ cannot change. So set $i$ to the clock cycle of the row above it in $P_{\mathit{mp}}$ and proceed to the next iteration of the inner loop. If there are no rows left for $i$ to index, then there is no possible counterexample for $k$ and so remove the bottom-most row of $P_{\mathit{mp}}$ and proceed to the next iteration of the outermost loop.
 - The case $P[i+1][\mathtt{val}] \neq P[k][\mathtt{val}]$ cannot occur because by construction of $i$, $\mathit{val}$ cannot change.
 - The case $P[i][\mathtt{mp}] \neq P[k][\mathtt{mp}]$ cannot occur because the list was constructed by selecting only elements with the same memory pointer.
 - This list exhausts the possibilities of condition (1).

When $P_{\mathit{mp}}$ consists of only two rows, it can contain no counter-examples. By applying the above procedure, we can reduce every correctly constructed sublist $P_{\mathit{mp}}$ to a list consisting of two rows. Therefore, for every $\mathit{mp}$, the sublist $P_{\mathit{mp}}$ is free of counter-examples to memory-consistency. Equivalently, $P$ is memory-consistent. $\square$

## Appendix

### Modified Memory Pointers by Instruction

|                  |    `osp`     |    `ramp`    |    `jsp`     |
|-----------------:|:------------:|:------------:|:------------:|
|            `pop` | $\mathsf{x}$ |              |              |
|     `push` + `a` | $\mathsf{x}$ |              |              |
|         `divine` | $\mathsf{x}$ |              |              |
|      `dup` + `i` | $\mathsf{x}$ |              |              |
|     `swap` + `i` |              |              |              |
|            `nop` |              |              |              |
|           `skiz` | $\mathsf{x}$ |              |              |
|     `call` + `d` |              |              | $\mathsf{x}$ |
|         `return` |              |              | $\mathsf{x}$ |
|        `recurse` |              |              |              |
|         `assert` | $\mathsf{x}$ |              |              |
|           `halt` |              |              |              |
|       `read_mem` |              | $\mathsf{x}$ |              |
|      `write_mem` |              | $\mathsf{x}$ |              |
|           `hash` |              |              |              |
| `divine_sibling` |              |              |              |
|  `assert_vector` |              |              |              |
|            `add` | $\mathsf{x}$ |              |              |
|            `mul` | $\mathsf{x}$ |              |              |
|         `invert` |              |              |              |
|          `split` | $\mathsf{x}$ |              |              |
|             `eq` | $\mathsf{x}$ |              |              |
|            `lsb` | $\mathsf{x}$ |              |              |
|          `xxadd` |              |              |              |
|          `xxmul` |              |              |              |
|        `xinvert` |              |              |              |
|          `xbmul` | $\mathsf{x}$ |              |              |
|        `read_io` | $\mathsf{x}$ |              |              |
|       `write_io` | $\mathsf{x}$ |              |              |



