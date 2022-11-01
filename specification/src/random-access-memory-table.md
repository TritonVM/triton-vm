# Random Access Memory Table

The RAM is accessible through `read_mem` and `write_mem` commands.
The RAM Table has six columns:

 - the cycle counter `clk`,
 - RAM address pointer `ramp`,
 - the value of the memory at that address `ramv`,
 - helper variable `iord` ("inverse of `ramp` difference", but elsewhere also "difference inverse" and `di` for short),
 - Bézout coefficient polynomial coefficient 0 `bcpc0`,
 - Bézout coefficient polynomial coefficient 1 `bcpc1`,
 - the inverse-or-zero of clock cycle differences minus 1 `clk_di`.

Columns `clk`, `ramv`, and `ramp` correspond to the columns of the same name in the Processor Table. A permutation argument with the Processor Table establishes that, selecting the columns with these labels, the two tables' sets of rows are identical.

Column `iord` helps with detecting a change of `ramp` across two RAM Table rows.
The function of `iord` is best explained in the context of sorting the RAM Table's rows, which is what the next section is about.

The Bézout coefficient polynomial coefficients `bcpc0` and `bcpc1` represent the coefficients of polynomials that are needed for the contiguity argument. This argument establishes that all regions of constant `ramp` are contiguous.

Column `clk_di` is used to discount clock cycle differences that are equal to one. It is necessary to establish the inner sorting by `clk` within contiguous regions of constant `ramp`.

## Sorting Rows

Up to order, the rows of the Hash Table in columns `clk`, `ramp`, `ramv` are identical to the rows in the Processor Table in columns `clk`, `st1`, and `ramv`.
In the Hash Table, the rows are sorted by memory address first, then by cycle counter.

Coming back to `iord`:
if the difference between `ramp` in row $i$ and row $i+1$ is 0, then `iord` in row $i$ is 0.
Otherwise, `iord` in row $i$ is the multiplicative inverse of the difference between `ramp` in row $i+1$ and `ramp` in row $i$.
In the last row, there being no next row, `iord` is 0.

An example of the mechanics can be found below.
For illustrative purposes only, we use four stack registers `st0` through `st3` in the example.
TritonVM has 16 stack registers, `st0` through `st15`.

Processor Table:

| `clk` | `ci`        | `nia`  | `st0` | `st1` | `st2` | `st3` | `ramv` |
|------:|:------------|:-------|------:|------:|------:|------:|-------:|
|     0 | `push`      | 5      |     0 |     0 |     0 |     0 |      0 |
|     1 | `push`      | 6      |     5 |     0 |     0 |     0 |      0 |
|     2 | `write_mem` | `pop`  |     6 |     5 |     0 |     0 |      0 |
|     3 | `pop`       | `pop`  |     6 |     5 |     0 |     0 |      6 |
|     4 | `pop`       | `push` |     5 |     0 |     0 |     0 |      0 |
|     5 | `push`      | 15     |     0 |     0 |     0 |     0 |      0 |
|     6 | `push`      | 16     |    15 |     0 |     0 |     0 |      0 |
|     7 | `write_mem` | `pop`  |    16 |    15 |     0 |     0 |      0 |
|     8 | `pop`       | `pop`  |    16 |    15 |     0 |     0 |     16 |
|     9 | `pop`       | `push` |    15 |     0 |     0 |     0 |      0 |
|    10 | `push`      | 5      |     0 |     0 |     0 |     0 |      0 |
|    11 | `push`      | 0      |     5 |     0 |     0 |     0 |      0 |
|    12 | `read_mem`  | `pop`  |     0 |     5 |     0 |     0 |      6 |
|    13 | `pop`       | `pop`  |     6 |     5 |     0 |     0 |      6 |
|    14 | `pop`       | `push` |     5 |     0 |     0 |     0 |      0 |
|    15 | `push`      | 15     |     0 |     0 |     0 |     0 |      0 |
|    16 | `push`      | 0      |    15 |     0 |     0 |     0 |      0 |
|    17 | `read_mem`  | `pop`  |     0 |    15 |     0 |     0 |     16 |
|    18 | `pop`       | `pop`  |    16 |    15 |     0 |     0 |     16 |
|    19 | `pop`       | `push` |    15 |     0 |     0 |     0 |      0 |
|    20 | `push`      | 5      |     0 |     0 |     0 |     0 |      0 |
|    21 | `push`      | 7      |     5 |     0 |     0 |     0 |      0 |
|    22 | `write_mem` | `pop`  |     7 |     5 |     0 |     0 |      6 |
|    23 | `pop`       | `pop`  |     7 |     5 |     0 |     0 |      7 |
|    24 | `pop`       | `push` |     5 |     0 |     0 |     0 |      0 |
|    25 | `push`      | 15     |     0 |     0 |     0 |     0 |      0 |
|    26 | `push`      | 0      |    15 |     0 |     0 |     0 |      0 |
|    27 | `read_mem`  | `push` |     0 |    15 |     0 |     0 |     16 |
|    28 | `push`      | 5      |    16 |    15 |     0 |     0 |     16 |
|    29 | `push`      | 0      |     5 |    16 |    15 |     0 |      0 |
|    30 | `read_mem`  | `halt` |     0 |     5 |    16 |    15 |      7 |
|    31 | `halt`      | `halt` |     7 |     5 |    16 |    15 |      7 |

RAM Table:

| `clk` | `ramp` (≘`st1`) | `ramv` | `iord`     |
|------:|----------------:|-------:|-----------:|
|     0 |               0 |      0 |          0 |
|     1 |               0 |      0 |          0 |
|     4 |               0 |      0 |          0 |
|     5 |               0 |      0 |          0 |
|     6 |               0 |      0 |          0 |
|     9 |               0 |      0 |          0 |
|    10 |               0 |      0 |          0 |
|    11 |               0 |      0 |          0 |
|    14 |               0 |      0 |          0 |
|    15 |               0 |      0 |          0 |
|    16 |               0 |      0 |          0 |
|    19 |               0 |      0 |          0 |
|    20 |               0 |      0 |          0 |
|    21 |               0 |      0 |          0 |
|    24 |               0 |      0 |          0 |
|    25 |               0 |      0 |          0 |
|    26 |               0 |      0 |   $5^{-1}$ |
|     2 |               5 |      0 |          0 |
|     3 |               5 |      6 |          0 |
|    12 |               5 |      6 |          0 |
|    13 |               5 |      6 |          0 |
|    22 |               5 |      6 |          0 |
|    23 |               5 |      7 |          0 |
|    30 |               5 |      7 |          0 |
|    31 |               5 |      7 |  $10^{-1}$ |
|     7 |              15 |      0 |          0 |
|     8 |              15 |     16 |          0 |
|    17 |              15 |     16 |          0 |
|    18 |              15 |     16 |          0 |
|    27 |              15 |     16 |          0 |
|    28 |              15 |     16 |          1 |
|    29 |              16 |      0 |          0 |


## Padding

A padding row is a direct copy of the RAM Table's row with the highest value for column `clk`, called template row, with the exception of the cycle count column `clk`.
In a padding row, the value of column `clk` is 1 greater than the value of column `clk` in the template row.
The padding row is inserted right below the template row.
Finally, the value of column `iord` is set to 0 in the template row.
These steps are repeated until the desired padded height is reached.
In total, above steps ensure that the Permutation Argument between the RAM Table and the [Processor Table](processor-table.md) holds up.

## Contiguity Argument

As a stepping stone to proving memory-consistency, it is necessary to prove that all regions of constant `ramp` are contiguous. In simpler terms, this condition stipulates that after filtering the rows in the RAM Table for any given `ramp` value, the resulting sublist of rows forms a contiguous sublist with no gaps. The contiguity establishes this property. What follows here is a summary of the [Contiguity Argument for RAM Consistency](memory-consistency.md#contiguity-for-ram-table).

The contiguity argument is a Randomized AIR without Preprocessing (RAP). In particular, there are 4 extension columns whose values depend on the verifier's challenge $\alpha$:

 - The running product polynomial `rpp`, which accumulates a factor $(\alpha - \mathsf{ramp})$ in every consecutive pair of rows (including the first) where the current row's `ramp` value is different from the previous row's `ramp` value.
 - The formal derivative `fd`, which is updated according to the product rule of differentiation and therefore tracks the formal derivative of `rpp`.
 - The Bézout coefficient 0 `bc0`, which is the evaluation of the polynomial defined by the coefficients of `bcpc0` in $\alpha$.
 - The Bézout coefficient 1 `bc1`, which is the evaluation of the polynomial defined by the coefficients of `bcpc1` in $\alpha$.

The contiguity of regions of constant `ramp` is implied by the square-freeness (as a polynomial in $\alpha$) of `rpp`. If `rpp` is square-free, then the Bézout relation

$$ \mathsf{bc0} \cdot \mathsf{rpp} + \mathsf{bc1} \cdot \mathsf{fd} = 1 $$

holds for any $\alpha$. However, if `rp` is *not* square-free, indicating a malicious prover, then the above relation holds in a negligible fraction of possible $\alpha$'s. Therefore, the AIR enforces the Bézout relation as a terminal boundary constraint.

## Inner Sorting

The second stepping stone to proving memory-consistency is to establish that the rows in each region of constant `ramp` are correctly sorted for `clk` in ascending order. To prove this property, we show that all differences of `clk` values difference greater than 1 in consecutive rows with the same `ramp` value – the *clock jump differences* – are contained in the `clk` table of the [Processor Table](processor-table.md). What follows here is a summary of the construction reduced to the RAM Table; the bulk of the logic and constraints that make this argument work is located in the Processor Table. The entirety of this construction is motivated and explained in [TIP-0003](https://github.com/TritonVM/triton-vm/blob/master/tips/tip-0003/tip-0003.md).

The inner sorting argument requires one extension column `cjdrp`, which contains a running product. Specifically, this column accumulates a factor $\beta - (\mathsf{clk}' - \mathsf{clk})$ in every pair of consecutive rows where a) the `ramp` value is the same, and b) the difference in `clk` values minus one is different from zero.

## Row Permutation Argument

Selecting for the columns `clk`, `ramp`, and `ramv`, the set of rows of the RAM Table is identical to the set of rows of the Processor Table. This argument requires one extension column, `rppa`, short for "running product for Permutation Argument". This column accumulates a factor $(a \cdot \mathsf{clk} + b \cdot \mathsf{ramp} + c \cdot \mathsf{ramv} - \gamma)$ in every row. In this expression, $a$, $b$, $c$, and $\gamma$ are challenges from the verifier.

## Constraints

### Initial Constraints

1. Cycle count `clk` is 0.
1. RAM pointer `ramp` is 0.
1. RAM value `ramv` is 0.
4. The first coefficient of the Bézout coefficient polynomial 0 `bcpc0` is 0.
5. The Bézout coefficient 0 `bc0` is 0.
6. The Bézout coefficient 1 `bc1` is equal to the first coefficient of the Bézout coefficient polynomial `bcpc1`.
7. The running product polynomial `rpp` starts with `α-ramp`.
8. The formal derivative `fd` starts with 1.
9. The running product for clock jump differences `rpcjd` starts with 1.
10. The running product for the permutation argument `rppa` has accumulated the first factor.

**Initial Constraints as Polynomials**

1. `clk`
1. `ramp`
1. `ramv`
4. `bcpc0`
5. `bc0`
6. `bc1 - bcpc1`
7. `rpp - α + ramp`
8. `fd - 1`
9. `rpcjd - 1`
10. `rppa - (a · clk + b · ramp + c · ramv - γ)`

### Consistency Constraints

None.

### Transition Constraints

1. If `(ramp - ramp')` is 0, then `iord` is 0, else `iord` is the multiplicative inverse of `(ramp' - ramp)`.
1. If the `ramp` changes, then the new `ramv` must be 0.
1. If the `ramp` does not change and the `ramv` does change, then the cycle counter `clk` must increase by 1.
4. The Bézout coefficient polynomial coefficients are allowed to change only when the `ramp` changes.
5. The running product polynomial `rpp` accumulates a factor `(α - ramp)` whenever `ramp` changes.
6. The clock difference inverse `clk_di` is the inverse-or-zero of the clock difference minus 1.
7. The running product for clock jump differences `rpcjd` accumulates a factor `(β - clk' + clk)` whenever that difference is greater than one and `ramp` is the same.
8. The running product for the permutation argument accumulates a factor for the new row.

Written as Disjunctive Normal Form, the same constraints can be expressed as:
1. `iord` is 0 or `iord` is the inverse of `(ramp' - ramp)`.
1. `(ramp' - ramp)` is zero or `iord` is the inverse of `(ramp' - ramp)`.
1. The `ramp` does not change or the new `ramv` is 0.
1. The `ramp` does change or the `ramv` does not change or the `clk` increases by 1.
5. `bcpc0' - bcpc0` is zero or `(ramp' - ramp)` is nonzero.
6. `bcpc1' - bcpc1` is zero or `(ramp' - ramp)` is nonzero.
7. `(ramp' - ramp)` is zero and `rpp' = rpp`; or `(ramp' - ramp)` is nonzero and `rpp' = rpp·(ramp'-α))` is zero.
8. the formal derivative `fd` applies the product rule of differentiation (as necessary).
9. Bézout coefficient 0 is evaluated correctly.
10. Bézout coefficient 1 is evaluated correctly.
11. `clk_di` is zero or the inverse of `(clk' - clk - 1)`.
12. `(clk' - clk - 1)` is zero or the inverse of `clk_di`.
13. `(clk' - clk - 1) ≠ 0` and `rpcjd' = rpcjd`;  or `ramp' - ramp ≠ 0` and `rpcjd' = rpcjd`; or `(clk' - clk - 1) = 0` and `ramp' - ramp = 0` and `rpcjd' = rpcjd · (β - clk' + clk)`.
14. `rppa' = rppa · (a · clk + b · ramp + c · ramv - γ)`

**Transition Constraints as Polynomials**

1. `iord·(iord·(ramp' - ramp) - 1)`
1. `(ramp' - ramp)·(iord·(ramp' - ramp) - 1)`
1. `(ramp' - ramp)·ramv'`
1. `(iord·(ramp' - ramp) - 1)·(ramv' - ramv)·(clk' - (clk + 1))`
5. `(iord·(ramp' - ramp) - 1)·(bcpc0' - bcpc0)`
6. `(iord·(ramp' - ramp) - 1)·(bcpc1' - bcpc1)`
7. `(iord·(ramp' - ramp) - 1)·(rpp' - rpp) + (ramp' - ramp)·(rpp' - rpp·(ramp'-α))`
8. `(iord·(ramp' - ramp) - 1)·(fd' - fd) + (ramp' - ramp)·(fd' - fd·(ramp'-α) - rpp)`
9. `(iord·(ramp' - ramp) - 1)·(bc0' - bc0) + (ramp' - ramp) · (bc0' - bc0 · α - bcpc0')`
10. `(iord·(ramp' - ramp) - 1)·(bc1' - bc1) + (ramp' - ramp) · (bc1' - bc1 · α - bcpc1')`
11. `clk_di · (clk_di · (clk' - clk - 1) - 1)`
12. `(clk' - clk - 1) · (clk_di · (clk' - clk - 1) - 1)`
13. `(clk' - clk - 1) · (rpcjd' - rpcjd) + (1 - (ramp' - ramp) · iord) · (rpcjd' - rpcjd) + (1 - (clk' - clk - 1) · clk_di) · ramp · (rpcjd' - rpcjd · (β - ramp))`
14. `rppa' - rppa · (a · clk + b · ramp + c · ramv - γ)`

### Terminal Constraints

1. The Bézout relation holds between `rp`, `fd`, `bc0`, and `bc1`.

**Terminal Constraints as Polynomials**

1. `rpp · bc0 + fd · bc1 - 1`

### Relations to Other Tables

1. A Permutation Argument establishes that the rows in the RAM Table correspond to the rows of the [Processor Table](processor-table.md), after selecting for columns `clk`, `ramp`, `ramv` in both tables. The running product for this argument is contained in the `rppa` column.
2. A multi-table Permutation Argument shows that all clock jump differences greater than one, from all memory-like tables (i.e., including the [OpStack Table](operational-stack-table.md) and the [JumpStack Table](jump-stack-table.md)), are contained in the `cjd` column of the [Processor Table](processor-table.md). The running product for this argument is contained in the `rpcjd` column.