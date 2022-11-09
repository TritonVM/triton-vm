# Random Access Memory Table

The RAM is accessible through `read_mem` and `write_mem` commands.

## Base Columns

The RAM Table has 7 columns:
1. the cycle counter `clk`,
1. the inverse-or-zero of clock cycle differences minus 1 `clk_di`.
1. RAM address pointer `ramp`,
1. the value of the memory at that address `ramv`,
1. helper variable `iord` ("inverse of `ramp` difference", but elsewhere also "difference inverse" and `di` for short),
1. Bézout coefficient polynomial coefficient 0 `bcpc0`,
1. Bézout coefficient polynomial coefficient 1 `bcpc1`,

| Clock | Inverse of Clock Difference Minus One | RAM Pointer | RAM Value | Inverse of RAM Pointer Difference | Bézout coefficient polynomial's coefficients 0 | Bézout coefficient polynomial's coefficients 1 |
|:------|:--------------------------------------|:------------|:----------|:----------------------------------|:-----------------------------------------------|:-----------------------------------------------|
| -     | -                                     | -           | -         | -                                 | -                                              | -                                              |

Columns `clk`, `ramv`, and `ramp` correspond to the columns of the same name in the [Processor Table](processor-table.md).
A permutation argument with the Processor Table establishes that, selecting the columns with these labels, the two tables' sets of rows are identical.

Column `iord` helps with detecting a change of `ramp` across two RAM Table rows.
The function of `iord` is best explained in the context of sorting the RAM Table's rows, which is what the next section is about.

The Bézout coefficient polynomial coefficients `bcpc0` and `bcpc1` represent the coefficients of polynomials that are needed for the [contiguity argument](memory-consistency.md#contiguity-for-ram-table).
This argument establishes that all regions of constant `ramp` are contiguous.

Column `clk_di` is used to discount clock cycle differences that are equal to one.
It is necessary to establish the [inner sorting](memory-consistency.md#clock-jump-differences-and-inner-sorting) by `clk` within contiguous regions of constant `ramp`.

## Extension Columns

The RAM Table has 2 extension columns, `rppa` and `rpcjd`.

1. A Permutation Argument establishes that the rows in the RAM Table correspond to the rows of the [Processor Table](processor-table.md), after selecting for columns `clk`, `ramp`, `ramv` in both tables.
    The running product for this argument is contained in the `rppa` column.
1. In order to achieve [memory consistency](memory-consistency.md), a [multi-table Permutation Argument](memory-consistency.md#memory-like-tables) shows that all clock jump differences greater than one, from all memory-like tables (i.e., including the [OpStack Table](operational-stack-table.md) and the [JumpStack Table](jump-stack-table.md)), are contained in the `cjd` column of the [Processor Table](processor-table.md).
    The running product for this argument is contained in the `rpcjd` column.

## Sorting Rows

Up to order, the rows of the Hash Table in columns `clk`, `ramp`, `ramv` are identical to the rows in the [Processor Table](processor-table.md) in columns `clk`, `ramp`, and `ramv`.
In the Hash Table, the rows are sorted by memory address first, then by cycle counter.

Coming back to `iord`:
if the difference between `ramp` in row $i$ and row $i+1$ is 0, then `iord` in row $i$ is 0.
Otherwise, `iord` in row $i$ is the multiplicative inverse of the difference between `ramp` in row $i+1$ and `ramp` in row $i$.
In the last row, there being no next row, `iord` is 0.

An example of the mechanics can be found below.
For illustrative purposes only, we use four stack registers `st0` through `st3` in the example.
TritonVM has 16 stack registers, `st0` through `st15`.

Processor Table:

| `clk` | `ci`        | `nia`  | `st0` | `st1` | `st2` | `st3` | `ramp` | `ramv` |
|------:|:------------|:-------|------:|------:|------:|------:|-------:|-------:|
|     0 | `push`      | 5      |     0 |     0 |     0 |     0 |      0 |      0 |
|     1 | `push`      | 6      |     5 |     0 |     0 |     0 |      0 |      0 |
|     2 | `write_mem` | `pop`  |     6 |     5 |     0 |     0 |      5 |      0 |
|     3 | `pop`       | `pop`  |     6 |     5 |     0 |     0 |      5 |      6 |
|     4 | `pop`       | `push` |     5 |     0 |     0 |     0 |      5 |      0 |
|     5 | `push`      | 15     |     0 |     0 |     0 |     0 |      5 |      0 |
|     6 | `push`      | 16     |    15 |     0 |     0 |     0 |      5 |      0 |
|     7 | `write_mem` | `pop`  |    16 |    15 |     0 |     0 |     15 |      0 |
|     8 | `pop`       | `pop`  |    16 |    15 |     0 |     0 |     15 |     16 |
|     9 | `pop`       | `push` |    15 |     0 |     0 |     0 |     15 |     16 |
|    10 | `push`      | 5      |     0 |     0 |     0 |     0 |     15 |     16 |
|    11 | `push`      | 0      |     5 |     0 |     0 |     0 |     15 |     16 |
|    12 | `read_mem`  | `pop`  |     0 |     5 |     0 |     0 |      5 |      6 |
|    13 | `pop`       | `pop`  |     6 |     5 |     0 |     0 |      5 |      6 |
|    14 | `pop`       | `push` |     5 |     0 |     0 |     0 |      5 |      6 |
|    15 | `push`      | 15     |     0 |     0 |     0 |     0 |      5 |      6 |
|    16 | `push`      | 0      |    15 |     0 |     0 |     0 |      5 |      6 |
|    17 | `read_mem`  | `pop`  |     0 |    15 |     0 |     0 |     15 |     16 |
|    18 | `pop`       | `pop`  |    16 |    15 |     0 |     0 |     15 |     16 |
|    19 | `pop`       | `push` |    15 |     0 |     0 |     0 |     15 |     16 |
|    20 | `push`      | 5      |     0 |     0 |     0 |     0 |     15 |     16 |
|    21 | `push`      | 7      |     5 |     0 |     0 |     0 |     15 |     16 |
|    22 | `write_mem` | `pop`  |     7 |     5 |     0 |     0 |      5 |      6 |
|    23 | `pop`       | `pop`  |     7 |     5 |     0 |     0 |      5 |      7 |
|    24 | `pop`       | `push` |     5 |     0 |     0 |     0 |      5 |      7 |
|    25 | `push`      | 15     |     0 |     0 |     0 |     0 |      5 |      7 |
|    26 | `push`      | 0      |    15 |     0 |     0 |     0 |      5 |      7 |
|    27 | `read_mem`  | `push` |     0 |    15 |     0 |     0 |     15 |     16 |
|    28 | `push`      | 5      |    16 |    15 |     0 |     0 |     15 |     16 |
|    29 | `push`      | 0      |     5 |    16 |    15 |     0 |     15 |     16 |
|    30 | `read_mem`  | `halt` |     0 |     5 |    16 |    15 |      5 |      7 |
|    31 | `halt`      | `halt` |     7 |     5 |    16 |    15 |      5 |      7 |

RAM Table:

| `clk` | `ramp` | `ramv` |      `iord` |
|------:|-------:|-------:|------------:|
|     0 |      0 |      0 |             |
|     1 |      0 |      0 |  5${}^{-1}$ |
|     2 |      5 |      0 |             |
|     3 |      5 |      6 |             |
|     4 |      5 |      0 |             |
|     5 |      5 |      0 |             |
|     6 |      5 |      0 |             |
|    12 |      5 |      6 |             |
|    13 |      5 |      6 |             |
|    14 |      5 |      6 |             |
|    15 |      5 |      6 |             |
|    16 |      5 |      6 |             |
|    22 |      5 |      6 |             |
|    23 |      5 |      7 |             |
|    24 |      5 |      7 |             |
|    25 |      5 |      7 |             |
|    26 |      5 |      7 |             |
|    30 |      5 |      7 |             |
|    31 |      5 |      7 | 10${}^{-1}$ |
|     7 |     15 |      0 |             |
|     8 |     15 |     16 |             |
|     9 |     15 |     16 |             |
|    10 |     15 |     16 |             |
|    11 |     15 |     16 |             |
|    17 |     15 |     16 |             |
|    18 |     15 |     16 |             |
|    19 |     15 |     16 |             |
|    20 |     15 |     16 |             |
|    21 |     15 |     16 |             |
|    27 |     15 |     16 |             |
|    28 |     15 |     16 |             |
|    29 |     15 |     16 |             |

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

# Arithmetic Intermediate Representation

Let all household items (🪥, 🛁, etc.) be challenges, concretely evaluation points, supplied by the verifier.
Let all fruit & vegetables (🥝, 🥥, etc.) be challenges, concretely weights to compress rows, supplied by the verifier.
Both types of challenges are X-field elements, _i.e._, elements of $\mathbb{F}_{p^3}$.

## Initial Constraints

1. Cycle count `clk` is 0.
1. RAM pointer `ramp` is 0.
1. RAM value `ramv` is 0.
1. The first coefficient of the Bézout coefficient polynomial 0 `bcpc0` is 0.
1. The Bézout coefficient 0 `bc0` is 0.
1. The Bézout coefficient 1 `bc1` is equal to the first coefficient of the Bézout coefficient polynomial `bcpc1`.
1. The running product polynomial `rpp` starts with `🧼 - ramp`.
1. The formal derivative `fd` starts with 1.
1. The running product for clock jump differences `rpcjd` starts with 1.
1. The running product for the permutation argument with the Processor Table `rppa` has absorbed the first row with respect to challenges 🍍, 🍈, and 🍎 and indeterminate 🛋.

### Initial Constraints as Polynomials

1. `clk`
1. `ramp`
1. `ramv`
1. `bcpc0`
1. `bc0`
1. `bc1 - bcpc1`
1. `rpp - 🧼 + ramp`
1. `fd - 1`
1. `rpcjd - 1`
1. `rppa - 🛋 - 🍍·clk - 🍈·ramp - 🍎·ramv`

## Consistency Constraints

None.

## Transition Constraints

1. If `(ramp - ramp')` is 0, then `iord` is 0, else `iord` is the multiplicative inverse of `(ramp' - ramp)`.
1. If the `ramp` changes, then the new `ramv` must be 0.
1. If the `ramp` does not change and the `ramv` does change, then the cycle counter `clk` must increase by 1.
1. The Bézout coefficient polynomial coefficients are allowed to change only when the `ramp` changes.
1. The running product polynomial `rpp` accumulates a factor `(🧼 - ramp)` whenever `ramp` changes.
1. The clock difference inverse `clk_di` is the inverse-or-zero of the clock difference minus 1.
1. The running product for clock jump differences `rpcjd` accumulates a factor `(🚿 - clk' + clk)` whenever that difference is greater than one and `ramp` is the same.
1. The running product for the permutation argument with the Processor Table `rppa` absorbs the next row with respect to challenges 🍍, 🍈, and 🍎 and indeterminate 🛋.

Written as Disjunctive Normal Form, the same constraints can be expressed as:
1. `iord` is 0 or `iord` is the inverse of `(ramp' - ramp)`.
1. `(ramp' - ramp)` is zero or `iord` is the inverse of `(ramp' - ramp)`.
1. The `ramp` does not change or the new `ramv` is 0.
1. The `ramp` does change or the `ramv` does not change or the `clk` increases by 1.
1. `bcpc0' - bcpc0` is zero or `(ramp' - ramp)` is nonzero.
1. `bcpc1' - bcpc1` is zero or `(ramp' - ramp)` is nonzero.
1. `(ramp' - ramp)` is zero and `rpp' = rpp`; or `(ramp' - ramp)` is nonzero and `rpp' = rpp·(ramp'-🧼))` is zero.
1. the formal derivative `fd` applies the product rule of differentiation (as necessary).
1. Bézout coefficient 0 is evaluated correctly.
1. Bézout coefficient 1 is evaluated correctly.
1. `clk_di` is zero or the inverse of `(clk' - clk - 1)`.
1. `(clk' - clk - 1)` is zero or the inverse of `clk_di`.
1. `(clk' - clk - 1) ≠ 0` and `rpcjd' = rpcjd`;
    or `ramp' - ramp ≠ 0` and `rpcjd' = rpcjd`;
    or `(clk' - clk - 1) = 0` and `ramp' - ramp = 0` and `rpcjd' = rpcjd·(🚿 - clk' + clk)`.
1. `rppa' = rppa·(🛋 - 🍍·clk' + 🍈·ramp' + 🍎·ramv')`

### Transition Constraints as Polynomials

1. `iord·(iord·(ramp' - ramp) - 1)`
1. `(ramp' - ramp)·(iord·(ramp' - ramp) - 1)`
1. `(ramp' - ramp)·ramv'`
1. `(iord·(ramp' - ramp) - 1)·(ramv' - ramv)·(clk' - (clk + 1))`
1. `(iord·(ramp' - ramp) - 1)·(bcpc0' - bcpc0)`
1. `(iord·(ramp' - ramp) - 1)·(bcpc1' - bcpc1)`
1. `(iord·(ramp' - ramp) - 1)·(rpp' - rpp) + (ramp' - ramp)·(rpp' - rpp·(ramp'-🧼))`
1. `(iord·(ramp' - ramp) - 1)·(fd' - fd) + (ramp' - ramp)·(fd' - fd·(ramp'-🧼) - rpp)`
1. `(iord·(ramp' - ramp) - 1)·(bc0' - bc0) + (ramp' - ramp)·(bc0' - bc0·🧼 - bcpc0')`
1. `(iord·(ramp' - ramp) - 1)·(bc1' - bc1) + (ramp' - ramp)·(bc1' - bc1·🧼 - bcpc1')`
1. `clk_di·(clk_di·(clk' - clk - 1) - 1)`
1. `(clk' - clk - 1)·(clk_di·(clk' - clk - 1) - 1)`
1. `(clk' - clk - 1)·(rpcjd' - rpcjd) + (1 - (ramp' - ramp)·iord)·(rpcjd' - rpcjd) + (1 - (clk' - clk - 1)·clk_di)·ramp·(rpcjd' - rpcjd·(🚿 - clk' + clk))`
1. `rppa' - rppa·(🛋 - 🍍·clk' + 🍈·ramp' + 🍎·ramv')`

## Terminal Constraints

1. The Bézout relation holds between `rp`, `fd`, `bc0`, and `bc1`.

### Terminal Constraints as Polynomials

1. `rpp·bc0 + fd·bc1 - 1`
