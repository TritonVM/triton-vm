# Random Access Memory Table

The purpose of the RAM Table is to ensure that the RAM is accessed in a consistent manner.
That is, all mutation of RAM happens explicitly through instruction `write_mem`,
and any invocations of instruction `read_mem` return the values last written.

The fundamental principle is identical to that of the [Op Stack Table](operational-stack-table.md).
The main difference is the absence of a dedicated stack pointer.
Instead, op stack element `st0` is used as RAM pointer `ram_pointer`. 

If some RAM address is read from before it is written to, the corresponding value is not determined.
This is one of interfaces for non-deterministic input in Triton VM.
Consecutive reads from any such address always returns the same value (until overwritten via `write_mem`).  

## Base Columns

The RAM Table has 7 base columns:
1. the cycle counter `clk`,
1. the executed `instruction_type` – 0 for “write”, 1 for “read”, 2 for padding rows,
1. RAM pointer `ram_pointer`,
1. RAM value `ram_value`,
1. helper variable "inverse of `ram_pointer` difference" `iord`,
1. Bézout coefficient polynomial coefficient 0 `bcpc0`,
1. Bézout coefficient polynomial coefficient 1 `bcpc1`,

Column `iord` helps with detecting a change of `ram_pointer` across two RAM Table rows.
The function of `iord` is best explained in the context of sorting the RAM Table's rows, which is what the next section is about.

The Bézout coefficient polynomial coefficients `bcpc0` and `bcpc1` represent the coefficients of polynomials that are needed for the [contiguity argument](memory-consistency.md#contiguity-for-ram-table).
This argument establishes that all regions of constant `ram_pointer` are contiguous.

## Extension Columns

The RAM Table has 6 extension columns:
1. `RunningProductOfRAMP`, accumulating next row's `ram_pointer` as a root whenever `ram_pointer` changes between two rows,
1. `FormalDerivative`, the (evaluated) formal derivative of `RunningProductOfRAMP`,
1. `BezoutCoefficient0`, the (evaluated) polynomial with base column `bcpc0` as coefficients,
1. `BezoutCoefficient1`, the (evaluated) polynomial with base column `bcpc1` as coefficients,
1. `RunningProductPermArg`, the [Permutation Argument](permutation-argument.md) with the [Processor Table](processor-table.md), and
1. `ClockJumpDifferenceLookupClientLogDerivative`, part of [memory consistency](clock-jump-differences-and-inner-sorting.md).

Columns `RunningProductOfRAMP`, `FormalDerivative`, `BezoutCoefficient0`, and `BezoutCoefficient1` are part of the [Contiguity Argument](contiguity-of-memory-pointer-regions.md).

## Sorting Rows

In the Hash Table, the rows are arranged such that they

1. form contiguous regions of `ram_pointer`, and
1. are sorted by cycle counter `clk` within each such region.

One way to achieve this is to sort by `ram_pointer` first, `clk` second.

Coming back to `iord`:
if the difference between `ram_pointer` in row $i$ and row $i+1$ is 0, then `iord` in row $i$ is 0.
Otherwise, `iord` in row $i$ is the multiplicative inverse of the difference between `ram_pointer` in row $i+1$ and `ram_pointer` in row $i$.
In the last row, there being no next row, `iord` is 0.

An example of the mechanics can be found below.
To increase visual clarity, stack registers holding value “0” are represented by an empty cell.
For illustrative purposes only, we use six stack registers `st0` through `st5` in the example.
Triton VM has 16 stack registers, `st0` through `st15`.

Processor Table:

| clk | ci        | nia | st0 | st1 | st2 | st3 | st4 | st5 |
|----:|:----------|:----|----:|----:|----:|----:|----:|----:|
|   0 | push      | 20  |     |     |     |     |     |     |
|   1 | push      | 100 |  20 |     |     |     |     |     |
|   2 | write_mem | 1   | 100 |  20 |     |     |     |     |
|   3 | pop       | 1   | 101 |     |     |     |     |     |
|   4 | push      | 5   |     |     |     |     |     |     |
|   5 | push      | 6   |   5 |     |     |     |     |     |
|   6 | push      | 7   |   6 |   5 |     |     |     |     |
|   7 | push      | 8   |   7 |   6 |   5 |     |     |     |
|   8 | push      | 9   |   8 |   7 |   6 |   5 |     |     |
|   9 | push      | 42  |   9 |   8 |   7 |   6 |   5 |     |
|  10 | write_mem | 5   |  42 |   9 |   8 |   7 |   6 |   5 |
|  11 | pop       | 1   |  47 |     |     |     |     |     |
|  12 | push      | 42  |     |     |     |     |     |     |
|  13 | read_mem  | 1   |  42 |     |     |     |     |     |
|  14 | pop       | 2   |  41 |   9 |     |     |     |     |
|  15 | push      | 45  |     |     |     |     |     |     |
|  16 | read_mem  | 3   |  45 |     |     |     |     |     |
|  17 | pop       | 4   |  42 |   8 |   7 |   6 |     |     |
|  18 | push      | 17  |     |     |     |     |     |     |
|  19 | push      | 18  |  17 |     |     |     |     |     |
|  20 | push      | 19  |  18 |  17 |     |     |     |     |
|  21 | push      | 43  |  19 |  18 |  17 |     |     |     |
|  22 | write_mem | 3   |  43 |  19 |  18 |  17 |     |     |
|  23 | pop       | 1   |  46 |     |     |     |     |     |
|  24 | push      | 46  |     |     |     |     |     |     |
|  25 | read_mem  | 5   |  46 |     |     |     |     |     |
|  26 | pop       | 1   |  41 |   9 |  19 |  18 |  17 |   5 |
|  27 | pop       | 5   |   9 |  19 |  18 |  17 |   5 |     |
|  28 | push      | 42  |     |     |     |     |     |     |
|  29 | read_mem  | 1   |  42 |     |     |     |     |     |
|  30 | pop       | 2   |  41 |   9 |     |     |     |     |
|  31 | push      | 100 |     |     |     |     |     |     |
|  32 | read_mem  | 1   | 100 |     |     |     |     |     |
|  33 | pop       | 2   |  99 |  20 |     |     |     |     |
|  34 | halt      |     |     |     |     |     |     |     |

RAM Table:

| clk | type  | pointer | value |      iord |
|----:|:------|--------:|------:|----------:|
|  10 | write |      42 |     9 |         0 |
|  13 | read  |      42 |     9 |         0 |
|  25 | read  |      42 |     9 |         0 |
|  29 | read  |      42 |     9 |         1 |
|  10 | write |      43 |     8 |         0 |
|  16 | read  |      43 |     8 |         0 |
|  22 | write |      43 |    19 |         0 |
|  25 | read  |      43 |    19 |         1 |
|  10 | write |      44 |     7 |         0 |
|  16 | read  |      44 |     7 |         0 |
|  22 | write |      44 |    18 |         0 |
|  25 | read  |      44 |    18 |         1 |
|  10 | write |      45 |     6 |         0 |
|  16 | read  |      45 |     6 |         0 |
|  22 | write |      45 |    17 |         0 |
|  25 | read  |      45 |    17 |         1 |
|  10 | write |      46 |     5 |         0 |
|  25 | read  |      46 |     5 | $54^{-1}$ |
|   2 | write |     100 |    20 |         0 |
|  32 | read  |     100 |    20 |         0 |

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

1. The first coefficient of the Bézout coefficient polynomial 0 `bcpc0` is 0.
1. The Bézout coefficient 0 `bc0` is 0.
1. The Bézout coefficient 1 `bc1` is equal to the first coefficient of the Bézout coefficient polynomial `bcpc1`.
1. The running product polynomial `rpp` starts with `🧼 - ramp`.
1. The formal derivative `fd` starts with 1.
1. The running product for the permutation argument with the Processor Table `rppa` has absorbed the first row with respect to challenges 🍍, 🍈, 🍎, and 🌽 and indeterminate 🛋.
1. The logarithmic derivative for the clock jump difference lookup `ClockJumpDifferenceLookupClientLogDerivative` is 0.

### Initial Constraints as Polynomials

1. `bcpc0`
1. `bc0`
1. `bc1 - bcpc1`
1. `rpp - 🧼 + ramp`
1. `fd - 1`
1. `rppa - 🛋 - 🍍·clk - 🍈·ramp - 🍎·ramv - 🌽·previous_instruction`
1. `ClockJumpDifferenceLookupClientLogDerivative`

## Consistency Constraints

None.

## Transition Constraints

1. If `(ramp - ramp')` is 0, then `iord` is 0, else `iord` is the multiplicative inverse of `(ramp' - ramp)`.
1. If the `ramp` does not change and `previous_instruction` in the next row is not `write_mem`, then the RAM value `ramv` does not change.
1. The Bézout coefficient polynomial coefficients are allowed to change only when the `ramp` changes.
1. The running product polynomial `rpp` accumulates a factor `(🧼 - ramp)` whenever `ramp` changes.
1. The running product for the permutation argument with the Processor Table `rppa` absorbs the next row with respect to challenges 🍍, 🍈, 🍎, and 🌽 and indeterminate 🛋.
1. If the RAM pointer `ramp` does not change, then the logarithmic derivative for the clock jump difference lookup `ClockJumpDifferenceLookupClientLogDerivative` accumulates a factor `(clk' - clk)` relative to indeterminate 🪞.
  Otherwise, it remains the same.

Written as Disjunctive Normal Form, the same constraints can be expressed as:
1. `iord` is 0 or `iord` is the inverse of `(ramp' - ramp)`.
1. `(ramp' - ramp)` is zero or `iord` is the inverse of `(ramp' - ramp)`.
1. `(ramp' - ramp)` non-zero or `previous_instruction'` is `opcode(write_mem)` or `ramv'` is `ramv`.
1. `bcpc0' - bcpc0` is zero or `(ramp' - ramp)` is nonzero.
1. `bcpc1' - bcpc1` is zero or `(ramp' - ramp)` is nonzero.
1. `(ramp' - ramp)` is zero and `rpp' = rpp`; or `(ramp' - ramp)` is nonzero and `rpp' = rpp·(ramp'-🧼))` is zero.
1. the formal derivative `fd` applies the product rule of differentiation (as necessary).
1. Bézout coefficient 0 is evaluated correctly.
1. Bézout coefficient 1 is evaluated correctly.
1. `rppa' = rppa·(🛋 - 🍍·clk' - 🍈·ramp' - 🍎·ramv' - 🌽·previous_instruction')`
1. - the `ramp` changes or the logarithmic derivative accumulates a summand, and
   - the `ramp` does not change or the logarithmic derivative does not change.

### Transition Constraints as Polynomials

1. `iord·(iord·(ramp' - ramp) - 1)`
1. `(ramp' - ramp)·(iord·(ramp' - ramp) - 1)`
1. `(1 - iord·(ramp' - ramp))·(previous_instruction - opcode(write_mem))·(ramv' - ramv)`
1. `(iord·(ramp' - ramp) - 1)·(bcpc0' - bcpc0)`
1. `(iord·(ramp' - ramp) - 1)·(bcpc1' - bcpc1)`
1. `(iord·(ramp' - ramp) - 1)·(rpp' - rpp) + (ramp' - ramp)·(rpp' - rpp·(ramp'-🧼))`
1. `(iord·(ramp' - ramp) - 1)·(fd' - fd) + (ramp' - ramp)·(fd' - fd·(ramp'-🧼) - rpp)`
1. `(iord·(ramp' - ramp) - 1)·(bc0' - bc0) + (ramp' - ramp)·(bc0' - bc0·🧼 - bcpc0')`
1. `(iord·(ramp' - ramp) - 1)·(bc1' - bc1) + (ramp' - ramp)·(bc1' - bc1·🧼 - bcpc1')`
1. `rppa' - rppa·(🛋 - 🍍·clk' - 🍈·ramp' - 🍎·ramv' - 🌽·previous_instruction')`
1. `(iord·(ramp' - ramp) - 1)·((ClockJumpDifferenceLookupClientLogDerivative' - ClockJumpDifferenceLookupClientLogDerivative) · (🪞 - clk' + clk) - 1)`<br />
   `+ (ramp' - ramp)·(ClockJumpDifferenceLookupClientLogDerivative' - ClockJumpDifferenceLookupClientLogDerivative)`

## Terminal Constraints

1. The Bézout relation holds between `rp`, `fd`, `bc0`, and `bc1`.

### Terminal Constraints as Polynomials

1. `rpp·bc0 + fd·bc1 - 1`
