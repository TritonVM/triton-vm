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

## Main Columns

The RAM Table has 7 main columns:
1. the cycle counter `clk`,
1. the executed `instruction_type` – 0 for “write”, 1 for “read”, 2 for padding rows,
1. RAM pointer `ram_pointer`,
1. RAM value `ram_value`,
1. helper variable "inverse of `ram_pointer` difference" `iord`,
1. Bézout coefficient polynomial coefficient 0 `bcpc0`,
1. Bézout coefficient polynomial coefficient 1 `bcpc1`,

Column `iord` helps with detecting a change of `ram_pointer` across two RAM Table rows.
The function of `iord` is best explained in the context of sorting the RAM Table's rows, which is what the next section is about.

The Bézout coefficient polynomial coefficients `bcpc0` and `bcpc1` represent the coefficients of polynomials that are needed for the [contiguity argument](contiguity-of-memory-pointer-regions.md#contiguity-for-ram-table).
This argument establishes that all regions of constant `ram_pointer` are contiguous.

## Auxiliary Columns

The RAM Table has 6 auxiliary columns:
1. `RunningProductOfRAMP`, accumulating next row's `ram_pointer` as a root whenever `ram_pointer` changes between two rows,
1. `FormalDerivative`, the (evaluated) formal derivative of `RunningProductOfRAMP`,
1. `BezoutCoefficient0`, the (evaluated) polynomial with main column `bcpc0` as coefficients,
1. `BezoutCoefficient1`, the (evaluated) polynomial with main column `bcpc1` as coefficients,
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

The row used for padding the RAM Table is its last row, with the `instruction_type` set to 2.

If the RAM Table is empty, the all-zero row with the following modifications is used instead:
- `instruction_type` is set to 2, and
- `bcpc1` is set to 1.

This ensures that the Contiguity Argument works correctly in the absence of any actual contiguous RAM pointer region. 

The padding row is inserted below the RAM Table until the desired height is reached.

## Row Permutation Argument

The permutation argument with the [Processor Table](processor-table.md) establishes that the RAM Table's rows correspond to the Processor Table's sent and received RAM values, at the correct cycle counter and RAM address.

# Arithmetic Intermediate Representation

Let all household items (🪥, 🛁, etc.) be challenges, concretely evaluation points, supplied by the verifier.
Let all fruit & vegetables (🥝, 🥥, etc.) be challenges, concretely weights to compress rows, supplied by the verifier.
Both types of challenges are X-field elements, _i.e._, elements of $\mathbb{F}_{p^3}$.

## Initial Constraints

1. The first coefficient of the Bézout coefficient polynomial 0 `bcpc0` is 0.
1. The Bézout coefficient 0 `bc0` is 0.
1. The Bézout coefficient 1 `bc1` is equal to the first coefficient of the Bézout coefficient polynomial `bcpc1`.
1. The running product polynomial `RunningProductOfRAMP` starts with `🧼 - ram_pointer`.
1. The formal derivative starts with 1.
1. If the first row is not a padding row, the running product for the permutation argument with the Processor Table `RunningProductPermArg` has absorbed the first row with respect to challenges 🍍, 🍈, 🍎, and 🌽 and indeterminate 🛋.<br />
    Else, the running product for the permutation argument with the Processor Table `RunningProductPermArg` is 1.
1. The logarithmic derivative for the clock jump difference lookup `ClockJumpDifferenceLookupClientLogDerivative` is 0.

### Initial Constraints as Polynomials

1. `bcpc0`
1. `bc0`
1. `bc1 - bcpc1`
1. `RunningProductOfRAMP - 🧼 + ram_pointer`
1. `FormalDerivative - 1`
1. `(RunningProductPermArg - 🛋 - 🍍·clk - 🍈·ram_pointer - 🍎·ram_value - 🌽·instruction_type)·(instruction_type - 2)`<br />
    `(RunningProductPermArg - 1)·(instruction_type - 1)·(instruction_type - 0)`
1. `ClockJumpDifferenceLookupClientLogDerivative`

## Consistency Constraints

None.

## Transition Constraints

1. If the current row is a padding row, then the next row is a padding row.
1. The “inverse of `ram_pointer` difference” `iord` is 0 or `iord` is the inverse of the difference between current and next row's `ram_pointer`.
1. The `ram_pointer` changes or `iord` is the inverse of the difference between current and next row's `ram_pointer`.
1. The `ram_pointer` changes or `instruction_type` is “write” or the `ram_value` remains unchanged.
1. The `bcbp0` changes if and only if the `ram_pointer` changes.
1. The `bcbp1` changes if and only if the `ram_pointer` changes.
1. If the `ram_pointer` changes, the `RunningProductOfRAMP` accumulates next `ram_pointer`.<br />
    Otherwise, it remains unchanged.
1. If the `ram_pointer` changes, the `FormalDerivative` updates under the product rule of differentiation.<br />
    Otherwise, it remains unchanged.
1. If the `ram_pointer` changes, Bézout coefficient 0 `bc0` updates according to the running evaluation rules with respect to `bcpc0`.<br />
    Otherwise, it remains unchanged.
1. If the `ram_pointer` changes, Bézout coefficient 1 `bc1` updates according to the running evaluation rules with respect to `bcpc1`.<br />
    Otherwise, it remains unchanged.
1. If the next row is not a padding row, the `RunningProductPermArg` accumulates the next row.<br />
    Otherwise, it remains unchanged.
1. If the `ram_pointer` does not change and the next row is not a padding row, the `ClockJumpDifferenceLookupClientLogDerivative` accumulates the difference of `clk`.<br />
    Otherwise, it remains unchanged.

### Transition Constraints as Polynomials

1. `(instruction_type - 0)·(instruction_type - 1)·(instruction_type' - 2)`
1. `(iord·(ram_pointer' - ram_pointer) - 1)·iord`
1. `(iord·(ram_pointer' - ram_pointer) - 1)·(ram_pointer' - ram_pointer)`
1. `(iord·(ram_pointer' - ram_pointer) - 1)·(instruction_type - 0)·(ram_value' - ram_value)`
1. `(iord·(ram_pointer' - ram_pointer) - 1)·(bcpc0' - bcpc0)`
1. `(iord·(ram_pointer' - ram_pointer) - 1)·(bcpc1' - bcpc1)`
1. `(iord·(ram_pointer' - ram_pointer) - 1)·(RunningProductOfRAMP' - RunningProductOfRAMP)`<br />
    ` + (ram_pointer' - ram_pointer)·(RunningProductOfRAMP' - RunningProductOfRAMP·(ram_pointer'-🧼))`
1. `(iord·(ram_pointer' - ram_pointer) - 1)·(FormalDerivative' - FormalDerivative)`<br />
    `+ (ram_pointer' - ram_pointer)·(FormalDerivative' - FormalDerivative·(ram_pointer'-🧼) - RunningProductOfRAMP)`
1. `(iord·(ram_pointer' - ram_pointer) - 1)·(bc0' - bc0)`<br />
     `+ (ram_pointer' - ram_pointer)·(bc0' - bc0·🧼 - bcpc0')`
1. `(iord·(ram_pointer' - ram_pointer) - 1)·(bc1' - bc1)`<br />
     `+ (ram_pointer' - ram_pointer)·(bc1' - bc1·🧼 - bcpc1')`
1. `(RunningProductPermArg' - RunningProductPermArg·(🛋 - 🍍·clk' - 🍈·ram_pointer' - 🍎·ram_value' - 🌽·instruction_type'))·(instruction_type' - 2)`<br />
    `(RunningProductPermArg' - RunningProductPermArg)·(instruction_type - 1)·(instruction_type - 0))`
1. `(iord·(ram_pointer' - ram_pointer) - 1)·(instruction_type' - 2)·((ClockJumpDifferenceLookupClientLogDerivative' - ClockJumpDifferenceLookupClientLogDerivative) · (🪞 - clk' + clk) - 1)`<br />
    `+ (ram_pointer' - ram_pointer)·(ClockJumpDifferenceLookupClientLogDerivative' - ClockJumpDifferenceLookupClientLogDerivative)`<br />
    `+ (instruction_type' - 1)·(instruction_type' - 0)·(ClockJumpDifferenceLookupClientLogDerivative' - ClockJumpDifferenceLookupClientLogDerivative)`

## Terminal Constraints

1. The Bézout relation holds between `RunningProductOfRAMP`, `FormalDerivative`, `bc0`, and `bc1`.

### Terminal Constraints as Polynomials

1. `RunningProductOfRAMP·bc0 + FormalDerivative·bc1 - 1`
