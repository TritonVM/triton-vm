# Random Access Memory Table

The RAM is accessible through `read_mem` and `write_mem` commands.
The RAM Table has four columns:
the cycle counter `clk`, RAM address pointer `ramp`, the value of the memory at that address `ramv`, and helper variable `InverseOfRampDifference`.
Columns `clk` and `ramv` correspond to the columns of the same name in the Processor Table.
Column `ramp` corresponds to the Processor Table's column `st1`.
Column `InverseOfRampDifference` helps with detecting a change of `ramp` across two RAM Table rows.
In order to explain `InverseOfRampDifference` more easily, we first explain how to sort the RAM Table's rows.

Up to order, the rows of the Hash Table in columns `clk`, `ramp`, `ramv` are identical to the rows in the Processor Table in columns `clk`, `st1`, and `ramv`.
In the Hash Table, the rows are sorted by memory address first, then by cycle counter.

Coming back to `InverseOfRampDifference`:
if the difference between `ramp` in row $i$ and row $i+1$ is 0, then `InverseOfRampDifference` in row $i$ is 0.
Otherwise, `InverseOfRampDifference` in row $i$ is the multiplicative inverse of the difference between `ramp` in row $i+1$ and `ramp` in row $i$.
In the last row, there being no next row, `InverseOfRampDifference` is 0.

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

| `clk` | `ramp` (≘`st1`) | `ramv` | `InverseOfRampDifference` |
|------:|----------------:|-------:|--------------------------:|
|     0 |               0 |      0 |                         0 |
|     1 |               0 |      0 |                         0 |
|     4 |               0 |      0 |                         0 |
|     5 |               0 |      0 |                         0 |
|     6 |               0 |      0 |                         0 |
|     9 |               0 |      0 |                         0 |
|    10 |               0 |      0 |                         0 |
|    11 |               0 |      0 |                         0 |
|    14 |               0 |      0 |                         0 |
|    15 |               0 |      0 |                         0 |
|    16 |               0 |      0 |                         0 |
|    19 |               0 |      0 |                         0 |
|    20 |               0 |      0 |                         0 |
|    21 |               0 |      0 |                         0 |
|    24 |               0 |      0 |                         0 |
|    25 |               0 |      0 |                         0 |
|    26 |               0 |      0 |                  $5^{-1}$ |
|     2 |               5 |      0 |                         0 |
|     3 |               5 |      6 |                         0 |
|    12 |               5 |      6 |                         0 |
|    13 |               5 |      6 |                         0 |
|    22 |               5 |      6 |                         0 |
|    23 |               5 |      7 |                         0 |
|    30 |               5 |      7 |                         0 |
|    31 |               5 |      7 |                 $10^{-1}$ |
|     7 |              15 |      0 |                         0 |
|     8 |              15 |     16 |                         0 |
|    17 |              15 |     16 |                         0 |
|    18 |              15 |     16 |                         0 |
|    27 |              15 |     16 |                         0 |
|    28 |              15 |     16 |                         1 |
|    29 |              16 |      0 |                         0 |


## Padding

A padding row is a direct copy of the RAM Table's row with the highest value for column `clk`, called template row, with the exception of the cycle count column `clk`.
In a padding row, the value of column `clk` is 1 greater than the value of column `clk` in the template row.
The padding row is inserted right below the template row.
Finally, the value of column `InverseOfRampDifference` is set to 0 in the template row.
These steps are repeated until the desired padded height is reached.
In total, above steps ensure that the Permutation Argument between the RAM Table and the [Processor Table](processor-table.md) holds up.

## Initial Constraints

1. Cycle count `clk` is 0.
1. RAM pointer `ramp` is 0.
1. RAM value `ramv` is 0.

**Initial Constraints as Polynomials**

1. `clk`
1. `ramp`
1. `ramv`

## Consistency Constraints

None.

## Transition Constraints

1. If `(ramp - ramp')` is 0, then `InverseOfRampDifference` is 0, else `InverseOfRampDifference` is the multiplicative inverse of `(ramp' - ramp)`.
1. If the `ramp` changes, then the new `ramv` must be 0.
1. If the `ramp` does not change and the `ramv` does change, then the cycle counter `clk` must increase by 1.

Written as Disjunctive Normal Form, the same constraints can be expressed as:
1. `InverseOfRampDifference` is 0 or `InverseOfRampDifference` is the inverse of `(ramp' - ramp)`.
1. `(ramp' - ramp)` is zero or `InverseOfRampDifference` is the inverse of `(ramp' - ramp)`.
1. The `ramp` does not change or the new `ramv` is 0.
1. The `ramp` does change or the `ramv` does not change or the `clk` increases by 1.

**Transition Constraints as Polynomials**

1. `InverseOfRampDifference·(InverseOfRampDifference·(ramp' - ramp) - 1)`
1. `(ramp' - ramp)·(InverseOfRampDifference·(ramp' - ramp) - 1)`
1. `(ramp' - ramp)·ramv'`
1. `(InverseOfRampDifference·(ramp' - ramp) - 1)·(ramv' - ramv)·(clk' - (clk + 1))`

## Terminal Constraints

None.

## Relations to Other Tables

1. A Permutation Argument establishes that the rows in the RAM Table correspond to the rows of the [Processor Table](processor-table.md).
