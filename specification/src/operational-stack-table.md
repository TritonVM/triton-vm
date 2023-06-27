# Operational Stack Table

The operational stack is where the program stores simple elementary operations, function arguments, and pointers to important objects.
There are 16 registers (`st0` through `st15`) that the program can access directly.
These registers correspond to the top of the stack.
They are recorded in the [Processor Table](processor-table.md).

The rest of the operational stack is stored in a dedicated memory object called “operational stack underflow memory”.
It is initially empty.
The evolution of the underflow memory is recorded in the Operational Stack Table.

## Base Columns

The Operational Stack Table consists of 4 columns:
1. the cycle counter `clk`
1. the shrink stack indicator `shrink_stack`
1. the operational stack value `osv`, and
1. the operational stack pointer `osp`.

| Clock | Shrink Stack Indicator | Op Stack Pointer | Op Stack Value |
|:------|:-----------------------|:-----------------|:---------------|
| -     | -                      | -                | -              |

Columns `clk`, `shrink_stack`, `osp`, and `osv` correspond to columns `clk`, `ib1`, `osp`, and `osv` in the [Processor Table](processor-table.md), respectively.
A [Permutation Argument](permutation-argument.md) with the Processor Table establishes that, selecting the columns with these labels, the two tables' sets of rows are identical.

In order to guarantee [memory consistency](memory-consistency.md), the rows of the operational stack table are sorted by operational stack pointer `osp` first, cycle count `clk` second.
The mechanics are best illustrated by an example.
Observe how the malicious manipulation of the Op Stack Underflow Memory, the change of “42” into “99” in cycle 8, is detected:
The transition constraints of the Op Stack Table stipulate that if `osp` does not change, then `osv` can only change if the shrink stack indicator `shrink_stack` is set.
Consequently, row `[5, push, 9, 42]` being followed by row `[_, _, 9, 99]` violates the constraints.
The shrink stack indicator being correct is guaranteed by the Permutation Argument between Op Stack Table and the [Processor Table](processor-table.md).

For illustrative purposes only, we use four stack registers `st0` through `st3` in the example.
TritonVM has 16 stack registers, `st0` through `st15`.

Execution trace:

| `clk` | `ci`   | `nia`  | `st0` | `st1` | `st2` | `st3` | Op Stack Underflow Memory | `osp` | `osv` |
|------:|:-------|:-------|------:|------:|------:|------:|:--------------------------|------:|------:|
|     0 | `push` | 42     |     0 |     0 |     0 |     0 | [ ]                       |     4 |     0 |
|     1 | `push` | 43     |    42 |     0 |     0 |     0 | [0]                       |     5 |     0 |
|     2 | `push` | 44     |    43 |    42 |     0 |     0 | [0,0]                     |     6 |     0 |
|     3 | `push` | 45     |    44 |    43 |    42 |     0 | [0,0,0]                   |     7 |     0 |
|     4 | `push` | 46     |    45 |    44 |    43 |    42 | [0,0,0,0]                 |     8 |     0 |
|     5 | `push` | 47     |    46 |    45 |    44 |    43 | [42,0,0,0,0]              |     9 |    42 |
|     6 | `push` | 48     |    47 |    46 |    45 |    44 | [43,42,0,0,0,0]           |    10 |    43 |
|     7 | `nop`  | `pop`  |    48 |    47 |    46 |    45 | [44,43,42,0,0,0,0]        |    11 |    44 |
|     8 | `pop`  | `pop`  |    48 |    47 |    46 |    45 | [44,43,99,0,0,0,0]        |    11 |    44 |
|     9 | `pop`  | `pop`  |    47 |    46 |    45 |    44 | [43,99,0,0,0,0]           |    10 |    43 |
|    10 | `pop`  | `pop`  |    46 |    45 |    44 |    43 | [99,0,0,0,0]              |     9 |    99 |
|    11 | `pop`  | `push` |    45 |    44 |    43 |    99 | [0,0,0,0]                 |     8 |     0 |
|    12 | `push` | 77     |    44 |    43 |    99 |     0 | [0,0,0]                   |     7 |     0 |
|    13 | `swap` | 4      |    77 |    44 |    43 |    99 | [0,0,0,0]                 |     8 |     0 |
|    14 | `push` | 78     |    99 |    44 |    43 |    77 | [0,0,0,0]                 |     8 |     0 |
|    15 | `swap` | 4      |    78 |    99 |    44 |    43 | [77,0,0,0,0]              |     9 |    77 |
|    16 | `push` | 79     |    43 |    99 |    44 |    78 | [77,0,0,0,0]              |     9 |    77 |
|    17 | `pop`  | `pop`  |    79 |    43 |    99 |    44 | [78,77,0,0,0,0]           |    10 |    78 |
|    18 | `pop`  | `pop`  |    43 |    99 |    44 |    78 | [77,0,0,0,0]              |     9 |    77 |
|    19 | `pop`  | `pop`  |    99 |    44 |    78 |    77 | [0,0,0,0]                 |     8 |     0 |
|    20 | `pop`  | `pop`  |    44 |    78 |    77 |     0 | [0,0,0]                   |     7 |     0 |
|    21 | `pop`  | `pop`  |    78 |    77 |     0 |     0 | [0,0]                     |     6 |     0 |
|    22 | `pop`  | `pop`  |    77 |     0 |     0 |     0 | [0]                       |     5 |     0 |
|    23 | `pop`  | `pop`  |     0 |     0 |     0 |     0 | [ ]                       |     4 |     0 |
|    24 | `pop`  | `nop`  |     0 |     0 |     0 |     0 | 💥                         |     3 |     0 |

Operational Stack Table:

| `clk` | `shrink_stack` | (comment) | `osp` | `osv` |
|------:|---------------:|:----------|------:|------:|
|     0 |              0 | (`push`)  |     4 |     0 |
|    23 |              1 | (`pop`)   |     4 |     0 |
|     1 |              0 | (`push`)  |     5 |     0 |
|    22 |              1 | (`pop`)   |     5 |     0 |
|     2 |              0 | (`push`)  |     6 |     0 |
|    21 |              1 | (`pop`)   |     6 |     0 |
|     3 |              0 | (`push`)  |     7 |     0 |
|    12 |              0 | (`push`)  |     7 |     0 |
|    20 |              1 | (`pop`)   |     7 |     0 |
|     4 |              0 | (`push`)  |     8 |     0 |
|    11 |              1 | (`pop`)   |     8 |     0 |
|    13 |              0 | (`swap`)  |     8 |     0 |
|    14 |              0 | (`push`)  |     8 |     0 |
|    19 |              1 | (`pop`)   |     8 |     0 |
|     5 |              0 | (`push`)  |     9 |    42 |
|    10 |              1 | (`pop`)   |     9 |    99 |
|    15 |              0 | (`swap`)  |     9 |    77 |
|    16 |              0 | (`push`)  |     9 |    77 |
|    18 |              1 | (`pop`)   |     9 |    77 |
|     6 |              0 | (`push`)  |    10 |    43 |
|     9 |              1 | (`pop`)   |    10 |    43 |
|    17 |              1 | (`pop`)   |    10 |    78 |
|     7 |              0 | (`nop`)   |    11 |    44 |
|     8 |              1 | (`pop`)   |    11 |    44 |

## Extension Columns

The Op Stack Table has 2 extension columns, `rppa` and `ClockJumpDifferenceLookupClientLogDerivative`.

1. A Permutation Argument establishes that the rows of the Op Stack Table correspond to the rows of the [Processor Table](processor-table.md).
  The running product for this argument is contained in the `rppa` column.
1. In order to achieve [memory consistency](memory-consistency.md), a [Lookup Argument](lookup-argument.md) shows that all clock jump differences are contained in the `clk` column of the [Processor Table](processor-table.md).
  The logarithmic derivative for this argument is contained in the `ClockJumpDifferenceLookupClientLogDerivative` column.

## Padding

A padding row is a direct copy of the Op Stack Table's row with the highest value for column `clk`, called template row, with the exception of the cycle count column `clk`.
In a padding row, the value of column `clk` is 1 greater than the value of column `clk` in the template row.
The padding row is inserted right below the template row.
These steps are repeated until the desired padded height is reached.
In total, above steps ensure that the Permutation Argument between the Op Stack Table and the [Processor Table](processor-table.md) holds up.

## Memory-Consistency

Memory-consistency follows from two more primitive properties:

1. Contiguity of regions of constant memory pointer.
  Since the memory pointer for the Op Stack table, `osp` can change by at most one per cycle, it is possible to enforce a full sorting using AIR constraints.
2. Correct inner-sorting within contiguous regions.
  Specifically, the rows within each contiguous region of constant memory pointer should be sorted for clock cycle.
  This property is established by the clock jump difference [Lookup Argument](lookup-argument.md).
  In a nutshell, every difference of consecutive clock cycles that occurs within one contiguous block of constant memory pointer is shown itself to be a valid clock cycle through a separate cross-table argument.

# Arithmetic Intermediate Representation

Let all household items (🪥, 🛁, etc.) be challenges, concretely evaluation points, supplied by the verifier.
Let all fruit & vegetables (🥝, 🥥, etc.) be challenges, concretely weights to compress rows, supplied by the verifier.
Both types of challenges are X-field elements, _i.e._, elements of $\mathbb{F}_{p^3}$.

## Initial Constraints

1. `clk` is 0
1. `osv` is 0.
1. `osp` is the number of available stack registers, _i.e._, 16.
1. The running product for the permutation argument with the Processor Table `rppa` starts off having accumulated the first row with respect to challenges 🍋, 🍊, 🍉, and 🫒 and indeterminate 🪤.
1. The logarithmic derivative for the clock jump difference lookup `ClockJumpDifferenceLookupClientLogDerivative` is 0.

### Initial Constraints as Polynomials

1. `clk`
1. `osv`
1. `osp - 16`
1. `rppa - (🪤 - 🍋·clk - 🍊·ib1 - 🍉·osp - 🫒osv)`
1. `ClockJumpDifferenceLookupClientLogDerivative`

## Consistency Constraints

None.

## Transition Constraints

1.
  - the `osp` increases by 1, *or*
  - the `osp` does not change AND the `osv` does not change, *or*
  - the `osp` does not change AND the shrink stack indicator `shrink_stack` is 1.
1. The running product for the permutation argument with the Processor Table `rppa` absorbs the next row with respect to challenges 🍋, 🍊, 🍉, and 🫒 and indeterminate 🪤.
1. If the op stack pointer `osp` does not change, then the logarithmic derivative for the clock jump difference lookup `ClockJumpDifferenceLookupClientLogDerivative` accumulates a factor `(clk' - clk)` relative to indeterminate 🪞.
  Otherwise, it remains the same.

Written as Disjunctive Normal Form, the same constraints can be expressed as:

1.
  - the `osp` increases by 1 or the `osp` does not change
  - the `osp` increases by 1 or the `osv` does not change or the shrink stack indicator `shrink_stack` is 1
1. `rppa' = rppa·(🪤 - 🍋·clk' - 🍊·ib1' - 🍉·osp' - 🫒osv')`
1. - the `osp` changes or the logarithmic derivative accumulates a summand, and
   - the `osp` does not change or the logarithmic derivative does not change.

### Transition Constraints as Polynomials

1. `(osp' - (osp + 1))·(osp' - osp)`
1. `(osp' - (osp + 1))·(osv' - osv)·(1 - shrink_stack)`
1. `rppa' - rppa·(🪤 - 🍋·clk' - 🍊·ib1' - 🍉·osp' - 🫒osv')`
1. `(osp' - (osp + 1))·((ClockJumpDifferenceLookupClientLogDerivative' - ClockJumpDifferenceLookupClientLogDerivative) · (🪞 - clk' + clk) - 1)`<br />
   `+ (osp' - osp)·(ClockJumpDifferenceLookupClientLogDerivative' - ClockJumpDifferenceLookupClientLogDerivative)`

## Terminal Constraints

None.
