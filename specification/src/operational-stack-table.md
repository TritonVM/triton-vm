# Operational Stack Table

The operational stack is where the program stores simple elementary operations, function arguments, and pointers to important objects.
There are 16 registers (`st0` through `st15`) that the program can access directly.
These registers correspond to the top of the stack.
The rest of the operational stack is stored in a dedicated memory object called Operational Stack Underflow Memory.
It is initially empty.

## Base Columns

The Operational Stack Table consists of 5 columns:
1. the cycle counter `clk`
1. the inverse-or-zero of the difference between two consecutive `clk` values minus one, `clk_di`,
1. the shrink stack indicator `shrink_stack`
1. the operational stack value `osv`, and
1. the operational stack pointer `osp`.

| Clock | Inverse of Clock Difference Minus One | Shrink Stack Indicator | Op Stack Pointer | Op Stack Value |
|:------|:--------------------------------------|:-----------------------|:-----------------|:---------------|
| -     | -                                     | -                      | -                | -              |

Columns `clk`, `shrink_stack`, `osp`, and `osv` correspond to columns `clk`, `ib1`, `osp`, and `osv` in the [Processor Table](processor-table.md), respectively.
A Permutation Argument with the Processor Table establishes that, selecting the columns with these labels, the two tables' sets of rows are identical.

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

| `clk` |               `clk_di` | `shrink_stack` | (comment) | `osp` | `osv` |
|------:|-----------------------:|---------------:|:----------|------:|------:|
|     0 |  (23 - 0 - 1)${}^{-1}$ |              0 | (`push`)  |     4 |     0 |
|    23 |  (1 - 23 - 1)${}^{-1}$ |              1 | (`pop`)   |     4 |     0 |
|     1 |  (22 - 1 - 1)${}^{-1}$ |              0 | (`push`)  |     5 |     0 |
|    22 |  (2 - 22 - 1)${}^{-1}$ |              1 | (`pop`)   |     5 |     0 |
|     2 |  (21 - 2 - 1)${}^{-1}$ |              0 | (`push`)  |     6 |     0 |
|    21 |  (3 - 21 - 1)${}^{-1}$ |              1 | (`pop`)   |     6 |     0 |
|     3 |  (12 - 3 - 1)${}^{-1}$ |              0 | (`push`)  |     7 |     0 |
|    12 | (20 - 12 - 1)${}^{-1}$ |              0 | (`push`)  |     7 |     0 |
|    20 |  (4 - 20 - 1)${}^{-1}$ |              1 | (`pop`)   |     7 |     0 |
|     4 |  (11 - 4 - 1)${}^{-1}$ |              0 | (`push`)  |     8 |     0 |
|    11 | (13 - 11 - 1)${}^{-1}$ |              1 | (`pop`)   |     8 |     0 |
|    13 | (14 - 13 - 1)${}^{-1}$ |              0 | (`swap`)  |     8 |     0 |
|    14 | (19 - 14 - 1)${}^{-1}$ |              0 | (`push`)  |     8 |     0 |
|    19 |  (5 - 19 - 1)${}^{-1}$ |              1 | (`pop`)   |     8 |     0 |
|     5 |  (10 - 5 - 1)${}^{-1}$ |              0 | (`push`)  |     9 |    42 |
|    10 | (15 - 10 - 1)${}^{-1}$ |              1 | (`pop`)   |     9 |    99 |
|    15 | (16 - 15 - 1)${}^{-1}$ |              0 | (`swap`)  |     9 |    77 |
|    16 | (18 - 16 - 1)${}^{-1}$ |              0 | (`push`)  |     9 |    77 |
|    18 |  (6 - 18 - 1)${}^{-1}$ |              1 | (`pop`)   |     9 |    77 |
|     6 |   (9 - 6 - 1)${}^{-1}$ |              0 | (`push`)  |    10 |    43 |
|     9 |  (17 - 9 - 1)${}^{-1}$ |              1 | (`pop`)   |    10 |    43 |
|    17 |  (7 - 17 - 1)${}^{-1}$ |              1 | (`pop`)   |    10 |    78 |
|     7 |   (8 - 7 - 1)${}^{-1}$ |              0 | (`nop`)   |    11 |    44 |
|     8 |                      0 |              1 | (`pop`)   |    11 |    44 |

## Extension Columns

The Op Stack Table has 2 extension columns, `rppa` and `rpcjd`.

1. A Permutation Argument establishes that the rows of the Op Stack Table correspond to the rows of the [Processor Table](processor-table.md).
  The running product for this argument is contained in the `rppa` column.
1. In order to achieve [memory consistency](memory-consistency.md), a [multi-table Permutation Argument](memory-consistency.md#memory-like-tables) shows that all clock jump differences greater than one, from all memory-like tables (i.e., including the [RAM Table](random-access-memory-table.md) and the [JumpStack Table](jump-stack-table.md)), are contained in the `cjd` column of the [Processor Table](processor-table.md).
  The running product for this argument is contained in the `rpcjd` column.

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
  This property is established by the clock jump difference lookup argument.
  In a nutshell, every difference of consecutive clock cycles that
  a) occurs within one contiguous block of constant memory pointer, and
  b) is greater than 1, is shown itself to be a valid clock cycle through a separate cross-table argument.
  The construction is described in more details in [Memory Consistency](memory-consistency.md#memory-like-tables).

# Arithmetic Intermediate Representation

Let all household items (🪥, 🛁, etc.) be challenges, concretely evaluation points, supplied by the verifier.
Let all fruit & vegetables (🥝, 🥥, etc.) be challenges, concretely weights to compress rows, supplied by the verifier.
Both types of challenges are X-field elements, _i.e._, elements of $\mathbb{F}_{p^3}$.

## Initial Constraints

1. `clk` is 0
1. `osv` is 0.
1. `osp` is the number of available stack registers, _i.e._, 16.
1. The running product for the permutation argument with the Processor Table `rppa` starts off having accumulated the first row with respect to challenges 🍋, 🍊, 🍉, and 🫒 and indeterminate 🪤.
1. The running product of clock jump differences `rpcjd` starts off with 1.

### Initial Constraints as Polynomials

1. `clk`
1. `osv`
1. `osp - 16`
1. `rppa - (🪤 - 🍋·clk - 🍊·ib1 - 🍉·osp - 🫒osv)`
1. `rpcjd - 1`

## Consistency Constraints

None.

## Transition Constraints

1.
  - the `osp` increases by 1, *or*
  - the `osp` does not change AND the `osv` does not change, *or*
  - the `osp` does not change AND the shrink stack indicator `shrink_stack` is 1.
1. The clock jump difference inverse column `clk_di` is the inverse of the clock jump difference minus one if a) the clock jump difference is greater than 1, and b) the op stack pointer remains the same.
1. The running product for the permutation argument with the Processor Table `rppa` absorbs the next row with respect to challenges 🍋, 🍊, 🍉, and 🫒 and indeterminate 🪤.
1. The running product for clock jump differences `rpcjd` accumulates a factor `(clk' - clk)` (relative to indeterminate `🚿`) if
  a) the clock jump difference is greater than 1, and if
  b) the op stack pointer does not change;
  and remains the same otherwise.

Written as Disjunctive Normal Form, the same constraints can be expressed as:

1.
  - the `osp` increases by 1 or the `osp` does not change
  - the `osp` increases by 1 or the `osv` does not change or the shrink stack indicator `shrink_stack` is 1
1. `osp' - osp = 1` or `clk_di` is the multiplicative inverse of `(clk' - clk - 1)` or `0` if that inverse does not exist.
1. `rppa' = rppa·(🪤 - 🍋·clk' - 🍊·ib1' - 🍉·osp' - 🫒osv')`
1. `rpcjd' = rpcjd` and `(clk' - clk - 1) = 0`;
  or `rpcjd' = rpcjd` and `osp' ≠ osp`;
  or `rpcjd' = rpcjd·(🚿 - clk' + clk)` and `(clk' - clk - 1)·clk_di = 1` and `osp' = osp`.

### Transition Constraints as Polynomials

1. `(osp' - (osp + 1))·(osp' - osp)`
1. `(osp' - (osp + 1))·(osv' - osv)·(1 - shrink_stack)`
1. `clk_di·(osp' - osp - 1)·(1 - clk_di·(clk' - clk - one))`
1. `(clk' - clk - one)·(osp' - osp - 1)·(1 - clk_di·(clk' - clk - one))`
1. `rppa' - rppa·(🪤 - 🍋·clk' - 🍊·ib1' - 🍉·osp' - 🫒osv')`
1. `(clk' - clk - 1)·(rpcjd' - rpcjd) + (osp' - osp - 1)·(rpcjd' - rpcjd) + (1 - (clk' - clk - 1)·clk_di)·(osp' - osp)·(rpcjd' - rpcjd·(🚿 - clk' + clk))`

## Terminal Constraints

None.
