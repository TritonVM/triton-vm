# Jump Stack Table

The Jump Stack Memory contains the underflow from the Jump Stack.

## Base Columns

The Jump Stack Table consists of 6 columns:
1. the cycle counter `clk`
1. the inverse-or-zero of the difference between two consecutive `clk` values minus one, `clk_di`,
1. current instruction `ci`,
1. the jump stack pointer `jsp`,
1. the last jump's origin `jso`, and
1. the last jump's destination `jsd`.

| Clock | Inverse of Clock Difference Minus One | Current Instruction | Jump Stack Pointer | Jump Stack Origin | Jump Stack Destination |
|:------|:--------------------------------------|:--------------------|:-------------------|:------------------|:-----------------------|
| -     | -                                     | -                   | -                  | -                 | -                      |

The rows are sorted by jump stack pointer `jsp`, then by cycle counter `clk`.
The column `jsd` contains the destination of stack-extending jump (`call`) as well as of the no-stack-change jump (`recurse`);
the column `jso` contains the source of the stack-extending jump (`call`) or equivalently the destination of the stack-shrinking jump (`return`).

The AIR for this table guarantees that the return address of a single cell of return address memory can change only if there was a `call` instruction.

An example program, execution trace, and jump stack table are shown below.

Program:

| `address` | `instruction` |
|:----------|:--------------|
| `0x00`    | `foo`         |
| `0x01`    | `bar`         |
| `0x02`    | `call`        |
| `0x03`    | `0xA0`        |
| `0x04`    | `buzz`        |
| `0x05`    | `bar`         |
| `0x06`    | `call`        |
| `0x07`    | `0xB0`        |
| `0x08`    | `foo`         |
| `0x09`    | `bar`         |
| ⋮         | ⋮             |
| `0xA0`    | `buzz`        |
| `0xA1`    | `foo`         |
| `0xA2`    | `bar`         |
| `0xA3`    | `return`      |
| `0xA4`    | `foo`         |
| ⋮         | ⋮             |
| `0xB0`    | `foo`         |
| `0xB1`    | `call`        |
| `0xB2`    | `0xC0`        |
| `0xB3`    | `return`      |
| `0xB4`    | `bazz`        |
| ⋮         | ⋮             |
| `0xC0`    | `buzz`        |
| `0xC1`    | `foo`         |
| `0xC2`    | `bar`         |
| `0xC3`    | `return`      |
| `0xC4`    | `buzz`        |

Execution trace:

| `clk` | `ip`   | `ci`     | `nia`    | `jsp` |  `jso` |  `jsd` | jump stack                           |
|------:|:-------|:---------|:---------|------:|-------:|-------:|:-------------------------------------|
|     0 | `0x00` | `foo`    | `bar`    |     0 | `0x00` | `0x00` | [ ]                                  |
|     1 | `0x01` | `bar`    | `call`   |     0 | `0x00` | `0x00` | [ ]                                  |
|     2 | `0x02` | `call`   | `0xA0`   |     0 | `0x00` | `0x00` | [ ]                                  |
|     3 | `0xA0` | `buzz`   | `foo`    |     1 | `0x04` | `0xA0` | [(`0x04`, `0xA0`)]                   |
|     4 | `0xA1` | `foo`    | `bar`    |     1 | `0x04` | `0xA0` | [(`0x04`, `0xA0`)]                   |
|     5 | `0xA2` | `bar`    | `return` |     1 | `0x04` | `0xA0` | [(`0x04`, `0xA0`)]                   |
|     6 | `0xA3` | `return` | `foo`    |     1 | `0x04` | `0xA0` | [(`0x04`, `0xA0`)]                   |
|     7 | `0x04` | `buzz`   | `bar`    |     0 | `0x00` | `0x00` | [ ]                                  |
|     8 | `0x05` | `bar`    | `call`   |     0 | `0x00` | `0x00` | [ ]                                  |
|     9 | `0x06` | `call`   | `0xB0`   |     0 | `0x00` | `0x00` | [ ]                                  |
|    10 | `0xB0` | `foo`    | `call`   |     1 | `0x08` | `0xB0` | [(`0x08`, `0xB0`)]                   |
|    11 | `0xB1` | `call`   | `0xC0`   |     1 | `0x08` | `0xB0` | [(`0x08`, `0xB0`)]                   |
|    12 | `0xC0` | `buzz`   | `foo`    |     2 | `0xB3` | `0xC0` | [(`0x08`, `0xB0`), (`0xB3`, `0xC0`)] |
|    13 | `0xC1` | `foo`    | `bar`    |     2 | `0xB3` | `0xC0` | [(`0x08`, `0xB0`), (`0xB3`, `0xC0`)] |
|    14 | `0xC2` | `bar`    | `return` |     2 | `0xB3` | `0xC0` | [(`0x08`, `0xB0`), (`0xB3`, `0xC0`)] |
|    15 | `0xC3` | `return` | `buzz`   |     2 | `0xB3` | `0xC0` | [(`0x08`, `0xB0`), (`0xB3`, `0xC0`)] |
|    16 | `0xB3` | `return` | `bazz`   |     1 | `0x08` | `0xB0` | [(`0x08`, `0xB0`)]                   |
|    17 | `0x08` | `foo`    | `bar`    |     0 | `0x00` | `0x00` | [ ]                                  |

Jump Stack Table:


| `clk` |               `clk_di` | `ci`     | `jsp` |  `jso` |  `jsd` |
|------:|-----------------------:|:---------|------:|-------:|-------:|
|     0 |                      0 | `foo`    |     0 | `0x00` | `0x00` |
|     1 |                      0 | `bar`    |     0 | `0x00` | `0x00` |
|     2 |   (7 - 2 - 1)${}^{-1}$ | `call`   |     0 | `0x00` | `0x00` |
|     7 |                      0 | `buzz`   |     0 | `0x00` | `0x00` |
|     8 |                      0 | `bar`    |     0 | `0x00` | `0x00` |
|     9 |  (17 - 9 - 1)${}^{-1}$ | `call`   |     0 | `0x00` | `0x00` |
|    17 |  (3 - 17 - 1)${}^{-1}$ | `foo`    |     0 | `0x00` | `0x00` |
|     3 |                      0 | `buzz`   |     1 | `0x04` | `0xA0` |
|     4 |                      0 | `foo`    |     1 | `0x04` | `0xA0` |
|     5 |                      0 | `bar`    |     1 | `0x04` | `0xA0` |
|     6 |  (10 - 6 - 1)${}^{-1}$ | `return` |     1 | `0x04` | `0xA0` |
|    10 |                      0 | `foo`    |     1 | `0x08` | `0xB0` |
|    11 | (16 - 11 - 1)${}^{-1}$ | `call`   |     1 | `0x08` | `0xB0` |
|    16 | (12 - 16 - 1)${}^{-1}$ | `return` |     1 | `0x08` | `0xB0` |
|    12 |                      0 | `buzz`   |     2 | `0xB3` | `0xC0` |
|    13 |                      0 | `foo`    |     2 | `0xB3` | `0xC0` |
|    14 |                      0 | `bar`    |     2 | `0xB3` | `0xC0` |
|    15 |                      0 | `return` |     2 | `0xB3` | `0xC0` |

## Extension Columns

The Jump Stack Table has 2 extension columns, `rppa` and `rpcjd`.

1. A Permutation Argument establishes that the rows of the Jump Stack Table match with the rows in the [Processor Table](processor-table.md).
  The running product for this argument is contained in the `rppa` column.
1. In order to achieve [memory consistency](memory-consistency.md), a [multi-table Permutation Argument](memory-consistency.md#memory-like-tables) shows that all clock jump differences greater than one, from all memory-like tables (i.e., including the [RAM Table](random-access-memory-table.md) and the [OpStack Table](operational-stack-table.md)), are contained in the `cjd` column of the Processor Table.
  The running product for this argument is contained in the `rpcjd` column.

## Padding

A padding row is a direct copy of the Jump Stack Table's row with the highest value for column `clk`, called template row, with the exception of the cycle count column `clk`.
In a padding row, the value of column `clk` is 1 greater than the value of column `clk` in the template row.
The padding row is inserted right below the template row.
These steps are repeated until the desired padded height is reached.
In total, above steps ensure that the Permutation Argument between the Jump Stack Table and the [Processor Table](processor-table.md) holds up.

## Memory-Consistency

Memory-consistency follows from two more primitive properties:

 1. Contiguity of regions of constant memory pointer. Since the memory pointer for the JumpStack table, `jsp` can change by at most one per cycle, it is possible to enforce a full sorting using AIR constraints.
 2. Correct inner-sorting within contiguous regions. Specifically, the rows within each contiguous region of constant memory pointer should be sorted for clock cycle. This property is established by the clock jump difference lookup argument. In a nutshell, every difference of consecutive clock cycles that a) occurs within one contiguous block of constant memory pointer, and b) is greater than 1, is shown itself to be a valid clock cycle through a separate cross-table argument. The construction is described in more details in [Memory Consistency](memory-consistency.md).

# Arithmetic Intermediate Representation

Let all household items (🪥, 🛁, etc.) be challenges, concretely evaluation points, supplied by the verifier.
Let all fruit & vegetables (🥝, 🥥, etc.) be challenges, concretely weights to compress rows, supplied by the verifier.
Both types of challenges are X-field elements, _i.e._, elements of $\mathbb{F}_{p^3}$.

## Initial Constraints

1. Cycle count `clk` is 0.
1. Jump Stack Pointer `jsp` is 0.
1. Jump Stack Origin `jso` is 0.
1. Jump Stack Destination `jsd` is 0.
1. The running product for the permutation argument with the Processor Table `rppa` has absorbed the first row with respect to challenges 🍇, 🍅, 🍌, 🍏, and 🍐 and indeterminate 🧴.
1. The running product of clock jump differences `rpcjd` starts off with 1.

### Initial Constraints as Polynomials

1. `clk`
1. `jsp`
1. `jso`
1. `jsd`
1. `rppa - (🧴 - 🍇·clk - 🍅·ci - 🍌·jsp - 🍏·jso - 🍐·jsd)`
1. `rpcjd - 1`

## Consistency Constraints

None.

## Transition Constraints

1. The jump stack pointer `jsp` increases by 1, *or*
1. (`jsp` does not change and `jso` does not change and `jsd` does not change and the cycle counter `clk` increases by 1), *or*
1. (`jsp` does not change and `jso` does not change and `jsd` does not change and the current instruction `ci` is `call`), *or*
1. (`jsp` does not change and the current instruction `ci` is `return`).
1. The clock jump difference inverse column `clk_di` is the inverse of the clock jump difference minus one if a) the clock jump difference is greater than 1, and b) the jump stack pointer remains the same.
1. If the memory pointer `jsp` does not change, then `clk_di` is the inverse-or-zero of the clock jump difference minus one.
1. The running product for the permutation argument `rppa` absorbs the next row  with respect to challenges 🍇, 🍅, 🍌, 🍏, and 🍐 and indeterminate 🧴.
1. The running product for clock jump differences `rpcjd` accumulates a factor `(clk' - clk - 1)` (relative to indeterminate `🚿`) if
  a) the clock jump difference is greater than 1, and if
  b) the jump stack pointer does not change;
  and remains the same otherwise.

Written as Disjunctive Normal Form, the same constraints can be expressed as:
1. The jump stack pointer `jsp` increases by 1 or the jump stack pointer `jsp` does not change
1. The jump stack pointer `jsp` increases by 1 or the jump stack origin `jso` does not change or current instruction `ci` is `return`
1. The jump stack pointer `jsp` increases by 1 or the jump stack destination `jsd` does not change or current instruction `ci` is `return`
1. The jump stack pointer `jsp` increases by 1 or the cycle count `clk` increases by 1 or current instruction `ci` is `call` or current instruction `ci` is `return`
1. Either `jsp' - jsp = 1` or `clk_di` is the inverse of `clk' - clk - 1` (or 0 if no inverse exists).
1. `rppa' = rppa·(🧴 - a·clk' - b·ci' - c·jsp' - d·jsp' - e·jsd)`
1. `rpcjd' = rpcjd` and `(clk' - clk - 1) = 0`;
  or `rpcjd' = rpcjd` and `jsp' ≠ jsp`;
  or `rpcjd' = rpcjd·(🚿 - clk' + clk)` and `(clk' - clk - 1)·clk_di = 1` and `jsp' = jsp`.

### Transition Constraints as Polynomials

1. `(jsp' - (jsp + 1))·(jsp' - jsp)`
1. `(jsp' - (jsp + 1))·(jso' - jso)·(ci - op_code(return))`
1. `(jsp' - (jsp + 1))·(jsd' - jsd)·(ci - op_code(return))`
1. `(jsp' - (jsp + 1))·(clk' - (clk + 1))·(ci - op_code(call))·(ci - op_code(return))`
1. `clk_di·(jsp' - jsp - 1)·(1 - clk_di·(clk' - clk - one))`
1. `(clk' - clk - one)·(jsp' - jsp - 1)·(1 - clk_di·(clk' - clk - one))`
1. `rppa' - rppa·(🧴 - 🍇·clk - 🍅·ci - 🍌·jsp - 🍏·jso - 🍐·jsd)`
1. `(clk' - clk - 1)·(rpcjd' - rpcjd) + (jsp' - jsp - 1)·(rpcjd' - rpcjd) + (1 - (clk' - clk - 1)·clk_di)·(jsp' - jsp)·(rpcjd' - rpcjd·(🚿 - clk' + clk))`

## Terminal Constraints

None.
