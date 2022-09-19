# Jump Stack Table

The Jump Stack Memory contains the underflow from the Jump Stack.
TritonVM defines three registers to deal with the Jump Stack:
1. `jsp`, the jump stack pointer, which points to a location in Jump Stack Memory
1. `jso`, the last jump's origin, which points to a location in Program Memory, and
1. `jsd`, the last jump's destination, which also points to a location in Program Memory.

The Jump Stack Table is a table whose columns are a subset of those of the Processor Table.
The rows are sorted by jump stack pointer (`jsp`), then by cycle counter (`clk`).
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

| `clk` | `ip`   | `ci`     | `nia`    | `jsp` | `jso`  | `jsd`  | jump stack                           |
|:------|:-------|:---------|:---------|:------|:-------|:-------|:-------------------------------------|
| 0     | `0x00` | `foo`    | `bar`    | 0     | `0x00` | `0x00` | [ ]                                  |
| 1     | `0x01` | `bar`    | `call`   | 0     | `0x00` | `0x00` | [ ]                                  |
| 2     | `0x02` | `call`   | `0xA0`   | 0     | `0x00` | `0x00` | [ ]                                  |
| 3     | `0xA0` | `buzz`   | `foo`    | 1     | `0x04` | `0xA0` | [(`0x04`, `0xA0`)]                   |
| 4     | `0xA1` | `foo`    | `bar`    | 1     | `0x04` | `0xA0` | [(`0x04`, `0xA0`)]                   |
| 5     | `0xA2` | `bar`    | `return` | 1     | `0x04` | `0xA0` | [(`0x04`, `0xA0`)]                   |
| 6     | `0xA3` | `return` | `foo`    | 1     | `0x04` | `0xA0` | [(`0x04`, `0xA0`)]                   |
| 7     | `0x04` | `buzz`   | `bar`    | 0     | `0x00` | `0x00` | [ ]                                  |
| 8     | `0x05` | `bar`    | `call`   | 0     | `0x00` | `0x00` | [ ]                                  |
| 9     | `0x06` | `call`   | `0xB0`   | 0     | `0x00` | `0x00` | [ ]                                  |
| 10    | `0xB0` | `foo`    | `call`   | 1     | `0x08` | `0xB0` | [(`0x08`, `0xB0`)]                   |
| 11    | `0xB1` | `call`   | `0xC0`   | 1     | `0x08` | `0xB0` | [(`0x08`, `0xB0`)]                   |
| 12    | `0xC0` | `buzz`   | `foo`    | 2     | `0xB3` | `0xC0` | [(`0x08`, `0xB0`), (`0xB3`, `0xC0`)] |
| 13    | `0xC1` | `foo`    | `bar`    | 2     | `0xB3` | `0xC0` | [(`0x08`, `0xB0`), (`0xB3`, `0xC0`)] |
| 14    | `0xC2` | `bar`    | `return` | 2     | `0xB3` | `0xC0` | [(`0x08`, `0xB0`), (`0xB3`, `0xC0`)] |
| 15    | `0xC3` | `return` | `buzz`   | 2     | `0xB3` | `0xC0` | [(`0x08`, `0xB0`), (`0xB3`, `0xC0`)] |
| 16    | `0xB3` | `return` | `bazz`   | 1     | `0x08` | `0xB0` | [(`0x08`, `0xB0`)]                   |
| 17    | `0x08` | `foo`    | `bar`    | 0     | `0x00` | `0x00` | [ ]                                  |

Jump Stack Table:


| `clk` | `ci`     | `jsp` | `jso`  | `jsd`  |
|:------|:---------|:------|:-------|:-------|
| 0     | `foo`    | 0     | `0x00` | `0x00` |
| 1     | `bar`    | 0     | `0x00` | `0x00` |
| 2     | `call`   | 0     | `0x00` | `0x00` |
| 7     | `buzz`   | 0     | `0x00` | `0x00` |
| 8     | `bar`    | 0     | `0x00` | `0x00` |
| 9     | `call`   | 0     | `0x00` | `0x00` |
| 17    | `foo`    | 0     | `0x00` | `0x00` |
| 3     | `buzz`   | 1     | `0x04` | `0xA0` |
| 4     | `foo`    | 1     | `0x04` | `0xA0` |
| 5     | `bar`    | 1     | `0x04` | `0xA0` |
| 6     | `return` | 1     | `0x04` | `0xA0` |
| 10    | `foo`    | 1     | `0x08` | `0xB0` |
| 11    | `call`   | 1     | `0x08` | `0xB0` |
| 16    | `return` | 1     | `0x08` | `0xB0` |
| 12    | `buzz`   | 2     | `0xB3` | `0xC0` |
| 13    | `foo`    | 2     | `0xB3` | `0xC0` |
| 14    | `bar`    | 2     | `0xB3` | `0xC0` |
| 15    | `return` | 2     | `0xB3` | `0xC0` |

**Padding**

After the Jump Stack Table is filled in, its length being $l$, the table is padded until a total length of $2^{\lceil\log_2 l\rceil}$ is reached (or 0 if $l=0$).
Each padding row is a direct copy of the Jump Stack Table's last row, with the exception of the cycle count column `clk`.
In a padding row, column `clk` is set to the table's current total length, a value in the interval $[l, 2^{\lceil\log_2 l\rceil})$.
This ensures that every value in the interval $[0, 2^{\lceil\log_2 l\rceil})$ appears exactly once in the Jump Stack Table's `clk` column.

**Initial Constraints**

1. Cycle count `clk` is 0.
1. Jump Stack Pointer `jsp` is 0.
1. Jump Stack Origin `jso` is 0.
1. Jump Stack Destination `jsd` is 0.

**Initial Constraints as Polynomials**

1. `clk`
1. `jsp`
1. `jso`
1. `jsd`

**Consistency Constraints**

None.

**Transition Constraints**

1. The jump stack pointer `jsp` increases by 1, *or*
1. (`jsp` does not change and `jso` does not change and `jsd` does not change and the cycle counter `clk` increases by 1), *or*
1. (`jsp` does not change and `jso` does not change and `jsd` does not change and the current instruction `ci` is `call`), *or*
1. (`jsp` does not change and the current instruction `ci` is `return`).

Written as Disjunctive Normal Form, the same constraints can be expressed as:
1. The jump stack pointer `jsp` increases by 1 or the jump stack pointer `jsp` does not change
1. The jump stack pointer `jsp` increases by 1 or the jump stack origin `jso` does not change or current instruction `ci` is `return`
1. The jump stack pointer `jsp` increases by 1 or the jump stack destination `jsd` does not change or current instruction `ci` is `return`
1. The jump stack pointer `jsp` increases by 1 or the cycle count `clk` increases by 1 or current instruction `ci` is `call` or current instruction `ci` is `return`

**Transition Constraints as Polynomials**

1. `(jsp' - (jsp + 1))·(jsp' - jsp)`
1. `(jsp' - (jsp + 1))·(jso' - jso)·(ci - op_code(return))`
1. `(jsp' - (jsp + 1))·(jsd' - jsd)·(ci - op_code(return))`
1. `(jsp' - (jsp + 1))·(clk' - (clk + 1))·(ci - op_code(call))·(ci - op_code(return))`

**Terminal Constraints**

None.
 
**Relations to Other Tables**

1. A Permutation Argument establishes that the rows match with the rows in the [Processor Table](processor-table.md).
