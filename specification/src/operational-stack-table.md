# Operational Stack Table

The operational stack is where the program stores simple elementary operations, function arguments, and pointers to important objects.
There are 16 registers (`st0` through `st15`) that the program can access directly.
These registers correspond to the top of the stack.
The rest of the operational stack is stored in a dedicated memory object called Operational Stack Underflow Memory.
It is initially empty.

The operational stack table contains a subset of the columns of the processor table –
specifically, the cycle counter `clk`, the current instruction `ci`, the operation stack value `osv` and pointer `osp`.
The rows of the operational stack table are sorted by operational stack pointer `osp` first, cycle count `clk` second.

The mechanics are best illustrated by an example.
Observe how the malicious manipulation of the OpStack Underflow Memory, the change of “42” into “99” in cycle 8, is detected:
The transition constraints of the OpStack Table stipulate that if `osp` does not change, then `osv` can only change if current instruction `ci` shrinks the OpStack.
Instruction `push` does _not_ shrink the OpStack.
Consequently, row `[5, push, 9, 42]` being followed by row `[_, _, 9, 99]` violates the constraints.

For illustrative purposes only, we use four stack registers `st0` through `st3` in the example.
TritonVM has 16 stack registers, `st0` through `st15`.

Execution trace:

| `clk` | `ci`   | `nia`  | `st0` | `st1` | `st2` | `st3` | OpStack Underflow Memory | `osp` | `osv` |
|------:|:-------|:-------|------:|------:|------:|------:|:-------------------------|------:|------:|
|     0 | `push` | 42     |     0 |     0 |     0 |     0 | [ ]                      |     4 |     0 |
|     1 | `push` | 43     |    42 |     0 |     0 |     0 | [0]                      |     5 |     0 |
|     2 | `push` | 44     |    43 |    42 |     0 |     0 | [0,0]                    |     6 |     0 |
|     3 | `push` | 45     |    44 |    43 |    42 |     0 | [0,0,0]                  |     7 |     0 |
|     4 | `push` | 46     |    45 |    44 |    43 |    42 | [0,0,0,0]                |     8 |     0 |
|     5 | `push` | 47     |    46 |    45 |    44 |    43 | [42,0,0,0,0]             |     9 |    42 |
|     6 | `push` | 48     |    47 |    46 |    45 |    44 | [43,42,0,0,0,0]          |    10 |    43 |
|     7 | `nop`  | `pop`  |    48 |    47 |    46 |    45 | [44,43,42,0,0,0,0]       |    11 |    44 |
|     8 | `pop`  | `pop`  |    48 |    47 |    46 |    45 | [44,43,99,0,0,0,0]       |    11 |    44 |
|     9 | `pop`  | `pop`  |    47 |    46 |    45 |    44 | [43,99,0,0,0,0]          |    10 |    43 |
|    10 | `pop`  | `pop`  |    46 |    45 |    44 |    43 | [99,0,0,0,0]             |     9 |    99 |
|    11 | `pop`  | `push` |    45 |    44 |    43 |    99 | [0,0,0,0]                |     8 |     0 |
|    12 | `push` | 77     |    44 |    43 |    99 |     0 | [0,0,0]                  |     7 |     0 |
|    13 | `swap` | 4      |    77 |    44 |    43 |    99 | [0,0,0,0]                |     8 |     0 |
|    14 | `push` | 78     |    99 |    44 |    43 |    77 | [0,0,0,0]                |     8 |     0 |
|    15 | `swap` | 4      |    78 |    99 |    44 |    43 | [77,0,0,0,0]             |     9 |    77 |
|    16 | `push` | 79     |    43 |    99 |    44 |    78 | [77,0,0,0,0]             |     9 |    77 |
|    17 | `pop`  | `pop`  |    79 |    43 |    99 |    44 | [78,77,0,0,0,0]          |    10 |    78 |
|    18 | `pop`  | `pop`  |    43 |    99 |    44 |    78 | [77,0,0,0,0]             |     9 |    77 |
|    19 | `pop`  | `pop`  |    99 |    44 |    78 |    77 | [0,0,0,0]                |     8 |     0 |
|    20 | `pop`  | `pop`  |    44 |    78 |    77 |     0 | [0,0,0]                  |     7 |     0 |
|    21 | `pop`  | `pop`  |    78 |    77 |     0 |     0 | [0,0]                    |     6 |     0 |
|    22 | `pop`  | `pop`  |    77 |     0 |     0 |     0 | [0]                      |     5 |     0 |
|    23 | `pop`  | `pop`  |     0 |     0 |     0 |     0 | [ ]                      |     4 |     0 |
|    24 | `pop`  | `nop`  |     0 |     0 |     0 |     0 | 💥                        |     3 |     0 |

Operational Stack Table:

| `clk` | `ci`   | `osp` | `osv` |
|------:|:-------|------:|------:|
|     0 | `push` |     4 |     0 |
|    23 | `pop`  |     4 |     0 |
|     1 | `push` |     5 |     0 |
|    22 | `pop`  |     5 |     0 |
|     2 | `push` |     6 |     0 |
|    21 | `pop`  |     6 |     0 |
|     3 | `push` |     7 |     0 |
|    12 | `push` |     7 |     0 |
|    20 | `pop`  |     7 |     0 |
|     4 | `push` |     8 |     0 |
|    11 | `pop`  |     8 |     0 |
|    13 | `swap` |     8 |     0 |
|    14 | `push` |     8 |     0 |
|    19 | `pop`  |     8 |     0 |
|     5 | `push` |     9 |    42 |
|    10 | `pop`  |     9 |    99 |
|    15 | `swap` |     9 |    77 |
|    16 | `push` |     9 |    77 |
|    18 | `pop`  |     9 |    77 |
|     6 | `push` |    10 |    43 |
|     9 | `pop`  |    10 |    43 |
|    17 | `pop`  |    10 |    78 |
|     7 | `nop`  |    11 |    44 |
|     8 | `pop`  |    11 |    44 |


## Padding

After the Op Stack Table is filled in, its length being $l$, the table is padded until a total length of $2^{\lceil\log_2 l\rceil}$ is reached (or 0 if $l=0$).
Each padding row is a direct copy of the Op Stack Table's last row, with the exception of the cycle count column `clk`.
In a padding row, column `clk` is set to the table's current total length, a value in the interval $[l, 2^{\lceil\log_2 l\rceil})$.
This ensures that every value in the interval $[0, 2^{\lceil\log_2 l\rceil})$ appears exactly once in the Op Stack Table's `clk` column.

## Initial Constraints

1. `clk` is 0
1. `osv` is 0.
1. `osp` is the number of available stack registers, i.e., 16.

**Initial Constraints as Polynomials**

1. `clk`
1. `osv`
1. `osp - 16`

## Consistency Constraints

None.

## Transition Constraints

1. the `osp` increases by 1, *or*
1. the `osp` does not change AND the `osv` does not change, *or*
1. the `osp` does not change AND the `ci` shrinks the OpStack.

Written as Disjunctive Normal Form, the same constraints can be expressed as:
1. the `osp` increases by 1 or the `osp` does not change
1. the `osp` increases by 1 or the `osv` does not change or the `ci` shrinks the OpStack

An instruction is OpStack-shrinking if it is
- in instruction group `shrink_stack`, or
- in instruction group `binop`, or
- `xbmul`.

**Transition Constraints as Polynomials**

1. `(osp' - (osp + 1))·(osp' - osp)`
1. `(osp' - (osp + 1))·(osv' - osv)·(ci - op_code(pop))·(ci - op_code(skiz))·(ci - op_code(assert))·(ci - op_code(add))·(ci - op_code(mul))·(ci - op_code(eq))·(ci - op_code(lt))·(ci - op_code(and))·(ci - op_code(xor))·(ci - op_code(xbmul))·(ci - op_code(write_io))`

## Terminal Constraints

None.

## Relations to Other Tables

1. A Permutation Argument establishes that the rows of the operational stack table correspond to the rows of the [Processor Table](processor-table.md).
