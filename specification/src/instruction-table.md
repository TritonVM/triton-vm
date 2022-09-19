# Instruction Table

The Instruction Table establishes the link between the program and the instructions that are being executed by the processor.
The table consists of three columns:
1. the instruction's `address`,
1. the `current_instruction`, and
1. the `next_instruction_or_arg`.

It contains
- one row for every instruction in the [Program Table](#program-table), i.e., one row for every available instruction, and
- one row for every cycle `clk` in the [Processor Table](#processor-table), i.e., one row for every executed instruction.

The rows are sorted by `address`.

When copying the [Program Table](#program-table) with its two columns into the Instruction Table with its three columns, the value in `next_instruction_or_arg` is the value from the Program Table's _next_ row's `instruction` column (or 0 if no next row exists).
For an example, see below.

Program Table:

| Address | Instruction |
|--------:|:------------|
|       0 | push        |
|       1 | 10          |
|       2 | push        |
|       3 | 5           |
|       4 | add         |
|       … | …           |

Instruction Table:

| `address` | `current_instruction` | `next_instruction_or_arg` | (comment)              |
|----------:|:----------------------|:--------------------------|:-----------------------|
|         0 | push                  | 10                        | (from Program Table)   |
|         0 | push                  | 10                        | (from Processor Table) |
|         1 | 10                    | push                      | (from Program Table)   |
|         2 | push                  | 5                         | (from Program Table)   |
|         2 | push                  | 5                         | (from Processor Table) |
|         3 | 5                     | add                       | (from Program Table)   |
|         4 | add                   | …                         | (from Program Table)   |
|         … | …                     | …                         | …                      |

**Padding**

After the Instruction Table is filled in, its length being $l$, the table is padded until a total length of $2^{\lceil\log_2 l\rceil}$ is reached (or 0 if $l=0$).
Each padding row is a direct copy of the Instruction Table's last row, with the exception of the column `address`.
Column `address` increases by 1 between any two consecutive rows if at least one of the two rows is a padding row.

**Initial Constraints**

None.

**Consistency Constraints**

None.

**Transition Constraints**

1. The address increases by 1 or `current_instruction` does not change.
1. The address increases by 1 or `next_instruction_or_arg` does not change.

**Transition Constraints as Polynomials**

1. `(address' - (address + 1))·(current_instruction' - current_instruction)`
1. `(address' - (address + 1))·(next_instruction_or_arg' - next_instruction_or_arg)`

**Terminal Constraints**

None.

**Relations to Other Tables**

1. An Evaluation Argument establishes that the set of unique rows corresponds to the instructions as given by the [Program Table](#program-table).
1. A Permutation Argument establishes that the rows not included in above Evaluation Argument correspond to the values of the registers (`ip, ci, nia`) of the [Processor Table](#processor-table).
