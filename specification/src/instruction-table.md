# Instruction Table

The Instruction Table establishes the link between the program and the instructions that are being executed by the processor.
The table consists of four columns:
1. the instruction's `address`,
1. the `current_instruction`,
1. the `next_instruction_or_arg`,
1. a padding indicator `is_padding`.

It contains
- one row for every instruction in the [Program Table](#program-table), i.e., one row for every available instruction, and
- one row for every cycle `clk` in the [Processor Table](#processor-table), i.e., one row for every executed instruction.

The rows are sorted by `address`.

When copying the [Program Table](#program-table) with its three columns into the Instruction Table with its four columns, the value in `next_instruction_or_arg` is the value from the Program Table's _next_ row's `instruction` column (or 0 if no next row exists).
For an example, see below.

Program Table:

| Address | Instruction | is_padding |
|--------:|:------------|-----------:|
|       0 | push        |          0 |
|       1 | 10          |          0 |
|       2 | push        |          0 |
|       3 | 5           |          0 |
|       4 | add         |          0 |
|       … | …           |          … |

Instruction Table:

| `address` | `current_instruction` | `next_instruction_or_arg` | `is_padding` | (comment)              |
|----------:|:----------------------|:--------------------------|-------------:|:-----------------------|
|         0 | push                  | 10                        |            0 | (from Program Table)   |
|         0 | push                  | 10                        |            0 | (from Processor Table) |
|         1 | 10                    | push                      |            0 | (from Program Table)   |
|         2 | push                  | 5                         |            0 | (from Program Table)   |
|         2 | push                  | 5                         |            0 | (from Processor Table) |
|         3 | 5                     | add                       |            0 | (from Program Table)   |
|         4 | add                   | …                         |            0 | (from Program Table)   |
|         … | …                     | …                         |            … | …                      |

## Padding

A padding row is a copy of the Instruction Table's last row with the following modifications:
1. column `address` is increased by 1, and
1. column `is_padding` is set to 1.

## Initial Constraints

None.

## Consistency Constraints

None.

## Transition Constraints

1. The address increases by 1 or `current_instruction` does not change.
1. The address increases by 1 or `next_instruction_or_arg` does not change.

**Transition Constraints as Polynomials**

1. `(address' - (address + 1))·(current_instruction' - current_instruction)`
1. `(address' - (address + 1))·(next_instruction_or_arg' - next_instruction_or_arg)`

## Terminal Constraints

None.

## Relations to Other Tables

1. An Evaluation Argument establishes that the set of unique non-padding rows corresponds to the instructions as given by the [Program Table](#program-table).
1. A Permutation Argument establishes that the non-padding rows not included in above Evaluation Argument correspond to the values of the registers (`ip, ci, nia`) of the [Processor Table](#processor-table).
