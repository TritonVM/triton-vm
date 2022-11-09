# Program Table

## Base Columns

The virtual machine's program memory is read-only.
The corresponding Program Table consists of 3 base columns:
`Address`, `Instruction`, and `IsPadding`.

| Address | Instruction | Is Padding |
|:--------|:------------|:-----------|
| -       | -           | -          |

The Program Table is static in the sense that it is fixed before the VM runs.
Moreover, the user can commit to the program by providing the Merkle root of the zipped FRI codeword.
This commitment assumes that the FRI domain is fixed, which implies an upper bound on program size.

## Extension Columns

An Evaluation Argument establishes that the non-padding rows of the Program Table match with the unique rows of the [Instruction Table](instruction-table.md).
Therefor, the Program Table has 1 extension column, `RunningEvaluation`.

## Padding

A padding row is a copy of the Program Table's last row with the following modifications:
1. column `Address` is increased by 1,
1. column `Instruction` is set to 0, and
1. column `IsPadding` is set to 1.

# Arithmetic Intermediate Representation

Let all household items (🪥, 🛁, etc.) be challenges, concretely evaluation points, supplied by the verifier.
Let all fruit & vegetables (🥝, 🥥, etc.) be challenges, concretely weights to compress rows, supplied by the verifier.
Both types of challenges are X-field elements, _i.e._, elements of $\mathbb{F}_{p^3}$.

## Initial Constraints

1. The first address is 0.
1. The running evaluation is 1.

### Initial Constraints as Polynomials

1. `Address`
1. `RunningEvaluation - 1`

## Consistency Constraints

None.

## Transition Constraints

1. The address increases by 1.
1. If the current row is not a padding row, the running evaluation absorbs the current row's address, the current row's instruction, and the next row's instruction with respect to challenges 🥝, 🥥, and 🫐 and indeterminate 🪥 respectively. Otherwise, it remains unchanged.

### Transition Constraints as Polynomials

1. `Address' - (Address + 1)`
1. `(1 - IsPadding) · (RunningEvaluation' - 🪥·RunningEvaluation - 🥝·Address - 🥥·Instruction - 🫐·Instruction') + IsPadding · (RunningEvaluation' - RunningEvaluation)`

## Terminal Constraints

None.
