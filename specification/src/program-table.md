# Program Table

The Virtual Machine's Program Memory is read-only.
The corresponding Program Table consists of three columns, `Address`, `Instruction`, and `IsPadding`.

| Address | Instruction | IsPadding |
|:--------|:------------|:----------|
| -       | -           |           |

The Program Table is static in the sense that it is fixed before the VM runs.
Moreover, the user can commit to the program by providing the Merkle root of the zipped FRI codeword.
This commitment assumes that the FRI domain is fixed, which implies an upper bound on program size.

## Padding

A padding row is a copy of the Program Table's last row with the following modifications:
1. column `Address` is increased by 1,
1. column `Instruction` is set to 0, and
1. column `IsPadding` is set to 1.

## Initial Constraints

1. The first address is 0.

**Initial Constraints as Polynomials**

1. `Address`

## Consistency Constraints

None.

## Transition Constraints

1. The address increases by 1.

**Transition Constraints as Polynomials**

1. `Address' - (Address + 1)`

## Terminal Constraints

None.

## Relations to Other Tables

1. An Evaluation Argument establishes that the non-padding rows of the Program Table match with the unique rows of the [Instruction Table](instruction-table.md).
