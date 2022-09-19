# Program Table

The Virtual Machine's Program Memory is read-only.
The corresponding Program Table consists of two columns, `address` and `instruction`.
The latter variable does not correspond to the processor's state but to the value of the memory at the given location.

| Address | Instruction |
|:--------|:------------|
| -       | -           |

The Program Table is static in the sense that it is fixed before the VM runs.
Moreover, the user can commit to the program by providing the Merkle root of the zipped FRI codeword.
This commitment assumes that the FRI domain is fixed, which implies an upper bound on program size.

**Padding**

After the Program Table is filled in, its length being $l$, the table is padded until a total length of $2^{\lceil\log_2 l\rceil}$ is reached (or 0 if $l=0$).
Each padding row is a direct copy of the Program Table's last row, with the exception of the column `address`.
Column `address` increases by 1 between any two consecutive rows, even padding rows.

**Initial Constraints**

1. The first address is 0.

**Initial Constraints as Polynomials**

1. `addr`

**Consistency Constraints**

None.

**Transition Constraints**

1. The address increases by 1.

**Transition Constraints as Polynomials**

1. `addr' - (addr + 1)`

**Terminal Constraints**

None.

**Relations to other Tables**

1. An Evaluation Argument establishes that the rows of the Program Table match with the unique rows of the [Instruction Table](instruction-table.md).
