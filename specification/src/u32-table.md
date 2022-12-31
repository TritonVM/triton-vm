# U32 Table

The U32 Operations Table arithmetizes the coprocessor for “difficult” 32-bit unsigned integer operations.
The two inputs to the U32 Operations Table are left-hand side (LHS) and right-hand side (RHS).
To allow efficient arithmetization, a [u32 instruction](instructions.md)'s result is constructed using multiple rows.
Crucially, the rows of the U32 table are independent of the processor's clock.
Hence, the result of the instruction can be transferred into the processor within one clock cycle.

In the U32 Table, the results for many different instructions are computed in parallel, each in their own column.
The processor's current instruction `CI` determines which of columns is copied to the processor as the result.

## Base Columns

| name             | description                                                                                                   |
|:-----------------|:--------------------------------------------------------------------------------------------------------------|
| `CopyFlag`       | Marks the beginning of an independent section within the U32 table.                                           |
| `Bits`           | The number of bits that LHS and RHS have already been shifted by.                                             |
| `BitsMinus33Inv` | The inverse-or-zero of the difference between 33 and `Bits`.                                                  |
| `CI`             | Current Instruction, the instruction the processor is currently executing.                                    |
| `LHS` (`st0`)    | Left-hand side of the operation.                                                                              |
| `RHS` (`st1`)    | Right-hand side of the operation.                                                                             |
| `LT`             | The result (or intermediate result) of LHS < RHS.                                                             |
| `AND`            | The result (or intermediate result) of LHS & RHS, _i.e._, bitwise and.                                        |
| `XOR`            | The result (or intermediate result) of LHS ^ RHS, _i.e._, bitwise xor.                                        |
| `Log2Floor`      | The number of bits in LHS minus 1, which is usually equivalent to the floor of log₂(LHS), except for LHS = 0. |
| `LhsCopy`        | A copy of LHS in the first row in the current, independent section, _i.e._, when `CopyFlag` is 1.             |
| `Pow`            | The result (or intermediate result) of $\texttt{LHS}^\texttt{RHS}$. Might overflow – care advised!            |
| `LhsInv`         | The inverse-or-zero of LHS. Needed to check whether `LHS` is unequal to 0.                                    |
| `RhsInv`         | The inverse-or-zero of RHS. Needed to check whether `RHS` is unequal to 0.                                    |

`LT` can take on the values 0, 1, or 2, where
- 0 means `LHS` >= `RHS` is definitely known in the current row,
- 1 means `LHS` < `RHS` is definitely known in the current row, and
- 2 means the result is unknown in the current row.
This value can never occur in the first row of a section, _i.e._, when `CopyFlag` is 1.

An example U32 Table follows.
Some columns are omitted for presentation reasons.
All cells in the U32 Table contain elements of the B-field.
For clearer illustration of the mechanics, the columns marked “$_2$” are presented in base 2, whereas columns marked “$_{10}$” are in the usual base 10.

| CopyFlag$_2$ | Bits$_{10}$ | LHS$_2$   | RHS$_2$   | LT$_{10}$ | AND$_2$   | XOR$_2$   | Log2Floor$_{10}$ |           Pow$_{10}$ |
|-------------:|------------:|:----------|:----------|----------:|:----------|:----------|-----------------:|---------------------:|
|            1 |           0 | 001**0**  | 101**0**  |         1 | 001**0**  | 100**0**  |                1 |                 1024 |
|            0 |           1 | 00**1**   | 10**1**   |         1 | 00**1**   | 10**0**   |                1 |                   32 |
|            0 |           2 | 0**0**    | 1**0**    |         1 | 0**0**    | 1**0**    |               -1 |                    4 |
|            0 |           3 | **0**     | **1**     |         1 | **0**     | **1**     |               -1 |                    2 |
|            0 |           4 | **0**     | **0**     |         2 | **0**     | **0**     |               -1 |                    1 |
|            1 |           0 | 1100**0** | 1101**0** |         1 | 1100**0** | 0001**0** |                4 | 11527596562258709312 |
|            0 |           1 | 110**0**  | 110**1**  |         1 | 110**0**  | 000**1**  |                4 |   876488338465357824 |
|            0 |           2 | 11**0**   | 11**0**   |         2 | 11**0**   | 00**0**   |                4 |            191102976 |
|            0 |           3 | 1**1**    | 1**1**    |         2 | 1**1**    | 0**0**    |                4 |                13824 |
|            0 |           4 | **1**     | **1**     |         2 | **1**     | **0**     |                4 |                   24 |
|            0 |           5 | **0**     | **0**     |         2 | **0**     | **0**     |               -1 |                    1 |

The AIR verifies the correct update of each consecutive pair of rows.
In every row one bit – the current least significant bit of both LHS and RHS – is eliminated.
Only when both `LHS` and `RHS` are 0 can a new row with `CopyFlag = 1` be inserted.
Inserting a row with `Bits` equal to 32 and `LHS` or `RHS` not 0, as well as
inserting a row with `Bits` equal to 33 makes it impossible to generate a proof of correct execution of Triton VM.

## Extension Columns

The U32 Table has 1 extension column, `RunningProductProcessor`.
It corresponds to the Permutation Argument with the [Processor Table](processor-table.md), establishing that whenever the processor executes a u32 instruction, the following holds:

- the processor's stack register `st0` is copied into `LHS`,
- the processor's stack register `st1` is copied into `RHS`,
- the processor's `CI` register is copied into `CI`, and
- the result, condition to `CI`, is copied to the processor.

More concretely, the result to be copied to the processor is

- `LT` if `CI` is opcode(`lt`),
- `AND` if `CI` is opcode(`and`),
- `XOR` if `CI` is opcode(`xor`),
- `Log2Floor` if `CI` is opcode(`log_2_floor`),
- `Pow` if `CI` is opcode(`pow`),
- `LT` if `CI` is opcode(`div`), and
- 0 if `CI` is 0.

The instruction `div` uses the U32 Table to ensure that the remainder `r` is smaller than the divisor `d`.
Hence, the result of `LT` is used.
The instruction `div` _also_ uses the U32 Table to ensure that the numerator `n` and the quotient `q` are u32.
For this range check, happening in its independent section, no result is required.
The instruction `split` also uses the U32 Table for range checking only, _i.e._, to ensure that the instruction's resulting “high bits” and “low bits” each fit in a u32.

To conditionally copy the required result to the processor, instruction de-selectors like in the Processor Table are used.
Concretely, with `u32_instructions = {lt, and, xor, log_2_floor, pow, div}`, the following aliases are used:

- `lt_div_deselector` = $\texttt{CI}\cdot\prod_{\substack{\texttt{i} \in \texttt{u32\_instructions}\\\texttt{i} \not\in \{\texttt{lt}, \texttt{div}\}}} \texttt{CI} - \texttt{opcode}(\texttt{i})$
- `and_deselector` = $\texttt{CI}\cdot\prod_{\substack{\texttt{i} \in \texttt{u32\_instructions}\\\texttt{i} \neq \texttt{and}}} \texttt{CI} - \texttt{opcode}(\texttt{i})$
- `xor_deselector` = $\texttt{CI}\cdot\prod_{\substack{\texttt{i} \in \texttt{u32\_instructions}\\\texttt{i} \neq \texttt{xor}}} \texttt{CI} - \texttt{opcode}(\texttt{i})$
- `log_2_floor_deselector` = $\texttt{CI}\cdot\prod_{\substack{\texttt{i} \in \texttt{u32\_instructions}\\\texttt{i} \neq \texttt{log\_2\_floor}}} \texttt{CI} - \texttt{opcode}(\texttt{i})$
- `pow_deselector` = $\texttt{CI}\cdot\prod_{\substack{\texttt{i} \in \texttt{u32\_instructions}\\\texttt{i} \neq \texttt{pow}}} \texttt{CI} - \texttt{opcode}(\texttt{i})$

Throughout the next sections, the alias `Result` corresponds to the polynomial
`LT·lt_div_deselector + AND·and_deselector + XOR·xor_deselector + Log2Floor·log_2_floor_deselector + Pow·pow_deselector`.

## Padding

Each padding row is the all-zero row with the exception of
- `BitsMinus33Inv`, which is $-33^{-1}$,
- `LT`, which is 2,
- `Log2Floor`, which is -1, and
- `Pow`, which is 1.

# Arithmetic Intermediate Representation

Let all household items (🪥, 🛁, etc.) be challenges, concretely evaluation points, supplied by the verifier.
Let all fruit & vegetables (🥝, 🥥, etc.) be challenges, concretely weights to compress rows, supplied by the verifier.
Both types of challenges are X-field elements, _i.e._, elements of $\mathbb{F}_{p^3}$.

## Initial Constraints

1. If the `CopyFlag` is 0, then `RunningProductProcessor` is 1.
1. If the `CopyFlag` is 1, then `RunningProductProcessor` has absorbed the first row with respect to challenges 🥜, 🌰, 🥑, and 🥕, and indeterminate 🧷.

### Initial Constraints as Polynomials

1. `(CopyFlag - 1)·(RunningProductProcessor - 1)`
1. `CopyFlag·(RunningProductProcessor - (🧷 - 🥜·LHS - 🌰·RHS - 🥑·CI - 🥕·Result))`

## Consistency Constraints

1. The `CopyFlag` is 0 or 1.
1. If `CopyFlag` is 1, then `Bits` is 0.
1. `BitsMinus33Inv` is the inverse of (`Bits` - 33).
1. `LhsInv` is 0 or the inverse of `LHS`.
1. `Lhs` is 0 or `LhsInv` is the inverse of `LHS`.
1. `RhsInv` is 0 or the inverse of `RHS`.
1. `Rhs` is 0 or `RhsInv` is the inverse of `RHS`.
1. If `CopyFlag` is 1, then `LhsCopy` is `LHS`.
1. If `CopyFlag` is 0 and `LHS` is 0 and `RHS` is 0, then `LT` is 2.
1. If `CopyFlag` is 0 and `LHS` is 0 and `RHS` is 0, then `AND` is 0.
1. If `CopyFlag` is 0 and `LHS` is 0 and `RHS` is 0, then `XOR` is 0.
1. If `CopyFlag` is 0 and `LHS` is 0 and `RHS` is 0, then `Pow` is 1.
1. If `LHS` is 0, then `Log2Floor` is -1.

Written in Disjunctive Normal Form, the same constraints can be expressed as:

1. The `CopyFlag` is 0 or 1.
1. `CopyFlag` is 0 or `Bits` is 0.
1. `BitsMinus33Inv` the inverse of (`Bits` - 33).
1. `LhsInv` is 0 or the inverse of `LHS`.
1. `Lhs` is 0 or `LhsInv` is the inverse of `LHS`.
1. `RhsInv` is 0 or the inverse of `RHS`.
1. `Rhs` is 0 or `RhsInv` is the inverse of `RHS`.
1. `CopyFlag` is 0 or `LhsCopy` is `LHS`.
1. `CopyFlag` is 1 or `LHS` is not 0 or `RHS` is not 0 or `LT` is 2.
1. `CopyFlag` is 1 or `LHS` is not 0 or `RHS` is not 0 or `AND` is 0.
1. `CopyFlag` is 1 or `LHS` is not 0 or `RHS` is not 0 or `XOR` is 0.
1. `CopyFlag` is 1 or `LHS` is not 0 or `RHS` is not 0 or `Pow` is 1.
1. `LHS` is not 0 or `Log2Floor` is -1.

### Consistency Constraints as Polynomials

1. `CopyFlag·(CopyFlag - 1)`
1. `CopyFlag·Bits`
1. `1 - BitsMinus33Inv·(Bits - 33)`
1. `LhsInv·(1 - LHS·LhsInv)`
1. `LHS·(1 - LHS·LhsInv)`
1. `RhsInv·(1 - RHS·RhsInv)`
1. `RHS·(1 - RHS·RhsInv)`
1. `CopyFlag·(LHS - LhsCopy)`
1. `(CopyFlag - 1)·(1 - LHS·LhsInv)·(1 - RHS·RhsInv)·(LT - 2)`
1. `(CopyFlag - 1)·(1 - LHS·LhsInv)·(1 - RHS·RhsInv)·AND`
1. `(CopyFlag - 1)·(1 - LHS·LhsInv)·(1 - RHS·RhsInv)·XOR`
1. `(CopyFlag - 1)·(1 - LHS·LhsInv)·(1 - RHS·RhsInv)·(Pow - 1)`
1. `(1 - LHS·LhsInv)·(Log2Floor + 1)`

## Transition Constraints

Even though they are never explicitly represented, it is useful to alias the `LHS`'s and `RHS`'s _least-significant bit_, or “lsb.”
Given two consecutive rows for `LHS`, the (current) least significant bit can be computed by subtracting twice the next row's `LHS` from the current row's `LHS`.
These aliases, _i.e._, `LhsLsb` = 2·`LHS`' - `LHS` and `RhsLsb` = 2·`RHS`' - `RHS`, are used throughout the following.

1. If the `CopyFlag` in the next row is 1, then `LHS` in the current row is 0.
1. If the `CopyFlag` in the next row is 1, then `RHS` in the current row is 0.
1. If the `CopyFlag` in the next row is 0, then `CI` in the next row is `CI` in the current row.
1. If the `CopyFlag` in the next row is 0, then `LhsCopy` in the next row is `LhsCopy` in the current row.
1. If the `CopyFlag` in the next row is 0 and `LHS` in the current row is unequal to 0, then `Bits` in the next row is `Bits` in the current row plus 1.
1. If the `CopyFlag` in the next row is 0 and `RHS` in the current row is unequal to 0, then `Bits` in the next row is `Bits` in the current row plus 1.
1. If the `CopyFlag` in the next row is 0, then `LhsLsb` is either 0 or 1.
1. If the `CopyFlag` in the next row is 0, then `RhsLsb` is either 0 or 1.
1. If the `CopyFlag` in the next row is 0 and `LT` in the next row is 0, then `LT` in the current row is 0.
1. If the `CopyFlag` in the next row is 0 and `LT` in the next row is 1, then `LT` in the current row is 1.
1. If the `CopyFlag` in the next row is 0 and `LT` in the next row is 2 and `LhsLsb` is 0 and `RhsLsb` is 1, then `LT` in the current row is 1.
1. If the `CopyFlag` in the next row is 0 and `LT` in the next row is 2 and `LhsLsb` is 1 and `RhsLsb` is 0, then `LT` in the current row is 0.
1. If the `CopyFlag` in the next row is 0 and `LT` in the next row is 2 and `LhsLsb` is `RhsLsb` and the `CopyFlag` in the current row is 0, then `LT` in the current row is 2.
1. If the `CopyFlag` in the next row is 0 and `LT` in the next row is 2 and `LhsLsb` is `RhsLsb` and the `CopyFlag` in the current row is 1, then `LT` in the current row is 0.
1. If the `CopyFlag` in the next row is 0, then `AND` in the current row is twice `AND` in the next row plus the product of `LhsLsb` and `RhsLsb`.
1. If the `CopyFlag` in the next row is 0, then `XOR` in the current row is twice `XOR` in the next row plus `LhsLsb` plus `RhsLsb` minus twice the product of `LhsLsb` and `RhsLsb`.
1. If the `CopyFlag` in the next row is 0 and `LHS` in the next row is 0 and `LHS` in the current row is not 0, then `Log2Floor` in the current row is `Bits`.
1. If the `CopyFlag` in the next row is 0 and `LHS` in the next row is not 0, then `Log2Floor` in the current row is `Log2Floor` in the next row.
1. If the `CopyFlag` in the next row is 0 and `RhsLsb` in the current row is 0, then `Pow` in the current row is `Pow` in the next row squared.
1. If the `CopyFlag` in the next row is 0 and `RhsLsb` in the current row is 1, then `Pow` in the current row is `Pow` in the next row squared times `LhsCopy` in the next row.
1. If the `CopyFlag` in the next row is 0, then `RunningProductProcessor` in the next row is `RunningProductProcessor` in the current row.
1. If the `CopyFlag` in the next row is 1, then `RunningProductProcessor` in the next row has absorbed the next row with respect to challenges 🥜, 🌰, 🥑, and 🥕, and indeterminate 🧷.

Written in Disjunctive Normal Form, the same constraints can be expressed as:

1. `CopyFlag`' is 0 or `LHS` is 0.
1. `CopyFlag`' is 0 or `RHS` is 0.
1. `CopyFlag`' is 1 or `CI`' is `CI`.
1. `CopyFlag`' is 1 or `LhsCopy`' is `LhsCopy`.
1. `CopyFlag`' is 1 or `LHS` is 0 or `Bits`' is `Bits` + 1.
1. `CopyFlag`' is 1 or `RHS` is 0 or `Bits`' is `Bits` + 1.
1. `CopyFlag`' is 1 or `LhsLsb` is 0 or `LhsLsb` is 1.
1. `CopyFlag`' is 1 or `RhsLsb` is 0 or `RhsLsb` is 1.
1. `CopyFlag`' is 1 or (`LT`' is 1 or 2) or `LT` is 0.
1. `CopyFlag`' is 1 or (`LT`' is 0 or 2) or `LT` is 1.
1. `CopyFlag`' is 1 or (`LT`' is 0 or 1) or `LhsLsb` is 1 or `RhsLsb` is 0 or `LT` is 1.
1. `CopyFlag`' is 1 or (`LT`' is 0 or 1) or `LhsLsb` is 0 or `RhsLsb` is 1 or `LT` is 0.
1. `CopyFlag`' is 1 or (`LT`' is 0 or 1) or `LhsLsb` is unequal to `RhsLsb` or `CopyFlag` is 1 or `LT` is 2.
1. `CopyFlag`' is 1 or (`LT`' is 0 or 1) or `LhsLsb` is unequal to `RhsLsb` or `CopyFlag` is 0 or `LT` is 0.
1. `CopyFlag`' is 1 or `AND` is twice `AND`' plus the product of `LhsLsb` and `RhsLsb`.
1. `CopyFlag`' is 1 or `XOR` is twice `XOR`' plus `LhsLsb` plus `RhsLsb` minus twice the product of `LhsLsb` and `RhsLsb`.
1. `CopyFlag`' is 1 or `LHS`' is not 0 or `LHS` is 0 or `Log2Floor` is `Bits`.
1. `CopyFlag`' is 1 or `LHS`' is 0 or `Log2Floor` is `Log2Floor`'.
1. `CopyFlag`' is 1 or `RhsLsb` is 1 or `Pow` is `Pow`' times `Pow`'.
1. `CopyFlag`' is 1 or `RhsLsb` is 0 or `Pow` is `Pow`' times `Pow`' times `LhsCopy`'.
1. `CopyFlag`' is 1 or `RunningProductProcessor`' is `RunningProductProcessor`.
1. `CopyFlag`' is 0 or `RunningProductProcessor`' is `RunningProductProcessor` times `(🧷 - 🥜·LHS' - 🌰·RHS' - 🥑·CI' - 🥕·Result')`.

### Transition Constraints as Polynomials

1. `CopyFlag'·LHS`
1. `CopyFlag'·RHS`
1. `(CopyFlag' - 1)·(CI' - CI)`
1. `(CopyFlag' - 1)·(LhsCopy' - LhsCopy)`
1. `(CopyFlag' - 1)·LHS·(Bits' - Bits - 1)`
1. `(CopyFlag' - 1)·RHS·(Bits' - Bits - 1)`
1. `(CopyFlag' - 1)·LhsLsb·(LhsLsb - 1)`
1. `(CopyFlag' - 1)·RhsLsb·(RhsLsb - 1)`
1. `(CopyFlag' - 1)·(LT' - 1)·(LT' - 2)·LT`
1. `(CopyFlag' - 1)·(LT' - 0)·(LT' - 2)·(LT - 1)`
1. `(CopyFlag' - 1)·(LT' - 0)·(LT' - 1)·(LhsLsb - 1)·RhsLsb·(LT - 1)`
1. `(CopyFlag' - 1)·(LT' - 0)·(LT' - 1)·LhsLsb·(RhsLsb - 1)·LT`
1. `(CopyFlag' - 1)·(LT' - 0)·(LT' - 1)·(1 - LhsLsb - RhsLsb + 2·LhsLsb·RhsLsb)·(CopyFlag - 1)·(LT - 2)`
1. `(CopyFlag' - 1)·(LT' - 0)·(LT' - 1)·(1 - LhsLsb - RhsLsb + 2·LhsLsb·RhsLsb)·CopyFlag·LT`
1. `(CopyFlag' - 1)·(AND - 2·AND' + LhsLsb·RhsLsb)`
1. `(CopyFlag' - 1)·(XOR - 2·XOR' + LhsLsb + RhsLsb - 2·LhsLsb·RhsLsb)`
1. `(CopyFlag' - 1)·(1 - LHS'·LhsInv')·LHS·(Log2Floor - Bits)`
1. `(CopyFlag' - 1)·LHS'·(Log2Floor' - Log2Floor)`
1. `(CopyFlag' - 1)·(RhsLsb - 1)·(Pow - Pow'·Pow')`
1. `(CopyFlag' - 1)·RhsLsb·(Pow - Pow'·Pow'·LhsCopy)`
1. `(CopyFlag' - 1)·(RunningProductProcessor' - RunningProductProcessor)`
1. `CopyFlag'·(RunningProductProcessor' - RunningProductProcessor·(🧷 - 🥜·LHS - 🌰·RHS - 🥑·CI - 🥕·Result))`

## Terminal Constraints

1. `LHS` is 0.
1. `RHS` is 0.

### Terminal Constraints as Polynomials

1. `LHS`
1. `RHS`
