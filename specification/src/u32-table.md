# U32 Table

The U32 Operations Table arithmetizes the coprocessor for “difficult” 32-bit unsigned integer operations.
The two inputs to the U32 Operations Table are left-hand side (LHS) and right-hand side (RHS), usually corresponding to the processor's `st0` and `st1`, respectively.
(For more details see the arithmetization of the specific [u32 instructions](instructions.md#bitwise-arithmetic) further below.)

To allow efficient arithmetization, a u32 instruction's result is constructed using multiple rows.
The collection of rows arithmetizing the execution of one instruction is called a _section_.
The U32 Table's sections are independent of each other.
The processor's current instruction `CI` is recorded within the section and dictates its arithmetization.

Crucially, the rows of the U32 table are independent of the processor's clock.
Hence, the result of the instruction can be transferred into the processor within one clock cycle.

## Main Columns

| name                 | description                                                                                     |
|:---------------------|:------------------------------------------------------------------------------------------------|
| `CopyFlag`           | The row to be copied between processor and u32 coprocessor. Marks the beginning of a section.   |
| `CI`                 | Current instruction, the instruction the processor is currently executing.                      |
| `Bits`               | The number of bits that LHS and RHS have already been shifted by.                               |
| `BitsMinus33Inv`     | The inverse-or-zero of the difference between 33 and `Bits`.                                    |
| `LHS`                | Left-hand side of the operation. Usually corresponds to the processor's `st0`.                  |
| `LhsInv`             | The inverse-or-zero of LHS. Needed to check whether `LHS` is unequal to 0.                      |
| `RHS`                | Right-hand side of the operation. Usually corresponds to the processor's `st1`.                 |
| `RhsInv`             | The inverse-or-zero of RHS. Needed to check whether `RHS` is unequal to 0.                      |
| `Result`             | The result (or intermediate result) of the instruction requested by the processor.              |
| `LookupMultiplicity` | The number of times the processor has executed the current instruction with the same arguments. |

An example U32 Table follows.
Some columns are omitted for presentation reasons.
All cells in the U32 Table contain elements of the B-field.
For clearer illustration of the mechanics, the columns marked “$_2$” are presented in base 2, whereas columns marked “$_{10}$” are in the usual base 10.

| CopyFlag$_2$ | CI          | Bits$_{10}$ | LHS$_2$    | RHS$_2$   | Result$_2$ | Result$_{10}$ |
|-------------:|:------------|------------:|:-----------|:----------|-----------:|--------------:|
|            1 | and         |           0 | 1100**0**  | 1101**0** |  1100**0** |            24 |
|            0 | and         |           1 | 110**0**   | 110**1**  |   110**0** |            12 |
|            0 | and         |           2 | 11**0**    | 11**0**   |    11**0** |             6 |
|            0 | and         |           3 | 1**1**     | 1**1**    |     1**1** |             3 |
|            0 | and         |           4 | **1**      | **1**     |      **1** |             1 |
|            0 | and         |           5 | **0**      | **0**     |      **0** |             0 |
|            1 | pow         |           0 | 10         | 10**1**   |     100000 |            32 |
|            0 | pow         |           1 | 10         | 1**0**    |        100 |             4 |
|            0 | pow         |           2 | 10         | **1**     |         10 |             2 |
|            0 | pow         |           3 | 10         | **0**     |          1 |             1 |
|            1 | log_2_floor |           0 | 10011**0** | 0         |        101 |             5 |
|            0 | log_2_floor |           1 | 1001**1**  | 0         |        101 |             5 |
|            0 | log_2_floor |           2 | 100**1**   | 0         |        101 |             5 |
|            0 | log_2_floor |           3 | 10**0**    | 0         |        101 |             5 |
|            0 | log_2_floor |           4 | 1**0**     | 0         |        101 |             5 |
|            0 | log_2_floor |           5 | **1**      | 0         |        101 |             5 |
|            0 | log_2_floor |           6 | **0**      | 0         |         -1 |            -1 |
|            1 | lt          |           0 | 11111      | 1101**1** |          0 |             0 |
|            0 | lt          |           1 | 1111       | 110**1**  |          0 |             0 |
|            0 | lt          |           2 | 111        | 11**0**   |          0 |             0 |
|            0 | lt          |           3 | 11         | 1**1**    |         10 |             2 |
|            0 | lt          |           4 | 1          | **1**     |         10 |             2 |
|            0 | lt          |           5 | 0          | **0**     |         10 |             2 |

The AIR verifies the correct update of each consecutive pair of rows.
For most instructions, the current least significant bit of both `LHS` and `RHS` is eliminated between two consecutive rows.
This eliminated bit is used to successively build the required result in the `Result` column.
For instruction `pow`, only the least significant bit `RHS` is eliminated, while `LHS` remains unchanged throughout the section.

There are 6 instructions the U32 Table is “aware” of: `split`, `lt`, `and`, `log_2_floor`, `pow`, and `pop_count`.
The instruction `split` uses the U32 Table for range checking only.
Concretely, the U32 Table ensures that the instruction `split`'s resulting “high bits” and “low bits” each fit in a u32.
Since the processor does not expect any result from instruction `split`, the `Result` must be 0 else the Lookup Argument fails.
Similarly, for instructions `log_2_floor` and `pop_count`, the processor always sets the `RHS` to 0.

For instruction `xor`, the processor requests the computation of the two arguments' `and` and converts the result using the following equality:

$$a \texttt{ xor } b = a + b - 2 \cdot (a \texttt{ and } b)$$

Credit for this trick goes, to the best of our knowledge, to [Daniel Lubarov](https://github.com/dlubarov).

For the remaining u32 instruction `div_mod`, the processor triggers the creation of two sections in the U32 Table:

- One section to ensure that the remainder `r` is smaller than the divisor `d`.
The processor requests the result of `lt` by setting the U32 Table's `CI` register to the opcode of `lt`.
This also guarantees that `r` and `d` each fit in a u32.
- One section to ensure that the quotient `q` and the numerator `n` each fit in a u32.
The processor needs no result, only the range checking capabilities, like for instruction `split`.
Consequently, the processor sets the U32 Table's `CI` register to the opcode of `split`.

If the current instruction is `lt`, the `Result` can take on the values 0, 1, or 2, where
- 0 means `LHS` >= `RHS` is definitely known in the current row,
- 1 means `LHS` < `RHS` is definitely known in the current row, and
- 2 means the result is unknown in the current row.
This is only an intermediate result.
It can never occur in the first row of a section, _i.e._, when `CopyFlag` is 1.

A new row with `CopyFlag = 1` can only be inserted if

1. `LHS` is 0 or the current instruction is `pow`, and
1. `RHS` is 0.

It is impossible to create a valid proof of correct execution of Triton VM if `Bits` is 33 in any row.

## Auxiliary Columns

The U32 Table has 1 auxiliary column, `U32LookupServerLogDerivative`.
It corresponds to the [Lookup Argument](lookup-argument.md) with the [Processor Table](processor-table.md), establishing that whenever the processor executes a u32 instruction, the following holds:

- the processor's requested left-hand side is copied into `LHS`,
- the processor's requested right-hand side is copied into `RHS`,
- the processor's requested u32 instruction is copied into `CI`, and
- the result `Result` is copied to the processor.

## Padding

Each padding row is the all-zero row with the exception of

- `CI`, which is the opcode of `split`, and
- `BitsMinus33Inv`, which is $-33^{-1}$.

Additionally, if the U32 Table is non-empty before applying padding, the padding row's columns `CI`, `LHS`, `LhsInv`, and `Result` are taken from the U32 Table's last row.

# Arithmetic Intermediate Representation

Let all household items (🪥, 🛁, etc.) be challenges, concretely evaluation points, supplied by the verifier.
Let all fruit & vegetables (🥝, 🥥, etc.) be challenges, concretely weights to compress rows, supplied by the verifier.
Both types of challenges are X-field elements, _i.e._, elements of $\mathbb{F}_{p^3}$.

## Initial Constraints

1. If the `CopyFlag` is 0, then `U32LookupServerLogDerivative` is 0.
Otherwise, the `U32LookupServerLogDerivative` has accumulated the first row with multiplicity `LookupMultiplicity` and with respect to challenges 🥜, 🌰, 🥑, and 🥕, and indeterminate 🧷.

### Initial Constraints as Polynomials

1. `(CopyFlag - 1)·U32LookupServerLogDerivative`<br />
    `+ CopyFlag·(U32LookupServerLogDerivative·(🧷 - 🥜·LHS - 🌰·RHS - 🥑·CI - 🥕·Result) - LookupMultiplicity)`

## Consistency Constraints

1. The `CopyFlag` is 0 or 1.
1. If `CopyFlag` is 1, then `Bits` is 0.
1. `BitsMinus33Inv` is the inverse of (`Bits` - 33).
1. `LhsInv` is 0 or the inverse of `LHS`.
1. `Lhs` is 0 or `LhsInv` is the inverse of `LHS`.
1. `RhsInv` is 0 or the inverse of `RHS`.
1. `Rhs` is 0 or `RhsInv` is the inverse of `RHS`.
1. If `CopyFlag` is 0 and the current instruction is `lt` and `LHS` is 0 and `RHS` is 0, then `Result` is 2.
1. If `CopyFlag` is 1 and the current instruction is `lt` and `LHS` is 0 and `RHS` is 0, then `Result` is 0.
1. If the current instruction is `and` and `LHS` is 0 and `RHS` is 0, then `Result` is 0.
1. If the current instruction is `pow` and `RHS` is 0, then `Result` is 1.
1. If `CopyFlag` is 0 and the current instruction is `log_2_floor` and `LHS` is 0, then `Result` is -1.
1. If `CopyFlag` is 1 and the current instruction is `log_2_floor` and `LHS` is 0, the VM crashes.
1. If `CopyFlag` is 0 and the current instruction is `pop_count` and `LHS` is 0, then `Result` is 0.
1. If `CopyFlag` is 0, then `LookupMultiplicity` is 0.

Written in Disjunctive Normal Form, the same constraints can be expressed as:

1. The `CopyFlag` is 0 or 1.
1. `CopyFlag` is 0 or `Bits` is 0.
1. `BitsMinus33Inv` the inverse of (`Bits` - 33).
1. `LhsInv` is 0 or the inverse of `LHS`.
1. `Lhs` is 0 or `LhsInv` is the inverse of `LHS`.
1. `RhsInv` is 0 or the inverse of `RHS`.
1. `Rhs` is 0 or `RhsInv` is the inverse of `RHS`.
1. `CopyFlag` is 1 or `CI` is the opcode of `split`, `and`, `pow`, `log_2_floor`, or `pop_count` or `LHS` is not 0 or `RHS` is not 0 or `Result` is 2.
1. `CopyFlag` is 0 or `CI` is the opcode of `split`, `and`, `pow`, `log_2_floor`, or `pop_count` or `LHS` is not 0 or `RHS` is not 0 or `Result` is 0.
1. `CI` is the opcode of `split`, `lt`, `pow`, `log_2_floor`, or `pop_count` or `LHS` is not 0 or `RHS` is not 0 or `Result` is 0.
1. `CI` is the opcode of `split`, `lt`, `and`, `log_2_floor`, or `pop_count` or `RHS` is not 0 or `Result` is 1.
1. `CopyFlag` is 1 or `CI` is the opcode of `split`, `lt`, `and`, `pow`, or `pop_count` or `LHS` is not 0 or `Result` is -1.
1. `CopyFlag` is 0 or `CI` is the opcode of `split`, `lt`, `and`, `pow`, or `pop_count` or `LHS` is not 0.
1. `CopyFlag` is 1 or `CI` is the opcode of `split`, `lt`, `and`, `pow`, or `log_2_floor` or `LHS` is not 0 or `Result` is 0.
1. `CopyFlag` is 1 or `LookupMultiplicity` is 0.

### Consistency Constraints as Polynomials

1. `CopyFlag·(CopyFlag - 1)`
1. `CopyFlag·Bits`
1. `1 - BitsMinus33Inv·(Bits - 33)`
1. `LhsInv·(1 - LHS·LhsInv)`
1. `LHS·(1 - LHS·LhsInv)`
1. `RhsInv·(1 - RHS·RhsInv)`
1. `RHS·(1 - RHS·RhsInv)`
1. `(CopyFlag - 1)·(CI - opcode(split))·(CI - opcode(and))·(CI - opcode(pow))·(CI - opcode(log_2_floor))·(CI - opcode(pop_count))·(1 - LHS·LhsInv)·(1 - RHS·RhsInv)·(Result - 2)`
1. `(CopyFlag - 0)·(CI - opcode(split))·(CI - opcode(and))·(CI - opcode(pow))·(CI - opcode(log_2_floor))·(CI - opcode(pop_count))·(1 - LHS·LhsInv)·(1 - RHS·RhsInv)·(Result - 0)`
1. `(CI - opcode(split))·(CI - opcode(lt))·(CI - opcode(pow))·(CI - opcode(log_2_floor))·(CI - opcode(pop_count))·(1 - LHS·LhsInv)·(1 - RHS·RhsInv)·Result`
1. `(CI - opcode(split))·(CI - opcode(lt))·(CI - opcode(and))·(CI - opcode(log_2_floor))·(CI - opcode(pop_count))·(1 - RHS·RhsInv)·(Result - 1)`
1. `(CopyFlag - 1)·(CI - opcode(split))·(CI - opcode(lt))·(CI - opcode(and))·(CI - opcode(pow))·(CI - opcode(pop_count))·(1 - LHS·LhsInv)·(Result + 1)`
1. `CopyFlag·(CI - opcode(split))·(CI - opcode(lt))·(CI - opcode(and))·(CI - opcode(pow))·(CI - opcode(pop_count))·(1 - LHS·LhsInv)`
1. `(CopyFlag - 1)·(CI - opcode(split))·(CI - opcode(lt))·(CI - opcode(and))·(CI - opcode(pow))·(CI - opcode(log_2_floor))·(1 - LHS·LhsInv)·(Result - 0)`
1. `(CopyFlag - 1)·LookupMultiplicity`

## Transition Constraints

Even though they are never explicitly represented, it is useful to alias the `LHS`'s and `RHS`'s _least significant bit_, or “lsb.”
Given two consecutive rows, the (current) least significant bit of `LHS` can be computed by subtracting twice the next row's `LHS` from the current row's `LHS`.
These aliases, _i.e._, `LhsLsb` := `LHS - 2·LHS'` and `RhsLsb` := `RHS - 2·RHS'`, are used throughout the following.

1. If the `CopyFlag` in the next row is 1 and the current instruction is not `pow`, then `LHS` in the current row is 0.
1. If the `CopyFlag` in the next row is 1, then `RHS` in the current row is 0.
1. If the `CopyFlag` in the next row is 0, then `CI` in the next row is `CI` in the current row.
1. If the `CopyFlag` in the next row is 0 and `LHS` in the current row is unequal to 0 and the current instruction is not `pow`, then `Bits` in the next row is `Bits` in the current row plus 1.
1. If the `CopyFlag` in the next row is 0 and `RHS` in the current row is unequal to 0, then `Bits` in the next row is `Bits` in the current row plus 1.
1. If the `CopyFlag` in the next row is 0 and the current instruction is not `pow`, then `LhsLsb` is either 0 or 1.
1. If the `CopyFlag` in the next row is 0, then `RhsLsb` is either 0 or 1.
1. If the `CopyFlag` in the next row is 0 and the current instruction is `lt` and `Result` in the next row is 0, then `Result` in the current row is 0.
1. If the `CopyFlag` in the next row is 0 and the current instruction is `lt` and `Result` in the next row is 1, then `Result` in the current row is 1.
1. If the `CopyFlag` in the next row is 0 and the current instruction is `lt` and `Result` in the next row is 2 and `LhsLsb` is 0 and `RhsLsb` is 1, then `Result` in the current row is 1.
1. If the `CopyFlag` in the next row is 0 and the current instruction is `lt` and `Result` in the next row is 2 and `LhsLsb` is 1 and `RhsLsb` is 0, then `Result` in the current row is 0.
1. If the `CopyFlag` in the next row is 0 and the current instruction is `lt` and `Result` in the next row is 2 and `LhsLsb` is `RhsLsb` and the `CopyFlag` in the current row is 0, then `Result` in the current row is 2.
1. If the `CopyFlag` in the next row is 0 and the current instruction is `lt` and `Result` in the next row is 2 and `LhsLsb` is `RhsLsb` and the `CopyFlag` in the current row is 1, then `Result` in the current row is 0.
1. If the `CopyFlag` in the next row is 0 and the current instruction is `and`, then `Result` in the current row is twice `Result` in the next row plus the product of `LhsLsb` and `RhsLsb`.
1. If the `CopyFlag` in the next row is 0 and the current instruction is `log_2_floor` and `LHS` in the next row is 0 and `LHS` in the current row is not 0, then `Result` in the current row is `Bits`.
1. If the `CopyFlag` in the next row is 0 and the current instruction is `log_2_floor` and `LHS` in the next row is not 0, then `Result` in the current row is `Result` in the next row.
1. If the `CopyFlag` in the next row is 0 and the current instruction is `pow`, then `LHS` remains unchanged.
1. If the `CopyFlag` in the next row is 0 and the current instruction is `pow` and `RhsLsb` in the current row is 0, then `Result` in the current row is `Result` in the next row squared.
1. If the `CopyFlag` in the next row is 0 and the current instruction is `pow` and `RhsLsb` in the current row is 1, then `Result` in the current row is `Result` in the next row squared times `LHS` in the current row.
1. If the `CopyFlag` in the next row is 0 and the current instruction is `pop_count`, then `Result` in the current row is `Result` in the next row plus `LhsLsb`.
1. If the `CopyFlag` in the next row is 0, then `U32LookupServerLogDerivative` in the next row is `U32LookupServerLogDerivative` in the current row.
1. If the `CopyFlag` in the next row is 1, then `U32LookupServerLogDerivative` in the next row has accumulated the next row with multiplicity `LookupMultiplicity` and with respect to challenges 🥜, 🌰, 🥑, and 🥕, and indeterminate 🧷.

Written in Disjunctive Normal Form, the same constraints can be expressed as:

1. `CopyFlag`' is 0 or `LHS` is 0 or `CI` is the opcode of `pow`.
1. `CopyFlag`' is 0 or `RHS` is 0.
1. `CopyFlag`' is 1 or `CI`' is `CI`.
1. `CopyFlag`' is 1 or `LHS` is 0 or `CI` is the opcode of `pow` or `Bits`' is `Bits` + 1.
1. `CopyFlag`' is 1 or `RHS` is 0 or `Bits`' is `Bits` + 1.
1. `CopyFlag`' is 1 or `CI` is the opcode of `pow` or `LhsLsb` is 0 or `LhsLsb` is 1.
1. `CopyFlag`' is 1 or `RhsLsb` is 0 or `RhsLsb` is 1.
1. `CopyFlag`' is 1 or `CI` is the opcode of `split`, `and`, `pow`, `log_2_floor`, or `pop_count` or (`Result`' is 1 or 2) or `Result` is 0.
1. `CopyFlag`' is 1 or `CI` is the opcode of `split`, `and`, `pow`, `log_2_floor`, or `pop_count` or (`Result`' is 0 or 2) or `Result` is 1.
1. `CopyFlag`' is 1 or `CI` is the opcode of `split`, `and`, `pow`, `log_2_floor`, or `pop_count` or (`Result`' is 0 or 1) or `LhsLsb` is 1 or `RhsLsb` is 0 or `Result` is 1.
1. `CopyFlag`' is 1 or `CI` is the opcode of `split`, `and`, `pow`, `log_2_floor`, or `pop_count` or (`Result`' is 0 or 1) or `LhsLsb` is 0 or `RhsLsb` is 1 or `Result` is 0.
1. `CopyFlag`' is 1 or `CI` is the opcode of `split`, `and`, `pow`, `log_2_floor`, or `pop_count` or (`Result`' is 0 or 1) or `LhsLsb` is unequal to `RhsLsb` or `CopyFlag` is 1 or `Result` is 2.
1. `CopyFlag`' is 1 or `CI` is the opcode of `split`, `and`, `pow`, `log_2_floor`, or `pop_count` or (`Result`' is 0 or 1) or `LhsLsb` is unequal to `RhsLsb` or `CopyFlag` is 0 or `Result` is 0.
1. `CopyFlag`' is 1 or `CI` is the opcode of `split`, `lt`, `pow`, `log_2_floor`, or `pop_count` or `Result` is twice `Result`' plus the product of `LhsLsb` and `RhsLsb`.
1. `CopyFlag`' is 1 or `CI` is the opcode of `split`, `lt`, `and`, `pow`, or `pop_count` or `LHS`' is not 0 or `LHS` is 0 or `Result` is `Bits`.
1. `CopyFlag`' is 1 or `CI` is the opcode of `split`, `lt`, `and`, `pow`, or `pop_count` or `LHS`' is 0 or `Result` is `Result`'.
1. `CopyFlag`' is 1 or `CI` is the opcode of `split`, `lt`, `and`, `log_2_floor`, or `pop_count` or `LHS`' is `LHS`.
1. `CopyFlag`' is 1 or `CI` is the opcode of `split`, `lt`, `and`, `log_2_floor`, or `pop_count` or `RhsLsb` is 1 or `Result` is `Result`' times `Result`'.
1. `CopyFlag`' is 1 or `CI` is the opcode of `split`, `lt`, `and`, `log_2_floor`, or `pop_count` or `RhsLsb` is 0 or `Result` is `Result`' times `Result`' times `LHS`.
1. `CopyFlag`' is 1 or `CI` is the opcode of `split`, `lt`, `and`, `pow`, or `log_2_floor` or `Result` is `Result`' plus `LhsLsb`.
1. `CopyFlag`' is 1 or `U32LookupServerLogDerivative`' is `U32LookupServerLogDerivative`.
1. `CopyFlag`' is 0 or the difference of `U32LookupServerLogDerivative`' and `U32LookupServerLogDerivative` times `(🧷 - 🥜·LHS' - 🌰·RHS' - 🥑·CI' - 🥕·Result')` is `LookupMultiplicity`'.

### Transition Constraints as Polynomials

1. `(CopyFlag' - 0)·LHS·(CI - opcode(pow))`
1. `(CopyFlag' - 0)·RHS`
1. `(CopyFlag' - 1)·(CI' - CI)`
1. `(CopyFlag' - 1)·LHS·(CI - opcode(pow))·(Bits' - Bits - 1)`
1. `(CopyFlag' - 1)·RHS·(Bits' - Bits - 1)`
1. `(CopyFlag' - 1)·(CI - opcode(pow))·LhsLsb·(LhsLsb - 1)`
1. `(CopyFlag' - 1)·RhsLsb·(RhsLsb - 1)`
1. `(CopyFlag' - 1)·(CI - opcode(split))·(CI - opcode(and))·(CI - opcode(pow))·(CI - opcode(log_2_floor))·(CI - opcode(pop_count))·(Result' - 1)·(Result' - 2)·(Result - 0)`
1. `(CopyFlag' - 1)·(CI - opcode(split))·(CI - opcode(and))·(CI - opcode(pow))·(CI - opcode(log_2_floor))·(CI - opcode(pop_count))·(Result' - 0)·(Result' - 2)·(Result - 1)`
1. `(CopyFlag' - 1)·(CI - opcode(split))·(CI - opcode(and))·(CI - opcode(pow))·(CI - opcode(log_2_floor))·(CI - opcode(pop_count))·(Result' - 0)·(Result' - 1)·(LhsLsb - 1)·(RhsLsb - 0)·(Result - 1)`
1. `(CopyFlag' - 1)·(CI - opcode(split))·(CI - opcode(and))·(CI - opcode(pow))·(CI - opcode(log_2_floor))·(CI - opcode(pop_count))·(Result' - 0)·(Result' - 1)·(LhsLsb - 0)·(RhsLsb - 1)·(Result - 0)`
1. `(CopyFlag' - 1)·(CI - opcode(split))·(CI - opcode(and))·(CI - opcode(pow))·(CI - opcode(log_2_floor))·(CI - opcode(pop_count))·(Result' - 0)·(Result' - 1)·(1 - LhsLsb - RhsLsb + 2·LhsLsb·RhsLsb)·(CopyFlag - 1)·(Result - 2)`
1. `(CopyFlag' - 1)·(CI - opcode(split))·(CI - opcode(and))·(CI - opcode(pow))·(CI - opcode(log_2_floor))·(CI - opcode(pop_count))·(Result' - 0)·(Result' - 1)·(1 - LhsLsb - RhsLsb + 2·LhsLsb·RhsLsb)·(CopyFlag - 0)·(Result - 0)`
1. `(CopyFlag' - 1)·(CI - opcode(split))·(CI - opcode(lt))·(CI - opcode(pow))·(CI - opcode(log_2_floor))·(CI - opcode(pop_count))·(Result - 2·Result' - LhsLsb·RhsLsb)`
1. `(CopyFlag' - 1)·(CI - opcode(split))·(CI - opcode(lt))·(CI - opcode(and))·(CI - opcode(pow))·(CI - opcode(pop_count))·(1 - LHS'·LhsInv')·LHS·(Result - Bits)`
1. `(CopyFlag' - 1)·(CI - opcode(split))·(CI - opcode(lt))·(CI - opcode(and))·(CI - opcode(pow))·(CI - opcode(pop_count))·LHS'·(Result' - Result)`
1. `(CopyFlag' - 1)·(CI - opcode(split))·(CI - opcode(lt))·(CI - opcode(and))·(CI - opcode(log_2_floor))·(CI - opcode(pop_count))·(LHS' - LHS)`
1. `(CopyFlag' - 1)·(CI - opcode(split))·(CI - opcode(lt))·(CI - opcode(and))·(CI - opcode(log_2_floor))·(CI - opcode(pop_count))·(RhsLsb - 1)·(Result - Result'·Result')`
1. `(CopyFlag' - 1)·(CI - opcode(split))·(CI - opcode(lt))·(CI - opcode(and))·(CI - opcode(log_2_floor))·(CI - opcode(pop_count))·(RhsLsb - 0)·(Result - Result'·Result'·LHS)`
1. `(CopyFlag' - 1)·(CI - opcode(split))·(CI - opcode(lt))·(CI - opcode(and))·(CI - opcode(pow))·(CI - opcode(log_2_floor))·(Result - Result' - LhsLsb)`
1. `(CopyFlag' - 1)·(U32LookupServerLogDerivative' - U32LookupServerLogDerivative)`
1. `(CopyFlag' - 0)·((U32LookupServerLogDerivative' - U32LookupServerLogDerivative)·(🧷 - 🥜·LHS - 🌰·RHS - 🥑·CI - 🥕·Result) - LookupMultiplicity')`

## Terminal Constraints

1. `LHS` is 0 or the current instruction is `pow`.
1. `RHS` is 0.

### Terminal Constraints as Polynomials

1. `LHS·(CI - opcode(pow))`
1. `RHS`
