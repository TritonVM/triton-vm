# Program Table

The Program Table contains the entire program as a read-only list of [instruction opcodes](instructions.md) and their arguments.
The [processor](processor-table.md) looks up instructions and arguments using its [instruction pointer `ip`](registers.md).

For [program attestation](program-attestation.md), the program is [padded](program-attestation.md#mechanics) and sent to the [Hash Table](hash-table.md) in chunks of size 10, which is the $\texttt{rate}$ of the [Tip5 hash function][tip5].
Program padding is one 1 followed by the minimal number of 0’s necessary to make the padded input length a multiple of the $\texttt{rate}$[^padding].

## Base Columns

The corresponding Program Table consists of 7 base columns:

| Column                   | Description                                                                                                     |
|:-------------------------|:----------------------------------------------------------------------------------------------------------------|
| `Address`                | an instruction's address                                                                                        |
| `Instruction`            | the (opcode of the) instruction                                                                                 |
| `LookupMultiplicity`     | how often an instruction has been executed                                                                      |
| `AbsorbCount`            | `Address` modulo the Tip5 $\texttt{rate}$, which is 10                                                          |
| `MaxMinusAbsorbCountInv` | the inverse-or-zero of $\texttt{rate} - 1 - \texttt{AbsorbCount}$                                               |
| `IsHashInputPadding`     | padding indicator for absorbing the program into the Sponge (see [program attestation](program-attestation.md)) |
| `IsTablePadding`         | padding indicator for rows only required due to the dominating length of some other table                       |

## Extension Columns

A [Lookup Argument](lookup-argument.md) with the [Processor Table](processor-table.md) establishes that the processor has loaded the correct instruction (and its argument) from program memory.
To establish the program memory's side of the Lookup Argument, the Program Table has extension column, `InstructionLookupServerLogDerivative`.

For sending the padded program to the [Hash Table](hash-table.md), a combination of two [Evaluation Arguments](evaluation-argument.md) is used.
The first, `PrepareChunkRunningEvaluation`, absorbs one chunk of $\texttt{rate}$ (_i.e._ 10) instructions at a time, after which it is reset and starts absorbing again.
The second, `SendChunkRunningEvaluation`, absorbs one such prepared chunk every 10 instructions.

## Padding

A padding row is a copy of the Program Table's last row with the following modifications:
1. column `Address` is increased by 1,
1. column `Instruction` is set to 0, 
1. column `LookupMultiplicity` is set to 0,
1. column `AbsorbCount` is set to `Address` mod $\texttt{rate}$,
1. column `MaxMinusAbsorbCountInv` is set to the inverse-or-zero of $\texttt{rate} - 1 - \texttt{AbsorbCount}$,
1. column `IsHashInputPadding` is set to 1, and
1. column `IsTablePadding` is set to 1.

Above procedure is iterated until the [necessary number of rows](arithmetization.md#padding) have been added.

As an exception to all other [table-linking arguments](table-linking.md), the Program Table's instruction [Lookup Argument](lookup-argument.md) records the argument's initial value in the first row.
This is necessary because an instruction's potential argument, or else the next instruction, is recorded in the next row.
Hence, verifying correct initialization of the logarithmic derivative requires access to both the current and the next row.
Only transition constraints can access two rows.
Therefore, the initial recorded value of the logarithmic derivative must be independent of the second row.
The logarithmic derivative's final value, allowing for a meaningful cross-table argument, is recorded in the first padding row.
This row is guaranteed to exist due to the hash-input padding mechanics.

# Arithmetic Intermediate Representation

Let all household items (🪥, 🛁, etc.) be challenges, concretely evaluation points, supplied by the verifier.
Let all fruit & vegetables (🥝, 🥥, etc.) be challenges, concretely weights to compress rows, supplied by the verifier.
Both types of challenges are X-field elements, _i.e._, elements of $\mathbb{F}_{p^3}$.

## Initial Constraints

1. The `Address` is 0.
1. The `AbsorbCount` is 0.
1. The indicator `IsHashInputPadding` is 0.
1. The `InstructionLookupServerLogDerivative` is 0.
1. `PrepareChunkRunningEvaluation` has absorbed `Instruction` with respect to challenge 🪑.
1. `SendChunkRunningEvaluation` is 1.

### Initial Constraints as Polynomials

1. `Address`
1. `AbsorbCount`
1. `IsHashInputPadding`
1. `InstructionLookupServerLogDerivative`
1. `PrepareChunkRunningEvaluation - 🪑 - Instruction`
1. `SendChunkRunningEvaluation - 1`

## Consistency Constraints

1. The `MaxMinusAbsorbCountInv` is zero or the inverse of $\texttt{rate} - 1 -{}$ `AbsorbCount`.
1. The `AbsorbCount` is $\texttt{rate} - 1$ or the `MaxMinusAbsorbCountInv` is the inverse of $\texttt{rate} - 1 -{}$ `AbsorbCount`.
1. Indicator `IsHashInputPadding` is either 0 or 1.
1. Indicator `IsTablePadding` is either 0 or 1.

### Consistency Constraints as Polynomials

1. `(1 - MaxMinusAbsorbCountInv · (rate - 1 - AbsorbCount)) · MaxMinusAbsorbCountInv`
1. `(1 - MaxMinusAbsorbCountInv · (rate - 1 - AbsorbCount)) · (rate - 1 - AbsorbCount)`
1. `IsHashInputPadding · (IsHashInputPadding - 1)`
1. `IsTablePadding · (IsTablePadding - 1)`

## Transition Constraints

1. The `Address` increases by 1.
1. If the `AbsorbCount` is not $\texttt{rate} - 1$, it increases by 1. Else, the `AbsorbCount` in the next row is 0.
1. The indicator `IsHashInputPadding` is 0 or remains unchanged.
1. The padding indicator `IsTablePadding` is 0 or remains unchanged.
1. If `IsHashInputPadding` is 0 in the current row and 1 in the next row, then `Instruction` in the next row is 1.
1. If `IsHashInputPadding` is 1 in the current row then `Instruction` in the next row is 0.
1. If `IsHashInputPadding` is 1 in the current row and `AbsorbCount` is $\texttt{rate} - 1$ in the current row then `IsTablePadding` is 1 in the next row.
1. If the current row is not a padding row, the logarithmic derivative accumulates the current row's address, the current row's instruction, and the next row's instruction with respect to challenges 🥝, 🥥, and 🫐 and indeterminate 🪥 respectively.
Otherwise, it remains unchanged.
1. If the `AbsorbCount` in the current row is not $\texttt{rate} - 1$, then `PrepareChunkRunningEvaluation` absorbs the `Instruction` in the next row with respect to challenge 🪑.
Otherwise, `PrepareChunkRunningEvaluation` resets and absorbs the `Instruction` in the next row with respect to challenge 🪑.
1. If the next row is not a padding row and the `AbsorbCount` in the next row is $\texttt{rate} - 1$, then `SendChunkRunningEvaluation` absorbs `PrepareChunkRunningEvaluation` in the next row with respect to variable 🪣.
Otherwise, it remains unchanged.

### Transition Constraints as Polynomials

1. `Address' - Address - 1`
1. `MaxMinusAbsorbCountInv · (AbsorbCount' - AbsorbCount - 1)`<br />
    ` + (1 - MaxMinusAbsorbCountInv · (rate - 1 - AbsorbCount)) · AbsorbCount'`
1. `IsHashInputPadding · (IsHashInputPadding' - IsHashInputPadding)`
1. `IsTablePadding · (IsTablePadding' - IsTablePadding)`
1. `(IsHashInputPadding - 1) · IsHashInputPadding' · (Instruction' - 1)`
1. `IsHashInputPadding · Instruction'`
1. `IsHashInputPadding · (1 - MaxMinusAbsorbCountInv · (rate - 1 - AbsorbCount)) · IsTablePadding'`
1. `(1 - IsHashInputPadding) · ((InstructionLookupServerLogDerivative' - InstructionLookupServerLogDerivative) · (🪥 - 🥝·Address - 🥥·Instruction - 🫐·Instruction') - LookupMultiplicity)`<br />
    ` + IsHashInputPadding · (InstructionLookupServerLogDerivative' - InstructionLookupServerLogDerivative)`
1. `(rate - 1 - AbsorbCount) · (PrepareChunkRunningEvaluation' - 🪑·PrepareChunkRunningEvaluation - Instruction')`<br />
    ` + (1 - MaxMinusAbsorbCountInv · (rate - 1 - AbsorbCount)) · (PrepareChunkRunningEvaluation' - 🪑 - Instruction')`
1. `(IsTablePadding' - 1) · (1 - MaxMinusAbsorbCountInv' · (rate - 1 - AbsorbCount')) · (SendChunkRunningEvaluation' - 🪣·SendChunkRunningEvaluation - PrepareChunkRunningEvaluation')`<br />
    ` + (SendChunkRunningEvaluation' - SendChunkRunningEvaluation) · IsTablePadding'`<br />
    ` + (SendChunkRunningEvaluation' - SendChunkRunningEvaluation) · (rate - 1 - AbsorbCount')`<br />

## Terminal Constraints

1. The indicator `IsHashInputPadding` is 1.
1. The `AbsorbCount` is $\texttt{rate} - 1$ or the indicator `IsTablePadding` is 1.

### Terminal Constraints as Polynomials

1. `IsHashInputPadding - 1`
1. `(rate - 1 - AbsorbCount) · (IsTablePadding - 1)`

---

[tip5]: https://eprint.iacr.org/2023/107.pdf

[^padding]:
See also section 2.5 “Fixed-Length versus Variable-Length” in the [Tip5 paper][tip5].
