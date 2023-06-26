# Program Table

The Program Table contains the entire program as a read-only list of [instruction opcodes](instructions.md) and their arguments.
The [processor](processor-table.md) looks up instructions and arguments using its [instruction pointer `ip`](registers.md).

For [program attestation](program-attestation.md), the program is [padded](program-attestation.md#mechanics) and sent to the [Hash Table](hash-table.md) in chunks of size 10, which is the $\texttt{rate}$ of the [Tip5 hash function][tip5].
Padding is one 1 followed by the minimal number of 0’s necessary to make the padded input length a multiple of the $\texttt{rate}$[^padding].

## Base Columns

The corresponding Program Table consists of 7 base columns:

| Column                   | Description                                                                               |
|:-------------------------|:------------------------------------------------------------------------------------------|
| `Address`                | an instruction's address                                                                  |
| `Instruction`            | the (opcode of the) instruction                                                           |
| `LookupMultiplicity`     | how often an instruction has been executed                                                |
| `AbsorbCount`            | `Address` modulo the Tip5 $\texttt{rate}$, which is 10                                    |
| `MaxMinusAbsorbCountInv` | the inverse-or-zero of $\texttt{rate} - 1 - \texttt{AbsorbCount}$                         |
| `IsHashInputPadding`     | padding indicator for absorbing the program into the Sponge                               |
| `IsTablePadding`         | padding indicator for rows only required due to the dominating length of some other table |

## Extension Columns

A Lookup Argument with the [Processor Table](processor-table.md) establishes that the processor has loaded the correct instruction (and its argument) from program memory.
To establish the program memory's side of the Lookup Argument, the Program Table has extension column, `InstructionLookupServerLogDerivative`.

For sending the padded program to the [Hash Table](hash-table.md), a combination of two evaluation arguments is used.
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
1. column `IsPadding` is set to 1.

Above procedure is iterated until the [necessary number of rows](arithmetization.md#padding) have been added.

# Arithmetic Intermediate Representation

Let all household items (🪥, 🛁, etc.) be challenges, concretely evaluation points, supplied by the verifier.
Let all fruit & vegetables (🥝, 🥥, etc.) be challenges, concretely weights to compress rows, supplied by the verifier.
Both types of challenges are X-field elements, _i.e._, elements of $\mathbb{F}_{p^3}$.

## Initial Constraints

1. The first address is 0.
1. The logarithmic derivative is 0.

### Initial Constraints as Polynomials

1. `Address`
1. `InstructionLookupServerLogDerivative`

## Consistency Constraints

1. The padding indicator `IsPadding` is either 0 or 1.

### Consistency Constraints as Polynomials

1. `IsPadding·(IsPadding - 1)`

## Transition Constraints

1. The address increases by 1.
1. The padding indicator `IsPadding` is 0 or remains unchanged.
1. If the current row is not a padding row, the logarithmic derivative accumulates the current row's address, the current row's instruction, and the next row's instruction with respect to challenges 🥝, 🥥, and 🫐 and indeterminate 🪥 respectively. Otherwise, it remains unchanged.

### Transition Constraints as Polynomials

1. `Address' - (Address + 1)`
1. `IsPadding·(IsPadding' - IsPadding)`
1. `(1 - IsPadding) · ((InstructionLookupServerLogDerivative' - InstructionLookupServerLogDerivative) · (🪥 - 🥝·Address - 🥥·Instruction - 🫐·Instruction') - LookupMultiplicity)`<br />
    ` + IsPadding · (InstructionLookupServerLogDerivative' - InstructionLookupServerLogDerivative)`

## Terminal Constraints

None.

---

[tip5]: https://eprint.iacr.org/2023/107.pdf

[^padding]:
See also section 2.5 “Fixed-Length versus Variable-Length” in the [Tip5 paper][tip5].
