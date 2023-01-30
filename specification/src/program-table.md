# Program Table

## Base Columns

The virtual machine's program memory is read-only.
The corresponding Program Table consists of 4 base columns:
`Address`, `Instruction`, `LookupMultiplicity`, and `IsPadding`.

| Address | Instruction | Lookup Multiplicity | Is Padding |
|:--------|:------------|:--------------------|:-----------|
| -       | -           | -                   | -          |

The column `LookupMultiplicity` indicates how often the instruction of the corresponding address has been looked up by the processor.

## Extension Columns

A Lookup Argument with the [Processor Table](processor-table.md) establishes that the processor has loaded the correct instruction (and its argument) from program memory.
To establish the program memory's side of the Lookup Argument, the Program Table has 1 extension column, `InstructionLookupServerLogDerivative`.

## Padding

A padding row is a copy of the Program Table's last row with the following modifications:
1. column `Address` is increased by 1,
1. column `Instruction` is set to 0, 
1. column `LookupMultiplicity` is set to 0, and
1. column `IsPadding` is set to 1.

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
