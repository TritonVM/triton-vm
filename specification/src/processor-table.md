# Processor Table

## Base Columns

The processor consists of all registers defined in the [Instruction Set Architecture](isa.md).
Each register is assigned a column in the processor table.

## Extension Colums

The Processor Table has the following extension columns, corresponding to [Evaluation Arguments](evaluation-argument.md), [Permutation Arguments](permutation-argument.md), and [Lookup Arguments](lookup-argument.md):

1. `RunningEvaluationStandardInput` for the Evaluation Argument with the input symbols.
1. `RunningEvaluationStandardOutput` for the Evaluation Argument with the output symbols.
1. `InstructionLookupClientLogDerivative` for the Lookup Argument with the [Program Table](program-table.md)
1. `RunningProductOpStackTable` for the Permutation Argument with the [OpStack Table](operational-stack-table.md).
1. `RunningProductRamTable` for the Permutation Argument with the [RAM Table](random-access-memory-table.md).
1. `RunningProductJumpStackTable` for the Permutation Argument with the [Jump Stack Table](jump-stack-table.md).
1. `RunningEvaluationHashInput` for the Evaluation Argument with the [Hash Table](hash-table.md) for copying the input to the hash function from the processor to the hash coprocessor.
1. `RunningEvaluationHashDigest` for the Evaluation Argument with the [Hash Table](hash-table.md) for copying the hash digest from the hash coprocessor to the processor.
1. `RunningEvaluationSponge` for the Evaluation Argument with the [Hash Table](hash-table.md) for copying the 10 next to-be-absorbed elements from the processor to the hash coprocessor or the 10 next squeezed elements from the hash coprocessor to the processor, depending on the instruction.
1. `U32LookupClientLogDerivative` for the Lookup Argument with the [U32 Table](u32-table.md).
1. `ClockJumpDifferenceLookupServerLogDerivative` for the Lookup Argument of clock jump differences with the [OpStack Table](operational-stack-table.md), the [RAM Table](random-access-memory-table.md), and the [JumpStack Table](jump-stack-table.md).

## Padding

A padding row is a copy of the Processor Table's last row with the following modifications:
1. column `clk` is increased by 1,
1. column `IsPadding` is set to 1,
1. column `cjd_mul` is set to 0,

A notable exception:
if the row with `clk` equal to 1 is a padding row, then the value of `cjd_mul` is not constrained in that row.
The reason for this exception is the lack of “awareness” of padding rows in the three memory-like tables.
In fact, all memory-like tables keep looking up clock jump differences in their padding section.
All these clock jumps are guaranteed to have magnitude 1 due to the [Permutation Arguments](permutation-argument.md) with the respective memory-like tables.

# Arithmetic Intermediate Representation

Let all household items (🪥, 🛁, etc.) be challenges, concretely evaluation points, supplied by the verifier.
Let all fruit & vegetables (🥝, 🥥, etc.) be challenges, concretely weights to compress rows, supplied by the verifier.
Both types of challenges are X-field elements, _i.e._, elements of $\mathbb{F}_{p^3}$.

Note, that the transition constraint's use of `some_column` vs `some_column_next` might be a little unintuitive.
For example, take the following part of some execution trace.

| Clock Cycle | Current Instruction |  st0 |  …  | st15 | Running Evaluation “To Hash Table”  | Running Evaluation “From Hash Table”    |
|:------------|:--------------------|-----:|:---:|-----:|:------------------------------------|:----------------------------------------|
| $i-1$       | `foo`               |   17 |  …  |   22 | $a$                                 | $b$                                     |
| $i$         | hash                |   17 |  …  |   22 | $🚪·a + \sum_j 🧄_j \cdot st_j$       | $b$                                     |
| $i+1$       | `bar`               | 1337 |  …  |   22 | $🚪·a + \sum_{j=0}^9 🧄_j \cdot st_j$ | $🪟·b + \sum_{j=0}^4 🧄_j \cdot st_{j+5}$ |

In order to verify the correctness of `RunningEvaluationHashInput`, the corresponding transition constraint needs to conditionally “activate” on row-tuple ($i-1$, $i$), where it is conditional on `ci_next` (not `ci`), and verifies absorption of the next row, _i.e._, row $i$.
However, in order to verify the correctness of `RunningEvaluationHashDigest`, the corresponding transition constraint needs to conditionally “activate” on row-tuple ($i$, $i+1$), where it is conditional on `ci` (not `ci_next`), and verifies absorption of the next row, _i.e._, row $i+1$.

## Initial Constraints

1. The cycle counter `clk` is 0.
1. The previous instruction `previous_instruction` is 0.
1. The instruction pointer `ip` is 0.
1. The jump address stack pointer `jsp` is 0.
1. The jump address origin `jso` is 0.
1. The jump address destination `jsd` is 0.
1. The operational stack element `st0` is 0.
1. The operational stack element `st1` is 0.
1. The operational stack element `st2` is 0.
1. The operational stack element `st3` is 0.
1. The operational stack element `st4` is 0.
1. The operational stack element `st5` is 0.
1. The operational stack element `st6` is 0.
1. The operational stack element `st7` is 0.
1. The operational stack element `st8` is 0.
1. The operational stack element `st9` is 0.
1. The operational stack element `st10` is 0.
1. The operational stack element `st11` is 0.
1. The operational stack element `st12` is 0.
1. The operational stack element `st13` is 0.
1. The operational stack element `st14` is 0.
1. The operational stack element `st15` is 0.
1. The operational stack pointer `osp` is 16.
1. The operational stack value `osv` is 0.
1. The RAM pointer `ramp` is 0.
1. The RAM value `ramv` is 0.
1. `RunningEvaluationStandardInput` is 1.
1. `RunningEvaluationStandardOutput` is 1.
1. `InstructionLookupClientLogDerivative` has absorbed the first row with respect to challenges 🥝, 🥥, and 🫐 and indeterminate 🪥.
1. `RunningProductOpStackTable` has absorbed the first row with respect to challenges 🍋, 🍊, 🍉, and 🫒 and indeterminate 🪤.
1. `RunningProductRamTable` has absorbed the first row with respect to challenges 🍍, 🍈, 🍎, and 🌽 and indeterminate 🛋.
1. `RunningProductJumpStackTable` has absorbed the first row with respect to challenges 🍇, 🍅, 🍌, 🍏, and 🍐 and indeterminate 🧴.
1. `RunningEvaluationHashInput` has absorbed the first row with respect to challenges 🧄₀ through 🧄₉ and indeterminate 🚪 if the current instruction is `hash`. Otherwise, it is 1.
1. `RunningEvaluationHashDigest` is 1.
1. `RunningEvaluationSponge` is 1.
1. `U32LookupClientLogDerivative` is 0.
1. `ClockJumpDifferenceLookupServerLogDerivative` is 0.

### Initial Constraints as Polynomials

1. `clk`
1. `previous_instruction`
1. `ip`
1. `jsp`
1. `jso`
1. `jsd`
1. `st0`
1. `st1`
1. `st2`
1. `st3`
1. `st4`
1. `st5`
1. `st6`
1. `st7`
1. `st8`
1. `st9`
1. `st10`
1. `st11`
1. `st12`
1. `st13`
1. `st14`
1. `st15`
1. `osp`
1. `osv`
1. `ramp`
1. `ramv`
1. `RunningEvaluationStandardInput - 1`
1. `RunningEvaluationStandardOutput - 1`
1. `InstructionLookupClientLogDerivative · (🪥 - 🥝·ip - 🥥·ci - 🫐·nia) - 1`
1. `RunningProductOpStackTable - (🪤 - 🍋·clk - 🍊·ib1 - 🍉·osp - 🫒·osv)`
1. `RunningProductRamTable - (🛋 - 🍍·clk - 🍈·ramp - 🍎·ramv - 🌽·previous_instruction)`
1. `RunningProductJumpStackTable - (🧴 - 🍇·clk - 🍅·ci - 🍌·jsp - 🍏·jso - 🍐·jsd)`
1. `(ci - opcode(hash))·(RunningEvaluationHashInput - 1)`<br />
    `+ hash_deselector·(RunningEvaluationHashInput - 🚪 - 🧄₀·st0 - 🧄₁·st1 - 🧄₂·st2 - 🧄₃·st3 - 🧄₄·st4 - 🧄₅·st5 - 🧄₆·st6 - 🧄₇·st7 - 🧄₈·st8 - 🧄₉·st9)`
1. `RunningEvaluationHashDigest - 1`
1. `RunningEvaluationSponge - 1`
1. `U32LookupClientLogDerivative`
1. `ClockJumpDifferenceLookupServerLogDerivative`

## Consistency Constraints

1. The composition of instruction bits `ib0` through `ib7` corresponds to the current instruction `ci`.
1. The instruction bit `ib0` is a bit.
1. The instruction bit `ib1` is a bit.
1. The instruction bit `ib2` is a bit.
1. The instruction bit `ib3` is a bit.
1. The instruction bit `ib4` is a bit.
1. The instruction bit `ib5` is a bit.
1. The instruction bit `ib6` is a bit.
1. The instruction bit `ib7` is a bit.
1. The padding indicator `IsPadding` is either 0 or 1.
1. If the current padding row is a padding row and `clk` is not 1, then the clock jump difference lookup multiplicity is 0.

### Consistency Constraints as Polynomials

1. `ci - (2^7·ib7 + 2^6·ib6 + 2^5·ib5 + 2^4·ib4 + 2^3·ib3 + 2^2·ib2 + 2^1·ib1 + 2^0·ib0)`
1. `ib0·(ib0 - 1)`
1. `ib1·(ib1 - 1)`
1. `ib2·(ib2 - 1)`
1. `ib3·(ib3 - 1)`
1. `ib4·(ib4 - 1)`
1. `ib5·(ib5 - 1)`
1. `ib6·(ib6 - 1)`
1. `ib7·(ib7 - 1)`
1. `IsPadding·(IsPadding - 1)`
1. `IsPadding·(clk - 1)·ClockJumpDifferenceLookupServerLogDerivative`

## Transition Constraints

Due to their complexity, instruction-specific constraints are defined [in their own section](instruction-specific-transition-constraints.md).
The following additional constraints also apply to every pair of rows.

1. The cycle counter `clk` increases by 1.
1. The padding indicator `IsPadding` is 0 or remains unchanged.
1. The current instruction `ci` in the current row is copied into `previous_instruction` in the next row or the next row is a padding row.
1. The running evaluation for standard input absorbs `st0` of the next row with respect to 🛏 if the current instruction is `read_io`, and remains unchanged otherwise.
1. The running evaluation for standard output absorbs `st0` of the next row with respect to 🧯 if the current instruction in the next row is `write_io`, and remains unchanged otherwise.
1. If the next row is not a padding row, the logarithmic derivative for the Program Table absorbs the next row with respect to challenges 🥝, 🥥, and 🫐 and indeterminate 🪥. Otherwise, it remains unchanged.
1. The running product for the OpStack Table absorbs the next row with respect to challenges 🍋, 🍊, 🍉, and 🫒 and indeterminate 🪤.
1. The running product for the RAM Table absorbs the next row with respect to challenges 🍍, 🍈, 🍎, and 🌽 and indeterminate 🛋.
1. The running product for the JumpStack Table absorbs the next row with respect to challenges 🍇, 🍅, 🍌, 🍏, and 🍐 and indeterminate 🧴.
1. If the current instruction in the next row is `hash`, the running evaluation “Hash Input” absorbs the next row with respect to challenges 🧄₀ through 🧄₉ and indeterminate 🚪. Otherwise, it remains unchanged.
1. If the current instruction is `hash`, the running evaluation “Hash Digest” absorbs the next row with respect to challenges 🧄₀ through 🧄₄ and indeterminate 🪟. Otherwise, it remains unchanged.
1. If the current instruction is `absorb_init`, `absorb`, or `squeeze`, then the running evaluation “Sponge” absorbs the current instruction and the next row with respect to challenges 🧅 and 🧄₀ through 🧄₉ and indeterminate 🧽. Otherwise, it remains unchanged.
1.  1. If the current instruction is `split`, then the logarithmic derivative for the Lookup Argument with the U32 Table accumulates `st0` and `st1` in the next row and `ci` in the current row with respect to challenges 🥜, 🌰, and 🥑, and indeterminate 🧷.
    1. If the current instruction is `lt`, `and`, `xor`, or `pow`, then the logarithmic derivative for the Lookup Argument with the U32 Table accumulates `st0`, `st1`, and `ci` in the current row and `st0` in the next row with respect to challenges 🥜, 🌰, 🥑, and 🥕, and indeterminate 🧷.
    1. If the current instruction is `log_2_floor`, then the logarithmic derivative for the Lookup Argument with the U32 Table accumulates `st0` and `ci` in the current row and `st0` in the next row with respect to challenges 🥜, 🥑, and 🥕, and indeterminate 🧷.
    1. If the current instruction is `div`, then the logarithmic derivative for the Lookup Argument with the U32 Table accumulates both
        1. `st0` in the next row and `st1` in the current row as well as the constants `opcode(lt)` and `1` with respect to challenges 🥜, 🌰, 🥑, and 🥕, and indeterminate 🧷.
        1. `st0` in the current row and `st1` in the next row as well as `opcode(split)` with respect to challenges 🥜, 🌰, and 🥑, and indeterminate 🧷.
    1. If the current instruction is `pop_count`, then the logarithmic derivative for the Lookup Argument with the U32 Table accumulates `st0` and `ci` in the current row and `st0` in the next row with respect to challenges 🥜, 🥑, and 🥕, and indeterminate 🧷.
    1. Else, _i.e._, if the current instruction is not a u32 instruction, the logarithmic derivative for the Lookup Argument with the U32 Table remains unchanged.
1. The running sum for the logarithmic derivative of the clock jump difference lookup argument accumulates the next row's `clk` with the appropriate multiplicity `cjd_mul` with respect to indeterminate 🪞.

### Transition Constraints as Polynomials

1. `clk' - (clk + 1)`
1. `IsPadding·(IsPadding' - IsPadding)`
1. `(1 - IsPadding')·(previous_instruction' - ci)`
1. `(ci - opcode(read_io))·(RunningEvaluationStandardInput' - RunningEvaluationStandardInput)`<br />
    `+ read_io_deselector·(RunningEvaluationStandardInput' - 🛏·RunningEvaluationStandardInput - st0')`
1. `(ci' - opcode(write_io))·(RunningEvaluationStandardOutput' - RunningEvaluationStandardOutput)`<br />
    `+ write_io_deselector'·(RunningEvaluationStandardOutput' - 🧯·RunningEvaluationStandardOutput - st0')`
1. `(1 - IsPadding') · ((InstructionLookupClientLogDerivative' - InstructionLookupClientLogDerivative) · (🛁 - 🥝·ip' - 🥥·ci' - 🫐·nia') - 1)`<br />
    `+ IsPadding'·(RunningProductInstructionTable' - RunningProductInstructionTable)`
1. `RunningProductOpStackTable' - RunningProductOpStackTable·(🪤 - 🍋·clk' - 🍊·ib1' - 🍉·osp' - 🫒·osv')`
1. `RunningProductRamTable' - RunningProductRamTable·(🛋 - 🍍·clk' - 🍈·ramp' - 🍎·ramv' - 🌽·previous_instruction')`
1. `RunningProductJumpStackTable' - RunningProductJumpStackTable·(🧴 - 🍇·clk' - 🍅·ci' - 🍌·jsp' - 🍏·jso' - 🍐·jsd')`
1. `(ci' - opcode(hash))·(RunningEvaluationHashInput' - RunningEvaluationHashInput)`<br />
    `+ hash_deselector'·(RunningEvaluationHashInput' - 🚪·RunningEvaluationHashInput - 🧄₀·st0' - 🧄₁·st1' - 🧄₂·st2' - 🧄₃·st3' - 🧄₄·st4' - 🧄₅·st5' - 🧄₆·st6' - 🧄₇·st7' - 🧄₈·st8' - 🧄₉·st9')`
1. `(ci - opcode(hash))·(RunningEvaluationHashDigest' - RunningEvaluationHashDigest)`<br />
    `+ hash_deselector·(RunningEvaluationHashDigest' - 🪟·RunningEvaluationHashDigest - 🧄₀·st5' - 🧄₁·st6' - 🧄₂·st7' - 🧄₃·st8' - 🧄₄·st9')`
1. `(ci - opcode(absorb_init))·(ci - opcode(absorb)·(ci - opcode(squeeze))·(RunningEvaluationHashDigest' - RunningEvaluationHashDigest)`<br />
    `+ (absorb_init_deselector + absorb_deselector + squeeze_deselector)`<br />
    `·(RunningEvaluationSponge' - 🧽·RunningEvaluationSponge - 🧅·ci - 🧄₀·st0' - 🧄₁·st1' - 🧄₂·st2' - 🧄₃·st3' - 🧄₄·st4' - 🧄₅·st5' - 🧄₆·st6' - 🧄₇·st7' - 🧄₈·st8' - 🧄₉·st9')`
1.  1. `split_deselector·((U32LookupClientLogDerivative' - U32LookupClientLogDerivative)·(🧷 - 🥜·st0' - 🌰·st1' - 🥑·ci) - 1)`
    1. `+ lt_deselector·((U32LookupClientLogDerivative' - U32LookupClientLogDerivative)·(🧷 - 🥜·st0 - 🌰·st1 - 🥑·ci - 🥕·st0') - 1)`
    1. `+ and_deselector·((U32LookupClientLogDerivative' - U32LookupClientLogDerivative)·(🧷 - 🥜·st0 - 🌰·st1 - 🥑·ci - 🥕·st0') - 1)`
    1. `+ xor_deselector·((U32LookupClientLogDerivative' - U32LookupClientLogDerivative)·(🧷 - 🥜·st0 - 🌰·st1 - 🥑·ci - 🥕·st0') - 1)`
    1. `+ pow_deselector·((U32LookupClientLogDerivative' - U32LookupClientLogDerivative)·(🧷 - 🥜·st0 - 🌰·st1 - 🥑·ci - 🥕·st0') - 1)`
    1. `+ log_2_floor_deselector·((U32LookupClientLogDerivative' - U32LookupClientLogDerivative)·(🧷 - 🥜·st0 - 🥑·ci - 🥕·st0') - 1)`
    1. `+ div_deselector·(`<br />
    &emsp;&emsp;`(U32LookupClientLogDerivative' - U32LookupClientLogDerivative)·(🧷 - 🥜·st0' - 🌰·st1 - 🥑·opcode(lt) - 🥕·1)·(🧷 - 🥜·st0 - 🌰·st1' - 🥑·opcode(split))`<br />
    &emsp;&emsp;`- (🧷 - 🥜·st0' - 🌰·st1 - 🥑·opcode(lt) - 🥕·1)`<br />
    &emsp;&emsp;`- (🧷 - 🥜·st0 - 🌰·st1' - 🥑·opcode(split))`<br />
    &emsp;`)`
    1. `+ pop_count_deselector·((U32LookupClientLogDerivative' - U32LookupClientLogDerivative)·(🧷 - 🥜·st0 - 🥑·ci - 🥕·st0') - 1)`
    1. `+ (1 - ib2)·(U32LookupClientLogDerivative' - U32LookupClientLogDerivative)`
1. `(ClockJumpDifferenceLookupServerLogDerivative' - ClockJumpDifferenceLookupServerLogDerivative)`<br />
    `·(🪞 - clk') - cjd_mul'`

## Terminal Constraints

1. In the last row, register “current instruction” `ci` is 0, corresponding to instruction `halt`.

### Terminal Constraints as Polynomials

1. `ci`
