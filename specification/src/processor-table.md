# Processor Table

The Processor Table records all of Triton VM states during execution of a particular program.
The states are recorded in chronological order.
The first row is the initial state, the last (non-[padding](arithmetization.md#padding)) row is the terminal state, _i.e._, the state after having executed instruction `halt`.
It is impossible to generate a valid proof if the instruction executed last is anything but `halt`.

It is worth highlighting the initialization of the operational stack.
Stack elements `st0` through `st10` are initially 0.
However, stack elements `st11` through `st15`, _i.e._, the very bottom of the stack, are initialized with the hash digest of the program that is being executed.
This is primarily useful for recursive verifiers: 
they can compare their own program digest to the program digest of the proof they are verifying.
This way, a recursive verifier can easily determine if they are actually recursing, or whether the proof they are checking was generated using an entirely different program.
A more detailed explanation of the mechanics can be found on the page about [program attestation](program-attestation.md).

## Main Columns

The processor consists of all registers defined in the [Instruction Set Architecture](isa.md).
Each register is assigned a column in the processor table.

## Auxiliary Columns

The Processor Table has the following auxiliary columns, corresponding to [Evaluation Arguments](evaluation-argument.md), [Permutation Arguments](permutation-argument.md), and [Lookup Arguments](lookup-argument.md):

1. `RunningEvaluationStandardInput` for the Evaluation Argument with the input symbols.
1. `RunningEvaluationStandardOutput` for the Evaluation Argument with the output symbols.
1. `InstructionLookupClientLogDerivative` for the Lookup Argument with the [Program Table](program-table.md)
1. `RunningProductOpStackTable` for the Permutation Argument with the [Op Stack Table](operational-stack-table.md).
1. `RunningProductRamTable` for the Permutation Argument with the [RAM Table](random-access-memory-table.md). Note that virtual column `instruction_type` holds value 1 for reads and 0 for writes.
1. `RunningProductJumpStackTable` for the Permutation Argument with the [Jump Stack Table](jump-stack-table.md).
1. `RunningEvaluationHashInput` for the Evaluation Argument with the [Hash Table](hash-table.md) for copying the input to the hash function from the processor to the hash coprocessor.
1. `RunningEvaluationHashDigest` for the Evaluation Argument with the [Hash Table](hash-table.md) for copying the hash digest from the hash coprocessor to the processor.
1. `RunningEvaluationSponge` for the Evaluation Argument with the [Hash Table](hash-table.md) for copying the 10 next to-be-absorbed elements from the processor to the hash coprocessor or the 10 next squeezed elements from the hash coprocessor to the processor, depending on the instruction.
1. `U32LookupClientLogDerivative` for the Lookup Argument with the [U32 Table](u32-table.md).
1. `ClockJumpDifferenceLookupServerLogDerivative` for the Lookup Argument of clock jump differences with the [Op Stack Table](operational-stack-table.md), the [RAM Table](random-access-memory-table.md), and the [Jump Stack Table](jump-stack-table.md).

### Permutation Argument with the Op Stack Table

The subset [Permutation Argument](permutation-argument.md) with the [Op Stack Table](operational-stack-table.md) `RunningProductOpStackTable` establishes consistency of the op stack underflow memory.
The number of factors incorporated into the running product at any given cycle depends on the executed instruction in this cycle:
for every element pushed to or popped from the stack, there is one factor.
Namely, if the op stack grows, every element spilling from `st15` into op stack underflow memory will be incorporated as one factor;
and if the op stack shrinks, every element from op stack underflow memory being transferred into `st15` will be one factor.

Notably, if an instruction shrinks the op stack by more than one element in a single clock cycle, each spilled element is incorporated as one factor.
The same holds true for instructions growing the op stack by more than one element in a single clock cycle.

One key insight for this Permutation Argument is that the processor will always have access to the elements that are to be read from or written to underflow memory:
if the instruction grows the op stack, then the elements in question currently reside in the directly accessible, top part of the stack;
if the instruction shrinks the op stack, then the elements in question will be in the top part of the stack in the next cycle.
In either case, the [Transition Constraint](arithmetization.md#arithmetic-intermediate-representation) for the Permutation Argument can incorporate the explicitly listed elements as well as the corresponding trivial-to-compute `op_stack_pointer`.

## Padding

A padding row is a copy of the Processor Table's last row with the following modifications:
1. column `clk` is increased by 1,
1. column `IsPadding` is set to 1,
1. column `cjd_mul` is set to 0,

A notable exception:
if the row with `clk` equal to 1 is a padding row, then the value of `cjd_mul` is not constrained in that row.
The reason for this exception is the lack of “awareness” of padding rows in the [Jump Stack Table](jump-stack-table.md):
it keeps looking up clock jump differences in its padding section.
All these clock jumps are guaranteed to have magnitude 1.

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
1. The [Evaluation Argument](evaluation-argument.md) of operational stack elements `st11` through `st15` with respect to indeterminate 🥬 equals the public part of program digest challenge, 🫑.
See [program attestation](program-attestation.md) for more details.
1. The `op_stack_pointer` is 16.
1. `RunningEvaluationStandardInput` is 1.
1. `RunningEvaluationStandardOutput` is 1.
1. `InstructionLookupClientLogDerivative` has absorbed the first row with respect to challenges 🥝, 🥥, and 🫐 and indeterminate 🪥.
1. `RunningProductOpStackTable` is 1.
1. `RunningProductRamTable` has absorbed the first row with respect to challenges 🍍, 🍈, 🍎, and 🌽 and indeterminate 🛋.
1. `RunningProductJumpStackTable` has absorbed the first row with respect to challenges 🍇, 🍅, 🍌, 🍏, and 🍐 and indeterminate 🧴.
1. `RunningEvaluationHashInput` has absorbed the first row with respect to challenges 🧄₀ through 🧄₉ and indeterminate 🚪 if the current instruction is `hash`. Otherwise, it is 1.
1. `RunningEvaluationHashDigest` is 1.
1. `RunningEvaluationSponge` is 1.
1. `U32LookupClientLogDerivative` is 0.
1. `ClockJumpDifferenceLookupServerLogDerivative` starts having accumulated the first contribution.

## Consistency Constraints

1. The composition of instruction bits `ib0` through `ib6` corresponds to the current instruction `ci`.
1. The instruction bit `ib0` is a bit.
1. The instruction bit `ib1` is a bit.
1. The instruction bit `ib2` is a bit.
1. The instruction bit `ib3` is a bit.
1. The instruction bit `ib4` is a bit.
1. The instruction bit `ib5` is a bit.
1. The instruction bit `ib6` is a bit.
1. The padding indicator `IsPadding` is either 0 or 1.
1. If the current padding row is a padding row and `clk` is not 1, then the clock jump difference lookup multiplicity is 0.

## Transition Constraints

Due to their complexity, instruction-specific constraints are defined [in their own section](instruction-specific-transition-constraints.md).
The following additional constraints also apply to every pair of rows.

1. The cycle counter `clk` increases by 1.
1. The padding indicator `IsPadding` is 0 or remains unchanged.
1. If the next row is not a padding row, the logarithmic derivative for the Program Table absorbs the next row with respect to challenges 🥝, 🥥, and 🫐 and indeterminate 🪥. Otherwise, it remains unchanged.
1. The running sum for the logarithmic derivative of the clock jump difference lookup argument accumulates the next row's `clk` with the appropriate multiplicity `cjd_mul` with respect to indeterminate 🪞.
1. The running product for the Jump Stack Table absorbs the next row with respect to challenges 🍇, 🍅, 🍌, 🍏, and 🍐 and indeterminate 🧴.
1.  1. If the current instruction in the next row is `hash`, the running evaluation “Hash Input” absorbs the next row with respect to challenges 🧄₀ through 🧄₉ and indeterminate 🚪.
    1. If the current instruction in the next row is `merkle_step` or `merkle_step_mem` and helper variable `hv5`…
        1. …is 0, the running evaluation “Hash Input” absorbs next row's `st0` through `st4` and `hv0` through `hv4`…
        1. …is 1, the running evaluation “Hash Input” absorbs next row's `hv0` through `hv4` and `st0` through `st4`…<br>
    …with respect to challenges 🧄₀ through 🧄₉ and indeterminate 🚪.
    1. Otherwise, it remains unchanged.
1. If the current instruction is `hash`, `merkle_step`, or `merkle_step_mem`, the running evaluation “Hash Digest” absorbs the next row with respect to challenges 🧄₀ through 🧄₄ and indeterminate 🪟. Otherwise, it remains unchanged.
1. If the current instruction is `sponge_init`, then the running evaluation “Sponge” absorbs the current instruction and the Sponge's default initial state with respect to challenges 🧅 and 🧄₀ through 🧄₉ and indeterminate 🧽.
    Else if the current instruction is `sponge_absorb`, then the running evaluation “Sponge” absorbs the current instruction and the current row with respect to challenges 🧅 and 🧄₀ through 🧄₉ and indeterminate 🧽.
    Else if the current instruction is `sponge_squeeze`, then the running evaluation “Sponge” absorbs the current instruction and the next row with respect to challenges 🧅 and 🧄₀ through 🧄₉ and indeterminate 🧽.
    Else if the current instruction is `sponge_absorb_mem`, then the running evaluation “Sponge” absorbs the opcode of instruction `sponge_absorb` and stack elements `st1` through `st4` and all 6 helper variables with respect to challenges 🧅 and 🧄₀ through 🧄₉ and indeterminate 🧽.
    Otherwise, the running evaluation remains unchanged.
1.  1. If the current instruction is `split`, then the logarithmic derivative for the Lookup Argument with the U32 Table accumulates `st0` and `st1` in the next row and `ci` in the current row with respect to challenges 🥜, 🌰, and 🥑, and indeterminate 🧷.
    1. If the current instruction is `lt`, `and`, `xor`, or `pow`, then the logarithmic derivative for the Lookup Argument with the U32 Table accumulates `st0`, `st1`, and `ci` in the current row and `st0` in the next row with respect to challenges 🥜, 🌰, 🥑, and 🥕, and indeterminate 🧷.
    1. If the current instruction is `log_2_floor`, then the logarithmic derivative for the Lookup Argument with the U32 Table accumulates `st0` and `ci` in the current row and `st0` in the next row with respect to challenges 🥜, 🥑, and 🥕, and indeterminate 🧷.
    1. If the current instruction is `div_mod`, then the logarithmic derivative for the Lookup Argument with the U32 Table accumulates both
        1. `st0` in the next row and `st1` in the current row as well as the constants `opcode(lt)` and `1` with respect to challenges 🥜, 🌰, 🥑, and 🥕, and indeterminate 🧷.
        1. `st0` in the current row and `st1` in the next row as well as `opcode(split)` with respect to challenges 🥜, 🌰, and 🥑, and indeterminate 🧷.
    1. If the current instruction is `merkle_step` or `merkle_step_mem`, then the logarithmic derivative for the Lookup Argument with the U32 Table accumulates `st5` from the current and next rows as well as `opcode(split)` with respect to challenges 🥜, 🌰, and 🥑, and indeterminate 🧷.
    1. If the current instruction is `pop_count`, then the logarithmic derivative for the Lookup Argument with the U32 Table accumulates `st0` and `ci` in the current row and `st0` in the next row with respect to challenges 🥜, 🥑, and 🥕, and indeterminate 🧷.
    1. Else, _i.e._, if the current instruction is not a u32 instruction, the logarithmic derivative for the Lookup Argument with the U32 Table remains unchanged.

## Terminal Constraints

1. In the last row, register “current instruction” `ci` is 0, corresponding to instruction `halt`.
