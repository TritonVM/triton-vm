# Hash Table

The instruction `hash` hashes the Op Stack's 10 top-most elements in one cycle.
Similarly, the Sponge instructions `sponge_init`, `sponge_absorb`, and `sponge_squeeze` also all complete in one cycle.
The main processor achieves this by using a hash coprocessor.
The Hash Table is part of the arithmetization of that coprocessor, the other two parts being the [Cascade Table](cascade-table.md) and the [Lookup Table](lookup-table.md).
In addition to accelerating these [hashing instructions](instructions.md#hashing), the Hash Table helps with [program attestation](program-attestation.md) by hashing the [program](program-table.md).
    
Note: the Hash Table is not “aware” of instruction `sponge_absorb_mem`.
Instead, the processor requests a “regular” `sponge_absorb` from the Hash Table, fetching the to-be-absorbed elements from RAM instead of the stack.

The arithmetization for instruction `hash`, the Sponge instructions `sponge_init`, `sponge_absorb`, and `sponge_squeeze`, and for program hashing are quite similar.
The main differences are in updates to the `state` registers between executions of the pseudo-random permutation used in Triton VM, the permutation of [Tip5](https://eprint.iacr.org/2023/107.pdf).
A summary of the four instructions' mechanics:

- Instruction `hash`
    1. sets all the hash coprocessor's rate registers (`state_0` through `state_9`) to equal the processor's stack registers `state_0` through `state_9`,
    1. sets all the hash coprocessor's capacity registers (`state_10` through `state_15`) to 1,
    1. executes the 5 rounds of the Tip5 permutation,
    1. overwrites the processor's stack registers `state_0` through `state_4` with 0, and
    1. overwrites the processor's stack registers `state_5` through `state_9` with the hash coprocessor's registers `state_0` through `state_4`.
- Instruction `sponge_init`
    1. sets all the hash coprocessor's registers (`state_0` through `state_15`) to 0.
- Instruction `sponge_absorb`
    1. overwrites the hash coprocessor's rate registers (`state_0` through `state_9`) with the processor's stack registers `state_0` through `state_9`, and
    1. executes the 5 rounds of the Tip5 permutation.
- Instruction `sponge_squeeze`
    1. overwrites the processor's stack registers `state_0` through `state_9` with the hash coprocessor's rate registers (`state_0` through `state_9`), and
    1. executes the 5 rounds of the Tip5 permutation.

Program hashing happens in the initialization phase of Triton VM.
The to-be-executed program has no control over it.
Program hashing is mechanically identical to performing instruction `sponge_absorb` as often as is necessary to hash the entire program.
A notable difference is the source of the to-be-absorbed elements:
they come from program memory, not the processor (which is not running yet).
Once all instructions have been absorbed, the resulting digest is checked against the publicly claimed digest.

Due to the various similar but distinct tasks of the Hash Table, it has an explicit `Mode` register.
The four separate modes are `program_hashing`, `sponge`, `hash`, and `pad`, and they evolve in that order.
Changing the mode is only possible when the permutation has been applied in full, _i.e._, when the round number is 5.
Once mode `pad` is reached, it is not possible to change the mode anymore.
It is not possible to skip mode `program_hashing`:
the program is always hashed.
Skipping any or all of the modes `sponge`, `hash`, or `pad` is possible in principle:

- if no Sponge instructions are executed, mode `sponge` will be skipped,
- if no `hash` instruction is executed, mode `hash` will be skipped, and
- if the Hash Table does not require any padding, mode `pad` will be skipped.

The distinct modes translate into distinct sections in the Hash Table, which are recorded in order:
First, the entire Sponge's transition of hashing the program is recorded.
Then, the Hash Table records all Sponge instructions in the order the processor executed them.
Then, the Hash Table records all `hash` instructions in the order the processor executed them.
Lastly, as many [padding](arithmetization.md#padding) rows as necessary are inserted.
In total, this separation allows the processor to execute `hash` instructions without affecting the Sponge's state, and keeps [program hashing](program-attestation.md) independent from both.

Note that `state_0` through `state_3`, corresponding to those states that are being split-and-looked-up in the Tip5 permutation, are not stored as a single field element.
Instead, four limbs “highest”, “mid high”, “mid low”, and “lowest” are recorded in the Hash Table.
This (basically) corresponds to storing the result of $\sigma(R \cdot \texttt{state\_element})$.
In the Hash Table, the resulting limbs are 16 bit wide, and hence, there are only 4 limbs;
the split into 8-bit limbs happens in the [Cascade Table](cascade-table.md).
For convenience, this document occasionally refers to those states as if they were a single register.
This is an alias for
$(2^{48}\cdot\texttt{state\_i\_highest\_lkin} + 2^{32}\cdot\texttt{state\_i\_mid\_high\_lkin} + 2^{16}\cdot\texttt{state\_i\_mid\_low\_lkin} + \texttt{state\_i\_lowest\_lkin})\cdot R^{-1}$.

## Main Columns

The Hash Table has 67 main columns:

- The `Mode` indicator, as described above.
It takes value
    + $1$ for mode `program_hashing`,
    + $2$ for mode `sponge`,
    + $3$ for mode `hash`, and
    + $0$ for mode `pad`.
- Current instruction `CI`, holding the instruction the processor is currently executing.
This column is only relevant for mode `sponge`.
- Round number indicator `round_no`, which can be one of $\{0, \dots, 5\}$.
    The Tip5 permutation has 5 rounds, indexed $\{0, \dots, 4\}$.
    The round number 5 indicates that the Tip5 permutation has been applied in full.
- 16 columns `state_i_highest_lkin`, `state_i_mid_high_lkin`, `state_i_mid_low_lkin`, `state_i_lowest_lkin` for the to-be-looked-up value of `state_0` through `state_4`, each of which holds one 16-bit wide limb.
- 16 columns `state_i_highest_lkout`, `state_i_mid_high_lkout`, `state_i_mid_low_lkout`, `state_i_lowest_lkout` for the looked-up value of `state_0` through `state_4`, each of which holds one 16-bit wide limb.
- 12 columns `state_5` through `state_15`.
- 4 columns `state_i_inv` establishing correct decomposition of `state_0_*_lkin` through `state_3_*_lkin` into 16-bit wide limbs.
- 16 columns `constant_i`, which hold the round constant for the round indicated by `RoundNumber`, or 0 if no round with this round number exists.

## Auxiliary Columns

The Hash Table has 20 auxiliary columns:

- `RunningEvaluationReceiveChunk` for the [Evaluation Argument](evaluation-argument.md) for copying chunks of size $\texttt{rate}$ from the [Program Table](program-table.md).
Relevant for [program attestation](program-attestation.md).
- `RunningEvaluationHashInput` for the Evaluation Argument for copying the input to the hash function from the processor to the hash coprocessor,
- `RunningEvaluationHashDigest` for the Evaluation Argument for copying the hash digest from the hash coprocessor to the processor,
- `RunningEvaluationSponge` for the Evaluation Argument for copying the 10 next to-be-absorbed elements from the processor to the hash coprocessor or the 10 next squeezed elements from the hash coprocessor to the processor, depending on the instruction,
- 16 columns `state_i_limb_LookupClientLogDerivative` (for `i` $\in \{0, \dots, 3\}$ and `limb` $\in \{$`highest`, `mid_high`, `mid_low`, `lowest` $\}$) establishing correct lookup of the respective limbs in the [Cascade Table](cascade-table.md).

## Padding

Each padding row is the all-zero row with the exception of
- `CI`, which is the opcode of instruction `hash`,
- `state_i_inv` for `i` $\in \{0, \dots, 3\}$, which is $(2^{32} - 1)^{-1}$, and
- `constant_i` for `i` $\in \{0, \dots, 15\}$, which is the `i`th constant for round 0.

# Arithmetic Intermediate Representation

Let all household items (🪥, 🛁, etc.) be challenges, concretely evaluation points, supplied by the verifier.
Let all fruit & vegetables (🥝, 🥥, etc.) be challenges, concretely weights to compress rows, supplied by the verifier.
Both types of challenges are X-field elements, _i.e._, elements of $\mathbb{F}_{p^3}$.

## Initial Constraints

1. The `Mode` is `program_hashing`.
1. The round number is 0.
1. `RunningEvaluationReceiveChunk` has absorbed the first chunk of instructions with respect to indeterminate 🪣.
1. `RunningEvaluationHashInput` is 1.
1. `RunningEvaluationHashDigest` is 1.
1. `RunningEvaluationSponge` is 1.
1. For `i` $\in \{0, \dots, 3\}$ and `limb` $\in \{$`highest`, `mid_high`, `mid_low`, `lowest` $\}$:<br />
    `state_i_limb_LookupClientLogDerivative` has accumulated `state_i_limb_lkin` and `state_i_limb_lkout` with respect to challenges 🍒, 🍓 and indeterminate 🧺.

### Initial Constraints as Polynomials

1. `Mode - 1`
1. `round_no`
1. `RunningEvaluationReceiveChunk - 🪣 - (🪑^10 + state_0·🪑^9 + state_1·🪑^8 + state_2·🪑^7 + state_3·🪑^6 + state_4·🪑^5 + state_5·🪑^4 + state_6·🪑^3 + state_7·🪑^2 + state_8·🪑 + state_9)`
1. `RunningEvaluationHashInput - 1`
1. `RunningEvaluationHashDigest - 1`
1. `RunningEvaluationSponge - 1`
1. For `i` $\in \{0, \dots, 3\}$ and `limb` $\in \{$`highest`, `mid_high`, `mid_low`, `lowest` $\}$:<br />
    `state_i_limb_LookupClientLogDerivative·(🧺 - 🍒·state_i_limb_lkin - 🍓·state_i_limb_lkout) - 1`

## Consistency Constraints

1. The `Mode` is a valid mode, _i.e._, $\in \{0, \dots, 3\}$.
1. If the `Mode` is `program_hashing`, `hash`, or `pad`, then the current instruction is the opcode of `hash`.
1. If the `Mode` is `sponge`, then the current instruction is a Sponge instruction.
1. If the `Mode` is `pad`, then the `round_no` is 0.
1. If the current instruction `CI` is `sponge_init`, then the `round_no` is 0.
1. For `i` $\in\{10, \dots, 15\}$:
If the current instruction `CI` is `sponge_init`, then register `state_i` is 0.
(_Note_: the remaining registers, corresponding to the rate, are guaranteed to be 0 through the Evaluation Argument “Sponge” with the processor.)
1. For `i` $\in\{10, \dots, 15\}$:
If the round number is 0 and the current `Mode` is `hash`, then register `state_i` is 1.
1. For `i` $\in\{0, \dots, 3\}$:
ensure that decomposition of `state_i` is unique.
That is, if both high limbs of `state_i` are the maximum value, then both low limbs are 0[^oxfoi].
To make the corresponding polynomials low degree, register `state_i_inv` holds the inverse-or-zero of the re-composed highest two limbs of `state_i` subtracted from their maximum value.
Let `state_i_hi_limbs_minus_2_pow_32` be an alias for that difference:
`state_i_hi_limbs_minus_2_pow_32` ${}:= 2^{32} - 1 - 2^{16} \cdot{}$`state_i_highest_lk_in`${}-{}$`state_i_mid_high_lk_in`.
    1. If the two high limbs of `state_i` are both the maximum possible value, then the two low limbs of `state_i` are both 0.
    1. The `state_i_inv` is the inverse of `state_i_hi_limbs_minus_2_pow_32` or `state_i_inv` is 0.
    1. The `state_i_inv` is the inverse of `state_i_hi_limbs_minus_2_pow_32` or `state_i_hi_limbs_minus_2_pow_32` is 0.
1. The round constants adhere to the specification of Tip5.

### Consistency Constraints as Polynomials

1. `(Mode - 0)·(Mode - 1)·(Mode - 2)·(Mode - 3)`
1. `(Mode - 2)·(CI - opcode(hash))`
1. `(Mode - 0)·(Mode - 1)·(Mode - 3)`<br />
    ` ·(CI - opcode(sponge_init))·(CI - opcode(sponge_absorb))·(CI - opcode(sponge_squeeze))`
1. `(Mode - 1)·(Mode - 2)·(Mode - 3)·(round_no - 0)`
1. `(CI - opcode(hash))·(CI - opcode(sponge_absorb))·(CI - opcode(sponge_squeeze))·(round_no - 0)`
1. For `i` $\in\{10, \dots, 15\}$:<br />
    `·(CI - opcode(hash))·(CI - opcode(sponge_absorb))·(CI - opcode(sponge_squeeze))`<br />
    `·(state_i - 0)`
1. For `i` $\in\{10, \dots, 15\}$:<br />
    `(round_no - 1)·(round_no - 2)·(round_no - 3)·(round_no - 4)·(round_no - 5)`<br />
    `·(Mode - 0)·(Mode - 1)·(Mode - 2)`<br />
    `·(state_i - 1)`
1. For `i` $\in\{0, \dots, 3\}$:
define `state_i_hi_limbs_minus_2_pow_32 := 2^32 - 1 - 2^16·state_i_highest_lk_in - state_i_mid_high_lk_in`.
    1. `(1 - state_i_inv · state_i_hi_limbs_minus_2_pow_32)·(2^16·state_i_mid_low_lk_in + state_i_lowest_lk_in)`
    1. `(1 - state_i_inv · state_i_hi_limbs_minus_2_pow_32)·state_i_inv`
    1. `(1 - state_i_inv · state_i_hi_limbs_minus_2_pow_32)·state_i_hi_limbs_minus_2_pow_32`

## Transition Constraints

1. If the `round_no` is 5, then the `round_no` in the next row is 0.
1. If the `Mode` is not `pad` and the current instruction `CI` is not `sponge_init` and the `round_no` is not 5, then the `round_no` increments by 1.
1. If the `Mode` in the next row is `program_hashing` and the `round_no` in the next row is 0, then receive a chunk of instructions with respect to challenges 🪣 and 🪑.
1. If the `Mode` changes from `program_hashing`, then the [Evaluation Argument](evaluation-argument.md) of `state_0` through `state_4` with respect to indeterminate 🥬 equals the public program digest challenge, 🫑.
1. If the `Mode` is `program_hashing` and the `Mode` in the next row is `sponge`, then the current instruction in the next row is `sponge_init`.
1. If the `round_no` is not 5 and the current instruction `CI` is not `sponge_init`, then the current instruction does not change.
1. If the `round_no` is not 5 and the current instruction `CI` is not `sponge_init`, then the `Mode` does not change.
1. If the `Mode` is `sponge`, then the `Mode` in the next row is `sponge` or `hash` or `pad`.
1. If the `Mode` is `hash`, then the `Mode` in the next row is `hash` or `pad`.
1. If the `Mode` is `pad`, then the `Mode` in the next row is `pad`.
1. If the `round_no` in the next row is 0
and the `Mode` in the next row is either `program_hashing` or `sponge`
and the instruction in the next row is either `sponge_absorb` or `sponge_init`,
then the capacity's state registers don't change.
1. If the `round_no` in the next row is 0 and the current instruction in the next row is `sponge_squeeze`, then none of the state registers change.
1. If the `round_no` in the next row is 0 and the `Mode` in the next row is `hash`, then `RunningEvaluationHashInput` accumulates the next row with respect to challenges 🧄₀ through 🧄₉ and indeterminate 🚪.
Otherwise, it remains unchanged.
1. If the `round_no` in the next row is 5 and the `Mode` in the next row is `hash`, then `RunningEvaluationHashDigest` accumulates the next row with respect to challenges 🧄₀ through 🧄₄ and indeterminate 🪟.
Otherwise, it remains unchanged.
1. If the `round_no` in the next row is 0 and the `Mode` in the next row is `sponge`, then `RunningEvaluationSponge` accumulates the next row with respect to challenges 🧅 and 🧄₀ through 🧄₉ and indeterminate 🧽.
Otherwise, it remains unchanged.
1. For `i` $\in \{0, \dots, 3\}$ and `limb` $\in \{$`highest`, `mid_high`, `mid_low`, `lowest` $\}$:<br />
If the next round number is not 5 and the `mode` in the next row is not `pad` and the current instruction `CI` in the next row is not `sponge_init`, then `state_i_limb_LookupClientLogDerivative` has accumulated `state_i_limb_lkin'` and `state_i_limb_lkout'` with respect to challenges 🍒, 🍓 and indeterminate 🧺.
Otherwise, `state_i_limb_LookupClientLogDerivative` remains unchanged.
1. For `r` $\in\{0, \dots, 4\}$:<br />
If the `round_no` is `r`, the `state` registers adhere to the rules of applying round `r` of the Tip5 permutation.

### Transition Constraints as Polynomials

1. `(round_no - 0)·(round_no - 1)·(round_no - 2)·(round_no - 3)·(round_no - 4)·(round_no' - 0)`
1. `(Mode - 0)·(round_no - 5)·(CI - opcode(sponge_init))·(round_no' - round_no - 1)`
1. `RunningEvaluationReceiveChunk' - 🪣·RunningEvaluationReceiveChunk - (🪑^10 + state_0·🪑^9 + state_1·🪑^8 + state_2·🪑^7 + state_3·🪑^6 + state_4·🪑^5 + state_5·🪑^4 + state_6·🪑^3 + state_7·🪑^2 + state_8·🪑 + state_9)`
1. `(Mode - 0)·(Mode - 2)·(Mode - 3)·(Mode' - 1)·(🥬^5 + state_0·🥬^4 + state_1·🥬^3 + state_2·🥬^2 + state_3·🥬^1 + state_4 - 🫑)`
1. `(Mode - 0)·(Mode - 2)·(Mode - 3)·(Mode' - 2)·(CI' - opcode(sponge_init))`
1. `(round_no - 5)·(CI - opcode(sponge_init))·(CI' - CI)`
1. `(round_no - 5)·(CI - opcode(sponge_init))·(Mode' - Mode)`
1. `(Mode - 0)·(Mode - 1)·(Mode - 3)·(Mode' - 0)·(Mode' - 2)·(Mode' - 3)`
1. `(Mode - 0)·(Mode - 1)·(Mode - 2)·(Mode' - 0)·(Mode' - 3)`
1. `(Mode - 1)·(Mode - 2)·(Mode - 3)·(Mode' - 0)`
1. `(round_no' - 1)·(round_no' - 2)·(round_no' - 3)·(round_no' - 4)·(round_no' - 5)`<br />
    `·(Mode' - 3)·(Mode' - 0)`<br />
    `·(CI' - opcode(sponge_init))`<br />
    `·(🧄₁₀·(state_10' - state_10) + 🧄₁₁·(state_11' - state_11) + 🧄₁₂·(state_12' - state_12) + 🧄₁₃·(state_13' - state_13) + 🧄₁₄·(state_14' - state_14) + 🧄₁₅·(state_15' - state_15))`
1. `(round_no' - 1)·(round_no' - 2)·(round_no' - 3)·(round_no' - 4)·(round_no' - 5)`<br />
    `·(CI' - opcode(hash))·(CI' - opcode(sponge_init))·(CI' - opcode(sponge_absorb))`<br />
    `·(🧄₀·(state_0' - state_0) + 🧄₁·(state_1' - state_1) + 🧄₂·(state_2' - state_2) + 🧄₃·(state_3' - state_3) + 🧄₄·(state_4' - state_4)`<br />
    ` + 🧄₅·(state_5' - state_5) + 🧄₆·(state_6' - state_6) + 🧄₇·(state_7' - state_7) + 🧄₈·(state_8' - state_8) + 🧄₉·(state_9' - state_9)`<br />
    ` + 🧄₁₀·(state_10' - state_10) + 🧄₁₁·(state_11' - state_11) + 🧄₁₂·(state_12' - state_12) + 🧄₁₃·(state_13' - state_13) + 🧄₁₄·(state_14' - state_14) + 🧄₁₅·(state_15' - state_15))`
1. `(round_no' - 0)·(round_no' - 1)·(round_no' - 2)·(round_no' - 3)·(round_no' - 4)`<br />
    `·(RunningEvaluationHashInput' - 🚪·RunningEvaluationHashInput - 🧄₀·state_0' - 🧄₁·state_1' - 🧄₂·state_2' - 🧄₃·state_3' - 🧄₄·state_4' - 🧄₅·state_5' - 🧄₆·state_6' - 🧄₇·state_7' - 🧄₈·state_8' - 🧄₉·state_9')`<br />
    `+ (round_no' - 0)·(RunningEvaluationHashInput' - RunningEvaluationHashInput)`<br />
    `+ (Mode' - 3)·(RunningEvaluationHashInput' - RunningEvaluationHashInput)`
1. `(round_no' - 0)·(round_no' - 1)·(round_no' - 2)·(round_no' - 3)·(round_no' - 4)`<br />
    `·(Mode' - 0)·(Mode' - 1)·(Mode' - 2)`<br />
    `·(RunningEvaluationHashDigest' - 🪟·RunningEvaluationHashDigest - 🧄₀·state_0' - 🧄₁·state_1' - 🧄₂·state_2' - 🧄₃·state_3' - 🧄₄·state_4')`<br />
    `+ (round_no' - 5)·(RunningEvaluationHashDigest' - RunningEvaluationHashDigest)`<br />
    `+ (Mode' - 3)·(RunningEvaluationHashDigest' - RunningEvaluationHashDigest)`
1. `(round_no' - 1)·(round_no' - 2)·(round_no' - 3)·(round_no' - 4)·(round_no' - 5)`<br />
    `·(CI' - opcode(hash))`<br />
    `·(RunningEvaluationSponge' - 🧽·RunningEvaluationSponge - 🧅·CI' - 🧄₀·state_0' - 🧄₁·state_1' - 🧄₂·state_2' - 🧄₃·state_3' - 🧄₄·state_4' - 🧄₅·state_5' - 🧄₆·state_6' - 🧄₇·state_7' - 🧄₈·state_8' - 🧄₉·state_9')`<br />
    `+ (RunningEvaluationSponge' - RunningEvaluationSponge)·(round_no' - 0)`<br />
    `+ (RunningEvaluationSponge' - RunningEvaluationSponge)·(CI' - opcode(sponge_init))·(CI' - opcode(sponge_absorb))·(CI' - opcode(sponge_squeeze))`
1. For `i` $\in \{0, \dots, 3\}$ and `limb` $\in \{$`highest`, `mid_high`, `mid_low`, `lowest` $\}$:<br />
    `(round_no' - 5)·(Mode' - 0)·(CI' - opcode(sponge_init))·((state_i_limb_LookupClientLogDerivative' - state_i_limb_LookupClientLogDerivative)·(🧺 - 🍒·state_i_limb_lkin' - 🍓·state_i_limb_lkout') - 1)`<br />
    `+ (round_no' - 0)·(round_no' - 1)·(round_no' - 2)·(round_no' - 3)·(round_no' - 4)`<br />
    `·(state_i_limb_LookupClientLogDerivative' - state_i_limb_LookupClientLogDerivative)`<br />
    `+ (CI' - opcode(hash))·(CI' - opcode(sponge_absorb))·(CI' - opcode(sponge_squeeze))`<br />
    `·(state_i_limb_LookupClientLogDerivative' - state_i_limb_LookupClientLogDerivative)`
1. The remaining constraints are left as an exercise to the reader.
For hints, see the [Tip5 paper](https://eprint.iacr.org/2023/107.pdf).

## Terminal Constraints

1. If the `Mode` is `program_hashing`, then the [Evaluation Argument](evaluation-argument.md) of `state_0` through `state_4` with respect to indeterminate 🥬 equals the public program digest challenge, 🫑.
1. If the `Mode` is not `pad` and the current instruction `CI` is not `sponge_init`, then the `round_no` is 5.

### Terminal Constraints as Polynomials

1. `🥬^5 + state_0·🥬^4 + state_1·🥬^3 + state_2·🥬^2 + state_3·🥬 + state_4 - 🫑`
1. `(Mode - 0)·(CI - opcode(sponge_init))·(round_no - 5)`

[^oxfoi]: This is a special property of the Oxfoi prime.
