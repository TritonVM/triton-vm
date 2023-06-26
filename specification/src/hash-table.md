# Hash Table

The instruction `hash` hashes the OpStack's 10 top-most elements in one cycle.
Similarly, the Sponge instructions `absorb_init`, `absorb`, and `squeeze` also all complete in one cycle.
The main processor achieves this by using a hash coprocessor.
The Hash Table is part of the arithmetization of that coprocessor, the other two parts being the [Cascade Table](cascade-table.md) and the [Lookup Table](lookup-table.md).
In addition to accelerating these [hashing instructions](instructions.md#hashing), the Hash Table helps with [program attestation](program-attestation.md) by hashing the [program](program-table.md).

The arithmetization for instruction `hash`, the Sponge instructions `absorb_init`, `absorb`, and `squeeze`, and for program hashing are quite similar.
The main differences are in updates to the `state` registers between executions of the pseudo-random permutation used in Triton VM, the permutation of [Tip5](https://eprint.iacr.org/2023/107.pdf).
A summary of the four instructions' mechanics:

- Instruction `hash`
    1. sets all the hash coprocessor's rate registers (`state0` through `state9`) to equal the processor's stack registers `st0` through `st9`,
    1. sets all the hash coprocessor's capacity registers (`state10` through `state15`) to 1,
    1. executes the 5 rounds of the Tip5 permutation,
    1. overwrites the processor's stack registers `st0` through `st4` with 0, and
    1. overwrites the processor's stack registers `st5` through `st9` with the hash coprocessor's registers `state0` through `state4`.
- Instruction `absorb_init`
    1. sets all the hash coprocessor's rate registers (`state0` through `state9`) to equal the processor's stack registers `st0` through `st9`,
    1. sets all the hash coprocessor's capacity registers (`state10` through `state15`) to 0, and
    1. executes the 5 rounds of the Tip5 permutation.
- Instruction `absorb`
    1. overwrites the hash coprocessor's rate registers (`state0` through `state9`) with the processor's stack registers `st0` through `st9`, and
    1. executes the 5 rounds of the Tip5 permutation.
- Instruction `squeeze`
    1. overwrites the processor's stack registers `st0` through `st9` with the hash coprocessor's rate registers (`state0` through `state9`), and
    1. executes the 5 rounds of the Tip5 permutation.

Program hashing happens in the initialization phase of Triton VM.
The to-be-executed program has no control over it.
Program hashing is mechanically identical to performing instruction `absorb` as often as is necessary to hash the entire program.
A notable difference is the source of the to-be-absorbed elements:
they come from program memory, not the processor (which is not running yet).
Once all instructions have been absorbed, the resulting digest is checked against the publicly claimed digest.

Due to the various similar but distinct tasks of the Hash Table, it has an explicit `mode` register.
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

Note that `state0` through `state3`, corresponding to those states that are being split-and-looked-up in the Tip5 permutation, are not stored as a single field element.
Instead, four limbs “highest”, “mid high”, “mid low”, and “lowest” are recorded in the Hash Table.
This (basically) corresponds to storing the result of $\sigma(R \cdot \texttt{state_element})$.
In the Hash Table, the resulting limbs are 16 bit wide, and hence, there are only 4 limbs;
the split into 8-bit limbs happens in the [Cascade Table](cascade-table.md).

## Base Columns

The Hash Table has 66 base columns:

- Round number indicator `round_no`, which can be one of $\{-1, 0, \dots, 5\}$.
    The Tip5 permutation has 5 rounds, indexed $\{0, \dots, 4\}$.
    The round number -1 indicates a padding row.
    The round number 5 indicates that the Tip5 permutation has been applied in full.
- Current instruction `CI`, holding the instruction the processor is currently executing.
- 16 columns `state_i_highest_lkin`, `state_i_midhigh_lkin`, `state_i_midlow_lkin`, `state_i_lowest_lkin` for the to-be-looked-up value of `state0` through `state4`, each of which holds one 16-bit wide limb.
- 16 columns `state_i_highest_lkout`, `state_i_midhigh_lkout`, `state_i_midlow_lkout`, `state_i_lowest_lkout` for the looked-up value of `state0` through `state4`, each of which holds one 16-bit wide limb.
- 12 columns `state5` through `state15`.
- 4 columns `state_i_inv` establishing correct decomposition of `state_0_*_lkin` through `state_4_*_lkin` into 16-bit wide limbs.
- 16 columns `constant_i`, which hold the round constant for the round indicated by `RoundNumber`, or 0 if no round with this round number exists.

## Extension Columns

The Hash Table has 19 extension columns:

- `RunningEvaluationHashInput` for the Evaluation Argument for copying the input to the hash function from the processor to the hash coprocessor,
- `RunningEvaluationHashDigest` for the Evaluation Argument for copying the hash digest from the hash coprocessor to the processor,
- `RunningEvaluationSponge` for the Evaluation Argument for copying the 10 next to-be-absorbed elements from the processor to the hash coprocessor or the 10 next squeezed elements from the hash coprocessor to the processor, depending on the instruction,
- 16 columns `state_i_limb_LookupClientLogDerivative` (for `i` $\in \{0, \dots, 3\}$, `limb` $\in \{$`highest`, `midhigh`, `midlow`, `lowest` $\}$) establishing correct lookup of the respective limbs in the [Cascade Table](cascade-table.md).

## Padding

Each padding row is the all-zero row with the exception of
- `round_no`, which is -1,
- `CI`, which is the opcode of instruction `hash`, and
- `state_i_inv` for `i` $\in \{0, \dots, 3\}$, which is $(2^{32} - 1)^{-1}$.

# Arithmetic Intermediate Representation

Let all household items (🪥, 🛁, etc.) be challenges, concretely evaluation points, supplied by the verifier.
Let all fruit & vegetables (🥝, 🥥, etc.) be challenges, concretely weights to compress rows, supplied by the verifier.
Both types of challenges are X-field elements, _i.e._, elements of $\mathbb{F}_{p^3}$.

## Initial Constraints

1. The round number is -1 or 0.
1. The current instruction is `hash` or `absorb_init`.
1. If the current instruction is `hash` and the round number is 0, then `RunningEvaluationHashInput` has accumulated the first row with respect to challenges 🧄₀ through 🧄₉ and indeterminate 🚪.
    Otherwise, `RunningEvaluationHashInput` is 1.
1. `RunningEvaluationHashDigest` is 1.
1. If the current instruction is `absorb_init`, then `RunningEvaluationSponge` has accumulated the first row with respect to challenges 🧅 and 🧄₀ through 🧄₉ and indeterminate 🧽.
    Otherwise, `RunningEvaluationSponge` is 1.
1. For `i` $\in \{0, \dots, 3\}$, `limb` $\in \{$`highest`, `midhigh`, `midlow`, `lowest` $\}$:<br />
    If the round number is 0, then `state_i_limb_LookupClientLogDerivative` has accumulated `state_i_limb_lkin` and `state_i_limb_lkout` with respect to challenges 🍒, 🍓 and indeterminate 🧺.
    Otherwise, `state_i_limb_LookupClientLogDerivative` is 0.

Written as Disjunctive Normal Form, the same constraints can be expressed as:

1. `round_no` is -1 or 0.
1. `CI` is the opcode of `hash` or of `absorb_init`.
1. (`CI` is the opcode of `absorb_init` or `round_no` is -1 or `RunningEvaluationHashInput` has accumulated the first row with respect to challenges 🧄₀ through 🧄₉ and indeterminate 🚪)<br />
    and (`CI` is the opcode of `hash` or `RunningEvaluationHashInput` is 1)<br />
    and (`round_no` is 0 or `RunningEvaluationHashInput` is 1).
1. (`CI` is the opcode of `hash` or `RunningEvaluationSponge` has accumulated the first row with respect to challenges 🧅 and 🧄₀ through 🧄₉ and indeterminate 🧽)<br />
    and (`CI` is the opcode of `absorb_init` or `RunningEvaluationSponge` is 1).
1. For `i` $\in \{0, \dots, 3\}$, `limb` $\in \{$`highest`, `midhigh`, `midlow`, `lowest` $\}$:<br />
    (`round_no` is -1 or `state_i_limb_LookupClientLogDerivative` has accumulated the first row)<br />
    and (`round_no` is 0 or `state_i_limb_LookupClientLogDerivative` is the default initial).

### Initial Constraints as Polynomials

1. `(round_no + 1)·round_no`
1. `(CI - opcode(hash))·(CI - opcode(absorb_init))`
1. `(CI - opcode(absorb_init))·(round_no + 1)·(RunningEvaluationHashInput - 🚪 - 🧄₀·st0 - 🧄₁·st1 - 🧄₂·st2 - 🧄₃·st3 - 🧄₄·st4 - 🧄₅·st5 - 🧄₆·st6 - 🧄₇·st7 - 🧄₈·st8 - 🧄₉·st9)`<br />
    `+ (CI - opcode(hash))·(RunningEvaluationHashInput - 1)`<br />
    `+ round_no·(RunningEvaluationHashInput - 1)`
1. `RunningEvaluationHashDigest - 1`
1. `(CI - opcode(hash))·(RunningEvaluationSponge - 🧽 - 🧅·CI - 🧄₀·st0 - 🧄₁·st1 - 🧄₂·st2 - 🧄₃·st3 - 🧄₄·st4 - 🧄₅·st5 - 🧄₆·st6 - 🧄₇·st7 - 🧄₈·st8 - 🧄₉·st9)`<br />
    `+ (CI - opcode(absorb_init))·(RunningEvaluationSponge - 1)`
1. For `i` $\in \{0, \dots, 3\}$, `limb` $\in \{$`highest`, `midhigh`, `midlow`, `lowest` $\}$:<br />
    `(round_no + 1)·(state_i_limb_LookupClientLogDerivative·(🧺 - 🍒·state_i_limb_lkin - 🍓·state_i_limb_lkout) - 1)`<br />
    `+ round_no·state_i_limb_LookupClientLogDerivative`

## Consistency Constraints

1. If the round number is -1, then the current instruction is `hash`.
1. If the round number is 0 and the current instruction is `hash`, then register `state10` is 1.
1. If the round number is 0 and the current instruction is `hash`, then register `state11` is 1.
1. If the round number is 0 and the current instruction is `hash`, then register `state12` is 1.
1. If the round number is 0 and the current instruction is `hash`, then register `state13` is 1.
1. If the round number is 0 and the current instruction is `hash`, then register `state14` is 1.
1. If the round number is 0 and the current instruction is `hash`, then register `state15` is 1.
1. If the round number is 0 and the current instruction is `absorb_init`, then register `state10` is 0.
1. If the round number is 0 and the current instruction is `absorb_init`, then register `state11` is 0.
1. If the round number is 0 and the current instruction is `absorb_init`, then register `state12` is 0.
1. If the round number is 0 and the current instruction is `absorb_init`, then register `state13` is 0.
1. If the round number is 0 and the current instruction is `absorb_init`, then register `state14` is 0.
1. If the round number is 0 and the current instruction is `absorb_init`, then register `state15` is 0.
1. If the round number is 0 and the current instruction is `absorb_init`, then register `state10` is 0.
1. The round constants adhere to the specification of Tip5.

Written as Disjunctive Normal Form, the same constraints can be expressed as:

1. The round number is 0 or 1 or 2 or 3 or 4 or 5 or `CI` is the opcode of `hash`.
1. The round number is -1 or 1 or 2 or 3 or 4 or 5 or `CI` is the opcode of `absorb_init` or `absorb` or `squeeze` or `state10` is 1.
1. The round number is -1 or 1 or 2 or 3 or 4 or 5 or `CI` is the opcode of `absorb_init` or `absorb` or `squeeze` or `state11` is 1.
1. The round number is -1 or 1 or 2 or 3 or 4 or 5 or `CI` is the opcode of `absorb_init` or `absorb` or `squeeze` or `state12` is 1.
1. The round number is -1 or 1 or 2 or 3 or 4 or 5 or `CI` is the opcode of `absorb_init` or `absorb` or `squeeze` or `state13` is 1.
1. The round number is -1 or 1 or 2 or 3 or 4 or 5 or `CI` is the opcode of `absorb_init` or `absorb` or `squeeze` or `state14` is 1.
1. The round number is -1 or 1 or 2 or 3 or 4 or 5 or `CI` is the opcode of `absorb_init` or `absorb` or `squeeze` or `state15` is 1.
1. The round number is -1 or 1 or 2 or 3 or 4 or 5 or `CI` is the opcode of `hash` or `absorb` or `squeeze` or `state10` is 0.
1. The round number is -1 or 1 or 2 or 3 or 4 or 5 or `CI` is the opcode of `hash` or `absorb` or `squeeze` or `state11` is 0.
1. The round number is -1 or 1 or 2 or 3 or 4 or 5 or `CI` is the opcode of `hash` or `absorb` or `squeeze` or `state12` is 0.
1. The round number is -1 or 1 or 2 or 3 or 4 or 5 or `CI` is the opcode of `hash` or `absorb` or `squeeze` or `state13` is 0.
1. The round number is -1 or 1 or 2 or 3 or 4 or 5 or `CI` is the opcode of `hash` or `absorb` or `squeeze` or `state14` is 0.
1. The round number is -1 or 1 or 2 or 3 or 4 or 5 or `CI` is the opcode of `hash` or `absorb` or `squeeze` or `state15` is 0.
1. The `constant_i` equals interpolant(`round_no`), where “interpolant” is the lowest-degree interpolant through (i, `constant_i`) for $-1 \leqslant i \leqslant 5$.

### Consistency Constraints as Polynomials

1. `(round_no - 0)·(round_no - 1)·(round_no - 2)·(round_no - 3)·(round_no - 4)·(round_no - 5)`<br />
    `·(CI - opcode(hash))`
1. `(round_no + 1)·(round_no - 1)·(round_no - 2)·(round_no - 3)·(round_no - 4)·(round_no - 5)`<br />
    `·(CI - opcode(absorb_init))·(CI - opcode(absorb))·(CI - opcode(squeeze))`<br />
    `·(state10 - 1)`
1. `(round_no + 1)·(round_no - 1)·(round_no - 2)·(round_no - 3)·(round_no - 4)·(round_no - 5)`<br />
    `·(CI - opcode(absorb_init))·(CI - opcode(absorb))·(CI - opcode(squeeze))`<br />
    `·(state11 - 1)`
1. `(round_no + 1)·(round_no - 1)·(round_no - 2)·(round_no - 3)·(round_no - 4)·(round_no - 5)`<br />
    `·(CI - opcode(absorb_init))·(CI - opcode(absorb))·(CI - opcode(squeeze))`<br />
    `·(state12 - 1)`
1. `(round_no + 1)·(round_no - 1)·(round_no - 2)·(round_no - 3)·(round_no - 4)·(round_no - 5)`<br />
    `·(CI - opcode(absorb_init))·(CI - opcode(absorb))·(CI - opcode(squeeze))`<br />
    `·(state13 - 1)`
1. `(round_no + 1)·(round_no - 1)·(round_no - 2)·(round_no - 3)·(round_no - 4)·(round_no - 5)`<br />
    `·(CI - opcode(absorb_init))·(CI - opcode(absorb))·(CI - opcode(squeeze))`<br />
    `·(state14 - 1)`
1. `(round_no + 1)·(round_no - 1)·(round_no - 2)·(round_no - 3)·(round_no - 4)·(round_no - 5)`<br />
    `·(CI - opcode(absorb_init))·(CI - opcode(absorb))·(CI - opcode(squeeze))`<br />
    `·(state15 - 1)`
1. `(round_no + 1)·(round_no - 1)·(round_no - 2)·(round_no - 3)·(round_no - 4)·(round_no - 5)`<br />
    `·(CI - opcode(hash))·(CI - opcode(absorb))·(CI - opcode(squeeze))`<br />
    `·state10`
1. `(round_no + 1)·(round_no - 1)·(round_no - 2)·(round_no - 3)·(round_no - 4)·(round_no - 5)`<br />
    `·(CI - opcode(hash))·(CI - opcode(absorb))·(CI - opcode(squeeze))`<br />
    `·state11`
1. `(round_no + 1)·(round_no - 1)·(round_no - 2)·(round_no - 3)·(round_no - 4)·(round_no - 5)`<br />
    `·(CI - opcode(hash))·(CI - opcode(absorb))·(CI - opcode(squeeze))`<br />
    `·state12`
1. `(round_no + 1)·(round_no - 1)·(round_no - 2)·(round_no - 3)·(round_no - 4)·(round_no - 5)`<br />
    `·(CI - opcode(hash))·(CI - opcode(absorb))·(CI - opcode(squeeze))`<br />
    `·state13`
1. `(round_no + 1)·(round_no - 1)·(round_no - 2)·(round_no - 3)·(round_no - 4)·(round_no - 5)`<br />
    `·(CI - opcode(hash))·(CI - opcode(absorb))·(CI - opcode(squeeze))`<br />
    `·state14`
1. `(round_no + 1)·(round_no - 1)·(round_no - 2)·(round_no - 3)·(round_no - 4)·(round_no - 5)`<br />
    `·(CI - opcode(hash))·(CI - opcode(absorb))·(CI - opcode(squeeze))`<br />
    `·state15`

## Transition Constraints

1. If the round number is -1, then the round number in the next row is -1.
1. If the round number is 1, 2, 3, or 4, then the round number in the next row is incremented by 1.
1. If the round number is 5, then the round number in the next row is either -1 or 0.
1. If the current instruction is `hash`, then the current instruction in the next row is `hash`.
1. If the round number is not 5, the current instruction in the next row is the current instruction in the current row.
1. If the round number in the next row is 0 and the current instruction in the next row is `absorb`, then the capacity's state registers don't change.
1. If the round number in the next row is 0 and the current instruction in the next row is `squeeze`, then none of the state registers change.
1. If the round number in the next row is 0 and the current instruction in the next row is `hash`, then `RunningEvaluationHashInput` accumulates the next row with respect to challenges 🧄₀ through 🧄₉ and indeterminate 🚪. Otherwise, it remains unchanged.
1. If the round number in the next row is 5 and the current instruction in the next row is `hash`, then `RunningEvaluationHashDigest` accumulates the next row with respect to challenges 🧄₀ through 🧄₄ and indeterminate 🪟. Otherwise, it remains unchanged.
1.  1. If the round number in the next row is 0 and the current instruction in the next row is `absorb_init`, `absorb`, or `squeeze`, then `RunningEvaluationSponge` accumulates the next row with respect to challenges 🧅 and 🧄₀ through 🧄₉ and indeterminate 🧽.
    1. If the round number in the next row is not 0, then `RunningEvaluationSponge` remains unchanged.
    1. If the current instruction in the next row is `hash`, then `RunningEvaluationSponge` remains unchanged.
1. If the round number is 0, the `state` registers adhere to the rules of applying round 0 of the Tip5 permutation.
1. If the round number is 1, the `state` registers adhere to the rules of applying round 1 of the Tip5 permutation.
1. If the round number is 2, the `state` registers adhere to the rules of applying round 2 of the Tip5 permutation.
1. If the round number is 3, the `state` registers adhere to the rules of applying round 3 of the Tip5 permutation.
1. If the round number is 4, the `state` registers adhere to the rules of applying round 4 of the Tip5 permutation.
1. For `i` $\in \{0, \dots, 3\}$, `limb` $\in \{$`highest`, `midhigh`, `midlow`, `lowest` $\}$:<br />
    If the next round number is 0, 1, 2, 3, or 4, then `state_i_limb_LookupClientLogDerivative` has accumulated `state_i_limb_lkin'` and `state_i_limb_lkout'` with respect to challenges 🍒, 🍓 and indeterminate 🧺.
    Otherwise, `state_i_limb_LookupClientLogDerivative` remains unchanged.

Written as Disjunctive Normal Form, the same constraints can be expressed as:

1. `round_no` is 0 or 1 or 2 or 3 or 4 or 5 or `round_no'` is -1.
1. `round_no` is -1 or 5 or `round_no'` is `round_no` + 1.
1. `round_no` is -1 or 0 or 1 or 2 or 3 or 4 or `round_no'` is -1 or 0.
1. `CI` is the opcode of `absorb_init` or `absorb` or `squeeze` or `CI'` is the opcode of `hash`.
1. `round_no` is 5 or `CI'` is `CI`.
1. `round_no'` is -1 or 1 or 2 or 3 or 4 or 5 or `CI'` is the opcode of `hash` or `absorb_init` or `squeeze` or the $🧄_i$-randomized sum of differences of the state registers `state10` through `state15` in the next row and the current row is 0.
1. `round_no'` is -1 or 1 or 2 or 3 or 4 or 5 or `CI'` is the opcode of `hash` or `absorb_init` or `absorb` or the $🧄_i$-randomized sum of differences of all state registers in the next row and the current row is 0.
1. (`round_no'` is -1 or 1 or 2 or 3 or 4 or 5 or `CI'` is the opcode of `absorb_init` or `absorb` or `squeeze` or `RunningEvaluationHashInput` accumulates the next row)<br />
    and (`round_no'` is 0 or `RunningEvaluationHashInput` remains unchanged)<br />
    and (`CI'` is the opcode of `hash` or `RunningEvaluationHashInput` remains unchanged).
1. (`round_no'` is -1 or 0 or 1 or 2 or 3 or 4 or `CI'` is the opcode of `absorb_init` or `absorb` or `squeeze` or `RunningEvaluationHashDigest` accumulates the next row)<br />
    and (`round_no'` is 5 or `RunningEvaluationHashDigest` remains unchanged)<br />
    and (`CI'` is the opcode of `hash` or `RunningEvaluationHashDigest` remains unchanged).

1.  1. (`round_no'` is -1 or 1 or 2 or 3 or 4 or 5 or `CI'` is the opcode of `hash` or `RunningEvaluationSponge` accumulates the next row)
    1. and (`round_no'` is 0 or `RunningEvaluationSponge` remains unchanged)
    1. and (`CI'` is the opcode of `absorb_init` or `absorb` or `squeeze` or `RunningEvaluationSponge` remains unchanged).
1. `round_no` is -1 or 1 or 2 or 3 or 4 or 5 or the `state` registers adhere to the rules of applying round 0 of the Tip5 permutation.
1. `round_no` is -1 or 0 or 2 or 3 or 4 or 5 or the `state` registers adhere to the rules of applying round 1 of the Tip5 permutation.
1. `round_no` is -1 or 0 or 1 or 3 or 4 or 5 or the `state` registers adhere to the rules of applying round 2 of the Tip5 permutation.
1. `round_no` is -1 or 0 or 1 or 2 or 4 or 5 or the `state` registers adhere to the rules of applying round 3 of the Tip5 permutation.
1. `round_no` is -1 or 0 or 1 or 2 or 3 or 5 or the `state` registers adhere to the rules of applying round 4 of the Tip5 permutation.
1. For `i` $\in \{0, \dots, 3\}$, `limb` $\in \{$`highest`, `midhigh`, `midlow`, `lowest` $\}$:<br />
    (`round_no'` is -1 or 5 or `state_i_limb_LookupClientLogDerivative` has accumulated the next row)<br />
    and (`round_no` is 0 or 1 or 2 or 3 or 4 or `state_i_limb_LookupClientLogDerivative'` is `state_i_limb_LookupClientLogDerivative`).

### Transition Constraints as Polynomials

1. `(round_no - 0)·(round_no - 1)·(round_no - 2)·(round_no - 3)·(round_no - 4)·(round_no' + 1)`
1. `(round_no + 1)·(round_no - 5)·(round_no' - round_no - 1)`
1. `(round_no + 1)·(round_no - 0)·(round_no - 1)·(round_no - 2)·(round_no - 3)·(round_no - 4)·(round_no' + 1)·(round_no' - 0)`
1. `(CI - opcode(absorb_init))·(CI - opcode(absorb))·(CI - opcode(squeeze))·(CI' - opcode(hash))`
1. `(round_no - 5)·(CI' - CI)`
1. `(round_no' + 1)·(round_no' - 1)·(round_no' - 2)·(round_no' - 3)·(round_no' - 4)·(round_no' - 5)`<br />
    `·(CI' - opcode(hash))·(CI' - opcode(absorb_init))·(CI' - opcode(squeeze))`<br />
    `·(🧄₁₀·(st10' - st10) + 🧄₁₁·(st11' - st11) + 🧄₁₂·(st12' - st12) + 🧄₁₃·(st13' - st13) + 🧄₁₄·(st14' - st14) + 🧄₁₅·(st15' - st15))`
1. `(round_no' + 1)·(round_no' - 1)·(round_no' - 2)·(round_no' - 3)·(round_no' - 4)·(round_no' - 5)`<br />
    `·(CI' - opcode(hash))·(CI' - opcode(absorb_init))·(CI' - opcode(absorb))`<br />
    `·(🧄₀·(st0' - st0) + 🧄₁·(st1' - st1) + 🧄₂·(st2' - st2) + 🧄₃·(st3' - st3) + 🧄₄·(st4' - st4) + 🧄₅·(st5' - st5) + 🧄₆·(st6' - st6) + 🧄₇·(st7' - st7) + 🧄₈·(st8' - st8) + 🧄₉·(st9' - st9) + 🧄₁₀·(st10' - st10) + 🧄₁₁·(st11' - st11) + 🧄₁₂·(st12' - st12) + 🧄₁₃·(st13' - st13) + 🧄₁₄·(st14' - st14) + 🧄₁₅·(st15' - st15))`
1. `(round_no' + 1)·(round_no' - 1)·(round_no' - 2)·(round_no' - 3)·(round_no' - 4)·(round_no' - 5)`<br />
    `·(CI' - opcode(absorb_init))·(CI' - opcode(absorb))·(CI' - opcode(squeeze))`<br />
    `·(RunningEvaluationHashInput' - 🚪·RunningEvaluationHashInput - 🧄₀·st0' - 🧄₁·st1' - 🧄₂·st2' - 🧄₃·st3' - 🧄₄·st4' - 🧄₅·st5' - 🧄₆·st6' - 🧄₇·st7' - 🧄₈·st8' - 🧄₉·st9')`<br />
    `+ round_no'·(RunningEvaluationHashInput' - RunningEvaluationHashInput)`<br />
    `+ (CI' - opcode(hash))·(RunningEvaluationHashInput' - RunningEvaluationHashInput)`
1. `(round_no' + 1)·(round_no' - 0)·(round_no' - 1)·(round_no' - 2)·(round_no' - 3)·(round_no' - 4)`<br />
    `·(CI' - opcode(absorb_init))·(CI' - opcode(absorb))·(CI' - opcode(squeeze))`<br />
    `·(RunningEvaluationHashDigest' - 🪟·RunningEvaluationHashDigest - 🧄₀·st0' - 🧄₁·st1' - 🧄₂·st2' - 🧄₃·st3' - 🧄₄·st4')`<br />
    `+ (round_no' - 5)·(RunningEvaluationHashDigest' - RunningEvaluationHashDigest)`<br />
    `+ (CI' - opcode(hash))·(RunningEvaluationHashDigest' - RunningEvaluationHashDigest)`
1.  1. `(round_no' + 1)·(round_no' - 1)·(round_no' - 2)·(round_no' - 3)·(round_no' - 4)·(round_no' - 5)`<br />
    `·(CI' - opcode(hash))`<br />
    `·(RunningEvaluationSponge' - 🧽·RunningEvaluationSponge - 🧅·CI' - 🧄₀·st0' - 🧄₁·st1' - 🧄₂·st2' - 🧄₃·st3' - 🧄₄·st4' - 🧄₅·st5' - 🧄₆·st6' - 🧄₇·st7' - 🧄₈·st8' - 🧄₉·st9')`<br />
    1. `+ (round_no' - 0)·(RunningEvaluationSponge' - RunningEvaluationSponge)`<br />
    1. `+ (CI' - opcode(absorb_init))·(CI' - opcode(absorb))·(CI' - opcode(squeeze))·(RunningEvaluationSponge' - RunningEvaluationSponge)`
1. For `i` $\in \{0, \dots, 3\}$, `limb` $\in \{$`highest`, `midhigh`, `midlow`, `lowest` $\}$:<br />
    `(round_no + 1)·(round_no - 5)·((state_i_limb_LookupClientLogDerivative' - state_i_limb_LookupClientLogDerivative)·(🧺 - 🍒·state_i_limb_lkin' - 🍓·state_i_limb_lkout') - 1)`<br />
    `+ (round_no - 0)·(round_no - 1)·(round_no - 2)·(round_no - 3)·(round_no - 4)·(state_i_limb_LookupClientLogDerivative' - state_i_limb_LookupClientLogDerivative)`
1. The remaining constraints are left as an exercise to the reader.
  For hints, see the [Tip5 paper](https://eprint.iacr.org/2023/107.pdf).

## Terminal Constraints

None.
