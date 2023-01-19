# Hash Table

The instruction `hash` hashes the OpStack's 10 top-most elements in one cycle.
Similarly, the Sponge instructions `absorb_init`, `absorb`, and `squeeze` all complete in one cycle.
The main processor achieves this by using a hash coprocessor.
The Hash Table is the arithmetization of that coprocessor.

Instruction `hash` and the Sponge instructions `absorb_init`, `absorb`, and `squeeze` are quite similar.
The main differences are in updates to the `state` registers between executions of Triton VM's pseudo-random permutation, [Rescue-XLIX](https://eprint.iacr.org/2020/1143.pdf).
A summary of the four instructions' mechanics:

- Instruction `hash`
    1. resets all state registers to 0,
    1. sets the eleventh state register, _i.e._, `st10`, to 1,
    1. adds the processor's stack registers `st0` through `st9` to the hash coprocessor's registers `state0` through `state9`
    1. executes the 8 rounds of Rescue-XLIX,
    1. overwrites the processor's stack registers `st0` through `st4` with 0, and
    1. overwrites the processor's stack registers with the hash coprocessor's registers `st5` through `st9` with `state0` through `state4`.
- Instruction `absorb_init`
    1. resets all state registers to 0,
    1. adds the processor's stack registers `st0` through `st9` to the hash coprocessor's registers `state0` through `state9`, and
    1. executes the 8 rounds of Rescue-XLIX.
- Instruction `absorb`
    1. adds the processor's stack registers `st0` through `st9` to the hash coprocessor's registers `state0` through `state9`, and
    1. executes the 8 rounds of Rescue-XLIX.
- Instruction `squeeze`
    1. overwrites the processor's stack registers `st0` through `st9` with the hash coprocessor's registers `state0` through `state9`, and
    1. executes the 8 rounds of Rescue-XLIX.

The Hash Table first records all Sponge instructions in the order the processor executed them.
Then, the Hash Table records all `hash` instructions in the order the processor executed them.
This allows the processor to execute `hash` instructions without affecting the Sponge's state.

## Base Columns

The Hash Table has 50 columns:
- one column `round_no` to indicate the round number,
- one column current instruction `CI`, holding the instruction the processor is currently executing,
- 16 state registers `state0` through `state15` to which the Rescue-XLIX rounds are applied, and
- 32 helper registers called `constant0A` through `constant15A` and `constant0B` through `constant15B` holding round constants.

## Extension Columns

The Hash Table has 5 extension columns:

- `RunningEvaluationHashInput`
- `RunningEvaluationHashDigest`
- `RunningEvaluationSpongeAbsorb`
- `RunningEvaluationSpongeSqueeze`
- `RunningEvaluationSpongeOrder`

Each column corresponds to one Evaluation Argument.
The respective Evaluation Arguments establish:

1. whenever the [processor](processor-table.md) executes a `hash` instruction, the values of the stack's 10 top-most registers correspond to some row in the Hash Table with round index equal to 1.
1. after having executed a `hash` instruction, stack registers `st5` through `st9` in the [processor](processor-table.md) correspond to the digest computed in the Hash Coprocessor, i.e., the first 5 values of the Hash Table's row with round index equal to 9.
1. whenever the processor executes an `absorb_init` or `absorb` instruction, the values of the stack's 10 top-most registers `st0` through `st9` in the processor correspond to (a) some row or (b) the difference of consecutive rows, respectively, in the Hash Table with round index equal to 1.
1. whenever the processor executes a `squeeze` instruction,  the values of the stack's 10 top-most registers correspond `st0` through `st9` in the processor correspond to the Sponge's current state, _i.e._, the first 10 values of the Hash Table's row with round index equal to 1.
1. whenever the processor executes any of the three Sponge instructions `absorb_init`, `absorb`, or `squeeze`, the register `CI` in the Hash Table corresponds to register `ci` in the Processor Table and the to-be-absorbed or to-be-squeezed elements are copied correctly between the two.

## Padding

Each padding row is the all-zero row with the exception of `CI`, which is the opcode of instruction `hash`.

# Arithmetic Intermediate Representation

Let all household items (🪥, 🛁, etc.) be challenges, concretely evaluation points, supplied by the verifier.
Let all fruit & vegetables (🥝, 🥥, etc.) be challenges, concretely weights to compress rows, supplied by the verifier.
Both types of challenges are X-field elements, _i.e._, elements of $\mathbb{F}_{p^3}$.

## Initial Constraints

1. The round number is 0 or 1.
1. The current instruction is `hash` or `absorb_init`.
1. If the current instruction is `hash` and the round number is 1, then `RunningEvaluationHashInput` has accumulated the first row with respect to challenges 🧄₀ through 🧄₉ and indeterminate 🚪.
    Otherwise, `RunningEvaluationHashInput` is 1.
1. `RunningEvaluationHashDigest` is 1.
1. If the current instruction is `absorb_init`, then `RunningEvaluationSpongeAbsorb` has accumulated the first row with respect to challenges 🧄₀ through 🧄₉ and indeterminate 🧽.
    Otherwise, `RunningEvaluationSpongeAbsorb` is 1.
1. `RunningEvaluationSpongeSqueeze` is 1.
1. If the current instruction is not `hash`, then `RunningEvaluationSpongeOrder` has accumulated `CI` and `st0` through `st9` with respect to challenges 🧅 and 🧄₀ through 🧄₉ and indeterminate 🪞.
    Otherwise, `RunningEvaluationSpongeOrder` is 1.

Written as Disjunctive Normal Form, the same constraints can be expressed as:

1. `round_no` is 0 or 1.
1. `CI` is the opcode of `hash` or of `absorb_init`.
1. (`CI` is the opcode of `absorb_init` or `round_no` is 0 or `RunningEvaluationHashInput` has accumulated the first row with respect to challenges 🧄₀ through 🧄₉ and indeterminate 🚪)<br />
    and (`CI` is the opcode of `hash` or `RunningEvaluationHashInput` is 1)<br />
    and (`round_no` is 1 or `RunningEvaluationHashInput` is 1).
1. `RunningEvaluationSpongeSqueeze` is 1.
1. (`CI` is the opcode of `hash` or `RunningEvaluationSpongeAbsorb` has accumulated the first row with respect to challenges 🧄₀ through 🧄₉ and indeterminate 🧽)<br />
    and (`CI` is the opcode of `absorb_init` or `RunningEvaluationSpongeAbsorb` is 1).
1. `RunningEvaluationHashDigest` is 1.
1. (`CI` is the opcode of `hash` or `RunningEvaluationSpongeOrder` has accumulated `CI` and `st0` through `st9` with respect to challenges 🧅 and 🧄₀ through 🧄₉ and indeterminate 🪞)<br />
    and (`CI` is the opcode of `absorb_init` or `RunningEvaluationSpongeOrder` is 1).

### Initial Constraints as Polynomials

1. `round_no·(round_no - 1)`
1. `(CI - opcode(hash))·(CI - opcode(absorb_init))`
1. `(CI - opcode(absorb_init))·round_no·(RunningEvaluationHashInput - 🚪 - 🧄₀·st0 - 🧄₁·st1 - 🧄₂·st2 - 🧄₃·st3 - 🧄₄·st4 - 🧄₅·st5 - 🧄₆·st6 - 🧄₇·st7 - 🧄₈·st8 - 🧄₉·st9)`<br />
    `+ (CI - opcode(hash))·(RunningEvaluationHashInput - 1)`<br />
    `+ (round_no - 1)·(RunningEvaluationHashInput - 1)`
1. `RunningEvaluationHashDigest - 1`
1. `(CI - opcode(hash))·(RunningEvaluationSpongeAbsorb - 🧽 - 🧄₀·st0 - 🧄₁·st1 - 🧄₂·st2 - 🧄₃·st3 - 🧄₄·st4 - 🧄₅·st5 - 🧄₆·st6 - 🧄₇·st7 - 🧄₈·st8 - 🧄₉·st9)`<br />
    `+ (CI - opcode(absorb_init))·(RunningEvaluationSpongeAbsorb - 1)`
1. `RunningEvaluationSpongeSqueeze - 1`
1. `(CI - opcode(hash))·(RunningEvaluationSpongeOrder - 🪞 - CI·🧅 - 🧄₀·st0 - 🧄₁·st1 - 🧄₂·st2 - 🧄₃·st3 - 🧄₄·st4 - 🧄₅·st5 - 🧄₆·st6 - 🧄₇·st7 - 🧄₈·st8 - 🧄₉·st9)`<br />
    `+ (CI - opcode(absorb_init))·(RunningEvaluationSpongeOrder - 1)`

## Consistency Constraints

1. If the round number is 0, then the current instruction is `hash`.
1. If the round number is 1 and the current instruction is `hash`, then register `state10` is 1.
1. If the round number is 1 and the current instruction is `absorb_init`, then register `state10` is 0.
1. If the round number is 1 and the current instruction is either `hash` or `absorb_init`, then register `state11` is 0.
1. If the round number is 1 and the current instruction is either `hash` or `absorb_init`, then register `state12` is 0.
1. If the round number is 1 and the current instruction is either `hash` or `absorb_init`, then register `state13` is 0.
1. If the round number is 1 and the current instruction is either `hash` or `absorb_init`, then register `state14` is 0.
1. If the round number is 1 and the current instruction is either `hash` or `absorb_init`, then register `state15` is 0.
1. The round constants adhere to the specification of Rescue Prime.

Written as Disjunctive Normal Form, the same constraints can be expressed as:

1. The round number is 1 or 2 or 3 or 4 or 5 or 6 or 7 or 8 or 9 or `CI` is the opcode of `hash`.
1. The round number is 0 or 2 or 3 or 4 or 5 or 6 or 7 or 8 or 9 or `CI` is the opcode of `absorb_init` or `absorb` or `squeeze` or `state10` is 1.
1. The round number is 0 or 2 or 3 or 4 or 5 or 6 or 7 or 8 or 9 or `CI` is the opcode of `hash` or `absorb` or `squeeze` or `state10` is 0.
1. The round number is 0 or 2 or 3 or 4 or 5 or 6 or 7 or 8 or 9 or `CI` is the opcode of `absorb` or `squeeze` or `state11` is 0.
1. The round number is 0 or 2 or 3 or 4 or 5 or 6 or 7 or 8 or 9 or `CI` is the opcode of `absorb` or `squeeze` or `state12` is 0.
1. The round number is 0 or 2 or 3 or 4 or 5 or 6 or 7 or 8 or 9 or `CI` is the opcode of `absorb` or `squeeze` or `state13` is 0.
1. The round number is 0 or 2 or 3 or 4 or 5 or 6 or 7 or 8 or 9 or `CI` is the opcode of `absorb` or `squeeze` or `state14` is 0.
1. The round number is 0 or 2 or 3 or 4 or 5 or 6 or 7 or 8 or 9 or `CI` is the opcode of `absorb` or `squeeze` or `state15` is 0.
1. The `constantiX` equals interpolant(`round_no`), where “interpolant” is the lowest-degree interpolant through (i, `constantiX`) for $1 \leqslant i \leqslant 9$, `X` $\in$ {A, B}.

### Consistency Constraints as Polynomials

1. `(round_no - 1)·(round_no - 2)·(round_no - 3)·(round_no - 4)·(round_no - 5)·(round_no - 6)·(round_no - 7)·(round_no - 8)·(round_no - 9)·(CI - opcode(hash))`
1. `(round_no - 0)·(round_no - 2)·(round_no - 3)·(round_no - 4)·(round_no - 5)·(round_no - 6)·(round_no - 7)·(round_no - 8)·(round_no - 9)·(CI - opcode(absorb_init))·(CI - opcode(absorb))·(CI - opcode(squeeze))·(state10 - 1)`
1. `(round_no - 0)·(round_no - 2)·(round_no - 3)·(round_no - 4)·(round_no - 5)·(round_no - 6)·(round_no - 7)·(round_no - 8)·(round_no - 9)·(CI - opcode(hash))·(CI - opcode(absorb))·(CI - opcode(squeeze))·state10`
1. `(round_no - 0)·(round_no - 2)·(round_no - 3)·(round_no - 4)·(round_no - 5)·(round_no - 6)·(round_no - 7)·(round_no - 8)·(round_no - 9)·(CI - opcode(absorb))·(CI - opcode(squeeze))·state11`
1. `(round_no - 0)·(round_no - 2)·(round_no - 3)·(round_no - 4)·(round_no - 5)·(round_no - 6)·(round_no - 7)·(round_no - 8)·(round_no - 9)·(CI - opcode(absorb))·(CI - opcode(squeeze))·state12`
1. `(round_no - 0)·(round_no - 2)·(round_no - 3)·(round_no - 4)·(round_no - 5)·(round_no - 6)·(round_no - 7)·(round_no - 8)·(round_no - 9)·(CI - opcode(absorb))·(CI - opcode(squeeze))·state13`
1. `(round_no - 0)·(round_no - 2)·(round_no - 3)·(round_no - 4)·(round_no - 5)·(round_no - 6)·(round_no - 7)·(round_no - 8)·(round_no - 9)·(CI - opcode(absorb))·(CI - opcode(squeeze))·state14`
1. `(round_no - 0)·(round_no - 2)·(round_no - 3)·(round_no - 4)·(round_no - 5)·(round_no - 6)·(round_no - 7)·(round_no - 8)·(round_no - 9)·(CI - opcode(absorb))·(CI - opcode(squeeze))·state15`

## Transition Constraints

1. If the round number is 0, the round number in the next row is 0.
1. If the round number is 1, 2, 3, 4, 5, 6, 7, or 8, then the round number in the next row is incremented by 1.
1. If the round number is 9, the round number in the next row is either 0 or 1.
1. If the current instruction is `hash`, then the current instruction in the next row is `hash`.
1. If the round number is not 9, the current instruction in the next row is the current instruction in the current row.
1. If the round number in the next row is 1 and the current instruction in the next row is `absorb`, then the capacity's state registers don't change.
1. If the round number in the next row is 1 and the current instruction in the next row is `squeeze`, then none of the state registers change.
1. If the round number in the next row is 1 and the current instruction in the next row is `hash`, then `RunningEvaluationHashInput` accumulates the next row with respect to challenges 🧄₀ through 🧄₉ and indeterminate 🚪. Otherwise, it remains unchanged.
1. If the round number in the next row is 9 and the current instruction in the next row is `hash`, then `RunningEvaluationHashDigest` accumulates the next row with respect to challenges 🧄₀ through 🧄₄ and indeterminate 🪟. Otherwise, it remains unchanged.
1.  1. If the round number in the next row is 1 and the current instruction in the next row is `absorb_init`, then `RunningEvaluationSpongeAbsorb` accumulates the next row with respect to challenges 🧄₀ through 🧄₉ and indeterminate 🧽.
    1. If the round number in the next row is 1 and the current instruction in the next row is `absorb`, then `RunningEvaluationSpongeAbsorb` accumulates the difference of the next row and the current row with respect to challenges 🧄₀ through 🧄₉ and indeterminate 🧽.
    1. If the round number in the next row is not 1, then `RunningEvaluationSpongeAbsorb` remains unchanged.
    1. If the current instruction in the next row is either `hash` or `squeeze`, then `RunningEvaluationSpongeAbsorb` remains unchanged.
1. If the round number in the next row is 1 and the current instruction is `squeeze`, then `RunningEvaluationSpongeSqueeze` accumulates the next row with respect to challenges 🧄₀ through 🧄₉ and indeterminate 🪣.
    Otherwise, it remains unchanged.
1.  1. If the round number in the next row is 1 and the current instruction in the next row is `absorb_init` or `squeeze`, then `RunningEvaluationSpongeOrder` accumulates `CI` and `st0` through `st9` in the next row with respect to challenges 🧅 and 🧄₀ through 🧄₉ and indeterminate 🪞.
    1. If the round number in the next row is 1 and the current instruction in the next row is `absorb`, then `RunningEvaluationSpongeOrder` accumulates `CI` in the next row and the differences of `st0` through `st9` in the next row and the current row with respect to challenges 🧅 and 🧄₀ through 🧄₉ and indeterminate 🪞.
    1. If the round number in the next row is not 1, then `RunningEvaluationSpongeOrder` remains unchanged.
    1. If `CI` in the next row is `hash`, then `RunningEvaluationSpongeOrder` remains unchanged.
1. If the round number is 1, the `state` registers adhere to the rules of applying Rescue-XLIX round 1.
1. If the round number is 2, the `state` registers adhere to the rules of applying Rescue-XLIX round 2.
1. If the round number is 3, the `state` registers adhere to the rules of applying Rescue-XLIX round 3.
1. If the round number is 4, the `state` registers adhere to the rules of applying Rescue-XLIX round 4.
1. If the round number is 5, the `state` registers adhere to the rules of applying Rescue-XLIX round 5.
1. If the round number is 6, the `state` registers adhere to the rules of applying Rescue-XLIX round 6.
1. If the round number is 7, the `state` registers adhere to the rules of applying Rescue-XLIX round 7.
1. If the round number is 8, the `state` registers adhere to the rules of applying Rescue-XLIX round 8.

Written as Disjunctive Normal Form, the same constraints can be expressed as:

1. `round_no` is 1 or 2 or 3 or 4 or 5 or 6 or 7 or 8 or 9 or `round_no'` is 0.
1. `round_no` is 0 or 9 or `round_no'` is `round_no` + 1.
1. `round_no` is 0 or 1 or 2 or 3 or 4 or 5 or 6 or 7 or 8 or `round_no'` is 0 or 1.
1. `CI` is the opcode of `absorb_init` or `absorb` or `squeeze` or `CI'` is the opcode of `hash`.
1. `round_no` is 9 or `CI'` is `CI`.
1. `round_no'` is 0 or 2 or 3 or 4 or 5 or 6 or 7 or 8 or 9 or `CI'` is the opcode of `hash` or `absorb_init` or `squeeze` or the $🧄_i$-randomized sum of differences of the state registers `state10` through `state15` in the next row and the current row is 0.
1. `round_no'` is 0 or 2 or 3 or 4 or 5 or 6 or 7 or 8 or 9 or `CI'` is the opcode of `hash` or `absorb_init` or `absorb` or the $🧄_i$-randomized sum of differences of all state registers in the next row and the current row is 0.
1. (`round_no'` is 0 or 2 or 3 or 4 or 5 or 6 or 7 or 8 or 9 or `CI'` is the opcode of `absorb_init` or `absorb` or `squeeze` or `RunningEvaluationHashInput` accumulates the next row)<br />
    and (`round_no'` is 1 or `RunningEvaluationHashInput` remains unchanged)<br />
    and (`CI'` is the opcode of `hash` or `RunningEvaluationHashInput` remains unchanged).
1. (`round_no'` is 0 or 1 or 2 or 3 or 4 or 5 or 6 or 7 or 8 or `CI'` is the opcode of `absorb_init` or `absorb` or `squeeze` or `RunningEvaluationHashDigest` accumulates the next row)<br />
    and (`round_no'` is 9 or `RunningEvaluationHashDigest` remains unchanged)<br />
    and (`CI'` is the opcode of `hash` or `RunningEvaluationHashDigest` remains unchanged).
1.  1. (`round_no'` is 0 or 2 or 3 or 4 or 5 or 6 or 7 or 8 or 9 or `CI'` is the opcode of `hash` or `absorb` or `squeeze` or `RunningEvaluationSpongeAbsorb` accumulates the next row)
    1. and (`round_no'` is 0 or 2 or 3 or 4 or 5 or 6 or 7 or 8 or 9 or `CI'` is the opcode of `hash` or `absorb_init` or `squeeze` or `RunningEvaluationSpongeAbsorb` accumulates the difference of the next row and the current row)
    1. and (`round_no'` is 1 or `RunningEvaluationSpongeAbsorb` remains unchanged)
    1. and (`CI'` is the opcode of `absorb_init` or `absorb` or `RunningEvaluationSpongeAbsorb` remains unchanged).
1. (`round_no'` is 0 or 2 or 3 or 4 or 5 or 6 or 7 or 8 or 9 or `CI'` is the opcode of `hash` or `absorb_init` or `absorb` or `RunningEvaluationSpongeSqueeze` accumulates the next row)<br />
    and (`round_no'` is 1 or `RunningEvaluationSpongeSqueeze` remains unchanged)<br />
    and (`CI'` is the opcode of `squeeze` or `RunningEvaluationSpongeSqueeze` remains unchanged).
1.  1. (`round_no'` is 0 or 2 or 3 or 4 or 5 or 6 or 7 or 8 or 9 or `CI'` is the opcode of `hash` or `absorb` or `RunningEvaluationSpongeOrder` accumulates the next row)
    1. (`round_no'` is 0 or 2 or 3 or 4 or 5 or 6 or 7 or 8 or 9 or `CI'` is the opcode of `hash` or `absorb_init` or `squeeze` or `RunningEvaluationSpongeOrder` accumulates the difference of the next row and the current row)
    1. and (`round_no'` is 1 or `RunningEvaluationSpongeOrder` remains unchanged)
    1. and (`CI'` is the opcode of `absorb_init` or `absorb` or `squeeze` or `RunningEvaluationSpongeOrder` remains unchanged).
1. `round_no` is 0 or 2 or 3 or 4 or 5 or 6 or 7 or 8 or 9 or the `state` registers adhere to the rules of applying Rescue-XLIX round 1.
1. `round_no` is 0 or 1 or 3 or 4 or 5 or 6 or 7 or 8 or 9 or the `state` registers adhere to the rules of applying Rescue-XLIX round 2.
1. `round_no` is 0 or 1 or 2 or 4 or 5 or 6 or 7 or 8 or 9 or the `state` registers adhere to the rules of applying Rescue-XLIX round 3.
1. `round_no` is 0 or 1 or 2 or 3 or 5 or 6 or 7 or 8 or 9 or the `state` registers adhere to the rules of applying Rescue-XLIX round 4.
1. `round_no` is 0 or 1 or 2 or 3 or 4 or 6 or 7 or 8 or 9 or the `state` registers adhere to the rules of applying Rescue-XLIX round 5.
1. `round_no` is 0 or 1 or 2 or 3 or 4 or 5 or 7 or 8 or 9 or the `state` registers adhere to the rules of applying Rescue-XLIX round 6.
1. `round_no` is 0 or 1 or 2 or 3 or 4 or 5 or 6 or 8 or 9 or the `state` registers adhere to the rules of applying Rescue-XLIX round 7.
1. `round_no` is 0 or 1 or 2 or 3 or 4 or 5 or 6 or 7 or 9 or the `state` registers adhere to the rules of applying Rescue-XLIX round 8.

### Transition Constraints as Polynomials

1. `(round_no - 1)·(round_no - 2)·(round_no - 3)·(round_no - 4)·(round_no - 5)·(round_no - 6)·(round_no - 7)·(round_no - 8)·(round_no-9)·(round_no' - 0)`
1. `(round_no - 0)·(round_no - 9)·(round_no' - round_no - 1)`
1. `(round_no - 0)·(round_no - 1)·(round_no - 2)·(round_no - 3)·(round_no - 4)·(round_no - 5)·(round_no - 6)·(round_no - 7)·(round_no-8)·(round_no' - 0)·(round_no' - 1)`
1. `(CI - opcode(absorb_init))·(CI - opcode(absorb))·(CI - opcode(squeeze))·(CI' - opcode(hash))`
1. `(round_no - 9)·(CI' - CI)`
1. `(round_no' - 0)·(round_no' - 2)·(round_no' - 3)·(round_no' - 4)·(round_no' - 5)·(round_no' - 6)·(round_no' - 7)·(round_no' - 8)·(round_no' - 9)`<br />
    `·(CI' - opcode(hash))·(CI' - opcode(absorb_init))·(CI' - opcode(squeeze))`<br />
    `·(🧄₁₀·(st10' - st10) + 🧄₁₁·(st11' - st11) + 🧄₁₂·(st12' - st12) + 🧄₁₃·(st13' - st13) + 🧄₁₄·(st14' - st14) + 🧄₁₅·(st15' - st15))`
1. `(round_no' - 0)·(round_no' - 2)·(round_no' - 3)·(round_no' - 4)·(round_no' - 5)·(round_no' - 6)·(round_no' - 7)·(round_no' - 8)·(round_no' - 9)`<br />
    `·(CI' - opcode(hash))·(CI' - opcode(absorb_init))·(CI' - opcode(absorb))`<br />
    `·(🧄₀·(st0' - st0) + 🧄₁·(st1' - st1) + 🧄₂·(st2' - st2) + 🧄₃·(st3' - st3) + 🧄₄·(st4' - st4) + 🧄₅·(st5' - st5) + 🧄₆·(st6' - st6) + 🧄₇·(st7' - st7) + 🧄₈·(st8' - st8) + 🧄₉·(st9' - st9) + 🧄₁₀·(st10' - st10) + 🧄₁₁·(st11' - st11) + 🧄₁₂·(st12' - st12) + 🧄₁₃·(st13' - st13) + 🧄₁₄·(st14' - st14) + 🧄₁₅·(st15' - st15))`
1. `(round_no' - 0)·(round_no' - 2)·(round_no' - 3)·(round_no' - 4)·(round_no' - 5)·(round_no' - 6)·(round_no' - 7)·(round_no' - 8)·(round_no' - 9)`<br />
    `·(CI' - opcode(absorb_init))·(CI' - opcode(absorb))·(CI' - opcode(squeeze))`<br />
    `·(RunningEvaluationHashInput' - 🚪·RunningEvaluationHashInput - 🧄₀·st0' - 🧄₁·st1' - 🧄₂·st2' - 🧄₃·st3' - 🧄₄·st4' - 🧄₅·st5' - 🧄₆·st6' - 🧄₇·st7' - 🧄₈·st8' - 🧄₉·st9')`<br />
    `+ (round_no' - 1)·(RunningEvaluationHashInput' - RunningEvaluationHashInput)`<br />
    `+ (CI' - opcode(hash))·(RunningEvaluationHashInput' - RunningEvaluationHashInput)`
1. `(round_no' - 0)·(round_no' - 1)·(round_no' - 2)·(round_no' - 3)·(round_no' - 4)·(round_no' - 5)·(round_no' - 6)·(round_no' - 7)·(round_no' - 8)`<br />
    `·(CI' - opcode(absorb_init))·(CI' - opcode(absorb))·(CI' - opcode(squeeze))`<br />
    `·(RunningEvaluationHashDigest' - 🪟·RunningEvaluationHashDigest - 🧄₀·st0' - 🧄₁·st1' - 🧄₂·st2' - 🧄₃·st3' - 🧄₄·st4')`<br />
    `+ (round_no' - 9)·(RunningEvaluationHashDigest' - RunningEvaluationHashDigest)`<br />
    `+ (CI' - opcode(hash))·(RunningEvaluationHashDigest' - RunningEvaluationHashDigest)`
1.  1. `(round_no' - 0)·(round_no' - 2)·(round_no' - 3)·(round_no' - 4)·(round_no' - 5)·(round_no' - 6)·(round_no' - 7)·(round_no' - 8)·(round_no' - 9)`<br />
    `·(CI' - opcode(hash))·(CI' - opcode(absorb))·(CI' - opcode(squeeze))`<br />
    `·(RunningEvaluationSpongeAbsorb' - 🧽·RunningEvaluationSpongeAbsorb - 🧄₀·st0' - 🧄₁·st1' - 🧄₂·st2' - 🧄₃·st3' - 🧄₄·st4' - 🧄₅·st5' - 🧄₆·st6' - 🧄₇·st7' - 🧄₈·st8' - 🧄₉·st9')`<br />
    1. `+ (round_no' - 0)·(round_no' - 2)·(round_no' - 3)·(round_no' - 4)·(round_no' - 5)·(round_no' - 6)·(round_no' - 7)·(round_no' - 8)·(round_no' - 9)`<br />
    `·(CI' - opcode(hash))·(CI' - opcode(absorb_init))·(CI' - opcode(squeeze))`<br />
    `·(RunningEvaluationSpongeAbsorb' - 🧽·RunningEvaluationSpongeAbsorb - 🧄₀·(st0' - st0) - 🧄₁·(st1' - st1) - 🧄₂·(st2' - st2) - 🧄₃·(st3' - st3) - 🧄₄·(st4' - st4) - 🧄₅·(st5' - st5) - 🧄₆·(st6' - st6) - 🧄₇·(st7' - st7) - 🧄₈·(st8' - st8) - 🧄₉·(st9' - st9))`<br />
    1. `+ (round_no' - 1)·(RunningEvaluationSpongeAbsorb' - RunningEvaluationSpongeAbsorb)`<br />
    1. `+ (CI' - opcode(absorb_init)·(CI' - opcode(absorb))·(RunningEvaluationSpongeAbsorb' - RunningEvaluationSpongeAbsorb))`
1. `(round_no' - 0)·(round_no' - 2)·(round_no' - 3)·(round_no' - 4)·(round_no' - 5)·(round_no' - 6)·(round_no' - 7)·(round_no' - 8)·(round_no' - 9)`<br />
    `·(CI' - opcode(hash))·(CI' - opcode(absorb_init))·(CI' - opcode(absorb))`<br />
    `·(RunningEvaluationSpongeSqueeze' - 🪣·RunningEvaluationSpongeSqueeze - 🧄₀·st0' - 🧄₁·st1' - 🧄₂·st2' - 🧄₃·st3' - 🧄₄·st4' - 🧄₅·st5' - 🧄₆·st6' - 🧄₇·st7' - 🧄₈·st8' - 🧄₉·st9')`<br />
    `+ (round_no' - 1)·(RunningEvaluationSpongeSqueeze' - RunningEvaluationSpongeSqueeze)`<br />
    `+ (CI' - opcode(absorb))·(RunningEvaluationSpongeSqueeze' - RunningEvaluationSpongeSqueeze)`
1.  1. `(round_no' - 0)·(round_no' - 2)·(round_no' - 3)·(round_no' - 4)·(round_no' - 5)·(round_no' - 6)·(round_no' - 7)·(round_no' - 8)·(round_no' - 9)`<br />
        `·(CI' - opcode(hash))·(CI' - opcode(absorb))`<br />
        `·(RunningEvaluationSpongeOrder' - 🪞·RunningEvaluationSpongeOrder - 🧅·CI' - 🧄₀·st0' - 🧄₁·st1' - 🧄₂·st2' - 🧄₃·st3' - 🧄₄·st4' - 🧄₅·st5' - 🧄₆·st6' - 🧄₇·st7' - 🧄₈·st8' - 🧄₉·st9')`<br />
    1. `(round_no' - 0)·(round_no' - 2)·(round_no' - 3)·(round_no' - 4)·(round_no' - 5)·(round_no' - 6)·(round_no' - 7)·(round_no' - 8)·(round_no' - 9)`<br />
        `·(CI' - opcode(hash))·(CI' - opcode(absorb_init)·(CI' - opcode(squeeze))`<br />
        `·(RunningEvaluationSpongeOrder' - 🪞·RunningEvaluationSpongeOrder - 🧅·CI' - 🧄₀·(st0' - st0) - 🧄₁·(st1' - st1) - 🧄₂·(st2' - st2) - 🧄₃·(st3' - st3) - 🧄₄·(st4' - st4) - 🧄₅·(st5' - st5) - 🧄₆·(st6' - st6) - 🧄₇·(st7' - st7) - 🧄₈·(st8' - st8) - 🧄₉·(st9' - st9))`<br />
    1. `+ (round_no' - 1)·(RunningEvaluationSpongeOrder' - RunningEvaluationSpongeOrder)`<br />
    1. `+ (CI' - opcode(absorb_init))·(CI' - opcode(absorb))·(CI' - opcode(squeeze))·(RunningEvaluationSpongeOrder' - RunningEvaluationSpongeOrder)`
1. The remaining constraints are left as an exercise to the reader.
  For hints, see the [Rescue-Prime Systematization of Knowledge, Sections 2.4 & 2.5](https://eprint.iacr.org/2020/1143.pdf#page=5).

## Terminal Constraints

None.
