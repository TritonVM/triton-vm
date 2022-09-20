# Hash Table

The instruction `hash` hashes the OpStack's 10 top-most elements in one cycle.
What happens in the background is that the registers `st0` through `st9` are copied to the Hash Coprocessor's registers `state0` through `state9`.
The eleventh state register, `state10` is set to 1; this is the domain separation bit.
The Hash Coprocessor's five remaining state registers, `state11` through `state15`, are set to 0.
Then, the Coprocessor runs the 8 rounds of Rescue-XLIX on its `state` registers.
Finally, the hash digest, i.e., the 5 values from `state0` through `state4`, are copied back to the OpStack.
This allows the (main) Processor to perform the hashing instruction in a single cycle.

The Hash Table has 49 columns:
- one column `rnd_nmbr` to indicate the round number,
- 16 state registers `state0` through `state15` to which the Rescue-XLIX rounds are applied, and
- 32 helper registers called `constant0A` through `constant15A` and `constant0B` through `constant15B` holding round constants.

## Padding

Each padding row is the all-zero row.

## Initial Constraints

1. The round number `rnd_nmbr` starts at 0 or 1.

**Initial Constraints as Polynomials**

1. `rnd_nmbr·(rnd_nmbr - 1)`

## Consistency Constraints

1. If the round number is 1, register `state10` is 0.
1. If the round number is 1, register `state11` is 0.
1. If the round number is 1, register `state12` is 0.
1. If the round number is 1, register `state13` is 0.
1. If the round number is 1, register `state14` is 0.
1. If the round number is 1, register `state15` is 0.
1. The round constants adhere to the specification of Rescue Prime.

Written as Disjunctive Normal Form, the same constraints can be expressed as:
1. The round number is 0 or 2 or 3 or 4 or 5 or 6 or 7 or 8 or `state10` is 0.
1. The round number is 0 or 2 or 3 or 4 or 5 or 6 or 7 or 8 or `state11` is 0.
1. The round number is 0 or 2 or 3 or 4 or 5 or 6 or 7 or 8 or `state12` is 0.
1. The round number is 0 or 2 or 3 or 4 or 5 or 6 or 7 or 8 or `state13` is 0.
1. The round number is 0 or 2 or 3 or 4 or 5 or 6 or 7 or 8 or `state14` is 0.
1. The round number is 0 or 2 or 3 or 4 or 5 or 6 or 7 or 8 or `state15` is 0.
1. The `constantiX` equals interpolant(`rnd_nmbr`), where “interpolant” is the lowest-degree interpolant through (i, `constantiX`) for $1 \leqslant i \leqslant 8$, `X` $\in$ {A, B}.

**Consistency Constraints as Polynomials**

1. `(rnd_nmbr - 0)·(rnd_nmbr - 2)·(rnd_nmbr - 3)·(rnd_nmbr - 4)·(rnd_nmbr - 5)·(rnd_nmbr - 6)·(rnd_nmbr - 7)·(rnd_nmbr - 8)·state10`
1. `(rnd_nmbr - 0)·(rnd_nmbr - 2)·(rnd_nmbr - 3)·(rnd_nmbr - 4)·(rnd_nmbr - 5)·(rnd_nmbr - 6)·(rnd_nmbr - 7)·(rnd_nmbr - 8)·state11`
1. `(rnd_nmbr - 0)·(rnd_nmbr - 2)·(rnd_nmbr - 3)·(rnd_nmbr - 4)·(rnd_nmbr - 5)·(rnd_nmbr - 6)·(rnd_nmbr - 7)·(rnd_nmbr - 8)·state12`
1. `(rnd_nmbr - 0)·(rnd_nmbr - 2)·(rnd_nmbr - 3)·(rnd_nmbr - 4)·(rnd_nmbr - 5)·(rnd_nmbr - 6)·(rnd_nmbr - 7)·(rnd_nmbr - 8)·state13`
1. `(rnd_nmbr - 0)·(rnd_nmbr - 2)·(rnd_nmbr - 3)·(rnd_nmbr - 4)·(rnd_nmbr - 5)·(rnd_nmbr - 6)·(rnd_nmbr - 7)·(rnd_nmbr - 8)·state14`
1. `(rnd_nmbr - 0)·(rnd_nmbr - 2)·(rnd_nmbr - 3)·(rnd_nmbr - 4)·(rnd_nmbr - 5)·(rnd_nmbr - 6)·(rnd_nmbr - 7)·(rnd_nmbr - 8)·state15`

## Transition Constraints

1. If the round number is 0, the next round number is 0.
1. If the round number is 1, the next round number is 2.
1. If the round number is 2, the next round number is 3.
1. If the round number is 3, the next round number is 4.
1. If the round number is 4, the next round number is 5.
1. If the round number is 5, the next round number is 6.
1. If the round number is 6, the next round number is 7.
1. If the round number is 7, the next round number is 8.
1. If the round number is 8, the next round number is either 0 or 1.
1. If the round number is 1, the `state` registers adhere to the rules of applying Rescue-XLIX round 1.
1. If the round number is 2, the `state` registers adhere to the rules of applying Rescue-XLIX round 2.
1. If the round number is 3, the `state` registers adhere to the rules of applying Rescue-XLIX round 3.
1. If the round number is 4, the `state` registers adhere to the rules of applying Rescue-XLIX round 4.
1. If the round number is 5, the `state` registers adhere to the rules of applying Rescue-XLIX round 5.
1. If the round number is 6, the `state` registers adhere to the rules of applying Rescue-XLIX round 6.
1. If the round number is 7, the `state` registers adhere to the rules of applying Rescue-XLIX round 7.
1. If the round number is 8, the `state` registers adhere to the rules of applying Rescue-XLIX round 8.

Written as Disjunctive Normal Form, the same constraints can be expressed as:
1. The round number is 1 or 2 or 3 or 4 or 5 or 6 or 7 or 8 or the next round number is 0.
1. The round number is 0 or 2 or 3 or 4 or 5 or 6 or 7 or 8 or the next round number is 2.
1. The round number is 0 or 1 or 3 or 4 or 5 or 6 or 7 or 8 or the next round number is 3.
1. The round number is 0 or 1 or 2 or 4 or 5 or 6 or 7 or 8 or the next round number is 4.
1. The round number is 0 or 1 or 2 or 3 or 5 or 6 or 7 or 8 or the next round number is 5.
1. The round number is 0 or 1 or 2 or 3 or 4 or 6 or 7 or 8 or the next round number is 6.
1. The round number is 0 or 1 or 2 or 3 or 4 or 5 or 7 or 8 or the next round number is 7.
1. The round number is 0 or 1 or 2 or 3 or 4 or 5 or 6 or 8 or the next round number is 8.
1. The round number is 0 or 1 or 2 or 3 or 4 or 5 or 6 or 7 or the next round number is 0 or 1.
1. The round number is 0 or 2 or 3 or 4 or 5 or 6 or 7 or 8 or the `state` registers adhere to the rules of applying Rescue-XLIX round 1.
1. The round number is 0 or 1 or 3 or 4 or 5 or 6 or 7 or 8 or the `state` registers adhere to the rules of applying Rescue-XLIX round 2.
1. The round number is 0 or 1 or 2 or 4 or 5 or 6 or 7 or 8 or the `state` registers adhere to the rules of applying Rescue-XLIX round 3.
1. The round number is 0 or 1 or 2 or 3 or 5 or 6 or 7 or 8 or the `state` registers adhere to the rules of applying Rescue-XLIX round 4.
1. The round number is 0 or 1 or 2 or 3 or 4 or 6 or 7 or 8 or the `state` registers adhere to the rules of applying Rescue-XLIX round 5.
1. The round number is 0 or 1 or 2 or 3 or 4 or 5 or 7 or 8 or the `state` registers adhere to the rules of applying Rescue-XLIX round 6.
1. The round number is 0 or 1 or 2 or 3 or 4 or 5 or 6 or 8 or the `state` registers adhere to the rules of applying Rescue-XLIX round 7.
1. The round number is 0 or 1 or 2 or 3 or 4 or 5 or 6 or 7 or the `state` registers adhere to the rules of applying Rescue-XLIX round 8.

**Transition Constraints as Polynomials**

1. `(rnd_nmbr - 1)·(rnd_nmbr - 2)·(rnd_nmbr - 3)·(rnd_nmbr - 4)·(rnd_nmbr - 5)·(rnd_nmbr - 6)·(rnd_nmbr - 7)·(rnd_nmbr - 8)·(rnd_nmbr' -  0)`
1. `(rnd_nmbr - 0)·(rnd_nmbr - 2)·(rnd_nmbr - 3)·(rnd_nmbr - 4)·(rnd_nmbr - 5)·(rnd_nmbr - 6)·(rnd_nmbr - 7)·(rnd_nmbr - 8)·(rnd_nmbr' -  2)`
1. `(rnd_nmbr - 0)·(rnd_nmbr - 1)·(rnd_nmbr - 3)·(rnd_nmbr - 4)·(rnd_nmbr - 5)·(rnd_nmbr - 6)·(rnd_nmbr - 7)·(rnd_nmbr - 8)·(rnd_nmbr' -  3)`
1. `(rnd_nmbr - 0)·(rnd_nmbr - 1)·(rnd_nmbr - 2)·(rnd_nmbr - 4)·(rnd_nmbr - 5)·(rnd_nmbr - 6)·(rnd_nmbr - 7)·(rnd_nmbr - 8)·(rnd_nmbr' -  4)`
1. `(rnd_nmbr - 0)·(rnd_nmbr - 1)·(rnd_nmbr - 2)·(rnd_nmbr - 3)·(rnd_nmbr - 5)·(rnd_nmbr - 6)·(rnd_nmbr - 7)·(rnd_nmbr - 8)·(rnd_nmbr' -  5)`
1. `(rnd_nmbr - 0)·(rnd_nmbr - 1)·(rnd_nmbr - 2)·(rnd_nmbr - 3)·(rnd_nmbr - 4)·(rnd_nmbr - 6)·(rnd_nmbr - 7)·(rnd_nmbr - 8)·(rnd_nmbr' -  6)`
1. `(rnd_nmbr - 0)·(rnd_nmbr - 1)·(rnd_nmbr - 2)·(rnd_nmbr - 3)·(rnd_nmbr - 4)·(rnd_nmbr - 5)·(rnd_nmbr - 7)·(rnd_nmbr - 8)·(rnd_nmbr' -  7)`
1. `(rnd_nmbr - 0)·(rnd_nmbr - 1)·(rnd_nmbr - 2)·(rnd_nmbr - 3)·(rnd_nmbr - 4)·(rnd_nmbr - 5)·(rnd_nmbr - 6)·(rnd_nmbr - 8)·(rnd_nmbr' -  8)`
1. `(rnd_nmbr - 0)·(rnd_nmbr - 1)·(rnd_nmbr - 2)·(rnd_nmbr - 3)·(rnd_nmbr - 4)·(rnd_nmbr - 5)·(rnd_nmbr - 6)·(rnd_nmbr - 7)·(rnd_nmbr' -  0)·(rnd_nmbr' -  1)`
1. The remaining 16 constraints are left as an exercise to the reader.
For hints, see the [Rescue-Prime Systematization of Knowledge, Sections 2.4 & 2.5](https://eprint.iacr.org/2020/1143.pdf#page=5).

## Terminal Constraints

None.

## Relations to Other Tables

1. An Evaluation Argument establishes that whenever the [processor](processor-table.md) executes a `hash` instruction, the values of the stack's 12 top-most registers correspond to some row in the Hash Table with round index equal to 1.
1. An Evaluation Argument establishes that after having executed a `hash` instruction, the top 6 stack registers in the [processor](processor-table.md) correspond to the digest computed in the Hash Coprocessor, i.e., the first six values of the Hash Table's row with round index equal to 8.
