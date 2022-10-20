# Processor Table

The processor consists of all registers defined in the [Instruction Set Architecture](isa.md).
Each register is assigned a column in the processor table.

## Padding

A padding row is a copy of the Processor Table's last row with the following modifications:
1. column `clk` is increased by 1, and
1. column `IsPadding` is set to 1.

## Inner Sorting Argument for Memory Consistency

The rows memory-like tables, which are the [RAM Table](random-access-memory-table.md), the [JumpStack Table](jump-stack-table.md), and the [OpStack Table](operational-stack-table.md), need to satisfy as particular ordering in order to establish memory-consistency. In particular, the regions of constant memory pointer need to be contiguous; and the rows in each such contiguous region must be sorted for clock cycle. The contiguity of regions is trivial for the JumpStack and OpStack Table, and for the RAM Table the Contiguity Argument of [TIP 0001](https://github.com/TritonVM/triton-vm/blob/master/tips/tip-0001/tip-0001.md) establishes this fact. The Inner Sorting Argument is described in [TIP 0003](https://github.com/TritonVM/triton-vm/blob/master/tips/tip-0003/tip-0003.md) and impacts the Processor Table quite substantially.

The construction introduces three extra base columns:

 - `cjd`, the list of all clock jump differences greater than 1 in all memory-like tables.
 - `invm`, the list of inverses of clock jump differences, counting multiplicities. This column helps to select all nonzero `cjd`'s.
 - `invu`, the list of inverses of unique clock jump differences, *i.e.*, without counting multiplicities. This column helps to select the unique nonzero `cjd`'s.

... and three extra extra extension columns:

 - `rer`, the running evaluation of relevant clock cycles.
 - `reu`, the running evaluation of unique clock cycle differences.
 - `rpm`, the running product of all clock jump differences, with multiplicities.

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
1. The operational stack element `st11` is 0.
1. The operational stack element `st12` is 0.
1. The operational stack element `st13` is 0.
1. The operational stack element `st14` is 0.
1. The operational stack element `st15` is 0.
1. The operational stack pointer `osp` is 16.
1. The operational stack value `osv` is 0.
1. The RAM value `ramv` is 0.
1. The running evaluation of relevant clock cycles starts with 1.
1. The running evaluation of unique clock jump differences starts off having applied one evaluation step with the clock jump difference.
1. The running product of all clock jump differences starts starts off having accumulated the first factor.

**Initial Constraints as Polynomials**

1. `clk`
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
1. `ramv`
1. `rer - 1`
1. `reu - β - cjd`
1. `rpm - (alpha - cjd)`

## Consistency Constraints

1. The composition of instruction buckets `ib0`-`ib5` corresponds the current instruction `ci`.
1. The inverse of clock jump difference with multiplicity `invm` is the inverse-or-zero of the the clock jump difference `cjd`.

**Consistency Constraints as Polynomials**

1. `ci - (2^5·ib5 + 2^4·ib4 + 2^3·ib3 + 2^2·ib2 + 2^1·ib1 + 2^0·ib0)`
1. `invm·(invm·cjd - 1)`
1. `cjd·(invm·cjd - 1)`

## Transition Constraints

Due to their complexity, instruction-specific constraints are defined [in their own section](processors-instruction-constraints.md).

The following constraint applies to every pair of rows.

1. The cycle counter `clk` increases by 1.
1. The unique inverse column `invu'` holds the inverse-or-zero of the difference of consecutive `cjd`'s, if `cjd'` is nonzero.
1. The running product `rpm` of `cjd`'s with multiplicities is accumulates a factor `α - cjd'` in every row, provided that `cjd'` is nonzero.
1. The running evaluation `reu` of unique `cjd`'s is updated relative to evaluation point β whenever the difference of `cjd`'s is nonzero *and* the next `cjd` is nonzero.
1. The running evaluation `rer` or relevant clock cycles is updated relative to evaluation point β or not at all.

**Transition Constraints as Polynomials**

1. `clk' - (clk + 1)`
1. `invu' · (invu' · (cjd' - cjd) - 1) · cjd'`
1. `(cjd - cjd') · (invu' · (cjd' - cjd) - 1) · cjd'`
1. `cjd' · (rpm' - rpm · (α - cjd')) + (cjd' · invm' - 1) · (rpm' - rpm)`
1. `(1 - (cjd' - cjd) · invu) · (reu' - reu) + (1 - cjd' · invm) · (reu' - reu) + cjd' · (cjd' - cjd) · (reu' - β · reu - cjd')`
1. `(rer' - rer ·  β - clk') · (rer' - rer)`

## Terminal Constraints

1. In the last row, register “current instruction” `ci` is 0, corresponding to instruction `halt`.
1. In the last row, the running evaluations `rer` and `reu` are equal.

**Terminal Constraints as Polynomials**

1. `ci`
1. `rer - reu`

## Relations to Other Tables

1. A Permutation Argument with the [Instruction Table](instruction-table.md).
1. An Evaluation Argument with the input symbols.
1. An Evaluation Argument with the output symbols.
1. A Permutation Argument with the [Jump Stack Table](jump-stack-table.md).
1. A Permutation Argument with the [OpStack Table](operational-stack-table.md).
1. A Permutation Argument with the [RAM Table](random-access-memory-table.md).
1. An Evaluation Argument with the [Hash Table](hash-table.md) for copying the input to the hash function from the Processor to the Hash Coprocessor.
1. An Evaluation Argument with the [Hash Table](hash-table.md) for copying the hash digest from the Hash Coprocessor to the Processor.
1. A Multi-Table Set Equality argument with the [RAM Table](random-access-memory-table.md), the [JumpStack Table](jump-stack-table.md), and the [OpStack Table](operational-stack-table.md).