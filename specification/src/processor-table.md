# Processor Table

The processor consists of all registers defined in the [Instruction Set Architecture](isa.md).
Each register is assigned a column in the processor table.

**Padding**

After the Processor Table is filled in, its length being $l$, the table is padded until a total length of $2^{\lceil\log_2 l\rceil}$ is reached (or 0 if $l=0$).
Each padding row is a direct copy of the Processor Table's last row, with the exception of the cycle count column `clk`.
Column `clk` increases by 1 between any two consecutive rows, even padding rows.

**Consistency Constraints**

1. The composition of instruction buckets `ib0`-`ib5` corresponds the current instruction `ci`.

**Consistency Constraints as Polynomials**

1. `ci - (2^5·ib5 + 2^4·ib4 + 2^3·ib3 + 2^2·ib2 + 2^1·ib1 + 2^0·ib0)`

**Boundary Constraints**

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
1. In the last row, current instruction register `ci` is 0, corresponding to instruction `halt`.

**Boundary Constraints as Polynomials**

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
1. `ci`

**Transition Constraints**

Due to their complexity, instruction-specific constraints are defined [in their own section](processors-transition-constraints.md).
The following constraint applies to every cycle.

1. The cycle counter `clk` increases by 1.

**Transition Constraints as Polynomials**

1. `clk' - (clk + 1)`

**Relations to Other Tables**

1. A Permutation Argument with the [Instruction Table](instruction-table.md).
1. An Evaluation Argument with the input symbols.
1. An Evaluation Argument with the output symbols.
1. A Permutation Argument with the [Jump Stack Table](jump-stack-table.md).
1. A Permutation Argument with the [OpStack Table](operational-stack-table.md).
1. A Permutation Argument with the [RAM Table](random-access-memory-table.md).
1. An Evaluation Argument with the [Hash Table](hash-table.md) for copying the input to the hash function from the Processor to the Hash Coprocessor.
1. An Evaluation Argument with the [Hash Table](hash-table.md) for copying the hash digest from the Hash Coprocessor to the Processor.
1. A Permutation Argument with the [U32 Table](u32-table.md).

