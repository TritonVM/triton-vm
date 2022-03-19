# Triton VM Arithmetization

This document describes the arithmetization of Triton VM, whose instruction set architecture is defined [here](isa.md). An arithmetization defines two things: a algebraic execution tables (AETs) and arithmetic intermediate representation (AIR) constraints. The nature of Triton VM is that the execution trace is spread out over multiple tables, but linked through permutation and evaluation arguments.

Elsewhere, the acronym AET stands for algebraic execution *trace*. In the nomenclature of this note, a trace is a special kind of table that tracks the values of a set of registers across time.

## Processor Table

The processor consists of 52 registers, each of which is assigned a column in the corresponding table.

 - `clk` cycle counter
 - `ip` instruction pointer
 - `ci` current instruction
 - `ib0`-`ib5` instruction bits
 - `if0`-`if9` instruction flags (used to keep the AIR low degree)
 - `ni` next instruction
 - `jsp` jump address stack pointer
 - `jsv` jump address stack value
 - `st0`-`st3` operational stack elements
 - `iszero` one if top of stack is zero
 - `osp` operational stack pointer
 - `osv` operational stack value
 - `hv0`-`hv4` helper variables
 - `ramp` RAM pointer
 - `ramv` RAM value
 - `aux0`-`aux15` auxiliary registers

**Consistency Constraints**

 1. The instruction bits `ib0`-`ib5` are binary and correspond to the binary expansion of the current instruction `ci`.
 2. The instruction flags `if0`-`if10` are binary and match with the instruction bits through their defining predicates.
 3. The zero indicator `iszero` is binary and set iff the top of the stack `st0` is zero.

**Boundary Constraints**

 1. The cycle counter `clk` is zero.
 2. The instruction pointer `ip` is zero.
 3. The jump address stack pointer and value `jsp` and `jsv` are zero.
 4. The operational stack elements `st0`-`st3` are zero.
 5. The zero indicator `iszero` is one.
 6. The operational stack pointer and value `osp` and `osv` are zero
 7. The RAM pointer and value `ramp` and `ramv` are zero.
 8. The auxiliary registers `aux0`-`aux15` are zero.

**Transition Constraints**

Instruction-specific constraints are defined by the instructions. The following constraints apply to every cycle.

 1. The cycle counter `clk` increases by one.

**Relations to Other Tables**

 1. A Permutation Argument with the Instruction Table.
 2. A pair of Evaluation Arguments with the Input and Output Tables.
 3. A Permutation Argument with the Jump Stack Table.
 4. A Permutation Argument with the Opstack Table.
 5. A Permutation Argument with the Memory Table.
 6. A Permutation Argument with the Hash Table.
 7. A Permutation Argument with the Uint32 Table.

## Program Table

The Virtual Machine's Program Memory is read-only. The corresponding Program Table consists of two columns, `address` and `instruction`. The latter variable does not correspond to the processor's state but to the value of the memory at the given location.

| Address | Instruction |
|---------|-------------|
| - | - |

The Program Table is static in the sense that it is fixed before the VM runs. Moreover, the user can commit to the program by providing the Merkle root of the zipped FRI codeword. This commitment assumes that the FRI domain is fixed, which implies an upper bound on program size.

**Boundary Constraints**

 1. The first address is zero.

**Transition Constraints**

 1. The addresses increase monotonically for, by one in each row.

**Relations to other Tables**

 1. A Program-Evaluation Argument establishes that the rows of the Program Table match with the unique rows of the Instruction Table.

## Instruction Table

The Instruction Table establishes the link between the program and the instructions that are being executed by the processor. The table consists of three columns, the instruction `address`, the `current_instruction` and the `next_instruction`. It contains
 - one row for every instruction in the program (padding with zero where needed), and
 - one row for every cycle in the execution trace.

*** Relations to Other Tables**

 1. A Program Evaluation Argument establishes that the set of rows corresponds to the instructions as given by the Program Table.
 2. A Permutation Argument establishes that the set of remaining rows corresponds to the values of the registers (`ip, ci, ni`) of the Processor Table.

## Jump Stack Table

The Jump Stack Memory contains the underflow from the Jump Stack. The virtual machine defines two registers to deal with the Jump Stack: `jsp`, the return address pointer, which points to a location in jump address stack Memory; and `jsv`, the return address value, which points to a location in Program Memory.

The Jump Stack Table is a table whose columns are a subset of those of the Processor Table with one addition: the destination address `dest`. The rows are sorted by return address pointer (`jsp`), then by cycle counter (`clk`). The column `dest` contains the destination of stack-extending jump (`call`) as well as of the no-stack-change jump (`recurse`); the column `jsv` contains the source of the stack-extending jump (`call`) or equivalently the destination of the stack-shrinking jump (`return`).

The AIR for this table guarantees that the return address of a single cell of return address memory can change only if there was a `call` instruction.

An example program, execution trace, and jump stack table are shown below.

Program:

| `address` | `instruction` |
|-----------|---------------|
| `0x00`    | `foo` |
| `0x01`    | `bar` |
| `0x02`    | `call` |
| `0x03`    | `0xA0` |
| `0x04`    | `buzz` |
| `0x05`    | `bar` |
| `0x06`    | `call` |
| `0x07`    | `0xB0` |
| `0x08`    | `foo` |
| `0x09`    | `bar` |
| ... | ... |
| `0xA0`    | `buzz` |
| `0xA1`    | `foo`  |
| `0xA2`    | `bar`  |
| `0xA3`    | `return` |
| `0xA4`    | `foo` |
| ... | ... |
| `0xB0`    | `foo` |
| `0xB1`    | `call` |
| `0xB2`    | `0xC0` |
| `0xB3`    | `return` |
| `0xB4`    | `bazz` |
| ... | ... |
| `0xC0` | `buzz` |
| `0xC1` | `foo` |
| `0xC2` | `bar` |
| `0xC3` | `return` |
| `0xC4` | `buzz` |

Execution trace:

| `clk` | `ip`   | `ci`     | `ni`     | `jsv`  | `jsp`  | jump address stack | destination address stack |
|-------|--------|----------|----------|--------|--------|----------------------|---------------------------|
| 0     | `0x00` | `foo`    | `bar`    | `0x00` | 0      | []                   | []
| 1     | `0x01` | `bar`    | `call`   | `0x00` | 0      | []                   | []
| 2     | `0x02` | `call`   | `0xA0`   | `0x00` | 0      | []                   | []
| 3     | `0xA0` | `buzz`   | `foo`    | `0x04` | 1      | [`0x04`]             | [`0xA0`]
| 4     | `0xA1` | `foo`    | `bar`    | `0x04` | 1      | [`0x04`]             | [`0xA0`]
| 5     | `0xA2` | `bar`    | `return` | `0x04` | 1      | [`0x04`]             | [`0xA0`]
| 6     | `0xA3` | `return` | `foo`    | `0x04` | 1      | [`0x04`]             | [`0xA0`]
| 7     | `0x04` | `buzz`   | `bar`    | `0x00` | 0      | []                   | []
| 8     | `0x05` | `bar`    | `call`   | `0x00` | 0      | []                   | []
| 9     | `0x06` | `call`   | `0xB0`   | `0x00` | 0      | []                   | []
| 10    | `0xB0` | `foo`    | `call`   | `0x08` | 1      | [`0x08`]             | [`0xB0`]
| 11    | `0xB1` | `call`   | `0xC0`   | `0x08` | 1      | [`0x08`]             | [`0xB0`]
| 12    | `0xC0` | `buzz`   | `foo`    | `0xB3` | 2      | [`0x08`, `0xB3`]     | [`0xB0`, `0xC0`]
| 13    | `0xC1` | `foo`    | `bar`    | `0xB3` | 2      | [`0x08`, `0xB3`]     | [`0xB0`, `0xC0`]
| 14    | `0xC2` | `bar`    | `return` | `0xB3` | 2      | [`0x08`, `0xB3`]     | [`0xB0`, `0xC0`]
| 15    | `0xC3` | `return` | `buzz`   | `0xB3` | 2      | [`0x08`, `0xB3`]     | [`0xB0`, `0xC0`]
| 16    | `0xB3` | `return` | `bazz`   | `0x08` | 1      | [`0x08`]             | [`0xB0`]
| 17    | `0x08` | `foo`    | `bar`    | `0x00` | 0      | []                   | []

Memory table (i.e., actual jump stack table):

| `clk` | `ci`     | `jsp`  | `jsv`  | `dest` |
|-------|----------|--------|--------|--------|
| 0     | `foo`    | 0      | `0x00` | `0x00`
| 1     | `bar`    | 0      | `0x00` | `0x00`
| 2     | `call`   | 0      | `0x00` | `0x00`
| 7     | `buz`    | 0      | `0x00` | `0x00`
| 8     | `bar`    | 0      | `0x00` | `0x00`
| 9     | `call`   | 0      | `0x00` | `0x00`
| 17    | `foo`    | 0      | `0x00` | `0x00`
| 3     | `buz`    | 1      | `0x04` | `0xA0`
| 4     | `foo`    | 1      | `0x04` | `0xA0`
| 5     | `bar`    | 1      | `0x04` | `0xA0`
| 6     | `return` | 1      | `0x04` | `0xA0`
| 10    | `foo`    | 1      | `0x08` | `0xB0`
| 11    | `call`   | 1      | `0x08` | `0xB0`
| 16    | `return` | 1      | `0x08` | `0xB0`
| 12    | `buzz`   | 2      | `0xB3` | `0xC0`
| 13    | `foo`    | 2      | `0xB3` | `0xC0`
| 14    | `bar`    | 2      | `0xB3` | `0xC0`
| 15    | `return` | 2      | `0xB3` | `0xC0`

**Boundary Constraints**

 1. All registers are initially zero.

**Transition Constraints**

 1. The jump address stack pointer `jsp` increases by one, *or*
 2. (`jsp`, `jsv` and `dest` remain the same and) the cycle counter `clk` increases by one, *or*
 3. (`jsp`, `jsv` and `dest` remain the same and) the current instruction `ci` is `call`, *or*
 4. (`jsp` remains the same and) the current instruction `ci` is `return`.

**Relations to Other Tables**

 1. A Permutation Argument establishes that the rows match with the rows in the Processor Table.

### Operational Stack

The operational stack is where the program stores simple elementary operations, function arguments, and pointers to important objects. There are four registers that the program can access directly; these registers correspond to the top of the stack. The rest of the operational stack is stored in a dedicated memory object called Operational Stack Memory.

The operational stack always contains at least 4 elements. Instructions that reduce the number of stack elements to less than four are illegal.

The operational stack table contains a subset of the rows of the processor table -- specifically, the cycle counter `clk`, the current instruction `current_instruction`, the operation stack value `osv` and pointer `osp`. The rows of the operational stack table are sorted by operational stack pointer `osp` first, cycle `clk` second.

The mechanics are best illustrated by an example.

Execution trace:

| `clk` | `current_instruction` | `next_instruction` | `osv` | `osp` | operational stack |
|-------|-----------------------|--------------------|-------|-------|-------------------|
| 0     | `push`                | 0x01               | 0     | 0     | [0,0,0,0] |
| 1     | `push`                | 0x02               | 0     | 1     | [0;0,0,0,0x01] |
| 2     | `push`                | 0x03               | 0     | 2     | [0,0;0,0,0x01,0x02] |
| 3     | `push`                | 0x04               | 0     | 3     | [0,0,0;0,0x01,0x02,0x03] |
| 4     | `push`                | 0x05               | 0     | 4     | [0,0,0,0;0x01,0x02,0x03,0x04] |
| 5     | `foo`                 | `add`              | 0x01  | 5     | [0,0,0,0,0x01;0x02,0x03,0x04,0x05] |
| 6     | `add`                 | `pop`              | 0x01  | 5     | [0,0,0,0,0x01;0x02,0x03,0x04,0x05] |
| 7     | `pop`                 | `add`              | 0     | 4     | [0,0,0,0;0x01,0x02,0x03,0x09] |
| 8     | `add`                 | `add`              | 0     | 3     | [0,0,0;0,0x01,0x02,0x03] |
| 9     | `add`                 | `pop`              | 0     | 2     | [0,0;0,0,0x01,0x05] |
| 10    | `pop`                 | `foo`              | 0     | 1     | [0;0,0,0,0x06] |
| 11    | `foo`                 | `pop`              | 0     | 0     | [0,0,0,0] |
| 12    | `pop`                 | -                  | 0     | 0     | **illegal command** |

Memory table (i.e., the actual operational stack table):

| `clk`  | `current_instruction` | `osv` | `osp` |
|--------|-----------------------|-------|-------|
| 0      | `push`                | 0     | 0     |
| 11     | `foo`                 | 0     | 0     |
| 1      | `push`                | 0     | 1     |
| 10     | `pop`                 | 0     | 1     |
| 2      | `push`                | 0     | 2     |
| 9      | `add`                 | 0     | 2     |
| 3      | `push`                | 0     | 3     |
| 8      | `add`                 | 0     | 3     |
| 4      | `push`                | 0     | 4     |
| 7      | `pop`                 | 0x01  | 4     |
| 5      | `foo`                 | 0x01  | 5     |
| 6      | `add`                 | 0x01  | 5     |

**Boundary Conditions**

None.

**Transition Constraints**

 1. The operational stack pointer `osp` increases by 1, *or*
 2. The `current_instruction` is `push`-like, *or*
 3. There is no change in operational stack value `osv`.

**Relations to Other Tables**

 1. A Permutation Argument establishes that the rows of the operational stack table correspond to the rows of the Processor Table.

## Random Access Memory

The RAM is accessible through `load` and `save` commands. The RAM Table has three columns: the cycle counter `clk`, RAM address `memory_address`, and the value of the memory at that address `memory_value`. The columns are identical to the columns of the same name in the Processor Table, up to the order of the rows. The rows are sorted by memory address first, then by cycle counter.

**Boundary Constraints**

None.

**Transition Constraints**

 1. If the `memory_address` changes, then the new `memory_value` must be zero
 2. If the `memory_address` does not change and the `memory_value` does change, then the cycle counter `clk` must increase by one.

**Relations to Other Tables**

 1. A Permutation Argument establishes that the rows in the RAM Table correspond to the rows of the Processor Table.

## I/O Tables

There are two I/O Tables: one for the input, and one for the output. Both consist of a single column. The input and output can be committed to in the form of the FRI codeword Merkle roots associated with their interpolants (which may or may not integrate randomness).

**Boundary Constraints**

None.

**Transition Constraints**

None.

**Relations to Other Tables**

 1. A pair of evaluation arguments establishe that the symbols read by the processor as input, or written as output, correspond with the symbols listed in the corresponding I/O Table.

## Hash Coprocessor

The processor has 16 auxiliary registers. The instruction `xlix` applies the Rescue-XLIX permutation to them in one cycle. What happens in the background is that the auxiliary registers are copied to the hash coprocessor, which then runs 7 the rounds of the Rescue-XLIX, and then copies the 16 values back. This single-cycle hashing instruction is enabled by a Hash Table of 17 columns -- one extra to indicate round index.

**Boundary Constraints**

 1. The round index starts at 0.

**Transition Constraints**

 1. The round index increases by 1 modulo 8.
 2. On multiples of 8 there is no other constraint.
 3. For all other rows, the $i$th round of Rescue-XLIX is applied, where $i$ is the round index.

**Relations to Other Tables**

 1. A Permutation Argument establishes that whenever the processor executes an `xlix` instruction, the values of auxiliary registers correspond to some row in the Hash Table with index 0 mod 8 and the values of the auxiliary registers in the next cycle correspond to the values of the Hash Table 7 rows layer.

## Uint32 Operations

The Uint32 Operations Table is a lookup table for 'difficult' 32-bit unsigned integer operations.

| `idc` | LHS      | RHS      | EQ     | LT    | AND       | OR       | XOR       | REV |
|-----|----------|----------|--------|-------|-----------|----------|-----------|-----|
| 1  | `a`      | `b`      | `a==b` | `a<b` | `a and b` | `a or b` | `a xor b` | `rev(a)` |
| 1  | `a >> 1` | `b >> 1` | - | - | - | - | - | - |
| ... | - | - | - | - | - | - | - | - | - | - |
| 0 | `0` | `0` | `1` | `0` | `0` | `0` | `0` | `0` |
| 1 | `c` | `d` | - | - | - | - | - | - |

The AIR verifies the correct update of each consecutive pair of rows. In every row one bit is eliminated. Only when the previous row is all zeros (with a 1 in the column for `EQ`) can a new row be inserted.

The AIR constraints establish that the entire table is consistent. Copy-constraints establish that logical and bitwise operations were computed correctly.

**Boundary Constrants**

None.

**Transition Constraints**

 1. The indicator `idc` is binary.
 2. If the indicator `idc` is zero, so are LHS, RHS, LT, AND, OR, XOR, REV and 1-EQ.
 3. If the indicator `idc` is nonzero across two rows, the current and next row follow the one-bit update rules.

**Terminal Constraints**

 1. The indicator `idc` in the last row is zero.

**Relations to Other Tables**

 1. A Permutation Argument establishes that whenever the processor executes a uint32 operation, the operands and result exist as a row in this table.
