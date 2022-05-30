# Triton VM Arithmetization

This document describes the arithmetization of Triton VM, whose instruction set architecture is defined [here](isa.md).
An arithmetization defines two things:
1. algebraic execution tables (AETs), and
1. arithmetic intermediate representation (AIR) constraints.
The nature of Triton VM is that the execution trace is spread out over multiple tables, but linked through permutation and evaluation arguments.

Elsewhere, the acronym AET stands for algebraic execution *trace*.
In the nomenclature of this note, a trace is a special kind of table that tracks the values of a set of registers across time.

The values of all registers, and consequently the elements on the stack, in memory, and so on, are elements of the _B-field_, i.e., $\mathbb{F}_p$ where $p=2^{64}-2^{32}+1$.
All values of columns corresponding to one such register are elements of the B-Field as well.
The entries of a table's columns corresponding to Evaluation or Permutation Arguments are elements from the _X-Field_ $\mathbb{F}_{p^3}$.

For each table, up to three lists containing constraints of different type are given:
1. Consistency Constraints, establishing consistency within any given row,
1. Boundary Constraints, defining values in a table's first row and, in one case, also the last, and
1. Transition Constraints, establishing the consistency of two consecutive rows in relation to each other.

Together, all these constraints constitute the AIR constraints.

## Algebraic Execution Tables

There are 10 tables in TritonVM.
Their relation is described by below figure.
A red arrow indicates an Evaluation Argument, a blue line indicates a Permutation Argument.

![](img/aet-relations.png)


### Processor Table

The processor consists of all registers defined in the [Instruction Set Architecture](isa.md).
Each register is assigned a column in the processor table.

**Consistency Constraints**

1. The composition of instruction buckets `ib0`-`ib5` corresponds the current instruction `ci`.
1. The inverse register `inv` contains the inverse of `st0` if it is nonzero and zero otherwise.

**Boundary Constraints**

1. The cycle counter `clk` is zero.
1. The instruction pointer `ip` is zero.
1. The jump address stack pointer and value `jsp` and `jsv` are zero.
1. The operational stack elements `st0`-`st7` are zero.
1. The inverse register `inv` is zero.
1. The operational stack pointer `osp` is `8`
1. The operational stack value `osv` is zero
1. The RAM pointer and value `ramp` and `ramv` are zero.
1. The auxiliary registers `aux0`-`aux15` are zero.
1. In the last row, `ci` is zero, corresponding to instruction `halt`.

**Transition Constraints**

Due to their complexity, instruction-specific constraints are defined [in their own section further below](#instruction-specific-transition-constraints).
The following constraints apply to every cycle.

1. The cycle counter `clk` increases by one.

**Relations to Other Tables**

1. A Permutation Argument with the [Instruction Table](#instruction-table).
1. A pair of Evaluation Arguments with the [Input and Output Tables](#io-tables).
1. A Permutation Argument with the [Jump Stack Table](#jump-stack-table).
1. A Permutation Argument with the [Opstack Table](#operational-stack-table).
1. A Permutation Argument with the [RAM Table](#random-access-memory-table).
1. A Permutation Argument with the [Hash Table](#hash-coprocessor-table).
1. A Permutation Argument with the [uint32 Table](#uint32-operations-table).

### Program Table

The Virtual Machine's Program Memory is read-only.
The corresponding Program Table consists of two columns, `address` and `instruction`.
The latter variable does not correspond to the processor's state but to the value of the memory at the given location.

| Address | Instruction |
|:--------|:------------|
| -       | -           |

The Program Table is static in the sense that it is fixed before the VM runs.
Moreover, the user can commit to the program by providing the Merkle root of the zipped FRI codeword.
This commitment assumes that the FRI domain is fixed, which implies an upper bound on program size.

**Boundary Constraints**

1. The first address is zero.

**Transition Constraints**

1. The address increases by one.

**Relations to other Tables**

1. An Evaluation Argument establishes that the rows of the Program Table match with the unique rows of the [Instruction Table](#instruction-table).

### Instruction Table

The Instruction Table establishes the link between the program and the instructions that are being executed by the processor.
The table consists of three columns:
1. the instruction's `address`,
1. the `current_instruction`, and
1. the `next_instruction`.

It contains
- one row for every instruction in the [Program Table](#program-table), i.e., one row for every available instruction, and
- one row for every cycle `clk` in the [Processor Table](#processor-table), i.e., one row for every executed instruction.
The rows are sorted by `address`.
 

*** Relations to Other Tables**

1. An Evaluation Argument establishes that the set of rows corresponds to the instructions as given by the [Program Table](#program-table).
1. A Permutation Argument establishes that the set of remaining rows corresponds to the values of the registers (`ip, ci, ni`) of the [Processor Table](#processor-table).

### Jump Stack Table

The Jump Stack Memory contains the underflow from the Jump Stack.
TritonVM defines three registers to deal with the Jump Stack:
1. `jsp`, the jump stack pointer, which points to a location in Jump Stack Memory
1. `jso`, the last jump's origin, which points to a location in Program Memory, and
1. `jsd`, the last jump's destination, which also points to a location in Program Memory.

The Jump Stack Table is a table whose columns are a subset of those of the Processor Table.
The rows are sorted by jump stack pointer (`jsp`), then by cycle counter (`clk`).
The column `jsd` contains the destination of stack-extending jump (`call`) as well as of the no-stack-change jump (`recurse`);
the column `jso` contains the source of the stack-extending jump (`call`) or equivalently the destination of the stack-shrinking jump (`return`).

The AIR for this table guarantees that the return address of a single cell of return address memory can change only if there was a `call` instruction.

An example program, execution trace, and jump stack table are shown below.

Program:

| `address` | `instruction` |
|:----------|:--------------|
| `0x00`    | `foo`         |
| `0x01`    | `bar`         |
| `0x02`    | `call`        |
| `0x03`    | `0xA0`        |
| `0x04`    | `buzz`        |
| `0x05`    | `bar`         |
| `0x06`    | `call`        |
| `0x07`    | `0xB0`        |
| `0x08`    | `foo`         |
| `0x09`    | `bar`         |
| ⋮         | ⋮             |
| `0xA0`    | `buzz`        |
| `0xA1`    | `foo`         |
| `0xA2`    | `bar`         |
| `0xA3`    | `return`      |
| `0xA4`    | `foo`         |
| ⋮         | ⋮             |
| `0xB0`    | `foo`         |
| `0xB1`    | `call`        |
| `0xB2`    | `0xC0`        |
| `0xB3`    | `return`      |
| `0xB4`    | `bazz`        |
| ⋮         | ⋮             |
| `0xC0`    | `buzz`        |
| `0xC1`    | `foo`         |
| `0xC2`    | `bar`         |
| `0xC3`    | `return`      |
| `0xC4`    | `buzz`        |

Execution trace:

| `clk` | `ip`   | `ci`     | `nia`    | `jsp` | `jso`  | `jsd`  | jump stack                           |
|:------|:-------|:---------|:---------|:------|:-------|:-------|:-------------------------------------|
| 0     | `0x00` | `foo`    | `bar`    | 0     | `0x00` | `0x00` | [ ]                                  |
| 1     | `0x01` | `bar`    | `call`   | 0     | `0x00` | `0x00` | [ ]                                  |
| 2     | `0x02` | `call`   | `0xA0`   | 0     | `0x00` | `0x00` | [ ]                                  |
| 3     | `0xA0` | `buzz`   | `foo`    | 1     | `0x04` | `0xA0` | [(`0x04`, `0xA0`)]                   |
| 4     | `0xA1` | `foo`    | `bar`    | 1     | `0x04` | `0xA0` | [(`0x04`, `0xA0`)]                   |
| 5     | `0xA2` | `bar`    | `return` | 1     | `0x04` | `0xA0` | [(`0x04`, `0xA0`)]                   |
| 6     | `0xA3` | `return` | `foo`    | 1     | `0x04` | `0xA0` | [(`0x04`, `0xA0`)]                   |
| 7     | `0x04` | `buzz`   | `bar`    | 0     | `0x00` | `0x00` | [ ]                                  |
| 8     | `0x05` | `bar`    | `call`   | 0     | `0x00` | `0x00` | [ ]                                  |
| 9     | `0x06` | `call`   | `0xB0`   | 0     | `0x00` | `0x00` | [ ]                                  |
| 10    | `0xB0` | `foo`    | `call`   | 1     | `0x08` | `0xB0` | [(`0x08`, `0xB0`)]                   |
| 11    | `0xB1` | `call`   | `0xC0`   | 1     | `0x08` | `0xB0` | [(`0x08`, `0xB0`)]                   |
| 12    | `0xC0` | `buzz`   | `foo`    | 2     | `0xB3` | `0xC0` | [(`0x08`, `0xB0`), (`0xB3`, `0xC0`)] |
| 13    | `0xC1` | `foo`    | `bar`    | 2     | `0xB3` | `0xC0` | [(`0x08`, `0xB0`), (`0xB3`, `0xC0`)] |
| 14    | `0xC2` | `bar`    | `return` | 2     | `0xB3` | `0xC0` | [(`0x08`, `0xB0`), (`0xB3`, `0xC0`)] |
| 15    | `0xC3` | `return` | `buzz`   | 2     | `0xB3` | `0xC0` | [(`0x08`, `0xB0`), (`0xB3`, `0xC0`)] |
| 16    | `0xB3` | `return` | `bazz`   | 1     | `0x08` | `0xB0` | [(`0x08`, `0xB0`)]                   |
| 17    | `0x08` | `foo`    | `bar`    | 0     | `0x00` | `0x00` | [ ]                                  |

Jump Stack Table:


| `clk` | `ci`     | `jsp` | `jso`  | `jsd`  |
|:------|:---------|:------|:-------|:-------|
| 0     | `foo`    | 0     | `0x00` | `0x00` |
| 1     | `bar`    | 0     | `0x00` | `0x00` |
| 2     | `call`   | 0     | `0x00` | `0x00` |
| 7     | `buzz`   | 0     | `0x00` | `0x00` |
| 8     | `bar`    | 0     | `0x00` | `0x00` |
| 9     | `call`   | 0     | `0x00` | `0x00` |
| 17    | `foo`    | 0     | `0x00` | `0x00` |
| 3     | `buzz`   | 1     | `0x04` | `0xA0` |
| 4     | `foo`    | 1     | `0x04` | `0xA0` |
| 5     | `bar`    | 1     | `0x04` | `0xA0` |
| 6     | `return` | 1     | `0x04` | `0xA0` |
| 10    | `foo`    | 1     | `0x08` | `0xB0` |
| 11    | `call`   | 1     | `0x08` | `0xB0` |
| 16    | `return` | 1     | `0x08` | `0xB0` |
| 12    | `buzz`   | 2     | `0xB3` | `0xC0` |
| 13    | `foo`    | 2     | `0xB3` | `0xC0` |
| 14    | `bar`    | 2     | `0xB3` | `0xC0` |
| 15    | `return` | 2     | `0xB3` | `0xC0` |

**Boundary Constraints**

1. All registers are initially zero.

**Transition Constraints**

1. The jump stack pointer `jsp` increases by one, *or*
1. (`jsp`, `jso` and `jsd` remain the same and the cycle counter `clk` increases by one), *or*
1. (`jsp`, `jso` and `jsd` remain the same and the current instruction `ci` is `call`), *or*
1. (`jsp` remains the same and the current instruction `ci` is `return`).

**Relations to Other Tables**

1. A Permutation Argument establishes that the rows match with the rows in the [Processor Table](#processor-table).

### Operational Stack Table

The operational stack is where the program stores simple elementary operations, function arguments, and pointers to important objects.
There are eight registers (`st0` through `st7`) that the program can access directly.
These registers correspond to the top of the stack.
The rest of the operational stack is stored in a dedicated memory object called Operational Stack Underflow Memory.

The operational stack table contains a subset of the columns of the processor table –
specifically, the cycle counter `clk`, the current instruction `ci`, the operation stack value `osv` and pointer `osp`.
The rows of the operational stack table are sorted by operational stack pointer `osp` first, cycle count `clk` second.

The mechanics are best illustrated by an example.
For illustrative purposes only, we use four stack registers `st0` through `st3` in the example.
TritonVM has eight stack registers, `st0` through `st7`.

Execution trace:

| `clk` | `ci`   | `nia` | `osv` | `osp` | OpStack Underflow Memory | `st3` | `st2` | `st1` | `st0` |
|:------|:-------|:------|:------|:------|:-------------------------|:------|:------|:------|:------|
| 0     | `push` | 42    | 0     | 4     | [ ]                      | 0     | 0     | 0     | 0     |
| 1     | `push` | 43    | 0     | 5     | [0]                      | 0     | 0     | 0     | 42    |
| 2     | `push` | 44    | 0     | 6     | [0,0]                    | 0     | 0     | 42    | 43    |
| 3     | `push` | 45    | 0     | 7     | [0,0,0]                  | 0     | 42    | 43    | 44    |
| 4     | `push` | 46    | 0     | 8     | [0,0,0,0]                | 42    | 43    | 44    | 45    |
| 5     | `foo`  | `add` | 42    | 9     | [0,0,0,0,42]             | 43    | 44    | 45    | 46    |
| 6     | `add`  | `pop` | 42    | 9     | [0,0,0,0,42]             | 43    | 44    | 45    | 46    |
| 7     | `pop`  | `add` | 0     | 8     | [0,0,0,0]                | 42    | 43    | 44    | 91    |
| 8     | `add`  | `add` | 0     | 7     | [0,0,0]                  | 0     | 42    | 43    | 44    |
| 9     | `add`  | `pop` | 0     | 6     | [0,0]                    | 0     | 0     | 42    | 87    |
| 10    | `pop`  | `foo` | 0     | 5     | [0]                      | 0     | 0     | 0     | 129   |
| 11    | `foo`  | `pop` | 0     | 4     | [ ]                      | 0     | 0     | 0     | 0     |
| 12    | `pop`  | `bar` | 0     | 3     | 💥                        | 0     | 0     | 0     | 0     |

Operational Stack Table:

| `clk` | `ci`   | `osv` | `osp` |
|:------|:-------|:------|:------|
| 0     | `push` | 0     | 4     |
| 11    | `foo`  | 0     | 4     |
| 1     | `push` | 0     | 5     |
| 10    | `pop`  | 0     | 5     |
| 2     | `push` | 0     | 6     |
| 9     | `add`  | 0     | 6     |
| 3     | `push` | 0     | 7     |
| 8     | `add`  | 0     | 7     |
| 4     | `push` | 0     | 8     |
| 7     | `pop`  | 0     | 8     |
| 5     | `foo`  | 42    | 9     |
| 6     | `add`  | 42    | 9     |

**Boundary Conditions**

1. `osv` is zero.
1. `osp` is the number of available stack registers, i.e., 8.

**Transition Constraints**

1. The operational stack pointer `osp` increases by 1, *or*
1. The `current_instruction` is `push`-like, *or*
1. There is no change in operational stack value `osv`.

**Relations to Other Tables**

1. A Permutation Argument establishes that the rows of the operational stack table correspond to the rows of the [Processor Table](#processor-table).

### Random Access Memory Table

The RAM is accessible through `read_mem` and `write_mem` commands.
The RAM Table has three columns:
the cycle counter `clk`, RAM address pointer `ramp`, and the value of the memory at that address `ramv`.
The columns are identical to the columns of the same name in the Processor Table, up to the order of the rows.
The rows are sorted by memory address first, then by cycle counter.

**Boundary Constraints**

None.

**Transition Constraints**

1. If the `ramp` changes, then the new `ramv` must be zero
1. If the `ramp` does not change and the `ramv` does change, then the cycle counter `clk` must increase by one.

**Relations to Other Tables**

1. A Permutation Argument establishes that the rows in the RAM Table correspond to the rows of the [Processor Table](#processor-table).

### I/O Tables

There are two I/O Tables:
one for the input, and one for the output.
Both consist of a single column.
The input and output can be committed to in the form of the FRI codeword Merkle roots associated with their interpolants (which may or may not integrate randomness).

**Boundary Constraints**

None.

**Transition Constraints**

None.

**Relations to Other Tables**

1. A pair of evaluation arguments establish that the symbols read by the [processor](#processor-table) as input, or written as output, correspond with the symbols listed in the corresponding I/O Table.

### Hash Coprocessor Table

The processor has 16 auxiliary registers.
The instruction `xlix` applies the Rescue-XLIX permutation to them in one cycle.
What happens in the background is that the auxiliary registers are copied to the hash coprocessor, which then runs 7 the rounds of the Rescue-XLIX, and then copies the 16 values back.
This single-cycle hashing instruction is enabled by a Hash Table of 17 columns – one extra to indicate round index.

**Boundary Constraints**

1. The round index starts at 0.

**Transition Constraints**

1. The round index increases by 1 modulo 8.
1. On multiples of 8 there is no other constraint.
1. For all other rows, the $i$th round of Rescue-XLIX is applied, where $i$ is the round index.

**Relations to Other Tables**

1. A Permutation Argument establishes that whenever the [processor](#processor-table) executes an `xlix` instruction, the values of auxiliary registers correspond to some row in the Hash Table with index 0 mod 8 and the values of the auxiliary registers in the next cycle correspond to the values of the Hash Table 7 rows layer.

### Uint32 Operations Table

The Uint32 Operations Table is a lookup table for 'difficult' 32-bit unsigned integer operations.

| `idc` | LHS      | RHS      | LT                  | AND                     | XOR                     | REV           |
|:------|:---------|:---------|:--------------------|:------------------------|:------------------------|:--------------|
| 1     | `a`      | `b`      | `a<b`               | `a and b`               | `a xor b`               | `rev(a)`      |
| 0     | `a >> 1` | `b >> 1` | `(a >> 1)<(b >> 1)` | `(a >> 1) and (b >> 1)` | `(a >> 1) xor (b >> 1)` | `rev(a >> 1)` |
| 0     | `a >> 2` | `b >> 2` | …                   | …                       | …                       | …             |
| …     | …        | …        | …                   | …                       | …                       | …             |
| 0     | `0`      | `0`      | `0`                 | `0`                     | `0`                     | `0`           |
| 1     | `c`      | `d`      | `c<d`               | `c and d`               | `c xor d`               | `rev(c)`      |
| 0     | `c >> 1` | `d >> 1` | …                   | …                       | …                       | …             |
| …     | …        | …        | …                   | …                       | …                       | …             |
| 0     | `0`      | `0`      | `0`                 | `0`                     | `0`                     | `0`           |
| …     | …        | …        | …                   | …                       | …                       | …             |

The AIR verifies the correct update of each consecutive pair of rows.
In every row one bit is eliminated.
Only when the previous row is all zeros can a new row be inserted.

The AIR constraints establish that the entire table is consistent.
Copy-constraints establish that logical and bitwise operations were computed correctly.

For every instruction in the `u32_op` instruction group (`lt`, `and`, `xor`, `reverse`, `div`), there is a dedicated Permutation Argument with the [Processor Table](#processor-table).

**Consistency Constraints**

1. The indicator `idc` is binary.

**Boundary Constrants**

1. The indicator `idc` in the first row is one.
1. The indicator `idc` in the last row is zero.

**Transition Constraints**

1. If the indicator `idc` is zero, so are LHS, RHS, LT, AND, XOR, and REV.
1. If the indicator `idc` is nonzero across two rows, the current and next row follow the one-bit update rules.

**Relations to Other Tables**

1. A Permutation Argument establishes that whenever the [processor](#processor-table) executes a uint32 operation, the operands and result exist as a row in the uint32 table.

## Instruction-Specific Transition Constraints

Due to their complexity, the transition constraints for the [Processor Table](#Processor-Table) are listed and explained in this section.

### Instruction Groups

To keep the degrees of the AIR polynomials low, instructions are grouped based on their effect.
An instruction's effect not captured by the groups it is part of needs to be arithmetized separately and is described in the next sections.

| group name      | description                                                                                         |
|:----------------|:----------------------------------------------------------------------------------------------------|
| `decompose_arg` | instruction's argument held in `nia` is binary decomposed into helper registers `hv0` through `hv3` |
| `step_1`        | instruction pointer `ip` increases by 1                                                             |
| `step_2`        | instruction pointer `ip` increases by 2                                                             |
| `no_ram_access` | no modification of registers concerning RAM, i.e., `ramp` and `ramv`                                |
| `no_aux_change` | no modification of `aux` registers                                                                  |
| `u32_op`        | instruction is a 32-bit unsigned integer instruction                                                |
| `grow_stack`    | a new element is put onto the stack, rest of the stack remains unchanged                            |
| `keep_stack`    | stack remains unchanged                                                                             |
| `shrink_stack`  | stack's top-most element is removed, rest of the stack remains unchanged. Needs `hv4`               |
| `unop`          | stack's top-most element is modified, rest of stack remains unchanged                               |
| `binop`         | stack's two top-most elements are modified, rest of stack remains unchanged                         |

A summary of all instructions and which groups they are part of is given in the following table.

| instruction      | `has_arg`* | `decompose_arg` | `step_1` | `step_2` | `no_ram_access` | `no_aux_change` | `u32_op` | `grow_stack` | `keep_stack` | `shrink_stack` | `unop` | `binop` |
|:-----------------|:-----------|:----------------|:---------|:---------|:----------------|:----------------|:---------|:-------------|:-------------|:---------------|:-------|:--------|
| `pop`            |            |                 | x        |          | x               | x               |          |              |              | x              |        |         |
| `push` + `a`     | x          |                 |          | x        | x               | x               |          | x            |              |                |        |         |
| `divine`         |            |                 | x        |          | x               | x               |          | x            |              |                |        |         |
| `dup` + `i`      | x          | x               |          | x        | x               | x               |          | x            |              |                |        |         |
| `swap` + `i`     | x          | x               |          | x        | x               | x               |          |              |              |                |        |         |
| `nop`            |            |                 | x        |          | x               | x               |          |              | x            |                |        |         |
| `skiz`           |            |                 |          |          | x               | x               |          |              |              | x              |        |         |
| `call` + `d`     | x          |                 |          |          | x               | x               |          |              | x            |                |        |         |
| `return`         |            |                 |          |          | x               | x               |          |              | x            |                |        |         |
| `recurse`        |            |                 |          |          | x               | x               |          |              | x            |                |        |         |
| `assert`         |            |                 | x        |          | x               | x               |          |              |              | x              |        |         |
| `halt`           |            |                 | x        |          | x               | x               |          |              | x            |                |        |         |
| `read_mem`       |            |                 | x        |          |                 | x               |          | x            |              |                |        |         |
| `write_mem`      |            |                 | x        |          |                 | x               |          |              |              | x              |        |         |
| `xlix`           |            |                 | x        |          | x               |                 |          |              | x            |                |        |         |
| `clearall`       |            |                 | x        |          | x               |                 |          |              | x            |                |        |         |
| `squeeze` + `i`  | x          | x               |          | x        | x               | x               |          | x            |              |                |        |         |
| `absorb` + `i`   | x          | x               |          | x        | x               |                 |          |              |              | x              |        |         |
| `divine_sibling` |            |                 | x        |          | x               |                 |          |              |              |                | x      |         |
| `compare_digest` |            |                 | x        |          | x               | x               |          | x            |              |                |        |         |
| `add`            |            |                 | x        |          | x               | x               |          |              |              |                |        | x       |
| `mul`            |            |                 | x        |          | x               | x               |          |              |              |                |        | x       |
| `invert`         |            |                 | x        |          | x               | x               |          |              |              |                | x      |         |
| `split`          |            |                 | x        |          | x               | x               |          |              |              |                |        |         |
| `eq`             |            |                 | x        |          | x               | x               |          |              |              |                |        | x       |
| `lt`             |            |                 | x        |          | x               | x               | x        |              |              |                |        | x       |
| `and`            |            |                 | x        |          | x               | x               | x        |              |              |                |        | x       |
| `xor`            |            |                 | x        |          | x               | x               | x        |              |              |                |        | x       |
| `reverse`        |            |                 | x        |          | x               | x               | x        |              |              |                | x      |         |
| `div`            |            |                 | x        |          | x               | x               | x        |              |              |                |        |         |
| `xxadd`          |            |                 | x        |          | x               | x               |          |              |              |                |        |         |
| `xxmul`          |            |                 | x        |          | x               | x               |          |              |              |                |        |         |
| `xinv`           |            |                 | x        |          | x               | x               |          |              |              |                |        |         |
| `xbmul`          |            |                 | x        |          | x               | x               |          |              |              |                |        |         |
| `read_io`        |            |                 | x        |          | x               | x               |          | x            |              |                |        |         |
| `write_io`       |            |                 | x        |          | x               | x               |          |              |              | x              |        |         |

\*
Instruction Group `has_arg` is a _virtual_ instruction group.
That is, this instruction group is not represented by the instruction bucket registers `ib`.
The virtual instruction group `has_arg` is required for correct behavior of instruction `skiz`, for which the instruction pointer `ip` needs to increment by either 1, or 2, or 3.
The concrete value depends on the top of the stack `st0` and the next instruction, held in `nia`.
If (and only if) the current instruction `ci` is instruction `skiz`, then the opcode held in register `nia` is deconstructed into helper variable registers `hv`.
This is similar to how `ci` is (always) deconstructed into instruction bucket registers `ib`.
The virtual instruction bucket `has_arg` helps in identifying optimal opcodes for all instructions during development of TritonVM.

In the following sections, a register marked with a `'` refers to the next state of that register.
For example, `st0' = st0 + 2` means that stack register `st0` is incremented by 2.
An alternative view for the same concept is that registers marked with `'` are those of the next row in the table.

#### Indicator Polynomials `ind_i(hv3, hv2, hv1, hv0)`

For instructions [`dup`](#instruction-dup-i), [`swap`](#instruction-swap-i), [`squeeze`](#instruction-squeeze-i), and [`absorb`](#instruction-absorb-i), it is beneficial to have polynomials that evaluate to 1 if the instruction's argument `i` is a specific value, and to 0 otherwise.
This allows indicating which registers are constraint, and in which way they are, depending on `i`.
This is the purpose of the _indicator polynomials_ `ind_i`.
Evaluated on the binary decomposition of `i`, they show the behavior described above.

For example, take `i = 13`.
The corresponding binary decomposition is `(hv3, hv2, hv1, hv0) = (1, 1, 0, 1)`.
Indicator polynomial `ind_13(hv3, hv2, hv1, hv0)` is `hv3·hv2·(1 - hv1)·hv0`.
It evaluates to 1 on `(1, 1, 0, 1)`, i.e., `ind_13(1, 1, 0, 1) = 1`.
Any other indicator polynomial, like `ind_7`, evaluates to 0 on `(1, 1, 0, 1)`.

Below, you can find a list of all 16 indicator polynomials.

1.  `ind_0(hv3, hv2, hv1, hv0) = (1 - hv3)·(1 - hv2)·(1 - hv1)·(1 - hv0)`
1.  `ind_1(hv3, hv2, hv1, hv0) = (1 - hv3)·(1 - hv2)·(1 - hv1)·hv0`
1.  `ind_2(hv3, hv2, hv1, hv0) = (1 - hv3)·(1 - hv2)·hv1·(1 - hv0)`
1.  `ind_3(hv3, hv2, hv1, hv0) = (1 - hv3)·(1 - hv2)·hv1·hv0`
1.  `ind_4(hv3, hv2, hv1, hv0) = (1 - hv3)·hv2·(1 - hv1)·(1 - hv0)`
1.  `ind_5(hv3, hv2, hv1, hv0) = (1 - hv3)·hv2·(1 - hv1)·hv0`
1.  `ind_6(hv3, hv2, hv1, hv0) = (1 - hv3)·hv2·hv1·(1 - hv0)`
1.  `ind_7(hv3, hv2, hv1, hv0) = (1 - hv3)·hv2·hv1·hv0`
1.  `ind_8(hv3, hv2, hv1, hv0) = hv3·(1 - hv2)·(1 - hv1)·(1 - hv0)`
1.  `ind_9(hv3, hv2, hv1, hv0) = hv3·(1 - hv2)·(1 - hv1)·hv0`
1. `ind_10(hv3, hv2, hv1, hv0) = hv3·(1 - hv2)·hv1·(1 - hv0)`
1. `ind_11(hv3, hv2, hv1, hv0) = hv3·(1 - hv2)·hv1·hv0`
1. `ind_12(hv3, hv2, hv1, hv0) = hv3·hv2·(1 - hv1)·(1 - hv0)`
1. `ind_13(hv3, hv2, hv1, hv0) = hv3·hv2·(1 - hv1)·hv0`
1. `ind_14(hv3, hv2, hv1, hv0) = hv3·hv2·hv1·(1 - hv0)`
1. `ind_15(hv3, hv2, hv1, hv0) = hv3·hv2·hv1·hv0`

#### Group `decompose_arg`

##### Description

1. The helper variables are the decomposition of the instruction's argument, which is held in register `nia`.
1. The helper variable `hv0` is either 0 or 1.
1. The helper variable `hv1` is either 0 or 1.
1. The helper variable `hv2` is either 0 or 1.
1. The helper variable `hv3` is either 0 or 1.

##### Polynomials

1. `nia - (8·hv3 + 4·hv2 + 2·hv1 + hv0)`
1. `hv0·(hv0 - 1)`
1. `hv1·(hv1 - 1)`
1. `hv2·(hv2 - 1)`
1. `hv3·(hv3 - 1)`

#### Group `step_1`

##### Description

1. The instruction pointer increments by 1.

##### Polynomials

1. `ip' - (ip + 1)`

#### Group `step_2`

##### Description

1. The instruction pointer increments by 2.

##### Polynomials

1. `ip' - (ip + 2)`

#### Group `no_ram_access`

##### Description

1. RAM pointer register `ramp` does not change.
1. RAM value register `ramv` does not change.

##### Polynomials

1. `ramp' - ramp`
1. `ramv' - ramv`

#### Group `no_aux_change`

##### Description

1. Auxiliary register `aux0` does not change.
1. Auxiliary register `aux1` does not change.
1. Auxiliary register `aux2` does not change.
1. Auxiliary register `aux3` does not change.
1. Auxiliary register `aux4` does not change.
1. Auxiliary register `aux5` does not change.
1. Auxiliary register `aux6` does not change.
1. Auxiliary register `aux7` does not change.
1. Auxiliary register `aux8` does not change.
1. Auxiliary register `aux9` does not change.
1. Auxiliary register `aux10` does not change.
1. Auxiliary register `aux11` does not change.
1. Auxiliary register `aux12` does not change.
1. Auxiliary register `aux13` does not change.
1. Auxiliary register `aux14` does not change.
1. Auxiliary register `aux15` does not change.

##### Polynomials

1. `aux0' - aux0`
1. `aux1' - aux1`
1. `aux2' - aux2`
1. `aux3' - aux3`
1. `aux4' - aux4`
1. `aux5' - aux5`
1. `aux6' - aux6`
1. `aux7' - aux7`
1. `aux8' - aux8`
1. `aux9' - aux9`
1. `aux10' - aux10`
1. `aux11' - aux11`
1. `aux12' - aux12`
1. `aux13' - aux13`
1. `aux14' - aux14`
1. `aux15' - aux15`

#### Group `u32_op`

This group has no constraints.
It is used for the Permutation Argument with the uint32 table.

#### Group `grow_stack`

##### Description

1. The stack element in `st0` is moved into `st1`.
1. The stack element in `st1` is moved into `st2`.
1. The stack element in `st2` is moved into `st3`.
1. The stack element in `st3` is moved into `st4`.
1. The stack element in `st4` is moved into `st5`.
1. The stack element in `st5` is moved into `st6`.
1. The stack element in `st6` is moved into `st7`.
1. The stack element in `st7` is moved to the top of OpStack underflow, i.e., `osv`.
1. The OpStack pointer is incremented by 1.

##### Polynomials

1. `st1' - st0`
1. `st2' - st1`
1. `st3' - st2`
1. `st4' - st3`
1. `st5' - st4`
1. `st6' - st5`
1. `st7' - st6`
1. `osv' - st7`
1. `osp' - (osp + 1)`

#### Group `keep_stack`

##### Description

1. The stack element in `st0` does not change.
1. The stack element in `st1` does not change.
1. The stack element in `st2` does not change.
1. The stack element in `st3` does not change.
1. The stack element in `st4` does not change.
1. The stack element in `st5` does not change.
1. The stack element in `st6` does not change.
1. The stack element in `st7` does not change.
1. The top of the OpStack underflow, i.e., `osv`, does not change.
1. The OpStack pointer does not change.

##### Polynomials

1. `st0' - st0`
1. `st1' - st1`
1. `st2' - st2`
1. `st3' - st3`
1. `st4' - st4`
1. `st5' - st5`
1. `st6' - st6`
1. `st7' - st7`
1. `osv' - osv`
1. `osp' - osp`

#### Group `shrink_stack`

This instruction group requires helper variable `hv4` to hold the multiplicative inverse of `(osp' - 7)`.
In effect, this means that the OpStack pointer can never be 7, which would indicate a stack of size -1.
Since the stack can only change by one element at a time, this prevents stack underflow.

##### Description

1. The stack element in `st1` is moved into `st0`.
1. The stack element in `st2` is moved into `st1`.
1. The stack element in `st3` is moved into `st2`.
1. The stack element in `st4` is moved into `st3`.
1. The stack element in `st5` is moved into `st4`.
1. The stack element in `st6` is moved into `st5`.
1. The stack element in `st7` is moved into `st6`.
1. The stack element at the top of OpStack underflow, i.e., `osv`, is moved into `st7`.
1. The OpStack pointer is decremented by 1.
1. The helper variable register `hv4` holds the inverse of `(osp' - 7)`.

##### Polynomials

1. `st0' - st1`
1. `st1' - st2`
1. `st2' - st3`
1. `st3' - st4`
1. `st4' - st5`
1. `st5' - st6`
1. `st6' - st7`
1. `st7' - osv`
1. `osp' - (osp - 1)`
1. `(osp' - 7)·hv4 - 1`

#### Group `unop`

##### Description

1. The stack element in `st1` does not change.
1. The stack element in `st2` does not change.
1. The stack element in `st3` does not change.
1. The stack element in `st4` does not change.
1. The stack element in `st5` does not change.
1. The stack element in `st6` does not change.
1. The stack element in `st7` does not change.
1. The top of the OpStack underflow, i.e., `osv`, does not change.
1. The OpStack pointer does not change.

##### Polynomials

1. `st1' - st1`
1. `st2' - st2`
1. `st3' - st3`
1. `st4' - st4`
1. `st5' - st5`
1. `st6' - st6`
1. `st7' - st7`
1. `osv' - osv`
1. `osp' - osp`

#### Group `binop`

##### Description

1. The stack element in `st2` does not change.
1. The stack element in `st3` does not change.
1. The stack element in `st4` does not change.
1. The stack element in `st5` does not change.
1. The stack element in `st6` does not change.
1. The stack element in `st7` does not change.
1. The top of the OpStack underflow, i.e., `osv`, does not change.
1. The OpStack pointer does not change.

##### Polynomials

1. `st2' - st2`
1. `st3' - st3`
1. `st4' - st4`
1. `st5' - st5`
1. `st6' - st6`
1. `st7' - st7`
1. `osv' - osv`
1. `osp' - osp`

#### Instruction `pop`

This instruction has no additional transition constraints.

#### Instruction `push` + `a`

##### Description

1. The instruction's argument `a` is moved onto the stack.

##### Polynomials

1. `st0' - nia`

#### Instruction `divine`

This instruction has no additional transition constraints.

#### Instruction `dup` + `i`

This instruction makes use of [indicator polynomials](#indicator-polynomials-ind_ihv3-hv2-hv1-hv0).
For their definition, please refer to the corresponding section.

##### Description

1. If `i` is 0, then `st0` is put on top of the stack.
1. If `i` is 1, then `st1` is put on top of the stack.
1. If `i` is 2, then `st2` is put on top of the stack.
1. If `i` is 3, then `st3` is put on top of the stack.
1. If `i` is 4, then `st4` is put on top of the stack.
1. If `i` is 5, then `st5` is put on top of the stack.
1. If `i` is 6, then `st6` is put on top of the stack.
1. If `i` is 7, then `st7` is put on top of the stack.

##### Polynomials

1. `ind_0(hv3, hv2, hv1, hv0)·(st0' - st0)`
1. `ind_1(hv3, hv2, hv1, hv0)·(st0' - st1)`
1. `ind_2(hv3, hv2, hv1, hv0)·(st0' - st2)`
1. `ind_3(hv3, hv2, hv1, hv0)·(st0' - st3)`
1. `ind_4(hv3, hv2, hv1, hv0)·(st0' - st4)`
1. `ind_5(hv3, hv2, hv1, hv0)·(st0' - st5)`
1. `ind_7(hv3, hv2, hv1, hv0)·(st0' - st7)`

#### Instruction `swap` + `i`

This instruction makes use of [indicator polynomials](#indicator-polynomials-ind_ihv3-hv2-hv1-hv0).
For their definition, please refer to the corresponding section.

##### Description

1. Argument `i` is not 0.
1. If `i` is 1, then `st1` is moved into `st0`.
1. If `i` is 1, then `st0` is moved into `st1`.
1. If `i` is not 1, then `st1` does not change.
1. If `i` is 2, then `st2` is moved into `st0`.
1. If `i` is 2, then `st0` is moved into `st2`.
1. If `i` is not 2, then `st2` does not change.
1. If `i` is 3, then `st3` is moved into `st0`.
1. If `i` is 3, then `st0` is moved into `st3`.
1. If `i` is not 3, then `st3` does not change.
1. If `i` is 4, then `st4` is moved into `st0`.
1. If `i` is 4, then `st0` is moved into `st4`.
1. If `i` is not 4, then `st4` does not change.
1. If `i` is 5, then `st5` is moved into `st0`.
1. If `i` is 5, then `st0` is moved into `st5`.
1. If `i` is not 5, then `st5` does not change.
1. If `i` is 6, then `st6` is moved into `st0`.
1. If `i` is 6, then `st0` is moved into `st6`.
1. If `i` is not 6, then `st6` does not change.
1. If `i` is 7, then `st7` is moved into `st0`.
1. If `i` is 7, then `st0` is moved into `st7`.
1. If `i` is not 7, then `st7` does not change.

##### Polynomials

1. `ind_0(hv3, hv2, hv1, hv0)`
1. `ind_1(hv3, hv2, hv1, hv0)·(st0' - st1)`
1. `ind_1(hv3, hv2, hv1, hv0)·(st1' - st0)`
1. `(1 - ind_1(hv3, hv2, hv1, hv0))·(st1' - st1)`
1. `ind_2(hv3, hv2, hv1, hv0)·(st0' - st2)`
1. `ind_2(hv3, hv2, hv1, hv0)·(st2' - st0)`
1. `(1 - ind_2(hv3, hv2, hv1, hv0))·(st2' - st2)`
1. `ind_3(hv3, hv2, hv1, hv0)·(st0' - st3)`
1. `ind_3(hv3, hv2, hv1, hv0)·(st3' - st0)`
1. `(1 - ind_3(hv3, hv2, hv1, hv0))·(st3' - st3)`
1. `ind_4(hv3, hv2, hv1, hv0)·(st0' - st4)`
1. `ind_4(hv3, hv2, hv1, hv0)·(st4' - st0)`
1. `(1 - ind_4(hv3, hv2, hv1, hv0))·(st4' - st4)`
1. `ind_5(hv3, hv2, hv1, hv0)·(st0' - st5)`
1. `ind_5(hv3, hv2, hv1, hv0)·(st5' - st0)`
1. `(1 - ind_5(hv3, hv2, hv1, hv0))·(st5' - st5)`
1. `ind_6(hv3, hv2, hv1, hv0)·(st0' - st6)`
1. `ind_6(hv3, hv2, hv1, hv0)·(st6' - st0)`
1. `(1 - ind_6(hv3, hv2, hv1, hv0))·(st6' - st6)`
1. `ind_7(hv3, hv2, hv1, hv0)·(st0' - st7)`
1. `ind_7(hv3, hv2, hv1, hv0)·(st7' - st0)`
1. `(1 - ind_7(hv3, hv2, hv1, hv0))·(st7' - st7)`

### Instruction `nop`

This instruction has no additional transition constraints.

#### Instruction `skiz`

##### Description

Note:
The concrete decomposition of `nia` into helper variables `hv` as well as the concretely relevant `hv` determining whether `nia` takes an argument (currently `hv0`) are to be determined.

1. The jump stack pointer `jsp` does not change.
1. The last jump's origin `jso` does not change.
1. The last jump's destination `jsd` does not change.
1. The next instruction `nia` is decomposed into helper variables `hv`.
1. The relevant helper variable `hv0` is either 0 or 1.
    Here, `hv0 == 1` means that `nia` takes an argument.
1. Register `ip` increments by (1 if `st0` is non-zero else (2 if `nia` takes no argument else 3)).

##### Polynomials

1. `jsp' - jsp`
1. `jso' - jso`
1. `jsd' - jsd`
1. `nia - (hv0 + 2·hv1)`
1. `hv0·(hv0 - 1)`
1. `ip' - (ip + 1 + st0·inv·(1 + hv0))`

#### Instruction `call` + `d`

##### Description

1. The jump stack pointer `jsp` is incremented by 1.
1. The jump's origin `jso` is set to the current instruction pointer `ip` plus 2.
1. The jump's destination `jsd` is set to the instruction's argument `d`.
1. The instruction pointer `ip` is set to the instruction's argument `d`.

##### Polynomials

1. `jsp' - (jsp + 1)`
1. `jso' - (ip + 2)`
1. `jsd' - nia`
1. `ip' - nia`

#### Instruction `return`

##### Description

1. The jump stack pointer `jsp` is decremented by 1.
1. The instruction pointer `ip` is set to the last call's origin `jso`.

##### Polynomials

1. `jsp' - (jsp - 1)`
1. `ip' - jso`

#### Instruction `recurse`

##### Description

1. The jump stack pointer `jsp` does not change.
1. The last jump's origin `jso` does not change.
1. The last jump's destination `jsd` does not change.
1. The instruction pointer `ip` is set to the last jump's destination `jsd`.

##### Polynomials

1. `jsp' - jsp`
1. `jso' - jso`
1. `jsd' - jsd`
1. `ip' - jsd`

#### Instruction `assert`

##### Description

1. The current top of the stack `st0` is 1.

##### Polynomials

1. `st0 - 1`

#### Instruction `halt`

##### Description

1. The instruction executed in the following step is instruction `halt`.

##### Polynomials

1. `ci' - ci`

#### Instruction `read_mem`

#### Instruction `write_mem`

#### Instruction `xlix`

#### Instruction `clearall`

#### Instruction `squeeze` + `i`

#### Instruction `absorb` + `i`

#### Instruction `divine_sibling`

#### Instruction `compare_digest`

#### Instruction `add`

##### Description

1. The sum of the top two stack elements is moved into the top of the stack.

##### Polynomials

1. `st0' - (st0 + st1)`

#### Instruction `mul`

##### Description

1. The product of the top two stack elements is moved into the top of the stack.

##### Polynomials

1. `st0' - st0·st1`

#### Instruction `invert`

##### Description

1. The top of the stack's inverse is moved into the top of the stack.

##### Polynomials

1. `st0'·st0 - 1`

#### Instruction `split`

#### Instruction `eq`

#### Instruction `lt`

#### Instruction `and`

#### Instruction `xor`

#### Instruction `reverse`

#### Instruction `div`

A Permutation Argument guarantees `r < d`.

##### Description

1. Denominator `d` is not zero.
1. Result of division, i.e., quotient `q` and remainder `r`, are moved into `st1` and `st0` respectively, and match with numerator `n` and denominator `d`.
1. The stack element in `st2` does not change.
1. The stack element in `st3` does not change.
1. The stack element in `st4` does not change.
1. The stack element in `st5` does not change.
1. The stack element in `st6` does not change.
1. The stack element in `st7` does not change.

##### Polynomials

1. `st0·inv - 1`
1. `st1 - st0·st1' - st0'`
1. `st2' - st2`
1. `st3' - st3`
1. `st4' - st4`
1. `st5' - st5`
1. `st6' - st6`
1. `st7' - st7`

#### Instruction `xxadd`

#### Instruction `xxmul`

#### Instruction `xinv`

#### Instruction `xbmul`

#### Instruction `read_io`

#### Instruction `write_io`
