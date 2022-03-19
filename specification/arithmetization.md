# Triton VM Arithmetization

This document describes the arithmetization of Triton VM, whose instruction set architecture is defined [here](isa.md). An arithmetization defines two things: a algebraic execution tables (AETs) and arithmetic intermediate representation (AIR) constraints. The nature of Triton VM is that the execution trace is spread out over multiple tables, but linked through permutation and evaluation arguments.

Elsewhere, the acronym AET stands for algebraic execution *trace*. In the nomenclature of this note, a trace is a special kind of table that tracks the values of a set of registers across time.

## Memory Tables

The *general format* of memory tables is as follows.

| Cycle | Instruction | Address | New Value | Old Value |
|-------|-------------|---------|-----------|-----------|
| - | - | - | - | - |
| - | - | - | - | - |
| - | - | - | - | - |

The rows are sorted by address, then by cycle. Consecutive rows with the same address can be verified to have consistent memory values, i.e., changing value only when the instruction is set to "write". Consecutive rows with distinct addresses represent distinct memory cells. The second row of such a pair is the first time the new memory cell is referenced, and so the Old Value of this row must be set to zero. A copy-constraint establishes that memory accesses of the register table are consistent with this table.

However, the specific memory tables used in Triton VM differ from this pattern in important ways. For instance, the cycle counter or the distinction between old and new values can be dropped.

### Program Table

The Virtual Machine's Program Memory is read-only. The corresponding Program Table consists of two columns, `address` and `instruction`. The latter variable does not correspond to the processor's state but to the value of the memory at the given location.

| Address $P_a$ | Instruction $P_i$ |
|---------------|-------------------|
| - | - |

The Program Table is static in the sense that it is fixed before the VM runs. Moreover, the user can commit to the program by providing the Merkle root of the zipped FRI codeword. This commitment assumes that the FRI domain is fixed, which implies an upper bound on program size.

**Boundary Constraints**

 1. The first address is zero: ${P_a}_{[0]} = 0$.

**Transition Constraints**

 1. The addresses increase monotonically for all $i$: ${P_a}_{[i+1]} = {P_a}_{[i]} + 1$.

**Relations to other Tables**

 1. A Program Evaluation Argument establishes that the rows of the Program Table match with the unique rows of the Instruction Table.

### Jump Stack Table

The Jump Stack Memory contains the underflow from the Jump Stack. The virtual machine defines two registers to deal with the Jump Stack: `rap`, the return address pointer, which points to a location in Return Address Stack Memory; and `rav`, the return address value, which points to a location in Program Memory.

The Jump Stack Table is a table whose columns are a subset of those of the Processor Table with one addition: the destination address `dest`. The rows are sorted by return address pointer (`rap`), then by cycle counter (`clk`). The column `dest` contains the destination of stack-extending jump (`call`) as well as of the no-stack-change jump (`recurse`); the column `rav` contains the source of the stack-extending jump (`call`) or equivalently the destination of the stack-shrinking jump (`return`).

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

| `clk` | `ip`   | `ci`     | `ni`     | `rav`  | `rap`  | return address stack | destination address stack |
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

| `clk` | `ci`     | `rap`  | `rav`  | `dest` |
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

 1. The return address stack pointer `rap` increases by one, *or*
 2. (`rap`, `rav` and `dest` remain the same and) the cycle counter `clk` increases by one, *or*
 3. (`rap`, `rav` and `dest` remain the same and) the current instruction `ci` is `call`, *or*
 4. (`rap` remains the same and) the current instruction `ci` is `return`.

### Operational Stack

The operational stack is where the program stores simple elementary operations, function arguments, and pointers to important objects. There are four registers that the program can access directly; these registers correspond to the top of the stack. The rest of the operational stack is stored in a dedicated memory object called Operational Stack Memory.

The operational stack memory table contains a subset of the rows of the Register Trace and *two* columns for the stored value -- the old value and the new value. In exchange, the Operational Stack Memory Table stores fewer rows. In fact, it only stores rows associated with a stack growth or shrinkage. A copy-constraint establishes that the stack reads and writes are correct.

The mechanics are best illustrated by an example.

Execution trace:

| `clk` | `pir` | `cir` | `os0` | `osp` | operational stack |
|-------|-------|-------|-------|-------|-------------------|
| 0 | 0 | 0 | 0 | 0 | [0;0,0,0] |
| 1 | 0 | `push` | 0 | 1 | [0,0;0,0,0] |
| 2 | `push` | 0x0A | 0 | 1 | [0,0;0,0,0x0A] |
| 3 | 0x0A | `push` | 0 | 2 | [0,0,0;0,0x0A,0] |
| 4 | `push` | 0x0B | 0 | 2 | [0,0,0;0,0x0A,0x0B] |
| 5 | 0x0B | `push` | 0 | 3 | [0,0,00;0x0A,0x0B,0] |
| 6 | `push` | 0x0C | 0 | 3 | [0,0,0,0;0x0A,0x0B,0x0C] |
| 7 | 0x0C | `push` | 0x0A | 4 | [0,0,0,0,0x0A;,0x0B,0x0C,0] |
| 8 | `push` | 0x0D | 0x0A | 4 | [0,0,0,0,0x0A;0x0B,0x0C,0x0D] |
| 9 | 0x0D | `push` | 0x0B | 5 | [0,0,0,0,0x0A,0x0B;0x0C,0x0D,0] |
| 10| `push` | 0x0E | 0x0B | 5 | [0,0,0,0,0x0A,0x0B;0x0C,0x0D,0x0E] |
| 11| 0x0E | `add` | 0x0A | 4 | [0,0,0,0,0x0A;0x0B,0x0C,0x1C] |
| 12| `add` | `add` | 0 | 3 | [0,0,0,0;0x0A,0x0B,0x28] |
| 13| `add` | `add` | 0 | 2 | [0,0,0;0,0x0A,0x31] |
| 14| `add` | `add` | 0 | 1 | [0,0;0,0,0x3B] |
| 15| `add` | `pop` | 0 | 0 | [0;0,0,0] |

Memory table:

| `clk`* | `cir` | `os0` old | `os0` new | `osp` |
|-------|-------|-----------|-----------|-------|
| 0*| 0 | 0 | 0 | 0 |
| 15| `pop` | 0 | 0 | 0 |
| 1 | `push` | 0 | 0 | 1 |
| 2*| 0x0A | 0 | 0 | 1 |
| 14| `add` | 0 | 0 | 1 |
| 3 | `push` | 0 | 0 | 2 |
| 4*| 0x0B | 0 | 0 | 2 |
| 13| `add` | 0 | 0 | 2 |
| 5 | `push` | 0 | 0 | 3 |
| 6*| 0x0C | 0 | 0 | 3 |
| 12| `add` | 0 | 0 | 3 |
| 7 | `push` | 0 | 0x0A | 4 |
| 8*| 0x0D | 0x0A | 0x0A | 4 |
| 11| `add` | 0x0A | 0x0A | 4 |
| 9 | `push` | 0 | 0x0B | 5 |
|10*| 0x0E | 0x0B | 0x0B | 5 |

The rows marked with asterisks are included for the sake of completeness in this explication. In practice they will be omitted because they neither shrink nor grow the stack. Additionally, there is no need to include the cycle counter column.

**Boundary Conditions**

 1. The old value, new value, and operational stack pointer, all start out at 0.

**Transition Constraints**

 1. The operatoinal stack pointer increases by 1, *or*
 2. The instruction is one that grows the stack by 1, *or*
 3. The new value is the same as the old value.

### Random Access Memory

The RAM is accessible in two ways: first, individual memory elements can be read to and written from the stack; second, chunks of four elements (words) can be written to and read from the SIMD register. To enable this, the address is split into the high part (everything but the least significant two bits), and low bart (least significant two bits).

| Cycle $R_c$ | Instr $R_i$ | IB0 $R_{i0}$ | ... | IB5 $R_{i5}$ | Addr $R_a$ | AddrHi $R_{ahi}$ | AddrLo $R_{alo}$ | NV0 $R_{nv0}$ | ... | NV3 $R_{nv3}$ | OV0 $R_{ov0}$ | ... | OV3 $R_{ov}$ | Val $R_{val}$ |
|------------|-------------------|----------------------|--------|---------------|------|-----|----|---------------|-------|----------------|-------|--------------|-|-|
| - | - | - |  | - | - | - | - | - |  | - | - |  | - | 

The polynomials interpolate the columns of this table over the trace domain $\{\omicron^i \, \vert \, 0 \leq i < 2^k\}$. Let $f_s(\mathbf{X})$ and $f_v(\mathbf{X})$ be the low degree polynomials that indicate write instructions from stack and SIMD registers, respectively. The AIR constraints are:
 - boundary constraints: 
    - initial values: $\forall j \in \{0, \ldots, 3\} \, . \, R_{ovj}(1) = 0$
 - consistency constraints, over $X \in \{\omicron^i \, \vert \, 0 \leq i < 2^k\}$:
    - valid decomposition of address: $4 \cdot R_{ahi}(X) + R_{alo}(X) - R_a(X) = 0$
    - valid least-significant two bits: $R_{alo}(X) \cdot ( 1 - R_{alo(X)} ) \cdot ( 2 - R_{alo}(X)) \cdot (3 - R_{alo}(X)) = 0$
 - transition constraints, for $X \in \{\omicron^i \vert \, 0 \leq i < 2^k-1\}$:
   - monotonicity of addresses: $R_a(\omicron \cdot X) - R_a(X) - 1 = 0$
   - conditional monotonicity of cycles: $(R_a(\omicron \cdot X) - R_a(X)) \cdot (R_c(\omicron \cdot X) - R_c(X)) = 0$
   - initial values of new memory cells: $\forall j \in \{0,\ldots,3\} \, . \, (R_a(\omicron \cdot X) - R_a(X)) \cdot R_{ovj}(\omicron \cdot X) = 0$
   - no changes unless instructed: $\forall j \in \{0, \ldots, 3\} \, . \, (R_{ovj}(X) - R_{nvj}(X)) \cdot ( f_s(\mathbf{X}) + f_v(\mathbf{X}) - 1 ) = 0$
   - when writing from stack, just the one register may change: $\forall j \in \{0, \ldots, 3\} \, . \, f_s(\mathbf{X}) \cdot (\prod_{j' \neq j} (R_{alo}(X) - j')) \cdot (R_{ovj}(X) - R_{nvj}(X)) = 0$

## Register Trace

The trace consists of 38 registers, Many of the names and functions are defined in the [instruction set architecture](isa.md). The remaining ones are:
 - `ibj` for `j` ranging from 0 to 5; these are the instruction bit registers. Each `ibj` should contain exactly one bit of the current instruction.
 - `if` immediate flag. The current instruction is not executed if this flag is set; instead, the previous is.

| `cir` $T_{cir}$ | `pir` $T_{pir}$ | `clk` $T_{clk}$ | `if` $T_{if}$ | `ib0` $T_{ib0}$ | ... | `ib5` $T_{ib5}$ | `ip` $T_{ip}$ | `rp` $T_{rp}$ | `sp` $T_{sp}$ | `st0` $T_{st0}$ | ... | `st3` $T_{st3}$ | `hv0` $T_{hv0}$ | ... | `hv4` $T_{hv4}$ | `sc0` $T_{sc0}$ | ... | `sc15` $T_{sc15}$ |
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|

### Register Trace AIR

The columns are interpolated over the trace domain $\{\omicron^i \, | \, 0 \leq i < 2^k\}$. The AIR constraints follow.

#### Boundary constraints

Zero initial values: $T_*(1) = 0$ for all polynomials. This implies that the first instruction is zero and that the stack initially contains four zero elements.

#### Consistency constraints

The consistency constraints hold for all $X \in \{\omicron^i \, | \, 0 \leq i < 2^k\}$.
- binary instruction bits: $\forall j \in \{0, \ldots, 5\} \, . \, T_{ibj}(X) \cdot ( 1 - T_{ibj}(X) ) = 0$
- valid instruction decomposition: $T_{cir}(X) - \sum_{j=0}^{5} 2^j \cdot T_{ibj}(X) = 0$
- correct decomposition of stack top: $T_{st3}(X)) - \sum_{j=0}^{3} 2^{16 \cdot j} \cdot T_{hvj}(X) = 0$
- unique decomposition of stack top: $\big(T_{hv3}(X) \cdot (2^{16} \cdot T_{hv3}(X) + T_{hv2}(X)) - 1\big) \cdot (2^{16} \cdot T_{hv1}(X) + T_{hv0}(X)) = 0$

#### Transition Constraints Instruction Fetching and Control Flow

The transition constraints are valid for all $X \in \{\omicron^i \, | \, 0 \leq i < 2^k-1\}$.

   - monotonicity of cycle: $T_{clk}(\omicron \cdot X) - T_{clk}(X) = 0$
   - correct movement of instructions: $T_{pir}(\omicron \cdot X) - T_{cir}(X) = 0$
   - control flow, with indicator functions $f_{skiz}(\mathbf{X})$, $f_{jumpa}(X)$, and $f_{jumpr}(X)$
     - regular flow: $(1 - f_{skiz}(X) - f_{jumpa}(X) - f_{jumpr}(X)) \cdot (T_{ip}(\omicron \cdot X) - T_{ip}(X) - 1) = 0$

#### Transition Constraints for Stack

The transition constraints are valid for all $X \in \{\omicron^i \, | \, 0 \leq i < 2^k-1\}$.

   - stack
     - growth
     - shrinkage
     - manipulation

#### Transition Constraints for Native Arithmetic

The transition constraints are valid for all $X \in \{\omicron^i \, | \, 0 \leq i < 2^k-1\}$.

   - native arithmetic operations TODO
  

### Register Trace Copy-Constraints

Columns from the Register Trace are involved in the following copy-constraints:
 - For each cycle: $(T_{ip}(X), T_{cir}(X))$ with $(P_a(X), P_v(X))$ from the Program Memory Table.
 - For each of `lt`, `eq`, `and`, `or`, `xor`, `reverse`: $(T_{st1}(\omicron \cdot X), T_{st2}(\omicron \cdot X), T_{st3}(\omicron \cdot X))$ with matching rows and columns of the Uint32 Operations Table.
 - For each `div`:
   1. $(T_{st2}(X), T_{st2}(\omicron \cdot X))$ with matching rows in the Uint32 Operations Table, and the simulated column corresponding to `lt` + `eq`.
   2. $(T_{st3}(X), T_{st3}(\omicron \cdot X))$ with matching rows in the Uint32 Operations Table, and the column corresponding to `lt`.
 - For each or `read` and `write`: $(T_{clk}(X), T_{cir}(X), T_{ib0}(X), \ldots, T_{ib5}(X), T_{rp}(X), T_{st3}(X))$ with matching rows in the RAM Table.
 - For each `vread` and `vwrite`: $(T_{clk}(X), T_{pir}(X), T_{ib0}(X), \ldots, T_{ib5}(X), T_{rp}(X), T_{scj}(X), \ldots, T_{scj+3}(X))$ (where $j$ takes the value $0, 4, 8, 12$ depending on the immediate argument) with the matching rows in the RAM Table.
 - For each `split`, a batch-copy-constraint establishes that $(T_{hv0}(X), T_{hv1}(X), T_{hv2}(X), T_{hv3}(X))$ have matching rows in the Dynamic 16-Bit Table.
 - For each `vsplit`, a batch-copy-constraint establishes that $(T_{sc0}(X), T_{sc15}(X))$ have matching rows in the Dynamic 16-Bit Table.
 - For each `gauss`: $(T_{st3}(X), T_{st3}(\omicron \cdot X))$ with matching rows in the 16-Bit/Gauss Table.
 - For each `vgauss`, a batch-copy-constraint establishes that $(T_{sc0}(X),\ldots,T_{sc15}(X),T_{sc0}(\omicron \cdot X),\ldots,T_{sc15}(\omicron \cdot X))$ are in one-to-one correspondence with 16 consecutive rows of the 16-Bit/Gauss table $(G_i(X), G_o(X))$.
 - For each `vsplit`, and for each $j \in \{0, 4, 8, 12\}$: $(T_{scj}(X), \ldots, T_{scj+3}(X))$ with matching rows in the 4-Wide 32-Bit Table.
 - For each `xlix`: consecutive rows of $(T_{sc0}(X), \ldots T_{sc15}(X))$ with rows apart by 8 in the Rescue-XLIX Table.

## Auxiliary Tables

### Rescue-XLIX

The Rescue-XLIX table applies the Rescue-XLIX permutation to 16 variables, one round at a time.

|  | H0 $H_0$ | ... | H15 $H_{15}$ |
|--|----------|-----|--------------|
|0.| -        | ... | - |
|1.| -        | ... | - |
|2.| -        | ... | - |
|3.| -        | ... | - |
|4.| -        | ... | - |
|5.| -        | ... | - |
|6.| -        | ... | - |
|7.| -        | ... | - |

The AIR makes no requirements for rows that are multiples of 8. Other rows apply one round of Rescue-XLIX, with the parameters determined by the row index modulo 8.

A copy-constraint establishes that, whenever the `xlix` instruction is executed, the old and new 16-tuples in the SIMD cache correspond to rows $8k$ and $8k+7$ in the XLIX Table for some integer $k$.

### Uint32 Operations

The Uint32 Operations Table is a lookup table for 'difficult' 32-bit unsigned integer operations.

|     | LHS      | RHS      | EQ     | LT    | AND       | OR       | XOR       | REV |
|-----|----------|----------|--------|-------|-----------|----------|-----------|-----|
| 1.  | `a`      | `b`      | `a==b` | `a<b` | `a and b` | `a or b` | `a xor b` | `rev(a)` |
| 2.  | `a >> 1` | `b >> 1` | - | - | - | - | - | - |
| ... | - | - | - | - | - | - | - | - | - | - |
| 32. | `0` | `0` | `1` | `0` | `0` | `0` | `0` | `0` |
| 33. | `c` | `d` | - | - | - | - | - | - |

The AIR verifies the correct update of each consecutive pair of rows. In every row one bit is eliminated. Only when the previous row is all zeros (with a 1 in the column for `EQ`) can a new row be inserted.

The AIR constraints establish that the entire table is consistent. Copy-constraints establish that logical and bitwise operations were computed correctly.
